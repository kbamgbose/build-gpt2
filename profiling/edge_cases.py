"""
Edge case testing:
  1. Extremely long sequences (>10k tokens) — memory limits and OOM behavior
  2. Numerical stability — NaN/inf detection across dtypes and sequence lengths

Run on the pod:
    python profiling/edge_cases.py
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import math

DEVICE = "cuda"


# ── minimal attention for isolated testing ────────────────────────────────────

def naive_attention(q, k, v):
    """Manual attention — used to stress test numerical stability."""
    hs = q.size(-1)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
    T = q.size(-2)
    mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
    att = att.masked_fill(~mask, float('-inf'))
    att = F.softmax(att, dim=-1)
    return att @ v

def flash_attention(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)

def has_nan_or_inf(tensor):
    return torch.isnan(tensor).any().item() or torch.isinf(tensor).any().item()


# ── experiment 1: long sequence memory limits ─────────────────────────────────

def long_sequence_test():
    print("=" * 65)
    print("EXPERIMENT 1: Long sequence memory limits")
    print("Config: B=1, n_head=12, head_size=64 (GPT-2 dims)")
    print("=" * 65)
    print(f"{'seq_len':>8} {'naive':>12} {'flash':>12} {'naive_mb':>10} {'flash_mb':>10}")
    print("-" * 58)

    seq_lens = [1024, 2048, 4096, 8192, 16384]
    B, nh, hs = 1, 12, 64

    for T in seq_lens:
        q = torch.randn(B, nh, T, hs, device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn(B, nh, T, hs, device=DEVICE, dtype=torch.bfloat16)
        v = torch.randn(B, nh, T, hs, device=DEVICE, dtype=torch.bfloat16)

        # naive
        naive_status = "OK"
        naive_mb = 0.0
        try:
            torch.cuda.reset_peak_memory_stats()
            _ = naive_attention(q, k, v)
            torch.cuda.synchronize()
            naive_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        except torch.cuda.OutOfMemoryError:
            naive_status = "OOM"
            torch.cuda.empty_cache()

        # flash
        flash_status = "OK"
        flash_mb = 0.0
        try:
            torch.cuda.reset_peak_memory_stats()
            _ = flash_attention(q, k, v)
            torch.cuda.synchronize()
            flash_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        except torch.cuda.OutOfMemoryError:
            flash_status = "OOM"
            torch.cuda.empty_cache()

        naive_col = f"{naive_mb:>8.1f} MB" if naive_status == "OK" else "     OOM"
        flash_col = f"{flash_mb:>8.1f} MB" if flash_status == "OK" else "     OOM"
        print(f"{T:>8} {naive_col:>12} {flash_col:>12}")

        del q, k, v
        torch.cuda.empty_cache()

    print()
    print("OOM = out of memory. FlashAttention should survive longer sequences")
    print("because it never materializes the full (T, T) matrix in HBM.")


# ── experiment 2: numerical stability ────────────────────────────────────────

def numerical_stability_test():
    print("\n" + "=" * 65)
    print("EXPERIMENT 2: Numerical stability")
    print("=" * 65)

    B, nh, hs = 2, 8, 64
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    seq_lens = [512, 2048, 8192]

    # 2a: dtype comparison at fixed sequence length
    print("\n--- 2a: dtype vs NaN/inf (T=1024) ---")
    print(f"{'dtype':>12} {'naive_nan':>12} {'flash_nan':>12} {'max_val':>12}")
    print("-" * 52)

    T = 1024
    for dtype in dtypes:
        q = torch.randn(B, nh, T, hs, device=DEVICE, dtype=dtype)
        k = torch.randn(B, nh, T, hs, device=DEVICE, dtype=dtype)
        v = torch.randn(B, nh, T, hs, device=DEVICE, dtype=dtype)

        try:
            naive_out = naive_attention(q, k, v)
            naive_bad = has_nan_or_inf(naive_out)
            max_val   = naive_out.abs().max().item()
        except Exception as e:
            naive_bad = f"ERR"
            max_val   = -1

        try:
            flash_out = flash_attention(q, k, v)
            flash_bad = has_nan_or_inf(flash_out)
        except Exception:
            flash_bad = "ERR"

        print(f"{str(dtype):>12} {str(naive_bad):>12} {str(flash_bad):>12} {max_val:>12.4f}")

    # 2b: growing sequence length in float16 (most likely to overflow)
    print("\n--- 2b: float16 stability at increasing T ---")
    print(f"{'T':>8} {'naive_nan':>12} {'flash_nan':>12}")
    print("-" * 36)

    for T in seq_lens:
        q = torch.randn(B, nh, T, hs, device=DEVICE, dtype=torch.float16)
        k = torch.randn(B, nh, T, hs, device=DEVICE, dtype=torch.float16)
        v = torch.randn(B, nh, T, hs, device=DEVICE, dtype=torch.float16)

        try:
            naive_out = naive_attention(q, k, v)
            naive_bad = has_nan_or_inf(naive_out)
        except torch.cuda.OutOfMemoryError:
            naive_bad = "OOM"

        try:
            flash_out = flash_attention(q, k, v)
            flash_bad = has_nan_or_inf(flash_out)
        except torch.cuda.OutOfMemoryError:
            flash_bad = "OOM"

        print(f"{T:>8} {str(naive_bad):>12} {str(flash_bad):>12}")
        torch.cuda.empty_cache()

    # 2c: pathological input — large magnitude values that stress the scaling
    print("\n--- 2c: large magnitude inputs (scale=100) ---")
    print(f"{'dtype':>12} {'naive_nan':>12} {'flash_nan':>12}")
    print("-" * 40)

    T = 512
    for dtype in dtypes:
        q = torch.randn(B, nh, T, hs, device=DEVICE, dtype=dtype) * 100
        k = torch.randn(B, nh, T, hs, device=DEVICE, dtype=dtype) * 100
        v = torch.randn(B, nh, T, hs, device=DEVICE, dtype=dtype) * 100

        try:
            naive_out = naive_attention(q, k, v)
            naive_bad = has_nan_or_inf(naive_out)
        except Exception:
            naive_bad = "ERR"

        try:
            flash_out = flash_attention(q, k, v)
            flash_bad = has_nan_or_inf(flash_out)
        except Exception:
            flash_bad = "ERR"

        print(f"{str(dtype):>12} {str(naive_bad):>12} {str(flash_bad):>12}")

    print()
    print("float16 + large T is the danger zone — scores overflow before softmax.")
    print("bfloat16 has wider dynamic range and is much more stable.")
    print("FlashAttention uses online softmax normalization — more stable by design.")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    long_sequence_test()
    numerical_stability_test()
