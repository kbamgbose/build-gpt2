"""
Profiles naive attention (manual mask + matmul) vs FlashAttention
(torch.nn.functional.scaled_dot_product_attention, which uses the
FlashAttention kernel automatically on CUDA with float16/bfloat16).

Run on the pod:
    python profiling/profile_attention.py

Outputs:
    profiling/attention_kernels.csv
"""
import csv
import math
import torch
import torch.nn.functional as F

DEVICE = "cuda"
DTYPE  = torch.bfloat16
RUNS   = 20       # timed iterations per config
WARMUP = 5        # warmup iterations (discarded)

def naive_attention(q, k, v):
    """Manual causal attention — materializes the full (T, T) matrix."""
    head_size = q.size(-1)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))
    T = q.size(-2)
    mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
    att = att.masked_fill(~mask, float('-inf'))
    att = F.softmax(att, dim=-1)
    return att @ v

def flash_attention(q, k, v):
    """PyTorch 2.x scaled_dot_product_attention uses FlashAttention on CUDA."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)

def benchmark(fn, q, k, v):
    """Returns (avg_ms, peak_memory_mb)."""
    # warmup
    for _ in range(WARMUP):
        _ = fn(q, k, v)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(RUNS):
        out = fn(q, k, v)
    end.record()
    torch.cuda.synchronize()

    avg_ms  = start.elapsed_time(end) / RUNS
    peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    return avg_ms, peak_mb

def run():
    configs = [
        # (batch, n_head, seq_len, head_size)
        (4,  12,  256,  64),
        (4,  12,  512,  64),
        (4,  12, 1024,  64),
        (4,  12, 2048,  64),
        (4,  12, 4096,  64),
    ]

    rows = []
    print(f"{'seq_len':>8} {'naive_ms':>12} {'flash_ms':>12} {'naive_mb':>12} {'flash_mb':>12} {'speedup':>8}")
    print("-" * 70)

    for B, nh, T, hs in configs:
        q = torch.randn(B, nh, T, hs, device=DEVICE, dtype=DTYPE)
        k = torch.randn(B, nh, T, hs, device=DEVICE, dtype=DTYPE)
        v = torch.randn(B, nh, T, hs, device=DEVICE, dtype=DTYPE)

        naive_ms, naive_mb = benchmark(naive_attention, q, k, v)
        flash_ms, flash_mb = benchmark(flash_attention, q, k, v)
        speedup = naive_ms / flash_ms

        print(f"{T:>8} {naive_ms:>12.2f} {flash_ms:>12.2f} {naive_mb:>12.1f} {flash_mb:>12.1f} {speedup:>8.2f}x")
        rows.append({
            "seq_len":   T,
            "batch":     B,
            "n_head":    nh,
            "head_size": hs,
            "naive_ms":  round(naive_ms, 3),
            "flash_ms":  round(flash_ms, 3),
            "naive_mb":  round(naive_mb, 1),
            "flash_mb":  round(flash_mb, 1),
            "speedup":   round(speedup, 2),
        })

    out_path = "profiling/attention_kernels.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved → {out_path}")

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    run()
