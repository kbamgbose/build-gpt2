"""
Block size (depth) experiment: compare 6, 12, 24 transformer layers.
Measures memory, latency, and throughput at fixed model width and sequence length.
Documents the O(n²) attention cost compounding across layers.

Run on the pod:
    python profiling/block_size_experiment.py
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

DEVICE = "cuda"
DTYPE  = torch.bfloat16
RUNS   = 10
WARMUP = 3

# ── model (same as scaling_experiment.py) ─────────────────────────────────────

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer:    int = 12
    n_head:     int = 12
    n_embd:     int = 768


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()
        hs = C // self.n_head
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        return self.lm_head(self.transformer.ln_f(x))


# ── benchmark ─────────────────────────────────────────────────────────────────

def benchmark(model, idx):
    for _ in range(WARMUP):
        _ = model(idx)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(RUNS):
        _ = model(idx)
    end.record()
    torch.cuda.synchronize()

    avg_ms  = start.elapsed_time(end) / RUNS
    peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    return avg_ms, peak_mb


# ── experiment ────────────────────────────────────────────────────────────────

def run():
    B, T = 4, 1024
    layer_counts = [6, 12, 24]

    print("=" * 65)
    print("EXPERIMENT: Varying transformer depth (n_layer)")
    print(f"Fixed: n_embd=768, n_head=12, T={T}, B={B}")
    print("=" * 65)
    print(f"{'n_layer':>8} {'params_M':>10} {'latency_ms':>12} {'peak_mb':>10} {'ms/layer':>10}")
    print("-" * 58)

    base_ms = None

    for n_layer in layer_counts:
        config = GPTConfig(block_size=T, n_layer=n_layer, n_embd=768, n_head=12)
        model  = GPT(config).to(DEVICE).to(DTYPE)
        idx    = torch.randint(0, config.vocab_size, (B, T), device=DEVICE)
        params = sum(p.numel() for p in model.parameters()) / 1e6

        avg_ms, peak_mb = benchmark(model, idx)
        ms_per_layer = avg_ms / n_layer

        if base_ms is None:
            base_ms = avg_ms

        print(f"{n_layer:>8} {params:>10.1f} {avg_ms:>12.2f} {peak_mb:>10.1f} {ms_per_layer:>10.2f}")
        del model

    print()
    print("Key observations to record:")
    print("  - latency should scale linearly with n_layer (attention cost per layer is fixed)")
    print("  - memory grows with depth due to activation storage for backprop")
    print("  - ms/layer stays roughly constant — depth is linear cost, not quadratic")
    print("  - contrast with T doubling (experiment 1): that was quadratic in attention")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    run()
