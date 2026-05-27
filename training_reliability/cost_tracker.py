"""
Tracks FLOPs, GPU memory, wall-clock time, and estimated dollar cost per training step.

Usage in train.py:
    tracker = CostTracker(raw_model, B, T, grad_accum_steps,
                          num_gpus=ddp_world_size,
                          gpu_cost_per_hr=float(os.environ.get('GPU_COST_PER_HR', '2.50')))
    # inside loop:
    tracker.step(step=step, dt_seconds=dt)
    # at end:
    tracker.save('logs/cost_report.json')
    tracker.print_summary()
"""

import json
import os
import time
import torch


def compute_flops_per_step(model, B, T, grad_accum_steps):
    """
    Precise FLOPs estimate for one optimizer step (forward + backward across all micro-steps).

    Per transformer layer:
      Attention:  QKV proj     2 * B * T * C * 3C  = 6 B T C²
                  attn scores  2 * B * T * T * C   = 2 B T² C
                  attn w-sum   2 * B * T * T * C   = 2 B T² C
                  out proj     2 * B * T * C * C   = 2 B T C²
                  subtotal: 8 B T C² + 4 B T² C

      MLP:        fc           2 * B * T * C * 4C  = 8 B T C²
                  proj         2 * B * T * 4C * C  = 8 B T C²
                  subtotal: 16 B T C²

    LM head (weight-tied):     2 * B * T * C * vocab_size

    Forward total = n_layer * (24 B T C² + 4 B T² C) + 2 B T C * vocab_size
    Backward ≈ 2× forward  →  multiply by 3
    Grad accum: multiply by grad_accum_steps
    """
    cfg = model.config
    n_layer   = cfg.n_layer
    C         = cfg.n_embd
    vocab     = cfg.vocab_size

    attn_flops = 8 * B * T * C * C + 4 * B * T * T * C
    mlp_flops  = 16 * B * T * C * C
    head_flops = 2 * B * T * C * vocab

    forward_flops = n_layer * (attn_flops + mlp_flops) + head_flops
    total_flops   = 3 * forward_flops * grad_accum_steps   # fwd + bwd
    return total_flops


class CostTracker:
    def __init__(self, model, B, T, grad_accum_steps,
                 num_gpus=1, gpu_cost_per_hr=2.50):
        self.flops_per_step   = compute_flops_per_step(model, B, T, grad_accum_steps) * num_gpus
        self.tokens_per_step  = B * T * grad_accum_steps * num_gpus
        self.num_gpus         = num_gpus
        self.gpu_cost_per_hr  = gpu_cost_per_hr   # total for all GPUs in the pod

        self._step_times: list[float] = []
        self._peak_memory_mb: list[float] = []
        self._cumulative_flops = 0
        self._wall_start = time.time()

    def step(self, step: int, dt_seconds: float):
        self._step_times.append(dt_seconds)
        self._cumulative_flops += self.flops_per_step

        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
            self._peak_memory_mb.append(peak_mb)
            torch.cuda.reset_peak_memory_stats()

    # ── derived metrics ───────────────────────────────────────────────────────

    @property
    def total_wall_seconds(self) -> float:
        return time.time() - self._wall_start

    @property
    def avg_step_ms(self) -> float:
        return (sum(self._step_times) / len(self._step_times) * 1000
                if self._step_times else 0.0)

    @property
    def tokens_per_sec(self) -> float:
        if not self._step_times:
            return 0.0
        return self.tokens_per_step / (sum(self._step_times) / len(self._step_times))

    @property
    def tflops_per_sec(self) -> float:
        if not self._step_times:
            return 0.0
        avg_dt = sum(self._step_times) / len(self._step_times)
        return self.flops_per_step / avg_dt / 1e12

    @property
    def tflops_per_sec_per_gpu(self) -> float:
        return self.tflops_per_sec / self.num_gpus

    @property
    def estimated_cost_so_far(self) -> float:
        return (self.total_wall_seconds / 3600) * self.gpu_cost_per_hr

    @property
    def peak_memory_mb(self) -> float:
        return max(self._peak_memory_mb) if self._peak_memory_mb else 0.0

    def cost_to_train_n_steps(self, n_steps: int) -> float:
        if not self._step_times:
            return 0.0
        avg_dt = sum(self._step_times) / len(self._step_times)
        hours  = (avg_dt * n_steps) / 3600
        return hours * self.gpu_cost_per_hr

    def cost_per_billion_tokens(self) -> float:
        if not self._step_times:
            return 0.0
        avg_dt    = sum(self._step_times) / len(self._step_times)
        cost_step = (avg_dt / 3600) * self.gpu_cost_per_hr
        tokens_B  = self.tokens_per_step / 1e9
        return cost_step / tokens_B if tokens_B > 0 else 0.0

    # ── output ────────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "steps_tracked":          len(self._step_times),
            "total_wall_seconds":     round(self.total_wall_seconds, 1),
            "avg_step_ms":            round(self.avg_step_ms, 1),
            "tokens_per_sec":         round(self.tokens_per_sec),
            "tflops_per_sec":         round(self.tflops_per_sec, 2),
            "tflops_per_sec_per_gpu": round(self.tflops_per_sec_per_gpu, 2),
            "cumulative_tflops":      round(self._cumulative_flops / 1e12, 2),
            "peak_memory_mb":         round(self.peak_memory_mb, 1),
            "estimated_cost_usd":     round(self.estimated_cost_so_far, 4),
            "cost_per_billion_tokens_usd": round(self.cost_per_billion_tokens(), 4),
            "num_gpus":               self.num_gpus,
            "gpu_cost_per_hr_usd":    self.gpu_cost_per_hr,
        }

    def print_summary(self):
        s = self.summary()
        print("\n── Cost Report ──────────────────────────────")
        print(f"  Steps tracked:        {s['steps_tracked']}")
        print(f"  Avg step time:        {s['avg_step_ms']:.1f} ms")
        print(f"  Tokens/sec:           {s['tokens_per_sec']:,}")
        print(f"  TFLOPs/sec (total):   {s['tflops_per_sec']:.2f}")
        print(f"  TFLOPs/sec/GPU:       {s['tflops_per_sec_per_gpu']:.2f}")
        print(f"  Cumulative TFLOPs:    {s['cumulative_tflops']:.2f}")
        if s['peak_memory_mb'] > 0:
            print(f"  Peak GPU memory:      {s['peak_memory_mb']:.0f} MB")
        print(f"  Wall time:            {s['total_wall_seconds']:.0f}s")
        print(f"  Cost so far:          ${s['estimated_cost_usd']:.4f}")
        print(f"  Cost/1B tokens:       ${s['cost_per_billion_tokens_usd']:.4f}")
        print("─────────────────────────────────────────────\n")

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.summary(), f, indent=2)
