"""
Scaling sweep: grad_accum_steps in [1, 2, 4, 8].
CPU demo — same tiny model as failure_modes.py.
Adapt model config and data loader for RunPod with real data.

Run:
    python3 experiments/scaling.py
    GPU_COST_PER_HR=2.50 python3 experiments/scaling.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

import json
import time
import torch

from failure_modes import LocalGPT, Config
from training_reliability.cost_tracker import CostTracker


CFG         = Config()
ACCUM_STEPS = [1, 2, 4, 8]
NUM_STEPS   = 100
B, T        = 4, 32
LR          = 3e-4
CLIP        = 1.0


def run_sweep(accum_steps: int, gpu_cost_per_hr: float) -> tuple:
    torch.manual_seed(42)
    model     = LocalGPT(CFG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    tracker   = CostTracker(model, B=B, T=T, grad_accum_steps=accum_steps,
                             num_gpus=1, gpu_cost_per_hr=gpu_cost_per_hr)
    records = []

    for step in range(NUM_STEPS):
        t0 = time.time()
        optimizer.zero_grad()
        step_loss = 0.0

        for _ in range(accum_steps):
            x = torch.randint(0, CFG.vocab_size, (B, T))
            y = torch.randint(0, CFG.vocab_size, (B, T))
            _, loss = model(x, y)
            (loss / accum_steps).backward()
            step_loss += loss.item() / accum_steps

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP).item()
        optimizer.step()

        dt = time.time() - t0
        tracker.step(step=step, dt_seconds=dt)

        records.append({
            "step":                step,
            "loss":                step_loss,
            "grad_norm":           grad_norm,
            "flops_cumulative":    tracker._cumulative_flops,
            "cost_usd_cumulative": tracker.estimated_cost_so_far,
        })

    return records, tracker


def write_jsonl(path: str, records: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def print_summary_table(rows: list):
    col_w = [10, 10, 12, 11, 18]
    header = (
        f"{'grad_accum':>{col_w[0]}} | "
        f"{'final_loss':>{col_w[1]}} | "
        f"{'total_tflops':>{col_w[2]}} | "
        f"{'avg_ms/step':>{col_w[3]}} | "
        f"{'cost_per_1B_tokens':>{col_w[4]}}"
    )
    sep = "-" * len(header)
    print()
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['grad_accum']:>{col_w[0]}} | "
            f"{r['final_loss']:>{col_w[1]}.4f} | "
            f"{r['total_tflops']:>{col_w[2]}.2f} | "
            f"{r['avg_ms_per_step']:>{col_w[3]}.1f} | "
            f"${r['cost_per_1b_tokens']:>{col_w[4] - 1}.4f}"
        )
    print()


def main():
    gpu_cost_per_hr = float(os.environ.get("GPU_COST_PER_HR", "2.50"))
    summary_rows    = []

    print("Scaling sweep: grad_accum_steps in [1, 2, 4, 8]")
    print(f"Config: block_size={CFG.block_size}, vocab_size={CFG.vocab_size}, "
          f"n_layer={CFG.n_layer}, n_head={CFG.n_head}, n_embd={CFG.n_embd}")
    print(f"Steps per sweep: {NUM_STEPS}  B={B}  T={T}  "
          f"gpu_cost_per_hr=${gpu_cost_per_hr:.2f}")

    for accum in ACCUM_STEPS:
        print(f"\n── accum={accum} " + "─" * 44)
        records, tracker = run_sweep(accum, gpu_cost_per_hr)

        out_path = f"logs/scaling/accum_{accum}/metrics.jsonl"
        write_jsonl(out_path, records)
        print(f"  wrote {len(records)} records → {out_path}")

        s = tracker.summary()
        row = {
            "grad_accum":         accum,
            "final_loss":         records[-1]["loss"],
            "total_tflops":       s["cumulative_tflops"],
            "avg_ms_per_step":    s["avg_step_ms"],
            "cost_per_1b_tokens": s["cost_per_billion_tokens_usd"],
        }
        summary_rows.append(row)
        print(f"  final_loss={row['final_loss']:.4f}  "
              f"total_tflops={row['total_tflops']:.2f}  "
              f"avg_ms={row['avg_ms_per_step']:.1f}  "
              f"cost/1B=${row['cost_per_1b_tokens']:.4f}")

    print_summary_table(summary_rows)

    summary_path = "logs/scaling/summary.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"Summary written → {summary_path}")


if __name__ == "__main__":
    main()
