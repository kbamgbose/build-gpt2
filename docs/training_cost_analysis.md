# Training Cost Analysis — GPT-2 124M on A100 SXM

Measured on RunPod A100 SXM 80GB pods. Model: GPT-2 124M, trained on FineWeb-Edu 10B with `torch.compile` + bfloat16. Each run: 2,000 steps, total batch size 524,288 tokens (2^19), B=16, T=1024.

---

## Results

| Metric | 1× A100 | 4× A100 | Scaling |
|---|---|---|---|
| Tokens/sec | 132,058 | 517,857 | **3.92×** |
| Avg step time | 3,970 ms | 1,012 ms | 3.92× faster |
| Wall time (2k steps) | 7,946s (~2.2 hr) | 2,030s (~34 min) | 3.91× |
| Peak GPU memory | 24,965 MB | 23,864 MB | — |
| Cost/hr | $1.52 | $5.99 | — |
| Cost/run | $3.35 | $3.38 | ~1:1 |
| **Cost/1B tokens** | **$3.20** | **$3.21** | **~1:1** |

---

## Key Findings

**Near-linear scaling at 4 GPUs.** 4× A100s deliver 3.92× the throughput of 1× A100 — 98% scaling efficiency. This is expected on A100 SXM with NVLink: inter-GPU bandwidth (600 GB/s) is high enough that NCCL all-reduce on a 124M parameter model is not the bottleneck.

**Cost/1B tokens is hardware-count invariant.** $3.20 vs $3.21 — statistically identical. The pod cost scales linearly with GPU count and so does throughput, so cost-per-token is constant. The only reason to use more GPUs is wall-clock time, not cost efficiency.

**MFU: ~36% on both configurations.** Real per-GPU throughput is ~112 TFLOPs/s against an A100 SXM bf16 peak of 312 TFLOPs/s. This is consistent across 1 and 4 GPU runs. The gap to peak (~60-65% in Karpathy's reference implementation) is attributable to:
- Standard attention (no FlashAttention) — memory bandwidth bound at T=1024
- `torch.compile` warmup amortized over 2,000 steps still skews the average
- CPU-side overhead (data loading, logging) in the step timer

**Memory headroom is substantial.** Peak 24-25 GB on an 80 GB card at B=16, T=1024 — 3× headroom before hitting the memory wall. Batch size could be increased significantly before needing gradient checkpointing.

---

## Dataset-Size Cost and Wall Time

Extrapolated from the measured $3.20/1B tokens and the throughput numbers above. Cost is hardware-count invariant; GPU count only buys time.

| Tokens | 1× A100 wall time | 4× A100 wall time | 8× A100 wall time | Total cost |
|---|---|---|---|---|
| 1B   | 2.1 hr   | 32 min   | 16 min   | $3.20 |
| 2.5B (Chinchilla-optimal for 124M) | 5.3 hr  | 1.3 hr  | 40 min | $8.00 |
| 10B (FineWeb-Edu, 4× over-trained) | 21 hr   | 5.4 hr  | 2.7 hr | $32 |
| 100B | 8.7 days | 53 hr   | 27 hr  | $320 |
| 300B (GPT-3-class token budget) | 26 days | 6.7 days | 3.4 days | $960 |

### Chinchilla-optimal point

Hoffmann et al. (2022) showed that compute-optimal training uses ~20 tokens per parameter. For 124M parameters this is ~2.5B tokens. Past that, loss continues to drop but with sharply diminishing returns per dollar.

The nano-gpt FineWeb-Edu 10B run trains 4× past Chinchilla-optimal. This is the standard "over-train small models so inference is cheap and easy to deploy" pattern (Llama-2 7B was trained on 2T tokens, ~30× past Chinchilla-optimal for 7B).

If the goal is the lowest-loss model for a fixed budget, the right question is usually "bigger model, fewer tokens" rather than "small model, more tokens." That tradeoff curve is not measured here; it requires sweeping model size, which is partially covered in `transformer_scaling_analysis.md`.

### Assumptions

- 124M GPT-2 (`n_layer=12, n_head=12, n_embd=768`), B=16, T=1024, bfloat16, `torch.compile`.
- Steady-state throughput: ~132k tokens/sec on 1× A100 SXM, ~518k on 4× A100. The 8× column is linear extrapolation from the 1×/4× pair (98% scaling efficiency observed at 4×), not directly measured.
- Pricing is RunPod on-demand A100 SXM 80GB. Spot instances run ~30-50% cheaper at the cost of preemption.
- The per-token cost is steady-state. Runs shorter than ~1B tokens pay a `torch.compile` warmup tax in the first 100-200 steps, so real cost on a 1B run is ~5-10% higher than the table suggests.
- Multi-GPU scaling at 8× assumes NVLink topology comparable to 4× A100 SXM. PCIe-attached A100s would not hit 98% efficiency.
- Loss trajectory is not measured here. Published nano-gpt reproductions land around 3.0-3.1 train loss at 10B tokens on FineWeb-Edu.

---

## Hardware & Config

- **GPU**: NVIDIA A100 SXM 80GB (RunPod)
- **Model**: GPT-2 124M (`n_layer=12, n_head=12, n_embd=768, vocab_size=50304`)
- **Data**: FineWeb-Edu 10B, pre-tokenized GPT-2 BPE shards
- **Precision**: bfloat16 (`torch.autocast`)
- **Compile**: `torch.compile(model)` — compiled before DDP wrap
- **Distributed**: `torchrun` + NCCL, `DistributedDataParallel`
- **Optimizer**: AdamW, lr=6e-4, cosine decay, 715 warmup steps
- **Gradient clipping**: 1.0
