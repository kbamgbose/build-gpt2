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

## Full Training Cost Projections

At steady-state $3.20/1B tokens and FineWeb-Edu 10B tokens:

| Config | Est. total cost | Est. wall time |
|---|---|---|
| 1× A100 ($1.52/hr) | ~$32 | ~22 hr |
| 4× A100 ($5.99/hr) | ~$32 | ~5.5 hr |
| 8× A100 (~$12/hr) | ~$32 | ~2.7 hr |

Cost is constant across GPU counts. GPU count only buys time.

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
