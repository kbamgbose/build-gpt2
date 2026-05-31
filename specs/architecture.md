# Architecture Overview

Known-good patterns and design decisions. Read this before adding a new component or refactoring an existing one.

---

## Component map

```
nano-gpt/
├── attention.py          CausalSelfAttention — fused QKV, causal mask, scaled dot-product
├── model.py              GPT, Block, MLP, GPTConfig — standalone, importable
├── train.py              Training loop — DDP, grad accum, cosine LR, NaN rollback
├── train_tiny.py         Tiny smoke-test loop (synthetic data, no dataset)
├── training_reliability/ Optional monitoring module (cost, loss rate, grad norm, anomaly)
├── profiling/            One-shot experiments for scaling analysis
├── experiments/          Failure mode demos, scaling runs
├── tests/                Invariant tests (fast, CPU-only)
├── docs/                 Analysis write-ups (attention walkthrough, scaling results, costs)
├── specs/                Design decisions and known-good patterns (this file)
└── .agent/               Agent rules, constitution, env example
```

---

## Data flow

```
DataLoaderLite (shard-based .npy files)
    │
    ▼
token batch (B, T)  →  GPT.forward()  →  logits (B, T, vocab_size)
                                      →  cross-entropy loss
                                      ▼
                               loss.backward()
                                      ▼
                         clip_grad_norm_ (max=1.0)
                                      ▼
                              AdamW.step()
                                      ▼
                         CostTracker / monitors (optional)
                                      ▼
                         checkpoint every CHECKPOINT_INTERVAL steps
```

---

## Key design decisions

### Residual stream scaling
Projection layers at the end of each residual branch (`MLP.c_proj`, `CausalSelfAttention.c_proj`) use a reduced init std: `0.02 * (2 * n_layer)^-0.5`. This keeps the residual stream variance stable as depth grows. The flag `NANOGPT_SCALE_INIT = 1` on those layers is what triggers the scaled init in `_init_weights`.

### vocab_size=50304 (not 50257)
Training uses 50304 to align the vocabulary to a multiple of 64. The logit projection (`lm_head`) is a `(n_embd, vocab_size)` matmul — alignment to powers of 2 improves GPU kernel efficiency. The 47 extra tokens are never sampled; they absorb no-op probability mass.

### Fused QKV projection
A single `Linear(n_embd, 3 * n_embd)` produces Q, K, V for all heads at once, then `split` separates them. This is one matmul instead of three — faster on GPU, same result.

### Weight tying
`wte.weight = lm_head.weight` — the input embedding and output projection share parameters. Standard in GPT-2. Cuts ~40M parameters and improves generalization (the model learns a consistent token representation used for both encoding and prediction).

### Grad accum as a first-class parameter
`grad_accum_steps = total_batch_size // (B * T * ddp_world_size)` — it is derived, not set. This means effective batch size is always `total_batch_size` regardless of how many GPUs are used. The assertion `total_batch_size % (B * T * ddp_world_size) == 0` enforces this at startup.

### NaN rollback
On a non-finite loss, training rolls back to the last checkpoint and halves LR. This handles transient spikes without stopping the run. Repeated rollbacks (tracked by `rollback_count`) indicate a structural problem — the final `print` surfaces the count.

### training_reliability as optional
`from training_reliability import ...` is wrapped in `try/except ImportError`. This lets `train.py` run in environments without the module (e.g., quick single-machine tests) while still using it when available. This pattern is intentional — do not remove the guard.

---

## Known-good hyperparameter baseline (GPT-2 small scale)

| Parameter | Value | Note |
|-----------|-------|------|
| n_layer | 12 | |
| n_head | 12 | |
| n_embd | 768 | |
| block_size | 1024 | |
| total_batch_size | 524288 tokens | ~0.5M |
| max_lr | 6e-4 | |
| min_lr | 6e-5 | 10% of max_lr |
| warmup_steps | 715 | |
| max_steps | 19073 | ≈1 epoch FineWeb-10B |
| weight_decay | 0.1 | on ≥2D params only |
| grad_clip | 1.0 | |
| betas | (0.9, 0.95) | AdamW |

---

## Profiling results summary

Full data: `docs/transformer_scaling_analysis.md`

- Sequence length is O(n²) in attention, O(n) everywhere else — it is the most expensive dimension to scale.
- Model width (n_embd) is O(n²) in all matmuls — second most expensive.
- Model depth (n_layer) is O(n) — cheapest to scale.
- FlashAttention reduces memory from O(n²) to O(n) with no FLOPs change. Speedup grows from ~9x at T=256 to ~24x at T=4096.
- float16 overflows on large magnitude inputs; bfloat16 matches float32's exponent range and is safe for training.
