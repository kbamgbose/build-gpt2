# build-gpt2

GPT-2 (124M) implemented from scratch in PyTorch. No library wrappers. Trained on FineWeb-Edu 10B across 8x A100 80GB with DDP.

The goal was not to reproduce the paper. It was to understand what breaks, why it breaks silently, and what the hardware actually does at scale.

---

## Results

**FlashAttention vs naive at increasing sequence length:**

| seq_len | naive_ms | flash_ms | speedup | naive_mb | flash_mb |
|---------|----------|----------|---------|----------|----------|
| 1024    | 0.94     | 0.08     | 12.2x   | 226.1    | 38.3     |
| 4096    | 15.41    | 0.65     | 23.7x   | 3208.1   | 128.9    |

Naive attention materializes the full `(T, T)` score matrix in HBM. Memory is O(n²). FlashAttention tiles the computation into SRAM and never materializes it. Memory is O(n). At T=4096 the difference is 25x.

**Depth scaling (ms/layer across n_layer):**

| n_layer | ms/layer |
|---------|----------|
| 6       | 0.74     |
| 12      | 0.62     |
| 24      | 0.50     |

Per-layer cost falls as depth increases. Kernel launch overhead is amortized across more sequential work. This matters when estimating compute budgets: deeper models are cheaper per layer than they appear.

---

## GPU benchmarks

Measured on RunPod A100 SXM 80GB. GPT-2 124M, FineWeb-Edu 10B, bfloat16, `torch.compile`.

| Config | Tokens/sec | TFLOPs/GPU | Cost/1B tokens | MFU |
|--------|-----------|------------|----------------|-----|
| 1× A100 ($1.52/hr) | 132,058 | 112.9 | $3.20 | 36% |
| 4× A100 ($5.99/hr) | 517,857 | 110.7 | $3.21 | 35% |

4× A100 delivers **3.92× throughput** (98% linear scaling via NVLink). Cost/1B tokens is hardware-count invariant — more GPUs buys wall-clock time, not cost efficiency. Full analysis in `docs/training_cost_analysis.md`.

---

## Fault-tolerant training

`train.py` checkpoints every N steps and recovers from NaN gradients without restarting the run.

- Checkpoint saved every 500 steps (configurable via `CHECKPOINT_INTERVAL`), keeps last 5
- NaN detected post-accumulation via `check_anomaly`; rolls back to latest checkpoint, halves LR, continues
- Resume from checkpoint on restart — data loader position saved and restored
- Demonstrated end-to-end in `train_fault_demo.py`: NaN injected at step 80, rollback to step 80, training continues with halved LR

---

## Training reliability

Live monitors attached to the training loop that fire during a run, not after. No diagnosing by eyeballing loss curves.

| Monitor | Condition | Warning |
|---------|-----------|---------|
| `loss_rate` | `loss_t / loss_{t-10} > 1.5` | loss spike |
| `loss_rate` | `<1% decrease over 20 steps` | training stalled |
| `grad_norm` | `grad_norm > 10.0` | gradient spike |
| `grad_norm` | `grad_norm < 1e-6` | vanishing gradients |
| `anomaly` | NaN or Inf in loss or grad_norm | numerical failure |
| `anomaly` | `activation_std > 50.0` | activation explosion |
| `anomaly` | `activation_std < 0.01` after step 5 | activation collapse |

Each monitor returns a `Warning(step, monitor, message, values)` namedtuple. Warnings print live and write to `logs/<experiment>/warnings.jsonl`. `training_reliability/postmortem.py` reads those logs and generates structured failure reports.

Four training failure experiments run against this monitor layer:

| Experiment | Perturbation | Expected failure |
|------------|--------------|-----------------|
| `high_lr`  | lr=1.0 vs 3e-4 | gradient explosion → NaN |
| `low_lr`   | lr=1e-7 vs 3e-4 | stalled training |
| `bad_init` | init std=1.0 vs 0.02 | activation explosion |
| `no_clip`  | clip=None vs 1.0 | gradient instability |

Full analysis in `postmortems/training_instability.md`.

---

## Architecture failure modes

Six bugs injected into a minimal model, each measured rather than described. Full writeup in `docs/failure_modes.md`.

| Bug | Metric | Broken | Correct |
|-----|--------|--------|---------|
| No causal mask | Max logit diff at corrupted positions | 0.053 | 0.000 |
| Softmax before masking | Attention row sum deviation | 0.484 | 0.000 |
| No sqrt(d_k) scaling | Pre-softmax score std | 2.11 | 0.37 |
| Wrong view/transpose order | Forward pass | RuntimeError | Passes |
| LR = 1.0 vs 3e-4 | Loss after 10 steps | 5.55 to 173.8 | Stable |
| No gradient clipping | Max grad norm | Undetectable at small scale | Dangerous at large scale |

The three silent failures are the ones that matter. Softmax before masking, missing sqrt(d_k), and no gradient clipping all allow training to proceed and loss to decrease while the model is wrong or fragile. The hard crash from wrong tensor ordering is the easiest failure to ship.

---

## Architecture

Decoder-only transformer, GPT-2 124M config.

| Parameter  | Value  |
|------------|--------|
| n_layer    | 12     |
| n_head     | 12     |
| n_embd     | 768    |
| block_size | 1024   |
| vocab_size | 50257  |

**Attention:** Manual causal self-attention in `attention.py`. Lower-triangular mask registered as a buffer (not a parameter). Shape annotations at every step. Manual implementation kept for transparency; production use should delegate to `F.scaled_dot_product_attention`.

**MLP:** Two-layer feedforward, 4x hidden expansion, GELU activation. Residual projection initialized with `std *= (2 * n_layer) ** -0.5`. Without this, activation variance grows with depth and training destabilizes.

**Weight tying:** Token embedding and LM head share the same weight matrix. Reduces parameters by ~38M at this vocab size and improves sample efficiency.

**Optimizer:** AdamW. Weight decay applied to 2D parameters only; biases and norms excluded. Fused AdamW kernel used on CUDA when available.

**Training:** DDP across 8 GPUs. Gradient accumulation to `total_batch_size = 524,288` tokens per step. Cosine LR decay with 715-step linear warmup.

---

## Correctness tests

```bash
python -m pytest tests/ -v
```

All tests use `n_layer=2, n_head=2, n_embd=64`. Runs in under 2 seconds on CPU.

`tests/test_transformer.py` checks invariants:

| Test | Invariant |
|------|-----------|
| `test_causal_masking` | Corrupting positions t+1..T leaves logits at 0..t unchanged |
| `test_attention_output_shape` | `(B, T, C) -> (B, T, C)` |
| `test_forward_pass_shape` | Logits are `(B, T, vocab_size)` |
| `test_loss_is_finite` | Cross-entropy is finite on valid input |
| `test_no_nans_forward_and_backward` | No NaN or inf in logits or gradients after one backward pass |

`tests/test_failure_modes.py` proves each bug in the failure modes table is detectable programmatically. Not just described, caught.

---

## Repo structure

```
attention.py              # CausalSelfAttention, manual masking, shape annotations
model.py                  # GPTConfig, MLP, Block, GPT, no side effects
train.py                  # DDP loop, 8 GPU, gradient accumulation, cosine LR
train_baseline.py         # Baseline training loop for reliability experiments (5 modes)
train_tiny.py             # Single-GPU smoke test on input.txt
fineweb.py                # FineWeb-Edu 10B tokenization and sharding

training_reliability/
  __init__.py             # Warning namedtuple
  loss_rate.py            # Loss spike and stall detection
  grad_norm.py            # Gradient spike and vanishing gradient detection
  anomaly.py              # NaN/Inf and activation std anomaly detection
  postmortem.py           # Reads logs/, generates structured failure reports

tests/
  test_transformer.py     # Invariant tests
  test_failure_modes.py   # Failure mode detection tests

experiments/
  failure_modes.py        # 6 injection experiments with measurements

profiling/
  profile_attention.py    # FlashAttention vs naive across sequence lengths
  scaling_experiment.py   # Context length and width scaling
  block_size_experiment.py# Depth scaling, ms/layer
  edge_cases.py           # Long sequences, dtype stability, large inputs

postmortems/
  training_instability.md # Per-failure analysis: what broke, how it was caught, why

docs/
  attention_walkthrough.md        # 8-step derivation of causal self-attention
  failure_modes.md                # Full failure mode analysis
  transformer_scaling_analysis.md # A100 profiling results
```

---

## Running

```bash
pip install -r requirements.txt

# Correctness tests
python -m pytest tests/ -v

# Architecture failure mode experiments (~11s on CPU)
python experiments/failure_modes.py

# Baseline training run (control — should converge cleanly)
python train_baseline.py baseline

# Training failure experiments (run each independently)
python train_baseline.py high_lr
python train_baseline.py low_lr
python train_baseline.py bad_init
python train_baseline.py no_clip

# Generate structured failure reports from logs/
python training_reliability/postmortem.py

# Single-GPU smoke test
python train_tiny.py

# Tokenize FineWeb-Edu (requires CUDA)
python fineweb.py

# Distributed training
torchrun --standalone --nproc_per_node=8 train.py
```

---

## Limitations

- `attention.py` uses manual masking, not `F.scaled_dot_product_attention`. Kept for transparency; not suitable for production training.
- No KV cache. Inference recomputes keys and values for all previous tokens at each step: O(n²) in sequence length.
- No inference utilities. No batched generation, sampling strategies, or beam search.
- `train.py` checkpoints at end of training only. No mid-run recovery.
- FineWeb run completed with no HellaSwag eval instrumented. Loss curves exist but no downstream benchmark numbers.

---

## Next steps

- **KV cache:** cache key/value tensors during generation to reduce inference from O(n²) to O(n) per token
- **FlashAttention comparison:** wire `F.scaled_dot_product_attention` into `attention.py` behind a flag and re-run the profiling suite against the manual implementation on the same model
- **Inference batching:** batched generation with padding masks; measure throughput vs latency tradeoffs across batch sizes
- **PagedAttention and continuous batching:** memory-bound serving
- **HellaSwag eval:** validate the FineWeb training run against paper numbers
