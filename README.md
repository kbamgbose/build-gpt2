# build-gpt2

GPT-2 style decoder-only transformer implemented from scratch in PyTorch. Built to understand the architecture deeply — not as a wrapper around existing libraries, but as a ground-up implementation with correctness tests, isolated failure mode experiments, and scaling profiling on real hardware.

---

## What this repo demonstrates

- A complete GPT-2 implementation with manual causal self-attention, MLP blocks, weight tying, and a cosine LR schedule with linear warmup
- Correctness verified through invariant tests (causal masking, shape contracts, finite loss, clean gradients)
- Six architectural and training failure modes isolated, quantified, and documented with real measurements
- Scaling behavior profiled on 8x A100 80GB — context length, model width, and depth — with FlashAttention vs naive attention comparison
- Distributed training via PyTorch DDP with gradient accumulation across 8 GPUs on the FineWeb-Edu 10B dataset

---

## Repository structure

```
attention.py              # CausalSelfAttention — manual masking with full shape annotations
model.py                  # GPTConfig, MLP, Block, GPT — importable, no side effects
train.py                  # DDP training loop — 8 GPU, gradient accumulation, cosine LR
train_tiny.py             # Single-GPU training on input.txt — end-to-end smoke test
fineweb.py                # FineWeb-Edu 10B dataset tokenization and sharding

tests/
  test_transformer.py     # 5 invariant tests — causal mask, shapes, finite loss, clean grads
  test_failure_modes.py   # 4 pytest tests — each proves one architectural bug is detectable

experiments/
  failure_modes.py        # 6 isolated failure mode experiments with quantified diagnostics

profiling/
  profile_attention.py    # Naive vs FlashAttention — latency and memory across sequence lengths
  scaling_experiment.py   # Context length and model width scaling
  block_size_experiment.py# Depth scaling — ms/layer across n_layer = 6, 12, 24
  edge_cases.py           # Long sequences, dtype stability, large magnitude inputs

docs/
  attention_walkthrough.md        # 8-step derivation of causal self-attention
  failure_modes.md                # All 6 failure modes — symptom, root cause, detection, fix
  transformer_scaling_analysis.md # A100 profiling results with analysis
```

---

## Architecture

Standard GPT-2 decoder-only transformer. Key design choices:

- **Attention:** Multi-head causal self-attention with a registered lower-triangular bias buffer for masking. Manual implementation in `attention.py` with explicit shape annotations at every step.
- **MLP:** Two-layer feedforward with 4x expansion and GELU activation. Residual projection uses scaled initialization (`std *= (2 * n_layer) ** -0.5`) to prevent activation growth with depth.
- **Weight tying:** Token embedding and LM head share weights — reduces parameters and improves sample efficiency.
- **Optimizer:** AdamW with separate weight decay groups (2D params decay, biases/norms do not). Fused AdamW on CUDA where available.
- **Training:** DDP across 8 GPUs with gradient accumulation to reach `total_batch_size=524288` tokens per step. Cosine LR decay with 715-step linear warmup.

Default config (GPT-2 124M):

| Parameter | Value |
|-----------|-------|
| n_layer   | 12    |
| n_head    | 12    |
| n_embd    | 768   |
| block_size| 1024  |
| vocab_size| 50257 |

---

## Correctness tests

```bash
python -m pytest tests/test_transformer.py -v
```

Five invariant tests in `tests/test_transformer.py`:

| Test | What it proves |
|------|---------------|
| `test_causal_masking` | Future tokens do not affect past logits — corrupting positions t+1..T leaves positions 0..t unchanged |
| `test_attention_output_shape` | Attention maps `(B, T, C) → (B, T, C)` exactly |
| `test_forward_pass_shape` | Full model produces logits of shape `(B, T, vocab_size)` |
| `test_loss_is_finite` | Cross-entropy loss is finite on valid inputs |
| `test_no_nans_forward_and_backward` | No NaN or inf in logits or any gradient after one backward pass |

All tests use a minimal config (`n_layer=2, n_head=2, n_embd=64`) and run in under 2 seconds on CPU.

Four additional tests in `tests/test_failure_modes.py` prove each architectural failure mode is detectable:

```bash
python -m pytest tests/test_failure_modes.py -v
```

---

## Failure mode experiments

```bash
python experiments/failure_modes.py
```

Six experiments in `experiments/failure_modes.py`, each injecting one bug into a minimal model and measuring the outcome. All run on CPU in ~11 seconds. Full documentation in `docs/failure_modes.md`.

### Results summary

| Bug | Detection metric | Result |
|-----|-----------------|--------|
| No causal mask | Max logit diff at past positions after corrupting future tokens | `0.053` (correct: `0.000`) |
| Softmax before masking | Attention row sum deviation from 1.0 | `0.484` (correct: `0.000`) |
| No sqrt(d_k) scaling | Pre-softmax score std; attention entropy | std `2.11` unscaled vs `0.37` scaled; entropy collapses |
| Wrong view/transpose order | RuntimeError on forward pass | Raised immediately |
| High learning rate (1.0 vs 3e-4) | Loss over 10 steps | `5.55 → 173.8`; grad norm spikes to `44.2` |
| No gradient clipping | Max grad norm clipped vs unclipped | Invisible on small models; dangerous at scale |

The three silent bugs — softmax before masking, no sqrt scaling, and no gradient clipping — are the most dangerous: training still runs, loss still decreases, but the model is learning incorrectly or is fragile to scale.

---

## Scaling experiments

Results from an 8x A100 80GB run. Full tables and analysis in `docs/transformer_scaling_analysis.md`.

**FlashAttention vs naive attention** (`profiling/profile_attention.py`):

| seq_len | naive_ms | flash_ms | speedup | naive_mb | flash_mb |
|---------|----------|----------|---------|----------|----------|
| 1024    | 0.94     | 0.08     | 12.2x   | 226.1    | 38.3     |
| 4096    | 15.41    | 0.65     | 23.7x   | 3208.1   | 128.9    |

Naive memory is O(n²) in sequence length — it materializes the full `(T, T)` attention matrix in HBM. FlashAttention tiles the computation into SRAM and is O(n) in memory. At T=4096, naive uses 25x more memory.

**Context length scaling** (`profiling/scaling_experiment.py`): memory grows sub-quadratically in this model because it uses FlashAttention. Latency tracks O(n²) more closely.

**Depth scaling** (`profiling/block_size_experiment.py`): `ms/layer` decreases from `0.74 → 0.50` as layers increase from 6 to 24. Adding layers is O(n_layer) in compute, and the per-layer cost falls as the GPU amortizes kernel launch overhead across more sequential work.

---

## Running the repo

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run all correctness tests:**
```bash
python -m pytest tests/ -v
```

**Run failure mode experiments:**
```bash
python experiments/failure_modes.py
```

**Train on Shakespeare (single GPU smoke test):**
```bash
python train_tiny.py
```

**Tokenize FineWeb-Edu dataset (requires CUDA pod):**
```bash
python fineweb.py
```

**Full distributed training (8 GPU):**
```bash
torchrun --standalone --nproc_per_node=8 train.py
```

---

## Limitations

- `attention.py` uses manual masking, not FlashAttention — the manual implementation is for transparency; swap to `F.scaled_dot_product_attention` for production training
- No KV cache — inference re-computes keys and values for all previous tokens at every step, making generation O(n²) in sequence length
- No inference utilities — no batched generation, sampling strategies, or beam search
- `train.py` checkpoints at the end of training only — no mid-run recovery
- The FineWeb run completed but no HellaSwag eval was instrumented; loss curves exist but no downstream benchmark numbers

---

## Next steps

- **KV cache:** cache key/value tensors during generation to reduce inference from O(n²) to O(n) per token
- **FlashAttention comparison:** wire `F.scaled_dot_product_attention` into `attention.py` behind a flag and re-run the profiling suite to compare against the manual implementation on the same model
- **Inference batching:** implement batched generation with padding masks and measure throughput vs latency tradeoffs across batch sizes
