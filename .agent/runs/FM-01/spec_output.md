# FM-01 Spec Output

**Status:** approved and implemented
**Output file:** experiments/failure_modes.py
**Runtime:** ~11s on CPU

---

## Overview

Standalone script that runs 6 isolated experiments, each injecting one failure mode into a minimal GPT-2-style transformer. All experiments share the same config and seed. No production imports. All exceptions caught and reported.

## Config

```
block_size=32, vocab_size=256, n_layer=2, n_head=2, n_embd=64
device=cpu, seed=42
```

## Invariants

1. BaseAttention is a local copy of production logic — any divergence is intentional (the bug under test)
2. LocalGPT accepts `attn_class` parameter injected into each LocalBlock
3. train_n_steps() checks `torch.isfinite(loss)` before calling backward — prevents silent NaN propagation
4. `clip=None` disables gradient clipping; `clip=1.0` enables it
5. Each experiment prints exactly one labeled block with quantified diagnostics
6. Script exits with code 0 regardless of per-experiment errors

## Failure Modes

### BUG 1 — NoCausalMask
Omits `masked_fill` entirely. Detection: run two forward passes on identical input except positions 8+. If logits at positions 0–7 differ, future tokens leaked.

Diagnostic: `leaked: bool`, `max logit diff at past positions: float`

### BUG 2 — SoftmaxBeforeMask
Applies softmax to raw scores before applying the causal mask, then re-normalizes. Rows that include future tokens will not sum to 1.0 after the second normalization.

Diagnostic: `attention row sum deviation from 1.0: float` (expect ~0.48 with 2 heads, block_size=32)

### BUG 3 — NoSqrtScaling
Omits the `* (1.0 / math.sqrt(head_size))` factor. Raw dot products have std proportional to sqrt(d_k), pushing softmax into saturation. Low entropy = uniform attention collapsed to one token.

Diagnostics: `pre-softmax score std (unscaled)`, `pre-softmax score std (scaled)`, `attention entropy (unscaled)`

### BUG 4 — WrongTranspose
Does `q.transpose(1,2).view(B, T, n_head, hs)` instead of the correct `q.view(B, T, n_head, hs).transpose(1,2)`. The transposed tensor is non-contiguous; `.view()` on a non-contiguous tensor raises RuntimeError.

Diagnostic: `RuntimeError raised (expected): <message>`

### BUG 5 — High Learning Rate
Baseline model with `lr=1.0` instead of `3e-4`. Runs 10 steps, prints step/loss/grad_norm table. Loss should visibly diverge.

Diagnostic: step-by-step loss and grad_norm table, `nan_occurred: bool`

### BUG 6 — No Gradient Clipping
Runs baseline model 1 step with and without `clip_grad_norm_`. Reports max grad norm from each run and ratio.

Diagnostic: `max grad norm (unclipped)`, `max grad norm (clipped)`, `ratio`

## Actual Output

```
FM-01: Failure Mode Experiments
Config: block_size=32, vocab_size=256, n_layer=2, n_head=2, n_embd=64
Device: CPU

============================================================
  [BUG 1] No causal mask
============================================================
  final loss:  5.5748
  leaked:      True
  max logit diff at past positions: 0.052683
  (correct model: leaked=False, max_diff≈0.000000)

============================================================
  [BUG 2] Softmax before masking
============================================================
  final loss:  5.5710
  attention row sum deviation from 1.0: 0.4839
  (correct model: deviation≈0.0000)

============================================================
  [BUG 3] No sqrt(d_k) scaling
============================================================
  final loss:  5.5770
  pre-softmax score std (unscaled): 2.1118
  pre-softmax score std (scaled):   0.3733
  attention entropy (unscaled): 1.4202
  (lower entropy = more saturated = vanishing gradients)

============================================================
  [BUG 4] Wrong view/transpose order
============================================================
  RuntimeError raised (expected):
  The size of tensor a (32) must match the size of tensor b (2) at non-singleton dimension 3

============================================================
  [BUG 5] High learning rate  (lr=1.0 vs normal 3e-4)
============================================================
  step        loss   grad_norm
  ------------------------------
     0      5.5514      1.7207
     1     42.8087     23.2321
     2     73.8712     22.3213
     3     91.5250     36.4169
     4     90.0542     17.3873
     5     98.3008     35.3817
     6    120.3836     37.3173
     7    127.5184     36.2885
     8    125.7747     44.2005
     9    173.7684     37.6538
  nan_occurred: False

============================================================
  [BUG 6] No gradient clipping
============================================================
  max grad norm (unclipped): 1.7207
  max grad norm (clipped):   1.7207
  ratio (unclipped/clipped): 1.00x
  final loss (unclipped):    5.5765
  final loss (clipped):      5.5777

============================================================
  All experiments complete.
============================================================
```

Exit code: 0
