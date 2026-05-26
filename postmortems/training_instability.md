# Training Instability Postmortems

Four controlled failure experiments on a 2-layer mini-GPT (block_size=32, vocab_size=256, n_embd=64) trained on CPU for 200 steps. Each experiment changes exactly one variable from the baseline (lr=3e-4, init_std=0.02, clip=1.0). Monitors ran live during every experiment.

Reproduce any experiment:
```bash
python train_baseline.py [high_lr|low_lr|bad_init|no_clip]
python training_reliability/postmortem.py
```

---

## Failure 1: LR Too High — Gradient Explosion

**Experiment:** `high_lr` — lr=1.0 vs baseline 3e-4

### What broke

Loss diverged from 5.56 to 159 by step 46. Activations exploded from std=0.033 to 375,000 by step 200. The model never recovered to a useful state, even though the cosine LR schedule eventually decayed to 0.1 and loss partially fell back to ~6 by step 200. Activation std did not recover — it remained above 370,000 at final step, meaning the model's internal representations were destroyed even as the loss number improved.

### How you noticed

| Step | Monitor | Signal | Value |
|------|---------|--------|-------|
| 3 | `anomaly` | activation explosion | act_std=284 (threshold: 50) |
| 7 | `grad_norm` | gradient spike | grad_norm=13.1 (threshold: 10) |
| 10 | `loss_rate` | loss spike | rate=6.24x (loss: 5.56 → 34.7) |

`anomaly` fired first at step 3 — before the gradient norm even crossed threshold. Activation std blew up before the gradient norms did because the large LR update to weights immediately amplified forward-pass activations in the next step.

### Why it broke

With lr=1.0, each AdamW parameter update is ~3,000x larger than normal. The weight update at step 0 lands far outside the basin of the loss landscape. On step 1, the forward pass through those displaced weights produces logits with far larger magnitude, which propagates into large activations. The loss then reflects this mismatch (cross-entropy of extreme logits). Gradients computed from those logits are large, causing the next update to be even more extreme. This is a positive feedback loop.

Gradient clipping at 1.0 was still active (high_lr only changes LR, not clip). This is why the model didn't go NaN — clipping bounded each update's gradient contribution. But clipping cannot prevent the underlying problem: the LR multiplies after clipping, so updates are still 1.0 × clipped_grad rather than 3e-4 × clipped_grad. Clipping buys time but does not fix a runaway LR.

### What fixed it

Lower LR to 3e-4. The cosine schedule with warmup is itself a partial fix — as LR decayed from 1.0 toward 0.1, the loss partially recovered. A warmup that starts from 0 and ramps up slowly (not from a high value) prevents the first few steps from seeing lr=1.0.

### What to monitor in production

- `activation_std > 50` within the first 10 steps is a definitive early signal — faster than waiting for grad_norm or loss to spike
- `loss_rate > 1.5` by step 10-15 confirms divergence
- If loss recovers but activation_std does not, the model is numerically damaged even if the loss number looks acceptable

---

## Failure 2: LR Too Low — Stalled Training

**Experiment:** `low_lr` — lr=1e-7 vs baseline 3e-4

### What broke

Loss stayed in the range [5.508, 5.594] for all 200 steps. The starting loss (5.557) reflects the entropy of a uniform distribution over 256 tokens: ln(256) ≈ 5.545. The model never moved off this prior. By step 200, loss was 5.571 — indistinguishable from step 0.

Gradient norms were completely healthy (1.6–1.9 throughout). Activation std was completely stable (0.033 throughout). The optimizer was running, gradients were flowing, but nothing was happening.

### How you noticed

| Step | Monitor | Signal | Value |
|------|---------|--------|-------|
| 19 | `loss_rate` | training stalled | rate=0.999 (loss[19]/loss[0]=0.999) |
| 20–199 | `loss_rate` | training stalled | fired every step thereafter |

The stall check fires when the loss over the last 20 steps shows less than 1% decrease. It became available at step 19 (requiring 20 steps of history) and fired continuously from that point forward.

Crucially: no gradient or anomaly monitor fired. Stalled training looks completely healthy by every metric except loss progress.

### Why it broke

lr=1e-7 means each parameter update is ≈3e-4/1e-7 = 3,000x too small. With AdamW, the effective update size is lr × (gradient / (sqrt(second_moment) + ε)). At lr=1e-7, this is on the order of 1e-7, which is smaller than floating point noise at float32 precision for typical weight magnitudes (~0.02). The weights are not moving in any meaningful sense.

The LR schedule compounds this: the warmup phase linearly ramps from 0 to 1e-7, meaning the first 10 steps receive even smaller updates (1e-8 to 1e-7).

### What fixed it

Increase LR by at least 3 orders of magnitude. For this model, 3e-4 is appropriate. A useful heuristic: starting loss should decrease by at least a few percent within the first 50 steps. If it doesn't, LR is too low (or the data has no learnable signal).

### What to monitor in production

- Loss stall is a silent failure — all other metrics look normal
- Monitor `loss[t] / loss[t-20] > 0.99` for 3+ consecutive windows: training is not learning
- Also track absolute LR value — if it's below ~1e-6 for typical transformer sizes, expect stall
- At large scale, distinguishing "still in warmup" from "genuinely stalled" requires careful baseline expectations per step range

---

## Failure 3: Bad Initialization — Activation Explosion

**Experiment:** `bad_init` — init_std=1.0 vs baseline 0.02

### What broke

The model was broken before training began. At step 0:
- Loss: 23.65 (vs baseline ~5.56) — the model's random predictions were far worse than uniform
- activation_std: 154.5 (vs baseline ~0.033) — activations were 4,600x too large
- grad_norm: 91.6 (vs baseline ~1.7) — gradients were 53x too large

After 200 steps of training, loss reached ~19-21 (the model partially learned) but never approached the baseline's starting loss of 5.56. Activation std oscillated between 150–165 throughout, never recovering. The model was learning, but from a starting position so far from a good basin that 200 steps was insufficient to escape it.

### How you noticed

| Step | Monitor | Signal | Value |
|------|---------|--------|-------|
| 0 | `grad_norm` | gradient spike | grad_norm=91.6 (threshold: 10) |
| 0 | `anomaly` | activation explosion | act_std=154.5 (threshold: 50) |

Both monitors fired on the very first step — before any training had occurred. This is detectable at initialization time, before the first gradient update.

### Why it broke

Standard GPT-2 initialization uses std=0.02, with residual projections scaled by `(2 * n_layer)^{-0.5}` to keep activation variance stable as depth increases. With init_std=1.0, every linear weight has ~50x larger magnitude.

In the forward pass, each matrix multiply scales the activation magnitude by approximately `std × sqrt(fan_in)`. For this model: std=1.0, fan_in=64, so each layer multiplies variance by ~8. After 2 layers plus the LM head, activations are ~8^3 = 512x larger than expected — consistent with the observed 154 vs 0.033 ratio (~4,600x). Cross-entropy with logits of this magnitude produces losses well above ln(vocab_size), which is why loss started at 23.65 instead of 5.56.

The model cannot efficiently learn from this starting point because the gradients are computed in a regime where the loss landscape is essentially flat or extremely steep — the normal gradient descent assumptions break down.

### What fixed it

Use std=0.02 for all linear weights, with `std *= (2 * n_layer) ** -0.5` for residual projections. This keeps activation variance approximately constant through depth, so the model starts in a region of the loss landscape where gradient information is meaningful.

### What to monitor in production

- Check activation_std and grad_norm at step 0, before updating — these are initialization diagnostics, not training diagnostics
- If `activation_std > 50` at step 0, the model has not been initialized correctly
- `initial_loss >> ln(vocab_size)` is a direct indicator: the model's random weights are producing worse-than-uniform predictions, which means the forward pass is numerically saturated
- A sanity check before training: run one forward pass with no gradient update and verify loss ≈ ln(vocab_size) ± 0.5

---

## Failure 4: No Gradient Clipping — Gradient Instability

**Experiment:** `no_clip` — clip=None vs baseline clip=1.0

### What broke

At small scale with lr=3e-4, almost nothing visibly broke. Loss stayed flat at ~5.55 (same as low_lr — the model was not learning efficiently on random data). Gradient norms stayed below 10.0 throughout, ranging from ~1.7 down to ~0.8. No gradient spike was detected.

The only detectable signal was a slow drift in activation_std: from 0.033 at step 0 to 0.12 by step 200. This 3.6x drift is subtle — activation_std never crossed the explosion threshold of 50.

### How you noticed

| Step | Monitor | Signal | Value |
|------|---------|--------|-------|
| 19 | `loss_rate` | training stalled | (same signal as low_lr) |

The monitor that fired was the stall detector, not the gradient instability detector. The correct failure mode was not detected by the monitors as configured.

### Why it broke (and why it's hard to see at small scale)

Gradient clipping matters at scale, not at small scale with conservative LR. With a 2-layer, 64-dim model and lr=3e-4, gradient norms are naturally small (1–2) and do not spike above the 10.0 threshold. Clipping at 1.0 would trim nothing in this regime. The instability is latent, not active.

The slow drift in activation_std is the real signal. Without clipping, occasional large-gradient steps (visible in the no_clip run: grad_norm=1.74 at step 0, ~0.8 by step 200 as the LR schedule decayed) are not bounded. At small scale, these rare spikes are small enough to not cause visible damage. At large scale:
- Model is much wider (larger fan_in → larger gradient magnitudes)
- Batch sizes are larger (more gradient accumulation → potential for larger norms)
- Training runs for 10,000s of steps (rare spike events become near-certain over long runs)
- A single unclipped gradient spike can shift the loss by enough to escape the current basin

The existing failure_modes.py experiment (exp6_no_grad_clipping) documents this directly: at small scale the max grad norm ratio is ~1x; at production scale it becomes a real danger.

### What fixed it

Add `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` before `optimizer.step()`. The threshold 1.0 is standard for transformer training.

### What to monitor in production

- The current `grad_norm > 10.0` threshold is too high to catch the slow-drift failure mode at small scale
- In production, track the 99th percentile grad_norm over a rolling window — a rising p99 without a rising mean is the signature of rare spikes
- The activation_std drift (0.033 → 0.12 over 200 steps) is the most reliable signal of unchecked gradient noise; set a tighter threshold for production: `activation_std > 3x its step-0 value` should trigger a warning
- This failure is genuinely invisible at small scale. Do not conclude it is safe because it did not cause a problem on a 2-layer model.

---

## Summary

| Experiment | First monitor | Step | Failure type | Detectable at init? |
|------------|--------------|------|--------------|---------------------|
| `high_lr` | `anomaly` (act explosion) | 3 | Immediate divergence | No — appears at step 3 |
| `low_lr` | `loss_rate` (stall) | 19 | Silent non-learning | No — needs 20 steps |
| `bad_init` | `grad_norm` + `anomaly` | 0 | Wrong from the start | Yes — step 0 forward pass |
| `no_clip` | `loss_rate` (stall, wrong signal) | 19 | Latent, scale-dependent | No — not reliably at small scale |

Three of the four failures produce degraded-but-not-crashed training. Only `high_lr` shows loss divergence. `low_lr` and `no_clip` both triggered the stall detector but for different underlying reasons. `bad_init` is the only failure detectable before any gradient step.

The practical implication: a training run that produces finite loss and stable gradient norms is not necessarily healthy. You need activation_std tracking and loss progress checks to catch the silent failures.
