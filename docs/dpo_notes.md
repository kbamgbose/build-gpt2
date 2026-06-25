# DPO Notes

Notes on `dpo.py` and `dpo_data.py`. Default target: SFT checkpoint from `sft.py` fine-tuned on a ~100-pair subset of `Intel/orca_dpo_pairs`. Reuses `sft_data.format_prompt` so policy and reference see identical formatting.

Run:
```bash
python dpo.py --ref-checkpoint checkpoints/sft/sft_<ts>.pt
```

`--ref-checkpoint` is required. There is no HF GPT-2 fallback because DPO from a non-instruction-tuned base is meaningless: the reference would not produce coherent responses, so "policy prefers chosen more than reference does" reduces to noise.

---

## What DPO is

Direct Preference Optimization. Given preference triples `(prompt, chosen, rejected)`, shift the policy so it assigns relatively higher probability to chosen than to rejected, compared against a frozen reference copy of itself. No reward model, no PPO, no value function. The reference is what RLHF calls a "KL anchor" but folded directly into the loss.

The thing DPO does NOT do: teach new capabilities. It re-weights distributions over completions the model can already produce. If the base model cannot reason through a problem, DPO will not make it reason.

---

## Loss function

For one preference triple:

```
logp_policy_c = sum_t log P_policy(chosen_t   | prompt, chosen_{<t})
logp_policy_r = sum_t log P_policy(rejected_t | prompt, rejected_{<t})
logp_ref_c, logp_ref_r = same under frozen reference

policy_margin = logp_policy_c - logp_policy_r
ref_margin    = logp_ref_c    - logp_ref_r
rel_margin    = policy_margin - ref_margin

loss = -log sigmoid(beta * rel_margin)
```

Each piece does specific work:

- `chosen - rejected`: the preference direction. We want this to grow.
- subtracting `ref_margin`: anchors the update against what the policy already preferred before training. Without this, large absolute margins on already-easy examples would dominate gradient.
- `beta`: how aggressively we enforce preferences. Standard default `0.1`. Larger beta means smaller permitted drift from reference; smaller beta means more drift but more risk of capability loss.
- `-log sigmoid(...)`: the Bradley-Terry preference model. When `rel_margin = 0` (policy equals reference), loss is `log(2) ≈ 0.693`. When `rel_margin > 0`, loss drops below `log(2)`. When `rel_margin < 0` (policy prefers rejected more than reference did), loss exceeds `log(2)`.

Logp is summed (not averaged) over completion tokens. This is vanilla DPO. Length-normalized variants exist (IPO, LN-DPO) and are not implemented here.

Completion-only: the prompt half of each sequence has `completion_mask = 0` and is excluded from the logp sum. Pad positions are also masked out.

---

## Reward metrics

DPO's headline diagnostic is not the loss number. Loss can decrease while the policy is doing something pathological. Two cheaper-than-free metrics expose what is actually happening:

- **`reward_margin = mean(rel_margin)`** across the batch. Should rise above zero as training progresses. Stays at zero if the policy is not differentiating chosen from rejected any better than reference.
- **`reward_accuracy = mean(rel_margin > 0)`** with ties counted as 0.5. Fraction of batch examples where the policy now prefers chosen *more* than the reference did. At init this is exactly 0.5; healthy training pushes it toward 1.0.

Both are computed from log-probs that already exist in the loss; near-zero compute overhead. Logged per step.

---

## Reference model

The reference is a deep copy of the policy taken before training begins. Two safety nets keep it frozen:

1. `model.eval()` affects modules like Dropout / LayerNorm running stats. Belt.
2. `for p in model.parameters(): p.requires_grad = False` actually prevents grads. Suspenders.

Plus every reference forward is wrapped in `torch.no_grad()`. The combination is verified by `tests/test_dpo_loop.py::test_reference_params_unchanged_after_step`, which snapshots reference params before training, runs five steps, and asserts bit-equality after.

Memory cost: two 124M models = ~1 GB of weights, plus activations on one of them (reference is no-grad, so its activations are dropped). Fits on any modern GPU. On a 16 GB MPS Mac, it is tight but possible at small block_size.

---

## Failure modes

### 1. Swapped chosen and rejected

**Symptom:** loss looks healthy, but generations get worse, reward_margin trends negative.

**Root cause:** somewhere in the data pipeline, chosen and rejected got swapped (in the dataset adapter, in the collator, in the loss-side argument order). The policy learns to prefer rejected.

**Detection:** `reward_margin` going negative over the first few steps. Also `test_loss_increases_when_policy_prefers_rejected_more` pins the loss sign from the math side, and `test_chosen_rejected_share_prompt_tokens` pins that both come from the same prompt (not a mismatched pairing).

### 2. Reference accidentally trains

**Symptom:** after a few steps, reference margins drift away from initial values. `reward_margin` near zero forever because both sides are moving together.

**Root cause:** missing `torch.no_grad()` on the reference forward, or reference params not properly frozen, or `policy = reference` (aliasing instead of deep copy).

**Detection:** `tests/test_dpo_loop.py::test_reference_params_unchanged_after_step` catches this directly by snapshotting and comparing. Also `assert reference.training is False` in `dpo.py:main` at load time.

### 3. Asymmetric EOS treatment

**Symptom:** reward_margin rises without obvious generation improvement; or worse, the policy learns a length preference that wasn't in the data.

**Root cause:** EOS appended to chosen but not rejected (or vice versa). This creates a systematic length difference in the summed logp that gets interpreted as preference.

**Detection:** `tests/test_dpo_data.py::test_eos_appended_to_both_completions` asserts both have EOS at the last position with mask 1.

### 4. Catastrophic forgetting

**Symptom:** HellaSwag drops post-DPO. Generation samples regress to incoherent or repetitive.

**Root cause:** beta too low, lr too high, too many epochs, or all three. DPO pushes the policy away from the reference distribution; far enough away and pretraining capabilities go with it.

**Detection:** run `evals/basic_eval.py` on the saved DPO checkpoint and compare to pre-DPO. Side-by-side generation samples at end of `dpo.py` (SFT-only via reference vs SFT+DPO via policy) also reveal qualitative regression.

**Mitigation:** raise beta, lower lr, or fewer epochs. None of these are tuned in v1.

### 5. Overfit on toy data

**Symptom:** reward_margin climbs steeply on train, holdout reward_margin stays flat or decreases. Generations memorize specific preference patterns from the small dataset.

**Root cause:** ~100 pairs over 3 epochs = ~37 effective steps. Easy to overfit at this scale.

**Detection:** compare train and holdout reward_margin/accuracy at the end of the run. Visible in the eval lines printed every `--eval-every` steps.

### 6. Per-token vs summed logp mismatch

**Symptom:** the loss looks fine but reward_margin behaves oddly across examples of different lengths.

**Root cause:** one path computes mean per-token logp and another sums. Different semantics, both look like "log probability."

**Detection:** `tests/test_dpo_loop.py::test_compute_completion_logp_matches_manual_sum` pins the convention (summed). If a future contributor introduces a mean-based variant, the test fails.

---

## Observed results

Run on a single A100 SXM (RunPod). SFT checkpoint regenerated from HF GPT-2 124M (`sft_20260625_032209.pt`, step 714) and used as both policy init and frozen reference. DPO config: 100 Intel/orca_dpo_pairs (90 train / 10 holdout), 3 epochs, micro batch 2, grad-accum 4 (effective 8), lr 5e-6, beta 0.1, 10-step warmup.

**Headline:** wiring is correct (init sanity numbers exact), severe overfit at this scale (reward margin saturates), but HellaSwag did not budge from SFT-only baseline. Generations changed in style, did not consistently improve.

### Init sanity

| Quantity | Value | Predicted |
|---|---|---|
| pre-DPO holdout loss     | 0.6931 | log(2) ≈ 0.6931 |
| pre-DPO holdout margin   | +0.0000 | 0.0 |
| pre-DPO holdout accuracy | 0.500 | 0.5 (ties counted as 0.5) |

All three match the math predictions to floating-point precision. This is the "DPO loss is implemented correctly" handshake.

### Training trajectory

Loss collapses and reward margin saturates within ~10 effective steps. Selected lines:

| Effective step | Train loss | Reward margin | Grad norm |
|---|---|---|---|
|  0 | 0.6931 | +0.00  | 127.36 |
|  5 | 0.6025 | +1.92  | 106.89 |
| 10 | 0.3644 | +9.75  |  50.01 |
| 15 | 0.0952 | +45.84 |  21.03 |
| 20 | 0.0430 | +37.41 |   9.60 |
| 25 | 0.0046 | +62.89 |   1.20 |
| 32 | 0.0215 | +60.49 |   2.51 |

By step 15 the model has nearly memorized the 90-example train set. Holdout at step 20 (`loss=0.1673, margin=+27.27, acc=1.000`) confirms generalization to the 10 held-out pairs, but the magnitude of `margin` is concerning. Multiplied by beta=0.1, the effective logit input to sigmoid is ~5, deep into the saturated tail of the loss curve.

Final holdout: `loss=0.1222, margin=+49.35, acc=0.900`. Training took 25.6s wall clock on the A100.

### HellaSwag delta

| | Pre-SFT (HF GPT-2) | SFT-only | SFT+DPO |
|---|---|---|---|
| acc          | 0.2858 | 0.2903 | 0.2907 |
| acc_norm     | 0.2955 | 0.2988 | 0.3008 |
| activitynet acc | 0.3395 | 0.3429 | 0.3441 |
| wikihow acc     | 0.2602 | 0.2652 | 0.2652 |

SFT-only numbers shown are from the deterministic SFT-01 run with identical seed and config (not yet pulled from this run's `sft_only_eval.log`). SFT-only → SFT+DPO movement is +0.04 pp acc / +0.20 pp acc_norm, both well within the ~0.45 pp standard-error band at n=10042. **No evidence of catastrophic forgetting** despite the saturated reward margins.

### Generation comparison

Same 5 held-out prompts as SFT-01, decoded greedily from the in-memory reference (SFT-only) and policy (SFT+DPO):

| Prompt | SFT-only | SFT+DPO | Read |
|---|---|---|---|
| List three colors of the rainbow. | Three colors of the rainbow are blue, green, and yellow. | Green | Length collapse |
| What is the capital of France? | The capital of France is Paris. | The capital of France is Paris. | Identical (stable on well-known facts) |
| Explain in one sentence why the sky appears blue. | The sky appears blue because it is a colorless, unreflective substance. | The sky appears blue because it is a natural color. | Shorter, still wrong |
| Write a haiku about a coding bug. | The code was not properly initialized. | The world is a vast and mysterious place, filled with mysteries and mysteries of unknown origin. | Quality regression |
| Convert 25 degrees Celsius to Fahrenheit and show your work. | (infinite repetition loop) | The temperature of the Earth's surface is 25 degrees Celsius (212 Fahrenheit). | Broke the loop, math still wrong (25C is 77F, not 212F) |

DPO changed 4 of 5 outputs. One change is positive (broke a repetition loop), one is negative (length collapse), two are minor style/length differences without quality change, one is identical. Across 5 prompts this is not a clear preference shift; it is mostly DPO's overfit on the orca training set leaking into a different decoding style.

### What we learned

- The init sanity check (loss = log(2), margin = 0, acc = 0.5) is the cheapest possible validation that the loss math is correct. Worth keeping as a printed pre-training line in any future preference-optimization loop.
- At 100 pairs with beta=0.1 and lr=5e-6, the model overfits within ~10 effective steps. Reward margins of +49 on holdout mean the policy is preferring chosen orders of magnitude more confidently than the reference did, which is far past the regime where DPO is doing what it says on the tin (a moderate shift in preference distribution). The honest interpretation: this is memorization with a thin generalization veneer.
- HellaSwag was completely untouched by DPO at this scale. This is the surprising-and-useful finding: even aggressive DPO on toy data did not destroy base capability. Real headroom exists to run with more data or smaller beta / higher lr without catastrophic forgetting being the immediate concern.
- Generation comparison was the most informative artifact. The headline numbers (loss, margin, accuracy, HellaSwag) all looked sensible, but only the side-by-side outputs revealed that DPO at this scale was not consistently improving anything. For preference optimization, qualitative samples are not optional; they catch what numbers miss.
- The grad-norm spike warning fired on every step here too, exactly as in SFT-01. The threshold of 10.0 in `training_reliability/grad_norm.py` is too aggressive for both SFT and DPO; the same follow-up applies (make it per-loop configurable).
- Total cost on the A100: ~$0.45 across the full sequence (SFT regen + SFT-only HellaSwag + DPO + SFT+DPO HellaSwag). DPO itself was ~$0.01.
- The toy 100-pair scale is the right place to wire-up DPO, but the wrong place to evaluate whether DPO actually works. If preference shift is the real goal, the next experiment is `--n-examples 1000` (or more) with lr cut by 5x and beta raised to 0.3-0.5.
