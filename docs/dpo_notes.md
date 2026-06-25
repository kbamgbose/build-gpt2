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

To be backfilled after the GPU run. Plan:

1. Regenerate the SFT checkpoint on RunPod (SFT-01 checkpoint was not saved off the pod).
2. Run `python evals/basic_eval.py --checkpoint checkpoints/sft/sft_<ts>.pt` for the SFT-only HellaSwag baseline.
3. Run `python dpo.py --ref-checkpoint checkpoints/sft/sft_<ts>.pt`.
4. Run `python evals/basic_eval.py --checkpoint checkpoints/dpo/dpo_<ts>.pt` for the SFT+DPO HellaSwag number.
5. Fill the tables below and paste the generation comparison.

| Metric | SFT-only | SFT+DPO | Δ |
|---|---|---|---|
| HellaSwag acc        | TBD | TBD | TBD |
| HellaSwag acc_norm   | TBD | TBD | TBD |
| Holdout DPO loss     | TBD | TBD | TBD |
| Holdout reward_margin   | TBD | TBD | TBD |
| Holdout reward_accuracy | TBD | TBD | TBD |
| Wall time (DPO)      | n/a | TBD | n/a |
| GPU                  | n/a | TBD | n/a |

Plus side-by-side SFT-only vs SFT+DPO generations on the five fixed held-out prompts (already printed at the end of `dpo.py`).
