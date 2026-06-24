# Supervised Fine-Tuning Notes

Notes on `sft.py` and `sft_data.py`. Default target is HF GPT-2 124M fine-tuned on a 1k subset of `tatsu-lab/alpaca`. Same loader path supports a custom checkpoint from `train.py`.

Run:
```bash
python sft.py --hf-pretrained gpt2 --n-examples 1000 --epochs 3 --batch-size 4 --lr 2e-5
```

---

## What SFT is

Take a base language model trained on next-token prediction over web text and teach it to follow a `(instruction, response)` schema. The model architecture, vocabulary, and tokenizer are unchanged. Only the weights move, and only the loss signal is restricted: the model is graded on the response half of each example, not the instruction half.

The point is not to add knowledge. The base model already has it. The point is to bias the conditional distribution `p(text | "### Instruction:\n{x}\n\n### Response:\n")` toward what a human would write as a useful response, instead of toward the most likely continuation of a web document.

---

## Loss function

Standard next-token cross-entropy with a mask. Three pieces:

1. **Build inputs and labels position-by-position.**
   ```
   input_ids = [p0, p1, ..., p_{P-1}, r0, r1, ..., r_{R-1}, EOS]
   labels    = [-100, -100, ..., -100, r0, r1, ..., r_{R-1}, EOS]
   ```
   The first `P` positions (prompt) get `-100`. The next `R + 1` positions (response + EOS) get the token id at that position.

2. **Forward pass produces logits aligned with input_ids.** `logits[i]` is the distribution over the token that should follow `input_ids[i]`. For next-token prediction we shift:
   ```python
   shift_logits = logits[..., :-1, :]   # predicts positions 1..T-1
   shift_labels = labels[..., 1:]       # targets at positions 1..T-1
   loss = F.cross_entropy(shift_logits.view(-1, V), shift_labels.view(-1),
                          ignore_index=-100)
   ```
   Positions whose shifted label is `-100` (every prompt position, every pad position) contribute zero and do not count toward the mean.

3. **EOS is trainable.** The EOS id is `50256` (tiktoken's `<|endoftext|>`). It is appended to the response in `tokenize_example` and gets a real label, not `-100`. Without this the model never learns where to stop and will generate until `block_size` runs out.

Padding uses `pad_id = 50256` too. The value does not matter because pad positions are masked. This avoids needing a dedicated pad token outside the vocabulary.

---

## Prompt template

Standard Alpaca format. Single source of truth in `sft_data.format_prompt`. Imported by both training and inference; never duplicated.

With input:
```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
```

Without input:
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
```

Response is concatenated after `### Response:\n` and is the portion the model is trained on.

---

## Truncation

If `len(prompt) + len(response) + 1 > block_size`, the prompt is left-truncated. The most recent prompt tokens are preserved (closer to the response). If the response alone is at least `block_size` tokens, the example is rejected with `ValueError`, not silently skipped. A silent skip would mean some examples become "instruction only with zero trainable tokens," which trains nothing while looking like a normal pass.

---

## Failure modes

### 1. Loss-masking bug

**Symptom:** Training looks healthy. Loss decreases. After training the model continues `### Instruction:` prompts by writing new instructions instead of responding.

**Root cause:** Prompt-position labels were not `-100`. The model was trained to predict every token in the formatted example, including the instruction itself. It learned the joint distribution `p(instruction, response)` instead of the conditional `p(response | instruction)`.

**Detection:** `tests/test_sft_data.py::test_loss_ignores_masked_tokens` builds two logit tensors that differ only on prompt positions and asserts cross-entropy returns identical loss. If prompt logits ever leak into the gradient, this test fails.

### 2. EOS appended but masked out

**Symptom:** Generation never terminates. Every sample hits `max_new_tokens` and gets cut.

**Root cause:** EOS was appended to the response string but the label at that position was set to `-100`. Model never sees a training signal to emit `50256`.

**Detection:** `tests/test_sft_data.py::test_eos_is_added_and_trainable` asserts the last non-pad label is `50256`.

### 3. Catastrophic forgetting

**Symptom:** Post-SFT HellaSwag accuracy drops noticeably from the base model. Generation in response to non-instruction prompts (free-form completion) becomes incoherent.

**Root cause:** SFT loss is computed only over response tokens in a narrow distribution (Alpaca-style instructions, short responses). Gradient updates to the entire model push weights away from the broader pretraining distribution. The smaller the SFT dataset and the larger the LR, the worse this gets.

**Detection:** Run `python evals/basic_eval.py --checkpoint <sft_checkpoint.pt>` and compare to the HF GPT-2 baseline. Expect a drop. The size of the drop is the cost of instruction-following.

**Mitigation:** lower LR, fewer epochs, mix pretraining tokens into the batch (rehearsal), or use parameter-efficient methods like LoRA. None of these are implemented; the current loop is full fine-tune.

### 4. Overfit on small data

**Symptom:** Train loss drops to near-zero within 1 to 2 epochs. Held-out instructions produce verbatim parrots of training responses.

**Root cause:** 1k examples is small relative to 124M parameters. With 3 epochs that is ~3000 gradient updates on the same examples. The model memorizes responses.

**Detection:** Compare `initial_holdout_loss` and `final_holdout_loss` printed at the end of a run. If holdout loss is much higher than train loss at the end, overfit is present. Both are also logged to the per-step jsonl.

### 5. Learning rate too high

**Symptom:** Loss spikes wildly in the first hundred steps and plateaus high. HellaSwag acc collapses near random.

**Root cause:** GPT-2 was pretrained with cosine schedule peaking at 6e-4. Cold-starting AdamW at that LR on a new objective destroys the pretrained features. SFT should use 1 to 2 orders of magnitude lower. The default `2e-5` with 100-step linear warmup is conservative and works.

**Detection:** The training log records `grad_norm` per step. If grad-norm stays above 10 for many consecutive steps, LR is too aggressive. The `check_grad_norm` reliability check from `training_reliability/` raises a warning on each spike.

### 6. Prompt template drift

**Symptom:** Model behaves much worse at inference than the holdout loss suggests.

**Root cause:** Training used template A, inference used template B. Subtle differences (a stray newline, missing colon, different preamble) put the inference prompt off the training distribution.

**Detection:** Only one `format_prompt` function exists in `sft_data.py`. Both `sft.py:generate` and (future) downstream code import it. As long as nobody copy-pastes the template into a second file, drift is impossible.

---

## Observed results

Run on a single A100 SXM 80GB (RunPod). Base model: HF GPT-2 124M. SFT config: 1000 Alpaca examples, 3 epochs, batch 4, lr 2e-5, 100-step linear warmup, grad clip 1.0, holdout 5%.

**Headline:** SFT taught the Alpaca format and EOS termination reliably. Holdout loss dropped ~25%. HellaSwag did not degrade.

### Holdout loss trajectory

| Step | Holdout loss | Notes |
|---|---|---|
| 0    (pre-SFT) | 3.2565 | base HF GPT-2 on Alpaca format |
| 100 (end warmup) | 2.4272 | |
| 200 | 2.3775 | |
| 300 | 2.3660 | minimum |
| 400 | 2.3654 | still near minimum |
| 500 | 2.4227 | climbing |
| 600 | 2.4464 | overfit visible |
| 700 | 2.4391 | |
| 714 (final) | 2.4502 | saved checkpoint |

Training: 714 steps in 37.5s wall clock. The minimum holdout loss occurred at step 300-400 (about 1.7 epochs), then climbed back as the model started memorizing training examples. This is failure mode #4 (overfit on small data) exactly as predicted. A natural fix is best-checkpoint-by-holdout-loss saving, not last-checkpoint; the current loop does the latter.

### HellaSwag delta

| Metric | Pre-SFT | Post-SFT | Δ |
|---|---|---|---|
| all acc          | 0.2858 | 0.2903 | +0.45 pp |
| all acc_norm     | 0.2955 | 0.2988 | +0.33 pp |
| activitynet acc      | 0.3395 | 0.3429 | +0.34 pp |
| activitynet acc_norm | 0.3475 | 0.3475 | 0.00 pp |
| wikihow acc          | 0.2602 | 0.2652 | +0.50 pp |
| wikihow acc_norm     | 0.2706 | 0.2756 | +0.50 pp |

Catastrophic forgetting (failure mode #3) did not materialize. Two ways to read this:

- **Statistical caveat.** At n=10042 with p ≈ 0.29, one standard error of a proportion is about 0.45 pp. The all-categories delta sits at one SE. The honest call is "no evidence of degradation," not "evidence of improvement." Per-chunk variance within a single eval run is bigger than the delta: the first 500 examples ran 35.2%, the final converged at 29.0%.
- **Mechanism hypothesis.** At lr=2e-5 over 750 weight updates on 1k examples, this is the gentle-SFT regime. Format and EOS get added without overwriting the broader pretraining distribution. Catastrophic forgetting kicks in at higher lr, more epochs, or larger SFT datasets. To actually test the hypothesis, run with `--epochs 10 --lr 5e-5` and watch for a HellaSwag drop.

### Generation samples (held-out instructions)

Pre-SFT (base HF GPT-2) would generate until `max_new_tokens`, never emitting `<|endoftext|>` and not following any response template. Post-SFT:

```
prompt: List three colors of the rainbow.
  [eos] Three colors of the rainbow are blue, green, and yellow.

prompt: What is the capital of France?
  [eos] The capital of France is Paris.

prompt: Explain in one sentence why the sky appears blue.
  [eos] The sky appears blue because it is a colorless, unreflective substance.

prompt: Write a haiku about a coding bug.
  [eos] The code was not properly initialized.

prompt: Convert 25 degrees Celsius to Fahrenheit and show your work.
  [trunc] The Convert 25 degrees Celsius to Fahrenheit task is a simple and effective
   way to convert 25 degrees Celsius to Fahrenheit. The Convert 25 degrees Celsius to
   Fahrenheit task is a simple and effective way to convert 25 degrees Celsius to
   Fahrenheit. ...
```

Four of five samples terminated with EOS, demonstrating the EOS-trainable invariant worked end-to-end. The fifth collapsed into a repetition loop, a well-known small-model failure mode that SFT does not address: the format is correct, but the underlying arithmetic capability is not present in 124M.

The "sky is colorless" answer is also notable: format is correct, factual content is wrong. SFT teaches the response shape, not the response content. Knowledge is whatever the base model already has.

### What we learned

- The loss-masking invariant held: EOS was emitted in 4 of 5 generations, and the holdout loss drop is real.
- The overfit-on-small-data failure mode (#4) showed up exactly as the doc predicted, visible in the holdout trajectory by epoch 2.
- The catastrophic-forgetting failure mode (#3) did not materialize at this scale. The conditions for it are higher than what was used here.
- The repetition-loop failure mode is downstream of capability, not SFT-related. Worth noting as a separate class of failure.
- The gradient-spike warning threshold of 10.0 in `training_reliability/grad_norm.py` is calibrated for pretraining and is too aggressive for SFT. Grad norms here sat at 12 to 25 throughout, firing the warning on nearly every step. A follow-up is to make the threshold configurable per-loop, not change the global default.
- A real GPU run cost about $0.25 on a single A100 (two SFT runs at ~40s each, two full HellaSwag evals at ~250s each). The first SFT run's checkpoint corrupted on disk write (silent failure); the second succeeded. A `torch.load` round-trip verify immediately after save would catch this class of bug.
