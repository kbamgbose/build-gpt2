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

To be backfilled after the GPU run. Plan:

1. Run `python evals/basic_eval.py --hf-pretrained gpt2` for the baseline.
2. Run `python sft.py --hf-pretrained gpt2 --n-examples 1000 --epochs 3` on RunPod.
3. Run `python evals/basic_eval.py --checkpoint checkpoints/sft/sft_<timestamp>.pt`.
4. Fill the table below.

| Metric | Pre-SFT (HF GPT-2 124M) | Post-SFT |
|---|---|---|
| HellaSwag acc        | TBD | TBD |
| HellaSwag acc_norm   | TBD | TBD |
| Holdout loss         | TBD | TBD |
| Train loss (final)   | n/a | TBD |
| Wall time            | n/a | TBD |
| GPU                  | n/a | TBD |

Plus 3 representative generation samples on held-out instructions.
