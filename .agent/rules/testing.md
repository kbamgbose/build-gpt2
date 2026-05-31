# Testing Rules

Applies to: `tests/`, any edit to `model.py`, `attention.py`, `train.py`

---

## When to run

Run `python -m pytest tests/ -v` after every edit to model or training code. Not after documentation or profiling changes — but when in doubt, run it.

## Existing tests and what they prove

| Test | File | Invariant |
|------|------|-----------|
| `test_causal_masking` | `test_transformer.py` | Changing tokens at position t+1..T-1 must not change logits at 0..t |
| `test_attention_output_shape` | `test_transformer.py` | `CausalSelfAttention` preserves (B, T, C) shape |
| `test_forward_pass_shape` | `test_transformer.py` | Full forward pass produces (B, T, vocab_size) logits |
| `test_loss_is_finite` | `test_transformer.py` | Forward pass with targets produces finite loss |
| `test_no_nans_forward_and_backward` | `test_transformer.py` | No NaN/inf in logits or gradients during a full training step |
| `test_compile_checkpoint` | `test_compile_checkpoint.py` | `torch.compile` model + checkpoint save/load roundtrip |
| `test_failure_modes` | `test_failure_modes.py` | Failure mode detection in `experiments/failure_modes.py` |

## Adding new tests

New tests go in `tests/`. Use the minimal config already defined (`GPTConfig(block_size=64, vocab_size=256, n_layer=2, n_head=2, n_embd=64)`) — it runs fast on CPU and covers the structural invariants. Tests that require CUDA or the `edu_fineweb10B` dataset are not run in the standard suite and must be explicitly marked or kept in `profiling/` / `experiments/`.

## Test config constraint

The test GPTConfig deliberately uses `vocab_size=256` (not 50257) so tests run fast without tiktoken or dataset access. Tests must not call `enc = tiktoken.get_encoding('gpt2')` at module level — that breaks offline/CI environments.

## What tests are NOT for

Tests prove structural invariants (shape, causality, finite gradients). They do not prove that the model learns well — loss curves, scaling behavior, and convergence are for `profiling/` and `experiments/`. Don't add loss-convergence assertions to the test suite; they will be flaky across hardware and PyTorch versions.
