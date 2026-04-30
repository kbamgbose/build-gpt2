# FM-01 Spec Request

**Prompt used:** `.agent/prompts/spec.md`

---

You are a correctness-focused ML systems engineer.

Task: FM-01 — Failure Mode Experiments

Write an implementation spec for a standalone experiment script that isolates and quantifies six transformer failure modes. The script must:

1. Run entirely on CPU with tiny config (block_size=32, vocab_size=256, n_layer=2, n_head=2, n_embd=64)
2. Not import from production attention.py or model.py — define local copies
3. Inject failure modes via subclassing, not monkey-patching
4. Catch all exceptions — no experiment should crash the script
5. Print one structured result block per failure mode with quantified diagnostics

Failure modes to cover:
- BUG 1: No causal mask — measure whether future tokens leak into past logits
- BUG 2: Softmax before masking — measure attention row sum deviation from 1.0
- BUG 3: No sqrt(d_k) scaling — measure pre-softmax score std and attention entropy
- BUG 4: Wrong view/transpose order — expect RuntimeError on forward pass
- BUG 5: High learning rate (lr=1.0) — show loss divergence over 10 steps
- BUG 6: No gradient clipping — compare max grad norm clipped vs unclipped

Acceptance criteria:
- Exit code 0
- All 6 labels printed
- BUG 1: leaked=True, max_diff > 0
- BUG 2: row sum deviation > 0.1
- BUG 3: unscaled entropy reported lower than scaled (more saturated)
- BUG 4: RuntimeError raised and caught
- BUG 5: loss diverges visibly over 10 steps
- BUG 6: ratio printed (may be 1.0x on small model at step 0)
