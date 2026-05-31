# Global Rules

This file is the index. Read the linked file when working in that area.

| Area | When to read | File |
|------|-------------|------|
| Training loop | Editing `train.py`, `train_tiny.py`, `train_baseline.py` | `training.md` |
| Model architecture | Editing `model.py`, `attention.py` | `model.md` |
| Tests | Adding, removing, or running tests | `testing.md` |
| Hard guardrails | Before any change to guarded code | `../constitution.md` |

---

## Cross-cutting rules (apply everywhere)

**No magic numbers.** Every numeric constant that affects behavior must trace to a named variable, env var, or documented formula. The only acceptable bare literals are mathematical constants (e.g., `0.5`, `math.pi`).

**No silent fallbacks.** If a required input is missing, raise with a clear message. The `MONITORS_AVAILABLE` pattern in `train.py` is the deliberate exception — it guards an *optional* module, not required logic.

**Prefer existing patterns.** Before adding a new utility, check `training_reliability/` — cost tracking, anomaly detection, grad-norm checking, and loss-rate monitoring are already there. Use them.

**Env vars over hardcodes.** Runtime knobs (batch size, LR, checkpoint interval) are already driven by `os.environ.get(...)`. New ones follow the same pattern. See `.agent/.env.example` for the full list.

**Keep train.py flat.** The training loop is intentionally procedural and readable top-to-bottom. Avoid introducing helper classes, decorators, or abstractions that fragment the flow across files.
