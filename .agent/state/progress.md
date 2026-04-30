# Project Progress

## Active Work: Failure Mode Analysis Series

| Task  | Status    | File(s)                          |
|-------|-----------|----------------------------------|
| FM-01 | completed | experiments/failure_modes.py     |
| FM-02 | pending   | tests/test_failure_modes.py      |
| FM-03 | pending   | docs/failure_modes.md            |
| FM-04 | pending   | profiling/scaling_experiment.py, profiling/block_size_experiment.py |

## Completed Milestones

- Core architecture: attention.py, model.py, train.py, train_tiny.py
- Profiling suite: profile_attention.py, scaling_experiment.py, block_size_experiment.py, edge_cases.py
- A100 profiling run complete — results in docs/transformer_scaling_analysis.md
- Invariant tests: tests/test_transformer.py (5 tests, all passing)
- FM-01: 6 failure mode experiments isolated and quantified

## Next

FM-02 → FM-03 → FM-04
