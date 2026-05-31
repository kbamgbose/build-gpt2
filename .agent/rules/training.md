# Training Rules

Applies to: `train.py`, `train_tiny.py`, `train_baseline.py`, `fineweb.py`

---

## Batch size invariant

`total_batch_size % (B * T * ddp_world_size) == 0` must hold.  
`grad_accum_steps` is derived from this — it is not an independent parameter. If you change `B`, `T`, or `total_batch_size`, update all three together and verify the assertion still passes.

## LR schedule

The cosine decay in `get_lr` has three segments: linear warmup → cosine decay → floor at `min_lr`. The formula assumes `warmup_steps < max_steps`. Do not decouple `min_lr` from `max_lr * 0.1` without understanding the implications for final convergence.

## NaN rollback

The rollback block (`train.py:446-460`) halves LR on each rollback. This is intentional — repeated NaNs usually mean the LR is too high, and halving it is the correct recovery. Do not change this to a fixed value or remove the halving.

## Checkpoint rotation

`MAX_CHECKPOINTS` controls how many `.pt` files survive. The rotation (`existing[:-MAX_CHECKPOINTS]`) keeps the newest N. Never remove or reorder this — storage is finite on training pods.

## Validation eval

Validation runs every 100 steps using 20 fixed batches. The `val_loader.reset()` before each eval is required — without it, val position drifts and the loss comparison across steps is meaningless.

## DDP safety

When adding any new per-rank state (data counters, metric accumulators), ask: does each rank need its own copy, or should this be reduced across ranks? Wrong choices cause rank divergence that's hard to detect until generation quality degrades.

## Metrics logging

`log_metrics()` appends one JSON line per step to `logs/metrics.jsonl`. The schema — `step`, `loss`, `grad_norm`, `lr`, `dt_ms`, `tok_per_sec` — is fixed. Downstream analysis scripts depend on it. Add new fields freely; never rename or remove existing ones.
