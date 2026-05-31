# Agent Constitution

Hard limits. Any change that would violate these requires explicit user approval before proceeding. Stop, explain the conflict, and wait.

---

## Do Not Touch Without Approval

**GPTConfig defaults** (`model.py:38-43`)
The defaults (block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768) match GPT-2 small and are required for `from_pretrained` weight loading. Changing them silently breaks HuggingFace weight import and invalidates existing checkpoints.

**Checkpoint format** (`train.py:329-344`)
The keys saved in `torch.save(...)` — `step`, `model`, `optimizer`, `loss`, `config`, `loader_shard`, `loader_pos` — are read by `load_checkpoint`. Add new keys freely, but never rename or remove existing ones without a migration path.

**Causal mask logic** (`attention.py:68-70`)
`masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))` is the causal mask. The `== 0` (lower-triangular allows, upper blocks) is intentional. Inverting this, changing the fill value, or altering the slice makes the model attend to future tokens — loss drops fast during training, generation collapses at inference.

**Weight tying** (`train.py:78`, `model.py:57`)
`self.transformer.wte.weight = self.lm_head.weight` — input embedding and output projection share the same tensor. Removing this breaks pretrained weight loading and wastes ~40M parameters.

**DDP barrier placement** (`train.py:364-368`)
`dist.barrier()` calls around checkpoint loading must stay paired and in the right order. Removing one causes rank desync on multi-GPU runs — hangs or silent divergence.

**training_reliability imports** (`train.py:16-23`)
The `try/except ImportError` guard is intentional. Never remove the guard or make the import unconditional — the module is optional for single-machine runs.

---

## Always Do

- Run `python -m pytest tests/ -v` after any edit to `model.py`, `attention.py`, or `train.py`. Do not report a task complete if tests fail.
- Run `python train_tiny.py` for a smoke test after training-loop changes.
- Confirm with the user before adding any entry to `requirements.txt` — each new dependency adds GPU installation time.
- Confirm with the user before adding a new CLI flag or environment variable to `train.py` — the interface is intentionally minimal.

---

## Limits on Parallelism

One feature change at a time. Do not open multiple edit paths across model, training, and reliability code simultaneously — the interaction surface is too large to reason about safely.

---

## What "Done" Means

A task is done when:
1. `python -m pytest tests/ -v` passes with no failures
2. `python train_tiny.py` exits cleanly (no NaN, no exception)
3. The change does not violate any rule above
