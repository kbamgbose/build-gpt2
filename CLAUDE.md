# nano-gpt

GPT-2 built from scratch — transformer architecture, distributed training (DDP), fault-tolerant checkpointing, and a training reliability monitoring module.

## Map

| Path | Role |
|------|------|
| `model.py` | Clean standalone GPT (GPTConfig, GPT, Block, MLP) |
| `attention.py` | CausalSelfAttention with causal mask and sqrt(d_k) scaling |
| `train.py` | Full training loop — DDP, grad accum, cosine LR, NaN rollback, checkpointing |
| `train_tiny.py` | Fast sanity loop (no dataset required) |
| `sft.py` | Supervised fine-tuning loop on Alpaca-format instruction data |
| `sft_data.py` | SFT prompt template, tokenizer, padding collator with `-100` masking |
| `model_loader.py` | Shared model loaders for evals and SFT (HF GPT-2 and local checkpoints) |
| `evals/` | Eval harnesses (currently `basic_eval.py`: HellaSwag multiple-choice) |
| `minbpe/` | Minimal regex BPE tokenizer (teaching artifact, not wired into training) |
| `training_reliability/` | Cost tracker, loss-rate monitor, grad-norm check, anomaly detector |
| `profiling/` | Scaling experiments (attention kernels, context length, width, depth) |
| `experiments/` | Failure-mode demos and scaling runs |
| `tests/` | Invariant tests — must pass before any PR |

## Rules

Before changing anything, read:
1. `.agent/constitution.md` — hard guardrails; anything listed there requires explicit user approval
2. `.agent/rules/global.md` — ruleset index that links to area-specific files
3. `.agent/workflow.md` — process for meaningful changes (prediction, plan-only, trace, learning, from-memory gate)

For meaningful changes (see `.agent/workflow.md`), do not implement immediately. Produce plan-only output first and prompt the human to fill the Prediction section of `.agent/runs/<TASK-ID>/task.md`.

## Running tests

```bash
python -m pytest tests/ -v
```

Run this after every edit to `model.py`, `attention.py`, or `train.py`.

## Quick training smoke-test

```bash
python train_tiny.py
```

No dataset needed. If it runs without NaN/error, the model and training loop are intact.

## Environment

See `.agent/.env.example` for all supported runtime knobs (checkpoint interval, GPU cost rate, max steps, etc.). Never hardcode these values in source — they are all driven by `os.environ.get(...)` calls already in `train.py`.
