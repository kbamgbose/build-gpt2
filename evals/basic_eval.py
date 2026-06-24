"""
Minimal eval harness: HellaSwag validation, log-likelihood multiple-choice scoring.

Usage:
    python evals/basic_eval.py                              # HF GPT-2 fallback
    python evals/basic_eval.py --checkpoint checkpoints/ckpt_05000.pt
    python evals/basic_eval.py --hf-pretrained gpt2-medium
    python evals/basic_eval.py --limit 1000                 # first N examples only

Reports acc and acc_norm, overall and per source category (activitynet, wikihow).
Published baseline for HF GPT-2 124M: ~28.5% acc, ~29.5% acc_norm (random = 25%).

Scoring:
    For each of 4 candidate endings, compute the average log-probability of the
    ending tokens given the shared context. acc picks the highest summed log-prob,
    acc_norm picks the highest per-token mean log-prob (controls for ending-length
    bias). acc_norm is the standard reported number for HellaSwag.
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.request
from datetime import datetime

import torch
import torch.nn.functional as F
import tiktoken

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model_loader import load_checkpoint, load_hf_pretrained

DATA_DIR      = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR   = os.path.join(os.path.dirname(__file__), "results")
HELLASWAG_URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def download_hellaswag():
    path = os.path.join(DATA_DIR, "hellaswag_val.jsonl")
    if os.path.exists(path):
        return path
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"downloading HellaSwag val to {path}")
    urllib.request.urlretrieve(HELLASWAG_URL, path)
    print(f"  done, {os.path.getsize(path) / 1e6:.1f} MB")
    return path


def load_hellaswag(limit=None):
    path = download_hellaswag()
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
            if limit and len(examples) >= limit:
                break
    return examples


def category_of(example):
    src = example.get("source_id", "")
    return src.split("~", 1)[0] if "~" in src else "unknown"


@torch.no_grad()
def score_ending(model, enc, ctx, ending, device):
    """Return (sum_log_prob, num_continuation_tokens) for ctx + " " + ending."""
    ctx_ids = enc.encode(ctx)
    end_ids = enc.encode(" " + ending)
    full_ids = ctx_ids + end_ids

    block_size = model.config.block_size
    if len(full_ids) > block_size:
        # Trim context from the left so the full sequence fits.
        excess = len(full_ids) - block_size
        ctx_ids = ctx_ids[excess:]
        full_ids = ctx_ids + end_ids

    x = torch.tensor([full_ids], dtype=torch.long, device=device)
    logits, _ = model(x)
    log_probs = F.log_softmax(logits[0], dim=-1)

    # log_probs[i] is the distribution over token i+1.
    # Ending tokens occupy positions [ctx_len, N); predicted by log_probs[ctx_len-1 : N-1].
    ctx_len = len(ctx_ids)
    target = torch.tensor(end_ids, dtype=torch.long, device=device)
    selected = log_probs[ctx_len - 1: len(full_ids) - 1].gather(1, target.unsqueeze(1)).squeeze(1)
    return selected.sum().item(), len(end_ids)


@torch.no_grad()
def score_example(model, enc, example, device):
    """Return (pred_idx_by_sum, pred_idx_by_mean) for the 4 candidate endings."""
    sums = []
    means = []
    for ending in example["endings"]:
        total, n = score_ending(model, enc, example["ctx"], ending, device)
        sums.append(total)
        means.append(total / max(n, 1))
    return (
        int(max(range(4), key=lambda i: sums[i])),
        int(max(range(4), key=lambda i: means[i])),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="path to a .pt checkpoint saved by train.py")
    parser.add_argument("--hf-pretrained", default="gpt2",
                        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
                        help="HF model name (used when --checkpoint is not given)")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N examples")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if args.checkpoint:
        model, step = load_checkpoint(args.checkpoint, device)
        model_label = f"checkpoint:{os.path.basename(args.checkpoint)}@step{step}"
    else:
        model, _ = load_hf_pretrained(args.hf_pretrained, device)
        model_label = f"hf:{args.hf_pretrained}"

    enc = tiktoken.get_encoding("gpt2")
    examples = load_hellaswag(limit=args.limit)

    print(f"model:    {model_label}")
    print(f"device:   {device}")
    print(f"examples: {len(examples)}")
    print(f"seed:     {args.seed}")
    print()

    correct      = {"all": 0}
    correct_norm = {"all": 0}
    total        = {"all": 0}

    t0 = time.time()
    for i, ex in enumerate(examples):
        pred_sum, pred_mean = score_example(model, enc, ex, device)
        label = ex["label"]
        cat = category_of(ex)

        for k in (cat, "all"):
            correct.setdefault(k, 0)
            correct_norm.setdefault(k, 0)
            total.setdefault(k, 0)
            correct[k]      += int(pred_sum == label)
            correct_norm[k] += int(pred_mean == label)
            total[k]        += 1

        if (i + 1) % 500 == 0:
            acc = correct["all"] / total["all"]
            rate = (i + 1) / (time.time() - t0)
            print(f"  [{i+1}/{len(examples)}] acc={acc:.4f}  {rate:.1f} ex/s")

    elapsed = time.time() - t0

    print()
    print("results")
    print("-------")
    print(f"{'category':<16} {'n':>6} {'acc':>8} {'acc_norm':>10}")
    for k in sorted(total.keys(), key=lambda x: (x != "all", x)):
        n = total[k]
        print(f"{k:<16} {n:>6} {correct[k]/n:>8.4f} {correct_norm[k]/n:>10.4f}")
    print()
    print(f"elapsed: {elapsed:.1f}s ({len(examples)/elapsed:.1f} ex/s)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "timestamp":  timestamp,
        "model":      model_label,
        "n_examples": len(examples),
        "seed":       args.seed,
        "elapsed_s":  elapsed,
        "per_category": {
            k: {"n": total[k],
                "acc": correct[k] / total[k],
                "acc_norm": correct_norm[k] / total[k]}
            for k in total
        },
    }
    out_path = os.path.join(RESULTS_DIR, f"hellaswag_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"results written to {out_path}")


if __name__ == "__main__":
    main()
