"""
SFT loop on Alpaca-format instruction data. Loss is masked over prompt and pad
positions (label = LOSS_IGNORE); only response tokens contribute gradient.
Standard next-token shift at loss time.
"""
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

import torch
import torch.nn.functional as F
import tiktoken
from torch.utils.data import DataLoader

from model_loader import load_checkpoint, load_hf_pretrained
from sft_data import (
    EOS_ID, LOSS_IGNORE,
    AlpacaSFTDataset, load_alpaca_subset, pad_batch, format_prompt,
)
from training_reliability.anomaly import check_anomaly
from training_reliability.grad_norm import check_grad_norm


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_lr_schedule(warmup_steps: int, base_lr: float):
    def lr_at(step: int) -> float:
        if warmup_steps <= 0:
            return base_lr
        if step < warmup_steps:
            return base_lr * (step + 1) / warmup_steps
        return base_lr
    return lr_at


def sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Shifted next-token cross-entropy with ignore_index=LOSS_IGNORE."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=LOSS_IGNORE,
    )


@torch.no_grad()
def eval_loss(model, loader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        logits, _ = model(input_ids)
        total += sft_loss(logits, labels).item()
        n += 1
    model.train()
    return total / max(n, 1)


@torch.no_grad()
def generate(model, enc, instruction: str, device, max_new_tokens: int = 128):
    prompt = format_prompt(instruction)
    ids = enc.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    emitted_eos = False
    was_training = model.training
    model.eval()
    for _ in range(max_new_tokens):
        if x.size(1) >= model.config.block_size:
            break
        logits, _ = model(x)
        next_id = int(logits[0, -1].argmax())
        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == EOS_ID:
            emitted_eos = True
            break
    if was_training:
        model.train()
    full = enc.decode(x[0].tolist())
    return full[len(prompt):], emitted_eos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None,
                   help="path to a .pt checkpoint saved by train.py")
    p.add_argument("--hf-pretrained", default="gpt2",
                   choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
    p.add_argument("--n-examples",   type=int,   default=1000)
    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--batch-size",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=2e-5)
    p.add_argument("--warmup-steps", type=int,   default=100)
    p.add_argument("--block-size",   type=int,   default=1024)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip",    type=float, default=1.0)
    p.add_argument("--holdout-frac", type=float, default=0.05)
    p.add_argument("--eval-every",   type=int,   default=100,
                   help="0 disables in-training holdout eval")
    p.add_argument("--log-every",    type=int,   default=10)
    p.add_argument("--out-dir",      default="checkpoints/sft")
    p.add_argument("--seed",         type=int,   default=1337)
    p.add_argument("--device",       default="auto",
                   choices=["auto", "cuda", "cpu", "mps"])
    args = p.parse_args()

    set_seed(args.seed)
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    if args.checkpoint:
        model, base_step = load_checkpoint(args.checkpoint, device)
        model_label = f"checkpoint:{os.path.basename(args.checkpoint)}@step{base_step}"
    else:
        model, _ = load_hf_pretrained(args.hf_pretrained, device)
        model_label = f"hf:{args.hf_pretrained}"
    model.train()

    enc = tiktoken.get_encoding("gpt2")
    examples = load_alpaca_subset(n=args.n_examples, seed=args.seed)
    n_holdout = max(1, int(len(examples) * args.holdout_frac))
    train_examples   = examples[:-n_holdout]
    holdout_examples = examples[-n_holdout:]

    train_ds   = AlpacaSFTDataset(train_examples,   enc, args.block_size)
    holdout_ds = AlpacaSFTDataset(holdout_examples, enc, args.block_size)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=pad_batch, drop_last=False,
    )
    holdout_loader = DataLoader(
        holdout_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=pad_batch, drop_last=False,
    )

    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        device=device,
    )
    lr_at = make_lr_schedule(args.warmup_steps, args.lr)

    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("logs", f"sft_{timestamp}.jsonl")

    print(f"model:     {model_label}")
    print(f"device:    {device}")
    print(f"train n:   {len(train_examples)}")
    print(f"holdout n: {len(holdout_examples)}")
    print(f"epochs:    {args.epochs}")
    print(f"batch:     {args.batch_size}")
    print(f"steps/ep:  {len(train_loader)}")
    print(f"lr:        {args.lr} (warmup {args.warmup_steps})")
    print(f"log:       {log_path}")
    print()

    initial_holdout = eval_loss(model, holdout_loader, device)
    print(f"pre-SFT holdout loss: {initial_holdout:.4f}")
    print()

    os.makedirs(args.out_dir, exist_ok=True)
    best_path = os.path.join(args.out_dir, f"sft_{timestamp}_best.pt")
    best_holdout = float("inf")

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for batch in train_loader:
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step)

            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(input_ids)
            loss = sft_loss(logits, labels)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
            optimizer.step()

            train_loss = loss.item()
            cur_lr     = optimizer.param_groups[0]["lr"]

            warnings = []
            if w := check_anomaly(step, train_loss, grad_norm):
                warnings.append(w.message)
            if w := check_grad_norm(step, grad_norm):
                warnings.append(w.message)

            record = {
                "step": step, "epoch": epoch,
                "train_loss": train_loss, "lr": cur_lr,
                "grad_norm": grad_norm,
            }
            if warnings:
                record["warnings"] = warnings
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            if step % args.log_every == 0:
                tail = (f" | {','.join(warnings)}" if warnings else "")
                print(f"step {step:>5d} | loss {train_loss:.4f} | lr {cur_lr:.2e} | gn {grad_norm:.3f}{tail}")

            if args.eval_every and step > 0 and step % args.eval_every == 0:
                hv = eval_loss(model, holdout_loader, device)
                print(f"  [eval @ step {step}] holdout loss: {hv:.4f}")
                with open(log_path, "a") as f:
                    f.write(json.dumps({"step": step, "holdout_loss": hv}) + "\n")
                if hv < best_holdout:
                    best_holdout = hv
                    torch.save({"step": step, "model": model.state_dict(),
                                "loss": hv, "config": model.config,
                                "sft_config": vars(args)}, best_path)

            step += 1

    elapsed = time.time() - t0
    final_holdout = eval_loss(model, holdout_loader, device)

    print()
    print(f"training complete: {step} steps in {elapsed:.1f}s")
    print(f"pre-SFT  holdout loss: {initial_holdout:.4f}")
    print(f"post-SFT holdout loss: {final_holdout:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"sft_{timestamp}.pt")
    torch.save({
        "step":                 step,
        "model":                model.state_dict(),
        "loss":                 final_holdout,
        "config":               model.config,
        "sft_config":           vars(args),
        "initial_holdout_loss": initial_holdout,
        "final_holdout_loss":   final_holdout,
    }, ckpt_path)
    torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"checkpoint: {ckpt_path}")
    if best_holdout < float("inf"):
        print(f"best-by-holdout: {best_path} (holdout loss {best_holdout:.4f})")

    SAMPLE_PROMPTS = [
        "List three colors of the rainbow.",
        "What is the capital of France?",
        "Explain in one sentence why the sky appears blue.",
        "Write a haiku about a coding bug.",
        "Convert 25 degrees Celsius to Fahrenheit and show your work.",
    ]
    print()
    print("generation samples")
    print("------------------")
    for prompt in SAMPLE_PROMPTS:
        gen, eos = generate(model, enc, prompt, device)
        marker = "[eos]" if eos else "[trunc]"
        print(f"prompt: {prompt}")
        print(f"  {marker} {gen.strip()[:300]}")
        print()


if __name__ == "__main__":
    main()
