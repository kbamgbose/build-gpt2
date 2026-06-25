"""DPO loop. Trains a policy from an SFT checkpoint against a frozen reference copy."""
import argparse
import copy
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import tiktoken
from torch.utils.data import DataLoader

from model_loader import load_checkpoint
from sft_data import format_prompt, EOS_ID
from dpo_data import OrcaDPODataset, load_orca_dpo_subset, pad_dpo_batch
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


def compute_completion_logp(model, input_ids: torch.Tensor, completion_mask: torch.Tensor) -> torch.Tensor:
    logits, _ = model(input_ids)
    log_probs    = F.log_softmax(logits[:, :-1, :], dim=-1)
    targets      = input_ids[:, 1:]
    shifted_mask = completion_mask[:, 1:].float()
    per_token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (per_token_lp * shifted_mask).sum(dim=-1)


def dpo_loss(policy_chosen_lp:   torch.Tensor,
             policy_rejected_lp: torch.Tensor,
             ref_chosen_lp:      torch.Tensor,
             ref_rejected_lp:    torch.Tensor,
             beta:               float) -> Tuple[torch.Tensor, Dict[str, float]]:
    policy_margin = policy_chosen_lp - policy_rejected_lp
    ref_margin    = ref_chosen_lp    - ref_rejected_lp
    rel_margin    = policy_margin - ref_margin
    loss          = -F.logsigmoid(beta * rel_margin).mean()
    acc = ((rel_margin > 0).float() + 0.5 * (rel_margin == 0).float()).mean()
    metrics = {
        "reward_margin":   rel_margin.mean().item(),
        "reward_accuracy": acc.item(),
        "policy_margin":   policy_margin.mean().item(),
        "ref_margin":      ref_margin.mean().item(),
    }
    return loss, metrics


def freeze(model) -> None:
    model.eval()
    for p in model.parameters():
        p.requires_grad = False


@torch.no_grad()
def eval_dpo(policy, reference, loader, beta: float, device) -> Dict[str, float]:
    policy.eval()
    total_loss = 0.0
    total_metrics = defaultdict(float)
    n = 0
    for batch in loader:
        ci = batch["chosen_ids"].to(device)
        cm = batch["chosen_mask"].to(device)
        ri = batch["rejected_ids"].to(device)
        rm = batch["rejected_mask"].to(device)
        pol_c = compute_completion_logp(policy,    ci, cm)
        pol_r = compute_completion_logp(policy,    ri, rm)
        ref_c = compute_completion_logp(reference, ci, cm)
        ref_r = compute_completion_logp(reference, ri, rm)
        loss, metrics = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)
        total_loss += loss.item()
        for k, v in metrics.items():
            total_metrics[k] += v
        n += 1
    policy.train()
    return {"loss": total_loss / max(n, 1),
            **{k: v / max(n, 1) for k, v in total_metrics.items()}}


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
    p.add_argument("--ref-checkpoint",   required=True)
    p.add_argument("--n-examples",       type=int,   default=100)
    p.add_argument("--epochs",           type=int,   default=3)
    p.add_argument("--batch-size",       type=int,   default=2)
    p.add_argument("--grad-accum-steps", type=int,   default=4)
    p.add_argument("--lr",               type=float, default=5e-6)
    p.add_argument("--warmup-steps",     type=int,   default=10)
    p.add_argument("--beta",             type=float, default=0.1)
    p.add_argument("--block-size",       type=int,   default=1024)
    p.add_argument("--weight-decay",     type=float, default=0.1)
    p.add_argument("--grad-clip",        type=float, default=1.0)
    p.add_argument("--holdout-frac",     type=float, default=0.1)
    p.add_argument("--eval-every",       type=int,   default=20)
    p.add_argument("--log-every",        type=int,   default=1)
    p.add_argument("--out-dir",          default="checkpoints/dpo")
    p.add_argument("--seed",             type=int,   default=1337)
    p.add_argument("--device",           default="auto",
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

    policy, base_step = load_checkpoint(args.ref_checkpoint, device)
    policy.train()
    reference = copy.deepcopy(policy)
    freeze(reference)
    reference.to(device)
    assert reference.training is False

    enc = tiktoken.get_encoding("gpt2")
    examples = load_orca_dpo_subset(n=args.n_examples, seed=args.seed)
    n_holdout = max(1, int(len(examples) * args.holdout_frac))
    train_examples   = examples[:-n_holdout]
    holdout_examples = examples[-n_holdout:]

    train_ds   = OrcaDPODataset(train_examples,   enc, args.block_size)
    holdout_ds = OrcaDPODataset(holdout_examples, enc, args.block_size)
    train_loader   = DataLoader(train_ds,   batch_size=args.batch_size, shuffle=True,
                                collate_fn=pad_dpo_batch, drop_last=False)
    holdout_loader = DataLoader(holdout_ds, batch_size=args.batch_size, shuffle=False,
                                collate_fn=pad_dpo_batch, drop_last=False)

    optimizer = policy.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        device=device,
    )
    lr_at = make_lr_schedule(args.warmup_steps, args.lr)

    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("logs", f"dpo_{timestamp}.jsonl")

    print(f"ref ckpt:    {args.ref_checkpoint} @ step {base_step}")
    print(f"device:      {device}")
    print(f"train n:     {len(train_examples)}")
    print(f"holdout n:   {len(holdout_examples)}")
    print(f"epochs:      {args.epochs}")
    print(f"micro batch: {args.batch_size}  grad-accum: {args.grad_accum_steps}  eff batch: {args.batch_size * args.grad_accum_steps}")
    print(f"lr:          {args.lr}  warmup-eff-steps: {args.warmup_steps}")
    print(f"beta:        {args.beta}")
    print(f"log:         {log_path}")
    print()

    initial_eval = eval_dpo(policy, reference, holdout_loader, args.beta, device)
    print(f"pre-DPO holdout: loss={initial_eval['loss']:.4f}  margin={initial_eval['reward_margin']:+.4f}  acc={initial_eval['reward_accuracy']:.3f}")
    print()

    effective_step = 0
    accum_count    = 0
    accum_loss     = 0.0
    accum_metrics  = defaultdict(float)
    optimizer.zero_grad(set_to_none=True)

    t0 = time.time()
    for epoch in range(args.epochs):
        for batch in train_loader:
            ci = batch["chosen_ids"].to(device)
            cm = batch["chosen_mask"].to(device)
            ri = batch["rejected_ids"].to(device)
            rm = batch["rejected_mask"].to(device)

            with torch.no_grad():
                ref_c = compute_completion_logp(reference, ci, cm)
                ref_r = compute_completion_logp(reference, ri, rm)

            pol_c = compute_completion_logp(policy, ci, cm)
            pol_r = compute_completion_logp(policy, ri, rm)

            loss, metrics = dpo_loss(pol_c, pol_r, ref_c, ref_r, args.beta)
            (loss / args.grad_accum_steps).backward()

            accum_loss += loss.item()
            for k, v in metrics.items():
                accum_metrics[k] += v
            accum_count += 1

            if accum_count < args.grad_accum_steps:
                continue

            for g in optimizer.param_groups:
                g["lr"] = lr_at(effective_step)
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip).item()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            mean_loss    = accum_loss / args.grad_accum_steps
            mean_metrics = {k: v / args.grad_accum_steps for k, v in accum_metrics.items()}
            cur_lr       = optimizer.param_groups[0]["lr"]

            warnings = []
            if w := check_anomaly(effective_step, mean_loss, grad_norm):
                warnings.append(w.message)
            if w := check_grad_norm(effective_step, grad_norm):
                warnings.append(w.message)

            record = {
                "step": effective_step, "epoch": epoch,
                "train_loss": mean_loss,
                "lr": cur_lr, "grad_norm": grad_norm,
                **mean_metrics,
            }
            if warnings:
                record["warnings"] = warnings
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            if effective_step % args.log_every == 0:
                tail = (f" | {','.join(warnings)}" if warnings else "")
                print(f"step {effective_step:>4d} | loss {mean_loss:.4f} | margin {mean_metrics['reward_margin']:+.4f} | "
                      f"acc {mean_metrics['reward_accuracy']:.3f} | lr {cur_lr:.2e} | gn {grad_norm:.3f}{tail}")

            if args.eval_every and effective_step > 0 and effective_step % args.eval_every == 0:
                hv = eval_dpo(policy, reference, holdout_loader, args.beta, device)
                print(f"  [eval @ step {effective_step}] loss={hv['loss']:.4f}  margin={hv['reward_margin']:+.4f}  acc={hv['reward_accuracy']:.3f}")
                with open(log_path, "a") as f:
                    f.write(json.dumps({"step": effective_step, "holdout": hv}) + "\n")

            effective_step += 1
            accum_count = 0
            accum_loss  = 0.0
            accum_metrics.clear()

    elapsed = time.time() - t0
    final_eval = eval_dpo(policy, reference, holdout_loader, args.beta, device)

    print()
    print(f"training complete: {effective_step} effective steps in {elapsed:.1f}s")
    print(f"pre-DPO  holdout: loss={initial_eval['loss']:.4f}  margin={initial_eval['reward_margin']:+.4f}  acc={initial_eval['reward_accuracy']:.3f}")
    print(f"post-DPO holdout: loss={final_eval['loss']:.4f}  margin={final_eval['reward_margin']:+.4f}  acc={final_eval['reward_accuracy']:.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"dpo_{timestamp}.pt")
    torch.save({
        "step":         effective_step,
        "model":        policy.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "loss":         final_eval["loss"],
        "config":       policy.config,
        "dpo_config":   vars(args),
        "initial_eval": initial_eval,
        "final_eval":   final_eval,
    }, ckpt_path)
    print(f"checkpoint: {ckpt_path}")

    SAMPLE_PROMPTS = [
        "List three colors of the rainbow.",
        "What is the capital of France?",
        "Explain in one sentence why the sky appears blue.",
        "Write a haiku about a coding bug.",
        "Convert 25 degrees Celsius to Fahrenheit and show your work.",
    ]
    print()
    print("generation comparison (SFT-only via frozen reference vs SFT+DPO via policy)")
    print("---------------------------------------------------------------------------")
    assert reference.training is False
    for prompt in SAMPLE_PROMPTS:
        gen_ref, eos_ref = generate(reference, enc, prompt, device)
        gen_dpo, eos_dpo = generate(policy,    enc, prompt, device)
        print(f"prompt: {prompt}")
        print(f"  SFT      {'[eos]' if eos_ref else '[trunc]'} {gen_ref.strip()[:300]}")
        print(f"  SFT+DPO  {'[eos]' if eos_dpo else '[trunc]'} {gen_dpo.strip()[:300]}")
        print()


if __name__ == "__main__":
    main()
