"""
Fault-tolerance demo: injects a NaN gradient at step 80, detects it via
check_anomaly, rolls back to the last checkpoint, halves LR, and resumes.
Trains a tiny LocalGPT (block_size=32) on synthetic data for 150 steps total.
"""
import json
import math
import os
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

from training_reliability.anomaly import check_anomaly
from training_reliability.cost_tracker import CostTracker


# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    block_size: int = 32
    vocab_size: int = 256
    n_layer:    int = 2
    n_head:     int = 2
    n_embd:     int = 64


CFG = Config()
TOTAL_STEPS  = 150
CKPT_EVERY   = 20
FAULT_STEP   = 80
BATCH        = 4
LOG_DIR      = 'training/logs/fault_tolerance_demo'
CKPT_DIR     = os.path.join(LOG_DIR, 'checkpoints')


# ── model (copied from experiments/failure_modes.py) ─────────────────────────

class BaseAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn   = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj   = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head    = config.n_head
        self.n_embd    = config.n_embd
        self.head_size = config.n_embd // config.n_head
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()
        hs = self.head_size
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class LocalMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class LocalBlock(nn.Module):
    def __init__(self, config, attn_class):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = attn_class(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = LocalMLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class LocalGPT(nn.Module):
    def __init__(self, config, attn_class=BaseAttention):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            h    = nn.ModuleList([LocalBlock(config, attn_class)
                                  for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                std = 0.02
                if hasattr(module, 'NANOGPT_SCALE_INIT'):
                    std *= (2 * self.config.n_layer) ** -0.5
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos  = torch.arange(T, device=idx.device)
        x    = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ── checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(step, model, optimizer, loss, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f'ckpt_{step:04d}.pt')
    torch.save({
        'step':      step,
        'model':     model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss':      loss,
    }, path)
    print(f'[CKPT] step {step} → {path}')
    return path


def latest_checkpoint(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return None
    pts = sorted(f for f in os.listdir(ckpt_dir) if f.endswith('.pt'))
    return os.path.join(ckpt_dir, pts[-1]) if pts else None


def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['step'], ckpt['loss']


# ── logging helpers ───────────────────────────────────────────────────────────

def append_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(record) + '\n')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    metrics_path  = os.path.join(LOG_DIR, 'metrics.jsonl')
    warnings_path = os.path.join(LOG_DIR, 'warnings.jsonl')
    cost_path     = os.path.join(LOG_DIR, 'cost_report.json')

    # clear stale logs and checkpoints from previous runs
    import shutil
    shutil.rmtree(CKPT_DIR, ignore_errors=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    for p in (metrics_path, warnings_path):
        if os.path.exists(p):
            os.remove(p)

    torch.manual_seed(42)
    model     = LocalGPT(CFG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    tracker   = CostTracker(model, B=BATCH, T=CFG.block_size,
                            grad_accum_steps=1, num_gpus=1, gpu_cost_per_hr=0.0)

    fault_injected = False
    last_loss      = float('nan')
    step           = 0

    torch.manual_seed(42)

    while step < TOTAL_STEPS:
        # save checkpoint before this step when it's a checkpoint boundary
        if step % CKPT_EVERY == 0:
            save_checkpoint(step, model, optimizer, last_loss, CKPT_DIR)

        x = torch.randint(0, CFG.vocab_size, (BATCH, CFG.block_size))
        y = torch.randint(0, CFG.vocab_size, (BATCH, CFG.block_size))

        t0 = time.time()

        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()

        if step == FAULT_STEP and not fault_injected:
            first_param = next(model.parameters())
            first_param.grad.data[0] = float('nan')
            print(f'[FAULT] NaN injected at step {FAULT_STEP}')
            fault_injected = True

        grad_norm = sum(
            p.grad.norm().item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5

        warning = check_anomaly(step, loss.item(), grad_norm, activation_std=1.0)

        if warning:
            append_jsonl(warnings_path, {
                'step':    warning.step,
                'monitor': warning.monitor,
                'message': warning.message,
                'values':  warning.values,
            })

            ckpt_path    = latest_checkpoint(CKPT_DIR)
            resumed_step, _ = load_checkpoint(ckpt_path, model, optimizer)

            for pg in optimizer.param_groups:
                pg['lr'] /= 2
            current_lr = optimizer.param_groups[0]['lr']

            print(f'[FAULT] Rolling back to step {resumed_step}, LR halved')
            print(f'[RESUMED] continuing from step {resumed_step} with lr={current_lr}')

            step = resumed_step + 1
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        dt        = time.time() - t0
        last_loss = loss.item()
        tracker.step(step=step, dt_seconds=dt)

        append_jsonl(metrics_path, {
            'step':      step,
            'loss':      last_loss,
            'grad_norm': grad_norm,
            'lr':        optimizer.param_groups[0]['lr'],
        })

        step += 1

    tracker.save(cost_path)
    tracker.print_summary()
    print(f'Done. Metrics → {metrics_path}')
    print(f'Warnings → {warnings_path}')
    print(f'Cost report → {cost_path}')


if __name__ == '__main__':
    main()
