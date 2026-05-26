"""
Baseline training loop for failure-mode experiments.
Usage: python train_baseline.py [baseline|high_lr|low_lr|bad_init|no_clip]
"""
import sys
import os
import json
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from collections import namedtuple

try:
    from training_reliability.loss_rate import check_loss_rate
    from training_reliability.grad_norm import check_grad_norm
    from training_reliability.anomaly import check_anomaly
    MONITORS_AVAILABLE = True
except ImportError:
    MONITORS_AVAILABLE = False

Warning = namedtuple('Warning', ['step', 'monitor', 'message', 'values'])


# ── model config ──────────────────────────────────────────────────────────────

@dataclass
class Config:
    block_size: int = 32
    vocab_size: int = 256
    n_layer:    int = 2
    n_head:     int = 2
    n_embd:     int = 64


CFG = Config()


# ── model (self-contained, no external imports) ───────────────────────────────

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
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = BaseAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = LocalMLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class LocalGPT(nn.Module):
    def __init__(self, config, init_std=0.02):
        super().__init__()
        self.config   = config
        self.init_std = init_std
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            h    = nn.ModuleList([LocalBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # bad_init mode: use raw init_std without scaling
                std = self.init_std
                if self.init_std == 0.02 and hasattr(module, 'NANOGPT_SCALE_INIT'):
                    std *= (2 * self.config.n_layer) ** -0.5
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=self.init_std)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos    = torch.arange(T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ── LR schedule ───────────────────────────────────────────────────────────────

def get_lr(step, base_lr, warmup_steps=10, max_steps=200):
    min_lr = base_lr * 0.1
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)


# ── experiment configs ────────────────────────────────────────────────────────

EXPERIMENT_CONFIGS = {
    'baseline': {'lr': 3e-4,  'init_std': 0.02, 'clip': 1.0},
    'high_lr':  {'lr': 1.0,   'init_std': 0.02, 'clip': 1.0},
    'low_lr':   {'lr': 1e-7,  'init_std': 0.02, 'clip': 1.0},
    'bad_init': {'lr': 3e-4,  'init_std': 1.0,  'clip': 1.0},
    'no_clip':  {'lr': 3e-4,  'init_std': 0.02, 'clip': None},
}


# ── main training loop ────────────────────────────────────────────────────────

def run(mode='baseline'):
    cfg = EXPERIMENT_CONFIGS[mode]
    lr, init_std, clip = cfg['lr'], cfg['init_std'], cfg['clip']

    print(f"\n{'='*60}")
    print(f"  Experiment: {mode}")
    print(f"  lr={lr}, init_std={init_std}, clip={clip}")
    print(f"  monitors: {'enabled' if MONITORS_AVAILABLE else 'NOT AVAILABLE'}")
    print(f"{'='*60}")

    log_dir = os.path.join('logs', mode)
    os.makedirs(log_dir, exist_ok=True)
    metrics_path  = os.path.join(log_dir, 'metrics.jsonl')
    warnings_path = os.path.join(log_dir, 'warnings.jsonl')

    torch.manual_seed(42)
    model = LocalGPT(CFG, init_std=init_std)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8)

    # activation std hook
    activation_buffer = []
    def activation_hook(module, input, output):
        activation_buffer.append(output.detach().std().item())
    model.transformer.h[-1].register_forward_hook(activation_hook)

    loss_history = []

    with open(metrics_path, 'w') as mf, open(warnings_path, 'w') as wf:
        for step in range(200):
            x = torch.randint(0, CFG.vocab_size, (4, CFG.block_size))
            y = torch.randint(0, CFG.vocab_size, (4, CFG.block_size))

            activation_buffer.clear()
            logits, loss = model(x, y)
            act_std = activation_buffer[-1] if activation_buffer else 0.0

            loss_val = loss.item()

            if not torch.isfinite(loss):
                print(f"  NaN/Inf loss at step {step}, stopping early.")
                mf.write(json.dumps({'step': step, 'loss': None, 'grad_norm': None,
                                     'activation_std': act_std, 'lr': get_lr(step, lr)}) + '\n')
                break

            optimizer.zero_grad()
            loss.backward()

            if clip is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip).item()
            else:
                grad_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in model.parameters() if p.grad is not None
                ) ** 0.5

            current_lr = get_lr(step, lr)
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr
            optimizer.step()

            loss_history.append(loss_val)

            metrics = {
                'step': step,
                'loss': loss_val,
                'grad_norm': grad_norm,
                'activation_std': act_std,
                'lr': current_lr,
            }
            mf.write(json.dumps(metrics) + '\n')

            # live monitors
            if MONITORS_AVAILABLE:
                candidates = [
                    check_loss_rate(step, loss_history),
                    check_grad_norm(step, grad_norm),
                    check_anomaly(step, loss_val, grad_norm, act_std),
                ]
                for w in candidates:
                    if w is not None:
                        print(f"  WARN [step {w.step:3d}] {w.monitor}: {w.message} {w.values}")
                        wf.write(json.dumps({
                            'step': w.step, 'monitor': w.monitor,
                            'message': w.message, 'values': w.values,
                        }) + '\n')
                        wf.flush()

            if step % 20 == 0 or step == 199:
                print(f"  step {step:3d} | loss: {loss_val:.4f} | "
                      f"grad_norm: {grad_norm:.4f} | act_std: {act_std:.4f} | lr: {current_lr:.2e}")
            mf.flush()

    print(f"  Logs written to {log_dir}/")


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'baseline'
    if mode not in EXPERIMENT_CONFIGS:
        print(f"Unknown mode '{mode}'. Choose from: {list(EXPERIMENT_CONFIGS)}")
        sys.exit(1)
    run(mode)
