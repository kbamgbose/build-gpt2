from dataclasses import dataclass
import glob
import json
import math
import torch
import torch.nn as nn
import inspect
import time
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
from attention import CausalSelfAttention

try:
    from training_reliability.cost_tracker import CostTracker
    from training_reliability.loss_rate import check_loss_rate
    from training_reliability.grad_norm import check_grad_norm
    from training_reliability.anomaly import check_anomaly
    MONITORS_AVAILABLE = True
except ImportError:
    MONITORS_AVAILABLE = False

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate = 'tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768 


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
            # idx is of shape (B, T)
            B, T  = idx.size()
            assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"
            # forward the token and position embeddings
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device) #shape (T)
            pos_emb = self.transformer.wpe(pos) # posiion embeddings of shape (T, n_embd)
            tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
            x = tok_emb + pos_emb
            # forward blocks of the transformer
            for block in self.transformer.h:
                x = block(x)
            # forward the final layernorn and the classifier
            x = self.transformer.ln_f(x)
            logits = self.lm_head(x) # (B, T, vocab_size)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        #n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),   #124M params
            'gpt2-medium': dict(n_layer=16, n_head=16, n_embd=1024),  #350M params
            'gpt2-large':  dict(n_layer=20, n_head=20, n_embd=1280),  #774M params
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),  #1550M params
        } [model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)

        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('attn.bias')]  # discard this mask / buffer

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Hugging Face keys, minus buffer-only entries
        sd_keys_hf = [k for k in sd_hf.keys()
                      if not k.endswith('.attn.masked_bias')
                      and not k.endswith('.attn.bias')]

        # Only keep keys that exist in our model
        common_keys = [k for k in sd_keys_hf if k in sd]

        # Optional: print what doesn't line up (for debugging)
        # extra_hf = [k for k in sd_keys_hf if k not in sd]
        # missing_hf = [k for k in sd_keys if k not in sd_hf]
        # print("extra HF keys:", len(extra_hf))
        # print("missing HF keys:", len(missing_hf))

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                      'mlp.c_fc.weight', 'mlp.c_proj.weight']

        for k in common_keys:
            if any(k.endswith(w) for w in transposed):
                # Conv1D -> Linear: need transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (That require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, )
        return optimizer
# -----------------------------------------------------------------------------------------------------------------

import tiktoken
import numpy as np

enc = tiktoken.get_encoding('gpt2')

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y

# --------------------------------------------------------------------------
# distributed data parallel

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    device_type = "cuda"
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 16 # micro batch size
T = 1024 # seq length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure the total batch size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process: 
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision('high')

#logits
model = GPT(GPTConfig(vocab_size = 50304))
model.to(device)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank]) 
raw_model = model.module if ddp else model

max_lr =6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073
def get_lr(it):
    # 1) linear warmip for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

#optimizer
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

# ── checkpoint helpers ────────────────────────────────────────────────────────
CHECKPOINT_DIR      = os.environ.get('CHECKPOINT_DIR', 'checkpoints')
CHECKPOINT_INTERVAL = int(os.environ.get('CHECKPOINT_INTERVAL', '500'))
MAX_CHECKPOINTS     = int(os.environ.get('MAX_CHECKPOINTS', '5'))

def save_checkpoint(step, loss_val):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"ckpt_{step:05d}.pt")
    torch.save({
        'step':      step,
        'model':     raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss':      loss_val,
        'config':    raw_model.config,
        'loader_shard': train_loader.current_shard,
        'loader_pos':   train_loader.current_position,
    }, path)
    existing = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, 'ckpt_*.pt')))
    for old in existing[:-MAX_CHECKPOINTS]:
        os.remove(old)
    return path

def latest_checkpoint():
    ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, 'ckpt_*.pt')))
    return ckpts[-1] if ckpts else None

def load_checkpoint(path):
    ckpt = torch.load(path, map_location=device)
    raw_model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    train_loader.current_shard    = ckpt.get('loader_shard', 0)
    train_loader.current_position = ckpt.get('loader_pos', 0)
    train_loader.tokens = load_tokens(train_loader.shards[train_loader.current_shard])
    return ckpt['step'], ckpt['loss']

# ── resume from checkpoint if one exists ──────────────────────────────────────
start_step = 0
resume_ckpt = latest_checkpoint()
if resume_ckpt and master_process:
    start_step, _ = load_checkpoint(resume_ckpt)
    if ddp:
        dist.barrier()
    print(f"Resumed from {resume_ckpt} at step {start_step}")
elif resume_ckpt and ddp:
    dist.barrier()

# ── cost tracker + monitor state ──────────────────────────────────────────────
if MONITORS_AVAILABLE:
    tracker = CostTracker(
        raw_model, B, T, grad_accum_steps,
        num_gpus=ddp_world_size,
        gpu_cost_per_hr=float(os.environ.get('GPU_COST_PER_HR', '2.50')),
    )
loss_history  = []
rollback_count = 0

for step in range(start_step, max_steps):
    t0 = time.time()

    # evals
    if step % 100 == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y  = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")

        # generate from the model occasionally
        if step > 0 and step % 100 == 0:
            model.eval()
            num_return_sequences = 4
            max_length = 32
            tokens = enc.encode("Hello, I'm a language model,")
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)
            while xgen.size(1) < max_length:
                # forward the model to get the logits
                with torch.no_grad():
                    logits, loss = model(xgen) # (B, T, vocab_size)
                    logits = logits[:, -1, :] # (B, vocab_size)
                    probs = F.softmax(logits, dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                    xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                    xgen = torch.cat((xgen, xcol), dim=1)

            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(f"rank {ddp_rank} sample {i}: {decoded}")

    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # ── NaN / explosion rollback ──────────────────────────────────────────────
    if not torch.isfinite(loss_accum):
        rollback_count += 1
        ckpt = latest_checkpoint()
        if ckpt and master_process:
            rolled_back_to, _ = load_checkpoint(ckpt)
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.5
            print(f"[FAULT] NaN at step {step} (rollback #{rollback_count}) "
                  f"→ restored {ckpt}, LR halved to {optimizer.param_groups[0]['lr']:.2e}")
        elif master_process:
            print(f"[FAULT] NaN at step {step}, no checkpoint to roll back to — stopping.")
            break
        if ddp:
            dist.barrier()
        continue

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    # ── cost tracking ─────────────────────────────────────────────────────────
    if MONITORS_AVAILABLE:
        tracker.step(step=step, dt_seconds=dt)

    # ── periodic checkpoint ───────────────────────────────────────────────────
    if step > 0 and step % CHECKPOINT_INTERVAL == 0 and master_process:
        path = save_checkpoint(step, loss_accum.item())
        print(f"[CKPT] step {step} → {path}")

    # ── live monitors ─────────────────────────────────────────────────────────
    loss_history.append(loss_accum.item())
    if MONITORS_AVAILABLE and master_process:
        for w in [check_loss_rate(step, loss_history),
                  check_grad_norm(step, norm.item()),
                  check_anomaly(step, loss_accum.item(), norm.item(), 0.0)]:
            if w:
                print(f"WARN [step {w.step:4d}] {w.monitor}: {w.message} {w.values}")

    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | "
              f"norm: {norm:.4f} | dt: {dt*1000:.2f} ms | tok/sec: {tokens_per_sec:.2f}")


if master_process:
    # final checkpoint
    path = save_checkpoint(max_steps, loss_history[-1] if loss_history else 0.0)
    print(f"Training complete. Final checkpoint: {path}")

    # cost report
    if MONITORS_AVAILABLE:
        tracker.print_summary()
        os.makedirs("logs", exist_ok=True)
        tracker.save("logs/cost_report.json")

    if rollback_count:
        print(f"Total rollbacks during run: {rollback_count}")

if ddp:
    destroy_process_group()
