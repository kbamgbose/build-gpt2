"""
Tests for sft.py training loop helpers.

Verifies wiring (forward → loss → backward → step) on a tiny model and pins the
LR warmup schedule. All tests run on CPU in seconds.

Run: python -m pytest tests/test_sft_loop.py -v
"""
import os
import sys

import torch
import tiktoken

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model import GPT, GPTConfig
from sft import make_lr_schedule, sft_loss
from sft_data import AlpacaSFTDataset, pad_batch


def _tiny_model():
    config = GPTConfig(block_size=256, vocab_size=50257, n_layer=2, n_head=2, n_embd=64)
    torch.manual_seed(0)
    return GPT(config)


def _tiny_batch():
    enc = tiktoken.get_encoding("gpt2")
    examples = [
        {"instruction": "Add 1 and 2.",        "input": None, "output": "3."},
        {"instruction": "What color is grass?", "input": None, "output": "Green."},
        {"instruction": "Name a vegetable.",    "input": None, "output": "Carrot."},
        {"instruction": "Spell cat.",           "input": None, "output": "c-a-t"},
    ]
    ds = AlpacaSFTDataset(examples, enc, block_size=256)
    return pad_batch([ds[i] for i in range(len(ds))])


def test_one_step_no_nan():
    """A single forward+backward+step must produce finite loss and finite gradients."""
    model = _tiny_model()
    batch = _tiny_batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    optimizer.zero_grad()
    logits, _ = model(batch["input_ids"])
    loss = sft_loss(logits, batch["labels"])
    assert torch.isfinite(loss)
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "non-finite grad"
    optimizer.step()


def test_loss_decreases_after_n_steps():
    """After 5 training steps on the same batch, loss must drop. Catches
    backward-disconnect, wrong-direction sign flips, or optimizer misconfiguration."""
    model = _tiny_model()
    batch = _tiny_batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        logits, _ = model(batch["input_ids"])
        loss = sft_loss(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


def test_lr_warmup_schedule():
    """Linear ramp from base_lr * (1/warmup_steps) to base_lr, constant after."""
    lr_at = make_lr_schedule(warmup_steps=100, base_lr=2e-5)
    assert abs(lr_at(0)  - 2e-5 * 1   / 100) < 1e-12
    assert abs(lr_at(49) - 2e-5 * 50  / 100) < 1e-12
    assert abs(lr_at(99) - 2e-5 * 100 / 100) < 1e-12
    assert lr_at(100) == 2e-5
    assert lr_at(500) == 2e-5
    lr_no_warmup = make_lr_schedule(warmup_steps=0, base_lr=5e-5)
    assert lr_no_warmup(0) == 5e-5
    assert lr_no_warmup(100) == 5e-5
