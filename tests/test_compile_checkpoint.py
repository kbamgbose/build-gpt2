"""
Smoke test for torch.compile + checkpoint round-trip.
Simulates the train.py initialization order on CPU (no GPU required).
Verifies:
  1. raw_model.state_dict() keys have no _orig_mod. prefix
  2. Forward pass works through the compiled model (fixed shape)
  3. Checkpoint save -> load -> forward doesn't break
  4. raw_model handles variable-length sequences (simulates generation loop)
"""
import os
import sys
import tempfile

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model import GPT, GPTConfig


VOCAB = 50304
CFG = dict(vocab_size=VOCAB, n_layer=2, n_head=2, n_embd=64, block_size=64)


@pytest.fixture
def setup():
    gpt = GPT(GPTConfig(**CFG))
    raw_model = gpt
    device_type = "cpu"
    if torch.cuda.is_available():
        gpt = gpt.cuda()
        raw_model = gpt
        device_type = "cuda"
        gpt = torch.compile(gpt)
    return gpt, raw_model, device_type


def test_state_dict_no_orig_mod_prefix(setup):
    _, raw_model, _ = setup
    keys = list(raw_model.state_dict().keys())
    bad = [k for k in keys if '_orig_mod' in k]
    assert not bad, f"_orig_mod prefix found in keys: {bad}"


def test_training_forward(setup):
    model, _, _ = setup
    device = next(model.parameters()).device
    x = torch.randint(0, VOCAB, (4, 64), device=device)
    _, loss = model(x, x)
    assert torch.isfinite(loss)


def test_checkpoint_round_trip(setup):
    model, raw_model, _ = setup
    device = next(raw_model.parameters()).device
    x = torch.randint(0, VOCAB, (4, 64), device=device)

    _, loss_before = model(x, x)

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
    try:
        torch.save({'model': raw_model.state_dict(), 'step': 0}, path)
        ckpt = torch.load(path, weights_only=False)
        raw_model.load_state_dict(ckpt['model'])
        _, loss_after = model(x, x)
        assert torch.isfinite(loss_after)
        assert torch.allclose(loss_before, loss_after)
    finally:
        os.unlink(path)


def test_generation_variable_length(setup):
    """raw_model must handle growing T without recompile errors."""
    _, raw_model, _ = setup
    device = next(raw_model.parameters()).device
    xgen = torch.randint(0, VOCAB, (2, 8), device=device)
    raw_model.eval()
    with torch.no_grad():
        for _ in range(8):
            logits, _ = raw_model(xgen)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            xgen = torch.cat((xgen, next_token), dim=1)
    assert xgen.size(1) == 16
