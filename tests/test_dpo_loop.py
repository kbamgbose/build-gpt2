import copy
import os
import sys

import torch
import torch.nn.functional as F
import tiktoken

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model import GPT, GPTConfig
from dpo import compute_completion_logp, dpo_loss, freeze, make_lr_schedule
from dpo_data import OrcaDPODataset, pad_dpo_batch


def _tiny_model():
    config = GPTConfig(block_size=256, vocab_size=50257, n_layer=2, n_head=2, n_embd=64)
    torch.manual_seed(0)
    return GPT(config)


def _tiny_batch():
    enc = tiktoken.get_encoding("gpt2")
    examples = [
        {"question": "Add 1 and 2.",         "chosen": "3.",           "rejected": "five"},
        {"question": "What color is grass?", "chosen": "Green.",       "rejected": "purple"},
        {"question": "Spell cat.",           "chosen": "c-a-t",        "rejected": "k-a-t"},
        {"question": "Name a vegetable.",    "chosen": "Carrot.",      "rejected": "Cake."},
    ]
    ds = OrcaDPODataset(examples, enc, block_size=256)
    return pad_dpo_batch([ds[i] for i in range(len(ds))])


def test_compute_completion_logp_matches_manual_sum():
    model = _tiny_model()
    batch = _tiny_batch()
    ids  = batch["chosen_ids"]
    mask = batch["chosen_mask"]

    with torch.no_grad():
        lp_vec = compute_completion_logp(model, ids, mask)

        logits, _ = model(ids)
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        targets       = ids[:, 1:]
        shifted_mask  = mask[:, 1:].float()
        per_token_lp  = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        manual = (per_token_lp * shifted_mask).sum(dim=-1)

    assert torch.allclose(lp_vec, manual, atol=1e-6)


def test_policy_and_ref_margins_zero_at_init():
    """Before any training step, policy and reference are identical → rel_margin = 0 exactly."""
    model = _tiny_model()
    reference = copy.deepcopy(model)
    freeze(reference)
    batch = _tiny_batch()

    with torch.no_grad():
        pol_c = compute_completion_logp(model,     batch["chosen_ids"],   batch["chosen_mask"])
        pol_r = compute_completion_logp(model,     batch["rejected_ids"], batch["rejected_mask"])
        ref_c = compute_completion_logp(reference, batch["chosen_ids"],   batch["chosen_mask"])
        ref_r = compute_completion_logp(reference, batch["rejected_ids"], batch["rejected_mask"])
        _, m = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1)

    assert abs(m["reward_margin"]) < 1e-6


def test_reference_params_unchanged_after_step():
    """A backward+step on the policy must leave reference parameters bit-exact."""
    model = _tiny_model()
    reference = copy.deepcopy(model)
    freeze(reference)
    batch = _tiny_batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ref_snapshot = [p.detach().clone() for p in reference.parameters()]

    for _ in range(5):
        optimizer.zero_grad()
        pol_c = compute_completion_logp(model, batch["chosen_ids"],   batch["chosen_mask"])
        pol_r = compute_completion_logp(model, batch["rejected_ids"], batch["rejected_mask"])
        with torch.no_grad():
            ref_c = compute_completion_logp(reference, batch["chosen_ids"],   batch["chosen_mask"])
            ref_r = compute_completion_logp(reference, batch["rejected_ids"], batch["rejected_mask"])
        loss, _ = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1)
        loss.backward()
        optimizer.step()

    for p_now, p_then in zip(reference.parameters(), ref_snapshot):
        assert torch.equal(p_now, p_then)


def test_reward_margin_increases_on_fixed_batch():
    """Tiny overfit: after several DPO steps on the same batch, reward_margin should rise above zero."""
    model = _tiny_model()
    reference = copy.deepcopy(model)
    freeze(reference)
    batch = _tiny_batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    margins = []
    for _ in range(10):
        optimizer.zero_grad()
        pol_c = compute_completion_logp(model, batch["chosen_ids"],   batch["chosen_mask"])
        pol_r = compute_completion_logp(model, batch["rejected_ids"], batch["rejected_mask"])
        with torch.no_grad():
            ref_c = compute_completion_logp(reference, batch["chosen_ids"],   batch["chosen_mask"])
            ref_r = compute_completion_logp(reference, batch["rejected_ids"], batch["rejected_mask"])
        loss, m = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1)
        loss.backward()
        optimizer.step()
        margins.append(m["reward_margin"])

    assert margins[-1] > 0.0
    assert margins[-1] > margins[0]


def test_lr_warmup_schedule():
    lr_at = make_lr_schedule(warmup_steps=10, base_lr=5e-6)
    assert abs(lr_at(0)  - 5e-6 * 1 / 10) < 1e-12
    assert abs(lr_at(9)  - 5e-6)          < 1e-12
    assert lr_at(10) == 5e-6
    assert lr_at(100) == 5e-6
