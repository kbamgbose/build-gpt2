import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dpo import dpo_loss


def _t(x):
    return torch.tensor(x, dtype=torch.float32)


def test_loss_at_policy_equals_ref_is_log2():
    z = _t([0.0, 0.0])
    loss, m = dpo_loss(z, z, z, z, beta=0.1)
    assert abs(loss.item() - math.log(2)) < 1e-6
    assert m["reward_margin"]   == 0.0
    assert m["reward_accuracy"] == 0.5


def test_loss_decreases_when_policy_prefers_chosen_more():
    ref_c, ref_r = _t([0.0]), _t([0.0])
    pol_c_low,  pol_r_low  = _t([0.5]), _t([-0.5])
    pol_c_high, pol_r_high = _t([2.0]), _t([-2.0])
    loss_low,  _ = dpo_loss(pol_c_low,  pol_r_low,  ref_c, ref_r, beta=1.0)
    loss_high, _ = dpo_loss(pol_c_high, pol_r_high, ref_c, ref_r, beta=1.0)
    assert loss_high.item() < loss_low.item()


def test_loss_increases_when_policy_prefers_rejected_more():
    ref_c, ref_r = _t([0.0]), _t([0.0])
    pol_c, pol_r = _t([-1.0]), _t([1.0])
    loss, m = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=1.0)
    assert loss.item() > math.log(2)
    assert m["reward_margin"]   == -2.0
    assert m["reward_accuracy"] == 0.0


def test_reward_accuracy_well_defined():
    pol_c = _t([1.0, -1.0, 0.5, -0.5])
    pol_r = _t([0.0,  0.0, 0.0,  0.0])
    ref_c = _t([0.0,  0.0, 0.0,  0.0])
    ref_r = _t([0.0,  0.0, 0.0,  0.0])
    _, m = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1)
    assert m["reward_accuracy"] == 0.5


def test_beta_scales_loss_response():
    pol_c, pol_r = _t([1.0]), _t([-1.0])
    ref_c, ref_r = _t([0.0]), _t([0.0])
    loss_low,  _ = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.01)
    loss_high, _ = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=10.0)
    assert loss_high.item() < loss_low.item()
