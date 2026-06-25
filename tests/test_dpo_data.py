"""
Tests for DPO data pipeline. Pins: completion-only mask, shared-prompt prefix
across chosen/rejected, EOS appended to both completions, padding excluded.
"""
import os
import sys

import pytest
import tiktoken

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sft_data import EOS_ID
from dpo_data import (
    PAD_ID,
    format_pair,
    tokenize_completion,
    tokenize_dpo_example,
    pad_dpo_batch,
)

ENC = tiktoken.get_encoding("gpt2")


def _build(question="What is 2 plus 2?", chosen="4.", rejected="five", block_size=128):
    return tokenize_dpo_example(ENC, question, chosen, rejected, block_size)


def test_orca_schema_to_triple():
    prompt, chosen, rejected = format_pair("Q?", "good", "bad")
    assert prompt.endswith("### Response:\n")
    assert chosen == "good"
    assert rejected == "bad"


def test_chosen_rejected_share_prompt_tokens():
    """The prompt prefix must tokenize identically across chosen and rejected examples
    (since DPO compares logprobs conditioned on the same prompt)."""
    ex = _build(chosen="answer A here", rejected="answer B different")
    chosen_prompt_len  = sum(1 for m in ex["chosen_mask"]   if m == 0)
    rejected_prompt_len = sum(1 for m in ex["rejected_mask"] if m == 0)
    assert chosen_prompt_len == rejected_prompt_len
    assert ex["chosen_ids"][:chosen_prompt_len] == ex["rejected_ids"][:rejected_prompt_len]


def test_completion_mask_excludes_prompt():
    """Prompt positions must have mask 0; completion positions must have mask 1."""
    ex = _build()
    for ids, mask in [(ex["chosen_ids"], ex["chosen_mask"]),
                      (ex["rejected_ids"], ex["rejected_mask"])]:
        first_one = mask.index(1)
        assert all(m == 0 for m in mask[:first_one])
        assert all(m == 1 for m in mask[first_one:])
        assert first_one > 0


def test_eos_appended_to_both_completions():
    """Both chosen and rejected must end with EOS, and both EOS positions must be in the mask."""
    ex = _build()
    assert ex["chosen_ids"][-1]    == EOS_ID
    assert ex["rejected_ids"][-1]  == EOS_ID
    assert ex["chosen_mask"][-1]   == 1
    assert ex["rejected_mask"][-1] == 1


def test_padding_excluded_from_mask():
    """In a batch with different completion lengths, pad positions in the shorter example
    must have completion_mask = 0 so they cannot contribute to the logp sum."""
    e_short = _build(chosen="ok", rejected="no")
    e_long  = _build(chosen="this is a much longer chosen response", rejected="and a longer rejected one too")
    batch = pad_dpo_batch([e_short, e_long])
    L_short = len(e_short["chosen_ids"])
    L_max = batch["chosen_ids"].shape[1]
    if L_max > L_short:
        assert (batch["chosen_mask"][0, L_short:] == 0).all()
        assert (batch["chosen_ids"][0,  L_short:] == PAD_ID).all()


def test_completion_too_long_raises():
    huge = "x " * 2000
    with pytest.raises(ValueError, match="completion alone"):
        _build(chosen=huge, block_size=128)
    with pytest.raises(ValueError, match="completion alone"):
        _build(rejected=huge, block_size=128)


def test_pad_dpo_batch_shapes():
    batch = pad_dpo_batch([_build(), _build(chosen="longer chosen here")])
    assert batch["chosen_ids"].shape    == batch["chosen_mask"].shape
    assert batch["rejected_ids"].shape  == batch["rejected_mask"].shape
    assert batch["chosen_ids"].shape[0] == 2
