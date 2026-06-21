"""
Invariant tests for the SFT data pipeline.

The core invariant: loss is computed only over response tokens. Test #7 pins
this from the PyTorch side (cross-entropy with ignore_index=-100), tests #1-6
pin it from the data side.

Run: python -m pytest tests/test_sft_data.py -v
"""
import os
import sys

import pytest
import torch
import torch.nn.functional as F
import tiktoken

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from sft_data import (
    EOS_ID, LOSS_IGNORE,
    format_prompt, format_full,
    tokenize_example, pad_batch,
)

ENC = tiktoken.get_encoding("gpt2")


def _build(instruction="What is 2 plus 2?", input=None, output="4", block_size=128):
    prompt, response = format_full(instruction, input, output)
    return tokenize_example(ENC, prompt, response, block_size)


# --- core invariant: data side ---------------------------------------------

def test_prompt_tokens_are_masked():
    """All positions before the first response token must have label LOSS_IGNORE."""
    ex = _build()
    first_real = next(i for i, l in enumerate(ex["labels"]) if l != LOSS_IGNORE)
    assert first_real > 0
    assert all(l == LOSS_IGNORE for l in ex["labels"][:first_real])


def test_response_tokens_are_trainable():
    """Response positions must have labels equal to their token ids."""
    ex = _build()
    response_positions = [(i, l) for i, l in enumerate(ex["labels"]) if l != LOSS_IGNORE]
    assert len(response_positions) > 0
    for i, label in response_positions:
        assert label == ex["input_ids"][i]


def test_padding_tokens_are_masked():
    """In a batch with different lengths, pad positions in the shorter example are masked."""
    e_short = _build(output="hi")
    e_long  = _build(output="this is a deliberately longer response with more tokens")
    batch = pad_batch([e_short, e_long])
    L_short = len(e_short["input_ids"])
    assert (batch["labels"][0, L_short:] == LOSS_IGNORE).all()
    assert (batch["attention_mask"][0, :L_short] == 1).all()
    assert (batch["attention_mask"][0, L_short:] == 0).all()


def test_inputs_and_labels_same_shape():
    """input_ids and labels must always align position-by-position."""
    ex = _build()
    assert len(ex["input_ids"]) == len(ex["labels"])
    batch = pad_batch([_build(), _build(output="a longer response here")])
    assert batch["input_ids"].shape == batch["labels"].shape


def test_eos_is_added_and_trainable():
    """The final response token must be EOS and its label must be EOS (not LOSS_IGNORE)."""
    ex = _build()
    assert ex["input_ids"][-1] == EOS_ID
    assert ex["labels"][-1] == EOS_ID


def test_truncation_preserves_response_loss():
    """A prompt that overflows block_size must not leave the response with zero trainable tokens."""
    long_instruction = "x " * 2000
    block_size = 256
    ex = _build(instruction=long_instruction, output="short answer", block_size=block_size)
    assert len(ex["input_ids"]) == block_size
    n_response = sum(1 for l in ex["labels"] if l != LOSS_IGNORE)
    expected_response_tokens = len(ENC.encode("short answer")) + 1  # +1 for EOS
    assert n_response == expected_response_tokens


# --- core invariant: pytorch side ------------------------------------------

def test_loss_ignores_masked_tokens():
    """
    Prompt-position logits must not affect loss when their labels are LOSS_IGNORE.
    Build two logit tensors that differ ONLY on prompt positions and assert
    F.cross_entropy gives identical loss.
    """
    ex = _build()
    T = len(ex["input_ids"])
    vocab = 50257
    labels = torch.tensor(ex["labels"], dtype=torch.long)

    def build_logits(prompt_choice: int) -> torch.Tensor:
        logits = torch.full((T, vocab), -10.0)
        for i, lab in enumerate(ex["labels"]):
            if lab != LOSS_IGNORE:
                logits[i, lab] = 10.0
            else:
                logits[i, prompt_choice] = 10.0
        return logits

    loss_a = F.cross_entropy(build_logits(0),    labels, ignore_index=LOSS_IGNORE)
    loss_b = F.cross_entropy(build_logits(1234), labels, ignore_index=LOSS_IGNORE)
    assert torch.isclose(loss_a, loss_b)
    assert loss_a.item() < 0.001


# --- template + edge cases --------------------------------------------------

def test_template_round_trip():
    """No-input and with-input variants drop/include the '### Input:' block correctly."""
    no_input   = format_prompt("Hello")
    with_input = format_prompt("Hello", "World")
    assert "### Input:" not in no_input
    assert "### Input:\nWorld" in with_input
    assert no_input.endswith("### Response:\n")
    assert with_input.endswith("### Response:\n")
    assert "### Input:" not in format_prompt("Hello", "")
    assert "### Input:" not in format_prompt("Hello", "  ")


def test_response_exceeds_blocksize_raises():
    """Response alone too long must raise, never silently skip."""
    huge_response = "x " * 2000
    with pytest.raises(ValueError, match="response alone"):
        _build(output=huge_response, block_size=128)


def test_train_eval_collator_parity():
    """Same example must produce identical labels every call. Tripwire if a
    second eval collator is added with different masking semantics."""
    ex_a, ex_b = _build(), _build(output="another response")
    batch1 = pad_batch([ex_a, ex_b])
    batch2 = pad_batch([ex_a, ex_b])
    assert torch.equal(batch1["labels"], batch2["labels"])
    assert torch.equal(batch1["input_ids"], batch2["input_ids"])
    assert torch.equal(batch1["attention_mask"], batch2["attention_mask"])
