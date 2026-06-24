"""
SFT data pipeline. Core invariant: prompt and pad positions get
LOSS_IGNORE labels so cross-entropy(ignore_index=-100) skips them.
format_prompt is the single source of truth for the template — import,
never duplicate, or train/eval drift silently.
"""
from typing import List, Optional, Tuple, Dict

import torch
from torch.utils.data import Dataset

EOS_ID      = 50256   # tiktoken "gpt2" <|endoftext|>
LOSS_IGNORE = -100    # F.cross_entropy default ignore_index

PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)
PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


def format_prompt(instruction: str, input: Optional[str] = None) -> str:
    if input and input.strip():
        return PROMPT_WITH_INPUT.format(instruction=instruction, input=input)
    return PROMPT_NO_INPUT.format(instruction=instruction)


def format_full(instruction: str, input: Optional[str], output: str) -> Tuple[str, str]:
    # EOS is appended in tokenize_example, not here.
    return format_prompt(instruction, input), output


def tokenize_example(enc, prompt_str: str, response_str: str, block_size: int) -> Dict[str, List[int]]:
    """
    Left-truncates the prompt to preserve every response token; raises if the
    response alone exceeds block_size (rejecting silent zero-loss examples).
    """
    prompt_ids = enc.encode(prompt_str)
    response_ids = enc.encode(response_str) + [EOS_ID]

    if len(response_ids) >= block_size:
        raise ValueError(
            f"response alone ({len(response_ids)} tokens) >= block_size "
            f"({block_size}); increase block_size or shorten the response."
        )

    max_prompt = block_size - len(response_ids)
    if len(prompt_ids) > max_prompt:
        prompt_ids = prompt_ids[-max_prompt:]

    input_ids = prompt_ids + response_ids
    labels    = [LOSS_IGNORE] * len(prompt_ids) + list(response_ids)
    return {"input_ids": input_ids, "labels": labels}


def pad_batch(examples: List[Dict[str, List[int]]], pad_id: int = EOS_ID) -> Dict[str, torch.Tensor]:
    # pad_id defaults to EOS but pad positions are masked, so the value never reaches the loss.
    max_len = max(len(ex["input_ids"]) for ex in examples)
    B = len(examples)
    input_ids      = torch.full((B, max_len), pad_id,      dtype=torch.long)
    labels         = torch.full((B, max_len), LOSS_IGNORE, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len),             dtype=torch.long)
    for i, ex in enumerate(examples):
        L = len(ex["input_ids"])
        input_ids[i, :L]      = torch.tensor(ex["input_ids"], dtype=torch.long)
        labels[i, :L]         = torch.tensor(ex["labels"],    dtype=torch.long)
        attention_mask[i, :L] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def load_alpaca_subset(n: int = 1000, seed: int = 1337) -> List[Dict[str, str]]:
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return [{"instruction": ex["instruction"],
             "input":       ex["input"],
             "output":      ex["output"]} for ex in ds]


class AlpacaSFTDataset(Dataset):
    """Pre-tokenized SFT dataset. Fails loudly at __init__ on bad examples."""

    def __init__(self, examples: List[Dict[str, str]], enc, block_size: int):
        self.block_size = block_size
        self._cached: List[Dict[str, List[int]]] = []
        for ex in examples:
            prompt, response = format_full(ex["instruction"], ex.get("input"), ex["output"])
            self._cached.append(tokenize_example(enc, prompt, response, block_size))

    def __len__(self) -> int:
        return len(self._cached)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self._cached[idx]
