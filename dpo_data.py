"""
DPO data pipeline. Each example is a (prompt, chosen, rejected) triple.
completion_mask is 1 over response tokens, 0 over prompt and pad tokens; only
mask=1 positions contribute to the sequence log-probability that DPO compares.
Prompt template is imported from sft_data so SFT and DPO see the same shape.
"""
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from sft_data import EOS_ID, format_prompt

PAD_ID = EOS_ID


def format_pair(question: str, chosen: str, rejected: str) -> Tuple[str, str, str]:
    return format_prompt(question), chosen, rejected


def tokenize_completion(enc, prompt_str: str, completion_str: str, block_size: int) -> Dict[str, List[int]]:
    """
    Returns {'input_ids': List[int], 'completion_mask': List[int]}.
    completion_mask is 1 over response tokens (including EOS), 0 over prompt.
    Left-truncates prompt if total exceeds block_size; raises if completion alone is too long.
    """
    prompt_ids = enc.encode(prompt_str)
    completion_ids = enc.encode(completion_str) + [EOS_ID]

    if len(completion_ids) >= block_size:
        raise ValueError(
            f"completion alone ({len(completion_ids)} tokens) >= block_size "
            f"({block_size}); increase block_size or shorten the completion."
        )

    max_prompt = block_size - len(completion_ids)
    if len(prompt_ids) > max_prompt:
        prompt_ids = prompt_ids[-max_prompt:]

    input_ids       = prompt_ids + completion_ids
    completion_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
    return {"input_ids": input_ids, "completion_mask": completion_mask}


def tokenize_dpo_example(enc, question: str, chosen: str, rejected: str, block_size: int) -> Dict[str, List[int]]:
    prompt_str, chosen_str, rejected_str = format_pair(question, chosen, rejected)
    c = tokenize_completion(enc, prompt_str, chosen_str,   block_size)
    r = tokenize_completion(enc, prompt_str, rejected_str, block_size)
    return {
        "chosen_ids":      c["input_ids"],
        "chosen_mask":     c["completion_mask"],
        "rejected_ids":    r["input_ids"],
        "rejected_mask":   r["completion_mask"],
    }


def pad_dpo_batch(examples: List[Dict[str, List[int]]], pad_id: int = PAD_ID) -> Dict[str, torch.Tensor]:
    """
    Right-pad chosen and rejected sides independently to their respective batch-max length.
    Pad positions get input id = pad_id and completion_mask = 0 (excluded from logp sum).
    """
    def pad_side(prefix: str) -> Tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(ex[f"{prefix}_ids"]) for ex in examples)
        B = len(examples)
        ids  = torch.full((B, max_len), pad_id, dtype=torch.long)
        mask = torch.zeros((B, max_len),         dtype=torch.long)
        for i, ex in enumerate(examples):
            L = len(ex[f"{prefix}_ids"])
            ids[i,  :L] = torch.tensor(ex[f"{prefix}_ids"],  dtype=torch.long)
            mask[i, :L] = torch.tensor(ex[f"{prefix}_mask"], dtype=torch.long)
        return ids, mask

    chosen_ids,   chosen_mask   = pad_side("chosen")
    rejected_ids, rejected_mask = pad_side("rejected")
    return {
        "chosen_ids":    chosen_ids,
        "chosen_mask":   chosen_mask,
        "rejected_ids":  rejected_ids,
        "rejected_mask": rejected_mask,
    }


def load_orca_dpo_subset(n: int = 100, seed: int = 1337) -> List[Dict[str, str]]:
    from datasets import load_dataset
    ds = load_dataset("Intel/orca_dpo_pairs", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return [{"question": ex["question"],
             "chosen":   ex["chosen"],
             "rejected": ex["rejected"]} for ex in ds]


class OrcaDPODataset(Dataset):
    """Pre-tokenized DPO dataset. Fails loud at __init__ on bad examples."""

    def __init__(self, examples: List[Dict[str, str]], enc, block_size: int):
        self.block_size = block_size
        self._cached: List[Dict[str, List[int]]] = []
        for ex in examples:
            self._cached.append(
                tokenize_dpo_example(enc, ex["question"], ex["chosen"], ex["rejected"], block_size)
            )

    def __len__(self) -> int:
        return len(self._cached)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self._cached[idx]
