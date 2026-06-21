"""
Model loaders shared by evals and SFT.

Two entry points:
    load_checkpoint(path, device) -> (model, step)
    load_hf_pretrained(model_type, device) -> (model, None)

Both return the model already moved to device and set to eval() mode. The caller
is responsible for switching to train() when appropriate.
"""
import torch

from model import GPT, GPTConfig


def load_checkpoint(path, device):
    """Load a checkpoint saved by train.py. Strips DDP and torch.compile prefixes."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = GPT(config)
    sd = ckpt["model"]
    cleaned = {}
    for k, v in sd.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    model.load_state_dict(cleaned)
    return model.to(device).eval(), ckpt.get("step")


def load_hf_pretrained(model_type, device):
    """Load HuggingFace GPT-2 weights into a nano-gpt GPT instance."""
    from transformers import GPT2LMHeadModel
    arch = {
        "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),
        "gpt2-medium": dict(n_layer=16, n_head=16, n_embd=1024),
        "gpt2-large":  dict(n_layer=20, n_head=20, n_embd=1280),
        "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600),
    }[model_type]
    config = GPTConfig(**arch, vocab_size=50257, block_size=1024)
    model = GPT(config)

    sd = model.state_dict()
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()
    sd_keys_hf = [k for k in sd_hf.keys()
                  if not k.endswith(".attn.masked_bias")
                  and not k.endswith(".attn.bias")]
    transposed = ("attn.c_attn.weight", "attn.c_proj.weight",
                  "mlp.c_fc.weight", "mlp.c_proj.weight")
    for k in sd_keys_hf:
        if k not in sd:
            continue
        if any(k.endswith(w) for w in transposed):
            with torch.no_grad():
                sd[k].copy_(sd_hf[k].t())
        else:
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])
    return model.to(device).eval(), None
