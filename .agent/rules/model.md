# Model Architecture Rules

Applies to: `model.py`, `attention.py`

---

## Weight initialization

`_init_weights` applies two different std values:
- Standard linear/embedding: `std = 0.02`
- Residual projection (`NANOGPT_SCALE_INIT = 1`): `std *= (2 * n_layer) ** -0.5`

The residual scaling is the GPT-2 paper's scheme — it keeps residual stream variance stable as depth grows. `NANOGPT_SCALE_INIT` is the flag that triggers it. It is set on `MLP.c_proj` and `CausalSelfAttention.c_proj` only. Do not add it to other layers without understanding why.

## Attention scaling

Pre-softmax scores are scaled by `1 / sqrt(head_size)`. Do not remove or adjust this. Without it, dot products grow with head size, softmax saturates, and gradients die.

## Causal mask buffer

The `bias` buffer in `CausalSelfAttention.__init__` is registered with `register_buffer` — it moves to the correct device automatically with `.to(device)` but is not a learnable parameter and will not appear in `optimizer.state_dict()`. Do not change it to a plain tensor or a `nn.Parameter`.

## Weight tying

`self.transformer.wte.weight = self.lm_head.weight` means both layers share the same underlying tensor. A gradient through `lm_head` updates `wte` and vice versa. This is intentional. If you add a new projection that should be tied, use the same assignment pattern — do not use `copy_()`.

## vocab_size=50304 in train.py

`train.py` creates `GPT(GPTConfig(vocab_size=50304))` rather than the default 50257. 50304 is the next multiple of 64 above 50257 — this makes the vocabulary size a power-of-2-aligned number, which improves GPU kernel efficiency on the final linear projection. The extra 47 token embeddings are never used in practice.

## from_pretrained weight loading

`GPT.from_pretrained` loads GPT-2 weights from HuggingFace. It transposes Conv1D weights to Linear layout. The four transposed layers are hardcoded in `transposed` list — if you add new attention projections with Conv1D origin, update that list. Loading only copies keys present in both models; extra HF keys are silently skipped.

## model.py vs train.py duplication

`model.py` intentionally re-defines `GPTConfig`, `GPT`, `Block`, and `MLP` as a standalone module. `train.py` inlines its own versions of the same classes so it can be run without importing `model`. If you change the model architecture, update **both** files and ensure they remain in sync.
