import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # fused Q, K, V projection — one matmul instead of three
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection: mixes the concatenated head outputs
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head  # hs

        # lower-triangular mask: 1 where attention is allowed, 0 where blocked
        # shape (1, 1, T, T) — broadcasts over (B, n_head, T, T)
        # buffer: moves with .to(device) but is not a learnable parameter
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, n_embd (C = n_head * head_size)

        # single matmul produces Q, K, V for all heads
        # (B, T, C) -> (B, T, 3*n_embd)
        qkv = self.c_attn(x)                          # (B, T, 3*n_embd)
        q, k, v = qkv.split(self.n_embd, dim=2)       # each: (B, T, n_embd)

        # split embedding dim into heads, then put head dim before sequence dim
        # each head gets its own (T, hs) slice and runs attention independently
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, n_head, T, hs)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, n_head, T, hs)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, n_head, T, hs)

        # attention scores: how much each query position attends to each key position
        # (B, n_head, T, hs) @ (B, n_head, hs, T) -> (B, n_head, T, T)
        #
        # WHY scale by 1/sqrt(head_size):
        # dot products grow in magnitude proportional to head_size.
        # large logits push softmax into saturation (outputs near 0 or 1),
        # which kills gradients. dividing by sqrt(d_k) keeps variance ~1
        # regardless of head size — this is from the original "Attention is All You Need".
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))  # (B, n_head, T, T)

        # causal mask: position i may only attend to positions 0..i
        # future positions become -inf so softmax assigns them exactly 0 weight
        #
        # WHY softmax happens AFTER masking:
        # softmax normalizes over all keys. if you softmax first, future tokens
        # get nonzero probability — zeroing them out afterward breaks the
        # "probabilities sum to 1" guarantee and leaks information.
        # mask first, then softmax: -inf -> exp(-inf) = 0, renormalization is correct.
        #
        # WHAT breaks if the mask is wrong:
        # an upper-triangular mask, off-by-one, or wrong shape means the model
        # can see future tokens during training. loss drops fast (it's cheating),
        # but at inference (no future tokens) generation collapses to repetition or noise.
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))  # (B, n_head, T, T)

        # normalize to probabilities along the key dimension
        #
        # WHERE NaNs come from:
        # if every position in a row is masked (-inf), softmax produces 0/0 = NaN.
        # this only happens with an empty/fully-masked sequence — rare in practice.
        # NaNs also appear if pre-softmax scores overflow float16; bfloat16 +
        # the sqrt(d_k) scaling above prevents this in normal training.
        att = F.softmax(att, dim=-1)                                          # (B, n_head, T, T)

        # weighted sum of values: each output token is a mixture of value vectors
        # (B, n_head, T, T) @ (B, n_head, T, hs) -> (B, n_head, T, hs)
        y = att @ v                                                            # (B, n_head, T, hs)

        # merge heads: transpose back then flatten head and head_size dims into C
        # contiguous() required because transpose makes storage non-contiguous,
        # and view() only works on contiguous tensors
        y = y.transpose(1, 2).contiguous().view(B, T, C)                     # (B, T, n_embd)

        # learned linear mix of all head outputs
        y = self.c_proj(y)                                                    # (B, T, n_embd)
        return y
