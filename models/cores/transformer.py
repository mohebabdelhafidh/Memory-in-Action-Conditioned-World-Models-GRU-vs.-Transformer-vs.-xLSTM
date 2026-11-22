"""Causal Transformer core with rotary positions (RoPE) and a full key/value cache.

Same interface as the other cores (see models/cores/base.py):
    forward(x_seq, state=None) -> (h_seq, state)   whole sequence at once (training); with a cache it continues from it
    step(x_t, state=None)      -> (h_t, state)     one step; the state is the KV cache of every past step

The "state" is NOT constant-size: it holds keys and values for every step seen so far, so memory grows
linearly with context length. That is deliberate (hypothesis H2): no sliding window, nothing is thrown away.

Default size is chosen so the core matches the GRU core in parameters (~1.18M):
    dim 256, 2 layers, 4 heads, MLP ratio 2, output projected to `out_dim` = 512 (same belief size as the GRU).
"""
import torch
import torch.nn.functional as F
from torch import nn

from models.cores.base import Core


def apply_rope(x, cos, sin):
    """x: (B, H, T, Dh); cos/sin: (T, Dh/2). Rotates consecutive channel pairs by position-dependent angles."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))

    def forward(self, x, cos, sin, past=None):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin).to(v.dtype), apply_rope(k, cos, sin).to(v.dtype)
        n_past = 0 if past is None else past[0].shape[2]
        if past is not None:  # cached keys are already rotated at their own absolute positions
            k, v = torch.cat([past[0], k], dim=2), torch.cat([past[1], v], dim=2)
        if n_past == 0:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1)
        elif T == 1:
            a = F.scaled_dot_product_attention(q, k, v)  # the one new query may see every past key
        else:  # a chunk continuing from a cache: query i may see keys 0 .. n_past + i
            mask = torch.ones(T, n_past + T, dtype=torch.bool, device=x.device).tril(diagonal=n_past)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        x = x + self.mlp(self.ln2(x))
        return x, (k, v)


class TransformerCore(Core):
    def __init__(self, dim: int = 256, out_dim: int = 512, layers: int = 2, heads: int = 4,
                 mlp_ratio: int = 2, rope_base: float = 10000.0):
        super().__init__()
        assert dim % heads == 0 and (dim // heads) % 2 == 0, "head size must be even for rotary positions"
        self.dim, self.out_dim = dim, out_dim
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, out_dim)
        head = dim // heads
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, head, 2).float() / head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, state=None):
        past_len = 0 if state is None else state[0][0].shape[2]
        pos = torch.arange(past_len, past_len + x.shape[1], device=x.device, dtype=torch.float32)
        freqs = pos[:, None] * self.inv_freq[None, :]  # angles stay in fp32; cos/sin are cast to x's dtype
        cos, sin = freqs.cos().to(x.dtype), freqs.sin().to(x.dtype)
        cache = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, cos, sin, None if state is None else state[i])
            cache.append(kv)
        return self.out(self.ln_f(x)), cache

    def step(self, x, state=None):
        h, state = self.forward(x.unsqueeze(1), state)
        return h.squeeze(1), state
