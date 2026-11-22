"""xLSTM core: a thin wrapper around the official `xlstm` package's xLSTMBlockStack (mLSTM blocks only).

What the installed xlstm 2.0.6 gives us (from its source):
    stack(x)            -> (B, T, D)             parallel pass over a whole sequence; takes NO state, returns NO state
    stack.step(x, state) -> (out, state)         one step; `state` is a nested dict {"block_i": {...}} that step() fills in place

So this wrapper follows the shared core interface like this:
    forward(x_seq)               : parallel pass, returns (h_seq, None). Used for training, probes and long-context sweeps.
    forward(x_seq, state=given)  : falls back to a step-by-step loop (slower) and returns the final state.
    step(x_t, state)             : one recurrent step with explicit constant-size state (planning, VRAM/latency tests).

Default size (dim 288, 2 blocks, 4 heads, output projected to out_dim 512) is meant to land near the GRU core's
parameter count (~1.18M). Check it with the parameter count script before training; dim is the knob to adjust.
Only the mLSTM block type is used (slstm_at=[]).
"""
import torch
from torch import nn

from models.cores.base import Core


class XLSTMCore(Core):
    def __init__(self, dim: int = 288, out_dim: int = 512, blocks: int = 2, heads: int = 4,
                 context_length: int = 4096, conv_kernel: int = 4, qkv_blocksize: int = 4):
        super().__init__()
        # same construction route as the smoke test that passed in this venv (strict dacite config)
        from dacite import Config as DaciteConfig
        from dacite import from_dict
        from omegaconf import OmegaConf
        from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig

        raw = OmegaConf.to_container(OmegaConf.create(dict(
            mlstm_block=dict(mlstm=dict(conv1d_kernel_size=conv_kernel, qkv_proj_blocksize=qkv_blocksize,
                                        num_heads=heads)),
            context_length=context_length, num_blocks=blocks, embedding_dim=dim, slstm_at=[],
        )))
        cfg = from_dict(xLSTMBlockStackConfig, raw, config=DaciteConfig(strict=True))
        self.dim, self.out_dim = dim, out_dim
        self.stack = xLSTMBlockStack(cfg)
        self.out = nn.Linear(dim, out_dim)
        self._step_3d = None  # does stack.step want (B, 1, D) or (B, D)? detected on the first call

    def forward(self, x, state=None):
        if state is None:
            return self.out(self.stack(x)), None
        hs = []
        for t in range(x.shape[1]):
            h, state = self.step(x[:, t], state)
            hs.append(h)
        return torch.stack(hs, dim=1), state

    def step(self, x, state=None):
        orig = {} if state is None else state
        attempts = [True, False] if self._step_3d is None else [self._step_3d]
        last = None
        for three_d in attempts:
            try:
                # stack.step fills the state dict in place, so hand it a shallow copy of the caller's dict
                h, new = self.stack.step(x.unsqueeze(1) if three_d else x, dict(orig))
                self._step_3d = three_d
                return self.out(h.reshape(x.shape[0], -1)), new
            except (RuntimeError, ValueError, IndexError, TypeError, AssertionError) as e:
                last = e
        raise last
