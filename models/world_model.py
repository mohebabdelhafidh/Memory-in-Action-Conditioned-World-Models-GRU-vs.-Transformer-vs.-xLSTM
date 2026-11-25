"""Action-conditioned world model: encoder + action embedding + dynamics core + decoder + reward head.

    frame_t -> Encoder -> z_t --+
                                +-> Linear -> [core] -> h_t (belief state) -+-> Decoder     -> predicted frame_{t+1}
    action_t -> Embedding ------+                                           +-> reward head -> reward_{t+1} logit

Swap the core by config (gru | memoryless now; transformer | xlstm later). Everything else is shared.
"""
import torch
from torch import nn

from models.decoder import Decoder
from models.encoder import Encoder


class WorldModel(nn.Module):
    def __init__(self, core, latent_dim: int = 256, action_dim: int = 32, num_actions: int = 4):
        super().__init__()
        self.core = core
        self.encoder = Encoder(latent_dim)
        self.action_emb = nn.Embedding(num_actions, action_dim)
        self.inp = nn.Linear(latent_dim + action_dim, core.dim)
        self.decoder = Decoder(in_dim=core.out_dim)
        self.reward_head = nn.Sequential(nn.Linear(core.out_dim, 128), nn.SiLU(), nn.Linear(128, 1))

    def _embed(self, frames, actions):
        z = self.encoder(frames)
        return self.inp(torch.cat([z, self.action_emb(actions)], dim=-1))

    def _heads(self, h, state):
        return dict(
            h=h,  # belief state: what the memory probes read
            next_frame=self.decoder(h),
            reward_logit=self.reward_head(h).squeeze(-1),
            state=state,
        )

    def forward(self, frames, actions, state=None):
        """frames (B, T, 64, 64, 3) uint8, actions (B, T) long. Output at step t predicts frame/reward t+1."""
        h, state = self.core(self._embed(frames, actions), state)
        return self._heads(h, state)

    def step(self, frame, action, state=None):
        """One step with explicit state: frame (B, 64, 64, 3), action (B,). Used for planning and efficiency tests."""
        h, state = self.core.step(self._embed(frame, action), state)
        return self._heads(h, state)


def build_core(c: dict):
    """Build a dynamics core from a config dict (shared by the simulated-task and the real-data world models)."""
    kind = c["type"]
    if kind == "gru":
        from models.cores.gru import GRUCore
        core = GRUCore(dim=c.get("dim", 256), hidden=c["hidden"], layers=c.get("layers", 1))
    elif kind == "memoryless":
        from models.cores.memoryless import MemorylessCore
        core = MemorylessCore(dim=c.get("dim", 256), hidden=c["hidden"])
    elif kind == "transformer":
        from models.cores.transformer import TransformerCore
        core = TransformerCore(dim=c.get("dim", 256), out_dim=c["hidden"], layers=c.get("layers", 2),
                               heads=c.get("heads", 4), mlp_ratio=c.get("mlp_ratio", 2))
    elif kind == "xlstm":
        from models.cores.xlstm_core import XLSTMCore
        core = XLSTMCore(dim=c.get("dim", 288), out_dim=c["hidden"], blocks=c.get("blocks", 2),
                         heads=c.get("heads", 4), context_length=c.get("context_length", 4096))
    else:
        raise NotImplementedError(f"core type {kind!r} is not written yet (gru, memoryless, transformer, xlstm are)")
    return core


def build_world_model(cfg: dict) -> WorldModel:
    m = cfg["model"]
    core = build_core(cfg["core"])
    return WorldModel(core, latent_dim=m.get("latent_dim", 256), action_dim=m.get("action_dim", 32))


def count_params(module: nn.Module) -> float:
    return sum(p.numel() for p in module.parameters()) / 1e6
