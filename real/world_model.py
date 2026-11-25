"""World model for real robot video: encoder + action network + dynamics core + decoder. No reward head.

    frame_t -> Encoder -> z_t --+                (or a learned "blind" vector when the frame is withheld)
                                +-> Linear -> [core] -> h_t -> Decoder -> predicted frame_{t+1}
    action_t -> MLP ------------+

Differences from the simulated-task model (models/world_model.py):
  * frames are (H, W) = (64, 112) by default instead of 64 x 64
  * actions are continuous vectors (8 numbers: joint-angle changes + gripper), embedded by a small MLP
  * observation dropout: at steps flagged `blind`, the encoder output is replaced by a learned vector, so the model
    still sees the action but no image. It must keep the scene in its belief state and predict from actions alone.
  * no reward head (real demonstrations have no useful reward signal here)
The dynamics cores (GRU / Transformer / xLSTM / memoryless) are the very same classes from models/cores/.
"""
import torch
from torch import nn

from models.world_model import build_core, count_params  # noqa: F401  (count_params re-exported)
from real.decoder import Decoder
from real.encoder import Encoder


class RealWorldModel(nn.Module):
    def __init__(self, core, latent_dim: int = 256, act_dim: int = 8, act_emb: int = 32, size=(64, 112)):
        super().__init__()
        self.core = core
        self.encoder = Encoder(latent_dim, size=size)
        self.action_net = nn.Sequential(nn.Linear(act_dim, act_emb), nn.SiLU(), nn.Linear(act_emb, act_emb))
        self.blind_z = nn.Parameter(torch.zeros(latent_dim))  # stands in for "no frame at this step"
        self.inp = nn.Linear(latent_dim + act_emb, core.dim)
        self.decoder = Decoder(in_dim=core.out_dim, size=size)

    def _embed(self, frames, actions, blind=None):
        z = self.encoder(frames)
        if blind is not None:
            z = torch.where(blind.unsqueeze(-1), self.blind_z.to(z.dtype).expand_as(z), z)
        return self.inp(torch.cat([z, self.action_net(actions)], dim=-1))

    def forward(self, frames, actions, blind=None, state=None):
        """frames (B, T, H, W, 3) uint8, actions (B, T, act_dim) float, blind (B, T) bool (True = frame withheld).
        The output at step t predicts frame t+1."""
        h, state = self.core(self._embed(frames, actions, blind), state)
        return dict(h=h, next_frame=self.decoder(h), state=state)

    def step(self, frame, action, blind=False, state=None):
        """One step with explicit state. frame (B, H, W, 3), action (B, act_dim); pass blind=True to withhold the frame."""
        if blind:
            z = self.blind_z.expand(action.shape[0], -1)
        else:
            z = self.encoder(frame)
        h, state = self.core.step(self.inp(torch.cat([z, self.action_net(action)], dim=-1)), state)
        return dict(h=h, next_frame=self.decoder(h), state=state)


def build_real_model(cfg: dict) -> RealWorldModel:
    m = cfg["model"]
    return RealWorldModel(build_core(cfg["core"]), latent_dim=m.get("latent_dim", 256), act_dim=m.get("act_dim", 8),
                          act_emb=m.get("act_emb", 32), size=tuple(m.get("size", [64, 112])))
