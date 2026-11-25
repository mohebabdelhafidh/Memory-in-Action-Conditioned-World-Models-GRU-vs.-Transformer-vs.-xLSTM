"""Frame encoder for the real-data world model: (..., H, W, 3) frames -> (..., latent_dim).

Same architecture as models/encoder.py (four stride-2 conv blocks + a linear layer), but the frame size is a
parameter (default 64 x 112, which keeps DROID's 16:9 shape). H and W must be divisible by 16.
"""
import torch
from torch import nn


def to_float(frames: torch.Tensor) -> torch.Tensor:
    """uint8 [0,255] -> float [0,1]; float input is assumed to be in [0,1] already."""
    return frames.float() / 255.0 if frames.dtype == torch.uint8 else frames.float()


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 256, channels=(32, 64, 128, 256), size=(64, 112)):
        super().__init__()
        n = 2 ** len(channels)
        assert size[0] % n == 0 and size[1] % n == 0, f"frame size {size} must be divisible by {n}"
        self.latent_dim, self.size = latent_dim, tuple(size)
        layers, c_in = [], 3
        for c in channels:
            layers += [nn.Conv2d(c_in, c, kernel_size=4, stride=2, padding=1), nn.GroupNorm(8, c), nn.SiLU()]
            c_in = c
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(c_in * (size[0] // n) * (size[1] // n), latent_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        lead = frames.shape[:-3]
        x = to_float(frames.reshape(-1, *frames.shape[-3:])).permute(0, 3, 1, 2) - 0.5
        return self.fc(self.conv(x).flatten(1)).reshape(*lead, self.latent_dim)
