"""Frame decoder for the real-data world model: (..., in_dim) -> (..., H, W, 3) in [0, 1] (unsquashed linear output).

Same architecture as models/decoder.py with the frame size as a parameter (H and W divisible by 16).
"""
import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self, in_dim: int = 512, channels=(256, 128, 64, 32), size=(64, 112)):
        super().__init__()
        n = 2 ** len(channels)
        assert size[0] % n == 0 and size[1] % n == 0, f"frame size {size} must be divisible by {n}"
        self.in_dim, self.size, self.c0 = in_dim, tuple(size), channels[0]
        self.h0, self.w0 = size[0] // n, size[1] // n
        self.fc = nn.Linear(in_dim, self.c0 * self.h0 * self.w0)
        layers = []
        for i, c_in in enumerate(channels):
            c_out = channels[i + 1] if i + 1 < len(channels) else channels[-1]
            layers += [nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1),
                       nn.GroupNorm(8, c_out), nn.SiLU()]
        self.up = nn.Sequential(*layers)
        self.out = nn.Conv2d(channels[-1], 3, kernel_size=3, padding=1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        lead = feat.shape[:-1]
        x = self.fc(feat.reshape(-1, self.in_dim)).reshape(-1, self.c0, self.h0, self.w0)
        x = self.out(self.up(x)) + 0.5
        return x.permute(0, 2, 3, 1).reshape(*lead, *self.size, 3)

    @staticmethod
    def to_uint8(frames: torch.Tensor) -> torch.Tensor:
        return (frames.clamp(0, 1) * 255).round().to(torch.uint8)
