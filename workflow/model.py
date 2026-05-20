"""FSRCNN model — repris de suxrobGM/fsrcnn (d=64, s=16, m=6, scale=4)."""
from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn


class FSRCNN(nn.Module):
    def __init__(self, scale: int = 4, d: int = 64, s: int = 16, m: int = 6) -> None:
        super().__init__()
        self.scale = scale

        self.feature = nn.Sequential(
            nn.Conv2d(1, d, kernel_size=5, padding=2),
            nn.PReLU(num_parameters=d),
        )
        self.shrink = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(num_parameters=s),
        )
        mapping = []
        for _ in range(m):
            mapping += [nn.Conv2d(s, s, kernel_size=3, padding=1), nn.PReLU(num_parameters=s)]
        self.map = nn.Sequential(*mapping)
        self.expand = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(num_parameters=d),
        )
        self.deconv = nn.ConvTranspose2d(
            d, 1, kernel_size=9, stride=scale, padding=4, output_padding=scale - 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature(x)
        x = self.shrink(x)
        x = self.map(x)
        x = self.expand(x)
        return self.deconv(x)


def load_model(ckpt_path: str | Path, device: str = "cpu") -> FSRCNN:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model = FSRCNN(scale=4, d=64, s=16, m=6)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model.to(device)
