"""Niveau 6 — ESPCN et FSRCNN (réseaux efficaces pour SR temps-réel)."""
import sys
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SCALE_FACTOR, DEVICE


class ESPCN(nn.Module):
    """Shi et al., 2016 — sub-pixel convolution."""
    def __init__(self, scale: int = SCALE_FACTOR, in_channels: int = 1):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, in_channels * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FSRCNN(nn.Module):
    """Dong et al., 2016 — version rapide de SRCNN."""
    def __init__(self, scale: int = SCALE_FACTOR, d: int = 56, s: int = 12, m: int = 4):
        super().__init__()
        self.scale = scale
        layers = [
            nn.Conv2d(1, d, kernel_size=5, padding=2), nn.PReLU(),
            nn.Conv2d(d, s, kernel_size=1), nn.PReLU(),
        ]
        for _ in range(m):
            layers += [nn.Conv2d(s, s, kernel_size=3, padding=1), nn.PReLU()]
        layers += [
            nn.Conv2d(s, d, kernel_size=1), nn.PReLU(),
            nn.ConvTranspose2d(d, 1, kernel_size=9, stride=scale,
                               padding=4, output_padding=scale - 1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)


def _from_tensor(tensor: torch.Tensor, color: bool = False) -> np.ndarray:
    arr = tensor.squeeze().detach().cpu().numpy()
    out = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR) if color else out


def infer(img: np.ndarray, model: nn.Module, device: str = DEVICE) -> np.ndarray:
    model.eval().to(device)
    inp = _to_tensor(img).to(device)
    with torch.no_grad():
        out = model(inp)
    return _from_tensor(out, color=(img.ndim == 3))


def load_espcn(weights_path: str, scale: int = SCALE_FACTOR, device: str = DEVICE) -> ESPCN:
    model = ESPCN(scale)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model


def load_fsrcnn(weights_path: str, scale: int = SCALE_FACTOR, device: str = DEVICE) -> FSRCNN:
    model = FSRCNN(scale)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model
