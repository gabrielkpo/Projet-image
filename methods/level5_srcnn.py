"""Niveau 5 — SRCNN (Dong et al., 2014)."""
import sys
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SCALE_FACTOR, DEVICE


class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def preprocess(img: np.ndarray, scale: int) -> tuple[torch.Tensor, np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    up = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(up.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    return tensor, up


def postprocess(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.squeeze().detach().cpu().numpy()
    return np.clip(arr * 255, 0, 255).astype(np.uint8)


def infer(img: np.ndarray, model: SRCNN, scale: int = SCALE_FACTOR,
          device: str = DEVICE) -> np.ndarray:
    model.eval()
    model.to(device)
    inp, _ = preprocess(img, scale)
    inp = inp.to(device)
    with torch.no_grad():
        out = model(inp)
    sr_gray = postprocess(out)
    if img.ndim == 3:
        sr_gray = cv2.cvtColor(sr_gray, cv2.COLOR_GRAY2BGR)
    return sr_gray


def load_model(weights_path: str, device: str = DEVICE) -> SRCNN:
    model = SRCNN()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model
