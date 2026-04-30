"""Entraînement ESPCN — entrée LR native, sortie HR via sub-pixel."""
import sys
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FRAMES_HR_DIR, FRAMES_LR_DIR, SCALE_FACTOR, PATCH_SIZE, STRIDE, BATCH_SIZE, LEARNING_RATE, EPOCHS, DEVICE
from methods.level6_cnn_optimized import ESPCN


class ESPCNDataset(Dataset):
    def __init__(self, lr_dirs: list[Path], hr_dirs: list[Path],
                 scale: int, patch_size: int, stride: int):
        self.scale = scale
        self.pairs: list[tuple[np.ndarray, np.ndarray]] = []
        lr_patch_size = patch_size // scale
        for lr_d, hr_d in zip(lr_dirs, hr_dirs):
            for lr_f in sorted(lr_d.glob("*.png")):
                hr_f = hr_d / lr_f.name
                if not hr_f.exists():
                    continue
                lr = cv2.cvtColor(cv2.imread(str(lr_f)), cv2.COLOR_BGR2GRAY)
                hr = cv2.cvtColor(cv2.imread(str(hr_f)), cv2.COLOR_BGR2GRAY)
                h, w = lr.shape
                for y in range(0, h - lr_patch_size + 1, stride // scale):
                    for x in range(0, w - lr_patch_size + 1, stride // scale):
                        lr_patch = lr[y:y + lr_patch_size, x:x + lr_patch_size]
                        hy, hx = y * scale, x * scale
                        hr_patch = hr[hy:hy + patch_size, hx:hx + patch_size]
                        if hr_patch.shape == (patch_size, patch_size):
                            self.pairs.append((lr_patch, hr_patch))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lr, hr = self.pairs[idx]
        to_t = lambda a: torch.from_numpy(a.astype(np.float32) / 255.0).unsqueeze(0)
        return to_t(lr), to_t(hr)


def train():
    lr_dirs = sorted(FRAMES_LR_DIR.iterdir())
    hr_dirs = sorted(FRAMES_HR_DIR.iterdir())
    dataset = ESPCNDataset(lr_dirs, hr_dirs, SCALE_FACTOR, PATCH_SIZE, STRIDE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = ESPCN(SCALE_FACTOR).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for lr_patch, hr_patch in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            lr_patch, hr_patch = lr_patch.to(DEVICE), hr_patch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(lr_patch), hr_patch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} — loss: {total_loss / len(loader):.6f}")

    out_path = Path(__file__).parent / "espcn_weights.pth"
    torch.save(model.state_dict(), out_path)
    print(f"Poids sauvegardés : {out_path}")


if __name__ == "__main__":
    train()
