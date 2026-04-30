"""Entraînement SRCNN sur les paires LR/HR."""
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
from methods.level5_srcnn import SRCNN


class SRDataset(Dataset):
    def __init__(self, lr_dirs: list[Path], hr_dirs: list[Path], patch_size: int, stride: int):
        self.pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for lr_d, hr_d in zip(lr_dirs, hr_dirs):
            for lr_f in sorted(lr_d.glob("*.png")):
                hr_f = hr_d / lr_f.name
                if not hr_f.exists():
                    continue
                lr = cv2.cvtColor(cv2.imread(str(lr_f)), cv2.COLOR_BGR2GRAY)
                hr = cv2.cvtColor(cv2.imread(str(hr_f)), cv2.COLOR_BGR2GRAY)
                lr_up = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)
                h, w = hr.shape
                for y in range(0, h - patch_size + 1, stride):
                    for x in range(0, w - patch_size + 1, stride):
                        self.pairs.append((
                            lr_up[y:y + patch_size, x:x + patch_size],
                            hr[y:y + patch_size, x:x + patch_size],
                        ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lr, hr = self.pairs[idx]
        to_t = lambda a: torch.from_numpy(a.astype(np.float32) / 255.0).unsqueeze(0)
        return to_t(lr), to_t(hr)


def train():
    lr_dirs = sorted(FRAMES_LR_DIR.iterdir())
    hr_dirs = sorted(FRAMES_HR_DIR.iterdir())
    dataset = SRDataset(lr_dirs, hr_dirs, PATCH_SIZE, STRIDE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = SRCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for lr_patch, hr_patch in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            lr_patch, hr_patch = lr_patch.to(DEVICE), hr_patch.to(DEVICE)
            optimizer.zero_grad()
            out = model(lr_patch)
            loss = criterion(out, hr_patch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} — loss: {total_loss / len(loader):.6f}")

    out_path = Path(__file__).parent / "srcnn_weights.pth"
    torch.save(model.state_dict(), out_path)
    print(f"Poids sauvegardés : {out_path}")


if __name__ == "__main__":
    train()
