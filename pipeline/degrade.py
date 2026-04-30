"""Génère des frames LR à partir des frames HR (bicubique + bruit optionnel)."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FRAMES_HR_DIR, FRAMES_LR_DIR, SCALE_FACTOR, BLUR_KERNEL_SIZE, NOISE_STD


def degrade(img: np.ndarray, scale: int = SCALE_FACTOR,
            blur_k: int = BLUR_KERNEL_SIZE, noise_std: float = NOISE_STD) -> np.ndarray:
    h, w = img.shape[:2]
    if blur_k > 1:
        img = cv2.GaussianBlur(img, (blur_k, blur_k), 0)
    lr = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, lr.shape).astype(np.float32)
        lr = np.clip(lr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return lr


def degrade_directory(hr_dir: Path, lr_dir: Path, **kwargs) -> None:
    lr_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(hr_dir.glob("*.png"))
    for f in frames:
        img = cv2.imread(str(f))
        lr = degrade(img, **kwargs)
        cv2.imwrite(str(lr_dir / f.name), lr)
    print(f"[degrade] {len(frames)} frames dégradées → {lr_dir}")


if __name__ == "__main__":
    for hr_subdir in FRAMES_HR_DIR.iterdir():
        if hr_subdir.is_dir():
            degrade_directory(hr_subdir, FRAMES_LR_DIR / hr_subdir.name)
