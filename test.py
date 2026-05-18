"""Test sparse coding SR — entraînement + évaluation visuelle. Non versionné."""
import sys
import cv2

import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, ".")
from config import FRAMES_HR_DIR, FRAMES_LR_PHASE1_DIR, VIDEO_SCALE, RESULTS_VISUALS_DIR
from pipeline.metrics import psnr
import methods.level1_interpolation as l1
import methods.level4_ml_classic as l4

# ── Paramètres ────────────────────────────────────────────────────────────────
VIDEO_ID     = "33mqqm4QlJ8"
N_TRAIN      = 20    # frames pour entraîner
N_TEST       = 5     # frames pour évaluer
SPARSE_MODEL = Path(__file__).parent / "train" / "sparse_sr.joblib"

scale    = VIDEO_SCALE[VIDEO_ID]
lr_root  = FRAMES_LR_PHASE1_DIR / VIDEO_ID
hr_root  = FRAMES_HR_DIR / VIDEO_ID
lr_files = sorted(lr_root.glob("*.png"))

RESULTS_VISUALS_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Entraînement si modèle absent ─────────────────────────────────────────
def train():
    lr_imgs, hr_imgs = [], []
    for lr_f in lr_files[:N_TRAIN]:
        lr = cv2.imread(str(lr_f))
        hr = cv2.imread(str(hr_root / lr_f.name))
        if lr is not None and hr is not None:
            lr_imgs.append(lr)
            hr_imgs.append(hr)

    model = l4.SparseSR(
        n_atoms=128, patch_size=8, stride_train=6,
        stride_infer=2, n_nonzero=8, scale=scale, max_patches=30_000,
    )
    model.fit(lr_imgs, hr_imgs)
    model.save(SPARSE_MODEL)


# ── 2. Évaluation PSNR / SSIM ────────────────────────────────────────────────
def evaluate():
    model = l4.SparseSR(scale=scale).load(SPARSE_MODEL)

    print(f"\n{'Frame':<12} {'Bicubic PSNR':>13} {'Sparse PSNR':>12} {'Δ PSNR':>8}")
    print("-" * 48)
    for lr_f in lr_files[N_TRAIN:N_TRAIN + N_TEST]:
        lr = cv2.imread(str(lr_f))
        hr = cv2.imread(str(hr_root / lr_f.name))

        bic = l1.upscale(lr, scale, "bicubic")
        spa = model.predict(lr)

        p_bic = psnr(hr, bic)
        p_spa = psnr(hr, spa)
        print(f"{lr_f.name:<12} {p_bic:>13.2f} {p_spa:>12.2f} {p_spa - p_bic:>+8.2f}")


# ── 3. Comparaison visuelle sur un patch ─────────────────────────────────────
def save_visual(frame_idx: int = N_TRAIN):
    lr_f = lr_files[frame_idx]
    lr   = cv2.imread(str(lr_f))
    hr   = cv2.imread(str(hr_root / lr_f.name))

    model = l4.SparseSR(scale=scale).load(SPARSE_MODEL)
    bic   = l1.upscale(lr, scale, "bicubic")
    spa   = model.predict(lr)

    # Patch centré
    H, W  = hr.shape[:2]
    py, px, ph, pw = H // 3, W // 3, 200, 200

    def crop(img): return cv2.cvtColor(img[py:py+ph, px:px+pw], cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    fig.suptitle(f"Sparse coding SR (×{scale}) — {lr_f.name}", fontsize=12, fontweight="bold")

    for ax, img, title, color in [
        (axes[0], hr,  "HR référence",                              "#2ecc71"),
        (axes[1], bic, f"Bicubique\nPSNR = {psnr(hr,bic):.2f} dB", "#95a5a6"),
        (axes[2], spa, f"Sparse SR\nPSNR = {psnr(hr,spa):.2f} dB", "#3498db"),
    ]:
        ax.imshow(crop(img))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_edgecolor(color); spine.set_linewidth(3)

    plt.tight_layout()
    out = RESULTS_VISUALS_DIR / f"sparse_patch_{VIDEO_ID}.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Image → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not SPARSE_MODEL.exists():
        print("=== Entraînement sparse coding ===")
        train()

    print("=== Évaluation ===")
    evaluate()

    print("\n=== Comparaison visuelle ===")
    save_visual()
