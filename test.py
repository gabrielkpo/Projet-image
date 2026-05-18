"""Test sparse coding SR — évaluation sur nos frames vidéo. Non versionné.

Pré-requis : entraîner le modèle sur DIV2K avant de lancer ce script.
    python train/train_sparse.py
"""
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
N_TEST       = 5
SPARSE_MODEL = Path(__file__).parent / "train" / "sparse_sr.joblib"

scale    = VIDEO_SCALE[VIDEO_ID]
lr_root  = FRAMES_LR_PHASE1_DIR / VIDEO_ID
hr_root  = FRAMES_HR_DIR / VIDEO_ID
lr_files = sorted(lr_root.glob("*.png"))

RESULTS_VISUALS_DIR.mkdir(parents=True, exist_ok=True)

# ── Vérification modèle ───────────────────────────────────────────────────────
if not SPARSE_MODEL.exists():
    print(f"[test] Modèle absent : {SPARSE_MODEL}")
    print("[test] Lancer d'abord : python train/train_sparse.py")
    sys.exit(1)

model = l4.SparseSR(scale=scale).load(SPARSE_MODEL)
print(f"[test] Modèle chargé : {SPARSE_MODEL}")


# ── 1. Évaluation PSNR / SSIM ────────────────────────────────────────────────
def evaluate():
    print(f"\n{'Frame':<16} {'Bicubic PSNR':>13} {'Sparse PSNR':>12} {'Δ PSNR':>8}")
    print("-" * 52)
    for lr_f in lr_files[:N_TEST]:
        lr = cv2.imread(str(lr_f))
        hr = cv2.imread(str(hr_root / lr_f.name))

        bic = l1.upscale(lr, scale, "bicubic")
        spa = model.predict(lr)

        p_bic = psnr(hr, bic)
        p_spa = psnr(hr, spa)
        print(f"{lr_f.name:<16} {p_bic:>13.2f} {p_spa:>12.2f} {p_spa - p_bic:>+8.2f}")


# ── 2. Comparaison visuelle sur un patch ─────────────────────────────────────
def save_visual(frame_idx: int = 0):
    lr_f = lr_files[frame_idx]
    lr   = cv2.imread(str(lr_f))
    hr   = cv2.imread(str(hr_root / lr_f.name))

    bic = l1.upscale(lr, scale, "bicubic")
    spa = model.predict(lr)

    H, W  = hr.shape[:2]
    py, px, ph, pw = H // 3, W // 3, 200, 200

    def crop(img):
        return cv2.cvtColor(img[py:py+ph, px:px+pw], cv2.COLOR_BGR2RGB)

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
    print(f"\nImage → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Évaluation sur frames vidéo ===")
    evaluate()

    print("\n=== Comparaison visuelle ===")
    save_visual()
