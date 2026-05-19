"""Test sparse coding SR — évaluation sur nos frames vidéo. Non versionné.

Pré-requis : entraîner le modèle avant de lancer ce script.
    python train/train_sparse.py --source frames --n_atoms 512 --n_nonzero 5
"""
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, ".")
from config import FRAMES_HR_DIR, FRAMES_LR_PHASE1_DIR, VIDEO_SCALE, RESULTS_VISUALS_DIR
from pipeline.metrics import psnr
import methods.level1_interpolation as l1
import methods.level4_ml_classic as l4

VIDEO_ID     = "33mqqm4QlJ8"
N_TEST       = 10
SEED         = 90          # changer pour tirer un autre échantillon
SPARSE_MODEL = Path(__file__).parent / "train" / "sparse_sr.joblib"

RESULTS_VISUALS_DIR.mkdir(parents=True, exist_ok=True)

if not SPARSE_MODEL.exists():
    print(f"[test] Modèle absent : {SPARSE_MODEL}")
    print("[test] Lancer d'abord : python train/train_sparse.py")
    sys.exit(1)

scale   = VIDEO_SCALE[VIDEO_ID]
model   = l4.SparseSR(scale=scale).load(SPARSE_MODEL)
hr_root = FRAMES_HR_DIR / VIDEO_ID

all_frames = sorted((FRAMES_LR_PHASE1_DIR / VIDEO_ID).glob("*.png"))
rng        = np.random.default_rng(SEED)
lr_files   = sorted(rng.choice(all_frames, size=min(N_TEST, len(all_frames)), replace=False))

print(f"[test] Modèle chargé — {VIDEO_ID} ×{scale}, {len(lr_files)} frames (seed={SEED})")


def evaluate() -> list:
    print(f"\n{'Frame':<16} {'Bicubic PSNR':>13} {'Sparse PSNR':>12} {'Δ PSNR':>8}")
    print("-" * 52)
    results = []
    for lr_f in lr_files:
        lr  = cv2.imread(str(lr_f))
        hr  = cv2.imread(str(hr_root / lr_f.name))
        bic = l1.upscale(lr, scale, "bicubic")
        spa = model.predict(lr)
        p_bic, p_spa = psnr(hr, bic), psnr(hr, spa)
        print(f"{lr_f.name:<16} {p_bic:>13.2f} {p_spa:>12.2f} {p_spa - p_bic:>+8.2f}")
        results.append((lr_f, hr, bic, spa, p_bic, p_spa))
    deltas = [r[5] - r[4] for r in results]
    print(f"{'Moyenne':<16} {'':>13} {'':>12} {np.mean(deltas):>+8.2f}")
    return results


def save_visuals(results: list):
    print("\n=== Comparaison visuelle ===")
    for lr_f, hr, bic, spa, p_bic, p_spa in results:
        H, W = hr.shape[:2]
        py, px, ph, pw = H // 3, W // 3, 200, 200

        def crop(img):
            return cv2.cvtColor(img[py:py+ph, px:px+pw], cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
        fig.suptitle(f"Sparse SR (×{scale}) — {VIDEO_ID} / {lr_f.name}",
                     fontsize=12, fontweight="bold")
        for ax, img, title, color in [
            (axes[0], hr,  "HR référence",                       "#2ecc71"),
            (axes[1], bic, f"Bicubique\nPSNR = {p_bic:.2f} dB", "#95a5a6"),
            (axes[2], spa, f"Sparse SR\nPSNR = {p_spa:.2f} dB", "#3498db"),
        ]:
            ax.imshow(crop(img))
            ax.set_title(title, fontsize=10)
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_visible(True); spine.set_edgecolor(color); spine.set_linewidth(3)
        plt.tight_layout()
        out = RESULTS_VISUALS_DIR / f"sparse_{VIDEO_ID}_{lr_f.stem}.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out}")


if __name__ == "__main__":
    results = evaluate()
    #save_visuals(results)
