"""Test du workflow SAD + FSRCNN sur frames consécutives."""
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from workflow.model    import load_model
from workflow.pipeline import SadFsrcnnPipeline
from config import FRAMES_HR_DIR, FRAMES_LR_PHASE1_DIR, VIDEO_SCALE, RESULTS_VISUALS_DIR
from pipeline.metrics import psnr

CKPT     = Path(__file__).parent.parent.parent / "fsrcnn-main" / "pretrained" / "fsrcnn_div2k.ckpt"
VIDEO_ID = "33mqqm4QlJ8"
N_FRAMES = 10


def main(sad_threshold: float = 1000.0):
    if not CKPT.exists():
        print(f"[run] Checkpoint introuvable : {CKPT}")
        sys.exit(1)

    print(f"[run] Chargement du modèle : {CKPT.name}")
    model    = load_model(CKPT)
    pipeline = SadFsrcnnPipeline(model, sad_threshold=sad_threshold)

    scale   = VIDEO_SCALE[VIDEO_ID]
    lr_root = FRAMES_LR_PHASE1_DIR / VIDEO_ID
    hr_root = FRAMES_HR_DIR / VIDEO_ID
    frames  = sorted(lr_root.glob("*.png"))[:N_FRAMES]

    RESULTS_VISUALS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Frame':<16} {'PSNR bic':>9} {'PSNR SR':>9} {'Δ PSNR':>7} {'Recyclé':>9}")
    print("-" * 56)

    prev_hr_y = None

    for lr_f in frames:
        lr  = cv2.imread(str(lr_f))
        hr  = cv2.imread(str(hr_root / lr_f.name))

        bic = cv2.resize(lr, (lr.shape[1] * scale, lr.shape[0] * scale),
                         interpolation=cv2.INTER_CUBIC)

        sr, hr_y, stats = pipeline.process(lr, prev_hr_y)
        prev_hr_y = hr_y

        p_bic = psnr(hr, bic)
        p_sr  = psnr(hr, sr)
        print(f"{lr_f.name:<16} {p_bic:>9.2f} {p_sr:>9.2f} {p_sr - p_bic:>+7.2f}"
              f" {stats['recycled_pct']:>8.1f}%")

        out = RESULTS_VISUALS_DIR / f"workflow_{VIDEO_ID}_{lr_f.stem}.png"
        _save_visual(hr, bic, sr, p_bic, p_sr, lr_f.name, scale, out)

    print(f"\nVisuels → {RESULTS_VISUALS_DIR}")


def _save_visual(hr, bic, sr, p_bic, p_sr, name, scale, out):
    import matplotlib.pyplot as plt
    H, W = hr.shape[:2]
    py, px, ph, pw = H // 3, W // 3, 200, 200

    def crop(img):
        return cv2.cvtColor(img[py:py+ph, px:px+pw], cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    fig.suptitle(f"SAD+FSRCNN (×{scale}) — {name}", fontsize=12, fontweight="bold")
    for ax, img, title, color in [
        (axes[0], hr,  "HR référence",                        "#2ecc71"),
        (axes[1], bic, f"Bicubique\nPSNR = {p_bic:.2f} dB",  "#95a5a6"),
        (axes[2], sr,  f"SAD+FSRCNN\nPSNR = {p_sr:.2f} dB",  "#3498db"),
    ]:
        ax.imshow(crop(img)); ax.set_title(title, fontsize=10); ax.axis("off")
        for s in ax.spines.values():
            s.set_visible(True); s.set_edgecolor(color); s.set_linewidth(3)
    plt.tight_layout()
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=1000.0,
                        help="Seuil SAD par bloc (défaut 1000)")
    args = parser.parse_args()
    main(args.threshold)
