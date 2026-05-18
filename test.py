"""Script de test local — modifier selon le niveau en cours. Non versionné."""
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import subprocess
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
from config import RUNS_DIR, VIDEO_SCALE, RESULTS_VISUALS_DIR

# ── Paramètres ────────────────────────────────────────────────────────────────
VIDEO_ID = "33mqqm4QlJ8"   # ou "DGbwtVtthu8"
N_FRAMES  = 5              # frames pour le benchmark rapide
# Patch zoomé : coin haut-gauche dans l'espace HR
PATCH_X, PATCH_Y, PATCH_W, PATCH_H = 800, 300, 280, 280

RESULTS_VISUALS_DIR.mkdir(parents=True, exist_ok=True)

scale  = VIDEO_SCALE[VIDEO_ID]
hr_dir = RUNS_DIR / VIDEO_ID / "frames_hr"
lr_dir = RUNS_DIR / VIDEO_ID / "frames_lr"
frames = sorted(lr_dir.glob("*.png"))[:N_FRAMES]

# ── Méthodes à tester ─────────────────────────────────────────────────────────
import methods.level2_frequency as l2
import methods.level1_interpolation as l1

METHODS = {
    "bicubic (réf)": lambda img, s: l1.upscale(img, s, "bicubic"),
    "fft_zeropad":   lambda img, s: l2.fft_zeropad(img, s),
    "dwt_haar":      lambda img, s: l2.dwt_upscale(img, s, wavelet="haar"),
    "dwt_db2":       lambda img, s: l2.dwt_upscale(img, s, wavelet="db2"),
}

# ── Imports métriques ─────────────────────────────────────────────────────────
from pipeline.metrics import psnr, ssim


# ── 1. Benchmark rapide ───────────────────────────────────────────────────────
def run_benchmark():
    scores = {name: {"psnr": [], "ssim": []} for name in METHODS}
    for lr_path in frames:
        hr = cv2.imread(str(hr_dir / lr_path.name))
        lr = cv2.imread(str(lr_path))
        for name, fn in METHODS.items():
            sr = fn(lr, scale)
            scores[name]["psnr"].append(psnr(hr, sr))
            scores[name]["ssim"].append(ssim(hr, sr))

    print(f"\n{'Méthode':<20} {'PSNR moy (dB)':>14} {'SSIM moy':>10}")
    print("-" * 46)
    for name, v in scores.items():
        print(f"{name:<20} {np.mean(v['psnr']):>14.2f} {np.mean(v['ssim']):>10.4f}")


# ── 2. Grille de patches ──────────────────────────────────────────────────────
def save_patch_grid(frame_idx: int = 0):
    lr_path = frames[frame_idx]
    lr = cv2.imread(str(lr_path))
    hr = cv2.imread(str(hr_dir / lr_path.name))

    n   = 1 + len(METHODS)
    fig = plt.figure(figsize=(4 * n, 8))
    gs  = gridspec.GridSpec(2, n, hspace=0.05, wspace=0.05)

    def _show(ax, img_bgr, title):
        ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=8, pad=3)
        ax.axis("off")

    def _patch(img):
        return img[PATCH_Y: PATCH_Y + PATCH_H, PATCH_X: PATCH_X + PATCH_W]

    _show(fig.add_subplot(gs[0, 0]), hr,        "HR (référence)")
    _show(fig.add_subplot(gs[1, 0]), _patch(hr), "patch HR")

    for col, (name, fn) in enumerate(METHODS.items(), 1):
        sr = fn(lr, scale)
        p, s = psnr(hr, sr), ssim(hr, sr)
        _show(fig.add_subplot(gs[0, col]), sr,         f"{name}\nPSNR {p:.2f} | SSIM {s:.4f}")
        _show(fig.add_subplot(gs[1, col]), _patch(sr),  f"patch {name}")

    out = RESULTS_VISUALS_DIR / f"patch_grid_{VIDEO_ID}_frame{frame_idx:02d}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Image → {out}")


# ── 3. Vidéo de comparaison côte à côte ──────────────────────────────────────
def save_comparison_video(seconds_per_frame: int = 2):
    TILE_W, TILE_H = 480, 270
    LABEL_H = 36
    FPS     = 24
    n_cols  = 1 + len(METHODS)
    out_w   = TILE_W * n_cols
    out_h   = TILE_H + LABEL_H
    labels  = ["HR (réf)"] + list(METHODS.keys())
    out_path = RESULTS_VISUALS_DIR / f"comparison_{VIDEO_ID}.mp4"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        idx = 0
        for lr_path in frames:
            lr    = cv2.imread(str(lr_path))
            hr    = cv2.imread(str(hr_dir / lr_path.name))
            tiles = [hr] + [fn(lr, scale) for fn in METHODS.values()]
            row   = np.hstack([cv2.resize(t, (TILE_W, TILE_H)) for t in tiles])
            bar   = np.full((LABEL_H, out_w, 3), 30, dtype=np.uint8)
            for i, lbl in enumerate(labels):
                cv2.putText(bar, lbl, (i * TILE_W + 8, LABEL_H - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            composed = np.vstack([bar, row])
            for _ in range(FPS * seconds_per_frame):
                cv2.imwrite(str(tmp / f"f{idx:05d}.png"), composed)
                idx += 1

        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(FPS),
            "-i", str(tmp / "f%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            str(out_path),
        ], check=True, capture_output=True)

    print(f"Vidéo → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Benchmark rapide ===")
    run_benchmark()

    print("\n=== Grille de patches (frame 0) ===")
    save_patch_grid(frame_idx=0)

    print("\n=== Vidéo de comparaison ===")
    save_comparison_video(seconds_per_frame=2)
