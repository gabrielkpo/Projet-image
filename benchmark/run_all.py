"""Lance tous les algorithmes SR sur le même jeu de frames et exporte les résultats."""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FRAMES_HR_DIR, FRAMES_LR_DIR, SCALE_FACTOR, RESULTS_DIR
from pipeline.metrics import evaluate, energy_tracker
import methods.level1_interpolation as l1
import methods.level2_frequency as l2
import methods.level3_denoising as l3
import methods.level4_ml_classic as l4

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_FRAMES = 20  # limiter pour un benchmark rapide


def collect_frame_pairs() -> list[tuple[Path, Path]]:
    pairs = []
    for lr_d in sorted(FRAMES_LR_DIR.iterdir()):
        if not lr_d.is_dir():
            continue
        hr_d = FRAMES_HR_DIR / lr_d.name
        for lr_f in sorted(lr_d.glob("*.png"))[:MAX_FRAMES]:
            hr_f = hr_d / lr_f.name
            if hr_f.exists():
                pairs.append((lr_f, hr_f))
    return pairs


def benchmark_method(name: str, fn, pairs: list, scale: int) -> dict:
    psnr_vals, ssim_vals, times = [], [], []
    for lr_path, hr_path in pairs:
        lr = cv2.imread(str(lr_path))
        hr = cv2.imread(str(hr_path))
        t0 = time.perf_counter()
        sr = fn(lr, scale)
        times.append(time.perf_counter() - t0)
        m = evaluate(hr, sr)
        psnr_vals.append(m["psnr"])
        ssim_vals.append(m["ssim"])
    return {
        "method": name,
        "psnr_mean": float(np.mean(psnr_vals)),
        "ssim_mean": float(np.mean(ssim_vals)),
        "time_mean_s": float(np.mean(times)),
    }


def main():
    pairs = collect_frame_pairs()
    if not pairs:
        print("Aucune paire LR/HR trouvée. Lancez d'abord extract_frames.py et degrade.py.")
        return

    print(f"{len(pairs)} paires trouvées — facteur x{SCALE_FACTOR}")

    methods = {
        "bicubic":   lambda img, s: l1.upscale(img, s, "bicubic"),
        "bilinear":  lambda img, s: l1.upscale(img, s, "bilinear"),
        "lanczos":   lambda img, s: l1.upscale(img, s, "lanczos"),
        "fft_zeropad": lambda img, s: l2.fft_zeropad(img, s),
        "dwt_upscale": lambda img, s: l2.dwt_upscale(img, s),
        "bilateral_sr": lambda img, s: l3.bilateral_sr(img, s),
        "nlm_sr":       lambda img, s: l3.nlm_sr(img, s),
        "pca_sr":       lambda img, s: l4.pca_sr(img, s),
    }

    rows = []
    for name, fn in methods.items():
        print(f"  Benchmarking {name}…", end=" ", flush=True)
        with energy_tracker(name) as get_energy:
            row = benchmark_method(name, fn, pairs, SCALE_FACTOR)
        emissions, energy = get_energy()
        row["emissions_kg"] = emissions
        row["energy_kwh"] = energy
        rows.append(row)
        print(f"PSNR={row['psnr_mean']:.2f} dB  SSIM={row['ssim_mean']:.4f}")

    df = pd.DataFrame(rows).sort_values("psnr_mean", ascending=False)
    csv_path = RESULTS_DIR / "benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nRésultats sauvegardés : {csv_path}")

    _plot(df)


def _plot(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    df_sorted = df.sort_values("psnr_mean", ascending=True)

    axes[0].barh(df_sorted["method"], df_sorted["psnr_mean"])
    axes[0].set_xlabel("PSNR moyen (dB)")
    axes[0].set_title("PSNR par méthode")

    axes[1].barh(df_sorted["method"], df_sorted["ssim_mean"])
    axes[1].set_xlabel("SSIM moyen")
    axes[1].set_title("SSIM par méthode")

    plt.tight_layout()
    fig_path = RESULTS_DIR / "benchmark.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Figure sauvegardée : {fig_path}")


if __name__ == "__main__":
    main()
