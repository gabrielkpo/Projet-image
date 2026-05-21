"""Niveau 6 — FSRCNN (réseau efficace pour SR temps-réel)."""
import sys
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SCALE_FACTOR, DEVICE


class FSRCNN(nn.Module):
    """Dong et al., 2016 — version rapide de SRCNN."""
    def __init__(self, scale: int = SCALE_FACTOR, d: int = 56, s: int = 12, m: int = 4):
        super().__init__()
        self.scale = scale
        self.feature = nn.Sequential(
            nn.Conv2d(1, d, kernel_size=5, padding=2),
            nn.PReLU(num_parameters=d),
        )
        self.shrink = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(num_parameters=s),
        )
        self.map = nn.Sequential(*[
            layer for _ in range(m)
            for layer in (nn.Conv2d(s, s, kernel_size=3, padding=1), nn.PReLU(num_parameters=s))
        ])
        self.expand = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(num_parameters=d),
        )
        self.deconv = nn.ConvTranspose2d(d, 1, kernel_size=9, stride=scale,
                                         padding=4, output_padding=scale - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature(x)
        x = self.shrink(x)
        x = self.map(x)
        x = self.expand(x)
        return self.deconv(x)


def _to_tensor(channel: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(channel.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)


def _from_tensor(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.squeeze().detach().cpu().numpy()
    return np.clip(arr * 255, 0, 255).astype(np.uint8)


def infer(img: np.ndarray, model: nn.Module, device: str = DEVICE) -> np.ndarray:
    """Inférence couleur : FSRCNN sur Y, bicubique sur Cb/Cr, fusion YCbCr."""
    model.eval().to(device)
    scale = model.scale
    h, w = img.shape[:2]

    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycbcr)

    inp = _to_tensor(y).to(device)
    with torch.no_grad():
        y_sr = _from_tensor(model(inp))

    cr_up = cv2.resize(cr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    cb_up = cv2.resize(cb, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    sr_ycbcr = cv2.merge([y_sr, cr_up, cb_up])
    return cv2.cvtColor(sr_ycbcr, cv2.COLOR_YCrCb2BGR)


def load_fsrcnn(weights_path: str, scale: int = SCALE_FACTOR, device: str = DEVICE) -> FSRCNN:
    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    # Détection automatique des hyperparamètres depuis les poids
    d = state_dict["feature.0.weight"].shape[0]
    s = state_dict["shrink.0.weight"].shape[0]
    m = sum(1 for k in state_dict if k.startswith("map.") and k.endswith(".bias"))
    model = FSRCNN(scale, d=d, s=s, m=m)
    model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    import argparse
    import time
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import (
        FRAMES_LR_PHASE1_DIR, FRAMES_HR_DIR, RUNS_DIR,
        VIDEO_SCALE, SCALE_FACTOR, RESULTS_VISUALS_DIR,
    )
    from pipeline.metrics import evaluate
    import methods.level1_interpolation as l1

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(Path(__file__).parent.parent / "workflow" / "fsrcnn_div2k.ckpt"))
    parser.add_argument("--video_id", default=None)
    parser.add_argument("--n_frames", type=int, default=10,
                        help="Nombre de frames a tester (defaut : 10)")
    parser.add_argument("--frame", type=int, default=None,
                        help="Numero de frame specifique (ex: 10 -> frame_00010.png)")
    parser.add_argument("--source", choices=["data", "runs"], default="data",
                        help="Source des frames : 'data' (1fps benchmark) ou 'runs' (FPS natif)")
    args = parser.parse_args()

    # Détection automatique du video_id
    if args.video_id:
        video_id = args.video_id
    else:
        dirs = [d for d in FRAMES_LR_PHASE1_DIR.iterdir() if d.is_dir()]
        if not dirs:
            print("Aucune donnee LR trouvee. Lancez d'abord extract_frames.py et degrade.py.")
            sys.exit(1)
        video_id = dirs[0].name

    scale = VIDEO_SCALE.get(video_id, SCALE_FACTOR)

    if args.source == "runs":
        lr_dir = RUNS_DIR / video_id / "frames_lr"
        hr_dir = RUNS_DIR / video_id / "frames_hr"
    else:
        lr_dir = FRAMES_LR_PHASE1_DIR / video_id
        hr_dir = FRAMES_HR_DIR / video_id

    import random
    all_frames = sorted(lr_dir.glob("frame_*.png"))
    if args.frame is not None:
        lr_frames = [f for f in all_frames if f.name == f"frame_{args.frame:05d}.png"]
    else:
        lr_frames = random.sample(all_frames, min(args.n_frames, len(all_frames)))
        lr_frames = sorted(lr_frames)
    if not lr_frames:
        print("Aucune frame trouvee.")
        sys.exit(1)

    print(f"Chargement des poids : {args.weights}")
    model = load_fsrcnn(args.weights, scale=scale)
    print(f"Test sur les {len(lr_frames)} dernieres frames ({video_id}, x{scale})\n")
    h = f"{'Frame':<20} {'PSNR Bic':>9} {'SSIM Bic':>9} {'T Bic':>7} {'PSNR FSRCNN':>12} {'SSIM FSRCNN':>12} {'T FSRCNN':>9}"
    print(h)
    print("-" * len(h))

    results = []
    for lr_path in lr_frames:
        hr_path = hr_dir / lr_path.name
        lr = cv2.imread(str(lr_path))
        hr = cv2.imread(str(hr_path))
        if lr is None or hr is None:
            continue

        t0 = time.perf_counter()
        bic = l1.upscale(lr, scale, "bicubic")
        t_bic = (time.perf_counter() - t0) * 1000
        m_bic = evaluate(hr, bic)

        t0 = time.perf_counter()
        sr = infer(lr, model)
        t_sr = (time.perf_counter() - t0) * 1000
        m_sr = evaluate(hr, sr)

        results.append((lr_path.name, lr, bic, sr, hr,
                        m_bic["psnr"], m_bic["ssim"], t_bic,
                        m_sr["psnr"],  m_sr["ssim"],  t_sr))
        print(f"{lr_path.name:<20} "
              f"{m_bic['psnr']:>9.2f} {m_bic['ssim']:>9.4f} {t_bic:>7.1f} "
              f"{m_sr['psnr']:>12.2f} {m_sr['ssim']:>12.4f} {t_sr:>9.1f}")

    print("-" * len(h))
    print(f"{'Moyenne':<20} "
          f"{sum(r[5] for r in results)/len(results):>9.2f} "
          f"{sum(r[6] for r in results)/len(results):>9.4f} "
          f"{sum(r[7] for r in results)/len(results):>7.1f} "
          f"{sum(r[8] for r in results)/len(results):>12.2f} "
          f"{sum(r[9] for r in results)/len(results):>12.4f} "
          f"{sum(r[10] for r in results)/len(results):>9.1f}")

    # Figure : LR / Bicubique / FSRCNN / HR pour chaque frame
    def to_rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    n = len(results)
    fig, axes = plt.subplots(n, 4, figsize=(20, 4 * n))
    if n == 1:
        axes = [axes]
    for row, (name, lr, bic, sr, hr, psnr_bic, ssim_bic, t_bic, psnr_sr, ssim_sr, t_sr) in zip(axes, results):
        panels = [
            ("LR", lr),
            (f"Bicubique\nPSNR {psnr_bic:.2f} dB | SSIM {ssim_bic:.4f}\nT {t_bic:.1f} ms", bic),
            (f"FSRCNN\nPSNR {psnr_sr:.2f} dB | SSIM {ssim_sr:.4f}\nT {t_sr:.1f} ms", sr),
            ("HR ref", hr),
        ]
        for ax, (title, img) in zip(row, panels):
            ax.imshow(to_rgb(img))
            ax.set_title(f"{title}\n{name}", fontsize=8)
            ax.axis("off")

    psnr_moy_bic = sum(r[5] for r in results) / len(results)
    psnr_moy_sr  = sum(r[8] for r in results) / len(results)
    fig.suptitle(
        f"FSRCNN x{scale} | {video_id} | "
        f"PSNR moyen : Bicubique {psnr_moy_bic:.2f} dB  vs  FSRCNN {psnr_moy_sr:.2f} dB",
        fontsize=12
    )
    plt.tight_layout()

    RESULTS_VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_VISUALS_DIR / f"fsrcnn_{video_id}.png"
    plt.savefig(out, dpi=150)
    print(f"\nFigure -> {out}")
    plt.show()
