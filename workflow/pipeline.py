"""Pipeline SAD + FSRCNN : batching de blocs 16×16 actifs."""
from __future__ import annotations

import numpy as np
import cv2
import torch

from .sad import compute_sad_mask
from .model import FSRCNN

BLOCK_SIZE = 16
SCALE      = 4
BLOCK_HR   = BLOCK_SIZE * SCALE   # 64


class SadFsrcnnPipeline:
    def __init__(
        self,
        model: FSRCNN,
        block_size: int = BLOCK_SIZE,
        sad_threshold: float = 1000.0,
        device: str = "cpu",
    ) -> None:
        self.model      = model
        self.block_size = block_size
        self.threshold  = sad_threshold
        self.device     = device
        self.scale      = model.scale

    def process(
        self,
        lr_bgr: np.ndarray,
        prev_hr_y: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        lr_bgr      : (H, W, 3) uint8
        prev_hr_y   : (H*scale, W*scale) uint8 ou None (première frame)
        Retourne    : (hr_bgr uint8, stats dict)
        """
        bs = self.block_size
        sc = self.scale
        bhr = bs * sc

        # ── 1. YCrCb split ──────────────────────────────────────────────────
        ycrcb  = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2YCrCb)
        lr_y, lr_cr, lr_cb = cv2.split(ycrcb)
        H, W = lr_y.shape
        H_hr, W_hr = H * sc, W * sc

        n_bh = H // bs
        n_bw = W // bs

        # ── 2. SAD mask (première frame → tout actif) ────────────────────────
        if prev_hr_y is None:
            mask = np.ones((n_bh, n_bw), dtype=bool)
        else:
            # Comparaison sur Y LR courant vs Y LR précédent (downscale du buffer)
            prev_lr_y = cv2.resize(prev_hr_y, (W, H), interpolation=cv2.INTER_AREA)
            mask = compute_sad_mask(lr_y, prev_lr_y, bs, self.threshold)

        n_active  = int(mask.sum())
        n_total   = n_bh * n_bw
        n_static  = n_total - n_active

        # ── 3. Extraction des blocs actifs ───────────────────────────────────
        positions   = [(by, bx) for by in range(n_bh) for bx in range(n_bw)]
        active_pos  = [(by, bx) for by, bx in positions if mask[by, bx]]

        # ── 4. Mosaïque Y HR ─────────────────────────────────────────────────
        hr_y = np.zeros((n_bh * bhr, n_bw * bhr), dtype=np.float32)

        # Blocs statiques : recopie depuis le buffer précédent
        if prev_hr_y is not None:
            for by, bx in positions:
                if not mask[by, bx]:
                    y_hr, x_hr = by * bhr, bx * bhr
                    hr_y[y_hr:y_hr + bhr, x_hr:x_hr + bhr] = \
                        prev_hr_y[y_hr:y_hr + bhr, x_hr:x_hr + bhr].astype(np.float32)

        # Blocs actifs : batch FSRCNN
        if n_active > 0:
            patches = np.stack([
                lr_y[by * bs:(by + 1) * bs, bx * bs:(bx + 1) * bs].astype(np.float32) / 255.0
                for by, bx in active_pos
            ])                                          # (N, bs, bs)
            tensor = torch.from_numpy(patches).unsqueeze(1).to(self.device)  # (N, 1, bs, bs)

            with torch.no_grad():
                out = self.model(tensor)                # (N, 1, bhr, bhr)

            out_np = out.squeeze(1).cpu().numpy() * 255.0  # (N, bhr, bhr)

            for i, (by, bx) in enumerate(active_pos):
                y_hr, x_hr = by * bhr, bx * bhr
                hr_y[y_hr:y_hr + bhr, x_hr:x_hr + bhr] = out_np[i]

        # ── 5. Lignes résiduelles (si H non divisible par bs) ────────────────
        hr_y_full = cv2.resize(lr_y.astype(np.float32), (W_hr, H_hr),
                               interpolation=cv2.INTER_CUBIC)
        grid_h = n_bh * bhr
        if grid_h < H_hr:
            hr_y_full[:grid_h, :n_bw * bhr] = hr_y
            hr_y = hr_y_full
        else:
            hr_y = hr_y

        hr_y_u8 = np.clip(hr_y, 0, 255).astype(np.uint8)

        # ── 6. Bicubic Cr/Cb ─────────────────────────────────────────────────
        hr_cr = cv2.resize(lr_cr, (W_hr, H_hr), interpolation=cv2.INTER_CUBIC)
        hr_cb = cv2.resize(lr_cb, (W_hr, H_hr), interpolation=cv2.INTER_CUBIC)

        # ── 7. Fusion BGR ────────────────────────────────────────────────────
        hr_bgr = cv2.cvtColor(cv2.merge([hr_y_u8, hr_cr, hr_cb]), cv2.COLOR_YCrCb2BGR)

        stats = {
            "n_active":      n_active,
            "n_static":      n_static,
            "n_total":       n_total,
            "recycled_pct":  100.0 * n_static / n_total,
        }
        return hr_bgr, hr_y_u8, stats
