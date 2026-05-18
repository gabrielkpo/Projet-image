"""Entraînement du Sparse Coding SR sur DIV2K.

Usage :
    python train/train_sparse.py [--n_imgs 50] [--n_atoms 256] [--n_nonzero 10]

Structure DIV2K attendue dans data/div2k/ :
    DIV2K_train_HR/        → images HR originales (0001.png … 0800.png)
    DIV2K_train_LR_bicubic/X4/  → images LR dégradées (0001x4.png …)

Téléchargement : https://data.vision.ee.ethz.ch/cvl/DIV2K/
"""
import sys
import argparse
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DIV2K_DIR, SCALE_FACTOR
from methods.level4_ml_classic import SparseSR

MODEL_PATH = Path(__file__).parent / "sparse_sr.joblib"


def load_div2k_pairs(n_imgs: int, scale: int) -> list:
    """
    Charge n_imgs paires (LR, HR) depuis DIV2K.
    Si les LR pré-calculées n'existent pas, les génère par dégradation bicubique.
    """
    hr_dir = DIV2K_DIR / "DIV2K_train_HR"
    lr_dir = DIV2K_DIR / "DIV2K_train_LR_bicubic" / f"X{scale}"

    if not hr_dir.exists():
        print(f"[train_sparse] DIV2K introuvable : {hr_dir}")
        print("  → Télécharger depuis https://data.vision.ee.ethz.ch/cvl/DIV2K/")
        print(f"  → Placer dans {DIV2K_DIR}/")
        sys.exit(1)

    pairs = []
    for hr_path in sorted(hr_dir.glob("*.png"))[:n_imgs]:
        hr_img = cv2.imread(str(hr_path))
        if hr_img is None:
            continue

        # Cherche le LR pré-calculé, sinon dégrade à la volée
        lr_path = lr_dir / f"{hr_path.stem}x{scale}.png"
        if lr_path.exists():
            lr_img = cv2.imread(str(lr_path))
        else:
            h, w   = hr_img.shape[:2]
            lr_img = cv2.resize(hr_img, (w // scale, h // scale),
                                interpolation=cv2.INTER_CUBIC)

        if lr_img is not None:
            pairs.append((lr_img, hr_img))

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_imgs",     type=int, default=50,  help="Nombre d'images DIV2K")
    parser.add_argument("--n_atoms",    type=int, default=256, help="Taille du dictionnaire")
    parser.add_argument("--n_nonzero",  type=int, default=10,  help="Sparsité (OMP)")
    parser.add_argument("--patch_size", type=int, default=8,   help="Taille des patches LR")
    args = parser.parse_args()

    scale = SCALE_FACTOR
    print(f"[train_sparse] Chargement de {args.n_imgs} images DIV2K (×{scale})…")

    pairs = load_div2k_pairs(args.n_imgs, scale)
    if not pairs:
        print("[train_sparse] Aucune paire chargée. Vérifier le dossier DIV2K.")
        sys.exit(1)

    print(f"[train_sparse] {len(pairs)} paires chargées.")

    lr_imgs = [p[0] for p in pairs]
    hr_imgs = [p[1] for p in pairs]

    model = SparseSR(
        n_atoms=args.n_atoms,
        patch_size=args.patch_size,
        stride_train=4,
        stride_infer=2,
        n_nonzero=args.n_nonzero,
        scale=scale,
    )
    model.fit(lr_imgs, hr_imgs)
    model.save(MODEL_PATH)


if __name__ == "__main__":
    main()
