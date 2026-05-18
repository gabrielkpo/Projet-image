"""Niveau 3 — Machine Learning classique : sparse coding sur patches LR/HR."""
import numpy as np
import cv2
import joblib
from pathlib import Path
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.linear_model import orthogonal_mp


# ══════════════════════════════════════════════════════════════════════════════
# Utilitaires partagés
# ══════════════════════════════════════════════════════════════════════════════

def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _sobel_features(patch: np.ndarray) -> np.ndarray:
    """Gradients Sobel (Gx, Gy) aplatis et normalisés — descripteur d'un patch LR."""
    p  = patch.astype(np.float32)
    gx = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
    feat = np.concatenate([gx.flatten(), gy.flatten()])
    norm = np.linalg.norm(feat)
    return feat / norm if norm > 1e-8 else feat


def _extract_pairs(lr_gray: np.ndarray, hr_gray: np.ndarray,
                   patch_size: int, stride: int, scale: int
                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrait les paires (descripteurs Sobel LR, pixels HR normalisés [0,1]).

    Chaque patch LR de taille patch_size×patch_size est associé au patch HR
    correspondant de taille (patch_size·scale)×(patch_size·scale).
    """
    h, w = lr_gray.shape
    lr_feats, hr_patches = [], []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            lr_patch = lr_gray[y:y + patch_size, x:x + patch_size]
            hr_patch = hr_gray[y * scale:(y + patch_size) * scale,
                                x * scale:(x + patch_size) * scale]
            lr_feats.append(_sobel_features(lr_patch))
            hr_patches.append(hr_patch.flatten().astype(np.float32) / 255.0)

    return np.array(lr_feats, dtype=np.float32), np.array(hr_patches, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Sparse Coding SR — dictionnaires couplés DLR / DHR
# ══════════════════════════════════════════════════════════════════════════════

class SparseSR:
    """
    Super-résolution par sparse coding avec dictionnaires couplés DLR et DHR.

    Hypothèse centrale (Yang et al., 2010) :
        pLR ≈ DLR · α    (descripteurs Sobel du patch LR)
        pHR ≈ DHR · α    (pixels du patch HR, même code parcimonieux α)

    Apprentissage (ODL, Mairal et al. 2010) :
        1. Extraction des descripteurs Sobel LR et des pixels HR
        2. DLR appris par MiniBatchDictionaryLearning (Online Dictionary Learning)
        3. Calcul des codes α via OMP sur DLR
        4. DHR appris par moindres carrés : DHR = X_hr · pinv(A)

    Inférence patch par patch :
        α* = OMP(DLR, features_LR)
        p̂HR = DHR · α*
    """

    def __init__(self, n_atoms: int = 256, patch_size: int = 8,
                 stride_train: int = 4, stride_infer: int = 2,
                 n_nonzero: int = 10, scale: int = 4,
                 max_patches: int = 60_000):
        self.n_atoms      = n_atoms
        self.patch_size   = patch_size
        self.stride_train = stride_train
        self.stride_infer = stride_infer
        self.n_nonzero    = n_nonzero
        self.scale        = scale
        self.max_patches  = max_patches
        self.D_lr = None  # (n_atoms, d_lr)  — convention sklearn (atoms en lignes)
        self.D_hr = None  # (d_hr,   n_atoms)

    # ── Étape 1+2+3+4 : entraînement ─────────────────────────────────────────

    def fit(self, lr_imgs: list, hr_imgs: list) -> "SparseSR":
        """Apprend DLR et DHR à partir de paires d'images (LR, HR)."""

        # Étape 1 — Extraction des patches LR (Sobel) et HR (pixels)
        print(f"[SparseSR] Extraction des patches ({len(lr_imgs)} images)…")
        all_lr, all_hr = [], []
        for lr, hr in zip(lr_imgs, hr_imgs):
            lf, hp = _extract_pairs(
                _to_gray(lr), _to_gray(hr),
                self.patch_size, self.stride_train, self.scale,
            )
            all_lr.append(lf)
            all_hr.append(hp)

        X_lr = np.vstack(all_lr)   # (N, d_lr)
        X_hr = np.vstack(all_hr)   # (N, d_hr)

        # Sous-échantillonnage si le dataset dépasse la limite mémoire
        if len(X_lr) > self.max_patches:
            idx  = np.random.default_rng(42).choice(len(X_lr), self.max_patches, replace=False)
            X_lr = X_lr[idx]
            X_hr = X_hr[idx]

        print(f"[SparseSR] {len(X_lr):,} patches — apprentissage DLR (ODL)…")

        # Étape 2 — Online Dictionary Learning sur les descripteurs LR
        odl = MiniBatchDictionaryLearning(
            n_components=self.n_atoms,
            transform_algorithm="omp",
            transform_n_nonzero_coefs=self.n_nonzero,
            max_iter=500,
            batch_size=256,
            random_state=42,
            verbose=1,
        )
        odl.fit(X_lr)
        self.D_lr = odl.components_   # (n_atoms, d_lr)

        # Étape 3 — Codes parcimonieux α pour tous les patches d'entraînement
        print("[SparseSR] Calcul des codes α via OMP…")
        A = odl.transform(X_lr)       # (N, n_atoms)

        # Étape 4 — DHR par moindres carrés : X_hr ≈ A @ DHR.T
        print("[SparseSR] Apprentissage DHR (moindres carrés)…")
        Z, _, _, _ = np.linalg.lstsq(A, X_hr, rcond=None)  # Z : (n_atoms, d_hr)
        self.D_hr = Z.T                                      # (d_hr, n_atoms)

        print("[SparseSR] Entraînement terminé.")
        return self

    # ── Inférence ─────────────────────────────────────────────────────────────

    def predict(self, lr_img: np.ndarray) -> np.ndarray:
        """Reconstruit l'image SR depuis lr_img par sparse coding patch par patch."""
        assert self.D_lr is not None, "Appeler fit() ou load() avant predict()."

        lr_gray = _to_gray(lr_img).astype(np.float32)
        h, w    = lr_gray.shape
        ph      = self.patch_size
        ph_hr   = ph * self.scale
        H, W    = h * self.scale, w * self.scale

        output  = np.zeros((H, W), dtype=np.float32)
        weights = np.zeros((H, W), dtype=np.float32)

        # Extraction vectorisée de tous les patches LR
        positions, feats = [], []
        for y in range(0, h - ph + 1, self.stride_infer):
            for x in range(0, w - ph + 1, self.stride_infer):
                feats.append(_sobel_features(lr_gray[y:y + ph, x:x + ph]))
                positions.append((y, x))

        X = np.array(feats, dtype=np.float32)   # (N_patches, d_lr)

        # OMP vectorisé : α* = OMP(DLR, features)
        # orthogonal_mp(X=D_lr.T, y=X.T) → (n_atoms, N_patches)
        A = orthogonal_mp(self.D_lr.T, X.T, n_nonzero_coefs=self.n_nonzero)

        # Reconstruction : p̂HR = DHR · α*
        HR_patches = self.D_hr @ A   # (d_hr, N_patches)

        # Réassemblage avec moyenne pondérée sur les zones de chevauchement
        for i, (y, x) in enumerate(positions):
            hr_patch = np.clip(HR_patches[:, i] * 255, 0, 255).reshape(ph_hr, ph_hr)
            y_hr, x_hr = y * self.scale, x * self.scale
            output [y_hr:y_hr + ph_hr, x_hr:x_hr + ph_hr] += hr_patch
            weights[y_hr:y_hr + ph_hr, x_hr:x_hr + ph_hr] += 1.0

        weights = np.maximum(weights, 1)
        out_gray = np.clip(output / weights, 0, 255).astype(np.uint8)

        return cv2.cvtColor(out_gray, cv2.COLOR_GRAY2BGR) if lr_img.ndim == 3 else out_gray

    # ── Sauvegarde / Chargement ───────────────────────────────────────────────

    def save(self, path: Path | str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({k: getattr(self, k) for k in
                     ("D_lr", "D_hr", "n_atoms", "patch_size",
                      "n_nonzero", "scale", "stride_train", "stride_infer")}, path)
        print(f"[SparseSR] Modèle sauvegardé → {path}")

    def load(self, path: Path | str) -> "SparseSR":
        for k, v in joblib.load(path).items():
            setattr(self, k, v)
        return self


def sparse_sr(lr_img: np.ndarray, scale: int,
              model_path: Path | str | None = None) -> np.ndarray:
    """Interface fonctionnelle : charge le modèle pré-entraîné et applique la SR."""
    if model_path is None or not Path(model_path).exists():
        raise FileNotFoundError(
            f"Modèle sparse introuvable : {model_path}\n"
            "Lancer d'abord : python train/train_sparse.py"
        )
    return SparseSR(scale=scale).load(model_path).predict(lr_img)
