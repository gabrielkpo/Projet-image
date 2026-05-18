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


def _lr_features(patch: np.ndarray) -> np.ndarray:
    """
    Features brutes du patch LR : pixels normalisés, moyenne soustraite.

    Plus robuste que Sobel quand les données d'entraînement sont peu diversifiées
    (quelques frames d'une seule vidéo). Avec DIV2K on peut revenir aux Sobel.
    """
    flat = patch.astype(np.float32).flatten() / 255.0
    return flat - flat.mean()


def _extract_pairs(lr_gray: np.ndarray, hr_gray: np.ndarray,
                   patch_size: int, stride: int, scale: int
                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrait les paires (features LR, résidu HR).

    Résidu HR = HR_patch - bicubique(LR_patch) : haute fréquence à apprendre.
    Garantie : patch homogène → features ≈ 0 → α ≈ 0 → sortie = bicubique.
    """
    h, w = lr_gray.shape
    lr_feats, hr_residuals = [], []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            lr_patch = lr_gray[y:y + patch_size, x:x + patch_size]
            hr_patch = hr_gray[y * scale:(y + patch_size) * scale,
                                x * scale:(x + patch_size) * scale]

            # Patch HR 8×8 aligné sur l'origine du patch LR (ratio 1:1)
            hr_patch = hr_gray[y * scale : y * scale + patch_size,
                                x * scale : x * scale + patch_size]

            # Baseline bicubique à l'origine pour calculer le résidu
            lr_up      = cv2.resize(lr_patch.astype(np.float32),
                                    (patch_size * scale, patch_size * scale),
                                    interpolation=cv2.INTER_CUBIC)
            bic_center = lr_up[0:patch_size, 0:patch_size]
            residual   = (hr_patch.astype(np.float32) - bic_center) / 128.0

            lr_feats.append(_lr_features(lr_patch))
            hr_residuals.append(residual.flatten())

    return np.array(lr_feats, dtype=np.float32), np.array(hr_residuals, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Sparse Coding SR — dictionnaires couplés DLR / DHR
# ══════════════════════════════════════════════════════════════════════════════

class SparseSR:
    """
    Super-résolution par sparse coding avec dictionnaires couplés DLR et DHR.

    Formulation résidu (Yang et al., 2010) :
        DLR apprend les descripteurs Sobel des patches LR
        DHR apprend les résidus haute-fréquence  HR - bicubique(LR)

    Garantie : si le patch est lisse (Sobel ≈ 0), α ≈ 0 et la sortie
    est égale au bicubique — degradation gracieuse assurée.

    Apprentissage :
        1. Extraction des descripteurs Sobel LR et des résidus HR
        2. DLR appris par MiniBatchDictionaryLearning (ODL)
        3. Codes α calculés par OMP sur DLR
        4. DHR appris par moindres carrés : DHR = X_residus · pinv(A)

    Inférence :
        α* = OMP(DLR, sobel(LR_patch))
        SR = bicubique(LR) + DHR · α*  (résidu ajouté à la baseline)
    """

    def __init__(self, n_atoms: int = 256, patch_size: int = 8,
                 stride_train: int = 4, stride_infer: int = 2,
                 n_nonzero: int = 3, scale: int = 4,
                 max_patches: int = 60_000):
        self.n_atoms      = n_atoms
        self.patch_size   = patch_size
        self.stride_train = stride_train
        self.stride_infer = stride_infer
        self.n_nonzero    = n_nonzero
        self.scale        = scale
        self.max_patches  = max_patches
        self.D_lr = None  # (n_atoms, d_lr)  — convention sklearn
        self.D_hr = None  # (d_hr,   n_atoms)

    # ── Entraînement ─────────────────────────────────────────────────────────

    def fit(self, lr_imgs: list, hr_imgs: list) -> "SparseSR":
        """Apprend DLR et DHR à partir de paires d'images (LR, HR)."""

        # Étape 1 — Extraction des descripteurs LR et résidus HR
        # Arrêt anticipé dès que max_patches est atteint pour éviter l'OOM
        print(f"[SparseSR] Extraction des patches ({len(lr_imgs)} images)…")
        all_lr, all_hr = [], []
        total = 0
        for lr, hr in zip(lr_imgs, hr_imgs):
            lf, hr_res = _extract_pairs(
                _to_gray(lr), _to_gray(hr),
                self.patch_size, self.stride_train, self.scale,
            )
            all_lr.append(lf)
            all_hr.append(hr_res)
            total += len(lf)
            if total >= self.max_patches:
                break

        X_lr  = np.vstack(all_lr)[:self.max_patches]
        X_res = np.vstack(all_hr)[:self.max_patches]

        print(f"[SparseSR] {len(X_lr):,} patches — apprentissage DLR (ODL)…")

        # Étape 2 — ODL sur les descripteurs Sobel LR
        odl = MiniBatchDictionaryLearning(
            n_components=self.n_atoms,
            alpha=1.0,
            transform_algorithm="omp",
            transform_n_nonzero_coefs=self.n_nonzero,
            max_iter=1000,
            batch_size=64,
            random_state=42,
            verbose=0,
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            odl.fit(X_lr)
        self.D_lr = odl.components_   # (n_atoms, d_lr)

        # Étape 3 — Codes α parcimonieux pour tous les patches d'entraînement
        print("[SparseSR] Calcul des codes α via OMP…")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A = odl.transform(X_lr)   # (N, n_atoms)

        # Étape 4 — DHR par moindres carrés : résidus ≈ A @ DHR.T
        print("[SparseSR] Apprentissage DHR (moindres carrés)…")
        Z, _, _, _ = np.linalg.lstsq(A, X_res, rcond=None)
        self.D_hr  = Z.T              # (d_hr, n_atoms)

        print("[SparseSR] Entraînement terminé.")
        return self

    # ── Inférence ─────────────────────────────────────────────────────────────

    def predict(self, lr_img: np.ndarray) -> np.ndarray:
        """Reconstruit l'image SR = bicubique(LR) + résidu sparse."""
        assert self.D_lr is not None, "Appeler fit() ou load() avant predict()."

        lr_gray = _to_gray(lr_img).astype(np.float32)
        h, w    = lr_gray.shape
        ph      = self.patch_size
        H, W    = h * self.scale, w * self.scale

        # Baseline bicubique — sortie sans résidu appris
        bicubic = cv2.resize(lr_gray, (W, H), interpolation=cv2.INTER_CUBIC)

        residual = np.zeros((H, W), dtype=np.float32)
        weights  = np.zeros((H, W), dtype=np.float32)

        # Extraction vectorisée des features LR
        positions, feats = [], []
        for y in range(0, h - ph + 1, self.stride_infer):
            for x in range(0, w - ph + 1, self.stride_infer):
                feats.append(_lr_features(lr_gray[y:y + ph, x:x + ph]))
                positions.append((y, x))

        X = np.array(feats, dtype=np.float32)   # (N_patches, d_lr)

        # OMP vectorisé : α* = OMP(DLR, X)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A = orthogonal_mp(self.D_lr.T, X.T, n_nonzero_coefs=self.n_nonzero)

        # Résidus prédits 8×8 (même taille que le patch LR)
        RES_patches = np.clip(self.D_hr @ A, -0.25, 0.25)   # (ph², N_patches)

        # Réassemblage : résidu placé à l'origine du patch HR (couverture complète)
        for i, (y, x) in enumerate(positions):
            res_patch = RES_patches[:, i].reshape(ph, ph)
            y_hr, x_hr = y * self.scale, x * self.scale
            if y_hr + ph <= H and x_hr + ph <= W:
                residual[y_hr:y_hr + ph, x_hr:x_hr + ph] += res_patch
                weights [y_hr:y_hr + ph, x_hr:x_hr + ph] += 1.0

        weights = np.maximum(weights, 1)
        output  = bicubic + (residual / weights) * 128.0
        out_gray = np.clip(output, 0, 255).astype(np.uint8)

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
