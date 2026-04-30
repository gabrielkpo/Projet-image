"""Niveau 4 — ML classique : sparse coding (approx.), random forest, PCA."""
import numpy as np
import cv2
from sklearn.decomposition import PCA, DictionaryLearning
from sklearn.ensemble import RandomForestRegressor


def _extract_patches(img: np.ndarray, patch_size: int, stride: int) -> tuple[np.ndarray, list]:
    h, w = img.shape[:2]
    patches, positions = [], []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patches.append(img[y:y + patch_size, x:x + patch_size].flatten())
            positions.append((y, x))
    return np.array(patches, dtype=np.float32), positions


def pca_sr(img: np.ndarray, scale: int, n_components: int = 32,
           patch_size: int = 8, stride: int = 4) -> np.ndarray:
    """PCA sur patches LR → reconstruction dans l'espace HR approximé."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    upscaled = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    patches, positions = _extract_patches(upscaled, patch_size, stride)
    pca = PCA(n_components=min(n_components, patches.shape[0], patches.shape[1]))
    transformed = pca.fit_transform(patches)
    reconstructed_patches = pca.inverse_transform(transformed)

    output = np.zeros_like(upscaled, dtype=np.float32)
    count = np.zeros_like(upscaled, dtype=np.float32)
    for patch, (y, x) in zip(reconstructed_patches, positions):
        output[y:y + patch_size, x:x + patch_size] += patch.reshape(patch_size, patch_size)
        count[y:y + patch_size, x:x + patch_size] += 1

    count = np.maximum(count, 1)
    out = np.clip(output / count, 0, 255).astype(np.uint8)
    if img.ndim == 3:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    return out


def random_forest_sr(lr_img: np.ndarray, hr_img: np.ndarray, test_img: np.ndarray,
                     scale: int, patch_size: int = 4, stride: int = 2,
                     n_estimators: int = 50) -> np.ndarray:
    """RF entraîné sur patches LR→HR fournis, puis appliqué à test_img."""
    def to_gray(i):
        return cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) if i.ndim == 3 else i

    lr_g = to_gray(lr_img).astype(np.float32)
    hr_g = to_gray(hr_img).astype(np.float32)

    lr_up = cv2.resize(lr_g, (lr_g.shape[1] * scale, lr_g.shape[0] * scale),
                       interpolation=cv2.INTER_CUBIC)
    X, _ = _extract_patches(lr_up, patch_size, stride)
    Y, positions = _extract_patches(hr_g, patch_size, stride)
    n = min(len(X), len(Y))
    X, Y = X[:n], Y[:n]

    rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1)
    rf.fit(X, Y)

    test_g = to_gray(test_img).astype(np.float32)
    test_up = cv2.resize(test_g, (test_g.shape[1] * scale, test_g.shape[0] * scale),
                         interpolation=cv2.INTER_CUBIC)
    X_test, pos_test = _extract_patches(test_up, patch_size, stride)
    Y_pred = rf.predict(X_test)

    output = np.zeros_like(test_up, dtype=np.float32)
    count = np.zeros_like(test_up, dtype=np.float32)
    for patch, (y, x) in zip(Y_pred, pos_test):
        output[y:y + patch_size, x:x + patch_size] += patch.reshape(patch_size, patch_size)
        count[y:y + patch_size, x:x + patch_size] += 1

    count = np.maximum(count, 1)
    out = np.clip(output / count, 0, 255).astype(np.uint8)
    if test_img.ndim == 3:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    return out
