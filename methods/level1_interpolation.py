"""Niveau 1 — interpolations classiques : bicubique, bilinéaire, Lanczos."""
import cv2
import numpy as np

_METHODS = {
    "bicubic":   cv2.INTER_CUBIC,
    "bilinear":  cv2.INTER_LINEAR,
    "lanczos":   cv2.INTER_LANCZOS4,
    "nearest":   cv2.INTER_NEAREST,
}


def upscale(img: np.ndarray, scale: int, method: str = "bicubic") -> np.ndarray:
    if method not in _METHODS:
        raise ValueError(f"Méthode inconnue : {method}. Choix : {list(_METHODS)}")
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=_METHODS[method])


def run_all(img: np.ndarray, scale: int) -> dict[str, np.ndarray]:
    return {name: upscale(img, scale, name) for name in _METHODS}
