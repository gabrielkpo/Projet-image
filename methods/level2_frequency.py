"""Niveau 2 — méthodes fréquentielles : DWT et FFT zero-padding."""
import numpy as np
import cv2
import pywt


def fft_zeropad(img: np.ndarray, scale: int) -> np.ndarray:
    """Zero-padding dans le domaine fréquentiel (FFT shift)."""
    def _upscale_channel(channel: np.ndarray) -> np.ndarray:
        h, w = channel.shape
        F = np.fft.fftshift(np.fft.fft2(channel))
        ph, pw = h * scale, w * scale
        pad_h, pad_w = (ph - h) // 2, (pw - w) // 2
        F_pad = np.zeros((ph, pw), dtype=complex)
        F_pad[pad_h:pad_h + h, pad_w:pad_w + w] = F * (scale ** 2)
        return np.abs(np.fft.ifft2(np.fft.ifftshift(F_pad)))

    if img.ndim == 2:
        out = _upscale_channel(img)
    else:
        out = np.stack([_upscale_channel(img[:, :, c]) for c in range(img.shape[2])], axis=2)

    return np.clip(out, 0, 255).astype(np.uint8)


def dwt_upscale(img: np.ndarray, scale: int, wavelet: str = "haar") -> np.ndarray:
    """Upscaling par extension de sous-bandes DWT."""
    def _upscale_channel(channel: np.ndarray) -> np.ndarray:
        levels = int(np.log2(scale))
        coeffs = pywt.wavedec2(channel.astype(float), wavelet, level=levels)
        # on étend les coefficients de détail à zéro (super-résolution naïve)
        new_coeffs = [coeffs[0]]
        for detail in coeffs[1:]:
            new_coeffs.append(tuple(np.zeros_like(d) for d in detail))
        rec = pywt.waverec2(new_coeffs, wavelet)
        return rec[:channel.shape[0] * scale, :channel.shape[1] * scale]

    if img.ndim == 2:
        out = _upscale_channel(img)
    else:
        out = np.stack([_upscale_channel(img[:, :, c]) for c in range(img.shape[2])], axis=2)

    return np.clip(out, 0, 255).astype(np.uint8)


def run_all(img: np.ndarray, scale: int) -> dict[str, np.ndarray]:
    return {
        "fft_zeropad": fft_zeropad(img, scale),
        "dwt_upscale":  dwt_upscale(img, scale),
    }
