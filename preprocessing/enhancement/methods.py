"""Classical enhancement operators for RGB arrays.

Inputs are uint8 in [0,255] or float32 in [0,1]. Outputs retain the corresponding
range and are returned as uint8 or float32."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

__all__ = [
    "apply_clahe",
    "apply_gamma_correction",
    "apply_log_transform",
    "apply_retinex",
    "histogram_equalization",
]


def _to_float(img: np.ndarray) -> Tuple[np.ndarray, str]:
    """Convert any input image to [0, 1] range float32 and remember the original dtype."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0, "uint8"
    return img.astype(np.float32), "float"


def _restore_dtype(img_float: np.ndarray, original: str) -> np.ndarray:
    """Restore the processed float image to the original dtype/range."""
    if original == "uint8":
        return np.clip(img_float * 255.0, 0.0, 255.0).astype(np.uint8)
    return np.clip(img_float, 0.0, 1.0).astype(np.float32)


def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE to the LAB lightness channel."""
    img_float, original_dtype = _to_float(img)
    img_uint8 = _restore_dtype(img_float, "uint8")

    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    return _restore_dtype(rgb_enhanced.astype(np.float32) / 255.0, original_dtype)


def apply_gamma_correction(img: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    """Apply x ** (1/gamma); gamma above one brightens values between zero and one."""
    img_float, original_dtype = _to_float(img)
    corrected = np.power(img_float, 1.0 / gamma)
    return _restore_dtype(corrected, original_dtype)


def apply_log_transform(img: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Apply c * log(1+x), normalized by the maximum transformed value."""
    img_float, original_dtype = _to_float(img)
    transformed = c * np.log1p(img_float)
    transformed /= transformed.max() + 1e-8  # normalize back to [0, 1] range
    return _restore_dtype(transformed, original_dtype)


def apply_retinex(img: np.ndarray, sigma: float = 50.0) -> np.ndarray:
    """Subtract the log Gaussian surround from each log RGB channel, then normalize globally."""
    img_float, original_dtype = _to_float(img)
    eps = 1e-8

    retinex = np.zeros_like(img_float)
    for c in range(3):
        channel = img_float[:, :, c]
        blur = cv2.GaussianBlur(channel, (0, 0), sigma)
        retinex[:, :, c] = np.log(channel + eps) - np.log(blur + eps)

    # normalize back to [0, 1] range
    retinex -= retinex.min()
    retinex /= retinex.max() + eps
    return _restore_dtype(retinex, original_dtype)


def histogram_equalization(img: np.ndarray) -> np.ndarray:
    """Equalize each RGB channel independently."""
    img_float, original_dtype = _to_float(img)
    img_uint8 = _restore_dtype(img_float, "uint8")
    channels = cv2.split(img_uint8)
    eq_channels = [cv2.equalizeHist(ch) for ch in channels]
    rgb_eq = cv2.merge(eq_channels)
    return _restore_dtype(rgb_eq.astype(np.float32) / 255.0, original_dtype)

