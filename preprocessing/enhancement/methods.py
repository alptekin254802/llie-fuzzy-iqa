"""
Classical image enhancement methods.

Each function accepts a numpy array of RGB images (uint8 [0, 255] or float32 [0, 1])
and returns an enhanced RGB array of the same dtype range. Modularizing the functions
allows us to attach them to different lines (notebooks, scripts, etc.) without repetition.
"""

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
    """
    Applies Contrast-Limited Adaptive Histogram Equalization (CLAHE).

    CLAHE, small squares are individually equalized to increase local contrast,
    while limiting noise amplification through clipping limit. We work on the
    lightness channel in the LAB color space to preserve natural colors.
    """
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
    """
    Applies gamma correction to brighten or darken the image.

    gamma > 1 values compress highlights and emphasize mid-tones, which can
    brighten dull images cleanly without over-saturating bright pixels.
    """
    img_float, original_dtype = _to_float(img)
    corrected = np.power(img_float, 1.0 / gamma)
    return _restore_dtype(corrected, original_dtype)


def apply_log_transform(img: np.ndarray, c: float = 1.0) -> np.ndarray:
    """
    Applies logarithmic transformation, which emphasizes darker densities over
    lighter ones. This is useful when details are hidden in deep shadows.
    """
    img_float, original_dtype = _to_float(img)
    transformed = c * np.log1p(img_float)
    transformed /= transformed.max() + 1e-8  # normalize back to [0, 1] range
    return _restore_dtype(transformed, original_dtype)


def apply_retinex(img: np.ndarray, sigma: float = 50.0) -> np.ndarray:
    """
    Applies Single-Scale Retinex (SSR).

    SSR, brightens the image as a blurred version of itself, and subtracts the
    log domain, which provides a better dynamic range and color stability.
    """
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
    """
    Applies histogram equalization per channel.

    Equalizing each RGB channel individually can increase global contrast and
    work well for quick, simple enhancements.
    """
    img_float, original_dtype = _to_float(img)
    img_uint8 = _restore_dtype(img_float, "uint8")
    channels = cv2.split(img_uint8)
    eq_channels = [cv2.equalizeHist(ch) for ch in channels]
    rgb_eq = cv2.merge(eq_channels)
    return _restore_dtype(rgb_eq.astype(np.float32) / 255.0, original_dtype)

