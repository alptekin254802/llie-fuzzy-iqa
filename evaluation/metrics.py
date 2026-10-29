"""
Low-light image enhancement evaluation metrics.
"""

from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity

__all__ = ["compute_psnr", "compute_ssim", "compute_entropy"]


def _prepare_pair(img: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """For metric calculations, convert a pair of images to [0, 1] range float32."""
    def _to_float(arr: np.ndarray) -> np.ndarray:
        if arr.dtype == np.uint8:
            return arr.astype(np.float32) / 255.0
        return arr.astype(np.float32)

    img_f = _to_float(img)
    gt_f = _to_float(gt)
    if img_f.shape != gt_f.shape:
        raise ValueError("Image and ground-truth must have matching shapes.")
    return img_f, gt_f


def compute_psnr(img: np.ndarray, gt: np.ndarray) -> float:
    """Return PSNR in dB; higher values mean lower pixel error against the reference."""
    img_f, gt_f = _prepare_pair(img, gt)
    mse = np.mean((img_f - gt_f) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(1.0 / math.sqrt(mse))


def compute_ssim(img: np.ndarray, gt: np.ndarray) -> float:
    """Return channel-averaged structural similarity against the reference."""
    img_f, gt_f = _prepare_pair(img, gt)
    # structural_similarity, win_size expects channel-last RGB inputs by default.
    return float(
        structural_similarity(
            gt_f,
            img_f,
            channel_axis=2,
            data_range=1.0,
        )
    )


def compute_entropy(img: np.ndarray, bins: int = 256) -> float:
    """Return grayscale histogram entropy in bits; this does not measure detail or quality directly."""
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Entropy expects a RGB image.")

    if img.dtype == np.uint8:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = cv2.cvtColor((np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
    hist = hist.flatten()
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))

