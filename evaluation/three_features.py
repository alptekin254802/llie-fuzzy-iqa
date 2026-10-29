"""The three deployed features from an enhanced RGB uint8 image."""
import cv2
import numpy as np


def from_gray(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist[hist > 0] / hist.sum()
    gf = gray.astype(np.float64)
    return (float(-(p * np.log2(p)).sum()), float(gf.std() / 255.0),
            float(cv2.Laplacian(gf, cv2.CV_64F, borderType=cv2.BORDER_REFLECT_101).var() / 255.0**2))


def extract(rgb8):
    """Returns entropy (bits), normalized contrast and normalized Laplacian variance."""
    if rgb8.dtype != np.uint8 or rgb8.ndim != 3 or rgb8.shape[2] != 3:
        raise ValueError("Expected an RGB uint8 image with three channels")
    return from_gray(cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY))
