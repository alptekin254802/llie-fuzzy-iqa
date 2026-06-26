"""
Stage B - Extract interpretable NO-REFERENCE features from enhanced images.

Place at: evaluation/nr_features.py      Run: python -m evaluation.nr_features

These features are computed from the enhanced image ALONE (no ground truth), so a
model built on them can run where no reference exists (autonomous driving, medical,
surveillance). All are cheap and human-interpretable:
  brightness   - mean luminance
  contrast     - RMS contrast (luminance std)
  entropy      - Shannon entropy (information / detail)
  colorfulness - Hasler-Susstrunk colorfulness
  sharpness    - variance of Laplacian (focus / edge strength)
  noise        - Immerkaer noise sigma estimate

Output: results/nr_features.csv   (filename, method, <features...>)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from preprocessing.enhancement.methods import (
    apply_clahe, apply_gamma_correction, apply_log_transform,
    apply_retinex, histogram_equalization,
)
from evaluation.metrics import compute_entropy

BASE_DIR = Path(__file__).resolve().parent.parent
LOW_DIR = BASE_DIR / "preprocessed" / "low"
CLASSICAL_CSV = BASE_DIR / "results" / "enhancement_metrics.csv"
OUTPUT_CSV = BASE_DIR / "results" / "nr_features.csv"

METHODS = {
    "clahe": apply_clahe, "gamma": apply_gamma_correction,
    "log": apply_log_transform, "retinex": apply_retinex,
    "hist_eq": histogram_equalization,
}


def resolve_image(directory: Path, filename: str) -> Path:
    p = directory / filename
    if p.exists():
        return p
    stem = Path(filename).stem
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        alt = directory / f"{stem}{ext}"
        if alt.exists():
            return alt
    raise FileNotFoundError(f"Image not found for '{filename}' in {directory}")


def load_rgb_float(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def nr_features(rgb01: np.ndarray) -> dict:
    """All features from the enhanced RGB image in [0,1]; no reference used."""
    rgb8 = np.clip(rgb01 * 255.0, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY).astype(np.float64)

    brightness = gray.mean() / 255.0
    contrast = gray.std() / 255.0

    R, G, B = (rgb8[..., 0].astype(np.float64),
               rgb8[..., 1].astype(np.float64),
               rgb8[..., 2].astype(np.float64))
    rg = R - G
    yb = 0.5 * (R + G) - B
    colorfulness = (np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                    + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)) / 255.0

    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var() / (255.0 ** 2)

    # Immerkaer fast noise sigma estimate
    Kmask = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = cv2.filter2D(gray, cv2.CV_64F, Kmask, borderType=cv2.BORDER_REPLICATE)
    H, W = gray.shape
    noise = (np.sum(np.abs(conv)) * np.sqrt(0.5 * np.pi)
             / (6.0 * max(W - 2, 1) * max(H - 2, 1))) / 255.0

    return {
        "brightness": float(brightness),
        "contrast": float(contrast),
        "entropy": float(compute_entropy(rgb01)),
        "colorfulness": float(colorfulness),
        "sharpness": float(sharpness),
        "noise": float(noise),
    }


def main() -> None:
    if not CLASSICAL_CSV.exists():
        sys.exit(f"Not found: {CLASSICAL_CSV}")
    df = pd.read_csv(CLASSICAL_CSV)
    filenames = sorted(df["filename"].unique())
    print(f"Images: {len(filenames)} | methods: {sorted(df['method'].unique())}")

    rows = []
    n = len(filenames)
    for i, fname in enumerate(filenames, 1):
        low = load_rgb_float(resolve_image(LOW_DIR, fname))
        for label, fn in METHODS.items():
            enh = np.clip(fn(low), 0.0, 1.0).astype(np.float32)
            rows.append({"filename": fname, "method": label, **nr_features(enh)})
        if i % 50 == 0 or i == n:
            print(f"  {i}/{n} images done")

    out = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}  ({len(out)} rows)")
    print("Feature ranges:")
    for c in ["brightness", "contrast", "entropy", "colorfulness", "sharpness", "noise"]:
        print(f"  {c:12s}: min={out[c].min():.4g}  med={out[c].median():.4g}  max={out[c].max():.4g}")


if __name__ == "__main__":
    main()
