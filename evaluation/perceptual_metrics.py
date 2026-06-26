"""
Stage 1a - Perceptual metric computation (LPIPS / NIQE / BRISQUE).

Place this file at:  evaluation/perceptual_metrics.py
Run from project root:  python -m evaluation.perceptual_metrics

For every (image, method) pair this script:
  1. Re-applies the five classical enhancement methods to the preprocessed
     low-light images, reusing preprocessing.enhancement.methods, so the
     enhanced images are IDENTICAL to those behind enhancement_metrics.csv.
  2. Computes three perceptual metrics that are NOT inputs to the fuzzy system:
       - LPIPS  : full-reference learned perceptual distance (lower = better)
       - NIQE   : no-reference natural-image quality          (lower = better)
       - BRISQUE: no-reference perceptual quality             (lower = better)
  3. Re-computes PSNR / SSIM / Entropy with the project's own metrics module as
     a CONSISTENCY CHECK against enhancement_metrics.csv, so we can prove the
     enhanced images are faithfully reproduced before trusting LPIPS.

Output: results/perceptual_metrics.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    import torch
    import pyiqa
except ImportError as exc:  # pragma: no cover
    sys.exit(
        "Missing dependency. Install with:\n"
        "  pip install pyiqa torch torchvision\n"
        f"(import error: {exc})"
    )

from preprocessing.enhancement.methods import (
    apply_clahe,
    apply_gamma_correction,
    apply_log_transform,
    apply_retinex,
    histogram_equalization,
)
from evaluation.metrics import compute_psnr, compute_ssim, compute_entropy

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
LOW_DIR = BASE_DIR / "preprocessed" / "low"
HIGH_DIR = BASE_DIR / "preprocessed" / "high"
CLASSICAL_CSV = BASE_DIR / "results" / "enhancement_metrics.csv"
OUTPUT_CSV = BASE_DIR / "results" / "perceptual_metrics.csv"

# CSV method label -> enhancement function.
# Update these keys if enhancement_metrics.csv uses different method labels.
METHODS = {
    "clahe": apply_clahe,
    "gamma": apply_gamma_correction,
    "log": apply_log_transform,
    "retinex": apply_retinex,
    "hist_eq": histogram_equalization,
}


def resolve_image(directory: Path, filename: str) -> Path:
    """Return the image path, trying the name as-is then common extensions."""
    candidate = directory / filename
    if candidate.exists():
        return candidate
    stem = Path(filename).stem
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        alt = directory / f"{stem}{ext}"
        if alt.exists():
            return alt
    raise FileNotFoundError(f"Image not found for '{filename}' in {directory}")


def load_rgb_float(path: Path) -> np.ndarray:
    """Load an image as RGB float32 in [0, 1]."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def to_tensor(img_rgb_float: np.ndarray, device: str) -> "torch.Tensor":
    """HWC RGB float[0,1] numpy -> (1,3,H,W) torch tensor on device."""
    t = torch.from_numpy(img_rgb_float).permute(2, 0, 1).unsqueeze(0).contiguous()
    return t.to(device).float()


def main() -> None:
    if not CLASSICAL_CSV.exists():
        sys.exit(f"Not found: {CLASSICAL_CSV} (run the enhancement pipeline first).")

    df = pd.read_csv(CLASSICAL_CSV)
    filenames = sorted(df["filename"].unique())
    csv_methods = sorted(df["method"].unique())
    print(f"Images: {len(filenames)} | methods in CSV: {csv_methods}")

    unknown = [m for m in csv_methods if m not in METHODS]
    if unknown:
        print(f"[WARN] CSV methods without a mapping in METHODS: {unknown}")
        print("       Edit the METHODS dict so the keys match the CSV labels.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Creating perceptual metrics (first run downloads small weights)...")
    lpips_m = pyiqa.create_metric("lpips", device=device)      # FR, lower=better
    niqe_m = pyiqa.create_metric("niqe", device=device)        # NR, lower=better
    brisque_m = pyiqa.create_metric("brisque", device=device)  # NR, lower=better

    rows = []
    n = len(filenames)
    for i, fname in enumerate(filenames, 1):
        low = load_rgb_float(resolve_image(LOW_DIR, fname))
        high = load_rgb_float(resolve_image(HIGH_DIR, fname))
        high_t = to_tensor(high, device)

        for label, fn in METHODS.items():
            enh = np.clip(fn(low), 0.0, 1.0).astype(np.float32)  # RGB float [0,1]
            enh_t = to_tensor(enh, device)
            with torch.no_grad():
                lpips_val = float(lpips_m(enh_t, high_t).item())
                niqe_val = float(niqe_m(enh_t).item())
                brisque_val = float(brisque_m(enh_t).item())
            rows.append(
                {
                    "filename": fname,
                    "method": label,
                    "lpips": lpips_val,
                    "niqe": niqe_val,
                    "brisque": brisque_val,
                    # consistency-check columns:
                    "psnr_check": compute_psnr(enh, high),
                    "ssim_check": compute_ssim(enh, high),
                    "entropy_check": compute_entropy(enh),
                }
            )
        if i % 25 == 0 or i == n:
            print(f"  {i}/{n} images done")

    out = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}  ({len(out)} rows)")

    # ---- consistency check against enhancement_metrics.csv ----
    print("\nConsistency check (recomputed vs enhancement_metrics.csv):")
    merged = out.merge(df, on=["filename", "method"], suffixes=("", "_csv"))
    ok = True
    for col in ["psnr", "ssim", "entropy"]:
        diff = (merged[f"{col}_check"] - merged[col]).abs()
        flag = "" if diff.max() < 1e-3 else "  <-- LARGE, check loading/color order"
        if diff.max() >= 1e-3:
            ok = False
        print(f"  {col:8s}: mean|d|={diff.mean():.4g}  max|d|={diff.max():.4g}{flag}")
    if ok:
        print("OK: enhanced images match enhancement_metrics.csv; LPIPS/NIQE/BRISQUE are trustworthy.")
    else:
        print("Mismatch detected; check color order, image loading, and enhancement labels.")


if __name__ == "__main__":
    main()
