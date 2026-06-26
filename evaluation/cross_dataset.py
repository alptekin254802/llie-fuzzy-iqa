"""
Step 3 - Cross-dataset feature + NR-anchor collector (unpaired datasets).

Place at: evaluation/cross_dataset.py     Run: python -m evaluation.cross_dataset

Addresses the "single dataset (LOL)" critique. Takes images from unpaired
real-world low-light datasets (LIME, DICM, ...), applies the SAME 5 classical
enhancement methods, and computes no-reference features + modern NR anchors.
There is NO ground truth here, so no PSNR/SSIM/LPIPS - the reference-free model
(trained on LOL) will be applied to these rows in the FINAL evaluation and its
score correlated against the NR anchors (MUSIQ / -NIQE) to demonstrate
cross-dataset generalisation with no changes to the model.

Data layout (you create this):
  datasets/unpaired/lime/*.png|jpg|bmp
  datasets/unpaired/dicm/*.png|jpg|bmp
  (optionally npe/, mef/, vv/ for more images)
Easiest single source for all of these: https://github.com/baidut/BIMEF  (data/ folder).

Output: results/cross_dataset.csv
  columns: dataset, filename, method, brightness, contrast, entropy,
           colorfulness, sharpness, noise, niqe, brisque, musiq, maniqa, clipiqa
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

import pyiqa
from preprocessing.enhancement.methods import (
    apply_clahe, apply_gamma_correction, apply_log_transform,
    apply_retinex, histogram_equalization,
)
from evaluation.metrics import compute_entropy

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE / "datasets" / "unpaired"     # contains one subfolder per dataset
OUTPUT_CSV = BASE / "results" / "cross_dataset.csv"
SIZE = 256                                         # match the LOL preprocessing
NR_METRICS = ["niqe", "brisque", "musiq", "maniqa", "clipiqa"]
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

METHODS = {
    "clahe": apply_clahe, "gamma": apply_gamma_correction,
    "log": apply_log_transform, "retinex": apply_retinex,
    "hist_eq": histogram_equalization,
}


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.float32) / 255.0


def to_tensor(rgb01: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(rgb01)).permute(2, 0, 1).unsqueeze(0).float().clamp(0, 1).to(device)


def nr_features(rgb01: np.ndarray) -> dict:
    rgb8 = np.clip(rgb01 * 255.0, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY).astype(np.float64)
    R, G, B = rgb8[..., 0].astype(np.float64), rgb8[..., 1].astype(np.float64), rgb8[..., 2].astype(np.float64)
    rg, yb = R - G, 0.5 * (R + G) - B
    Kmask = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = cv2.filter2D(gray, cv2.CV_64F, Kmask, borderType=cv2.BORDER_REPLICATE)
    H, W = gray.shape
    return {
        "brightness": float(gray.mean() / 255.0),
        "contrast": float(gray.std() / 255.0),
        "entropy": compute_entropy(rgb8),
        "colorfulness": float((np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)) / 255.0),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var() / (255.0 ** 2)),
        "noise": float(np.sum(np.abs(conv)) * np.sqrt(0.5 * np.pi) / (6.0 * max(W - 2, 1) * max(H - 2, 1)) / 255.0),
    }


def main() -> None:
    if not DATASETS_DIR.exists():
        sys.exit(f"Create {DATASETS_DIR} with one subfolder per dataset (e.g. lime/, dicm/).\n"
                 f"Easiest source: https://github.com/baidut/BIMEF (data/ folder).")
    subdirs = sorted([d for d in DATASETS_DIR.iterdir() if d.is_dir()])
    if not subdirs:
        sys.exit(f"No dataset subfolders found in {DATASETS_DIR}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    nr_models = {}
    for name in NR_METRICS:
        try:
            nr_models[name] = pyiqa.create_metric(name, device=device)
        except Exception as e:
            print(f"  [skip] {name}: {e}")

    rows = []
    for ds in subdirs:
        imgs = sorted([p for p in ds.iterdir() if p.suffix.lower() in EXTS])
        print(f"\n{ds.name}: {len(imgs)} images")
        for j, p in enumerate(imgs, 1):
            low = load_rgb(p)
            if low is None:
                print(f"  [skip unreadable] {p.name}")
                continue
            for label, fn in METHODS.items():
                enh = np.clip(fn(low), 0.0, 1.0).astype(np.float32)
                rec = {"dataset": ds.name, "filename": p.name, "method": label}
                rec.update(nr_features(enh))
                t = to_tensor(enh, device)
                with torch.no_grad():
                    for name, m in nr_models.items():
                        try:
                            rec[name] = float(m(t).item())
                        except Exception:
                            rec[name] = np.nan
                rows.append(rec)
            if j % 20 == 0 or j == len(imgs):
                print(f"  {j}/{len(imgs)}")

    out = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}  ({len(out)} rows)")
    print("Per-dataset enhanced-image counts:")
    print(out.groupby("dataset").size().to_string())

    # peek: do our raw features already track MUSIQ on this unseen data?
    from scipy.stats import spearmanr
    m = out.dropna(subset=["musiq"])
    if len(m) > 10:
        print(f"\nSanity (image-level Spearman vs MUSIQ, n={len(m)}):")
        for f in ["entropy", "contrast", "sharpness"]:
            print(f"  {f:10s}: {spearmanr(m[f], m['musiq'])[0]:+.3f}")
        print("  [the trained fuzzy combination of these is applied in the final evaluation]")


if __name__ == "__main__":
    main()
