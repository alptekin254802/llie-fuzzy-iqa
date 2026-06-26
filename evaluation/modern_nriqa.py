"""
Step 1 - Modern learned no-reference IQA metrics for the enhanced images.

Place at: evaluation/modern_nriqa.py    Run: python -m evaluation.modern_nriqa

Computes MUSIQ, MANIQA and CLIP-IQA (all no-reference, human-MOS-trained deep
metrics) for every enhanced image, using the same pipeline as the other scripts
(regenerate the enhanced image from the LOL low-light input via methods.py, then
score it). These serve two purposes later:
  (i)  a modern NR-IQA comparison row in Table 1 (their alignment with -LPIPS);
  (ii) an additional no-reference anchor for cross-dataset validation (Step 3).

Output: results/modern_nriqa.csv   (filename, method, musiq, maniqa, clipiqa)

Notes:
  * Needs `pyiqa` and `torch` (already used for LPIPS/NIQE/BRISQUE).
  * A GPU is strongly recommended; on CPU this is slow (thousands of deep
    forward passes). Trim METRICS if needed.
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

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
LOW_DIR = BASE_DIR / "preprocessed" / "low"
CLASSICAL_CSV = BASE_DIR / "results" / "enhancement_metrics.csv"
PERCEPTUAL_CSV = BASE_DIR / "results" / "perceptual_metrics.csv"   # for a sanity check
OUTPUT_CSV = BASE_DIR / "results" / "modern_nriqa.csv"

METRICS = ["musiq", "maniqa", "clipiqa"]   # all no-reference; comment out any to speed up

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


def to_tensor(rgb01: np.ndarray, device) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(rgb01)).permute(2, 0, 1).unsqueeze(0).float()
    return t.clamp(0, 1).to(device)


def main() -> None:
    if not CLASSICAL_CSV.exists():
        sys.exit(f"Not found: {CLASSICAL_CSV}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading metrics: {METRICS}")
    models = {}
    for name in METRICS:
        try:
            models[name] = pyiqa.create_metric(name, device=device)
            lo_better = getattr(models[name], "lower_better", None)
            print(f"  {name}: ready (lower_better={lo_better})")
        except Exception as e:
            print(f"  [skip] {name}: {e}")
    if not models:
        sys.exit("No metrics could be created.")

    df = pd.read_csv(CLASSICAL_CSV)
    filenames = sorted(df["filename"].unique())
    n = len(filenames)
    print(f"Images: {n} | methods: {sorted(df['method'].unique())}")

    rows = []
    for i, fname in enumerate(filenames, 1):
        low = load_rgb_float(resolve_image(LOW_DIR, fname))
        for label, fn in METHODS.items():
            enh = np.clip(fn(low), 0.0, 1.0).astype(np.float32)
            x = to_tensor(enh, device)
            rec = {"filename": fname, "method": label}
            with torch.no_grad():
                for name, m in models.items():
                    try:
                        rec[name] = float(m(x).item())
                    except Exception:
                        rec[name] = np.nan
            rows.append(rec)
        if i % 25 == 0 or i == n:
            print(f"  {i}/{n} images done")

    out = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}  ({len(out)} rows)")
    print("Ranges:")
    for name in models:
        if name in out:
            print(f"  {name:8s}: min={out[name].min():.4g}  med={out[name].median():.4g}  max={out[name].max():.4g}")

    # quick sanity: do these modern NR metrics track perceptual quality (-LPIPS)?
    if PERCEPTUAL_CSV.exists():
        from scipy.stats import spearmanr
        pc = pd.read_csv(PERCEPTUAL_CSV)[["filename", "method", "lpips"]]
        m = out.merge(pc, on=["filename", "method"]).dropna()
        tgt = -m["lpips"].to_numpy()
        print(f"\nImage-level Spearman with -LPIPS (n={len(m)}):")
        for name in models:
            if name in m:
                r = spearmanr(m[name].to_numpy(), tgt)[0]
                print(f"  {name:8s}: {r:+.3f}")
        print("  [for reference: PSNR 0.574, our NR-free fuzzy 0.753]")


if __name__ == "__main__":
    main()
