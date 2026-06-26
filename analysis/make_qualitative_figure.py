"""
Compact qualitative demonstration figure (PDF + SVG).

Place at: analysis/make_qualitative_figure.py
Run:      python -m analysis.make_qualitative_figure

Shows scenes where PSNR prefers histogram equalisation (higher PSNR) but the
reference-free fuzzy score - like human perception (LPIPS) - prefers the log
transform. Each row: low-light input, hist-eq, log, reference. The hist-eq and
log panels are annotated with PSNR (reference-based) and the proposed
reference-free score, making the disagreement visible.

Edit SCENES to pick different examples. Strong candidates (from the data):
  100.png, 189.png, 151.png, 143.png, 212.png, 598.png
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocessing.enhancement.methods import (
    apply_log_transform, histogram_equalization,
)

BASE = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("RESULTS_DIR", BASE / "results"))
LOW_DIR = BASE / "preprocessed" / "low"
HIGH_DIR = BASE / "preprocessed" / "high"
OUT = RESULTS / "figures"

SCENES = ["100.png", "189.png"]          # two demonstration scenes
# columns: (label, kind). kind in {"low","high", method-name}
COLUMNS = [("Low-light", "low"), ("Hist. Eq.", "hist_eq"),
           ("Log", "log"), ("Reference", "high")]
ENH = {"hist_eq": histogram_equalization, "log": apply_log_transform}

mpl.rcParams.update({"font.family": "serif", "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.02, "figure.dpi": 300,
                     "svg.fonttype": "none"})


def resolve(directory, fname):
    p = directory / fname
    if p.exists():
        return p
    stem = Path(fname).stem
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        if (directory / f"{stem}{ext}").exists():
            return directory / f"{stem}{ext}"
    raise FileNotFoundError(f"{fname} not in {directory}")


def load_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def metric_lookup():
    em = pd.read_csv(RESULTS / "enhancement_metrics.csv")
    pc = pd.read_csv(RESULTS / "perceptual_metrics.csv")
    nr = pd.read_csv(RESULTS / "nr_fuzzy_scores.csv")
    m = (em[["filename", "method", "psnr"]]
         .merge(pc[["filename", "method", "lpips"]], on=["filename", "method"])
         .merge(nr[["filename", "method", "nr_fuzzy_free"]], on=["filename", "method"]))
    return m.set_index(["filename", "method"])


def main():
    if not (RESULTS / "nr_fuzzy_scores.csv").exists():
        sys.exit("Run reference_free_fuzzy.py first (needs nr_fuzzy_scores.csv).")
    mt = metric_lookup()
    nrows, ncols = len(SCENES), len(COLUMNS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.1 * ncols, 2.5 * nrows))
    if nrows == 1:
        axes = axes[None, :]

    for r, scene in enumerate(SCENES):
        low = load_rgb(resolve(LOW_DIR, scene))
        high = load_rgb(resolve(HIGH_DIR, scene))
        for c, (label, kind) in enumerate(COLUMNS):
            ax = axes[r, c]
            if kind == "low":
                img = low
            elif kind == "high":
                img = high
            else:
                img = np.clip(ENH[kind](low), 0, 1)
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(label, fontsize=10)
            # annotate metrics for the two enhancement columns
            if kind in ENH:
                row = mt.loc[(scene, kind)]
                txt = (f"PSNR {row['psnr']:.1f} dB\n"
                       f"NR-score {row['nr_fuzzy_free']:.0f}")
                ax.set_xlabel(txt, fontsize=8)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(OUT / f"fig6_qualitative.{ext}")
    plt.close(fig)
    print(f"Saved fig6_qualitative.pdf / .svg to {OUT}")
    print("Annotated scenes:", SCENES)


if __name__ == "__main__":
    main()
