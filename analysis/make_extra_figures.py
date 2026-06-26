"""
Generate the kept/regenerated 'original' figures (+ a couple of extras) as PDF+SVG.

Place at: analysis/make_extra_figures.py    Run: python -m analysis.make_extra_figures

Produces (results/figures/):
  figS_membership_functions  - trapezoidal MFs of the 3 NR features (Supplementary)
  fig_method_means           - per-method PSNR vs -LPIPS vs NR-score (metric disagreement)
  fig7_methods_grid          - one scene, all 5 methods + low + reference (needs images)

The first two are computed from CSVs only; the grid needs preprocessed images and
preprocessing.enhancement.methods, so it is skipped if those are unavailable.
"""
from __future__ import annotations

import itertools
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("RESULTS_DIR", BASE / "results"))
LOW_DIR = BASE / "preprocessed" / "low"
HIGH_DIR = BASE / "preprocessed" / "high"
OUT = RESULTS / "figures"
OUT.mkdir(parents=True, exist_ok=True)

NR_FEATURES = ["entropy", "contrast", "sharpness"]
SAMPLE = "10.png"                         # scene for the method grid
GRID_METHODS = ["clahe", "gamma", "hist_eq", "log", "retinex"]

mpl.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, GREEN, ORANGE, RED = "#4c72b0", "#55a868", "#dd8452", "#c44e52"


def save(fig, name):
    for ext in ("pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  saved {name}.pdf / .svg")


def trapmf(x, a, b, c, d):
    x = np.asarray(x, float)
    left = np.where(b > a, (x - a) / np.maximum(b - a, 1e-12), 1.0)
    right = np.where(d > c, (d - x) / np.maximum(d - c, 1e-12), 1.0)
    return np.clip(np.minimum(np.minimum(left, 1.0), right), 0, 1)


def mfset(v):
    q = np.quantile(v, [0, .2, .4, .5, .6, .8, 1])
    return [(q[0], q[0], q[1], q[2]), (q[1], q[2], q[4], q[5]), (q[4], q[5], q[6], q[6])]


def fig_membership_functions():
    nr = pd.read_csv(RESULTS / "nr_features.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
    names = {"entropy": "Entropy (bit)", "contrast": "Contrast", "sharpness": "Sharpness"}
    colors = [BLUE, GREEN, RED]
    handles = None
    for ax, feat in zip(axes, NR_FEATURES):
        v = nr[feat].to_numpy()
        mfs = mfset(v)
        # clip the plotting window to avoid heavy right-skew squashing the curves
        lo, hi = v.min(), float(np.quantile(v, 0.98))
        xs = np.linspace(lo, hi, 500)
        hs = []
        for (lab, mf, col) in zip(["low", "med", "high"], mfs, colors):
            h, = ax.plot(xs, trapmf(xs, *mf), color=col, label=lab, lw=1.7)
            hs.append(h)
        handles = hs
        ax.set_xlim(lo, hi); ax.set_ylim(-0.03, 1.1)
        ax.set_title(names[feat]); ax.set_xlabel("value"); ax.set_ylabel("membership")
    fig.legend(handles, ["low", "med", "high"], frameon=False, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.08), fontsize=9)
    fig.suptitle("Trapezoidal membership functions of the no-reference features", y=1.02)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "figS_membership_functions")


def fig_method_means():
    em = pd.read_csv(RESULTS / "enhancement_metrics.csv")
    pc = pd.read_csv(RESULTS / "perceptual_metrics.csv")
    nr = pd.read_csv(RESULTS / "nr_fuzzy_scores.csv")
    df = (em[["filename", "method", "psnr"]]
          .merge(pc[["filename", "method", "lpips"]], on=["filename", "method"])
          .merge(nr[["filename", "method", "nr_fuzzy_free"]], on=["filename", "method"]))
    agg = df.groupby("method").agg(psnr=("psnr", "mean"),
                                   neg_lpips=("lpips", lambda s: -s.mean()),
                                   nr=("nr_fuzzy_free", "mean"))
    # normalise each column to [0,1] across methods (higher = better for all)
    norm = (agg - agg.min()) / (agg.max() - agg.min())
    methods = list(norm.index)
    x = np.arange(len(methods)); w = 0.26
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.bar(x - w, norm["psnr"], w, label="PSNR (reference)", color=BLUE, edgecolor="black", linewidth=0.4)
    ax.bar(x, norm["neg_lpips"], w, label="$-$LPIPS (perceptual)", color=GREEN, edgecolor="black", linewidth=0.4)
    ax.bar(x + w, norm["nr"], w, label="NR-score (ours)", color=RED, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel("normalised method-level mean")
    ax.set_title("Methods rank differently under PSNR vs perception")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "fig_method_means")


def fig_methods_grid():
    try:
        import cv2
        from preprocessing.enhancement.methods import (
            apply_clahe, apply_gamma_correction, apply_log_transform,
            apply_retinex, histogram_equalization,
        )
    except Exception as e:
        print(f"  [skip] methods grid (needs images + methods.py): {e}")
        return
    enh = {"clahe": apply_clahe, "gamma": apply_gamma_correction,
           "hist_eq": histogram_equalization, "log": apply_log_transform,
           "retinex": apply_retinex}

    def resolve(d, f):
        p = d / f
        if p.exists():
            return p
        stem = Path(f).stem
        for ext in (".png", ".jpg", ".jpeg", ".bmp"):
            if (d / f"{stem}{ext}").exists():
                return d / f"{stem}{ext}"
        raise FileNotFoundError(f)

    def load(d, f):
        bgr = cv2.imread(str(resolve(d, f)), cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    em = pd.read_csv(RESULTS / "enhancement_metrics.csv")
    nr = pd.read_csv(RESULTS / "nr_fuzzy_scores.csv")
    mt = (em[["filename", "method", "psnr"]]
          .merge(nr[["filename", "method", "nr_fuzzy_free"]], on=["filename", "method"])
          .set_index(["filename", "method"]))

    low = load(LOW_DIR, SAMPLE); high = load(HIGH_DIR, SAMPLE)
    panels = [("Low-light", low, None)]
    for m in GRID_METHODS:
        panels.append((m, np.clip(enh[m](low), 0, 1), m))
    panels.append(("Reference", high, None))

    fig, axes = plt.subplots(1, len(panels), figsize=(1.7 * len(panels), 2.1))
    for ax, (label, img, m) in zip(axes, panels):
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label, fontsize=9)
        if m is not None:
            r = mt.loc[(SAMPLE, m)]
            ax.set_xlabel(f"PSNR {r['psnr']:.1f}\nNR {r['nr_fuzzy_free']:.0f}", fontsize=7.5)
    fig.suptitle(f"Enhancement methods on a sample scene ({SAMPLE})", y=1.04, fontsize=10)
    fig.tight_layout()
    save(fig, "fig7_methods_grid")


def main():
    print(f"Reading from: {RESULTS}\nWriting to: {OUT}")
    fig_membership_functions()
    fig_method_means()
    fig_methods_grid()
    print("Done.")


if __name__ == "__main__":
    main()
