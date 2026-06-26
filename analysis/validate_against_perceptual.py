"""
Stage 1b - Image-level validation against an INDEPENDENT perceptual anchor.

Place this file at:  analysis/validate_against_perceptual.py
Run from project root:  python -m analysis.validate_against_perceptual

Merges the fuzzy results with perceptual_metrics.csv and answers the key
question: across all enhanced images, does the fuzzy quality score track
perceptual quality (LPIPS / NIQE / BRISQUE) better than any single classical
metric does? Because LPIPS/NIQE/BRISQUE are NOT inputs to the fuzzy system,
this is an external (non-circular) validation, computed image-level (n ~ 2425)
instead of method-level (n = 5).

Outputs:
  - results/perceptual_validation_summary.csv
  - results/figures/imagelevel_correlation_matrix.png  (correct; replaces old Fig.11)
  - results/figures/alignment_with_lpips.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
FUZZY_CSV = BASE_DIR / "results" / "fuzzy_enhancement_results.csv"
PERCEPTUAL_CSV = BASE_DIR / "results" / "perceptual_metrics.csv"
FIG_DIR = BASE_DIR / "results" / "figures"
SUMMARY_CSV = BASE_DIR / "results" / "perceptual_validation_summary.csv"

# Perceptual metrics are all "lower = better"; we flip the sign so that, like
# the classical metrics and the fuzzy score, "higher = better quality".
PERCEPTUAL = ["lpips", "niqe", "brisque"]
PREDICTORS = ["psnr", "ssim", "entropy", "fuzzy_score"]
PRIMARY = "lpips"        # main perceptual anchor for the headline result
N_BOOT = 1000
RNG = np.random.default_rng(42)


def _rho(x: np.ndarray, y: np.ndarray) -> float:
    return float(spearmanr(x, y)[0])


def _tau(x: np.ndarray, y: np.ndarray) -> float:
    return float(kendalltau(x, y)[0])


def load_merged() -> pd.DataFrame:
    if not FUZZY_CSV.exists():
        sys.exit(f"Not found: {FUZZY_CSV}")
    if not PERCEPTUAL_CSV.exists():
        sys.exit(f"Not found: {PERCEPTUAL_CSV} (run evaluation.perceptual_metrics first).")
    fz = pd.read_csv(FUZZY_CSV)
    pc = pd.read_csv(PERCEPTUAL_CSV)
    merged = fz.merge(
        pc[["filename", "method"] + PERCEPTUAL],
        on=["filename", "method"],
        how="inner",
    ).dropna(subset=PREDICTORS + PERCEPTUAL)
    print(f"Merged rows: {len(merged)}")
    return merged


def bootstrap_rho(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT):
    """Point estimate and 95% CI of Spearman rho, plus the bootstrap samples."""
    n = len(x)
    idx = np.arange(n)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        s = RNG.choice(idx, size=n, replace=True)
        boots[b] = _rho(x[s], y[s])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return _rho(x, y), lo, hi, boots


def main() -> None:
    df = load_merged()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Sign-flipped perceptual targets (higher = better).
    target = {m: -df[m].to_numpy() for m in PERCEPTUAL}

    # ---- point correlations for every predictor x perceptual pair ----
    summary_rows = []
    for pred in PREDICTORS:
        x = df[pred].to_numpy()
        for m in PERCEPTUAL:
            summary_rows.append(
                {
                    "predictor": pred,
                    "perceptual": m,
                    "spearman": _rho(x, target[m]),
                    "kendall": _tau(x, target[m]),
                }
            )
    summary = pd.DataFrame(summary_rows)

    # ---- bootstrap CIs vs the primary anchor (LPIPS) ----
    boot_store = {}
    ci_lo, ci_hi = {}, {}
    for pred in PREDICTORS:
        rho, lo, hi, boots = bootstrap_rho(df[pred].to_numpy(), target[PRIMARY])
        boot_store[pred] = boots
        ci_lo[pred], ci_hi[pred] = lo, hi
    summary["spearman_lo"] = summary.apply(
        lambda r: ci_lo[r["predictor"]] if r["perceptual"] == PRIMARY else np.nan, axis=1
    )
    summary["spearman_hi"] = summary.apply(
        lambda r: ci_hi[r["predictor"]] if r["perceptual"] == PRIMARY else np.nan, axis=1
    )
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved summary: {SUMMARY_CSV}")

    # ---- headline: fuzzy vs PSNR alignment with LPIPS, difference CI ----
    print(f"\n=== Alignment with perceptual quality (-{PRIMARY.upper()}), image level ===")
    sub = summary[summary["perceptual"] == PRIMARY].set_index("predictor")
    for pred in PREDICTORS:
        r = sub.loc[pred]
        print(
            f"  {pred:11s}: Spearman={r['spearman']:+.3f} "
            f"[{r['spearman_lo']:+.3f}, {r['spearman_hi']:+.3f}]  "
            f"Kendall={r['kendall']:+.3f}"
        )
    diff = boot_store["fuzzy_score"] - boot_store["psnr"]
    d_lo, d_hi = np.percentile(diff, [2.5, 97.5])
    print(
        f"\n  fuzzy_score - PSNR (alignment difference): {diff.mean():+.3f} "
        f"95% CI [{d_lo:+.3f}, {d_hi:+.3f}]"
    )
    if d_lo > 0:
        print("  -> Fuzzy tracks perception significantly better than PSNR (CI > 0).")
    elif d_hi < 0:
        print("  -> PSNR tracks perception better than fuzzy here (CI < 0).")
    else:
        print("  -> Difference CI includes 0 (not significant vs PSNR).")

    # ---- figure 1: correct image-level Spearman correlation matrix ----
    corr_cols = ["psnr", "ssim", "entropy", "lpips", "niqe", "brisque", "fuzzy_score"]
    corr_cols = [c for c in corr_cols if c in df.columns]
    mat = df[corr_cols].corr(method="spearman").to_numpy()
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    im = ax.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_cols)))
    ax.set_yticks(range(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=45, ha="right")
    ax.set_yticklabels(corr_cols)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax.text(
                j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                color="white" if abs(mat[i, j]) > 0.5 else "black", fontsize=9,
            )
    ax.set_title(f"Image-level Spearman correlation (n={len(df)})")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Spearman rho")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "imagelevel_correlation_matrix.png", dpi=300)
    plt.close(fig)

    # ---- figure 2: alignment with -LPIPS, bootstrap error bars ----
    sub = summary[summary["perceptual"] == PRIMARY].set_index("predictor").loc[PREDICTORS]
    vals = sub["spearman"].to_numpy()
    err_lo = vals - sub["spearman_lo"].to_numpy()
    err_hi = sub["spearman_hi"].to_numpy() - vals
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.bar(PREDICTORS, vals, yerr=[err_lo, err_hi], capsize=5, edgecolor="black")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel(f"Spearman with perceptual quality (-{PRIMARY.upper()})")
    ax.set_title(f"Which score tracks perception best? (image level, n={len(df)})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "alignment_with_lpips.png", dpi=300)
    plt.close(fig)

    print(f"\nFigures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
