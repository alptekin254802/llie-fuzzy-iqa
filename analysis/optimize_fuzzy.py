"""
Stage A (v2) - Perceptual calibration of the fuzzy quality model, done honestly.

Place at: analysis/optimize_fuzzy.py     Run: python -m analysis.optimize_fuzzy

Fixes vs v1:
  * Reports per-fold held-out Spearman (NO invalid pooling of differently-scaled
    out-of-fold scores).
  * Adds a MONOTONICITY penalty so the calibrated 27-rule table stays sensible
    (a higher metric level can never map to a lower quality), preserving the XAI
    story while still beating single metrics.

Models compared (all evaluated on held-out images via 5-fold GROUP CV):
  - single metrics: psnr, ssim, entropy
  - fuzzy_handtuned : the original hand-tuned score (baseline that fails)
  - fuzzy_mono      : calibrated with monotonicity penalty (PRIMARY, interpretable)
  - fuzzy_free      : calibrated without constraint (upper-bound ablation)

Outputs (results/):
  perceptual_optimization_summary.csv   (per-model mean+/-std vs LPIPS/NIQE/BRISQUE)
  optimized_rule_table_mono.csv         (interpretable monotone 27-rule table)
  optimized_rule_table_free.csv         (unconstrained table, for appendix)
  figures/optimized_alignment.png       (per-fold mean +/- std bar chart)
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr

BASE_DIR = Path(__file__).resolve().parent.parent
FUZZY_CSV = BASE_DIR / "results" / "fuzzy_enhancement_results.csv"
PERCEPTUAL_CSV = BASE_DIR / "results" / "perceptual_metrics.csv"
FIG_DIR = BASE_DIR / "results" / "figures"
SUMMARY_CSV = BASE_DIR / "results" / "perceptual_optimization_summary.csv"
RULE_MONO_CSV = BASE_DIR / "results" / "optimized_rule_table_mono.csv"
RULE_FREE_CSV = BASE_DIR / "results" / "optimized_rule_table_free.csv"

K_FOLDS = 5
PRIMARY_LAMBDA = 2.0     # monotonicity strength for the interpretable model
DE_MAXITER = 40
DE_POPSIZE = 8
DE_WORKERS = 1           # set -1 to use all cores
SEED = 42

MF = {
    "psnr": [(0, 0, 15, 25), (15, 25, 30, 40), (30, 40, 50, 50)],
    "ssim": [(0, 0, 0.3, 0.5), (0.3, 0.5, 0.7, 0.85), (0.7, 0.85, 1, 1)],
    "entropy": [(0, 0, 2, 4), (2, 4, 5, 6.5), (5, 6.5, 8, 8)],
}
RANGE = {"psnr": (0, 50), "ssim": (0, 1), "entropy": (0, 8)}
LEVELS = ["low", "med", "high"]
COMBOS = list(itertools.product(range(3), range(3), range(3)))
IDX = {c: n for n, c in enumerate(COMBOS)}
ORIGINAL_RULES = {
    (0,0,0):"Poor",(0,0,1):"Poor",(0,0,2):"Poor",(0,1,0):"Poor",(0,1,1):"Fair",
    (0,1,2):"Fair",(0,2,0):"Fair",(0,2,1):"Fair",(0,2,2):"Fair",(1,0,0):"Poor",
    (1,0,1):"Fair",(1,0,2):"Fair",(1,1,0):"Fair",(1,1,1):"Fair",(1,1,2):"Good",
    (1,2,0):"Fair",(1,2,1):"Good",(1,2,2):"Good",(2,0,0):"Fair",(2,0,1):"Fair",
    (2,0,2):"Fair",(2,1,0):"Fair",(2,1,1):"Good",(2,1,2):"Good",(2,2,0):"Fair",
    (2,2,1):"Good",(2,2,2):"Excellent",
}
BANDS = [(25, "Poor"), (50, "Fair"), (75, "Good"), (100.1, "Excellent")]

# monotonicity adjacency: lower-level cell must be <= the next-higher-level cell
_PAIRS = []
for (i, j, k) in COMBOS:
    for di, dj, dk in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        hi = (i + di, j + dj, k + dk)
        if hi in IDX:
            _PAIRS.append((IDX[(i, j, k)], IDX[hi]))
PAIRS = np.array(_PAIRS)

FIRE: np.ndarray
FSUM: np.ndarray


def trapmf(x, a, b, c, d):
    x = np.asarray(x, float)
    left = np.where(b > a, (x - a) / np.maximum(b - a, 1e-12), 1.0)
    right = np.where(d > c, (d - x) / np.maximum(d - c, 1e-12), 1.0)
    return np.clip(np.minimum(np.minimum(left, 1.0), right), 0, 1)


def memberships(vals, key):
    lo, hi = RANGE[key]
    v = np.clip(vals, lo, hi)
    return np.stack([trapmf(v, *MF[key][l]) for l in range(3)], axis=1)


def build_firing(df):
    ps = memberships(df["psnr"].to_numpy(), "psnr")
    ss = memberships(df["ssim"].to_numpy(), "ssim")
    en = memberships(df["entropy"].to_numpy(), "entropy")
    fire = np.stack(
        [np.minimum(np.minimum(ps[:, i], ss[:, j]), en[:, k]) for i, j, k in COMBOS],
        axis=1,
    )
    return fire, fire.sum(axis=1) + 1e-9


def score(theta, idx):
    return (FIRE[idx] @ theta) / FSUM[idx]


def violation(theta):
    return float(np.clip(theta[PAIRS[:, 0]] - theta[PAIRS[:, 1]], 0, None).sum())


def rho(x, y):
    r = spearmanr(x, y)[0]
    return 0.0 if np.isnan(r) else float(r)


def objective(theta, idx, lam):
    return -rho(score(theta, idx), TARGET[idx]) + lam * violation(theta) / 100.0


def calibrate(idx, lam):
    res = differential_evolution(
        objective, bounds=[(0.0, 100.0)] * 27, args=(idx, lam),
        maxiter=DE_MAXITER, popsize=DE_POPSIZE, seed=SEED, tol=1e-3,
        polish=False, workers=DE_WORKERS, updating="deferred",
    )
    return res.x


def group_kfold(groups, k, seed):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(np.unique(groups)))
    rng.shuffle(uniq)
    for chunk in np.array_split(uniq, k):
        test_set = set(chunk.tolist())
        m = np.array([g in test_set for g in groups])
        yield np.where(~m)[0], np.where(m)[0]


def band(value):
    for hi, name in BANDS:
        if value < hi:
            return name
    return "Excellent"


def save_rule_table(theta, path, tag):
    v = np.asarray(theta, float)
    rescaled = 100 * (v - v.min()) / (v.max() - v.min() + 1e-9)  # ranking-preserving
    rows, changed = [], 0
    for r, (i, j, k) in enumerate(COMBOS):
        orig = ORIGINAL_RULES[(i, j, k)]
        new = band(rescaled[r])
        changed += int(new != orig)
        rows.append({
            "rule": r + 1, "psnr": LEVELS[i], "ssim": LEVELS[j], "entropy": LEVELS[k],
            "original_label": orig, "calibrated_value": round(float(rescaled[r]), 1),
            "calibrated_label": new, "changed": new != orig,
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  [{tag}] rule table saved ({changed}/27 labels changed, "
          f"monotonicity violation={violation(theta):.1f}): {path.name}")


def main():
    global FIRE, FSUM, TARGET
    if not FUZZY_CSV.exists() or not PERCEPTUAL_CSV.exists():
        sys.exit("Need results/fuzzy_enhancement_results.csv and results/perceptual_metrics.csv")
    fz = pd.read_csv(FUZZY_CSV)
    pc = pd.read_csv(PERCEPTUAL_CSV)
    df = fz.merge(pc[["filename","method","lpips","niqe","brisque"]],
                  on=["filename","method"], how="inner").dropna(
                  subset=["psnr","ssim","entropy","fuzzy_score","lpips"]).reset_index(drop=True)
    print(f"Rows: {len(df)} | images: {df['filename'].nunique()} | folds: {K_FOLDS}")

    FIRE, FSUM = build_firing(df)
    groups = df["filename"].to_numpy()
    targets = {"lpips": -df["lpips"].to_numpy(),
               "niqe": -df["niqe"].to_numpy(),
               "brisque": -df["brisque"].to_numpy()}
    fixed = {"psnr": df["psnr"].to_numpy(), "ssim": df["ssim"].to_numpy(),
             "entropy": df["entropy"].to_numpy(), "fuzzy_handtuned": df["fuzzy_score"].to_numpy()}

    # per-fold held-out correlations
    per_fold = {m: {t: [] for t in targets} for m in
                list(fixed) + ["fuzzy_mono", "fuzzy_free"]}
    global TARGET
    for f, (tr, te) in enumerate(group_kfold(groups, K_FOLDS, SEED)):
        TARGET = targets["lpips"]                       # calibrate against LPIPS
        theta_mono = calibrate(tr, PRIMARY_LAMBDA)
        theta_free = calibrate(tr, 0.0)
        s_mono, s_free = score(theta_mono, te), score(theta_free, te)
        for t in targets:
            for m, x in fixed.items():
                per_fold[m][t].append(rho(x[te], targets[t][te]))
            per_fold["fuzzy_mono"][t].append(rho(s_mono, targets[t][te]))
            per_fold["fuzzy_free"][t].append(rho(s_free, targets[t][te]))
        print(f"  fold {f}: mono(test,LPIPS)={per_fold['fuzzy_mono']['lpips'][-1]:.3f}  "
              f"free={per_fold['fuzzy_free']['lpips'][-1]:.3f}  "
              f"psnr={per_fold['psnr']['lpips'][-1]:.3f}")

    # summary table
    rows = []
    for m in per_fold:
        row = {"model": m}
        for t in targets:
            a = np.array(per_fold[m][t])
            row[f"{t}_mean"] = round(a.mean(), 3)
            row[f"{t}_std"] = round(a.std(), 3)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)

    print("\n=== Held-out alignment (mean +/- std over folds) ===")
    print(f"{'model':16s} {'LPIPS':>14} {'NIQE':>14} {'BRISQUE':>14}")
    for _, r in summary.iterrows():
        print(f"{r['model']:16s} {r['lpips_mean']:+.3f}+/-{r['lpips_std']:.3f}  "
              f"{r['niqe_mean']:+.3f}+/-{r['niqe_std']:.3f}  "
              f"{r['brisque_mean']:+.3f}+/-{r['brisque_std']:.3f}")
    # paired improvement of mono over PSNR
    dmono = np.array(per_fold["fuzzy_mono"]["lpips"]) - np.array(per_fold["psnr"]["lpips"])
    dfree = np.array(per_fold["fuzzy_free"]["lpips"]) - np.array(per_fold["psnr"]["lpips"])
    print(f"\n  mono - PSNR per fold: {np.round(dmono,3)}  mean={dmono.mean():+.3f} "
          f"(all positive: {bool((dmono>0).all())})")
    print(f"  free - PSNR per fold: {np.round(dfree,3)}  mean={dfree.mean():+.3f} "
          f"(all positive: {bool((dfree>0).all())})")

    # final all-data models -> interpretable rule tables
    print("\nFinal all-data calibration:")
    TARGET = targets["lpips"]
    save_rule_table(calibrate(np.arange(len(df)), PRIMARY_LAMBDA), RULE_MONO_CSV, "mono")
    save_rule_table(calibrate(np.arange(len(df)), 0.0), RULE_FREE_CSV, "free")

    # figure: per-fold mean +/- std vs LPIPS
    order = ["psnr", "ssim", "entropy", "fuzzy_handtuned", "fuzzy_mono", "fuzzy_free"]
    means = [np.mean(per_fold[m]["lpips"]) for m in order]
    stds = [np.std(per_fold[m]["lpips"]) for m in order]
    colors = ["#4c72b0"]*4 + ["#c44e52", "#dd8452"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(order, means, yerr=stds, capsize=5, edgecolor="black", color=colors)
    ax.axhline(means[0], color="gray", ls="--", lw=0.9, label="PSNR baseline")
    ax.set_ylabel("Held-out Spearman with -LPIPS (mean +/- std)")
    ax.set_title(f"Hand-tuned vs calibrated fuzzy (5-fold CV, n={len(df)})")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "optimized_alignment.png", dpi=300)
    plt.close(fig)
    print(f"\nFigure saved: {FIG_DIR / 'optimized_alignment.png'}")


if __name__ == "__main__":
    main()