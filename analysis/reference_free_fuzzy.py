"""
Stage B (final) - Reference-free, explainable, calibrated fuzzy perceptual quality.

Place at: analysis/reference_free_fuzzy.py   Run: python -m analysis.reference_free_fuzzy

Builds the paper's main model: a 27-rule fuzzy system over THREE primitive,
human-interpretable NO-REFERENCE features (entropy, contrast, sharpness),
calibrated against perceptual quality (-LPIPS) with image-level 5-fold CV.
It needs no ground truth at inference, yet is compared head-to-head (same folds)
against reference-based baselines.

Compared models (held-out):
  psnr, ssim                      - reference-based single metrics
  entropy, contrast, sharpness    - the no-reference single features
  fr_fuzzy_free                   - calibrated fuzzy on PSNR+SSIM+entropy (Direction A)
  nr_fuzzy_free                   - calibrated fuzzy on the 3 NR features  (MAIN)
  nr_fuzzy_mono                   - same, monotone/interpretable variant

Outputs (results/):
  reference_free_summary.csv          per-model mean+/-std vs LPIPS/NIQE/BRISQUE
  nr_rule_table_mono.csv, nr_rule_table_free.csv
  nr_fuzzy_scores.csv                 all-data refit scores (for later plots/draft)
  figures/reference_free_alignment.png
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

BASE = Path(__file__).resolve().parent.parent
NR_CSV = BASE / "results" / "nr_features.csv"
PC_CSV = BASE / "results" / "perceptual_metrics.csv"
FZ_CSV = BASE / "results" / "fuzzy_enhancement_results.csv"
FIG_DIR = BASE / "results" / "figures"
SUMMARY_CSV = BASE / "results" / "reference_free_summary.csv"
RULE_MONO = BASE / "results" / "nr_rule_table_mono.csv"
RULE_FREE = BASE / "results" / "nr_rule_table_free.csv"
SCORES_CSV = BASE / "results" / "nr_fuzzy_scores.csv"

NR_FEATURES = ["entropy", "contrast", "sharpness"]   # the main no-reference inputs
FR_FEATURES = ["psnr", "ssim", "entropy"]            # Direction-A baseline
K_FOLDS = 5
PRIMARY_LAMBDA = 2.0
DE_MAXITER = 40
DE_POPSIZE = 8
DE_WORKERS = 1
SEED = 42

COMBOS = list(itertools.product(range(3), range(3), range(3)))
IDX = {c: n for n, c in enumerate(COMBOS)}
LEVELS = ["low", "med", "high"]
BANDS = [(25, "Poor"), (50, "Fair"), (75, "Good"), (100.1, "Excellent")]
_PAIRS = []
for (i, j, k) in COMBOS:
    for di, dj, dk in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        h = (i + di, j + dj, k + dk)
        if h in IDX:
            _PAIRS.append((IDX[(i, j, k)], IDX[h]))
PAIRS = np.array(_PAIRS)


def trapmf(x, a, b, c, d):
    x = np.asarray(x, float)
    left = np.where(b > a, (x - a) / np.maximum(b - a, 1e-12), 1.0)
    right = np.where(d > c, (d - x) / np.maximum(d - c, 1e-12), 1.0)
    return np.clip(np.minimum(np.minimum(left, 1.0), right), 0, 1)


def mfset(v):
    q = np.quantile(v, [0, .2, .4, .5, .6, .8, 1])
    return [(q[0], q[0], q[1], q[2]), (q[1], q[2], q[4], q[5]), (q[4], q[5], q[6], q[6])]


def membership(v):
    mf = mfset(v)
    return np.stack([trapmf(v, *mf[l]) for l in range(3)], axis=1)


def orient(df, cols, target):
    """Return feature arrays flipped so that higher = better (positive vs target)."""
    out = []
    for c in cols:
        x = df[c].to_numpy(float)
        s = spearmanr(x, target)[0]
        out.append(-x if s < 0 else x)
    return out


def build_firing(feat_arrays):
    M = [membership(a) for a in feat_arrays]
    fire = np.stack(
        [np.minimum(np.minimum(M[0][:, i], M[1][:, j]), M[2][:, k]) for i, j, k in COMBOS],
        axis=1,
    )
    return fire, fire.sum(axis=1) + 1e-9


def _membership_from_mfs(v, mfs):
    v = np.asarray(v, float)
    return np.stack([trapmf(v, *mfs[l]) for l in range(3)], axis=1)


_SCORER_CACHE = None


def _build_score_features_model():
    for p in (NR_CSV, PC_CSV, SCORES_CSV):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")

    df = (pd.read_csv(NR_CSV)
          .merge(pd.read_csv(PC_CSV)[["filename", "method", "lpips"]],
                 on=["filename", "method"])
          .merge(pd.read_csv(SCORES_CSV)[["filename", "method", "nr_fuzzy_free"]],
                 on=["filename", "method"])
          .dropna().reset_index(drop=True))
    target = -df["lpips"].to_numpy(float)

    signs = []
    mfs = []
    oriented = []
    for c in NR_FEATURES:
        x = df[c].to_numpy(float)
        sign = -1.0 if spearmanr(x, target)[0] < 0 else 1.0
        xo = sign * x
        signs.append(sign)
        mfs.append(mfset(xo))
        oriented.append(xo)

    memberships = [_membership_from_mfs(a, mf) for a, mf in zip(oriented, mfs)]
    fire = np.stack(
        [np.minimum(np.minimum(memberships[0][:, i], memberships[1][:, j]), memberships[2][:, k])
         for i, j, k in COMBOS],
        axis=1,
    )
    design = fire / (fire.sum(axis=1, keepdims=True) + 1e-9)
    theta, *_ = np.linalg.lstsq(design, df["nr_fuzzy_free"].to_numpy(float), rcond=None)
    return np.asarray(signs), mfs, theta


def score_features(entropy, contrast, sharpness):
    """Infer the calibrated all-data NR-fuzzy score for one feature triplet."""
    global _SCORER_CACHE
    if _SCORER_CACHE is None:
        _SCORER_CACHE = _build_score_features_model()
    signs, mfs, theta = _SCORER_CACHE
    vals = signs * np.asarray([entropy, contrast, sharpness], dtype=float)
    memberships = [_membership_from_mfs([vals[i]], mfs[i])[0] for i in range(3)]
    fire = np.array([min(memberships[0][i], memberships[1][j], memberships[2][k])
                     for i, j, k in COMBOS], dtype=float)
    fsum = fire.sum()
    return float((fire @ theta) / (fsum + 1e-9)) if fsum > 1e-12 else 0.0


def rho(x, y):
    r = spearmanr(x, y)[0]
    return 0.0 if np.isnan(r) else float(r)


def violation(theta):
    return float(np.clip(theta[PAIRS[:, 0]] - theta[PAIRS[:, 1]], 0, None).sum())


def calibrate(fire, fsum, target, idx, lam):
    def obj(theta):
        s = (fire[idx] @ theta) / fsum[idx]
        return -rho(s, target[idx]) + lam * violation(theta) / 100.0
    res = differential_evolution(
        obj, bounds=[(0.0, 100.0)] * 27, maxiter=DE_MAXITER, popsize=DE_POPSIZE,
        seed=SEED, tol=1e-3, polish=False, workers=DE_WORKERS, updating="deferred",
    )
    return res.x


def group_kfold(groups, k, seed):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(np.unique(groups)))
    rng.shuffle(uniq)
    for chunk in np.array_split(uniq, k):
        ts = set(chunk.tolist())
        m = np.array([g in ts for g in groups])
        yield np.where(~m)[0], np.where(m)[0]


def band(v):
    for hi, name in BANDS:
        if v < hi:
            return name
    return "Excellent"


def save_rules(theta, feats, path, tag):
    v = np.asarray(theta, float)
    resc = 100 * (v - v.min()) / (v.max() - v.min() + 1e-9)
    rows = [{
        "rule": r + 1, feats[0]: LEVELS[i], feats[1]: LEVELS[j], feats[2]: LEVELS[k],
        "value": round(float(resc[r]), 1), "label": band(resc[r]),
    } for r, (i, j, k) in enumerate(COMBOS)]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  [{tag}] rule table saved (monotonicity violation={violation(theta):.1f}): {path.name}")


def main():
    for p in (NR_CSV, PC_CSV, FZ_CSV):
        if not p.exists():
            sys.exit(f"Missing {p}")
    df = (pd.read_csv(NR_CSV)
          .merge(pd.read_csv(PC_CSV)[["filename", "method", "lpips", "niqe", "brisque"]],
                 on=["filename", "method"])
          .merge(pd.read_csv(FZ_CSV)[["filename", "method", "psnr", "ssim", "fuzzy_score"]],
                 on=["filename", "method"])
          .dropna().reset_index(drop=True))
    print(f"Rows: {len(df)} | images: {df['filename'].nunique()} | folds: {K_FOLDS}")

    groups = df["filename"].to_numpy()
    targets = {"lpips": -df["lpips"].to_numpy(),
               "niqe": -df["niqe"].to_numpy(),
               "brisque": -df["brisque"].to_numpy()}
    cal_target = targets["lpips"]

    nr_feats = orient(df, NR_FEATURES, cal_target)
    fr_feats = orient(df, FR_FEATURES, cal_target)
    nr_fire, nr_fsum = build_firing(nr_feats)
    fr_fire, fr_fsum = build_firing(fr_feats)

    singles = {c: df[c].to_numpy() for c in ["psnr", "ssim", "entropy", "contrast", "sharpness"]}
    singles["fuzzy_handtuned"] = df["fuzzy_score"].to_numpy()

    models = list(singles) + ["fr_fuzzy_free", "nr_fuzzy_free", "nr_fuzzy_mono"]
    per_fold = {m: {t: [] for t in targets} for m in models}

    for f, (tr, te) in enumerate(group_kfold(groups, K_FOLDS, SEED)):
        th_fr = calibrate(fr_fire, fr_fsum, cal_target, tr, 0.0)
        th_nr = calibrate(nr_fire, nr_fsum, cal_target, tr, 0.0)
        th_nm = calibrate(nr_fire, nr_fsum, cal_target, tr, PRIMARY_LAMBDA)
        s_fr = (fr_fire[te] @ th_fr) / fr_fsum[te]
        s_nr = (nr_fire[te] @ th_nr) / nr_fsum[te]
        s_nm = (nr_fire[te] @ th_nm) / nr_fsum[te]
        for t in targets:
            for c, x in singles.items():
                per_fold[c][t].append(rho(x[te], targets[t][te]))
            per_fold["fr_fuzzy_free"][t].append(rho(s_fr, targets[t][te]))
            per_fold["nr_fuzzy_free"][t].append(rho(s_nr, targets[t][te]))
            per_fold["nr_fuzzy_mono"][t].append(rho(s_nm, targets[t][te]))
        print(f"  fold {f}: nr_free={per_fold['nr_fuzzy_free']['lpips'][-1]:.3f}  "
              f"nr_mono={per_fold['nr_fuzzy_mono']['lpips'][-1]:.3f}  "
              f"fr_free={per_fold['fr_fuzzy_free']['lpips'][-1]:.3f}  "
              f"psnr={per_fold['psnr']['lpips'][-1]:.3f}")

    # summary
    rows = []
    for m in models:
        row = {"model": m}
        for t in targets:
            a = np.array(per_fold[m][t])
            row[f"{t}_mean"], row[f"{t}_std"] = round(a.mean(), 3), round(a.std(), 3)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)
    print("\n=== Held-out alignment (mean +/- std over folds) ===")
    print(f"{'model':16s} {'LPIPS':>13} {'NIQE':>13} {'BRISQUE':>13}   (reference-free?)")
    reffree = {"entropy", "contrast", "sharpness", "nr_fuzzy_free", "nr_fuzzy_mono"}
    for _, r in summary.iterrows():
        tag = "YES" if r["model"] in reffree else "no"
        print(f"{r['model']:16s} {r['lpips_mean']:+.3f}+-{r['lpips_std']:.3f} "
              f"{r['niqe_mean']:+.3f}+-{r['niqe_std']:.3f} "
              f"{r['brisque_mean']:+.3f}+-{r['brisque_std']:.3f}    {tag}")
    d = np.array(per_fold["nr_fuzzy_free"]["lpips"]) - np.array(per_fold["psnr"]["lpips"])
    print(f"\n  nr_fuzzy_free - PSNR per fold: {np.round(d,3)}  mean={d.mean():+.3f} "
          f"(all positive: {bool((d>0).all())})")

    # final all-data refit -> rule tables + scores + within-method robustness
    print("\nFinal all-data calibration:")
    th_free = calibrate(nr_fire, nr_fsum, cal_target, np.arange(len(df)), 0.0)
    th_mono = calibrate(nr_fire, nr_fsum, cal_target, np.arange(len(df)), PRIMARY_LAMBDA)
    save_rules(th_mono, NR_FEATURES, RULE_MONO, "mono")
    save_rules(th_free, NR_FEATURES, RULE_FREE, "free")
    score_all = (nr_fire @ th_free) / nr_fsum
    pd.DataFrame({"filename": df["filename"], "method": df["method"],
                  "nr_fuzzy_free": score_all,
                  "nr_fuzzy_mono": (nr_fire @ th_mono) / nr_fsum}).to_csv(SCORES_CSV, index=False)

    print("\nWithin-method Spearman vs -LPIPS (robustness; all-data nr_fuzzy_free):")
    for m in sorted(df["method"].unique()):
        mask = df["method"].to_numpy() == m
        print(f"  {m:8s}: {rho(score_all[mask], cal_target[mask]):+.3f}  (n={mask.sum()})")

    # figure
    order = ["psnr", "ssim", "entropy", "contrast", "sharpness",
             "fuzzy_handtuned", "fr_fuzzy_free", "nr_fuzzy_mono", "nr_fuzzy_free"]
    means = [np.mean(per_fold[m]["lpips"]) for m in order]
    stds = [np.std(per_fold[m]["lpips"]) for m in order]
    colors = ["#4c72b0", "#4c72b0", "#55a868", "#55a868", "#55a868",
              "#4c72b0", "#8172b3", "#dd8452", "#c44e52"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(order, means, yerr=stds, capsize=4, edgecolor="black", color=colors)
    ax.axhline(means[0], color="gray", ls="--", lw=0.9, label="PSNR (reference-based)")
    ax.set_ylabel("Held-out Spearman with -LPIPS")
    ax.set_title(f"Reference-free fuzzy vs reference-based metrics (5-fold CV, n={len(df)})")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "reference_free_alignment.png", dpi=300)
    plt.close(fig)
    print(f"\nFigure saved: {FIG_DIR / 'reference_free_alignment.png'}")


if __name__ == "__main__":
    main()
