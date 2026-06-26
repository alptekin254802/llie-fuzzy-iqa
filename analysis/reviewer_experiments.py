"""
Reviewer-response experiments  -  SELF-CONTAINED.

  (A) LEAKAGE FIX     - membership-function (MF) breakpoints computed PER TRAIN
                        FOLD ONLY vs. the manuscript's global quantiles. Shows
                        the held-out numbers barely move -> no meaningful leakage.
  (B) BASELINES       - same three oriented features fed to plain learned
                        regressors (Linear, SVR-RBF, RandomForest), same group
                        CV. Isolates "fuzzy inference" vs. "any learned fusion".
  (C) ALGO-DISJOINT   - leave-one-enhancer-out: calibrate on the other methods,
                        test on the held-out method (+ deep Zero-DCE / SCI as
                        fully held-out enhancers). Tests robustness to the
                        algorithm/style bias the reviewer raised.
  (D) SEED STABILITY  - NR-free held-out mean rho across 5 random seeds.

Run from the folder that contains your result CSVs, or set RESULTS_DIR:
    python reviewer_experiments.py
Inputs (same as make_figures.py):
    nr_features.csv, perceptual_metrics.csv, modern_nriqa.csv,
    fuzzy_enhancement_results.csv, deep_zerodce_all.csv, deep_sci_all.csv
Outputs:
    reviewer_experiments.csv   (+ a printed summary)
Needs: numpy, pandas, scipy, scikit-learn
"""
from __future__ import annotations

import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# --------------------------------------------------------------------------- #
# paths / constants  (mirrors make_figures.py)
# --------------------------------------------------------------------------- #
BASE = Path(__file__).resolve().parent.parent
# look in results/ first, then alongside the script, then cwd
_cands = [Path(os.environ["RESULTS_DIR"])] if os.environ.get("RESULTS_DIR") else []
_cands += [BASE / "results", Path(__file__).resolve().parent, Path.cwd()]
RESULTS = next((p for p in _cands if (p / "nr_features.csv").exists()), _cands[0])
print(f"[i] reading CSVs from: {RESULTS}")

NR_FEATURES = ["entropy", "contrast", "sharpness"]
FR_FEATURES = ["psnr", "ssim", "entropy"]
K_FOLDS, PRIMARY_LAMBDA, DE_MAXITER, DE_POPSIZE, SEED = 5, 2.0, 40, 8, 42
SEEDS = [42, 1, 7, 123, 2024]
COMBOS = list(itertools.product(range(3), range(3), range(3)))
IDX = {c: n for n, c in enumerate(COMBOS)}
_P = []
for (i, j, k) in COMBOS:
    for di, dj, dk in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        h = (i + di, j + dj, k + dk)
        if h in IDX:
            _P.append((IDX[(i, j, k)], IDX[h]))
PAIRS = np.array(_P)


# --------------------------------------------------------------------------- #
# fuzzy machinery  (verbatim logic from make_figures.py)
# --------------------------------------------------------------------------- #
def trapmf(x, a, b, c, d):
    x = np.asarray(x, float)
    left = np.where(b > a, (x - a) / np.maximum(b - a, 1e-12), 1.0)
    right = np.where(d > c, (d - x) / np.maximum(d - c, 1e-12), 1.0)
    return np.clip(np.minimum(np.minimum(left, 1.0), right), 0, 1)


def mfset(v):
    q = np.quantile(v, [0, .2, .4, .5, .6, .8, 1])
    return [(q[0], q[0], q[1], q[2]), (q[1], q[2], q[4], q[5]), (q[4], q[5], q[6], q[6])]


def rho(x, y):
    r = spearmanr(x, y)[0]
    return 0.0 if np.isnan(r) else float(r)


def violation(t):
    return float(np.clip(t[PAIRS[:, 0]] - t[PAIRS[:, 1]], 0, None).sum())


def orient_arrays(df, cols, target, sign_idx=None):
    """Return oriented feature arrays for ALL rows. Flip signs are derived from
    rows `sign_idx` only (train fold) when given, else from all rows."""
    s = np.arange(len(df)) if sign_idx is None else sign_idx
    arrs, signs = [], {}
    for c in cols:
        x = df[c].to_numpy(float)
        flip = spearmanr(x[s], target[s])[0] < 0
        signs[c] = flip
        arrs.append(-x if flip else x)
    return arrs, signs


def firing(arrs, mfs):
    Ms = [np.stack([trapmf(a, *mfs[n][l]) for l in range(3)], axis=1) for n, a in enumerate(arrs)]
    fire = np.stack([np.minimum(np.minimum(Ms[0][:, i], Ms[1][:, j]), Ms[2][:, k])
                     for i, j, k in COMBOS], axis=1)
    return fire, fire.sum(axis=1) + 1e-9


def mfs_from(arrs, idx=None):
    """MF breakpoints from rows `idx` only (train fold) or all rows."""
    return [mfset(a if idx is None else a[idx]) for a in arrs]


def calibrate(fire, fsum, target, idx, lam, seed=SEED):
    def obj(t):
        s = (fire[idx] @ t) / fsum[idx]
        return -rho(s, target[idx]) + lam * violation(t) / 100.0
    return differential_evolution(obj, bounds=[(0., 100.)] * 27, maxiter=DE_MAXITER,
                                  popsize=DE_POPSIZE, seed=seed, tol=1e-3, polish=False,
                                  workers=1, updating="deferred").x


def group_kfold(groups, k, seed):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(np.unique(groups))); rng.shuffle(uniq)
    for chunk in np.array_split(uniq, k):
        ts = set(chunk.tolist()); m = np.array([g in ts for g in groups])
        yield np.where(~m)[0], np.where(m)[0]


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_data():
    nf = pd.read_csv(RESULTS / "nr_features.csv")[["filename", "method", "entropy", "contrast", "sharpness"]]
    pc = pd.read_csv(RESULTS / "perceptual_metrics.csv")[["filename", "method", "lpips", "niqe", "brisque"]]
    mo = pd.read_csv(RESULTS / "modern_nriqa.csv")[["filename", "method", "musiq", "maniqa", "clipiqa"]]
    fz = pd.read_csv(RESULTS / "fuzzy_enhancement_results.csv")[["filename", "method", "psnr", "ssim", "fuzzy_score"]]
    classical = (nf.merge(pc, on=["filename", "method"]).merge(mo, on=["filename", "method"])
                 .merge(fz, on=["filename", "method"]).dropna().reset_index(drop=True))
    cols = ["filename", "method", "entropy", "contrast", "sharpness", "lpips", "niqe",
            "brisque", "musiq", "maniqa", "clipiqa", "psnr", "ssim"]
    dz = pd.read_csv(RESULTS / "deep_zerodce_all.csv")[cols]
    ds = pd.read_csv(RESULTS / "deep_sci_all.csv")[cols]
    return classical, cols, dz, ds


# --------------------------------------------------------------------------- #
# (A) leakage: global MF vs per-fold MF
# --------------------------------------------------------------------------- #
def fuzzy_cv(df, feats, lam, per_fold_mf, seed=SEED):
    """Held-out Spearman vs -LPIPS over group folds.
    per_fold_mf=False -> global MF + global orientation (manuscript setting).
    per_fold_mf=True  -> MF breakpoints AND orientation signs from train fold only.
    """
    g = df["filename"].to_numpy()
    tgt = -df["lpips"].to_numpy()
    out = []
    if not per_fold_mf:
        arrs, _ = orient_arrays(df, feats, tgt)            # global orientation
        fire, fsum = firing(arrs, mfs_from(arrs))          # global MF
    for tr, te in group_kfold(g, K_FOLDS, seed):
        if per_fold_mf:
            arrs, _ = orient_arrays(df, feats, tgt, sign_idx=tr)   # train-only signs
            fire, fsum = firing(arrs, mfs_from(arrs, idx=tr))      # train-only MF
        th = calibrate(fire, fsum, tgt, tr, lam, seed=seed)
        s = (fire[te] @ th) / fsum[te]
        out.append(rho(s, tgt[te]))
    return float(np.mean(out)), float(np.std(out))


def exp_leakage(classical):
    print("\n=== (A) LEAKAGE: global MF vs per-fold MF (held-out rho vs -LPIPS) ===")
    rows = []
    specs = [("FR-calibrated", FR_FEATURES, 0.0),
             ("NR-free", NR_FEATURES, 0.0),
             ("NR-mono", NR_FEATURES, PRIMARY_LAMBDA)]
    for name, feats, lam in specs:
        gm, gs = fuzzy_cv(classical, feats, lam, per_fold_mf=False)
        pm, ps = fuzzy_cv(classical, feats, lam, per_fold_mf=True)
        d = pm - gm
        print(f"  {name:14s} global {gm:.3f}+-{gs:.3f} | per-fold {pm:.3f}+-{ps:.3f} | delta {d:+.3f}")
        rows.append(dict(experiment="leakage", model=name, global_mean=gm, global_std=gs,
                         perfold_mean=pm, perfold_std=ps, delta=d))
    return rows


# --------------------------------------------------------------------------- #
# (B) lightweight non-fuzzy baselines, same 3 oriented features
# --------------------------------------------------------------------------- #
def reg_cv(df, feats, make_model, seed=SEED):
    g = df["filename"].to_numpy()
    tgt = -df["lpips"].to_numpy(); niqe = -df["niqe"].to_numpy()
    L, N = [], []
    arrs, _ = orient_arrays(df, feats, tgt)            # same orientation as fuzzy
    X = np.column_stack(arrs)
    for tr, te in group_kfold(g, K_FOLDS, seed):
        m = make_model()
        m.fit(X[tr], tgt[tr])
        p = m.predict(X[te])
        L.append(rho(p, tgt[te])); N.append(rho(p, niqe[te]))
    return (float(np.mean(L)), float(np.std(L)), float(np.mean(N)), float(np.std(N)))


def exp_baselines(classical):
    print("\n=== (B) BASELINES on the same 3 oriented features (held-out rho) ===")
    models = {
        "Linear":       lambda: make_pipeline(StandardScaler(), LinearRegression()),
        "SVR-RBF":      lambda: make_pipeline(StandardScaler(), SVR(C=10.0, gamma="scale")),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
    }
    rows = []
    # fuzzy reference (global MF, matches manuscript)
    fm, fs = fuzzy_cv(classical, NR_FEATURES, 0.0, per_fold_mf=False)
    print(f"  {'NR-free (fuzzy)':16s} vs-LPIPS {fm:.3f}+-{fs:.3f}   (manuscript reference)")
    rows.append(dict(experiment="baseline", model="NR-free (fuzzy)",
                     lpips_mean=fm, lpips_std=fs, niqe_mean=np.nan, niqe_std=np.nan))
    for name, mk in models.items():
        lm, ls, nm, ns = reg_cv(classical, NR_FEATURES, mk)
        print(f"  {name:16s} vs-LPIPS {lm:.3f}+-{ls:.3f} | vs-NIQE {nm:.3f}+-{ns:.3f}")
        rows.append(dict(experiment="baseline", model=name,
                         lpips_mean=lm, lpips_std=ls, niqe_mean=nm, niqe_std=ns))
    return rows


# --------------------------------------------------------------------------- #
# (C) algorithm-disjoint: leave-one-enhancer-out  (+ deep held-out)
# --------------------------------------------------------------------------- #
def fit_score_disjoint(train_df, test_df, feats=NR_FEATURES, lam=0.0, seed=SEED):
    """Calibrate on train_df (train-only MF + orientation), score test_df."""
    tgt_tr = -train_df["lpips"].to_numpy()
    arr_tr, signs = orient_arrays(train_df, feats, tgt_tr)
    mfs = mfs_from(arr_tr)
    fire_tr, fsum_tr = firing(arr_tr, mfs)
    th = calibrate(fire_tr, fsum_tr, tgt_tr, np.arange(len(train_df)), lam, seed=seed)
    # apply SAME signs + MFs to test
    arr_te = [(-test_df[c].to_numpy(float) if signs[c] else test_df[c].to_numpy(float)) for c in feats]
    fire_te, fsum_te = firing(arr_te, mfs)
    s = (fire_te @ th) / fsum_te
    return rho(s, -test_df["lpips"].to_numpy())


def exp_algo_disjoint(classical, dz, ds):
    print("\n=== (C) ALGORITHM-DISJOINT (calibrate on other methods, test on held-out) ===")
    rows = []
    methods = sorted(classical["method"].unique())
    for m in methods:
        tr = classical[classical["method"] != m].reset_index(drop=True)
        te = classical[classical["method"] == m].reset_index(drop=True)
        r = fit_score_disjoint(tr, te)
        print(f"  leave-out {m:10s} -> held-out rho {r:.3f}  (n={len(te)})")
        rows.append(dict(experiment="algo_disjoint", held_out=m, rho=r, n=len(te)))
    # deep enhancers fully held out: train on ALL classical, test on each deep set
    for name, dfd in [("zerodce", dz), ("sci", ds)]:
        r = fit_score_disjoint(classical, dfd.dropna().reset_index(drop=True))
        print(f"  deep held-out {name:7s} -> held-out rho {r:.3f}  (n={len(dfd)})")
        rows.append(dict(experiment="algo_disjoint", held_out=name, rho=r, n=len(dfd)))
    return rows


# --------------------------------------------------------------------------- #
# (D) seed stability
# --------------------------------------------------------------------------- #
def exp_seed_stability(classical):
    print("\n=== (D) SEED STABILITY: NR-free held-out mean rho across seeds ===")
    vals = []
    for sd in SEEDS:
        mean, _ = fuzzy_cv(classical, NR_FEATURES, 0.0, per_fold_mf=False, seed=sd)
        vals.append(mean)
        print(f"  seed {sd:5d} -> {mean:.3f}")
    vals = np.array(vals)
    spread = float(vals.max() - vals.min())
    std = float(vals.std())
    print(f"  => mean {vals.mean():.3f}, std {std:.3f}, max-min spread {spread:.3f}")
    print(f"  => mean {vals.mean():.3f}, std {std:.3f}, max-min spread {spread:.3f}")
    return [dict(experiment="seed_stability", seed=int(s), mean_rho=float(v)) for s, v in zip(SEEDS, vals)] + \
           [dict(experiment="seed_stability", seed="summary", mean_rho=float(vals.mean()),
                 std=std, spread=spread)]


# --------------------------------------------------------------------------- #
def main():
    classical, cols, dz, ds = load_data()
    print(f"[i] classical n={len(classical)}, zerodce n={len(dz)}, sci n={len(ds)}")
    rows = []
    rows += exp_leakage(classical)
    rows += exp_baselines(classical)
    rows += exp_algo_disjoint(classical, dz, ds)
    rows += exp_seed_stability(classical)
    out = RESULTS / "reviewer_experiments.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n[ok] wrote {out}")
    print("[ok] reviewer_experiments.csv contains the robustness and baseline summaries.")


if __name__ == "__main__":
    main()
