"""
RLIE human-MOS evaluation  

Tests the proposed assessor against ACTUAL HUMAN OPINION scores on the RLIE
dataset (1,540 real-world enhanced low-light images, 154 scenes, Bradley-Terry
scores from 103,950 2AFC annotations; ACM MM 2025, github.com/CQUPT-HuBo90/RLIE).

Three evaluations:
  (1) FROZEN TRANSFER     - the LOL-calibrated NR-free model (consequents + MF
                            breakpoints fit on LOL against -LPIPS) applied
                            UNCHANGED to RLIE; SRCC/PLCC vs human MOS. Zero-shot.
  (2) MOS RE-CALIBRATION  - re-calibrate the SAME fuzzy architecture directly
                            against the human MOS, scene-disjoint 5-fold (answers
                            "calibrate to a human anchor, not only LPIPS").
  (3) BASELINES           - Linear / SVR / RF on the same 3 features, MOS-calibrated,
                            same folds. Plus published deep references (MIIHDP, IACA)
                            cited in the paper, not recomputed here.

Reports SRCC, KRCC and PLCC (after the standard 5-param logistic mapping).

=============================  WHAT YOU MUST DO  =============================
1. Download RLIE (Google Drive / Baidu links in the repo README) and extract.
2. Fill in load_rlie() below so it returns a DataFrame with columns:
       img_path  (absolute path to each enhanced image)
       scene     (scene id, 1..154 -- used for scene-disjoint folds)
       mos       (the Bradley-Terry / MOS score, higher = better)
   If RLIE ships an official fold/split file, also return a 'fold' column
   (int 0..4); otherwise scene-disjoint GroupKFold is used automatically.
3. Make sure the LOL result CSVs (nr_features.csv, perceptual_metrics.csv) are
   reachable (RESULTS_DIR or results/), so the frozen LOL model can be built.
Run:  python rlie_human_mos.py  --rlie_root  /path/to/RLIE
=============================================================================
Needs: numpy, pandas, scipy, scikit-learn, opencv-python
"""
from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, differential_evolution
from scipy.stats import kendalltau, pearsonr, spearmanr

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# --------------------------------------------------------------------------- #
NR_FEATURES = ["entropy", "contrast", "sharpness"]
K_FOLDS, DE_MAXITER, DE_POPSIZE, SEED = 5, 40, 8, 42
COMBOS = list(itertools.product(range(3), range(3), range(3)))

BASE = Path(__file__).resolve().parent.parent
_cands = [Path(os.environ["RESULTS_DIR"])] if os.environ.get("RESULTS_DIR") else []
_cands += [BASE / "results", Path(__file__).resolve().parent, Path.cwd()]
RESULTS = next((p for p in _cands if (p / "nr_features.csv").exists()), _cands[0])


# ============================  RLIE loader (finalised)  ==================== #
def load_rlie(root: Path) -> pd.DataFrame:
    """RLIE layout: enhancement/{scene:03d}_{algo}.png (1540), ref/{scene:03d}.png,
    and two score files with columns [Image Name, Score].
    We use normalized_scores.csv (MOS-like scale); bt_scores.csv gives identical
    SRCC/KRCC (monotone transform). No official split file ships in the download,
    so scene-disjoint GroupKFold is used (see main)."""
    score_file = root / "normalized_scores.csv"      # swap to bt_scores.csv if preferred
    sc = pd.read_csv(score_file).rename(columns={"Image Name": "name", "Score": "mos"})
    sc["scene"] = sc["name"].str.slice(0, 3).astype(int)
    sc["img_path"] = sc["name"].apply(lambda nm: str(root / "enhancement" / nm))
    miss = [p for p in sc["img_path"] if not Path(p).exists()]
    if miss:
        print(f"[warn] {len(miss)} image paths not found, e.g. {miss[0]}")
    return sc[["img_path", "scene", "mos"]].reset_index(drop=True)
# =========================================================================== #


# --------------------------- feature extractor ----------------------------- #
# Identical to the manuscript's nr_features (BT.601 luma, RMS contrast,
# 256-bin Shannon entropy, 3x3 Laplacian-variance sharpness, all normalised).
def nr_features_from_path(path: str) -> dict:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY).astype(np.float64)
    contrast = gray.std() / 255.0
    hist = np.bincount(gray.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    p = hist[hist > 0] / hist.sum()
    entropy = float(-(p * np.log2(p)).sum())
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var() / (255.0 ** 2)
    return {"entropy": entropy, "contrast": float(contrast), "sharpness": float(sharpness)}


# ----------------------------- fuzzy machinery ----------------------------- #
def trapmf(x, a, b, c, d):
    x = np.asarray(x, float)
    left = np.where(b > a, (x - a) / np.maximum(b - a, 1e-12), 1.0)
    right = np.where(d > c, (d - x) / np.maximum(d - c, 1e-12), 1.0)
    return np.clip(np.minimum(np.minimum(left, 1.0), right), 0, 1)


def mfset(v):
    q = np.quantile(v, [0, .2, .4, .5, .6, .8, 1])
    return [(q[0], q[0], q[1], q[2]), (q[1], q[2], q[4], q[5]), (q[4], q[5], q[6], q[6])]


def srho(x, y):
    r = spearmanr(x, y)[0]
    return 0.0 if np.isnan(r) else float(r)


def firing(arrs, mfs):
    Ms = [np.stack([trapmf(a, *mfs[n][l]) for l in range(3)], axis=1) for n, a in enumerate(arrs)]
    fire = np.stack([np.minimum(np.minimum(Ms[0][:, i], Ms[1][:, j]), Ms[2][:, k])
                     for i, j, k in COMBOS], axis=1)
    return fire, fire.sum(axis=1) + 1e-9


def calibrate(fire, fsum, target, idx):
    def obj(t):
        s = (fire[idx] @ t) / fsum[idx]
        return -srho(s, target[idx])
    return differential_evolution(obj, bounds=[(0., 100.)] * 27, maxiter=DE_MAXITER,
                                  popsize=DE_POPSIZE, seed=SEED, tol=1e-3, polish=False,
                                  workers=1, updating="deferred").x


def fit_signs(X, target):
    return {c: (spearmanr(X[c].to_numpy(float), target)[0] < 0) for c in NR_FEATURES}


def oriented(X, signs):
    return [(-X[c].to_numpy(float) if signs[c] else X[c].to_numpy(float)) for c in NR_FEATURES]


def group_kfold(groups, k, seed):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(pd.unique(groups))); rng.shuffle(uniq)
    for chunk in np.array_split(uniq, k):
        ts = set(chunk.tolist()); m = np.array([g in ts for g in groups])
        yield np.where(~m)[0], np.where(m)[0]


# ------------------------------- IQA metrics ------------------------------- #
def _logistic5(x, b1, b2, b3, b4, b5):
    return b1 * (0.5 - 1.0 / (1.0 + np.exp(b2 * (x - b3)))) + b4 * x + b5


def iqa_scores(pred, mos):
    pred = np.asarray(pred, float); mos = np.asarray(mos, float)
    srcc = srho(pred, mos)
    krcc = kendalltau(pred, mos)[0]
    try:
        p0 = [np.max(mos) - np.min(mos), 1.0 / (np.std(pred) + 1e-6),
              float(np.mean(pred)), 0.0, float(np.mean(mos))]
        b, _ = curve_fit(_logistic5, pred, mos, p0=p0, maxfev=20000)
        plcc = pearsonr(_logistic5(pred, *b), mos)[0]
    except Exception:
        plcc = pearsonr(pred, mos)[0]
    return float(srcc), float(krcc), float(plcc)


# -------------------------------- pipeline --------------------------------- #
def build_lol_model():
    """Fit NR-free on LOL (full data, vs -LPIPS) -> (signs, mfs, theta)."""
    nf = pd.read_csv(RESULTS / "nr_features.csv")[["filename", "method"] + NR_FEATURES]
    pc = pd.read_csv(RESULTS / "perceptual_metrics.csv")[["filename", "method", "lpips"]]
    lol = nf.merge(pc, on=["filename", "method"]).dropna().reset_index(drop=True)
    tgt = -lol["lpips"].to_numpy()
    signs = fit_signs(lol, tgt)
    arrs = oriented(lol, signs)
    mfs = [mfset(a) for a in arrs]
    fire, fsum = firing(arrs, mfs)
    theta = calibrate(fire, fsum, tgt, np.arange(len(lol)))
    return signs, mfs, theta


def fuzzy_predict(X, signs, mfs, theta):
    arrs = oriented(X, signs)
    fire, fsum = firing(arrs, mfs)
    return (fire @ theta) / fsum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rlie_root", required=True)
    ap.add_argument("--cache", default="rlie_features.csv",
                    help="cache computed features so re-runs are instant")
    args = ap.parse_args()
    root = Path(args.rlie_root)

    # ---- load RLIE + compute our 3 features (cached) ----
    meta = load_rlie(root).reset_index(drop=True)
    cache = RESULTS / args.cache
    if cache.exists():
        feats = pd.read_csv(cache)
    else:
        print(f"[i] computing features for {len(meta)} images...")
        rows = [nr_features_from_path(p) for p in meta["img_path"]]
        feats = pd.concat([meta[["scene", "mos"]].reset_index(drop=True),
                           pd.DataFrame(rows)], axis=1)
        if "fold" in meta.columns:
            feats["fold"] = meta["fold"].values
        feats.to_csv(cache, index=False)
        print(f"[ok] cached -> {cache}")
    mos = feats["mos"].to_numpy(float)
    n = len(feats)
    print(f"[i] RLIE n={n}, scenes={feats['scene'].nunique()}")

    out = []

    # ---- (1) frozen LOL -> RLIE ----
    print("\n=== (1) FROZEN LOL-calibrated model -> RLIE (zero-shot, vs human MOS) ===")
    signs, mfs, theta = build_lol_model()
    pred = fuzzy_predict(feats, signs, mfs, theta)
    s, k, pl = iqa_scores(pred, mos)
    print(f"  NR-free (frozen)   SRCC {s:.3f} | KRCC {k:.3f} | PLCC {pl:.3f}")
    out.append(dict(eval="frozen_transfer", model="NR-free(LOL-frozen)", srcc=s, krcc=k, plcc=pl))

    # ---- folds for RLIE-native calibration ----
    if "fold" in feats.columns:
        folds = [(np.where(feats["fold"].to_numpy() != f)[0],
                  np.where(feats["fold"].to_numpy() == f)[0]) for f in sorted(feats["fold"].unique())]
        print("[i] using official RLIE folds")
    else:
        folds = list(group_kfold(feats["scene"].to_numpy(), K_FOLDS, SEED))
        print("[i] using scene-disjoint GroupKFold (no official split file provided)")

    # ---- (2) RLIE-native MOS calibration: fuzzy ----
    print("\n=== (2) RE-CALIBRATED to human MOS, scene-disjoint CV (held-out) ===")
    preds = np.full(n, np.nan)
    for tr, te in folds:
        sg = fit_signs(feats.iloc[tr], mos[tr])
        arrs = oriented(feats, sg)
        mfs_tr = [mfset(a[tr]) for a in arrs]
        fire, fsum = firing(arrs, mfs_tr)
        th = calibrate(fire, fsum, mos, tr)
        preds[te] = (fire[te] @ th) / fsum[te]
    s, k, pl = iqa_scores(preds, mos)
    print(f"  NR-free (MOS-cal)  SRCC {s:.3f} | KRCC {k:.3f} | PLCC {pl:.3f}")
    out.append(dict(eval="mos_calibrated", model="NR-free(MOS-cal)", srcc=s, krcc=k, plcc=pl))

    # ---- (3) baselines: Linear / SVR / RF, MOS-calibrated, same folds ----
    print("\n=== (3) BASELINES (same 3 features), MOS-calibrated, same folds ===")
    makers = {
        "Linear":       lambda: make_pipeline(StandardScaler(), LinearRegression()),
        "SVR-RBF":      lambda: make_pipeline(StandardScaler(), SVR(C=10.0, gamma="scale")),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
    }
    # use MOS-orientation signs (global) for the feature matrix
    sg = fit_signs(feats, mos)
    X = np.column_stack(oriented(feats, sg))
    for name, mk in makers.items():
        pr = np.full(n, np.nan)
        for tr, te in folds:
            m = mk(); m.fit(X[tr], mos[tr]); pr[te] = m.predict(X[te])
        s, k, pl = iqa_scores(pr, mos)
        print(f"  {name:12s}       SRCC {s:.3f} | KRCC {k:.3f} | PLCC {pl:.3f}")
        out.append(dict(eval="mos_calibrated", model=name, srcc=s, krcc=k, plcc=pl))

    print("\n[note] Deep RLIE-native references (cite, do not recompute): "
          "MIIHDP and IACA report higher SRCC/PLCC -- they are heavy networks "
          "trained on the human labels; our point is lightweight + interpretable + "
          "reference-free human-MOS alignment.")
    pd.DataFrame(out).to_csv(RESULTS / "rlie_human_mos.csv", index=False)
    print(f"[ok] wrote {RESULTS / 'rlie_human_mos.csv'}")


if __name__ == "__main__":
    main()
