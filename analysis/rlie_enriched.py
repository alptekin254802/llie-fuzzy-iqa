"""
Enriched-feature test  -  does adding colour / brightness / noise
to the three primitive features improve alignment, in-distribution (LOL vs
-LPIPS) and against real human MOS (RLIE)?

Clean instrument = learned regressors (Linear / SVR / RF). The fuzzy system is
NOT extended to 6 inputs (that is 3^6 = 729 rules, defeating interpretability);
the regressor 3-vs-6 contrast isolates feature sufficiency.

Two parts:
  PART A  LOL,  target -LPIPS,  group-CV by source image          (CSV only)
  PART B  RLIE, target human MOS, scene-disjoint 5-fold           (needs images)

Feature sets:
  3 = entropy, contrast, sharpness            (the paper's set)
  6 = + brightness, colorfulness, noise

Run:
  python rlie_enriched.py                          # Part A only (LOL)
  python rlie_enriched.py --rlie_root <RLIE_ROOT>  # Part A + Part B
Needs: numpy, pandas, scipy, scikit-learn, opencv-python
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import kendalltau, pearsonr, spearmanr

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

F3 = ["entropy", "contrast", "sharpness"]
F6 = ["entropy", "contrast", "sharpness", "brightness", "colorfulness", "noise"]
K_FOLDS, SEED = 5, 42

BASE = Path(__file__).resolve().parent.parent
_cands = [Path(os.environ["RESULTS_DIR"])] if os.environ.get("RESULTS_DIR") else []
_cands += [BASE / "results", Path(__file__).resolve().parent, Path.cwd()]
RESULTS = next((p for p in _cands if (p / "nr_features.csv").exists()), _cands[0])


def load_rlie(root: Path) -> pd.DataFrame:
    sc = pd.read_csv(root / "normalized_scores.csv").rename(
        columns={"Image Name": "name", "Score": "mos"})
    sc["scene"] = sc["name"].str.slice(0, 3).astype(int)
    sc["img_path"] = sc["name"].apply(lambda nm: str(root / "enhancement" / nm))
    return sc[["img_path", "scene", "mos"]].reset_index(drop=True)


def all6_from_path(path: str) -> dict:
    """Six NR features, identical definitions to the manuscript extractor."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY).astype(np.float64)
    brightness = gray.mean() / 255.0
    contrast = gray.std() / 255.0
    hist = np.bincount(gray.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    p = hist[hist > 0] / hist.sum()
    entropy = float(-(p * np.log2(p)).sum())
    R, G, B = rgb8[..., 0].astype(np.float64), rgb8[..., 1].astype(np.float64), rgb8[..., 2].astype(np.float64)
    rg, yb = R - G, 0.5 * (R + G) - B
    colorfulness = (np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                    + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)) / 255.0
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var() / (255.0 ** 2)
    Kmask = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = cv2.filter2D(gray, cv2.CV_64F, Kmask, borderType=cv2.BORDER_REPLICATE)
    H, W = gray.shape
    noise = (np.sum(np.abs(conv)) * np.sqrt(0.5 * np.pi)
             / (6.0 * max(W - 2, 1) * max(H - 2, 1))) / 255.0
    return {"entropy": entropy, "contrast": float(contrast), "sharpness": float(sharpness),
            "brightness": float(brightness), "colorfulness": float(colorfulness), "noise": float(noise)}


def srho(x, y):
    r = spearmanr(x, y)[0]
    return 0.0 if np.isnan(r) else float(r)


def group_kfold(groups, k, seed):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(pd.unique(groups))); rng.shuffle(uniq)
    for chunk in np.array_split(uniq, k):
        ts = set(chunk.tolist()); m = np.array([g in ts for g in groups])
        yield np.where(~m)[0], np.where(m)[0]


def _logistic5(x, b1, b2, b3, b4, b5):
    return b1 * (0.5 - 1.0 / (1.0 + np.exp(b2 * (x - b3)))) + b4 * x + b5


def iqa_scores(pred, y):
    pred = np.asarray(pred, float); y = np.asarray(y, float)
    s = srho(pred, y); k = kendalltau(pred, y)[0]
    try:
        p0 = [y.max() - y.min(), 1.0 / (pred.std() + 1e-6), float(pred.mean()), 0.0, float(y.mean())]
        b, _ = curve_fit(_logistic5, pred, y, p0=p0, maxfev=20000)
        pl = pearsonr(_logistic5(pred, *b), y)[0]
    except Exception:
        pl = pearsonr(pred, y)[0]
    return float(s), float(k), float(pl)


MAKERS = {
    "Linear":       lambda: make_pipeline(StandardScaler(), LinearRegression()),
    "SVR-RBF":      lambda: make_pipeline(StandardScaler(), SVR(C=10.0, gamma="scale")),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
}


def regressor_cv(df, feats, target, folds):
    X = df[feats].to_numpy(float); n = len(df)
    res = {}
    for name, mk in MAKERS.items():
        pr = np.full(n, np.nan)
        for tr, te in folds:
            m = mk(); m.fit(X[tr], target[tr]); pr[te] = m.predict(X[te])
        res[name] = iqa_scores(pr, target)
    return res


def report(title, df, target, folds, fuzzy3_ref):
    print(f"\n=== {title} ===")
    print(f"  {'fuzzy NR-free (3 feat)':22s}  SRCC {fuzzy3_ref}")
    rows = []
    for label, feats in [("3 feat", F3), ("6 feat", F6)]:
        res = regressor_cv(df, feats, target, folds)
        for name, (s, k, pl) in res.items():
            print(f"  {name+' ('+label+')':22s}  SRCC {s:.3f} | KRCC {k:.3f} | PLCC {pl:.3f}")
            rows.append(dict(dataset=title, model=name, feats=label, srcc=s, krcc=k, plcc=pl))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rlie_root", default=None)
    args = ap.parse_args()
    out = []

    # ---------- PART A: LOL, target -LPIPS ----------
    nf = pd.read_csv(RESULTS / "nr_features.csv")
    pc = pd.read_csv(RESULTS / "perceptual_metrics.csv")[["filename", "method", "lpips"]]
    lol = nf.merge(pc, on=["filename", "method"]).dropna(subset=F6 + ["lpips"]).reset_index(drop=True)
    tgtL = -lol["lpips"].to_numpy()
    foldsL = list(group_kfold(lol["filename"].to_numpy(), K_FOLDS, SEED))
    out += report("LOL (vs -LPIPS, in-distribution)", lol, tgtL, foldsL, fuzzy3_ref="0.753")

    # ---------- PART B: RLIE, target human MOS ----------
    if args.rlie_root:
        root = Path(args.rlie_root)
        cache = RESULTS / "rlie_features6.csv"
        if cache.exists():
            r = pd.read_csv(cache)
        else:
            meta = load_rlie(root)
            print(f"\n[i] computing 6 features for {len(meta)} RLIE images...")
            rows = [all6_from_path(p) for p in meta["img_path"]]
            r = pd.concat([meta[["scene", "mos"]].reset_index(drop=True), pd.DataFrame(rows)], axis=1)
            r.to_csv(cache, index=False); print(f"[ok] cached -> {cache}")
        tgtR = r["mos"].to_numpy(float)
        foldsR = list(group_kfold(r["scene"].to_numpy(), K_FOLDS, SEED))
        out += report("RLIE (vs human MOS, scene-disjoint)", r, tgtR, foldsR, fuzzy3_ref="0.267")
    else:
        print("\n[i] no --rlie_root given: ran Part A (LOL) only. "
              "Add --rlie_root to run the RLIE human-MOS test.")

    pd.DataFrame(out).to_csv(RESULTS / "enriched_feature_test.csv", index=False)
    print(f"\n[ok] wrote {RESULTS / 'enriched_feature_test.csv'}")


if __name__ == "__main__":
    main()
