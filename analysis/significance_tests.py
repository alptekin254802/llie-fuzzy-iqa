"""Paired scene-cluster bootstrap of fixed out-of-fold predictions.

Intervals are conditional on the fitted models. Resample complete scenes, keeping
all five enhanced outputs together. Report pooled and fold-mean SRCC separately;
Holm adjustment covers the eight comparisons per anchor and estimand.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

RESULTS = Path(__file__).resolve().parents[1] / "results"
COMPARATORS = {"MUSIQ": "musiq", "MANIQA": "maniqa", "CLIP-IQA": "clipiqa", "CLIP-IQA+": "clipiqa_plus",
               "PSNR": "psnr", "SSIM": "ssim", "Linear": "oof_linear", "RF": "oof_rf"}


def rho(a, b):
    a, b = rankdata(a), rankdata(b)
    a -= a.mean(); b -= b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den else 0.0


def paired_bootstrap(a, b, target, groups, fold, replicates=2000, seed=20261018):
    a, b, y, groups, fold = map(np.asarray, (a, b, target, groups, fold))
    ids = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    folds = np.unique(fold)
    rng = np.random.default_rng(seed)
    def estimate(ii):
        return [rho(a[ii], y[ii]) - rho(b[ii], y[ii]),
                float(np.mean([rho(a[ii[fold[ii] == f]], y[ii[fold[ii] == f]]) -
                               rho(b[ii[fold[ii] == f]], y[ii[fold[ii] == f]]) for f in folds]))]
    observed = estimate(np.arange(len(y)))
    draws = np.array([estimate(np.concatenate([ids[j] for j in rng.integers(len(ids), size=len(ids))]))
                      for _ in range(replicates)])
    result = {}
    for i, key in enumerate(("pooled", "fold_mean")):
        lo, hi = np.quantile(draws[:, i], [.025, .975])
        result[key] = {"difference": observed[i], "ci95": [float(lo), float(hi)],
                       "p_two_sided_null_centered": float((1 + np.count_nonzero(np.abs(draws[:, i] - observed[i]) >= abs(observed[i]))) / (replicates + 1)),
                       "excludes_zero": bool(lo > 0 or hi < 0), "replicates": replicates, "seed": seed}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--output", type=Path, default=RESULTS / "significance_tests.json")
    args = ap.parse_args()
    if args.replicates < 1:
        ap.error("replicates must be positive")
    d = pd.read_csv(RESULTS / "validation_predictions.csv")
    for name in ("modern_nriqa.csv", "clipiqa_plus_scores.csv"):
        d = d.merge(pd.read_csv(RESULTS / name), on=["filename", "method"], how="left", validate="one_to_one")
    columns = list(COMPARATORS.values()) + ["lpips", "niqe", "oof_nr_free"]
    if len(d) != 2425 or d[columns].isna().any().any() or d.filename.nunique() != 485:
        raise ValueError("Expected all 2425 predictions from 485 complete scenes")
    result = {"n_images": len(d), "n_scenes": d.filename.nunique(), "anchors": {}}
    for anchor in ("lpips", "niqe"):
        values = {}
        for name, col in COMPARATORS.items():
            values[name] = paired_bootstrap(d.oof_nr_free, d[col], -d[anchor], d.filename, d.fold, args.replicates)
        for estimand in ("pooled", "fold_mean"):
            order = sorted(values, key=lambda n: values[n][estimand]["p_two_sided_null_centered"])
            last = 0
            for i, name in enumerate(order):
                last = max(last, min(1, (len(order)-i) * values[name][estimand]["p_two_sided_null_centered"]))
                values[name][estimand]["p_holm_8"] = last
        result["anchors"][anchor] = values
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
