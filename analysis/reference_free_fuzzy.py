"""Fit the reference-free fuzzy models and evaluate five scene-disjoint folds.

NR-free uses entropy, normalized contrast and normalized Laplacian variance
with 27 singleton rules calibrated to negative LPIPS. NR-mono adds a
monotonicity constraint. Reference-based and single-feature baselines use
the same held-out scenes. Deployment fits use all calibration scenes.

Run ``python -m analysis.reference_free_fuzzy --output calibration_run``.
The output directory receives scores, summaries, rule tables and model files."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr
from analysis.monotone_fuzzy import fit as fit_monotone, design as product_design
from evaluation.quality_labels import quality_label

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
FR_FEATURES = ["psnr", "ssim", "entropy"]            # reference-based baseline
K_FOLDS = 5
DE_MAXITER = 40
DE_POPSIZE = 8
DE_WORKERS = 1
SEED = 42

COMBOS = list(itertools.product(range(3), range(3), range(3)))
IDX = {c: n for n, c in enumerate(COMBOS)}
LEVELS = ["low", "med", "high"]
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
    from evaluation.frozen_scorer import FrozenScorer
    model = FrozenScorer("nr_free")
    return model.signs, model.mfs, model.theta


def score_features(entropy, contrast, sharpness):
    """Infer the frozen all-scene deployment model without fitting at runtime."""
    global _SCORER_CACHE
    if _SCORER_CACHE is None:
        from evaluation.frozen_scorer import FrozenScorer
        _SCORER_CACHE = FrozenScorer("nr_free")
    return _SCORER_CACHE((entropy, contrast, sharpness))


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
    return quality_label(v)


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
    import argparse
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('calibration_run'))
    output=parser.parse_args().output.resolve()
    if output==BASE.resolve() or output.is_relative_to((BASE/'results').resolve()) or output.is_relative_to((BASE/'models').resolve()):
        raise ValueError('Choose a separate output directory for a new calibration')
    output.mkdir(parents=True,exist_ok=True);(output/'models').mkdir(exist_ok=True)
    global SUMMARY_CSV,RULE_MONO,RULE_FREE,SCORES_CSV
    SUMMARY_CSV=output/'reference_free_summary.csv';RULE_MONO=output/'nr_rule_table_mono.csv';RULE_FREE=output/'nr_rule_table_free.csv';SCORES_CSV=output/'nr_fuzzy_scores.csv'

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
        mono_mfs = [mfset(a[tr]) for a in nr_feats]
        th_nm = fit_monotone(np.column_stack(nr_feats)[tr], cal_target[tr], mono_mfs)
        s_fr = (fr_fire[te] @ th_fr) / fr_fsum[te]
        s_nr = (nr_fire[te] @ th_nr) / nr_fsum[te]
        s_nm = product_design(np.column_stack(nr_feats)[te], mono_mfs) @ th_nm
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
    mono_mfs = [mfset(a) for a in nr_feats]
    th_mono = fit_monotone(np.column_stack(nr_feats), cal_target, mono_mfs)
    for model_name, theta_values, norm in (("nr_free", th_free, "minimum"), ("nr_mono", th_mono, "product")):
        model_path = BASE / "models" / f"{model_name}.json"
        metadata = json.loads(model_path.read_text())
        metadata.update(theta=theta_values.tolist(), mfs=mono_mfs,
                        signs={c: bool(spearmanr(df[c], cal_target)[0] < 0) for c in NR_FEATURES}, t_norm=norm)
        (output / "models" / f"{model_name}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    save_rules(th_mono, NR_FEATURES, RULE_MONO, "mono")
    save_rules(th_free, NR_FEATURES, RULE_FREE, "free")
    score_all = (nr_fire @ th_free) / nr_fsum
    pd.DataFrame({"filename": df["filename"], "method": df["method"],
                  "nr_fuzzy_free": score_all,
                  "nr_fuzzy_mono": product_design(np.column_stack(nr_feats), mono_mfs) @ th_mono}).to_csv(SCORES_CSV, index=False)

    print("\nWithin-method Spearman vs -LPIPS (robustness; all-data nr_fuzzy_free):")
    for m in sorted(df["method"].unique()):
        mask = df["method"].to_numpy() == m
        print(f"  {m:8s}: {rho(score_all[mask], cal_target[mask]):+.3f}  (n={mask.sum()})")

if __name__ == "__main__":
    main()
