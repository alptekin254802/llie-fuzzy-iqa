"""
Statistical-significance tests  -  SELF-CONTAINED.

Reproduces the manuscript's NR-free model (same CSVs, same group CV, global MF,
seed=42) and reports:

  (1) WILLIAMS' TEST for two DEPENDENT correlations that share the target
      (-LPIPS): is the NR-free model's rank correlation with perceptual quality
      significantly higher than each competitor's (PSNR, SSIM, MUSIQ, CLIP-IQA)?
      Run on rank-transformed data, so it tests Spearman differences.
  (2) GROUP-AWARE BOOTSTRAP 95% CIs for the headline Spearman correlations
      (resamples whole source scenes, matching the group-CV design).

Run from the folder with your result CSVs, or set RESULTS_DIR:
    python significance_tests.py
Inputs (same as make_figures.py):
    nr_features.csv, perceptual_metrics.csv, modern_nriqa.csv,
    fuzzy_enhancement_results.csv
Outputs:
    significance_tests.csv   (+ printed summary)
Needs: numpy, pandas, scipy
"""
from __future__ import annotations

import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import rankdata, spearmanr, t as student_t

# --------------------------------------------------------------------------- #
BASE = Path(__file__).resolve().parent.parent
_cands = [Path(os.environ["RESULTS_DIR"])] if os.environ.get("RESULTS_DIR") else []
_cands += [BASE / "results", Path(__file__).resolve().parent, Path.cwd()]
RESULTS = next((p for p in _cands if (p / "nr_features.csv").exists()), _cands[0])
print(f"[i] reading CSVs from: {RESULTS}")

NR_FEATURES = ["entropy", "contrast", "sharpness"]
FR_FEATURES = ["psnr", "ssim", "entropy"]
K_FOLDS, PRIMARY_LAMBDA, DE_MAXITER, DE_POPSIZE, SEED = 5, 2.0, 40, 8, 42
N_BOOT = 2000
COMBOS = list(itertools.product(range(3), range(3), range(3)))


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


def orient(df, cols, target):
    arrs = []
    for c in cols:
        x = df[c].to_numpy(float)
        arrs.append(-x if spearmanr(x, target)[0] < 0 else x)
    return arrs


def firing(arrs, mfs):
    Ms = [np.stack([trapmf(a, *mfs[n][l]) for l in range(3)], axis=1) for n, a in enumerate(arrs)]
    fire = np.stack([np.minimum(np.minimum(Ms[0][:, i], Ms[1][:, j]), Ms[2][:, k])
                     for i, j, k in COMBOS], axis=1)
    return fire, fire.sum(axis=1) + 1e-9


def calibrate(fire, fsum, target, idx, lam):
    def obj_real(t):
        s = (fire[idx] @ t) / fsum[idx]
        return -rho(s, target[idx])
    return differential_evolution(obj_real, bounds=[(0., 100.)] * 27, maxiter=DE_MAXITER,
                                  popsize=DE_POPSIZE, seed=SEED, tol=1e-3, polish=False,
                                  workers=1, updating="deferred").x


def group_kfold(groups, k, seed):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(np.unique(groups))); rng.shuffle(uniq)
    for chunk in np.array_split(uniq, k):
        ts = set(chunk.tolist()); m = np.array([g in ts for g in groups])
        yield np.where(~m)[0], np.where(m)[0]


def heldout_scores(df, feats, target):
    """Assemble per-image held-out NR/FR-fuzzy scores (global MF, per-fold consequents)."""
    arrs = orient(df, feats, target)
    fire, fsum = firing(arrs, [mfset(a) for a in arrs])
    s = np.full(len(df), np.nan)
    for tr, te in group_kfold(df["filename"].to_numpy(), K_FOLDS, SEED):
        th = calibrate(fire, fsum, target, tr, 0.0)
        s[te] = (fire[te] @ th) / fsum[te]
    return s


# --------------------------------------------------------------------------- #
def williams_t(r_ta, r_tb, r_ab, n):
    """Williams' test (Steiger 1980) for two dependent correlations sharing t.
    H0: r_ta == r_tb. Returns (t, df, two-sided p). Run on ranks => Spearman."""
    detR = 1 - r_ta**2 - r_tb**2 - r_ab**2 + 2 * r_ta * r_tb * r_ab
    num = (r_ta - r_tb) * np.sqrt((n - 1) * (1 + r_ab))
    den = np.sqrt(2 * ((n - 1) / (n - 3)) * detR + ((r_ta + r_tb)**2 / 4) * (1 - r_ab)**3)
    tval = num / den
    df = n - 3
    p = 2 * student_t.sf(abs(tval), df)
    return float(tval), int(df), float(p)


def dependent_corr_test(a, b, target, n):
    """a = our score, b = competitor, both vs target. Returns rho_a, rho_b, williams."""
    ra = rankdata(a); rb = rankdata(b); rt = rankdata(target)
    r_ta = np.corrcoef(rt, ra)[0, 1]
    r_tb = np.corrcoef(rt, rb)[0, 1]
    r_ab = np.corrcoef(ra, rb)[0, 1]
    tval, df, p = williams_t(r_ta, r_tb, r_ab, n)
    return r_ta, r_tb, r_ab, tval, df, p


def group_bootstrap_ci(score, target, groups, n_boot=N_BOOT, seed=SEED):
    """95% CI for Spearman(score, target) by resampling whole groups (scenes)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    by_g = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_g[g] for g in pick])
        vals.append(rho(score[idx], target[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
def main():
    nf = pd.read_csv(RESULTS / "nr_features.csv")[["filename", "method", "entropy", "contrast", "sharpness"]]
    pc = pd.read_csv(RESULTS / "perceptual_metrics.csv")[["filename", "method", "lpips", "niqe"]]
    mo = pd.read_csv(RESULTS / "modern_nriqa.csv")[["filename", "method", "musiq", "maniqa", "clipiqa"]]
    fz = pd.read_csv(RESULTS / "fuzzy_enhancement_results.csv")[["filename", "method", "psnr", "ssim"]]
    df = (nf.merge(pc, on=["filename", "method"]).merge(mo, on=["filename", "method"])
          .merge(fz, on=["filename", "method"]).dropna().reset_index(drop=True))
    n = len(df)
    g = df["filename"].to_numpy()
    tgt = -df["lpips"].to_numpy()
    print(f"[i] n={n}")

    print("[i] assembling held-out NR-free scores (~1 min)...")
    nrfree = heldout_scores(df, NR_FEATURES, tgt)
    print(f"    held-out Spearman(NR-free, -LPIPS) = {rho(nrfree, tgt):.3f}")

    # (1) Williams' tests: NR-free vs each competitor, target = -LPIPS
    print("\n=== (1) WILLIAMS' TEST: is NR-free's correlation with -LPIPS higher? ===")
    rows = []
    competitors = {"PSNR": df["psnr"].to_numpy(), "SSIM": df["ssim"].to_numpy(),
                   "MUSIQ": df["musiq"].to_numpy(), "CLIP-IQA": df["clipiqa"].to_numpy()}
    for name, comp in competitors.items():
        r_ta, r_tb, r_ab, tval, ddf, p = dependent_corr_test(nrfree, comp, tgt, n)
        sig = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"
        print(f"  NR-free({r_ta:.3f}) vs {name:8s}({r_tb:.3f}): Williams t={tval:7.2f}, "
              f"df={ddf}, p={p:.2e}  {sig}")
        rows.append(dict(test="williams", competitor=name, rho_nrfree=r_ta, rho_comp=r_tb,
                         r_ab=r_ab, williams_t=tval, df=ddf, p_value=p))

    # (2) Group-aware bootstrap CIs for headline correlations vs -LPIPS
    print("\n=== (2) GROUP-BOOTSTRAP 95% CIs (Spearman vs -LPIPS) ===")
    headline = {"NR-free": nrfree, "PSNR": df["psnr"].to_numpy(), "SSIM": df["ssim"].to_numpy(),
                "MUSIQ": df["musiq"].to_numpy(), "MANIQA": df["maniqa"].to_numpy(),
                "CLIP-IQA": df["clipiqa"].to_numpy()}
    for name, sc in headline.items():
        r = rho(sc, tgt)
        lo, hi = group_bootstrap_ci(sc, tgt, g)
        print(f"  {name:9s} rho={r:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
        rows.append(dict(test="bootstrap_ci", model=name, rho=r, ci_lo=lo, ci_hi=hi))

    out = RESULTS / "significance_tests.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n[ok] wrote {out}")
    print("[ok] significance_tests.csv contains the manuscript significance summaries.")


if __name__ == "__main__":
    main()
