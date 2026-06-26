"""
Tier-3 (minimal): does a BROADER, more diverse calibration pool improve
operator-disjoint generalisation? (reviewer Q3 / Retinex failure)

For each held-out enhancer, calibrate the NR-free model on the OTHER methods and
score the held-out one, two ways:
   narrow pool = other CLASSICAL methods only (the manuscript's disjoint setting)
   broad  pool = ALL other methods, classical + deep (Zero-DCE, SCI)
Train-only MF breakpoints + orientation; held-out Spearman vs -LPIPS.

Inputs (same as before): nr_features.csv, perceptual_metrics.csv,
                         deep_zerodce_all.csv, deep_sci_all.csv
Output: tier3_broaden.csv  (+ printed table)
Run:    python tier3_broaden.py
Needs: numpy, pandas, scipy
"""
from __future__ import annotations
import itertools, os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr

NR = ["entropy", "contrast", "sharpness"]
DE_MAXITER, DE_POPSIZE, SEED = 40, 8, 42
COMBOS = list(itertools.product(range(3), range(3), range(3)))
CLASSICAL = ["clahe", "gamma", "log", "retinex", "hist_eq"]

BASE = Path(__file__).resolve().parent.parent
_c = [Path(os.environ["RESULTS_DIR"])] if os.environ.get("RESULTS_DIR") else []
_c += [BASE / "results", Path(__file__).resolve().parent, Path.cwd()]
RESULTS = next((p for p in _c if (p / "nr_features.csv").exists()), _c[0])


def trapmf(x, a, b, c, d):
    x = np.asarray(x, float)
    l = np.where(b > a, (x - a) / np.maximum(b - a, 1e-12), 1.0)
    r = np.where(d > c, (d - x) / np.maximum(d - c, 1e-12), 1.0)
    return np.clip(np.minimum(np.minimum(l, 1.0), r), 0, 1)


def mfset(v):
    q = np.quantile(v, [0, .2, .4, .5, .6, .8, 1])
    return [(q[0], q[0], q[1], q[2]), (q[1], q[2], q[4], q[5]), (q[4], q[5], q[6], q[6])]


def rho(x, y):
    r = spearmanr(x, y)[0]
    return 0.0 if np.isnan(r) else float(r)


def firing(arrs, mfs):
    Ms = [np.stack([trapmf(a, *mfs[n][l]) for l in range(3)], axis=1) for n, a in enumerate(arrs)]
    fire = np.stack([np.minimum(np.minimum(Ms[0][:, i], Ms[1][:, j]), Ms[2][:, k])
                     for i, j, k in COMBOS], axis=1)
    return fire, fire.sum(axis=1) + 1e-9


def calibrate(fire, fsum, tgt, idx):
    def obj(t):
        s = (fire[idx] @ t) / fsum[idx]
        return -rho(s, tgt[idx])
    return differential_evolution(obj, bounds=[(0., 100.)] * 27, maxiter=DE_MAXITER,
                                  popsize=DE_POPSIZE, seed=SEED, tol=1e-3, polish=False,
                                  workers=1, updating="deferred").x


def fit_score(train, test):
    tgt = -train["lpips"].to_numpy()
    signs = {c: (spearmanr(train[c].to_numpy(float), tgt)[0] < 0) for c in NR}
    arr_tr = [(-train[c].to_numpy(float) if signs[c] else train[c].to_numpy(float)) for c in NR]
    mfs = [mfset(a) for a in arr_tr]
    fire, fsum = firing(arr_tr, mfs)
    th = calibrate(fire, fsum, tgt, np.arange(len(train)))
    arr_te = [(-test[c].to_numpy(float) if signs[c] else test[c].to_numpy(float)) for c in NR]
    fte, fste = firing(arr_te, mfs)
    return rho((fte @ th) / fste, -test["lpips"].to_numpy())


def main():
    cols = ["filename", "method"] + NR + ["lpips"]
    nf = pd.read_csv(RESULTS / "nr_features.csv")[["filename", "method"] + NR]
    pc = pd.read_csv(RESULTS / "perceptual_metrics.csv")[["filename", "method", "lpips"]]
    classical = nf.merge(pc, on=["filename", "method"]).dropna()
    dz = pd.read_csv(RESULTS / "deep_zerodce_all.csv")[cols].dropna()
    ds = pd.read_csv(RESULTS / "deep_sci_all.csv")[cols].dropna()
    pool = pd.concat([classical, dz, ds], ignore_index=True)
    methods = CLASSICAL + ["zerodce", "sci"]
    print(f"[i] pool: {len(pool)} images, methods={methods}")

    rows = []
    print("\nheld-out      narrow(classical-only)   broad(classical+deep)   delta")
    for m in methods:
        te = pool[pool["method"] == m]
        narrow_tr = pool[(pool["method"] != m) & (pool["method"].isin(CLASSICAL))]
        broad_tr = pool[pool["method"] != m]
        rn = fit_score(narrow_tr, te)
        rb = fit_score(broad_tr, te)
        print(f"  {m:10s}      {rn:+.3f}                  {rb:+.3f}              {rb-rn:+.3f}")
        rows.append(dict(held_out=m, narrow=rn, broad=rb, delta=rb - rn, n=len(te)))
    pd.DataFrame(rows).to_csv(RESULTS / "tier3_broaden.csv", index=False)
    print(f"\n[ok] wrote {RESULTS / 'tier3_broaden.csv'}")
    print("[ok] tier3_broaden.csv contains the broadened-pool transfer summary.")


if __name__ == "__main__":
    main()
