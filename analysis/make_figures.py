"""
Paper figures (PDF + SVG) for the revised manuscript - SELF-CONTAINED.

Place at: analysis/make_figures.py     Run: python -m analysis.make_figures

Reads the raw result CSVs, runs the fuzzy model + 5-fold CV internally
(reproduces the manuscript numbers, seed=42), and writes vector figures to
results/figures/. Figures carry no in-figure titles; titles live in the
manuscript captions. SVG text stays editable (svg.fonttype=none).

Inputs expected in results/ (same folder as your other CSVs):
  nr_features.csv, perceptual_metrics.csv, modern_nriqa.csv,
  clipiqa_plus_scores.csv, fuzzy_enhancement_results.csv,
  deep_zerodce_all.csv, deep_sci_all.csv
Outputs (results/figures/, each as .pdf and .svg):
  fig2_correlation_matrix, fig3_main_alignment, fig4_generalization,
  fig4b_within_method, fig5_rule_heatmap, figS_membership_functions
  + results/figures_numbers.csv  (numbers used by manuscript tables)

Architecture diagram (Fig 1) is inline TikZ in main.tex.
Qualitative figure (Fig 6) needs the LOL images -> make_qualitative_figure.py.
"""
from __future__ import annotations

import itertools
import os
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("RESULTS_DIR", BASE / "results"))
OUT = RESULTS / "figures"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "none",          # keep SVG text editable
})
BLUE, GREEN, PURPLE, ORANGE, RED, GRAY = "#4c72b0", "#55a868", "#8172b3", "#dd8452", "#c44e52", "#8c8c8c"

# ---------------- fuzzy model (verbatim logic from reference_free_fuzzy.py) ----
NR_FEATURES = ["entropy", "contrast", "sharpness"]
FR_FEATURES = ["psnr", "ssim", "entropy"]
K_FOLDS, PRIMARY_LAMBDA, DE_MAXITER, DE_POPSIZE, SEED = 5, 2.0, 40, 8, 42
COMBOS = list(itertools.product(range(3), range(3), range(3)))
IDX = {c: n for n, c in enumerate(COMBOS)}
LEVELS = ["low", "med", "high"]
BANDS = [(25, "Poor"), (50, "Fair"), (75, "Good"), (100.1, "Excellent")]
_P = []
for (i, j, k) in COMBOS:
    for di, dj, dk in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        h = (i + di, j + dj, k + dk)
        if h in IDX:
            _P.append((IDX[(i, j, k)], IDX[h]))
PAIRS = np.array(_P)


def trapmf(x, a, b, c, d):
    x = np.asarray(x, float)
    left = np.where(b > a, (x - a) / np.maximum(b - a, 1e-12), 1.0)
    right = np.where(d > c, (d - x) / np.maximum(d - c, 1e-12), 1.0)
    return np.clip(np.minimum(np.minimum(left, 1.0), right), 0, 1)


def mfset(v):
    q = np.quantile(v, [0, .2, .4, .5, .6, .8, 1])
    return [(q[0], q[0], q[1], q[2]), (q[1], q[2], q[4], q[5]), (q[4], q[5], q[6], q[6])]


def membership(v, mf=None):
    mf = mf or mfset(v)
    return np.stack([trapmf(v, *mf[l]) for l in range(3)], axis=1), mf


def rho(x, y):
    r = spearmanr(x, y)[0]
    return 0.0 if np.isnan(r) else float(r)


def violation(t):
    return float(np.clip(t[PAIRS[:, 0]] - t[PAIRS[:, 1]], 0, None).sum())


def orient(df, cols, target):
    arrs, flips = [], {}
    for c in cols:
        x = df[c].to_numpy(float)
        fl = spearmanr(x, target)[0] < 0
        flips[c] = fl
        arrs.append(-x if fl else x)
    return arrs, flips


def build_firing(arrs, mfs=None):
    Ms, out = [], []
    for n, a in enumerate(arrs):
        M, mf = membership(a, None if mfs is None else mfs[n])
        Ms.append(M); out.append(mf)
    fire = np.stack([np.minimum(np.minimum(Ms[0][:, i], Ms[1][:, j]), Ms[2][:, k]) for i, j, k in COMBOS], axis=1)
    return fire, fire.sum(axis=1) + 1e-9, out


def calibrate(fire, fsum, target, idx, lam):
    def obj(t):
        s = (fire[idx] @ t) / fsum[idx]
        return -rho(s, target[idx]) + lam * violation(t) / 100.0
    return differential_evolution(obj, bounds=[(0., 100.)] * 27, maxiter=DE_MAXITER,
                                  popsize=DE_POPSIZE, seed=SEED, tol=1e-3, polish=False,
                                  workers=1, updating="deferred").x


def group_kfold(groups, k, seed):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(np.unique(groups))); rng.shuffle(uniq)
    for chunk in np.array_split(uniq, k):
        ts = set(chunk.tolist()); m = np.array([g in ts for g in groups])
        yield np.where(~m)[0], np.where(m)[0]


def save(fig, name):
    for ext in ("pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  saved {name}.pdf / .svg")


# ----------------------------- data loading ----------------------------------
def load_data():
    nf = pd.read_csv(RESULTS / "nr_features.csv")[["filename", "method", "entropy", "contrast", "sharpness"]]
    pc = pd.read_csv(RESULTS / "perceptual_metrics.csv")[["filename", "method", "lpips", "niqe", "brisque"]]
    mo = pd.read_csv(RESULTS / "modern_nriqa.csv")[["filename", "method", "musiq", "maniqa", "clipiqa"]]
    cp = pd.read_csv(RESULTS / "clipiqa_plus_scores.csv")[["filename", "method", "clipiqa_plus"]]
    fz = pd.read_csv(RESULTS / "fuzzy_enhancement_results.csv")[["filename", "method", "psnr", "ssim", "fuzzy_score"]]
    classical = (nf.merge(pc, on=["filename", "method"]).merge(mo, on=["filename", "method"])
                 .merge(cp, on=["filename", "method"])
                 .merge(fz, on=["filename", "method"]).dropna().reset_index(drop=True))
    cols = ["filename", "method", "entropy", "contrast", "sharpness", "lpips", "niqe",
            "brisque", "musiq", "maniqa", "clipiqa", "psnr", "ssim"]
    dz = pd.read_csv(RESULTS / "deep_zerodce_all.csv")[cols]
    ds = pd.read_csv(RESULTS / "deep_sci_all.csv")[cols]
    expanded = pd.concat([classical[cols], dz, ds], ignore_index=True).dropna().reset_index(drop=True)
    return classical, expanded


def cv_summary(df, models_single):
    """5-fold held-out Spearman vs -LPIPS and -NIQE for single metrics + fuzzy models."""
    g = df["filename"].to_numpy()
    tgt = -df["lpips"].to_numpy(); niqe = -df["niqe"].to_numpy()
    nr_arr, _ = orient(df, NR_FEATURES, tgt); fr_arr, _ = orient(df, FR_FEATURES, tgt)
    nf_, ns_, _ = build_firing(nr_arr); ff_, fs_, _ = build_firing(fr_arr)
    rows = {m: {"lpips": [], "niqe": []} for m in models_single + ["fr_fuzzy_free", "nr_fuzzy_free", "nr_fuzzy_mono"]}
    for tr, te in group_kfold(g, K_FOLDS, SEED):
        thfr = calibrate(ff_, fs_, tgt, tr, 0.0)
        thnr = calibrate(nf_, ns_, tgt, tr, 0.0)
        thnm = calibrate(nf_, ns_, tgt, tr, PRIMARY_LAMBDA)
        sfr, snr, snm = (ff_[te] @ thfr) / fs_[te], (nf_[te] @ thnr) / ns_[te], (nf_[te] @ thnm) / ns_[te]
        for m in models_single:
            rows[m]["lpips"].append(rho(df[m].to_numpy()[te], tgt[te]))
            rows[m]["niqe"].append(rho(df[m].to_numpy()[te], niqe[te]))
        for nm, s in [("fr_fuzzy_free", sfr), ("nr_fuzzy_free", snr), ("nr_fuzzy_mono", snm)]:
            rows[nm]["lpips"].append(rho(s, tgt[te])); rows[nm]["niqe"].append(rho(s, niqe[te]))
    out = []
    for m, d in rows.items():
        out.append({"model": m, "lpips_mean": np.mean(d["lpips"]), "lpips_std": np.std(d["lpips"]),
                    "niqe_mean": np.mean(d["niqe"]), "niqe_std": np.std(d["niqe"])})
    return pd.DataFrame(out).set_index("model")


# ----------------------------- figures ---------------------------------------
def fig_correlation_matrix(df):
    cols = ["psnr", "ssim", "entropy", "lpips", "niqe", "musiq", "maniqa", "fuzzy_score", "nr_fuzzy_free"]
    labels = ["PSNR", "SSIM", "Entropy", "LPIPS", "NIQE", "MUSIQ", "MANIQA", "Fuzzy$_{\\mathrm{hand}}$", "NR-free"]
    have = [c for c in cols if c in df.columns]
    labels = [labels[cols.index(c)] for c in have]
    M = df[have].corr(method="spearman").to_numpy()
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(M, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(have))); ax.set_yticks(range(len(have)))
    ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
    for i in range(len(have)):
        for j in range(len(have)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    color="white" if abs(M[i, j]) > 0.55 else "black", fontsize=6.5)
    fig.colorbar(im, ax=ax, shrink=0.85, label="Spearman $\\rho$")
    save(fig, "fig2_correlation_matrix")


def fig_main_alignment(sc):
    order = ["psnr", "ssim", "entropy", "clipiqa", "clipiqa_plus", "musiq", "maniqa",
             "fuzzy_handtuned", "fr_fuzzy_free", "nr_fuzzy_mono", "nr_fuzzy_free"]
    nice = ["PSNR", "SSIM", "Entropy", "CLIP-IQA", "CLIP-IQA+", "MUSIQ", "MANIQA",
            "Fuzzy$_{\\mathrm{hand}}$", "FR-calibrated", "NR-mono", "NR-free"]
    colors = [BLUE, BLUE, GREEN, GRAY, GRAY, GRAY, GRAY, BLUE, PURPLE, ORANGE, RED]
    vals = sc.loc[order, "lpips_mean"].to_numpy(); errs = sc.loc[order, "lpips_std"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(range(len(order)), vals, yerr=errs, capsize=3, edgecolor="black", linewidth=0.6, color=colors)
    ax.axhline(vals[0], color="gray", ls="--", lw=0.8)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(nice, rotation=25, ha="right")
    ax.set_ylabel("Held-out Spearman with $-$LPIPS")
    ax.set_ylim(0, max(vals) + 0.10)
    ax.legend(handles=[Patch(color=GRAY, label="modern deep NR-IQA"),
                       Patch(color=RED, label="ours (reference-free)")],
              frameon=False, loc="upper left")
    save(fig, "fig3_main_alignment")


def fig_generalization(sc):
    order = ["psnr", "ssim", "fuzzy_handtuned", "fr_fuzzy_free", "nr_fuzzy_free"]
    nice = ["PSNR", "SSIM", "Fuzzy$_{\\mathrm{hand}}$", "FR-calibrated", "NR-free"]
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.bar(x - w/2, sc.loc[order, "lpips_mean"], w, yerr=sc.loc[order, "lpips_std"], capsize=3,
           label="vs $-$LPIPS (target)", color=BLUE, edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, sc.loc[order, "niqe_mean"], w, yerr=sc.loc[order, "niqe_std"], capsize=3,
           label="vs $-$NIQE (held-out anchor)", color=ORANGE, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(nice, rotation=20, ha="right")
    ax.set_ylabel("Held-out Spearman"); ax.legend(frameon=False)
    save(fig, "fig4_generalization")


def fig_within_method(expanded):
    """7-method within-method validity (all-data nr_fuzzy_free)."""
    tgt = -expanded["lpips"].to_numpy()
    arrs, _ = orient(expanded, NR_FEATURES, tgt)
    fire, fsum, _ = build_firing(arrs)
    theta = calibrate(fire, fsum, tgt, np.arange(len(expanded)), 0.0)
    score = (fire @ theta) / fsum
    meth = expanded["method"].to_numpy()
    vals = {m: rho(score[meth == m], tgt[meth == m]) for m in np.unique(meth)}
    order = sorted(vals, key=lambda m: vals[m], reverse=True)
    deep = {"zerodce", "sci"}
    colors = [PURPLE if m in deep else RED for m in order]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.bar(range(len(order)), [vals[m] for m in order], color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("Within-method Spearman with $-$LPIPS")
    ax.legend(handles=[Patch(color=RED, label="classical"), Patch(color=PURPLE, label="deep (Zero-DCE, SCI)")],
              frameon=False, loc="upper right")
    save(fig, "fig4b_within_method")
    return vals


def fig_rule_heatmap(classical):
    """Calibrated monotone rule base on classical all-data."""
    tgt = -classical["lpips"].to_numpy()
    arrs, _ = orient(classical, NR_FEATURES, tgt)
    fire, fsum, _ = build_firing(arrs)
    theta = calibrate(fire, fsum, tgt, np.arange(len(classical)), PRIMARY_LAMBDA)
    v = 100 * (theta - theta.min()) / (theta.max() - theta.min() + 1e-9)
    rules = pd.DataFrame([{"entropy": LEVELS[i], "contrast": LEVELS[j], "sharpness": LEVELS[k], "value": v[r]}
                          for r, (i, j, k) in enumerate(COMBOS)])
    lv = LEVELS
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.6), constrained_layout=True)
    for s_i, s in enumerate(lv):
        grid = np.zeros((3, 3))
        for e_i, e in enumerate(lv):
            for c_i, c in enumerate(lv):
                grid[e_i, c_i] = rules[(rules.entropy == e) & (rules.contrast == c) & (rules.sharpness == s)]["value"].iloc[0]
        ax = axes[s_i]
        im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=100, origin="lower")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(lv); ax.set_yticklabels(lv); ax.set_xlabel("contrast")
        if s_i == 0:
            ax.set_ylabel("entropy")
        ax.set_title(f"sharpness$^*$ = {s}", fontsize=9)   # panel label (a/b/c style), allowed
        for e_i in range(3):
            for c_i in range(3):
                ax.text(c_i, e_i, f"{grid[e_i,c_i]:.0f}", ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(im, ax=axes, shrink=0.8, label="calibrated quality")
    save(fig, "fig5_rule_heatmap")
    rules.to_csv(RESULTS / "rule_table_mono.csv", index=False)


def fig_membership(classical):
    tgt = -classical["lpips"].to_numpy()
    arrs, flips = orient(classical, NR_FEATURES, tgt)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for n, (feat, a) in enumerate(zip(NR_FEATURES, arrs)):
        mf = mfset(a)
        lo, hi = np.quantile(a, 0.01), np.quantile(a, 0.99)
        xs = np.linspace(lo, hi, 400)
        ax = axes[n]
        for l, col in zip(range(3), [BLUE, GREEN, RED]):
            ax.plot(xs, trapmf(xs, *mf[l]), color=col, lw=1.5, label=LEVELS[l])
        star = "$^*$" if flips[feat] else ""
        ax.set_xlabel(f"{feat}{star} (oriented)"); ax.set_ylim(-0.05, 1.1)
        if n == 0:
            ax.set_ylabel("membership"); ax.legend(frameon=False, loc="lower center", ncol=3)
    save(fig, "figS_membership_functions")


def main():
    print(f"Reading from: {RESULTS}\nWriting to: {OUT}")
    classical, expanded = load_data()
    print(f"classical n={len(classical)} | expanded n={len(expanded)} (methods: {sorted(expanded['method'].unique())})")

    single = ["psnr", "ssim", "entropy", "musiq", "maniqa", "clipiqa", "clipiqa_plus"]
    print("Running 5-fold CV on classical set (~45s)...")
    sc = cv_summary(classical, single)
    # add hand-tuned fuzzy as a 'single' column (from fuzzy_score)
    g = classical["filename"].to_numpy(); tgt = -classical["lpips"].to_numpy(); niqe = -classical["niqe"].to_numpy()
    fh_l, fh_n = [], []
    for tr, te in group_kfold(g, K_FOLDS, SEED):
        fh_l.append(rho(classical["fuzzy_score"].to_numpy()[te], tgt[te]))
        fh_n.append(rho(classical["fuzzy_score"].to_numpy()[te], niqe[te]))
    sc.loc["fuzzy_handtuned"] = [np.mean(fh_l), np.std(fh_l), np.mean(fh_n), np.std(fh_n)]

    # all-data classical NR-free score (so the correlation matrix can include it)
    arrs_c, _ = orient(classical, NR_FEATURES, tgt)
    fc, sc_, _ = build_firing(arrs_c)
    classical["nr_fuzzy_free"] = (fc @ calibrate(fc, sc_, tgt, np.arange(len(classical)), 0.0)) / sc_

    fig_correlation_matrix(classical)
    fig_main_alignment(sc)
    fig_generalization(sc)
    wm = fig_within_method(expanded)
    fig_rule_heatmap(classical)
    fig_membership(classical)

    sc.round(3).to_csv(RESULTS / "figures_numbers.csv")
    print("\nwithin-method (7):", {k: round(v, 3) for k, v in wm.items()})
    print("Done. Figures in", OUT)
    print("Saved CSV summaries in", RESULTS)


if __name__ == "__main__":
    main()
