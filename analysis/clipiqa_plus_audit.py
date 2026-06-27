"""
CLIP-IQA+ audit for the LLIE NR-IQA study.

This script evaluates the prompt-tuned CLIP-IQA+ baseline without modifying any
of the main result CSVs. It can either reuse a cached score file or compute
CLIP-IQA+ scores from the same enhanced-image regeneration pipeline used by
evaluation/modern_nriqa.py.

Outputs:
  results/clipiqa_plus_scores.csv
      filename, method, clipiqa_plus

  results/clipiqa_plus_summary.csv
      5-fold image-group Spearman with -LPIPS and -NIQE, plus overall
      correlations, parameter count, timing, and a Williams-test comparison
      against NR-free.

  results/clipiqa_plus_runtime.csv
      standalone timing row for CLIP-IQA+.

Usage:
  python -m analysis.clipiqa_plus_audit
  python -m analysis.clipiqa_plus_audit --recompute-scores
  python -m analysis.clipiqa_plus_audit --timing-n 300

Notes:
  * Recomputing scores requires preprocessed/low and pyiqa/torch.
  * The first CLIP-IQA+ run may download the pyiqa CLIP-IQA+ prompt weights.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, t as student_t

try:
    import torch
    import pyiqa
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency. Install with: pip install torch torchvision pyiqa\n"
        f"Import error: {exc}"
    )

from preprocessing.enhancement.methods import (
    apply_clahe,
    apply_gamma_correction,
    apply_log_transform,
    apply_retinex,
    histogram_equalization,
)


BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
LOW_DIR = BASE / "preprocessed" / "low"
CLASSICAL_CSV = RESULTS / "enhancement_metrics.csv"
PERCEPTUAL_CSV = RESULTS / "perceptual_metrics.csv"
MODERN_CSV = RESULTS / "modern_nriqa.csv"
NR_FUZZY_CSV = RESULTS / "nr_fuzzy_scores.csv"
SCORES_CSV = RESULTS / "clipiqa_plus_scores.csv"
SUMMARY_CSV = RESULTS / "clipiqa_plus_summary.csv"
RUNTIME_CSV = RESULTS / "clipiqa_plus_runtime.csv"

METHODS = {
    "clahe": apply_clahe,
    "gamma": apply_gamma_correction,
    "log": apply_log_transform,
    "retinex": apply_retinex,
    "hist_eq": histogram_equalization,
}


def resolve_image(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if candidate.exists():
        return candidate
    stem = Path(filename).stem
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        alt = directory / f"{stem}{ext}"
        if alt.exists():
            return alt
    raise FileNotFoundError(f"Image not found for {filename!r} in {directory}")


def load_rgb_float(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def to_tensor(rgb01: np.ndarray, device: str) -> "torch.Tensor":
    arr = np.ascontiguousarray(np.clip(rgb01, 0.0, 1.0))
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(device)


def make_metric(device: str):
    return pyiqa.create_metric("clipiqa+", device=device)


def compute_scores(device: str) -> pd.DataFrame:
    if not CLASSICAL_CSV.exists():
        raise SystemExit(f"Missing {CLASSICAL_CSV}")
    if not LOW_DIR.exists():
        raise SystemExit(f"Missing {LOW_DIR}; cannot recompute CLIP-IQA+ scores.")

    metric = make_metric(device)
    df = pd.read_csv(CLASSICAL_CSV)
    filenames = sorted(df["filename"].unique())
    methods_in_csv = sorted(df["method"].unique())
    unknown = [m for m in methods_in_csv if m not in METHODS]
    if unknown:
        raise SystemExit(f"Unknown methods in {CLASSICAL_CSV.name}: {unknown}")

    print(f"Computing CLIP-IQA+ scores on {device}")
    print(f"Images: {len(filenames)} | methods: {methods_in_csv}")
    rows = []
    t0 = time.perf_counter()
    for i, fname in enumerate(filenames, 1):
        low = load_rgb_float(resolve_image(LOW_DIR, fname))
        for label, fn in METHODS.items():
            enh = np.clip(fn(low), 0.0, 1.0).astype(np.float32)
            x = to_tensor(enh, device)
            with torch.no_grad():
                if device == "cuda":
                    torch.cuda.synchronize()
                score = float(metric(x).item())
                if device == "cuda":
                    torch.cuda.synchronize()
            rows.append({"filename": fname, "method": label, "clipiqa_plus": score})
        if i % 25 == 0 or i == len(filenames):
            print(f"  {i}/{len(filenames)} images done ({time.perf_counter() - t0:.1f}s)")

    out = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(SCORES_CSV, index=False)
    print(f"Saved: {SCORES_CSV} ({len(out)} rows)")
    return out


def rho(x, y) -> float:
    r = spearmanr(x, y)[0]
    return 0.0 if np.isnan(r) else float(r)


def group_kfold(groups, k=5, seed=42):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(np.unique(groups)))
    rng.shuffle(uniq)
    for chunk in np.array_split(uniq, k):
        test_groups = set(chunk.tolist())
        mask = np.array([g in test_groups for g in groups])
        yield np.where(~mask)[0], np.where(mask)[0]


def cv_spearman(df: pd.DataFrame, score_col: str, target_col: str) -> tuple[np.ndarray, float, float]:
    groups = df["filename"].to_numpy()
    scores = df[score_col].to_numpy(float)
    target = df[target_col].to_numpy(float)
    folds = []
    for _, test_idx in group_kfold(groups):
        folds.append(rho(scores[test_idx], target[test_idx]))
    arr = np.asarray(folds, dtype=float)
    return arr, float(arr.mean()), float(arr.std())


def pearson(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.std() <= 1e-12 or y.std() <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def williams_test(r12, r13, r23, n):
    """Williams test for two dependent correlations sharing one variable."""
    if n <= 3:
        return np.nan, np.nan, np.nan
    r12 = float(np.clip(r12, -0.999999, 0.999999))
    r13 = float(np.clip(r13, -0.999999, 0.999999))
    r23 = float(np.clip(r23, -0.999999, 0.999999))
    k = 1.0 - r23
    denominator = 2.0 * k * (
        (n - 1.0) / (n - 3.0)
        + ((r12 + r13) ** 2 / 4.0) * (1.0 - r23) ** 3
    )
    if denominator <= 0:
        return np.nan, n - 3.0, np.nan
    t = (r12 - r13) * np.sqrt((n - 1.0) * (1.0 + r23) / denominator)
    df = n - 3.0
    p = 2.0 * (1.0 - student_t.cdf(abs(t), df))
    return float(t), float(df), float(p)


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    for path in (PERCEPTUAL_CSV, NR_FUZZY_CSV):
        if not path.exists():
            raise SystemExit(f"Missing {path}")

    pc = pd.read_csv(PERCEPTUAL_CSV)[["filename", "method", "lpips", "niqe"]]
    nr = pd.read_csv(NR_FUZZY_CSV)[["filename", "method", "nr_fuzzy_free"]]
    df = scores.merge(pc, on=["filename", "method"]).merge(nr, on=["filename", "method"]).dropna()
    df = df.reset_index(drop=True)
    df["minus_lpips"] = -df["lpips"].to_numpy(float)
    df["minus_niqe"] = -df["niqe"].to_numpy(float)

    rows = []
    for model, col in [("CLIP-IQA+", "clipiqa_plus"), ("NR-free", "nr_fuzzy_free")]:
        lp_folds, lp_mean, lp_std = cv_spearman(df, col, "minus_lpips")
        nq_folds, nq_mean, nq_std = cv_spearman(df, col, "minus_niqe")
        rows.append({
            "model": model,
            "lpips_mean": lp_mean,
            "lpips_std": lp_std,
            "niqe_mean": nq_mean,
            "niqe_std": nq_std,
            "lpips_overall": rho(df[col], df["minus_lpips"]),
            "niqe_overall": rho(df[col], df["minus_niqe"]),
            "lpips_folds": " ".join(f"{v:.6f}" for v in lp_folds),
            "niqe_folds": " ".join(f"{v:.6f}" for v in nq_folds),
        })

    if MODERN_CSV.exists() and "clipiqa" in pd.read_csv(MODERN_CSV, nrows=0).columns:
        old = pd.read_csv(MODERN_CSV)[["filename", "method", "clipiqa"]]
        old_df = old.merge(pc, on=["filename", "method"]).dropna().reset_index(drop=True)
        old_df["minus_lpips"] = -old_df["lpips"].to_numpy(float)
        old_df["minus_niqe"] = -old_df["niqe"].to_numpy(float)
        lp_folds, lp_mean, lp_std = cv_spearman(old_df, "clipiqa", "minus_lpips")
        nq_folds, nq_mean, nq_std = cv_spearman(old_df, "clipiqa", "minus_niqe")
        rows.append({
            "model": "CLIP-IQA",
            "lpips_mean": lp_mean,
            "lpips_std": lp_std,
            "niqe_mean": nq_mean,
            "niqe_std": nq_std,
            "lpips_overall": rho(old_df["clipiqa"], old_df["minus_lpips"]),
            "niqe_overall": rho(old_df["clipiqa"], old_df["minus_niqe"]),
            "lpips_folds": " ".join(f"{v:.6f}" for v in lp_folds),
            "niqe_folds": " ".join(f"{v:.6f}" for v in nq_folds),
        })

    ranks_target = rankdata(df["minus_lpips"].to_numpy(float))
    ranks_nr = rankdata(df["nr_fuzzy_free"].to_numpy(float))
    ranks_plus = rankdata(df["clipiqa_plus"].to_numpy(float))
    r_nr = pearson(ranks_nr, ranks_target)
    r_plus = pearson(ranks_plus, ranks_target)
    r_between = pearson(ranks_nr, ranks_plus)
    t_val, df_val, p_val = williams_test(r_nr, r_plus, r_between, len(df))

    rows.append({
        "model": "Williams NR-free vs CLIP-IQA+",
        "lpips_mean": r_nr,
        "lpips_std": np.nan,
        "niqe_mean": r_plus,
        "niqe_std": np.nan,
        "lpips_overall": r_between,
        "niqe_overall": np.nan,
        "williams_t": t_val,
        "williams_df": df_val,
        "williams_p": p_val,
    })

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved: {SUMMARY_CSV}")
    return summary


def synthetic_images(n: int, h=256, w=256, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def benchmark_metric(metric, device: str, n: int, warmup: int) -> dict:
    warm = list(synthetic_images(warmup, seed=123))
    for im in warm:
        x = torch.from_numpy(im.transpose(2, 0, 1)[None]).float().to(device) / 255.0
        with torch.no_grad():
            metric(x)
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for im in synthetic_images(n, seed=42):
        x = torch.from_numpy(im.transpose(2, 0, 1)[None]).float().to(device) / 255.0
        start = time.perf_counter()
        with torch.no_grad():
            metric(x)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e3)

    arr = np.asarray(times, dtype=float)
    med = float(np.median(arr))
    return {
        "method": "clipiqa+",
        "device": device,
        "ms_per_image": med,
        "mean_ms_per_image": float(arr.mean()),
        "p95_ms_per_image": float(np.percentile(arr, 95)),
        "img_per_s": 1000.0 / med if med > 0 else np.inf,
        "params_M": sum(p.numel() for p in metric.parameters()) / 1e6,
        "n": n,
    }


def print_summary(summary: pd.DataFrame, runtime: dict):
    print("\nCorrelation summary:")
    for _, row in summary.iterrows():
        model = row["model"]
        if model.startswith("Williams"):
            print(
                f"  Williams NR-free vs CLIP-IQA+: "
                f"t={row['williams_t']:.3f}, df={row['williams_df']:.0f}, "
                f"p={row['williams_p']:.3e}"
            )
        else:
            print(
                f"  {model:<10s} vs -LPIPS {row['lpips_mean']:.3f} +/- {row['lpips_std']:.3f} | "
                f"vs -NIQE {row['niqe_mean']:.3f} +/- {row['niqe_std']:.3f}"
            )
    print("\nRuntime summary:")
    print(
        f"  CLIP-IQA+ params={runtime['params_M']:.6f} M | "
        f"median={runtime['ms_per_image']:.3f} ms/image | "
        f"p95={runtime['p95_ms_per_image']:.3f} ms/image | "
        f"throughput={runtime['img_per_s']:.1f} img/s | device={runtime['device']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--recompute-scores", action="store_true",
                        help="ignore any cached clipiqa_plus_scores.csv and recompute scores")
    parser.add_argument("--timing-n", type=int, default=300,
                        help="number of synthetic 256x256 images for timing")
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    RESULTS.mkdir(parents=True, exist_ok=True)
    metric = make_metric(device)

    if SCORES_CSV.exists() and not args.recompute_scores:
        print(f"Using cached scores: {SCORES_CSV}")
        scores = pd.read_csv(SCORES_CSV)
    else:
        scores = compute_scores(device)

    summary = summarize(scores)
    runtime = benchmark_metric(metric, device, args.timing_n, args.warmup)
    pd.DataFrame([runtime]).to_csv(RUNTIME_CSV, index=False)
    print(f"Saved: {RUNTIME_CSV}")
    print_summary(summary, runtime)


if __name__ == "__main__":
    main()
