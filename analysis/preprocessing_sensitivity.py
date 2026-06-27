"""
Small preprocessing-sensitivity audit for the NR-fuzzy assessor.

The reviewer asked whether the headline result depends on choices such as
luminance conversion, entropy histogram binning, Laplacian kernel size, or resize
resolution. This script recomputes the three no-reference features under a small
set of alternatives and reruns the same image-group 5-fold calibration protocol.

It needs the LOL low-light images under preprocessed/low, which are not included
in the lightweight public package. The resulting CSV is included for reproducible
reporting:

    results/preprocessing_sensitivity.csv
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from analysis.reviewer_experiments import (
    K_FOLDS,
    NR_FEATURES,
    SEED,
    calibrate,
    firing,
    group_kfold,
    mfs_from,
    orient_arrays,
    rho,
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
OUT_CSV = RESULTS / "preprocessing_sensitivity.csv"

METHODS = {
    "clahe": apply_clahe,
    "gamma": apply_gamma_correction,
    "log": apply_log_transform,
    "retinex": apply_retinex,
    "hist_eq": histogram_equalization,
}

VARIANTS = [
    {
        "variant": "manuscript",
        "description": "BT.601 luminance, 256-bin entropy, default Laplacian, 256x256",
        "luma": "bt601",
        "entropy_bins": 256,
        "laplacian_ksize": 1,
        "resize": 256,
    },
    {
        "variant": "entropy_128_bins",
        "description": "128-bin entropy histogram",
        "luma": "bt601",
        "entropy_bins": 128,
        "laplacian_ksize": 1,
        "resize": 256,
    },
    {
        "variant": "entropy_64_bins",
        "description": "64-bin entropy histogram",
        "luma": "bt601",
        "entropy_bins": 64,
        "laplacian_ksize": 1,
        "resize": 256,
    },
    {
        "variant": "bt709_luminance",
        "description": "BT.709 luminance coefficients",
        "luma": "bt709",
        "entropy_bins": 256,
        "laplacian_ksize": 1,
        "resize": 256,
    },
    {
        "variant": "laplacian_5x5",
        "description": "larger 5x5 Laplacian aperture for sharpness",
        "luma": "bt601",
        "entropy_bins": 256,
        "laplacian_ksize": 5,
        "resize": 256,
    },
    {
        "variant": "resize_224",
        "description": "features computed after resizing enhanced image to 224x224",
        "luma": "bt601",
        "entropy_bins": 256,
        "laplacian_ksize": 1,
        "resize": 224,
    },
]


def resolve_low_dir() -> Path:
    for root in (BASE, BASE.parent):
        candidate = root / "preprocessed" / "low"
        if candidate.exists():
            return candidate
    raise SystemExit("Could not find preprocessed/low under the package or its parent.")


def resolve_image(directory: Path, filename: str) -> Path:
    p = directory / filename
    if p.exists():
        return p
    stem = Path(filename).stem
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        alt = directory / f"{stem}{ext}"
        if alt.exists():
            return alt
    raise FileNotFoundError(f"Image not found for {filename} in {directory}")


def load_rgb_float(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def luminance(rgb8: np.ndarray, mode: str) -> np.ndarray:
    if mode == "bt601":
        return cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY).astype(np.float64)
    if mode == "bt709":
        rgb = rgb8.astype(np.float64)
        return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    raise ValueError(f"Unknown luminance mode: {mode}")


def entropy_from_gray(gray: np.ndarray, bins: int) -> float:
    hist = cv2.calcHist([np.clip(gray, 0, 255).astype(np.uint8)], [0], None, [bins], [0, 256])
    prob = hist.flatten()
    prob = prob / prob.sum()
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def features(rgb01: np.ndarray, spec: dict) -> dict:
    rgb = np.clip(rgb01, 0.0, 1.0)
    if spec["resize"] != 256:
        rgb = cv2.resize(rgb, (spec["resize"], spec["resize"]), interpolation=cv2.INTER_AREA)
    rgb8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    gray = luminance(rgb8, spec["luma"])
    sharpness = cv2.Laplacian(
        gray, cv2.CV_64F, ksize=spec["laplacian_ksize"], borderType=cv2.BORDER_REPLICATE
    ).var() / (255.0 ** 2)
    return {
        "entropy": entropy_from_gray(gray, spec["entropy_bins"]),
        "contrast": float(gray.std() / 255.0),
        "sharpness": float(sharpness),
    }


def compute_feature_table(spec: dict, low_dir: Path) -> pd.DataFrame:
    base = pd.read_csv(RESULTS / "enhancement_metrics.csv")
    filenames = sorted(base["filename"].unique())
    rows = []
    for i, filename in enumerate(filenames, 1):
        low = load_rgb_float(resolve_image(low_dir, filename))
        for method, fn in METHODS.items():
            enhanced = np.clip(fn(low), 0.0, 1.0).astype(np.float32)
            rows.append({"filename": filename, "method": method, **features(enhanced, spec)})
        if i % 100 == 0 or i == len(filenames):
            print(f"  {spec['variant']}: {i}/{len(filenames)} images")
    return pd.DataFrame(rows)


def fuzzy_cv_lpips_niqe(df: pd.DataFrame) -> tuple[float, float, float, float]:
    groups = df["filename"].to_numpy()
    target = -df["lpips"].to_numpy(float)
    niqe = -df["niqe"].to_numpy(float)
    lpips_folds = []
    niqe_folds = []

    arrs, _ = orient_arrays(df, NR_FEATURES, target)
    fire, fsum = firing(arrs, mfs_from(arrs))
    for train_idx, test_idx in group_kfold(groups, K_FOLDS, SEED):
        theta = calibrate(fire, fsum, target, train_idx, lam=0.0, seed=SEED)
        score = (fire[test_idx] @ theta) / fsum[test_idx]
        lpips_folds.append(rho(score, target[test_idx]))
        niqe_folds.append(rho(score, niqe[test_idx]))

    lp = np.asarray(lpips_folds, dtype=float)
    nq = np.asarray(niqe_folds, dtype=float)
    return float(lp.mean()), float(lp.std()), float(nq.mean()), float(nq.std())


def main() -> None:
    low_dir = resolve_low_dir()
    perceptual = pd.read_csv(RESULTS / "perceptual_metrics.csv")[
        ["filename", "method", "lpips", "niqe"]
    ]
    rows = []
    baseline_lpips = None
    baseline_niqe = None

    print(f"Using low-light images from: {low_dir}")
    for spec in VARIANTS:
        feature_df = compute_feature_table(spec, low_dir)
        df = feature_df.merge(perceptual, on=["filename", "method"]).dropna().reset_index(drop=True)
        lp_m, lp_s, nq_m, nq_s = fuzzy_cv_lpips_niqe(df)
        if spec["variant"] == "manuscript":
            baseline_lpips = lp_m
            baseline_niqe = nq_m
        rows.append({
            "variant": spec["variant"],
            "description": spec["description"],
            "lpips_mean": lp_m,
            "lpips_std": lp_s,
            "niqe_mean": nq_m,
            "niqe_std": nq_s,
            "delta_lpips": lp_m - baseline_lpips if baseline_lpips is not None else 0.0,
            "delta_niqe": nq_m - baseline_niqe if baseline_niqe is not None else 0.0,
            "n": len(df),
        })
        print(
            f"  {spec['variant']:<18s} LPIPS {lp_m:.3f}+/-{lp_s:.3f} "
            f"NIQE {nq_m:.3f}+/-{nq_s:.3f}"
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
