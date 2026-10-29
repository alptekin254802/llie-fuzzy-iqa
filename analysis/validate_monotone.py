"""Reproduce NR-mono fold validation and record exact-rank / near-tie sensitivity."""
from pathlib import Path
import json
import hashlib
import platform
import scipy
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from analysis.monotone_fuzzy import design, fit
from analysis.reference_free_fuzzy import mfset, group_kfold, NR_FEATURES
from evaluation.frozen_scorer import FrozenScorer


def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    data = pd.read_csv(results / "nr_features.csv").merge(
        pd.read_csv(results / "perceptual_metrics.csv")[["filename", "method", "lpips", "niqe", "brisque"]],
        on=["filename", "method"], validate="one_to_one").merge(
        pd.read_csv(results / "enhancement_metrics.csv")[["filename", "method", "psnr", "ssim"]],
        on=["filename", "method"], validate="one_to_one").dropna().reset_index(drop=True)
    x = data[NR_FEATURES].to_numpy() * FrozenScorer("nr_mono").signs
    y = -data.lpips.to_numpy()
    predictions = np.empty(len(data)); fold_ids = np.empty(len(data), int); models = []
    for i, (train, test) in enumerate(group_kfold(data.filename, 5, 42)):
        mfs = [mfset(x[train, n]) for n in range(3)]
        theta = fit(x[train], y[train], mfs)
        predictions[test] = design(x[test], mfs) @ theta
        fold_ids[test] = i
        models.append({"mfs": mfs, "theta": theta.tolist()})
    output = {"protocol": "LOL orientations; train-fold membership breakpoints; constrained SSE; C-order design",
              "ranking": "exact floating-point predictions", "models": models,
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__, "blas_threads": 1},
              "input_sha256": {n: hashlib.sha256((results / n).read_bytes()).hexdigest() for n in ("nr_features.csv", "perceptual_metrics.csv", "enhancement_metrics.csv")},
              "deployment_sha256": hashlib.sha256((root / "models/nr_mono.json").read_bytes()).hexdigest()}
    scene_values = []
    for _, idx in data.groupby("filename").indices.items():
        r = spearmanr(predictions[idx], y[idx]).statistic
        scene_values.append(0.0 if np.isnan(r) else float(r))
    output["within_scene_srcc"] = float(np.mean(scene_values))
    for anchor in ("lpips", "niqe", "brisque"):
        target = -data[anchor].to_numpy()
        correlations = [float(spearmanr(predictions[fold_ids == i], target[fold_ids == i]).statistic) for i in range(5)]
        output[anchor + "_mean"] = float(np.mean(correlations))
        output[anchor + "_std"] = float(np.std(correlations))
        output[anchor + "_folds"] = correlations
    rounded = np.round(predictions, 9)
    output["near_tie_sensitivity"] = {"interpretation": "diagnostic; does not replace exact-rank primary estimates",
                                      "decimal_places": 9, "lpips_fold_mean": float(np.mean([
                                          spearmanr(rounded[fold_ids == i], y[fold_ids == i]).statistic for i in range(5)]))}
    (results / "monotone_validation.json").write_text(json.dumps(output, indent=2) + "\n")
    data[["filename", "method"]].assign(fold=fold_ids, prediction=predictions).to_csv(results / "nr_mono_oof.csv", index=False)
    print(json.dumps({k:v for k,v in output.items() if k != "models"}, indent=2))


if __name__ == "__main__":
    main()
