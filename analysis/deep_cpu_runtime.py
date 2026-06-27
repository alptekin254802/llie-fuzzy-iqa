"""
CPU timing for deep NR-IQA baselines.

This small audit complements benchmark_runtime.py by timing pyiqa baselines on
CPU, using the same 256x256 per-image setting reported for the proposed fuzzy
assessor. It is intended for reviewer-facing CPU-vs-CPU context rather than for
score generation.

Run from the repository root:

    python -m analysis.deep_cpu_runtime --metrics clipiqa+ --n 30

Output:
    results/deep_cpu_runtime.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
OUT_CSV = RESULTS / "deep_cpu_runtime.csv"


def synthetic_tensor(seed: int, size: int, device: str) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    return torch.from_numpy(arr.transpose(2, 0, 1)[None]).float().to(device) / 255.0


def benchmark_metric(name: str, n: int, warmup: int, size: int, threads: str) -> dict:
    import pyiqa

    if threads == "single":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    device = "cpu"
    metric = pyiqa.create_metric(name, device=device)
    metric.eval()

    with torch.no_grad():
        for i in range(warmup):
            metric(synthetic_tensor(10_000 + i, size, device))

    times = []
    with torch.no_grad():
        for i in range(n):
            x = synthetic_tensor(20_000 + i, size, device)
            start = time.perf_counter()
            metric(x)
            times.append((time.perf_counter() - start) * 1e3)

    arr = np.asarray(times, dtype=float)
    median = float(np.median(arr))
    params = sum(p.numel() for p in metric.parameters()) / 1e6
    return {
        "method": name,
        "device": device,
        "threads": threads,
        "input_size": f"{size}x{size}",
        "ms_per_image": median,
        "mean_ms_per_image": float(arr.mean()),
        "p95_ms_per_image": float(np.percentile(arr, 95)),
        "img_per_s": 1000.0 / median if median > 0 else np.inf,
        "params_M_pyiqa": params,
        "n": n,
        "warmup": warmup,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", default=["clipiqa+"],
                        help="pyiqa metric names to benchmark, e.g. clipiqa+ musiq")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--threads", choices=["single", "default"], default="single")
    args = parser.parse_args()

    rows = []
    for metric in args.metrics:
        print(f"Benchmarking {metric} on CPU ({args.threads} thread setting)...")
        rows.append(benchmark_metric(metric, args.n, args.warmup, args.size, args.threads))

    out = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
