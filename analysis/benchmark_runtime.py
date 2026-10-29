"""Stage timing for the deployed singleton scorers, with isolated process sessions.

The source images are resized to controlled input sizes before timing. File reads
use a warm filesystem cache. Each end-to-end path is timed directly; medians of
separate stages are not added. Model loading and warmup are excluded.
"""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import cv2
import numpy as np
from evaluation.frozen_scorer import FrozenScorer, MODEL_DIR
from evaluation.three_features import extract, from_gray


def summarize(values):
    return {"n": len(values), "median_ms": float(np.median(values)),
            "p95_ms": float(np.quantile(values, .95)), "mean_ms": float(np.mean(values))}


def power_scheme():
    if sys.platform != "win32":
        return "record externally"
    result = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True)
    return result.stdout.decode(errors="replace").strip()


def session(args):
    cv2.setNumThreads(1)
    names = sorted(args.images.glob("*.png"))[:50]
    if not names:
        raise ValueError("The image directory must contain PNG images")
    sources = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in names]
    if any(im is None for im in sources):
        raise ValueError("Unreadable input image")
    eq, mono = FrozenScorer("nr_free"), FrozenScorer("nr_mono")
    raw = {}

    def measure(key, fn, count=None, cycle=None):
        n = cycle or min(10, len(sources))
        for j in range(args.warmup):
            fn(j % n)
        values = []
        for j in range(count or args.samples):
            start = time.perf_counter_ns()
            fn(j % n)
            values.append((time.perf_counter_ns() - start) / 1e6)
        raw[key] = values

    started = datetime.now(timezone.utc).isoformat()
    with tempfile.TemporaryDirectory(prefix="iqa_timing_") as folder:
        for width, height in ((256, 256), (600, 400), (1280, 720), (1920, 1080)):
            tag = f"{width}x{height}"
            big = [cv2.resize(im, (width, height), interpolation=cv2.INTER_CUBIC) for im in sources[:10]]
            rgb = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in big]
            small = [cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA) for im in rgb]
            gray = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in small]
            features = [extract(im) for im in small]
            for ext in ("png", "jpg"):
                paths = [Path(folder) / f"{tag}_{i}.{ext}" for i in range(len(big))]
                for path, im in zip(paths, big):
                    if not cv2.imwrite(str(path), im, [cv2.IMWRITE_JPEG_QUALITY, 95] if ext == "jpg" else []):
                        raise OSError(f"Cannot save timing asset {path}")
                measure(tag + "/imread_" + ext, lambda i: cv2.imread(str(paths[i])))
            png = [Path(folder) / f"{tag}_{i}.png" for i in range(len(big))]
            measure(tag + "/bgr_rgb_gray", lambda i: cv2.cvtColor(cv2.cvtColor(big[i], cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2GRAY))
            for label, interp in (("area", cv2.INTER_AREA), ("linear", cv2.INTER_LINEAR)):
                measure(tag + "/resize_" + label, lambda i: cv2.resize(rgb[i], (256, 256), interpolation=interp))
                measure(tag + "/memory_e2e_" + label, lambda i: eq(extract(cv2.resize(rgb[i], (256, 256), interpolation=interp))))
                measure(tag + "/png_e2e_" + label, lambda i: eq(extract(cv2.resize(cv2.cvtColor(cv2.imread(str(png[i])), cv2.COLOR_BGR2RGB), (256, 256), interpolation=interp))))
            measure(tag + "/features_rgb", lambda i: extract(small[i]))
            measure(tag + "/features_gray", lambda i: from_gray(gray[i]))
            measure(tag + "/eq3", lambda i: eq(features[i]))
            measure(tag + "/nr_mono", lambda i: mono(features[i]))
            measure(tag + "/native_features_score", lambda i: eq(extract(rgb[i])), min(args.samples, 50))
    versions = {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__}
    if args.deep:
        import torch
        import pyiqa
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        versions.update(torch=torch.__version__, pyiqa=pyiqa.__version__)
        tensors = [torch.from_numpy(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (256, 256), interpolation=cv2.INTER_AREA)).permute(2, 0, 1)[None] for im in sources]
        for name in ("clipiqa+", "musiq"):
            model = pyiqa.create_metric(name, device="cpu")
            with torch.inference_mode():
                for j in range(3):
                    model(tensors[j % len(tensors)].float() / 255)
                times = []
                for tensor in tensors:
                    start = time.perf_counter_ns()
                    model(tensor.float() / 255)
                    times.append((time.perf_counter_ns() - start) / 1e6)
            raw["cpu/" + name] = times
            del model
    return {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(), "cpu": platform.processor(), "platform": platform.platform(),
            "power_scheme": power_scheme(), "threads": 1, "versions": versions,
            "images": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in names],
            "model_sha256": {n: hashlib.sha256((MODEL_DIR / f"{n}.json").read_bytes()).hexdigest() for n in ("nr_free", "nr_mono")},
            "cache": "warm filesystem cache", "resolutions": "controlled resampling of source images",
            "raw_ms": raw, "summary": {k: summarize(v) for k, v in raw.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("results/timing_run.json"))
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if min(args.sessions, args.samples, args.warmup) < 1:
        ap.error("sessions, samples and warmup must be positive")
    if args.output.exists():
        raise FileExistsError(f"Use a new output path: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        result = session(args)
    else:
        runs = []
        for i in range(args.sessions):
            path = args.output.with_name(f"{args.output.stem}_session{i+1}.json")
            command = [sys.executable, "-B", "-m", "analysis.benchmark_runtime", "--worker", "--images", str(args.images.resolve()),
                       "--output", str(path.resolve()), "--samples", str(args.samples), "--warmup", str(args.warmup)]
            if args.deep:
                command.append("--deep")
            subprocess.run(command, check=True)
            runs.append(json.loads(path.read_text()))
        pooled = {k: summarize([v for run in runs for v in run["raw_ms"][k]]) |
                  {"session_medians_ms": [run["summary"][k]["median_ms"] for run in runs]}
                  for k in runs[0]["raw_ms"]}
        result = {"sessions": runs, "pooled": pooled,
                  "interpretation": "Single-score latency; parameter search requires separate end-to-end timing."}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
