"""
benchmark_runtime.py
====================
Runtime / computational-complexity benchmark for the reference-free fuzzy
NR-IQA method (entropy + contrast + sharpness* -> 27-rule Mamdani FIS).

Purpose
-------
Produces the real per-image latency, throughput (images/s) and frame rate (fps)
needed for the "Computational Efficiency" section required by the Journal of
Real-Time Image Processing. Everything runs on CPU, single image at a time, with
no GPU and no learned weights, which is the whole point of the comparison against
deep no-reference IQA networks.

What it measures
----------------
  1. Feature extraction time  (entropy, contrast, sharpness) per image
  2. Fuzzy inference time      (27-rule Mamdani, centroid defuzzification)
  3. End-to-end time           (features + inference)
  -> reported as median ms/image, plus throughput (img/s) and fps.

Notes on fidelity
-----------------
* The three features use the same definitions as the paper:
    entropy   = Shannon entropy of the 256-bin grayscale histogram
    contrast  = standard deviation of the grayscale luminance
    sharpness = variance of the Laplacian
* The 27-rule Mamdani inference is implemented efficiently (vectorised,
  closed-form trapezoidal membership functions, centroid defuzzification on a
  configurable universe). The *runtime* of the method does not depend on whether
  the rule consequents are hand-set or LPIPS-calibrated, so arbitrary valid
  consequents are used here; the timing is representative of the deployed system.
* If needed for another implementation, replace
  `extract_features` and `FuzzyFIS.infer` with imports from your `analysis`
  package; the harness (timing, sweep, reporting) stays the same.

Usage
-----
  # synthetic images at the LOL resolution (default), single thread:
  python benchmark_runtime.py

  # time on your real enhanced images:
  python benchmark_runtime.py --images /path/to/enhanced --limit 500

  # resolution sweep (256, 512, 600x400, 1024):
  python benchmark_runtime.py --sweep

  # also time deep NR-IQA baselines (needs `pip install pyiqa torch`):
  python benchmark_runtime.py --deep

Outputs: prints a summary table and writes results/runtime_results.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import platform
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise SystemExit("OpenCV is required: pip install opencv-python-headless") from e


# --------------------------------------------------------------------------- #
#  Feature extraction (same definitions as the paper)
# --------------------------------------------------------------------------- #
def _to_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 1) * 255.0 if img.max() <= 1.0 else img
            img = img.astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img.astype(np.uint8)


def feat_entropy(gray: np.ndarray, bins: int = 256) -> float:
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256]).flatten()
    prob = hist / max(hist.sum(), 1.0)
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def feat_contrast(gray: np.ndarray) -> float:
    return float(gray.astype(np.float32).std())


def feat_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_features(img: np.ndarray) -> Tuple[float, float, float]:
    """Return (entropy, contrast, sharpness) for an RGB image."""
    gray = _to_gray_u8(img)
    return feat_entropy(gray), feat_contrast(gray), feat_sharpness(gray)


# --------------------------------------------------------------------------- #
#  27-rule Mamdani fuzzy inference (efficient, closed-form)
# --------------------------------------------------------------------------- #
def trapmf(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    left = np.where(b > a, (x - a) / np.maximum(b - a, 1e-12), 1.0)
    right = np.where(d > c, (d - x) / np.maximum(d - c, 1e-12), 1.0)
    return np.clip(np.minimum(np.minimum(left, 1.0), right), 0.0, 1.0)


class FuzzyFIS:
    """
    Three inputs (entropy, contrast, sharpness*), three trapezoidal sets each
    (low/medium/high) -> 3^3 = 27 rules. Mamdani min-implication, max-aggregation,
    centroid defuzzification over a quality universe.

    Membership breakpoints and rule consequents below are placeholders: they do
    NOT affect runtime. Swap in calibrated parameters if score values are also
    needed; the timing is identical.
    """

    def __init__(self, ranges: Dict[str, Tuple[float, float]] | None = None,
                 universe_res: int = 101):
        ranges = ranges or {
            "entropy": (0.0, 8.0),
            "contrast": (0.0, 128.0),
            "sharpness": (0.0, 2000.0),
        }
        self.ranges = ranges
        # three trapezoidal sets per input, placed at 20/40/60/80% of the range
        self.mf: Dict[str, List[Tuple[float, float, float, float]]] = {}
        for name, (lo, hi) in ranges.items():
            span = hi - lo
            q = [lo + span * p for p in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)]
            self.mf[name] = [
                (q[0], q[0], q[1], q[2]),   # low
                (q[1], q[2], q[4], q[5]),   # medium
                (q[4], q[5], q[6], q[6]),   # high
            ]
        # 27 rule consequents (centres on the quality universe), placeholder values
        rng = np.random.default_rng(42)
        self.universe = np.linspace(0.0, 100.0, universe_res)
        # output sets: poor/fair/good/excellent as trapezoids
        self.out_sets = [
            (0.0, 0.0, 20.0, 40.0),
            (20.0, 40.0, 50.0, 65.0),
            (50.0, 65.0, 75.0, 90.0),
            (75.0, 90.0, 100.0, 100.0),
        ]
        self.out_mf = np.stack([trapmf(self.universe, *s) for s in self.out_sets])
        # assign each of 27 rules to one output set (placeholder mapping)
        self.rule_out = rng.integers(0, len(self.out_sets), size=27)

    def infer(self, entropy: float, contrast: float, sharpness: float) -> float:
        # antecedent memberships (3 each)
        me = np.array([trapmf(entropy, *m) for m in self.mf["entropy"]])
        mc = np.array([trapmf(contrast, *m) for m in self.mf["contrast"]])
        ms = np.array([trapmf(sharpness, *m) for m in self.mf["sharpness"]])
        # 27 rule firing strengths = min over antecedents (combinatorial)
        # build via broadcasting then flatten
        fire = np.minimum.reduce(
            np.broadcast_arrays(me[:, None, None], mc[None, :, None], ms[None, None, :])
        ).ravel()  # length 27
        # aggregate clipped output sets (max), then centroid defuzzify
        agg = np.zeros_like(self.universe)
        for r in range(27):
            agg = np.maximum(agg, np.minimum(fire[r], self.out_mf[self.rule_out[r]]))
        denom = agg.sum()
        if denom <= 1e-12:
            return 0.0
        return float((self.universe * agg).sum() / denom)


# --------------------------------------------------------------------------- #
#  Timing harness
# --------------------------------------------------------------------------- #
def _now() -> float:
    return time.perf_counter()


def load_images(folder: str, limit: int,
                resize: Tuple[int, int] | None = None) -> List[np.ndarray]:
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        paths += glob.glob(os.path.join(folder, "**", ext), recursive=True)
    paths = sorted(paths)[:limit]
    imgs = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is not None:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            if resize is not None:
                # match the pipeline: bilinear resize to the operating resolution
                h, w = resize
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            imgs.append(im)
    return imgs


def synth_images(n: int, h: int, w: int, seed: int = 0) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8) for _ in range(n)]


def bench(images: List[np.ndarray], fis: FuzzyFIS, warmup: int = 10) -> Dict[str, float]:
    # warmup
    for im in images[:warmup]:
        e, c, s = extract_features(im)
        fis.infer(e, c, s)

    t_feat, t_inf, t_tot = [], [], []
    for im in images:
        t0 = _now()
        e, c, s = extract_features(im)
        t1 = _now()
        fis.infer(e, c, s)
        t2 = _now()
        t_feat.append((t1 - t0) * 1e3)   # ms
        t_inf.append((t2 - t1) * 1e3)
        t_tot.append((t2 - t0) * 1e3)

    def stats(a):
        a = np.array(a)
        return dict(median=float(np.median(a)),
                    mean=float(a.mean()),
                    p95=float(np.percentile(a, 95)))

    feat, inf, tot = stats(t_feat), stats(t_inf), stats(t_tot)
    tot_med_s = tot["median"] / 1e3
    return {
        "feat_ms": feat["median"], "inf_ms": inf["median"], "total_ms": tot["median"],
        "total_mean_ms": tot["mean"], "total_p95_ms": tot["p95"],
        "img_per_s": (1.0 / tot_med_s) if tot_med_s > 0 else float("inf"),
        "fps": (1.0 / tot_med_s) if tot_med_s > 0 else float("inf"),
        "n": len(images),
    }


def set_single_thread():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
#  Optional: deep NR-IQA baselines (MUSIQ / MANIQA / CLIP-IQA) via pyiqa
# --------------------------------------------------------------------------- #
def bench_deep(images: List[np.ndarray]) -> List[Dict]:
    try:
        import torch
        import pyiqa
    except Exception:
        print("\n[deep] pyiqa/torch not available -> skipping deep baselines "
              "(pip install pyiqa torch)")
        return []
    rows = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name in ("musiq", "maniqa", "clipiqa"):
        try:
            metric = pyiqa.create_metric(name, device=device)
        except Exception as ex:
            print(f"[deep] could not load {name}: {ex}")
            continue
        nparam = sum(p.numel() for p in metric.parameters()) if hasattr(metric, "parameters") else None
        # warmup
        t = torch.from_numpy(images[0].transpose(2, 0, 1)[None]).float() / 255.0
        t = t.to(device)
        for _ in range(3):
            with torch.no_grad():
                metric(t)
        if device == "cuda":
            torch.cuda.synchronize()
        ts = []
        for im in images[:200]:
            x = torch.from_numpy(im.transpose(2, 0, 1)[None]).float().to(device) / 255.0
            t0 = _now()
            with torch.no_grad():
                metric(x)
            if device == "cuda":
                torch.cuda.synchronize()
            ts.append((_now() - t0) * 1e3)
        med = float(np.median(ts))
        rows.append({"method": name, "device": device,
                     "ms_per_image": med, "img_per_s": 1e3 / med,
                     "params_M": (nparam / 1e6) if nparam else None})
    return rows


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None, help="folder of enhanced images")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--size", default="256x256", help="synthetic size HxW (pipeline uses 256x256)")
    ap.add_argument("--resize", default="256x256",
                    help="resize real --images to HxW before timing (pipeline operating "
                         "resolution); use 'none' to time at native resolution")
    ap.add_argument("--sweep", action="store_true", help="resolution sweep")
    ap.add_argument("--threads", default="single", choices=["single", "default"])
    ap.add_argument("--deep", action="store_true", help="also time deep baselines")
    ap.add_argument("--res", type=int, default=101, help="defuzz universe resolution")
    args = ap.parse_args()

    if args.threads == "single":
        set_single_thread()

    print("=" * 64)
    print("Reference-free fuzzy NR-IQA - runtime benchmark")
    print("=" * 64)
    print(f"CPU      : {platform.processor() or platform.machine()}")
    print(f"Python   : {platform.python_version()}   OpenCV: {cv2.__version__}")
    print(f"Threads  : {args.threads}")
    print(f"Defuzz   : centroid over {args.res} points")
    print("-" * 64)

    fis = FuzzyFIS(universe_res=args.res)
    results = []

    if args.images:
        rs = None if args.resize.lower() == "none" else tuple(
            int(x) for x in args.resize.lower().split("x"))
        imgs = load_images(args.images, args.limit, resize=rs)
        if not imgs:
            raise SystemExit(f"No images found under {args.images}")
        h, w = imgs[0].shape[:2]
        r = bench(imgs, fis)
        tag = "real images" if rs is None else f"real images, resized to {h}x{w}"
        r["resolution"] = f"{h}x{w} ({tag})"
        results.append(r)
    elif args.sweep:
        sizes = [(256, 256), (384, 384), (512, 512), (1024, 1024)]
        for (h, w) in sizes:
            imgs = synth_images(min(args.limit, 300), h, w)
            r = bench(imgs, fis)
            r["resolution"] = f"{h}x{w}"
            results.append(r)
    else:
        h, w = (int(x) for x in args.size.lower().split("x"))
        imgs = synth_images(args.limit, h, w)
        r = bench(imgs, fis)
        r["resolution"] = f"{h}x{w}"
        results.append(r)

    # report
    print(f"{'resolution':<20}{'feat ms':>9}{'infer ms':>10}"
          f"{'total ms':>10}{'img/s':>9}{'fps':>8}{'n':>6}")
    for r in results:
        print(f"{r['resolution']:<20}{r['feat_ms']:>9.3f}{r['inf_ms']:>10.4f}"
              f"{r['total_ms']:>10.3f}{r['img_per_s']:>9.0f}{r['fps']:>8.0f}{r['n']:>6d}")

    # CSV
    import csv
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "runtime_results.csv"
    with open(out, "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w_.writeheader()
        for r in results:
            w_.writerow(r)
    print(f"\nSaved: {out}")

    if args.deep:
        imgs = imgs if 'imgs' in dir() else synth_images(200, 256, 256)
        rows = bench_deep(imgs)
        if rows:
            print("\nDeep NR-IQA baselines:")
            print(f"{'method':<10}{'device':>8}{'ms/img':>10}{'img/s':>9}{'params(M)':>11}")
            for r in rows:
                pm = f"{r['params_M']:.1f}" if r['params_M'] else "n/a"
                print(f"{r['method']:<10}{r['device']:>8}{r['ms_per_image']:>10.2f}"
                      f"{r['img_per_s']:>9.1f}{pm:>11}")
            deep_out = out_dir / "runtime_deep.csv"
            with open(deep_out, "w", newline="") as f:
                w_ = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w_.writeheader()
                for r in rows:
                    w_.writerow(r)
            print(f"Saved: {deep_out}")


if __name__ == "__main__":
    main()
