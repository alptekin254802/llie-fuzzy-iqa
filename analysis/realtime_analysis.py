"""
realtime_analysis.py
====================
Derives the real-time *engineering* numbers quoted in the manuscript's
"Computational Efficiency and Real-Time Behaviour" section from the raw
benchmark outputs, so every figure in the text is reproducible:

  * median and 95th-percentile (worst-case / jitter) per-image latency
  * frame-budget occupancy at 30 / 60 fps  (latency / frame_budget)
  * resolution sweep (latency, throughput) and the minimum fps across it
  * live pipeline summary (sustained fps, per-stage medians) from results/pipeline_log.csv

Inputs (produced earlier):
  --runtime   results/runtime_results.csv   (from benchmark_runtime.py --sweep)
  --pipeline  results/pipeline_log.csv      (from realtime_pipeline_demo.py)

Usage:
  python analysis/realtime_analysis.py
  python analysis/realtime_analysis.py --runtime results/runtime_results.csv \
                                       --pipeline results/pipeline_log.csv
"""
from __future__ import annotations
import argparse, csv, statistics as st
from pathlib import Path


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def frame_budget_table(latency_ms, rates=(30, 60, 120)):
    rows = []
    for r in rates:
        budget = 1000.0 / r
        rows.append((r, budget, latency_ms, 100.0 * latency_ms / budget))
    return rows


def analyse_runtime(path):
    rows = _read_csv(path)
    # operating resolution row (256x256 if present, else first)
    op = next((r for r in rows if r.get("resolution", "").startswith("256")), rows[0])
    total = float(op["total_ms"])
    p95 = float(op.get("total_p95_ms", "nan"))
    print("=" * 64)
    print("PER-IMAGE LATENCY (operating resolution)")
    print("=" * 64)
    print(f"  resolution      : {op['resolution']}")
    print(f"  feature ms      : {float(op['feat_ms']):.3f}")
    print(f"  inference ms    : {float(op['inf_ms']):.4f}")
    print(f"  total median ms : {total:.3f}")
    print(f"  total p95 ms    : {p95:.3f}   (worst-case / jitter bound)")
    print(f"  throughput      : {float(op['img_per_s']):.0f} img/s")

    print("\nFRAME-BUDGET OCCUPANCY (median latency / frame budget)")
    for r, budget, lat, pct in frame_budget_table(total):
        print(f"  {r:3d} fps -> budget {budget:5.2f} ms : assessor uses {pct:4.1f} %")

    print("\nRESOLUTION SWEEP (real-time design trade-off)")
    print(f"  {'resolution':<12}{'feat ms':>9}{'infer ms':>10}{'total ms':>10}{'fps':>8}")
    min_fps = 1e9
    for r in rows:
        ips = float(r["img_per_s"])
        min_fps = min(min_fps, ips)
        print(f"  {r['resolution']:<12}{float(r['feat_ms']):>9.2f}"
              f"{float(r['inf_ms']):>10.3f}{float(r['total_ms']):>10.2f}{ips:>8.0f}")
    print(f"  -> minimum throughput across all tested resolutions: {min_fps:.0f} fps "
          f"({'>' if min_fps > 30 else '<='} 30 fps)")
    return total, p95


def analyse_pipeline(path):
    rows = _read_csv(path)
    cap = [float(r["cap_ms"]) for r in rows]
    enh = [float(r["enh_ms"]) for r in rows]
    ass = [float(r["assess_ms"]) for r in rows]
    tot = [float(r["total_ms"]) for r in rows]
    n = len(rows)
    sustained = n / (sum(tot) / 1e3) if sum(tot) > 0 else 0.0
    print("\n" + "=" * 64)
    print("LIVE PIPELINE (capture -> enhance -> assess)")
    print("=" * 64)
    print(f"  frames            : {n}")
    print(f"  capture  median ms: {st.median(cap):.3f}")
    print(f"  enhance  median ms: {st.median(enh):.3f}")
    print(f"  assess   median ms: {st.median(ass):.3f}   <-- proposed method")
    print(f"  total    median ms: {st.median(tot):.3f}")
    print(f"  sustained pipeline: {sustained:.1f} fps")
    print(f"  min instantaneous : {1000.0 / max(tot):.1f} fps "
          f"(stays {'above' if 1000.0/max(tot) > 30 else 'below'} 30 fps)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", default="results/runtime_results.csv")
    ap.add_argument("--pipeline", default="results/pipeline_log.csv")
    args = ap.parse_args()
    if Path(args.runtime).exists():
        analyse_runtime(args.runtime)
    else:
        print(f"[skip] {args.runtime} not found")
    if Path(args.pipeline).exists():
        analyse_pipeline(args.pipeline)
    else:
        print(f"[skip] {args.pipeline} not found")


if __name__ == "__main__":
    main()
