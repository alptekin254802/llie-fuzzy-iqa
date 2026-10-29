"""Summarize measured stage latencies and recorded capture-enhance-assess logs."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def analyse_runtime(path):
    data = json.loads(Path(path).read_text())
    for stage, row in data["pooled"].items():
        print(f"{stage:40s} median={row['median_ms']:.4f} ms  p95={row['p95_ms']:.4f} ms")
    print("p95 is an empirical percentile, not a worst-case bound. Stage medians are not additive.")


def analyse_pipeline(path):
    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))
    for stage in ("cap_ms", "enh_ms", "assess_ms", "total_ms"):
        print(f"{stage}: median={statistics.median(float(r[stage]) for r in rows):.4f} ms")
    elapsed = sum(float(r["total_ms"]) for r in rows) / 1000
    print(f"Recorded pipeline throughput: {len(rows) / elapsed:.4f} frames/s ({len(rows)} frames)")
    print("This measures capture-enhance-assess, not multi-candidate parameter search.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runtime", type=Path, default=Path("results/timing_benchmark.json"))
    ap.add_argument("--pipeline", type=Path, default=Path("results/pipeline_log.csv"))
    args = ap.parse_args()
    analyse_runtime(args.runtime)
    analyse_pipeline(args.pipeline)


if __name__ == "__main__":
    main()
