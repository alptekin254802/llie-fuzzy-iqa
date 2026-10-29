"""Regenerate both pipeline panels from the same recorded run."""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", type=Path, default=Path("results/pipeline_log.csv"))
    ap.add_argument("--output", type=Path, default=Path("results/figures"))
    args = ap.parse_args()
    data = pd.read_csv(args.log)
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"svg.fonttype": "none", "font.family": "serif", "font.size": 10})
    for name, values, label in (("fps", 1000 / data.total_ms, "Instantaneous throughput\n(frames/s)"),
                                ("score", data.score, "LPIPS-calibrated score")):
        fig, ax = plt.subplots(figsize=(6, 2.7))
        ax.plot(data.t_s, values, linewidth=.8)
        ax.set_xlabel("Time (s)"); ax.set_ylabel(label)
        if name == "fps":
            ax.axhline(30, linestyle="--", color="gray", linewidth=.8, label="30 frames/s")
            ax.legend(frameon=False)
        fig.tight_layout()
        for ext in ("pdf", "svg"):
            fig.savefig(args.output / f"fig6_pipeline_{name}.{ext}")
        plt.close(fig)


if __name__ == "__main__":
    main()
