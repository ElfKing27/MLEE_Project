"""Build a small, reproducible results folder suitable for committing to Git."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_results import mean_interval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("bo_comparison/sensitivity"))
    parser.add_argument("--output", type=Path, default=Path("bo_comparison/report"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("compare_methods.py")
    subprocess.run([sys.executable, str(script), "--input", str(args.input)],
                   check=True, capture_output=True, text=True,
                   env={**os.environ, "MPLBACKEND": "Agg"})
    summary = pd.read_csv(args.input / "balanced_comparison.csv")
    runs = pd.read_csv(args.input / "runs.csv")
    summary.to_csv(args.output / "comparison.csv", index=False, lineterminator="\n")
    for name in ("runs.csv", "paired_changes.csv"):
        (args.output / name).write_text((args.input / name).read_text(), encoding="utf-8", newline="\n")
    study = json.loads((args.input / "study.json").read_text())
    settings = {}
    for beta in (3, 2, 1):
        folder = args.input / f"beta_{beta}_budget_100"
        settings[str(beta)] = json.loads((folder / "manifest.json").read_text())
    metadata = {"study": study, "experiment_manifests": settings}
    (args.output / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n")

    devices = ["planar_L500_tox10", "planar_L1000_tox15", "planar_L2000_tox20"]
    labels = ["500 / 10 nm", "1,000 / 15 nm", "2,000 / 20 nm"]
    methods = [("EI", 3, "EI", "#CF503C"),
               ("SafeOpt", 3, "SafeOpt β=3", "#A5C8C1"),
               ("SafeOpt", 2, "SafeOpt β=2", "#4DAA9D"),
               ("SafeOpt", 1, "SafeOpt β=1", "#006C61")]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    panels = [("mean_violations", "Constraint violations", "Mean violating queries"),
              ("final_gap_mV", "Final safe recipe quality", "Gap to best safe pool objective (mV)"),
              ("distinct_recipes", "Exploration", "Mean distinct recipes queried")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for axis, (metric, title, ylabel) in zip(axes, panels):
        for offset, (method, beta, label, color) in enumerate(methods):
            values = summary[(summary.method == method) & (summary.beta == beta) & (summary.budget == 50)]
            values = values.set_index("device").loc[devices, metric]
            bars = axis.bar(np.arange(3) + (offset - 1.5) * .19, values, .18, color=color, label=label)
            axis.bar_label(bars, fmt="%.1f", fontsize=8, padding=3)
        axis.set_xticks(range(3), labels, rotation=10)
        axis.set(title=title, ylabel=ylabel, xlabel="Gate length / oxide thickness")
        axis.set_ylim(0, max(1, axis.get_ylim()[1]) * 1.15)
        axis.grid(axis="y", alpha=.2)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.suptitle("EI and SafeOpt at the same 50-query budget", y=.98, fontsize=17)
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.5, .91), ncol=4, frameon=False)
    fig.tight_layout(rect=(0, .03, 1, .82))
    fig.savefig(args.output / "comparison_50_queries.png", dpi=180)
    plt.close(fig)

    # Use paired initialization choices as the bootstrap sampling units.
    fig, axes = plt.subplots(3, 2, figsize=(11, 10.5))
    rng = np.random.default_rng(731)
    for row, device in enumerate(devices):
        for method, color in (("EI", "#CF503C"), ("SafeOpt", "#006C61")):
            paths = sorted((args.input / "beta_1_budget_100" / device).glob(f"run_*/{method}_curves.csv"))
            traces = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
            traces = traces[traces.evaluation <= 50]
            for column, metric in enumerate(("cumulative_violations", "simple_safe_regret")):
                values = traces.pivot(index="run", columns="evaluation", values=metric)
                scale = 1000 if column == 1 else 1
                mean, low, high = mean_interval(values.to_numpy() * scale, rng)
                axis = axes[row, column]
                axis.plot(values.columns, mean, color=color, label=method)
                axis.fill_between(values.columns, low, high, color=color, alpha=.15)
                axis.set(xlabel="Queries including the 3 safe starts",
                         ylabel="Cumulative violations" if column == 0 else "Best safe objective gap (mV)",
                         title=labels[row])
                axis.grid(alpha=.2)
                axis.legend()
    fig.suptitle("Learning curves for EI and SafeOpt β=1\n30 paired starts per device, means with 95% bootstrap bands", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(args.output / "learning_curves_beta_1.png", dpi=180)
    plt.close(fig)
    shutil.copyfile(args.input / "plots/balanced_beta_1_budget_50.png", args.output / "safety_and_quality.png")
    shutil.copyfile(args.input / "plots/sensitivity_overview.png", args.output / "budget_and_beta.png")
    print(f"Built the compact report in {args.output.resolve()}")


if __name__ == "__main__":
    main()
