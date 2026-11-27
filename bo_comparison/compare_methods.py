"""Compare optimization quality and safety together using saved study runs."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("bo_comparison/sensitivity"))
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    runs = pd.read_csv(args.input / "runs.csv")
    # This joint metric enforces the original requirement throughout the run.
    runs["safe_optimum_success"] = (
        (runs["zero_violations"] == 1) & (runs["pool_optimum_found"] == 1)
    ).astype(int)
    group_keys = ["device", "beta", "budget", "method"]
    summary = runs.groupby(group_keys, sort=False).agg(
        runs=("run", "size"),
        mean_violations=("violations", "mean"),
        zero_violation_fraction=("zero_violations", "mean"),
        optimum_found_fraction=("pool_optimum_found", "mean"),
        safe_optimum_successes=("safe_optimum_success", "sum"),
        safe_optimum_success_fraction=("safe_optimum_success", "mean"),
        final_gap_V=("final_simple_safe_regret", "mean"),
        average_gap_V=("mean_best_safe_gap", "mean"),
        distinct_recipes=("distinct_recipes", "mean"),
        repeats=("repeat_queries", "mean"),
    ).reset_index()
    summary["final_gap_mV"] = 1000 * summary.final_gap_V
    summary["average_gap_mV"] = 1000 * summary.average_gap_V
    summary.to_csv(args.input / "balanced_comparison.csv", index=False)

    chosen = summary[(summary.beta == args.beta) & (summary.budget == args.budget)]
    if chosen.empty:
        parser.error("The requested beta and budget are not in the study")
    devices = sorted(chosen.device.unique(), key=lambda name: int(name.split("_L")[1].split("_")[0]))
    labels = [name.split("_L")[1].split("_")[0] + " nm" for name in devices]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    panels = [
        ("zero_violation_fraction", "Completed without violations", "Percent of runs", 100),
        ("final_gap_mV", "Final best safe recipe gap", "Mean gap to best safe pool objective (mV)", 1),
        ("safe_optimum_success_fraction", "Found pool optimum with zero violations", "Percent of runs", 100),
    ]
    x = np.arange(len(devices))
    for axis, (metric, title, ylabel, scale) in zip(axes, panels):
        for method, offset, color in (("EI", -.19, "#CF503C"), ("SafeOpt", .19, "#007F73")):
            rows = chosen[chosen.method == method].set_index("device").loc[devices]
            bars = axis.bar(x + offset, rows[metric] * scale, .36, color=color, label=method)
            axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
        axis.set_xticks(x, labels)
        axis.set(title=title, ylabel=ylabel, xlabel="Gate length")
        axis.set_ylim(0, max(1, axis.get_ylim()[1]) * 1.15)
        axis.grid(axis="y", alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend()
    fig.suptitle(f"Safety and optimization together — {args.budget} queries, SafeOpt β = {args.beta:g}")
    destination = args.input / "plots" / f"balanced_beta_{args.beta:g}_budget_{args.budget}.png"
    fig.savefig(destination, dpi=180)
    if args.show:
        plt.show()
    plt.close(fig)
    report = chosen[["device", "method", "mean_violations", "final_gap_mV", "average_gap_mV",
                     "safe_optimum_successes", "runs", "distinct_recipes"]].to_string(index=False)
    text = (
        f"Safety and performance at beta={args.beta:g}, budget={args.budget}\n\n{report}\n\n"
        "Final gap measures distance from the best safe pool objective. Average gap\n"
        "measures how quickly good safe recipes were found over the budget. Lower is better.\n"
        "A joint success means the run found the exact pool optimum and had zero violations.\n"
        "Initial recipes count, including cases that already started at the optimum.\n"
        "For a hard no-fail requirement, a run with any violation fails the safety criterion.\n"
        "The offline replay continues after violations to measure diagnostic performance.\n"
        "Beta 1 was the best confidence setting explored in this study. These are\n"
        "exploratory results on fixed finite pools, with repeated queries charged equally.\n"
        "They do not establish a universal ranking or a hardware safety guarantee.\n"
    )
    (args.input / "balanced_interpretation.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
