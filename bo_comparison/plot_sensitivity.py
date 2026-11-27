"""Plot the effect of budget and confidence on safety and optimization."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_results import mean_interval, plot_learning_curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("bo_comparison/sensitivity"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    study = json.loads((args.input / "study.json").read_text())
    runs = pd.read_csv(args.input / "runs.csv")
    betas = study["betas"]
    budgets = study["budgets"]
    devices = sorted(runs.device.unique(), key=lambda name: int(name.split("_L")[1].split("_")[0]))
    plots = args.input / "plots"
    plots.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    rng = np.random.default_rng(182)
    panels = [
        ("violations", "Constraint violations", "Mean count", 1),
        ("final_simple_safe_regret", "Final best safe recipe gap", "Final safe gap (mV)", 1000),
        ("distinct_recipes", "Exploration", "Mean distinct recipes", 1),
    ]
    colors = {50: "#626B75", 100: "#007F73"}
    fig, axes = plt.subplots(len(devices), 3, figsize=(15, 10), squeeze=False)
    x = np.arange(len(betas))
    for row, device in enumerate(devices):
        for axis, (metric, title, ylabel, scale) in zip(axes[row], panels):
            for budget in budgets:
                means, lows, highs = [], [], []
                for beta in betas:
                    selected = runs[(runs.device == device) & (runs.beta == beta) &
                                    (runs.budget == budget) & (runs.method == "SafeOpt")]
                    mean, low, high = mean_interval(selected[metric].to_numpy() * scale, rng)
                    means.append(mean)
                    lows.append(low)
                    highs.append(high)
                axis.plot(x, means, marker="o", label=f"SafeOpt, {budget} queries", color=colors[budget],
                          linestyle="-" if budget == 50 else "--", markersize=8 if budget == 50 else 4,
                          markerfacecolor="none" if budget == 50 else colors[budget])
                axis.fill_between(x, lows, highs, color=colors[budget], alpha=.12)
                baseline = runs[(runs.device == device) & (runs.beta == 3) &
                                (runs.budget == budget) & (runs.method == "EI")]
                axis.axhline(baseline[metric].mean() * scale, color=colors[budget],
                             linestyle=":" if budget == 50 else "--", alpha=.7,
                             label=f"EI, {budget} queries")
            axis.set_xticks(x, [f"β = {beta:g}" for beta in betas])
            axis.set(title=f"{device}\n{title}", xlabel="Confidence multiplier β", ylabel=ylabel)
            axis.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Budget and safety-confidence sensitivity", y=.995, fontsize=17)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .972), ncol=4, frameon=False)
    fig.text(.5, .922, "Means with 95% bootstrap bands. SafeOpt outcomes overlap at 50 and 100 queries.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .915), h_pad=2, w_pad=2)
    fig.savefig(plots / "sensitivity_overview.png", dpi=180, bbox_inches="tight")

    # Keep the familiar EI-vs-SafeOpt learning curves for every beta setting.
    for beta in betas:
        case = args.input / f"beta_{beta:g}_budget_100"
        frames = [pd.read_csv(path) for path in sorted(case.glob("*/run_*/*_curves.csv"))]
        curves = pd.concat(frames, ignore_index=True)
        case_plots = plots / f"beta_{beta:g}"
        case_plots.mkdir(exist_ok=True)
        plot_learning_curves(curves, case_plots, title_suffix=f", β = {beta:g}")
    if args.show:
        plt.show()
    plt.close("all")
    print(f"Saved sensitivity overview and per-setting learning curves to {plots.resolve()}")


if __name__ == "__main__":
    main()
