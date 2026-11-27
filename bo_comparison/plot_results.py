"""Turn completed paired runs into CSV tables and Matplotlib figures."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"EI": "#CF503C", "SafeOpt": "#007F73"}
METRICS = [
    "violations", "queries_used", "final_simple_safe_regret",
    "cumulative_best_safe_gap", "zero_violations", "pool_optimum_found", "distinct_recipes",
]


def mean_interval(values, rng):
    """Percentile bootstrap interval for the mean of independent run pairs."""
    values = np.asarray(values)
    sampled = rng.integers(0, len(values), size=(3000, len(values)))
    bootstrap_means = values[sampled].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975], axis=0)
    return values.mean(axis=0), low, high


def make_tables(runs, output):
    rng = np.random.default_rng(731)
    summary_rows = []
    for (device, method), group in runs.groupby(["device", "method"], sort=False):
        record = {"device": device, "method": method, "runs": len(group)}
        for metric in METRICS:
            record[metric + "_mean"] = group[metric].mean()
            record[metric + "_std"] = group[metric].std(ddof=1)
        record["early_stops"] = int((group["stop_reason"] != "budget_reached").sum())
        record["runtime_s_mean"] = group["runtime_s"].mean()
        summary_rows.append(record)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "summary.csv", index=False)

    # Join on BOTH device and run so every difference uses matched seeds.
    ei = runs[runs.method == "EI"].set_index(["device", "run"])
    safe = runs[runs.method == "SafeOpt"].set_index(["device", "run"])
    differences = safe[METRICS] - ei[METRICS]
    differences.to_csv(output / "paired_differences.csv")
    intervals = []
    for device, group in differences.groupby(level="device", sort=False):
        for metric in METRICS:
            values = group[metric].to_numpy()
            mean, low, high = mean_interval(values, rng)
            intervals.append({
                "device": device, "metric": metric,
                "difference": "SafeOpt minus EI", "mean_difference": mean,
                "bootstrap_95_low": low, "bootstrap_95_high": high,
                "pairs": len(values), "negative_pairs": int((values < -1e-12).sum()),
                "tied_pairs": int((np.abs(values) <= 1e-12).sum()),
                "positive_pairs": int((values > 1e-12).sum()),
            })
    pd.DataFrame(intervals).to_csv(output / "paired_comparisons.csv", index=False)
    display = summary[[
        "device", "method", "runs", "violations_mean", "zero_violations_mean",
        "queries_used_mean", "final_simple_safe_regret_mean",
        "cumulative_best_safe_gap_mean", "distinct_recipes_mean", "early_stops",
    ]]
    report = (
        "Finite-pool EI vs SafeOpt comparison\n\n" + display.to_string(index=False) +
        "\n\nLower violations and regret are better. Higher zero-violation share is better.\n"
        "All intervals describe variation over random initial safe seeds on fixed pools.\n"
        "They do not describe uncertainty across fabricated devices or new process ranges.\n"
        "SafeOpt minus EI < 0 favors SafeOpt for violations and regret.\n"
        "Queries include the three initial seeds. After an early stop, best-so-far and\n"
        "violation totals carry forward to the common query-budget cap for plots.\n"
        "Cumulative best-safe gap sums incumbent gaps, not losses at queried recipes.\n"
        "Repeated queries cost one budget unit each. Distinct recipe counts show exploration.\n"
    )
    (output / "results.txt").write_text(report, encoding="utf-8")
    print(report)
    return summary


def plot_learning_curves(curves, output, title_suffix=""):
    rng = np.random.default_rng(810)
    panels = [
        ("cumulative_violations", "Constraint violations", "Cumulative count"),
        ("simple_safe_regret", "Best safe recipe gap", "Gap to best safe pool objective (V)"),
        ("cumulative_best_safe_gap", "Cumulative best-safe gap", "Sum of gaps (V × budget steps)"),
    ]
    for device, group in curves.groupby("device", sort=False):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
        for axis, (metric, title, ylabel) in zip(axes, panels):
            for method in ("EI", "SafeOpt"):
                matrix = group[group.method == method].pivot(
                    index="run", columns="evaluation", values=metric
                )
                mean, low, high = mean_interval(matrix.to_numpy(), rng)
                axis.plot(matrix.columns, mean, label=method, color=COLORS[method])
                axis.fill_between(matrix.columns, low, high, color=COLORS[method], alpha=.16)
            axis.set(title=title, xlabel="Query-budget step (including seeds)", ylabel=ylabel)
            axis.grid(alpha=.2)
            axis.legend()
        fig.suptitle(f"{device}{title_suffix} — means across paired runs with 95% bootstrap bands")
        fig.savefig(output / f"{device}_learning_curves.png", dpi=180)


def plot_summary(summary, runs, output):
    devices = list(summary.device.unique())
    labels = [d.replace("planar_", "").replace("_tox", " / oxide ") + " nm" for d in devices]
    x = np.arange(len(devices))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    panels = [
        ("violations_mean", "Violations per run", "Mean count", 1),
        ("zero_violations_mean", "Runs with zero violations", "Percent of runs", 100),
        ("distinct_recipes_mean", "Distinct recipes queried", "Mean unique recipe count", 1),
    ]
    for axis, (metric, title, ylabel, scale) in zip(axes, panels):
        for offset, method in zip((-.19, .19), ("EI", "SafeOpt")):
            rows = summary[summary.method == method].set_index("device").loc[devices]
            values = rows[metric].to_numpy() * scale
            bars = axis.bar(x + offset, values, .36, label=method, color=COLORS[method])
            axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
        axis.set_xticks(x, labels, rotation=12, ha="right", fontsize=9)
        axis.set(title=title, ylabel=ylabel)
        axis.set_ylim(0, max(1, axis.get_ylim()[1]) * 1.15)
        axis.grid(axis="y", alpha=.2)
    axes[0].legend()
    fig.savefig(output / "safety_and_budget.png", dpi=180)

    # One dot for each pair shows the performance/safety trade-off directly.
    ei = runs[runs.method == "EI"].set_index(["device", "run"])
    safe = runs[runs.method == "SafeOpt"].set_index(["device", "run"])
    delta = safe[METRICS] - ei[METRICS]
    fig, axes = plt.subplots(1, len(devices), figsize=(14, 4.4), constrained_layout=True, squeeze=False)
    for axis, device in zip(axes[0], devices):
        points = delta.loc[device]
        axis.scatter(points.violations, points.cumulative_best_safe_gap, color=COLORS["SafeOpt"], alpha=.65)
        axis.axhline(0, color="gray", linewidth=1)
        axis.axvline(0, color="gray", linewidth=1)
        axis.set(title=device, xlabel="Violations difference (SafeOpt − EI)",
                 ylabel="Cumulative best-safe gap difference")
        axis.grid(alpha=.2)
    fig.suptitle("Paired differences — lower left favors SafeOpt on both measures")
    fig.savefig(output / "paired_tradeoff.png", dpi=180)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs_bo"))
    parser.add_argument("--show", action="store_true", help="Open native Matplotlib windows")
    args = parser.parse_args()
    runs = pd.read_csv(args.input / "runs.csv")
    manifest = json.loads((args.input / "manifest.json").read_text())
    expected = len(manifest["datasets"]) * manifest["settings"]["runs"] * 2
    if len(runs) != expected or runs.duplicated(["device", "run", "method"]).any():
        raise ValueError("Missing or duplicate method results. Resume the experiment first.")
    frames = []
    for device, run, method in runs[["device", "run", "method"]].itertuples(index=False, name=None):
        frames.append(pd.read_csv(args.input / device / f"run_{run:02d}" / f"{method}_curves.csv"))
    curves = pd.concat(frames, ignore_index=True)
    budget = manifest["settings"]["budget"]
    if len(curves) != expected * budget:
        raise ValueError("Incomplete learning curves")
    plots = args.input / "plots"
    plots.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    summary = make_tables(runs, args.input)
    plot_learning_curves(curves, plots)
    plot_summary(summary, runs, plots)
    if args.show:
        plt.show()
    plt.close("all")
    print(f"Plots saved in {plots.resolve()}")


if __name__ == "__main__":
    main()
