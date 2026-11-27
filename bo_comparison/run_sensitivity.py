"""Compare confidence settings and budgets using the same initial recipes.

Run 100 queries, then score both the 50-query prefix and the complete run.
This is equivalent to stopping the same sequential policy at either budget.
"""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from run_experiment import file_hash, load_pool, save_json, score_trace
from verify_results import main as verify_saved_runs


def read_trace(path, budget):
    with path.open(newline="") as stream:
        trace = list(csv.DictReader(stream))[:budget]
    for row in trace:
        row["evaluation"] = int(row["evaluation"])
        row["candidate_index"] = int(row["candidate_index"])
        row["unsafe"] = int(row["unsafe"])
        row["objective"] = float(row["objective"])
    return trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs_parallel_100"))
    parser.add_argument("--output", type=Path, default=Path("bo_comparison/sensitivity"))
    parser.add_argument("--baseline", type=Path, default=Path("bo_comparison/results_equal_budget"))
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    source = Path(__file__).parent
    max_budget = 100
    betas = [3.0, 2.0, 1.0]
    budgets = [50, 100]
    args.output.mkdir(parents=True, exist_ok=True)

    # Finish the high-budget runs first. No policy knows its future outcomes.
    for beta in betas:
        folder = args.output / f"beta_{beta:g}_budget_{max_budget}"
        print(f"\nRunning beta={beta:g}, budget={max_budget}", flush=True)
        command = [sys.executable, str(source / "run_experiment.py"),
                   "--input", str(args.input), "--output", str(folder),
                   "--runs", str(args.runs), "--workers", str(args.workers),
                   "--budget", str(max_budget), "--beta", str(beta)]
        if (folder / "manifest.json").exists():
            command.append("--resume")
        with (args.output / f"beta_{beta:g}_run.log").open("a", encoding="utf-8") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        verify_saved_runs(["--datasets", str(args.input), "--results", str(folder)])
        print(f"Completed and audited beta={beta:g}", flush=True)

    # Confirm that EI is unchanged by beta, which only affects SafeOpt.
    records = []
    reference_ei = {}
    baseline = args.baseline / "manifest.json"
    baseline_manifest = json.loads(baseline.read_text()) if baseline.exists() else None
    inputs = []
    for path in sorted(args.input.glob("*/dataset.csv")):
        rows = load_pool(path)
        device = path.parent.name
        inputs.append({"device": device, "sha256": file_hash(path)})
        for run in range(args.runs):
            seeds_by_beta = []
            for beta in betas:
                folder = args.output / f"beta_{beta:g}_budget_{max_budget}" / device / f"run_{run:02d}"
                pair = json.loads((folder / "pair.json").read_text())
                seeds_by_beta.append(pair[0]["seed_indices"])
                for method in ("EI", "SafeOpt"):
                    trace = read_trace(folder / f"{method}_queries.csv", max_budget)
                    ids = [row["evaluation_id"] for row in trace]
                    if method == "EI":
                        key = (device, run)
                        if beta == 3:
                            reference_ei[key] = ids
                        elif reference_ei[key] != ids:
                            raise ValueError("EI changed when only SafeOpt beta changed")
                    # Check the original 50-query experiment against its prefix.
                    if beta == 3 and baseline_manifest and run < baseline_manifest["settings"]["runs"]:
                        original = args.baseline / device / f"run_{run:02d}" / f"{method}_queries.csv"
                        original_ids = [r["evaluation_id"] for r in read_trace(original, 50)]
                        if original_ids != ids[:50]:
                            raise ValueError("The original baseline does not match its 100-query prefix")
                    for budget in budgets:
                        curves, summary = score_trace(rows, trace[:budget], budget)
                        records.append({
                            "device": device, "run": run, "beta": beta,
                            "budget": budget, "method": method, **summary,
                            "mean_best_safe_gap": summary["cumulative_best_safe_gap"] / budget,
                        })
            if any(seeds != seeds_by_beta[0] for seeds in seeds_by_beta):
                raise ValueError("Initial recipes changed across confidence settings")
    results = pd.DataFrame(records)
    results.to_csv(args.output / "runs.csv", index=False)
    metrics = ["violations", "zero_violations", "distinct_recipes", "repeat_queries",
               "queries_used", "final_simple_safe_regret", "cumulative_best_safe_gap",
               "mean_best_safe_gap", "pool_optimum_found"]
    grouped = results.groupby(["device", "beta", "budget", "method"], sort=False)
    summary = grouped[metrics].mean().reset_index()
    summary.to_csv(args.output / "summary.csv", index=False)

    # Each bootstrap resample selects complete matched runs.
    rng = np.random.default_rng(821)
    contrasts = []
    indexed = results.set_index(["device", "run", "beta", "budget", "method"])
    for device in results.device.unique():
        for beta in betas:
            for method in ("EI", "SafeOpt"):
                first = indexed.xs((device, beta, 50, method), level=("device", "beta", "budget", "method"))
                second = indexed.xs((device, beta, 100, method), level=("device", "beta", "budget", "method"))
                comparisons = [(f"budget 100 minus 50 at beta {beta:g}", second, first)]
                if method == "SafeOpt" and beta != 3:
                    for budget in budgets:
                        lower = indexed.xs((device, beta, budget, method), level=("device", "beta", "budget", "method"))
                        higher = indexed.xs((device, 3.0, budget, method), level=("device", "beta", "budget", "method"))
                        comparisons.append((f"beta {beta:g} minus 3 at budget {budget}", lower, higher))
                for label, changed, reference in comparisons:
                    for metric in ("violations", "final_simple_safe_regret", "mean_best_safe_gap", "distinct_recipes"):
                        delta = (changed[metric] - reference[metric]).to_numpy()
                        sampled = rng.integers(len(delta), size=(3000, len(delta)))
                        low, high = np.quantile(delta[sampled].mean(axis=1), [.025, .975])
                        contrasts.append({"device": device, "method": method, "comparison": label,
                                          "metric": metric, "mean_difference": delta.mean(),
                                          "bootstrap_95_low": low, "bootstrap_95_high": high})
    pd.DataFrame(contrasts).to_csv(args.output / "paired_changes.csv", index=False)
    save_json(args.output / "study.json", {
        "betas": betas, "budgets": budgets, "runs_per_device": args.runs,
        "datasets": inputs, "baseline_prefix_checked": baseline_manifest is not None,
        "scoring": "50-query prefixes and full 100-query traces of the same policies",
        "beta_meaning": "Direct standard-deviation multiplier on GP confidence bounds",
        "repeats": "Allowed for both methods; every query costs one budget unit",
        "settings": "All other parameters match the original fixed-GP experiment",
        "uncertainty": "Paired bootstrap over initial seed choices on fixed datasets",
        "source_sha256": file_hash(Path(__file__)),
    })
    print(summary.to_string(index=False), flush=True)
    print(f"Saved sensitivity tables to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
