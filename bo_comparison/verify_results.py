"""Audit completed runs against the original simulation CSV files."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from run_experiment import file_hash, load_pool, score_trace


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, default=Path("outputs_parallel_100"))
    parser.add_argument("--results", type=Path, default=Path("outputs_bo"))
    args = parser.parse_args(argv)
    manifest = json.loads((args.results / "manifest.json").read_text())
    budget = manifest["settings"]["budget"]
    number_of_runs = manifest["settings"]["runs"]
    initial = manifest["settings"]["initial_safe"]
    checked = 0
    for dataset in manifest["datasets"]:
        device = dataset["device"]
        source = args.datasets / device / "dataset.csv"
        assert file_hash(source) == dataset["sha256"], "The source dataset changed"
        rows = load_pool(source)
        for run in range(number_of_runs):
            folder = args.results / device / f"run_{run:02d}"
            ei, safe = json.loads((folder / "pair.json").read_text())
            assert ei["seed_indices"] == safe["seed_indices"], "Unpaired starting recipes"
            assert all(not rows[i]["unsafe"] for i in ei["seed_indices"])
            for summary in (ei, safe):
                method = summary["method"]
                with (folder / f"{method}_queries.csv").open(newline="") as stream:
                    trace = list(csv.DictReader(stream))
                assert len(trace) == summary["queries_used"] <= budget
                assert len({row["candidate_index"] for row in trace}) == summary["distinct_recipes"]
                assert [int(r["candidate_index"]) for r in trace[:initial]] == ei["seed_indices"]
                for step, result in enumerate(trace, 1):
                    oracle = rows[int(result["candidate_index"])]
                    assert int(result["evaluation"]) == step
                    assert result["evaluation_id"] == oracle["evaluation_id"]
                    assert int(result["unsafe"]) == oracle["unsafe"]
                    for name in ("objective", "safety_margin", "vth_V", "ioff_A_per_um"):
                        result[name] = float(result[name])
                        assert np.isclose(result[name], oracle[name], rtol=1e-12, atol=1e-25)
                    result["unsafe"] = int(result["unsafe"])
                    if method == "SafeOpt" and step > initial:
                        earlier = {int(r["candidate_index"]) for r in trace[:step - 1]}
                        assert float(result["predicted_safety_lower"]) >= 0 or int(result["candidate_index"]) in earlier
                curves, recomputed = score_trace(rows, trace, budget)
                for name in ("violations", "queries_used", "final_simple_safe_regret", "cumulative_best_safe_gap"):
                    assert np.isclose(summary[name], recomputed[name], rtol=1e-12)
                with (folder / f"{method}_curves.csv").open(newline="") as stream:
                    recorded = list(csv.DictReader(stream))
                assert len(recorded) == budget
                for expected, actual in zip(curves, recorded):
                    for key, value in expected.items():
                        assert np.isclose(float(actual[key]), value, rtol=1e-12, atol=1e-14)
                checked += 1
    print(f"Verified {checked} method runs, matched seeds, query counts, oracle outcomes, and all learning curves.")


if __name__ == "__main__":
    main()
