"""Replay EI and SafeOpt on saved simulation results. No TCAD is launched.

Example from the project root:
    python bo_comparison/run_experiment.py --input outputs_parallel_100 \
        --output outputs_bo --runs 30 --budget 50 --workers 3
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

# Each worker solves tiny GP systems. Large BLAS thread pools slow these down.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from algorithms import choose_recipe


INPUT_COLUMNS = ["implant_dose_cm2", "anneal_time_s", "anneal_temperature_C"]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_pool(path):
    """Validate the finite benchmark before running any optimizer."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    identities = set()
    recipes = set()
    for row in rows:
        for key in INPUT_COLUMNS + ["objective", "safety_margin", "vth_V", "ioff_A_per_um"]:
            row[key] = float(row[key])
            if not np.isfinite(row[key]):
                raise ValueError(f"Nonfinite {key} in {path}")
        if row["solver_status"] != "ok" or row["extraction_status"] != "ok":
            raise ValueError(f"Unresolved simulation in {path}")
        if row["constraint_violated"] not in ("True", "False"):
            raise ValueError(f"Unknown safety label in {path}")
        row["unsafe"] = row["constraint_violated"] == "True"
        if row["unsafe"] != (row["safety_margin"] < 0.0):
            raise ValueError(f"Label disagrees with margin in {path}")
        if row["implant_dose_cm2"] <= 0.0 or row["ioff_A_per_um"] <= 0.0:
            raise ValueError(f"Nonpositive dose or leakage in {path}")
        limit = float(row["leakage_limit_A_per_um"])
        target = float(row["target_vth_V"])
        if not np.isclose(row["safety_margin"], np.log(limit / row["ioff_A_per_um"])):
            raise ValueError(f"Margin disagrees with leakage in {path}")
        if not np.isclose(row["objective"], -abs(row["vth_V"] - target)):
            raise ValueError(f"Objective disagrees with threshold voltage in {path}")
        identities.add(row["evaluation_id"])
        recipes.add(tuple(row[key] for key in INPUT_COLUMNS))
    if len(identities) != len(rows) or len(recipes) != len(rows):
        raise ValueError(f"Duplicate evaluation or recipe in {path}")
    if {row["device_id"] for row in rows} != {path.parent.name}:
        raise ValueError(f"Mixed or incorrect device IDs in {path}")
    if np.mean([row["unsafe"] for row in rows]) <= 0.3:
        raise ValueError(f"Pool does not exceed 30% unsafe: {path}")
    return rows


def normalize_inputs(rows):
    """Use coordinates only, without looking at unqueried responses."""
    X = np.array([[row[key] for key in INPUT_COLUMNS] for row in rows])
    X[:, 0] = np.log10(X[:, 0])
    minimum = X.min(axis=0)
    span = X.max(axis=0) - minimum
    span[span == 0.0] = 1.0
    return (X - minimum) / span


def run_method(rows, X, seeds, method, settings):
    """Select using observations, then reveal one CSV outcome per query."""
    observed = []
    objectives = []
    margins = []
    trace = []
    stop_reason = "budget_reached"
    objective_center = None
    start = time.perf_counter()
    for step in range(settings["budget"]):
        if step < len(seeds):
            index = int(seeds[step])
            prediction = {
                "selection_role": "known_safe_seed",
                "predicted_objective": "", "objective_std": "",
                "acquisition_value": "", "eligible_candidates": "",
                "predicted_safety_lower": "", "predicted_safety_upper": "",
            }
        else:
            if objective_center is None:
                objective_center = float(np.mean(objectives))
            index, prediction = choose_recipe(
                X, observed, objectives, margins, method, settings, objective_center
            )
            if index is None:
                stop_reason = prediction["stop_reason"]
                break
        # This is the only place the optimizer receives a new true response.
        result = rows[index]
        observed.append(index)
        objectives.append(result["objective"])
        margins.append(result["safety_margin"])
        trace.append({
            "evaluation": step + 1, "candidate_index": index,
            "evaluation_id": result["evaluation_id"],
            **{key: result[key] for key in INPUT_COLUMNS},
            "objective": result["objective"], "safety_margin": result["safety_margin"],
            "unsafe": int(result["unsafe"]), "vth_V": result["vth_V"],
            "ioff_A_per_um": result["ioff_A_per_um"], **prediction,
        })
    return trace, stop_reason, time.perf_counter() - start


def score_trace(rows, trace, budget):
    """The scoring code can see the full oracle. Selection code cannot."""
    optimum = max(row["objective"] for row in rows if not row["unsafe"])
    best_safe = -np.inf
    violations = 0
    curves = []
    cumulative_gap = 0.0
    best_id = ""
    for step in range(budget):
        queried = step < len(trace)
        if queried:
            result = trace[step]
            violations += result["unsafe"]
            if not result["unsafe"] and result["objective"] > best_safe:
                best_safe = result["objective"]
                best_id = result["evaluation_id"]
        gap = max(0.0, optimum - best_safe)
        cumulative_gap += gap
        curves.append({
            "evaluation": step + 1, "queried": int(queried),
            "queries_used": min(step + 1, len(trace)),
            "cumulative_violations": violations,
            "best_safe_objective": best_safe, "simple_safe_regret": gap,
            "cumulative_best_safe_gap": cumulative_gap,
        })
    summary = {
        "queries_used": len(trace), "violations": violations,
        "distinct_recipes": len({row["candidate_index"] for row in trace}),
        "repeat_queries": len(trace) - len({row["candidate_index"] for row in trace}),
        "zero_violations": int(violations == 0),
        "first_violation_evaluation": next((r["evaluation"] for r in trace if r["unsafe"]), ""),
        "final_best_safe_objective": best_safe,
        "final_simple_safe_regret": curves[-1]["simple_safe_regret"],
        "cumulative_best_safe_gap": cumulative_gap,
        "pool_best_safe_objective": optimum, "best_safe_evaluation_id": best_id,
        "pool_optimum_found": int(np.isclose(best_safe, optimum, rtol=0, atol=1e-12)),
    }
    return curves, summary


def run_pair(job):
    """One worker runs both methods on exactly the same starting recipes."""
    dataset, output, run, settings = job
    rows = load_pool(Path(dataset))
    device = rows[0]["device_id"]
    folder = Path(output) / device / f"run_{run:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    safe_indices = np.array([i for i, row in enumerate(rows) if not row["unsafe"]])
    gate_length = int(rows[0]["gate_length_nm"])
    rng = np.random.default_rng(np.random.SeedSequence([settings["seed"], gate_length, run]))
    seeds = rng.choice(safe_indices, size=settings["initial_safe"], replace=False).tolist()
    X = normalize_inputs(rows)
    result = []
    with threadpool_limits(limits=1):
        for method in ("EI", "SafeOpt"):
            trace, stop_reason, seconds = run_method(rows, X, seeds, method, settings)
            curves, summary = score_trace(rows, trace, settings["budget"])
            identifiers = {"device": device, "run": run, "method": method}
            write_csv(folder / f"{method}_queries.csv", [{**identifiers, **r} for r in trace])
            write_csv(folder / f"{method}_curves.csv", [{**identifiers, **r} for r in curves])
            result.append({
                **identifiers, **summary, "stop_reason": stop_reason,
                "runtime_s": seconds, "seed_indices": seeds,
                "seed_evaluation_ids": [rows[i]["evaluation_id"] for i in seeds],
            })
    save_json(folder / "pair.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs_parallel_100"))
    parser.add_argument("--output", type=Path, default=Path("outputs_bo"))
    parser.add_argument("--device", help="Optional one-device folder name")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--initial-safe", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--length-scale", type=float, default=0.4)
    parser.add_argument("--objective-scale", type=float, default=0.25)
    parser.add_argument("--safety-scale", type=float, default=10.0)
    parser.add_argument("--noise-variance", type=float, default=1e-6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    settings = vars(args).copy()
    for name in ("input", "output", "device", "resume", "workers"):
        settings.pop(name)
    if not (args.runs >= 1 and args.workers >= 1 and 1 <= args.initial_safe <= args.budget):
        parser.error("Use positive runs/workers and 1 <= initial-safe <= budget")
    if args.seed < 0 or any(settings[k] <= 0 for k in (
        "beta", "length_scale", "objective_scale", "safety_scale", "noise_variance"
    )):
        parser.error("Seed must be nonnegative and GP parameters must be positive")
    datasets = sorted(args.input.glob("*/dataset.csv"))
    if args.device:
        datasets = [p for p in datasets if p.parent.name == args.device]
    if not datasets:
        parser.error("No final dataset.csv files found under --input")
    inputs = []
    for path in datasets:
        rows = load_pool(path)
        safe = sum(not row["unsafe"] for row in rows)
        if args.budget > len(rows) or args.initial_safe > safe:
            parser.error(f"Insufficient candidates or safe seeds in {path}")
        inputs.append({
            "device": path.parent.name, "sha256": file_hash(path),
            "recipes": len(rows), "safe": safe, "unsafe": len(rows) - safe,
        })
    source = Path(__file__).parent
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "scikit-learn": sklearn.__version__}
    manifest = {
        "settings": settings, "datasets": inputs, "versions": versions,
        "source_sha256": {name: file_hash(source / name) for name in (
            "algorithms.py", "run_experiment.py"
        )},
        "domain": "100-point finite pools, no interpolation or TCAD calls",
        "initialization": "Uniform random choice from known-safe pool rows, shared within pair",
        "models": "Fixed Matern 5/2, no response-based hyperparameter tuning",
        "safety_confidence": "Empirical beta multiplier, no formal no-fail guarantee",
        "query_policy": "Both methods may repeat recipes; every query including repeats and seeds costs one budget unit",
        "regret": "Best-safe gap and its cumulative sum, not cumulative queried-point regret",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            parser.error("Output exists. Use --resume or choose a new output directory")
        if json.loads(manifest_path.read_text()) != manifest:
            parser.error("Resume settings, data, code, or package versions differ")
    else:
        if any(args.output.iterdir()):
            parser.error("Output is nonempty without a manifest. Choose a new folder")
        save_json(manifest_path, manifest)

    jobs = []
    for path in datasets:
        for run in range(args.runs):
            completed = args.output / path.parent.name / f"run_{run:02d}" / "pair.json"
            if not completed.exists():
                jobs.append((str(path.resolve()), str(args.output.resolve()), run, settings))
    print(f"Running {len(jobs)} pairs with {args.workers} workers", flush=True)
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_pair, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            pair = future.result()
            print(f"[{done}/{len(jobs)}] {pair[0]['device']} run {pair[0]['run']:02d}: "
                  f"EI {pair[0]['violations']} violations / {pair[0]['queries_used']} queries; "
                  f"SafeOpt {pair[1]['violations']} / {pair[1]['queries_used']}", flush=True)
    summaries = []
    for path in datasets:
        for run in range(args.runs):
            summaries.extend(json.loads((args.output / path.parent.name / f"run_{run:02d}" / "pair.json").read_text()))
    write_csv(args.output / "runs.csv", summaries)
    print(f"Finished in {time.perf_counter() - start:.1f} s. Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
