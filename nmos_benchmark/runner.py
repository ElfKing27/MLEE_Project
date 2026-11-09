"""Reproducible pilot-domain selection and independent validation dataset."""
from __future__ import annotations

import argparse
import copy
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = ("implant_dose_cm2", "anneal_time_s", "anneal_temperature_C")


def merge(base: dict, updates: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_devices(path: Path) -> list[dict]:
    document = json.loads(path.read_text())
    devices = [merge(document["defaults"], d) for d in document["devices"]]
    ids = [d["id"] for d in devices]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", i) for i in ids):
        raise ValueError("Device IDs must be unique, simple filenames.")
    for device in devices:
        for bounds in device["domain_candidates"]:
            for parameter in PARAMETERS:
                low, high = bounds[parameter]
                hard_low, hard_high = device["hard_bounds"][parameter]
                if not hard_low <= low < high <= hard_high:
                    raise ValueError(f"Candidate domain exceeds hard bounds: {parameter}")
        if device["electrical"]["leakage_limit_A_per_um"] <= 0:
            raise ValueError("Leakage limit must be positive.")
    return devices


def sample(bounds: dict, count: int, seed: int) -> list[dict]:
    # IID uniform in LOG DOSE, LINEAR TIME, LINEAR TEMPERATURE. This explicit
    # measure defines the unsafe fraction; it is not a uniform physical-dose claim.
    rng = np.random.default_rng(seed)
    values = rng.random((count, 3))
    recipes = []
    for point in values:
        recipe = {}
        for j, parameter in enumerate(PARAMETERS):
            low, high = bounds[parameter]
            if parameter == "implant_dose_cm2":
                low, high = np.log10([low, high])
                value = 10 ** (low + point[j] * (high - low))
            else:
                value = low + point[j] * (high - low)
            recipe[parameter] = float(value)
        recipes.append(recipe)
    return recipes


def summarize(rows: list[dict], alpha: float = 0.05) -> dict:
    n = len(rows)
    bad = sum(r.get("constraint_violated") is True for r in rows)
    good = sum(r.get("constraint_violated") is False for r in rows)
    unknown = n - bad - good
    # Unknowns are treated as NON-violations for the lower bound, so numerical
    # failures cannot help meet the benchmark. No failures are silently removed.
    lower = 0.0 if bad == 0 else float(beta.ppf(alpha, bad, n - bad + 1))
    upper = 1.0 if bad + unknown == n else float(beta.ppf(1 - alpha, bad + unknown + 1, good))
    complete = sum(r.get("solver_status") == "ok" and r.get("extraction_status") == "ok"
                   for r in rows)
    return {"n": n, "unsafe": bad, "safe": good, "unknown": unknown,
            "complete_evaluations": complete,
            "observed_unsafe_fraction_min": bad / n if n else None,
            "observed_unsafe_fraction_max": (bad + unknown) / n if n else None,
            "unsafe_fraction_lower_one_sided_95": lower,
            "unsafe_fraction_upper_one_sided_95": upper}


def evaluate(device: dict, recipe: dict, output: Path, process_only: bool, timeout: float) -> dict:
    dependencies = {p: importlib.metadata.version(p) for p in ("ViennaPS", "devsim", "numpy", "scipy")}
    source_hash = hashlib.sha256(b"".join(
        path.read_bytes() for path in sorted((ROOT / "nmos_benchmark").glob("*.py"))
    ) + (ROOT / "data/modeldb/provenance.json").read_bytes()).hexdigest()
    request = {"device": device, "recipe": recipe, "process_only": process_only,
               "dependencies": dependencies, "source_sha256": source_hash}
    fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:24]
    folder = output / "evaluations" / device["id"] / fingerprint
    folder.mkdir(parents=True, exist_ok=True)
    request_path, result_path = folder / "request.json", folder / "result.json"
    cached = result_path.is_file()
    if not cached:
        request_path.write_text(json.dumps(request, indent=2))
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        begin = time.monotonic()
        with (folder / "simulator.log").open("w") as log:
            for stage in (["process"] if process_only else ["process", "device"]):
                try:
                    remaining = max(0.1, timeout - (time.monotonic() - begin))
                    completed = subprocess.run([sys.executable, "-m", "nmos_benchmark.worker", str(request_path), stage],
                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
                    status = "native_worker_failure"
                    error = f"{stage} worker exit code {completed.returncode}; see simulator.log"
                except subprocess.TimeoutExpired:
                    status, error = "timeout", f"Evaluation exceeded {timeout:g} seconds"
                    break
                if completed.returncode != 0:
                    break
                if stage == "process":
                    stage_result = json.loads((folder / "process_result.json").read_text())
                    if process_only or stage_result["solver_status"] != "process_only":
                        result_path.write_text(json.dumps(stage_result, indent=2))
                        break
        if not result_path.is_file():
            result_path.write_text(json.dumps({"solver_status": status, "error": error,
                "constraint_violated": None, "elapsed_s": time.monotonic() - begin}, indent=2))
    result = json.loads(result_path.read_text())
    return {"device_id": device["id"], "evaluation_id": fingerprint, **recipe, **result,
            "cached": cached, "config_path": str(request_path.resolve()),
            "gate_length_nm": device["geometry"]["gate_length_nm"],
            "oxide_thickness_nm": device["geometry"]["oxide_thickness_nm"],
            "substrate_doping_cm3": device["fixed"]["substrate_doping_cm3"],
            "leakage_limit_A_per_um": device["electrical"]["leakage_limit_A_per_um"],
            "target_vth_V": device["electrical"]["target_vth_V"]}


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_batch(device, bounds, count, seed, output, phase, timeout, *,
                   workers=1, process_only=False):
    recipes = sample(bounds, count, seed)
    rows = {}

    def record(i, row):
        row.update(phase=phase, sample_index=i)
        rows[i] = row
        # Only the coordinator writes the shared CSV. Restore sample order
        # even when a later simulation finishes before an earlier one.
        write_rows(output / device["id"] / f"{phase}.csv", [rows[j] for j in sorted(rows)])
        print(f"{device['id']} {phase} {len(rows)}/{count} (recipe {i + 1}): {row['solver_status']}, "
              f"unsafe={row.get('constraint_violated')}", flush=True)

    if workers == 1 or count <= 1:
        for i, recipe in enumerate(recipes):
            record(i, evaluate(device, recipe, output, process_only, timeout))
    else:
        # Threads coordinate jobs; ViennaPS and DEVSIM still run in separate
        # subprocesses. Native simulation state and logs stay isolated.
        pool = ThreadPoolExecutor(max_workers=min(workers, count))
        try:
            jobs = {pool.submit(evaluate, device, recipe, output, process_only, timeout): i
                    for i, recipe in enumerate(recipes)}
            for job in as_completed(jobs):
                record(jobs[job], job.result())
        finally:
            # Cancel queued jobs on interruption. Active recipes can finish
            # and leave cached results for a restart.
            pool.shutdown(wait=True, cancel_futures=True)
    return [rows[i] for i in range(count)]


def generate(device, args, index):
    worker_options = {"workers": args.workers} if hasattr(args, "workers") else {}
    folder = args.output / device["id"]
    folder.mkdir(parents=True, exist_ok=True)
    frozen_path = folder / "frozen_domain.json"
    # A failed final validation must not trigger repeated fishing on that holdout.
    # Change the experiment/config and use a new output directory for a new study.
    if frozen_path.is_file():
        frozen = json.loads(frozen_path.read_text())
        if frozen["device"] != device or frozen["seed"] != args.seed:
            raise ValueError("Existing frozen domain belongs to different settings; use a new output.")
        selected = frozen["domain"]
    else:
        selected = None
        pilots = []
        for j, bounds in enumerate(device["domain_candidates"]):
            rows = evaluate_batch(device, bounds, args.pilot, args.seed + 10000 * index + j,
                                  args.output, f"pilot_{j}", args.timeout, **worker_options)
            stats = summarize(rows)
            pilots.append({"domain": bounds, "statistics": stats})
            # Exploratory rule only; statistical claim comes from new samples.
            if (stats["complete_evaluations"] == args.pilot and stats["safe"] >= 3
                    and 0.35 <= stats["unsafe"] / args.pilot <= 0.8):
                selected = bounds
                break
        (folder / "pilot_report.json").write_text(json.dumps(pilots, indent=2))
        if selected is None:
            (folder / "benchmark_report.json").write_text(json.dumps({"accepted": False,
                "reason": "No preregistered domain passed pilot criteria. Do not alter thresholds or fabricate labels.",
                "device_id": device["id"]}, indent=2))
            return False
        frozen_path.write_text(json.dumps({"device": device, "domain": selected, "seed": args.seed,
            "measure": "uniform_log10_dose_linear_time_linear_temperature",
            "final_count": args.points}, indent=2))
    frozen = json.loads(frozen_path.read_text())
    if frozen["final_count"] != args.points:
        raise ValueError("Final sample count was frozen; use a new output for a different count.")
    rows = evaluate_batch(device, selected, args.points, args.seed + 10000 * index + 5000,
                          args.output, "dataset", args.timeout, **worker_options)
    stats = summarize(rows)
    accepted = (stats["unsafe_fraction_lower_one_sided_95"] > 0.3
                and stats["safe"] >= 3 and stats["complete_evaluations"] == args.points)
    pilot_count = 0
    for path in folder.glob("pilot_*.csv"):
        with path.open(newline="") as f:
            pilot_count += sum(1 for _ in csv.DictReader(f))
    report = {"device_id": device["id"], "accepted": accepted, "statistics": stats,
        "reason": "PASS" if accepted else "Independent validation did not establish >30% unsafe with complete responses and safe seeds.",
        "domain": selected, "measure": frozen["measure"],
        "confidence_scope": "95% one-sided per-device, not simultaneous across devices",
        "cost": {"dataset_recipe_evaluations": args.points,
                 "pilot_recipe_evaluations": pilot_count,
                 "note": "One recipe includes one process simulation, bias ramps and a gate sweep. Counts are not primitive PDE solve counts."}}
    (folder / "benchmark_report.json").write_text(json.dumps(report, indent=2))
    seeds = [row for row in rows if row.get("constraint_violated") is False
             and row.get("extraction_status") == "ok"]
    write_rows(folder / "safe_seed_candidates.csv", seeds)
    return accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["doctor", "smoke", "process-only", "generate"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/devices.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--device", help="One configured device ID (default: all)")
    parser.add_argument("--points", type=int, default=100)
    parser.add_argument("--pilot", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--workers", type=int, default=2,
                        help="Concurrent recipes for generate and process-only (default: 2; 1 = serial)")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.points < 1 or args.pilot < 6:
        parser.error("points must be positive; pilot must be at least 6")
    if args.workers < 1:
        parser.error("workers must be at least 1")
    devices = load_devices(args.config)
    indexed = [(i, d) for i, d in enumerate(devices) if args.device is None or d["id"] == args.device]
    if not indexed:
        parser.error("Unknown device ID")
    if args.command == "doctor":
        from .process import verify_tables
        verify_tables()
        import viennaps as vps
        for name in ("ImplantTableModel", "IonImplantation", "Anneal"):
            if not hasattr(vps, name):
                raise RuntimeError(f"ViennaPS lacks {name}; check the installed release")
        from .device import load_devsim
        load_devsim()
        print("ViennaPS, DEVSIM and source-table checks passed.")
        return
    passed = []
    for index, device in indexed:
        if args.command == "generate":
            passed.append(generate(device, args, index))
        elif args.command == "smoke":
            recipe = {"implant_dose_cm2": 1e12, "anneal_time_s": 30., "anneal_temperature_C": 1050.}
            row = evaluate(device, recipe, args.output, False, args.timeout)
            write_rows(args.output / device["id"] / "smoke.csv", [row])
            print(json.dumps(row, indent=2))
            passed.append(row["solver_status"] == "ok" and row.get("extraction_status") == "ok")
        else:
            # Profiles only: never claim safety from process data alone.
            bounds = device["domain_candidates"][0]
            rows = evaluate_batch(device, bounds, args.points, args.seed + index * 10000,
                                  args.output, "process_only", args.timeout,
                                  workers=args.workers, process_only=True)
            passed.append(all(r["solver_status"] == "process_only" for r in rows))
    if not all(passed):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
