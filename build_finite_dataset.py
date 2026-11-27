"""Complete a finite NMOS benchmark using existing pilot results and random top-ups."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

from nmos_benchmark.runner import PARAMETERS, ROOT, evaluate_batch, load_devices, sample, write_rows


def recipe_key(row):
    return tuple(float(row[k]) for k in PARAMETERS)


def safety(row):
    """Return a boolean violation label, or None if the result is incomplete."""
    if row.get("solver_status") != "ok" or row.get("extraction_status") != "ok":
        return None
    try:
        if not all(math.isfinite(float(row[k])) for k in (*PARAMETERS, "vth_V", "ioff_A_per_um", "objective", "safety_margin")):
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return {"true": True, "false": False}.get(str(row.get("constraint_violated")).lower())


def simulator_identity():
    source = b"".join(path.read_bytes() for path in sorted((ROOT / "nmos_benchmark").glob("*.py")))
    source += (ROOT / "data/modeldb/provenance.json").read_bytes()
    versions = {p: importlib.metadata.version(p) for p in ("ViennaPS", "devsim", "numpy", "scipy")}
    return hashlib.sha256(source).hexdigest(), versions


def load_pilots(output, device):
    result, seen = [], set()
    source_hash, versions = simulator_identity()
    directory = output / device["id"]
    for path in sorted(directory.glob("pilot_*.csv")):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if safety(row) is None:
                    continue
                key = recipe_key(row)
                if key in seen:
                    continue
                eid = row.get("evaluation_id", "")
                if len(eid) != 24 or any(c not in "0123456789abcdef" for c in eid):
                    raise ValueError(f"Invalid evaluation ID in {path}")
                request_path = output / "evaluations" / device["id"] / eid / "request.json"
                request = json.loads(request_path.read_text())
                if (request["device"] != device or request["source_sha256"] != source_hash
                        or request["dependencies"] != versions or request["process_only"]
                        or recipe_key(request["recipe"]) != key):
                    raise ValueError(f"Pilot settings differ from the current evaluator: {request_path}")
                seen.add(key)
                result.append({**row, "source_phase": path.stem, "source_sample_index": row.get("sample_index")})
    return result


def complete(args):
    device = next((d for d in load_devices(args.config) if d["id"] == args.device), None)
    if device is None:
        raise ValueError(f"Unknown device: {args.device}")
    directory = args.output / device["id"]
    dataset = directory / "dataset.csv"
    report_path = directory / "finite_benchmark_report.json"
    if dataset.exists():
        if report_path.is_file():
            report = json.loads(report_path.read_text())
            if report["statistics"]["n"] == args.target and report["device_id"] == args.device:
                print(f"Finite dataset already exists: {dataset}")
                return bool(report["accepted"])
        raise ValueError(f"Refusing to overwrite an existing dataset: {dataset}")

    pilots = load_pilots(args.output, device)
    if len(pilots) != args.expect_pilots:
        raise ValueError(f"Expected {args.expect_pilots} complete distinct pilot rows, found {len(pilots)}")
    needed = args.target - len(pilots)
    if needed <= 0:
        raise ValueError("Target must be larger than the existing pilot pool")
    max_attempts = needed if args.max_attempts is None else args.max_attempts
    if max_attempts < needed:
        raise ValueError(f"--max-attempts must be at least {needed}")
    counts = Counter(safety(row) for row in pilots)
    bounds = device["hard_bounds"]
    print(f"{device['id']}: reuse {len(pilots)} pilot recipes ({counts[True]} unsafe, {counts[False]} safe); "
          f"need {needed} new complete recipes, at most {max_attempts} attempts, {args.workers} workers.", flush=True)
    if args.dry_run:
        print(f"Would write {dataset}. No simulations or files changed.")
        return True

    # Freeze top-up inputs so resuming cannot silently mix a different sample.
    plan_path = directory / "finite_topup_plan.json"
    plan = {"device": device, "pilot_ids": [r["evaluation_id"] for r in pilots], "seed": args.seed,
            "target": args.target, "measure": "uniform_log10_dose_linear_time_linear_temperature"}
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("Top-up plan already exists with different settings; resume with the original settings")
    else:
        plan_path.write_text(json.dumps(plan, indent=2))

    rows, seen, attempted, batch = list(pilots), {recipe_key(r) for r in pilots}, 0, 0
    while len(rows) < args.target and attempted < max_attempts:
        count = min(args.target - len(rows), max_attempts - attempted)
        seed = args.seed + batch
        phase = f"finite_topup_{batch}"
        keys = [recipe_key(r) for r in sample(bounds, count, seed)]
        if len(set(keys)) != len(keys) or any(k in seen for k in keys):
            raise ValueError("Top-up seed produced a duplicate recipe; choose a different study seed before starting")
        observed = evaluate_batch(device, bounds, count, seed, args.output, phase, args.timeout,
                                  workers=args.workers)
        for row in observed:
            if safety(row) is not None:
                rows.append({**row, "source_phase": phase, "source_sample_index": row["sample_index"]})
        seen.update(keys)
        attempted += count
        batch += 1

    if len(rows) != args.target:
        print(f"Have {len(rows)}/{args.target} complete recipes. Progress is cached. "
              "Rerun with the same arguments plus a larger --max-attempts to replace failed recipes.")
        return False
    unsafe = sum(safety(row) is True for row in rows)
    safe = len(rows) - unsafe
    accepted = unsafe / len(rows) > .30 and safe >= 3
    report = {"device_id": device["id"], "benchmark_type": "finite_candidate_pool", "accepted": accepted,
              "statistics": {"n": len(rows), "unsafe": unsafe, "safe": safe, "unknown": 0,
                             "observed_unsafe_fraction": unsafe / len(rows)},
              "sources": {"pilot_recipes": len(pilots), "new_complete_recipes": needed,
                          "topup_attempts": attempted}, "topup_bounds": bounds,
              "scope": "Exact count on this finite pool; no claim about continuous design-space volume."}
    for i, row in enumerate(rows):
        row.update(sample_index=i, phase="finite_dataset")
    temp = dataset.with_suffix(".tmp")
    write_rows(temp, rows)
    write_rows(directory / "safe_seed_candidates.csv", [r for r in rows if safety(r) is False])
    report_temp = report_path.with_suffix(".tmp")
    report_temp.write_text(json.dumps(report, indent=2))
    report_temp.replace(report_path)
    temp.replace(dataset)
    print(f"Saved {dataset}: {safe} safe, {unsafe} unsafe ({unsafe / len(rows):.1%}), finite benchmark accepted={accepted}.")
    return accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/devices.json")
    parser.add_argument("--device", default="planar_L1000_tox15")
    parser.add_argument("--output", type=Path, default=Path("outputs_parallel_100"))
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--expect-pilots", type=int, default=72)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--max-attempts", type=int, help="Allow additional draws to replace failed evaluations")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.timeout <= 0 or args.seed < 0:
        parser.error("Use a positive worker count and timeout, and a nonnegative seed")
    args.output = args.output.resolve()
    try:
        passed = complete(args)
    except (ValueError, KeyError, OSError, csv.Error) as exc:
        parser.exit(1, f"Finite dataset not completed: {exc}\n")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
