"""Read stable snapshots of NMOS benchmark CSVs."""
from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
import time


INPUTS = ("implant_dose_cm2", "anneal_time_s", "anneal_temperature_C")
OUTPUTS = ("vth_V", "ioff_A_per_um", "safety_margin", "objective",
           "target_vth_V", "leakage_limit_A_per_um")


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def read_snapshot(path: Path) -> list[dict]:
    """Retry if a simulator batch is rewriting the CSV as we read it."""
    for _ in range(4):
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        if (before.st_size == after.st_size == len(data) and
                before.st_mtime_ns == after.st_mtime_ns and data.endswith(b"\n")):
            reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig", errors="replace")), strict=True)
            if not set(INPUTS).issubset(reader.fieldnames or []):
                raise ValueError(f"Missing recipe columns in {path}")
            rows = list(reader)
            if all(None not in row and all(v is not None for v in row.values()) for row in rows):
                return rows
        time.sleep(0.05)
    raise ValueError(f"Could not read a stable CSV snapshot: {path}. Retry after the current write.")


def read_dataset(path: Path) -> dict:
    raw = read_snapshot(path)
    rows = []
    for row in raw:
        recipe = {key: number(row[key]) for key in INPUTS}
        if any(value is None for value in recipe.values()) or recipe[INPUTS[0]] <= 0:
            continue
        label = (row.get("constraint_violated") or "").strip().lower()
        solver = row.get("solver_status", "")
        responses = {key: number(row.get(key)) for key in OUTPUTS}
        # An incomplete threshold extraction can still have a valid leakage
        # label, but a failed solver or missing leakage cannot certify safety.
        status = "unknown"
        if solver == "ok" and responses["ioff_A_per_um"] is not None:
            status = {"true": "unsafe", "false": "safe"}.get(label, "unknown")
        rows.append({**recipe, **responses, "status": status, "solver": solver,
                     "extraction": row.get("extraction_status", ""),
                     "sample": row.get("sample_index") or str(len(rows)),
                     "evaluation_id": row.get("evaluation_id", "")})
    if not rows:
        raise ValueError(f"No valid recipe rows in {path}")
    device = raw[0].get("device_id") or path.parent.name
    if any((row.get("device_id") or device) != device for row in raw):
        raise ValueError(f"Expected one device per CSV: {path}")
    phase = "Final dataset" if path.name == "dataset.csv" else path.stem.replace("_", " ").capitalize()
    qualification = "Exploratory; not a final dataset"
    if path.name == "dataset.csv":
        qualification = "Final sample in progress or awaiting benchmark report"
        try:
            report = json.loads(path.with_name("benchmark_report.json").read_text(encoding="utf-8-sig"))
            if report.get("statistics", {}).get("n") == len(raw) and len(rows) == len(raw):
                qualification = ("Benchmark accepted" if report.get("accepted") is True
                                 else "Benchmark did not qualify")
        except (OSError, ValueError):
            pass
    return {"device": device, "phase": phase, "qualification": qualification,
            "source": str(path.resolve()), "skipped": len(raw) - len(rows), "rows": rows}


def discover(source: Path, include_pilots: bool) -> list[Path]:
    if source.is_file() and source.suffix.lower() == ".csv":
        return [source]
    if not source.is_dir():
        raise ValueError(f"Input must be a CSV or an output folder: {source}")
    paths = {*source.glob("dataset.csv"), *source.glob("*/dataset.csv")}
    if include_pilots:
        paths.update(source.glob("pilot_*.csv"))
        paths.update(source.glob("*/pilot_*.csv"))
    if not paths:
        raise ValueError("No final dataset.csv found. Use --include-pilots to preview exploratory samples.")
    return sorted(paths, key=lambda p: (p.parent.name, p.name != "dataset.csv", p.name))
