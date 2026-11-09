"""One isolated native-simulator evaluation. Invoked by the CLI, not directly."""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path


def main():
    request_path = Path(sys.argv[1])
    request = json.loads(request_path.read_text())
    task = sys.argv[2]
    result_path = request_path.with_name("process_result.json" if task == "process" else "result.json")
    folder = request_path.parent
    start = time.monotonic()
    stage = "process_failure"
    result = {"solver_status": "unknown", "constraint_violated": None,
              "safety_margin": None, "vth_V": None, "ioff_A_per_um": None,
              "objective": None, "yield_estimate": None}
    try:
        if task == "process":
            from .process import run_process
            result.update(run_process(request["device"], request["recipe"], folder / "profile.npz"))
            result["solver_status"] = "process_only"
        else:
            result.update(json.loads((folder / "process_result.json").read_text()))
            stage = "device_solver_failure"
            from .device import run_device
            result.update(run_device(request["device"], folder / "profile.npz", folder / "iv.csv"))
            result["solver_status"] = "ok"
    except Exception as exc:
        result.update(solver_status=stage, error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    if task == "device":
        result["process_elapsed_s"] = result["elapsed_s"]
        result["device_elapsed_s"] = time.monotonic() - start
        result["elapsed_s"] += result["device_elapsed_s"]
    else:
        result["elapsed_s"] = time.monotonic() - start
    temporary = result_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
    temporary.replace(result_path)


if __name__ == "__main__":
    main()
