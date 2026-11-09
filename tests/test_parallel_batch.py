"""Check concurrent execution and deterministic dataset ordering."""
import csv
import tempfile
import unittest
from pathlib import Path
from threading import Barrier, Event
from unittest.mock import patch

from nmos_benchmark.runner import ROOT, evaluate_batch, load_devices, sample, write_rows


class ParallelBatchTests(unittest.TestCase):
    def test_parallel_jobs_overlap_and_record_rows_in_sample_order(self):
        device = load_devices(ROOT / "configs/devices.json")[0]
        bounds = device["domain_candidates"][0]
        recipes = sample(bounds, 2, 45)
        both_running = Barrier(2, timeout=5)
        second_saved = Event()
        snapshots = []

        def fake_evaluate(dev, recipe, output, process_only, timeout):
            both_running.wait()  # Raises if the evaluations run serially.
            if recipe == recipes[0]:
                self.assertTrue(second_saved.wait(5))
            return {**recipe, "solver_status": "ok", "constraint_violated": False}

        def save(path, rows):
            write_rows(path, rows)
            snapshots.append([row["sample_index"] for row in rows])
            second_saved.set()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("nmos_benchmark.runner.evaluate", side_effect=fake_evaluate), \
                 patch("nmos_benchmark.runner.write_rows", side_effect=save):
                rows = evaluate_batch(device, bounds, 2, 45, output, "pilot_0", 30, workers=2)
            self.assertEqual(snapshots, [[1], [0, 1]])
            self.assertEqual([r["implant_dose_cm2"] for r in rows],
                             [r["implant_dose_cm2"] for r in recipes])
            with (output / device["id"] / "pilot_0.csv").open(newline="") as f:
                self.assertEqual([r["sample_index"] for r in csv.DictReader(f)], ["0", "1"])


if __name__ == "__main__":
    unittest.main()
