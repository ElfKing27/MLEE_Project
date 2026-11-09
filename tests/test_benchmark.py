import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from nmos_benchmark.runner import ROOT, generate, load_devices, sample, summarize, write_rows
from nmos_benchmark.process import verify_tables


class BenchmarkTests(unittest.TestCase):
    def test_sampling_measure_and_reproducibility(self):
        bounds = {"implant_dose_cm2": [1e11, 1e13], "anneal_time_s": [5, 120],
                  "anneal_temperature_C": [1000, 1100]}
        rows = sample(bounds, 100, 45)
        self.assertEqual(rows, sample(bounds, 100, 45))
        self.assertNotEqual(rows, sample(bounds, 100, 46))
        for row in rows:
            for key, (low, high) in bounds.items():
                self.assertTrue(low <= row[key] <= high)
        values = np.log10([r["implant_dose_cm2"] for r in sample(bounds, 10000, 4)])
        self.assertAlmostEqual(values.mean(), 12, delta=0.02)

    def test_unknowns_cannot_inflate_unsafe_fraction(self):
        rows = ([{"constraint_violated": True}] * 30
                + [{"constraint_violated": False}] * 10
                + [{"constraint_violated": None}] * 60)
        report = summarize(rows)
        self.assertEqual(report["unknown"], 60)
        self.assertEqual(report["observed_unsafe_fraction_min"], 0.3)
        self.assertLess(report["unsafe_fraction_lower_one_sided_95"], 0.3)

    def test_observed_thirty_percent_does_not_certify_thirty_percent(self):
        report = summarize([{"constraint_violated": True}] * 31
                           + [{"constraint_violated": False}] * 69)
        self.assertLess(report["unsafe_fraction_lower_one_sided_95"], 0.3)
        report = summarize([{"constraint_violated": True}] * 50
                           + [{"constraint_violated": False}] * 50)
        self.assertGreater(report["unsafe_fraction_lower_one_sided_95"], 0.3)

    def test_statistical_endpoints(self):
        self.assertEqual(summarize([{"constraint_violated": False}] * 100)
                         ["unsafe_fraction_lower_one_sided_95"], 0)
        self.assertEqual(summarize([{"constraint_violated": True}] * 100)
                         ["unsafe_fraction_upper_one_sided_95"], 1)

    def test_configuration_and_table_provenance(self):
        devices = load_devices(ROOT / "configs/devices.json")
        self.assertEqual(len(devices), 3)
        self.assertEqual(devices[0]["geometry"]["gate_length_nm"], 500)
        self.assertEqual(devices[1]["geometry"]["gate_length_nm"], 1000)
        verify_tables()

    def test_csv_retains_unknown_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            write_rows(path, [{"solver_status": "timeout", "constraint_violated": None}])
            self.assertIn("timeout,", path.read_text())

    def test_domain_selection_uses_independent_validation_and_freezes(self):
        device = load_devices(ROOT / "configs/devices.json")[0]
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory), pilot=6, points=100, seed=99, timeout=10)
            calls = []

            def fake_batch(dev, bounds, count, seed, output, phase, timeout):
                calls.append((phase, seed))
                rows = [{"constraint_violated": i < count // 2, "solver_status": "ok",
                         "extraction_status": "ok"} for i in range(count)]
                write_rows(output / dev["id"] / f"{phase}.csv", rows)
                return rows

            with patch("nmos_benchmark.runner.evaluate_batch", side_effect=fake_batch):
                self.assertTrue(generate(device, args, 0))
            self.assertEqual(calls, [("pilot_0", 99), ("dataset", 5099)])
            calls.clear()
            with patch("nmos_benchmark.runner.evaluate_batch", side_effect=fake_batch):
                self.assertTrue(generate(device, args, 0))
            self.assertEqual(calls, [("dataset", 5099)])

    def test_no_pilot_domain_means_no_fabricated_dataset(self):
        device = load_devices(ROOT / "configs/devices.json")[0]
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory), pilot=6, points=100, seed=99, timeout=10)
            with patch("nmos_benchmark.runner.evaluate_batch", return_value=[
                {"constraint_violated": False, "solver_status": "ok", "extraction_status": "ok"}] * 6):
                self.assertFalse(generate(device, args, 0))
            self.assertFalse((args.output / device["id"] / "dataset.csv").exists())


if __name__ == "__main__":
    unittest.main()
