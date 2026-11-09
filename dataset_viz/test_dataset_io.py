import csv
import json
from pathlib import Path
import tempfile
import unittest

from dataset_io import discover, read_dataset


class DatasetReaderTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.device = self.root / "reference"
        self.device.mkdir()

    def make_csv(self, name):
        path = self.device / name
        rows = [
            dict(implant_dose_cm2=1e11, anneal_time_s=10, anneal_temperature_C=1000,
                 solver_status="ok", constraint_violated="False", vth_V=.48, ioff_A_per_um=1e-13),
            dict(implant_dose_cm2=1e12, anneal_time_s=30, anneal_temperature_C=1050,
                 solver_status="ok", constraint_violated="True", vth_V=.4, ioff_A_per_um=1e-11),
            dict(implant_dose_cm2=1e13, anneal_time_s=60, anneal_temperature_C=1100,
                 solver_status="timeout", constraint_violated="", vth_V="", ioff_A_per_um=""),
        ]
        for i, row in enumerate(rows):
            row.update(sample_index=i, device_id="reference", target_vth_V=.5,
                       leakage_limit_A_per_um=1e-12, evaluation_id="</script><img src=x>")
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_final_is_default_and_pilot_is_separate(self):
        pilot = self.make_csv("pilot_0.csv")
        with self.assertRaisesRegex(ValueError, "No final dataset"):
            discover(self.root, False)
        final = self.make_csv("dataset.csv")
        self.assertEqual(discover(self.root, False), [final])
        self.assertEqual(discover(self.root, True), [final, pilot])
        self.assertIn("Exploratory", read_dataset(pilot)["qualification"])

    def test_qualification_and_unknown_responses(self):
        final = self.make_csv("dataset.csv")
        data = read_dataset(final)
        self.assertEqual([row["status"] for row in data["rows"]], ["safe", "unsafe", "unknown"])
        self.assertIsNone(data["rows"][2]["vth_V"])
        self.assertIn("awaiting", data["qualification"])
        final.with_name("benchmark_report.json").write_text(json.dumps({
            "accepted": False, "statistics": {"n": 3}}))
        self.assertEqual(read_dataset(final)["qualification"], "Benchmark did not qualify")
