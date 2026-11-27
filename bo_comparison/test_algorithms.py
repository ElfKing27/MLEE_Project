"""Small regression checks for acquisition math and hidden-response handling."""

import unittest
from unittest.mock import patch

import numpy as np

from algorithms import choose_recipe, expected_improvement
from run_experiment import run_method, score_trace


SETTINGS = {
    "length_scale": .4, "noise_variance": 1e-6, "beta": 3.0,
    "objective_scale": .25, "safety_scale": 10.0, "budget": 5,
}


class AcquisitionTests(unittest.TestCase):
    def test_ei_handles_zero_uncertainty(self):
        actual = expected_improvement(np.array([2., -1., 0.]), np.array([0., 0., 1.]), 0.)
        np.testing.assert_allclose(actual, [2., 0., 1 / np.sqrt(2 * np.pi)])

    def test_safeopt_can_select_an_expander_outside_the_maximizers(self):
        class ObjectiveGP:
            def predict(self, X, return_std):
                return np.array([0., 1., -1., 0.]), np.array([.01, .1, .4, .3])

        class SafetyGP:
            def predict(self, X, return_cov):
                covariance = np.diag([1e-6, .0001, .01, .1])
                covariance[2, 3] = covariance[3, 2] = .025
                return np.array([1., 1., .8, -.1]), covariance

        # Candidate 2 cannot beat candidate 1 on objective confidence bounds,
        # but its optimistic safety response would certify candidate 3.
        settings = {**SETTINGS, "objective_scale": 1., "safety_scale": 1.}
        with patch("algorithms.fit_gp", side_effect=[ObjectiveGP(), SafetyGP()]):
            index, details = choose_recipe(np.zeros((4, 3)), [0], [0.], [1.], "SafeOpt", settings, 0.)
        self.assertEqual(index, 2)
        self.assertEqual(details["selection_role"], "expander")

    def test_unqueried_responses_cannot_change_the_next_selection(self):
        X = np.column_stack([np.linspace(0, 1, 8)] * 3)
        rows = [{
            "evaluation_id": str(i), "implant_dose_cm2": 1e12,
            "anneal_time_s": 20., "anneal_temperature_C": 1050.,
            "objective": -.2 + i * .01, "safety_margin": 3., "unsafe": False,
            "vth_V": .7 - i * .01, "ioff_A_per_um": 1e-14,
        } for i in range(8)]
        changed = [dict(row) for row in rows]
        for row in changed[3:]:
            row.update(objective=1000., safety_margin=-1000., unsafe=True)
        settings = {**SETTINGS, "budget": 4}
        for method in ("EI", "SafeOpt"):
            first, _, _ = run_method(rows, X, [0, 1, 2], method, settings)
            second, _, _ = run_method(changed, X, [0, 1, 2], method, settings)
            self.assertEqual([r["candidate_index"] for r in first], [r["candidate_index"] for r in second])

    def test_early_stop_carries_best_forward_without_new_queries(self):
        rows = [{"objective": -.1, "unsafe": False}, {"objective": 0., "unsafe": False}]
        trace = [{"evaluation": 1, "candidate_index": 0, "objective": -.1, "unsafe": 0, "evaluation_id": "seed"}]
        curves, result = score_trace(rows, trace, 5)
        self.assertEqual(sum(row["queried"] for row in curves), 1)
        self.assertEqual(result["queries_used"], 1)
        self.assertAlmostEqual(result["cumulative_best_safe_gap"], .5)


if __name__ == "__main__":
    unittest.main()
