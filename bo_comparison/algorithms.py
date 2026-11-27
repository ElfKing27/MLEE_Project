"""The GP models and the two recipe-selection rules.

Only observed outcomes enter these functions. X contains all recipe coordinates.
SafeOpt uses the GP-confidence version with a separate safety constraint.
"""

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


def fit_gp(X, observed, values, center, scale, settings):
    """Fit a GP to queried recipes, in fixed, scaled output units."""
    kernel = Matern(length_scale=settings["length_scale"], nu=2.5)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=settings["noise_variance"],
        optimizer=None,
        normalize_y=False,
    )
    scaled_values = (np.asarray(values) - center) / scale
    gp.fit(X[observed], scaled_values)
    return gp


def expected_improvement(mean, std, best_observed):
    """Expected improvement for maximization, without a safety filter."""
    improvement = mean - best_observed
    score = np.maximum(improvement, 0.0)
    uncertain = std > 1e-12
    z = improvement[uncertain] / std[uncertain]
    score[uncertain] = (
        improvement[uncertain] * norm.cdf(z) + std[uncertain] * norm.pdf(z)
    )
    return score


def choose_recipe(X, observed, objectives, margins, method, settings, objective_center):
    """Return a candidate index and prediction details, or a stopping reason.

    Unlike a constraint-filtered EI rule, SafeOpt considers both potential
    objective maximizers and potential safe-set expanders. Both methods may
    revisit recipes, and each visit uses one unit of the evaluation budget.
    """
    objective_scale = settings["objective_scale"]
    objective_gp = fit_gp(
        X, observed, objectives, objective_center, objective_scale, settings
    )
    mean_f, std_f = objective_gp.predict(X, return_std=True)

    if method == "EI":
        best = (max(objectives) - objective_center) / objective_scale
        scores = expected_improvement(mean_f, std_f, best)
        index = int(np.argmax(scores))
        return index, {
            "selection_role": "expected_improvement",
            "predicted_objective": mean_f[index] * objective_scale + objective_center,
            "objective_std": std_f[index] * objective_scale,
            "acquisition_value": float(scores[index]) * objective_scale,
            "eligible_candidates": len(X),
            "predicted_safety_lower": "",
            "predicted_safety_upper": "",
        }

    # Margin = ln(leakage limit / leakage). A nonnegative value is safe.
    safety_scale = settings["safety_scale"]
    safety_gp = fit_gp(X, observed, margins, 0.0, safety_scale, settings)
    mean_g, covariance_g = safety_gp.predict(X, return_cov=True)
    variance_g = np.maximum(np.diag(covariance_g), 0.0)
    std_g = np.sqrt(variance_g)
    beta = settings["beta"]  # Direct multiplier of standard deviation.
    lower_g = mean_g - beta * std_g
    upper_g = mean_g + beta * std_g
    lower_f = mean_f - beta * std_f
    upper_f = mean_f + beta * std_f

    safe = lower_g >= 0.0
    # Exact outcomes of queried recipes take precedence over GP estimates.
    safe[observed] = np.asarray(margins) >= 0.0
    if not safe.any():
        return None, {"stop_reason": "no_certified_recipe"}

    # A potential maximizer could beat the best safe objective lower bound.
    best_lower_bound = np.max(lower_f[safe])
    maximizers = safe & (upper_f >= best_lower_bound)

    # An expander could certify a presently uncertified candidate if its
    # observed safety margin turns out to equal its optimistic upper bound.
    # The GP conditioning formula avoids refitting a GP for every fantasy.
    expanders = np.zeros(len(X), dtype=bool)
    uncertified = ~safe
    uncertainty = np.maximum(std_f, std_g)
    best_index = None
    if maximizers.any():
        candidates = np.flatnonzero(maximizers)
        best_index = int(candidates[np.argmax(uncertainty[candidates])])

    # Search from most uncertain down. Once an expander is found, any later
    # one has no larger acquisition value, so it cannot change the choice.
    ordered = np.argsort(-uncertainty, kind="stable")
    for index in ordered:
        if not safe[index] or maximizers[index]:
            continue
        if best_index is not None and uncertainty[index] <= uncertainty[best_index]:
            break
        if not uncertified.any():
            break
        denominator = variance_g[index] + settings["noise_variance"]
        cross_covariance = covariance_g[uncertified, index]
        optimistic_mean = mean_g[uncertified] + (
            cross_covariance / denominator * (upper_g[index] - mean_g[index])
        )
        optimistic_variance = np.maximum(
            variance_g[uncertified] - cross_covariance**2 / denominator, 0.0
        )
        optimistic_lower = optimistic_mean - beta * np.sqrt(optimistic_variance)
        if np.any(optimistic_lower >= 0.0):
            expanders[index] = True
            best_index = int(index)
            break

    if best_index is None:
        return None, {"stop_reason": "no_maximizer_or_expander"}

    index = best_index
    return index, {
        "selection_role": "maximizer" if maximizers[index] else "expander",
        "predicted_objective": mean_f[index] * objective_scale + objective_center,
        "objective_std": std_f[index] * objective_scale,
        "acquisition_value": float(2.0 * beta * uncertainty[index]),
        "eligible_candidates": int(safe.sum()),
        "predicted_safety_lower": float(lower_g[index] * safety_scale),
        "predicted_safety_upper": float(upper_g[index] * safety_scale),
    }
