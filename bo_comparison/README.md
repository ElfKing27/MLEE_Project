# EI and SafeOpt comparison

This folder runs a finite-pool replay using your saved ViennaPS–DEVSIM results.
No new device simulations are needed. Each device has its own 100 candidate
recipes. A query reveals one existing CSV row to the optimizer.

The [compact comparison report](report/README.md) is the starting point for
reviewing the results. It includes all tested beta/budget settings, with the
best explored setting highlighted. The tracked CSV snapshots under
`benchmark_data` make the optimizer study reproducible without TCAD access.
The `sensitivity` and `results_equal_budget` folders contain local raw outputs
and are ignored by Git.

## Commands from the project root

Run all 30 pairs per device, then make tables and plots:

```powershell
python .\bo_comparison\run_experiment.py --input outputs_parallel_100 --output outputs_bo --runs 30 --budget 50 --workers 3
python .\bo_comparison\plot_results.py --input outputs_bo --show
```

The completed results supplied with this folder are in `bo_comparison/results_equal_budget`.
To open their plots without running the optimizers again:

```powershell
python .\bo_comparison\plot_results.py --input .\bo_comparison\results_equal_budget --show
```

If the experiment is interrupted, run the same generation command with
`--resume`. Completed pairs are reused. Changed datasets, parameters, code or
package versions require a new output folder. Plots can be regenerated at any
time. Leave out `--show` to save PNGs without opening Matplotlib windows.

The tested Python environment already contains the required packages. For
another environment install only this experiment's dependencies:

```powershell
python -m pip install -r .\bo_comparison\requirements.txt
```

## How to read the code

1. `run_experiment.py` loads and checks the CSVs. `run_pair` chooses three
   random known-safe seeds and gives the same seeds to both methods.
   `run_method` selects a recipe using only observations, then reveals its true
   outcome. `score_trace` measures performance using the hidden pool optimum.
2. `algorithms.py` creates the GPs and chooses the next candidate. Read the EI
   branch first. The SafeOpt branch builds a confidence-based safe set, finds
   possible objective maximizers, checks possible safe-set expanders, and
   chooses the most uncertain eligible recipe in scaled output units.
3. `plot_results.py` reads saved traces and makes tables and figures. It joins
   results by device and run number before calculating paired differences.
4. `verify_results.py` checks saved query traces and metrics against the source
   datasets. `test_algorithms.py` checks EI math, expander selection, hidden
   response handling, and accounting after an early stop.

```powershell
python .\bo_comparison\verify_results.py --datasets outputs_parallel_100 --results .\bo_comparison\results_equal_budget
python -m unittest discover -s bo_comparison -p test_algorithms.py -v
```

## Frozen settings for the supplied experiment

- 30 paired runs per device, 3 devices, 2 methods: 180 method runs.
- Each run uses 50 recipe queries including 3 initial seeds. Repeated queries
  are allowed for both methods, and count toward the same budget.
- Seed selection is uniform over the known-safe pool rows. The seed provider
  sees the safety labels to provide those starting recipes. The algorithms
  receive no other unqueried objective or constraint outcomes.
- Inputs are log10 dose, time and temperature, each scaled to [0, 1] using
  candidate coordinates. All candidate coordinates are public to both methods.
- Both objective GPs use a fixed Matern 5/2 kernel, length scale 0.4, and
  normalized observation-noise variance 1e-6. The prior mean is the mean of
  the three starting objectives, kept fixed afterward. Output scale is 0.25 V.
- SafeOpt uses a second GP for `ln(leakage_limit / leakage)`. Its prior mean
  is zero and its output scale is 10 natural-log units. The kernel and noise
  settings match the objective GP. Its lower confidence bound is mean minus
  3 standard deviations. The CLI calls this multiplier `--beta`.
- Hyperparameters stay fixed. These are declared modeling choices, with no
  tuning on the unqueried responses. Beta 3 is an empirical confidence setting.
  It does not establish a formal zero-failure guarantee for these simulators.
- A positive safety margin denotes compliance with the leakage limit. The
  objective is `-abs(Vth - 0.5)`. Manufacturing yield is not simulated.

## What this SafeOpt implementation does

This is a small scikit-learn implementation of the GP-confidence SafeOpt
variant with a separate safety GP. It follows the maximizer/expander procedure
documented by Berkenkamp's SafeOpt package, including optimistic conditioning
of the safety GP. It does not use GPy or require the legacy SafeOpt package.
The code computes the hypothetical GP update directly from posterior covariance.
The regression test checks that an expander can win over an objective maximizer.

SafeOpt can revisit a known-safe recipe while it remains uncertain about other
recipes. Both methods may revisit recipes, and every visit costs one query.
This follows the usual SafeOpt selection over the full candidate set. The
`distinct_recipes` and `repeat_queries` columns distinguish exploration from
repetition. Repeated calls return the same deterministic CSV outcome. The
small GP observation-noise variance acts as numerical regularization; the
replay does not add random measurement noise to the saved outcomes.

The 500 nm pool contains only 32 safe recipes. It is impossible to make 50
distinct safe queries from that pool. Allowing repetitions lets both methods
use an equal 50-query budget. Interpret this as equal nominal evaluation cost
in an offline replay. No new simulation runtime is consumed. In a physical
experiment, a catastrophic violation could force termination. This diagnostic
replay continues after violations to measure their cumulative count, and also
records the first violation and zero-violation runs.

The 1,000 nm dataset combines 72 pilot outcomes and 28 additional outcomes. Its
50% unsafe fraction describes its finite candidate pool. All three comparisons
here are restricted to their saved pools, so no interpolation or continuous
space safety claim is needed.

## Tables and figures

- `manifest.json` records settings, source-code hashes, dataset hashes and
  scientific package versions. `runs.csv` holds one row per method run.
- `summary.csv` contains mean and sample standard deviation per device/method.
  It includes violations, zero-violation fraction, query counts, regret and stops.
- `paired_differences.csv` contains SafeOpt minus EI for each matched pair.
- `paired_comparisons.csv` contains paired mean differences and 95% percentile
  bootstrap intervals from 3,000 resamples of the run pairs.
- `results.txt` is the plain-text summary. `plots/` contains PNG figures.
- `<device>/run_XX/` contains each query, GP predictions at selection time,
  recorded stopping reasons and per-budget learning curves.

`simple_safe_regret` is the best safe pool objective minus the best safe
objective observed so far, in volts. `cumulative_best_safe_gap` sums that gap
over the budget steps. This is cumulative incumbent regret. It differs from
cumulative loss at the actual queried recipes. An unsafe recipe cannot improve
the recorded best-safe objective.

If an exceptional early stop occurs, the recorded best objective and violation
count carry forward to the 50-step comparison horizon. `queried=0` explicitly
identifies these padded steps. `queries_used` counts actual queries, including
repeats. The supplied equal-budget results must be checked for early stops
before describing them as 50-query runs. Learning-curve
bands and paired intervals show variation across starting seeds on the fixed
datasets, rather than uncertainty across fabricated devices. They are
per-comparison intervals, with no adjustment for multiple comparisons.

## Sources

- Sui et al., Safe Exploration for Optimization with Gaussian Processes,
  ICML 2015. https://proceedings.mlr.press/v37/sui15.html
- Berkenkamp et al., Bayesian Optimization with Safety Constraints.
  https://arxiv.org/abs/1602.04450
- SafeOpt API and GP-confidence maximizer/expander implementation.
  https://safeopt.readthedocs.io/en/latest/safeopt.SafeOpt.html
  https://github.com/befelix/SafeOpt/blob/master/safeopt/gp_opt.py
- scikit-learn GaussianProcessRegressor.
  https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
