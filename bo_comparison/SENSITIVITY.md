# Budget and confidence comparison

This study compares budgets of 50 and 100 queries and SafeOpt confidence
multipliers of 3, 2 and 1. It uses 30 paired initializations on each of the
three saved device pools. Each pair starts from the same three safe recipes.

## Observations from the completed study

The table shows SafeOpt's mean final gap to the best safe recipe in each
pool, in millivolts. Lower values mean better optimization. These final gaps
were identical at 50 and 100 queries for every tested beta.

| Gate length | Beta 3 | Beta 2 | Beta 1 |
| --- | ---: | ---: | ---: |
| 500 nm | 17.45 | 17.45 | 17.45 |
| 1,000 nm | 188.25 | 174.48 | 1.32 |
| 2,000 nm | 37.95 | 37.40 | 11.29 |

All tested SafeOpt runs had zero observed constraint violations. At beta 1,
it explored a mean of 3.37, 26.20 and 11.87 distinct recipes for the 500,
1,000 and 2,000 nm devices. The additional queries from 51 to 100 revisited
previous recipes in every SafeOpt run. Lowering beta improved performance
on two devices. Increasing the budget beyond 50 did not improve its final
recipe on any of these pools.

EI found the best safe pool recipe by 50 queries in every run. Its mean
violation counts rose from 22.53 to 30.03 on the 500 nm device, from 15.50 to
16.27 on the 1,000 nm device, and from 17.23 to 23.17 on the 2,000 nm device
when the budget increased to 100. Its seed and query sequences were unchanged
across beta values.

The starting recipes also match across all confidence settings. EI ignores
beta, so its query sequence must stay identical when beta changes. The
controller checks this and checks that the first 50 queries of the beta-3
run match the earlier experiment. It audits every saved query and learning
curve against the simulation CSVs.

## Running and viewing the study

From the project root:

```powershell
python .\bo_comparison\run_sensitivity.py --input outputs_parallel_100 --output .\bo_comparison\sensitivity --workers 3
python .\bo_comparison\plot_sensitivity.py --input .\bo_comparison\sensitivity --show
```

To view the supplied results, run only the second command. Omitting `--show`
saves PNGs without opening Matplotlib windows. An interrupted study resumes
completed pairs when you rerun the first command with the same arguments.

## What changes

The GP confidence interval is mean plus or minus beta times standard
deviation. A smaller beta makes the lower safety bound less pessimistic. It
can let SafeOpt try more recipes. It also changes the confidence bounds used
to identify potential objective maximizers and safe-set expanders. The GP
kernel, length scale, output scales and numerical noise are unchanged.

The controller runs each policy for 100 queries and scores its first 50
queries separately. The optimizer decides sequentially using revealed
outcomes. Its selection rule does not depend on the final budget. Therefore
these prefixes represent the same policy stopped at 50 queries.

Both methods may revisit recipes, and every visit costs one query, including
initialization. The replay uses the saved deterministic response again for
a revisit. The source pools contain 100 distinct recipes each. Larger query
budgets can be spent on repetition when the optimizer cannot certify new
recipes or has already found the best recipe it wants to exploit.

## Reading the results

- `summary.csv` has one row for each device, beta, budget and method. It shows
  mean violations, fraction of zero-violation runs, distinct recipes queried,
  final safe-objective gap, cumulative best-safe gap and average best-safe gap.
- `runs.csv` contains the same measures for each individual run.
- `paired_changes.csv` compares budgets within a beta and compares lower betas
  with beta 3 at the same budget. Differences pair the same initial recipes.
  It includes per-comparison 95% bootstrap intervals over the 30 run pairs.
- `study.json` records the tested grid, dataset hashes and scoring choices.
- `plots/sensitivity_overview.png` shows SafeOpt's violations, final safe gap
  and exploration across the settings. Horizontal lines show the EI baseline.
- `plots/beta_1`, `plots/beta_2` and `plots/beta_3` contain the familiar learning
  curves over the full 100 queries.
- Each `beta_*_budget_100` folder contains raw query logs and its own manifest.

Lower final safe-objective gap means the optimizer found a recipe closer to
the best safe candidate in the pool. Lower violations means fewer unsafe
evaluations. Higher distinct-recipe count means more exploration.

Raw cumulative gap depends on how many budget steps are summed. Use final
gap and `mean_best_safe_gap` to compare 50 with 100 queries. The latter divides
the cumulative gap by the budget. The study keeps the raw sum as well.

These are exploratory comparisons on fixed finite pools. A smaller beta can
permit more violations. Zero observed violations in this study does not
establish a formal safety guarantee. The bootstrap intervals describe variation
across initial safe-recipe choices. The settings share seeds and prefixes, so
their results are matched observations. Results from several betas should
not be counted as independent additional EI experiments.
