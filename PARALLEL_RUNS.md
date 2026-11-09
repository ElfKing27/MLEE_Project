# Dataset workflow and parallel runs

Run from the directory containing `nmos_benchmark` and `configs`:

```powershell
python -m nmos_benchmark generate --device planar_L500_tox10 --points 10 --pilot 6 --workers 2 --timeout 300 --output outputs_parallel_quick
```

This is a quick pipeline test: six pilot recipes per candidate domain, then ten
fresh recipes if a domain qualifies. Use `--pilot 24 --points 100` for the
original study settings. A small pilot may fail to select a domain. Omit
`--device` to process all configured devices, one device batch at a time.

1. `runner.py` loads `configs/devices.json`. Device geometry, baseline doping,
   electrical specifications, and candidate search bounds are fixed for a case.
2. The runner samples a pilot batch uniformly in log dose, time, and
   temperature. Each recipe is a distinct combination of those three inputs.
3. `worker.py` calls `process.py` in a separate Python process. ViennaPS implants
   boron into the silicon slab, models diffusion and solubility-limited
   activation during annealing, and exports an active boron depth profile.
4. A second process calls `device.py`. DEVSIM maps the profile onto the 2D mesh,
   adds the prescribed source/drain and substrate doping, and solves for
   threshold voltage and off-state drain leakage. This models a boron
   threshold-adjust implant; the source/drain implant flow is not simulated.
5. The result records the threshold error and whether leakage exceeds the
   specification. A timeout or solver failure leaves safety unknown.
6. The runner checks the full pilot and tries the next candidate domain if
   needed. It freezes a qualifying domain and samples the final dataset using
   an independent seed. The final 95% one-sided lower bound on the unsafe
   fraction must exceed 30%, with complete responses and safe seed candidates.
7. Read `dataset.csv` and `benchmark_report.json` in the per-device output
   folder. This pipeline does not train a GP or compare EI with SafeOpt.

## What runs at the same time?

With `--workers 2` (the default), two recipes can be evaluated at once. Within
each recipe, ViennaPS still finishes before DEVSIM starts. The two recipes can
then use separate DEVSIM processes simultaneously. Each solver remains limited
to one native thread. This improves batch throughput without changing the
individual solver's mesh, precision, or bias steps.

The coordinator alone writes CSV files and keeps rows in sampling order, even
if completion is out of order. It still finishes one pilot before selecting
another domain or starting final validation. `--workers 1` restores serial
execution. `--workers` also applies to `process-only`; `smoke` remains a
single-recipe-per-device diagnostic. On Ctrl+C, queued recipes are cancelled;
active evaluations may take time to exit and can save their cached results.

Start with two workers. Increase the count only if the computer has spare CPU
and memory; throughput does not necessarily scale linearly. The `--timeout`
limit applies to each recipe independently.

Keep previous outputs in their existing directories. This runner update changes
the code fingerprint, so the old runner's cache is not automatically reused.
Using a new output directory keeps the previous CSV files separate. Restarts
with the updated runner reuse matching cached results in that directory.
