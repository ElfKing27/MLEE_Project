# Planar nMOS process/device benchmark dataset

Generate approximately 100 evaluated recipes **per device**, then compare
expected-improvement BO and SafeOpt using the saved simulation outcomes.

The completed [comparison report](bo_comparison/report/README.md) includes
recipe quality, violations, exploration counts and budget/confidence sensitivity.
The optimizer study is a finite-pool numerical benchmark. It measures the
safety-performance trade-off, with nominal query accounting and fixed GP settings.

## What is implemented

* Three configurable planar silicon reference devices: L=500, 1000, 2000 nm;
  oxide thickness=10, 15, 20 nm. They are illustrative numerical devices,
  **not calibrated foundry technologies or published transistor datasets**.
* ViennaPS blanket **boron threshold-adjust implant**, followed by intrinsic
  diffusion and solubility-limited activation.
* Active boron profiles exported as depth-versus-concentration NPZ files.
* DEVSIM 2D Poisson + drift-diffusion with an ideal metal gate, SRH and constant
  mobilities from DEVSIM's simple_physics models. Source/drain and body contacts
  have prescribed analytical profiles formed **after** the channel anneal.
* Leakage at VGS=0 and VDS=1 V; constant-current Vth at VDS=0.05 V. The threshold
  criterion is Id/W=1e-7/L_um A/um. The gate sweep stops at its first crossing.
* Domain selection using separate pilot samples, followed by **fresh IID
  validation samples**, normally 100. No forced safe/unsafe class quotas.
* A one-sided 95% exact binomial lower bound must exceed 0.30. Observing 31
  unsafe points out of 100 is NOT sufficient to establish this claim.
* Caching, per-evaluation logs, frozen configurations, source-table checksums,
  isolated native workers, wall time and DC-solve counts.

**A process profile alone cannot determine electrical safety.** The default
`generate` command runs BOTH stages before labeling any point.

## Quick start (PowerShell, from this directory)

The existing Python 3.12 environment already has the tested packages. For a new
environment use `python -m pip install -r requirements.txt` with Python 3.12.
On Windows DEVSIM needs a working BLAS/LAPACK runtime, for example Intel MKL.
The loader checks `DEVSIM_MATH_LIBS`, then `sys.prefix/Library/bin/mkl_rt*.dll`.
It changes only the current process environment. If needed, install MKL in the
chosen environment (`python -m pip install mkl`) or set `DEVSIM_MATH_LIBS` to
your installed runtime. No DLL renaming or duplicate-OpenMP bypass is used.

```powershell
python -m nmos_benchmark doctor
python -m unittest discover -s tests -v

# One recipe per device: confirm the pipeline before a larger run.
python -m nmos_benchmark smoke --output outputs_smoke

# Start with one device. This can take tens of minutes or longer.
python -m nmos_benchmark generate --device planar_L500_tox10 --points 100 --output outputs

# All three devices, same fixed protocol.
python -m nmos_benchmark generate --points 100 --output outputs

# Only if electrical labeling is deliberately postponed:
python -m nmos_benchmark process-only --points 100 --output outputs_profiles
```

`process-only` leaves electrical response and safety labels unknown. It cannot
certify the unsafe fraction. `generate` integrates the DEVSIM stage already;
you do not need to manually pass its profiles to another script.

ViennaPS and DEVSIM run in **different processes**. On this Windows environment
their native OpenMP runtimes conflict if a process executes both solvers.

## Device and range configuration

Edit `configs/devices.json`. Each case merges a device entry into `defaults`.
Change geometry, background doping, work function, fixed S/D profiles, bias
conditions and electrical specifications there. IDs must be safe filenames.

The starting engineering specification is Ioff/W <= 1e-12 A/um, and the target
Vth is 0.5 V. **These are declared illustrative benchmark specifications**, not
published safe operating limits. Fix justified limits for your intended use
BEFORE dataset construction. The generator never adjusts the leakage limit.

The default hard bounds are dose 1e11..1e13 cm^-2, anneal 5..120 s,
temperature 1000..1100 C. Candidate domains are predeclared subsets of those
bounds. They are experimental search windows, NOT manufacturer-qualified
process windows. Add or override candidate domains per device as required.
The code tries each in order and stops at the first pilot-qualified domain.

The public ViennaPS database is a **model-parameter database**, not a database
of safe MOSFET recipes. It cannot supply device-specific electrical feasibility
ranges. The downloaded implant shape is tabulated at 1e13 cm^-2; varying dose
uses analytical dose scaling. This is a stated reduced-model approximation,
not calibrated dose-dependent implant damage physics. Energy is fixed to a
tabulated value; unsupported energies are rejected, not extrapolated. The
1000 C lower bound follows the documented >=1000 C fit of the selected
solubility parameters. No automatic extension beyond the hard bounds occurs.

## Selection, validation and dataset accounting

1. For each candidate domain, run 24 exploratory IID pilot recipes. At least
   three must be safe, 35..80% must be unsafe, and every response must be complete.
2. Freeze the selected domain and device specifications in `frozen_domain.json`.
3. Draw **100 new IID recipes** and run ViennaPS + DEVSIM on each. These are the
   final `dataset.csv`; they are also the independent validation sample.
4. Accept only if the 95% one-sided lower bound on unsafe fraction is >30%, at
   least three safe points exist, and every point has a complete response.
5. If validation fails, keep the data/report, mark `accepted=false` and stop.
   Do not repeatedly reselect a domain using that same holdout. A subsequent
   exploratory study requires a new output directory, predeclared settings and
   an independent seed. Report unsuccessful attempts, too.

Uniformity means **uniform in log10(dose), linear time and linear temperature**.
Changing this measure changes the meaning of "30% of the space". Intervals
are 95% per device, not a simultaneous 95% guarantee for the whole family.
Even a passing statistical test is evidence, not an absolute guarantee.

Pilot evaluations are ADDITIONAL to the 100 final rows. Three devices with
one successful 24-point pilot each use 372 recipe evaluations, before smoke
tests or BO. Each recipe itself uses one process simulation and many electrical
DC solves. `device_dc_solve_calls` and elapsed time help distinguish these costs.
Extended-precision model/equation/solver evaluation is enabled to resolve tiny
off-state currents. This can be slow on Windows; smoke tests are not timing
estimates for every recipe, and the default 180-second per-recipe timeout may
need increasing via `--timeout` after reviewing the simulator logs.
No physical catastrophe is implied merely by an illustrative leakage violation.

No matching domain may exist under your fixed models/specifications/bounds.
The code reports this instead of manufacturing a dataset to meet the target.
One hundred samples are a small pilot dataset in 3D: they do not resolve every
narrow failure boundary, certify every candidate, or establish an exact optimum.

## Outputs

For each device under `outputs/`:

* `pilot_0.csv`, etc.: every domain-search observation, including failures.
* `pilot_report.json`: exploratory domain-selection results.
* `frozen_domain.json`: selected domain, device configuration and seed.
* `dataset.csv`: 100 final validation recipes; failed points remain in the file.
* `benchmark_report.json`: counts, confidence bounds and acceptance decision.
* `safe_seed_candidates.csv`: safe final observations, with responses, not GP certification.

Per-recipe `evaluations/<device>/<hash>/` folders contain request/configuration,
source hash, library versions, process profile, partial IV curve, structured
results and the simulator log. Identical requests resume from cache. Failed
evaluations are retained; use a new configuration/output for a changed study.

Main columns include device ID, dose (cm^-2), time (s), temperature (C), Vth (V),
Ioff/W (A/um), safety margin `ln(limit/Ioff)`, violation flag, nominal objective
`-|Vth-target|`, extraction status, solver status, timing and profile/curve paths.
`yield_estimate` stays blank: this implementation does not simulate manufacturing
variability and must not be described as measuring yield.

Numerical/process failures have an UNKNOWN safety label. They never count as
unsafe just to inflate the fraction. A missing Vth crossing is reported as
`threshold_not_bracketed`, not invented or clipped to the sweep boundary.

## BO comparison and reproduction

The `bo_comparison` folder contains the optimizer code, a compact tracked report
and the 300 simulated recipe responses needed to reproduce the study. Raw
simulation caches and detailed optimizer traces stay local. Start with the
[report and reproduction commands](bo_comparison/report/README.md) or the
[code walkthrough](bo_comparison/README.md).

The full dataset is a hidden benchmark oracle, not initial training data for
both optimizers. If operating on the 100-point finite pool, disclose that finite
domain and withhold all unqueried responses. For continuous BO, use the live
evaluator or explicitly disclose an additional validated surrogate. Do not claim
that interpolation of 100 rows is exact TCAD. Charge initial seed observations
equally to both algorithms; a known-safe seed here is model-safe, not a hardware
safety certification. Domain qualification and dataset construction are separate
offline costs. The report records the fixed GP/kernel/noise/confidence choices
used in the completed experiment.

## Physical limits and validation still needed

This is a working **reduced numerical benchmark**. Device configurations and
electrical models still need independent calibration and mesh-convergence checks
before semiconductor engineering claims. No full process sequence, dynamic
activation kinetics, implant-damage/TED calibration, concentration-dependent
mobility, gate tunneling, breakdown, quantum effects or measured yield is modeled.
Consequently temperature/time effects may be weaker than in a calibrated process.
Parameters labelled "source/drain" are fixed analytical device profiles, not
another simulated implant in this version. Do not generalize conclusions to all
planar nMOSFETs, and do not include scaled devices beyond these model assumptions.

## Sources and provenance

* ViennaPS implantation: https://viennatools.github.io/ViennaPS/models/prebuilt/ionImplantation.html
* ViennaPS annealing: https://viennatools.github.io/ViennaPS/models/prebuilt/anneal.html
* Model database: https://viennatools.github.io/ViennaPS/models/modelDatabase.html
* DEVSIM models: https://github.com/devsim/devsim/tree/main/python_packages
* DEVSIM native-mesh contact requirements: https://devsim.net/meshing.html

`data/modeldb/provenance.json` records the upstream revision and SHA-256 hashes
of the two public tables. Their headers retain original provenance, including
Pichler's diffusivity/solubility sources and MCIMPL-II implant-profile provenance.
ViennaPS is GPL-3.0; consult its upstream license for redistribution/use. DEVSIM
is Apache-2.0. No third-party Python source is copied into this project.
