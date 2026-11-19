# Imaging: virgil against eht-imaging (WP-I)

Regularised-maximum-likelihood image reconstruction on simulated visibilities, virgil against eht-imaging 1.3.2, one
laptop CPU, 1 thread primary. Protocol: `reports/benchmarks-plan.md` section 6 and `../README.md`; settings and cases are
in `cases.yml` (predeclared; changes are logged in `CHANGES.md`). Nothing here edits `src/`, `tests/`, `trust/` or
`pyproject.toml`; the vv bridge worker `src/external_bridge/ehtim_worker.py` is imported by path, not edited.

## Files

| File | Role |
| --- | --- |
| `cases.yml` | predeclared cases, `matched` / `tight` / `native` settings, ladder, gate and accuracy rules, with sources |
| `make_data.py` | deterministic simulated data (`crosscheck`), `manifest.json` (sha256, committed), shared helpers (start and truth images, eht-imaging rows, neutral data dict) |
| `run_virgil.py`, `run_ehtim.py` | drivers: `gate` and `run` (one fresh process per row, scored by the neutral scorer outside the measured process) |
| `ehtim_bench_worker.py` | eht-imaging side, runs in `EHTIM_PYTHON`: timing, `maxit` control, the documented `native` recipe; imports `_imager` / `objective` from the bridge worker |
| `compare.py` | driver internals, neutral scoring, gate report, time-to-precision |
| `test_imaging.py` | tests of the helpers (`cd benchmarks/paper_performance && pyquick $BENCH_PYTHON -m pytest imaging/test_imaging.py`) |
| `gate_results.json` | output of `compare.py gate-report` (committed: the gate numbers below) |

## Commands (from `benchmarks/paper_performance/imaging`, after `source ~/.cache/virgil-bench/activate.sh`)

```
~/.claude/bin/pyquick $BENCH_PYTHON make_data.py              # writes ~/.cache/virgil-bench/data/imaging/*, manifest.json
~/.claude/bin/pyquick $BENCH_PYTHON run_virgil.py gate --case I-sim-24
~/.claude/bin/pyquick $BENCH_PYTHON run_ehtim.py   gate --case I-sim-24
~/.claude/bin/pyquick $BENCH_PYTHON compare.py gate-report               # all cases; writes gate_results.json
~/.claude/bin/pyquick $BENCH_PYTHON run_virgil.py run --case I-sim-24 --task ladder --size tiny --out-dir DIR     # smoke
~/.claude/bin/pyquick $BENCH_PYTHON run_ehtim.py   run --case I-sim-24 --task ladder --size tiny --out-dir DIR
```

Measured rows (WP-R only): the same without `--size tiny` and `--out-dir`, `--wrapper pyslot`, `--threads 1|2`, one
process per call, rows in `results/raw/`. Tasks: `ladder` (matched; `--method default` gives virgil's second row,
fit's own default method), `per_eval`, `stop_default` (matched, each code's own stopping rule), `stop_tight` (config
`tight`), `native`. Gate output is read from `~/.cache/virgil-bench/data/imaging/gate/`; a case whose gate failed gets
`status: gate_failed` on its `matched` / `tight` rows (measurements kept).
After the ladder rows of both codes exist, `compare.py time-to-precision --virgil ROW.json --ehtim ROW.json [--virgil-lm ROW.json]` gives the
interpolated time to each precision rung of `cases.yml` (`ladder.time_to_precision`): neutral J within 10 %, 3 %, 1 %, 0.3 %,
0.1 % of the lowest J either code reaches; the headline is the tightest rung both codes reach.

## What is measured

* **Problem.** The `well_posed` problem of the vv test `tests/test_ehtim_reconstruction.py` (3 stations, 13 hour angles,
  6 channels, sigma 0.002, seed 1: 234 amplitudes and 78 closure phases) on 24, 48 and 96 pixel grids over 12 mas;
  a ring scene (48 px); and a gated 4-station case (150 baselines-snapshots x 30 channels). Same data on every grid.
* **Matched.** Same start (circular Gaussian, FWHM 4 mas), amplitude + closure-phase data terms, regulariser weight
  7.5e5 on every grid, exact direct Fourier sum in both codes (eht-imaging `ttype="direct"`, virgil `ft="dft"`),
  L-BFGS-B in both (`maxcor=50` in eht-imaging; virgil `lbfgs_memory=50` to equal it). Budget ladder 25..3200 steps from
  the common start, tolerances set so that the budget binds; every returned image is scored by `neutral.py`:
  `J = 1/2 sum r_amp^2 + 1/2 sum r_cp^2 + 7.5e5 sum (db)^2` on the unit-flux image. Time to target and the `J` against
  wall-time curve come from these scores, never from a code's own objective value.
* **Per evaluation, stopping rules, native.** See `cases.yml`. `native` for eht-imaging is the recipe of its
  `examples/example_im_closure.py` (v1.3.2); for virgil the RML route of `docs/imaging_rml.md`.
* **Timing convention.** virgil: `fit` call to returned model (problem set-up and L-BFGS-B included; the file read is
  `t_load`). eht-imaging: building the Obsdata and Imager from the rows through the returned image
  (`init_imager` and the minimiser included; the simulated rows are built once and counted in `t_load`).
  `t_cold` of a ladder row is the first call (budget 25) in a fresh process; `t_warm` the repeats at that budget; the whole
  curve is in `accuracy.ladder` (`t_warm` per budget, the neutral score, steps / evaluations, stop reason).
  eht-imaging reports function and gradient evaluations (counted by wrappers, removed after each run), virgil steps.
  A row's `size` is `tiny` (smoke, budgets 25 and 50 only) or `full`.

## Gate (reproduced through the new runners; `gate_results.json`)

At the common start and at the rendered truth, on each grid. eht-imaging: its `objective` task with the flux term excluded
(`sum alpha_d (chi2_d - 1) + beta tv2`, constants added back). virgil: `1/2 sum r^2 + TSV` from `whitened_residuals` and
`TSV`. Reference: `neutral.image_objective` (edge convention matched to each code). Relative differences, worst of start
and truth:

| Case | eht-imaging vs neutral | virgil vs neutral | Result |
| --- | --- | --- | --- |
| I-sim-24 | 2.2e-16 | 3.3e-16 | pass |
| I-sim-48 | 6.5e-15 | 3.1e-15 | pass |
| I-sim-96 | 2.8e-14 | 5.9e-15 | pass |
| I-sim-ring | 6.7e-16 | 4.4e-16 | pass |
| I-sim-large | 1.7e-14 | 2.3e-3 (start), 1.8e-2 (truth) | **fail** (virgil) |

The tolerance is 1e-9. The pass confirms `neutral.pixel_grid`'s pixel-centre convention (copied from eht-imaging's
`xlist`/`ylist`) and the Fourier-sign mapping against eht-imaging's own objective, to rounding. The data counts agree
(`n_amp`, `n_cphase` from eht-imaging equal the file's).

**Finding, I-sim-large.** With four stations the closure triangles share baselines. virgil's `OIData` then builds a
`ClosureNoise` model (the whitened-sine residual plus a periodic penalty block, `likelihood._whiten_phases`; 4500
residuals for 2250 triangles), while eht-imaging's term is independent chords. At the rendered truth the amplitude
terms are equal (2250.2819737, both) and the closure-phase terms are 1018.2 (virgil) against 1086.5 (neutral and
eht-imaging). The `matched` and `tight` rows of I-sim-large therefore carry `gate_failed`. `OIData` offers no documented
switch for independent closure-phase errors in the pinned version; setting its `cp_noise` field to `None` after
loading would give the chord form, but that is an internal field and was not used (plan: no change to a gated case in
response to the gate). The three-station cases are unaffected (one triangle per snapshot, no correlation).

## Smoke runs (this package, `--size tiny`, 24 px, budgets 25 and 50, one repeat; `pyquick`, 1 thread, machine shared with other agents)

| Row (I-sim-24, tiny) | virgil | eht-imaging |
| --- | --- | --- |
| ladder: wall of the process | 0.54 s outside the worker, import 5.0 s, load 0.46 s | 0.56 s, import 3.0 s, load 0.002 s |
| ladder budget 25: cold / warm | 1.22 s / 0.039 s | 0.039 s / 0.039 s |
| ladder budget 25: neutral `J` | 3.35e4 | 7.20e4 |
| ladder budget 50: warm; `J` | 0.092 s; 7.83e3 | 0.054 s; 2.44e4 |
| per evaluation (5 calls), cold / warm | 0.52 s / 0.58 ms | 4.2 ms / 0.64 ms |
| stopping rule: default (virgil 9551 steps, eht-imaging 217 evaluations) wall; `J` | 29.1 s; 4.93e3 | 1.0 s; 5.42e3 |
| stopping rule: tight (virgil 5000 steps, limit reached; eht-imaging 5475 evaluations) wall; `J` | 7.2 s; 4.94e3 | 4.2 s; 5.02e3 |
| native: cold / warm; grid | 28.8 s / 14.2 s; 41 x 41 | 9.5 s / 4.2 s; 24 x 24 |

Other smoke rows written without error: virgil `--method default` ladder (25 and 50 steps reach `J` 5.0e3 and 4.95e3), and the
budget-25/50 ladders of both codes on I-sim-96, I-sim-ring and I-sim-large. The I-sim-large virgil row is written with
`status: gate_failed` (see the gate). The per-evaluation and stopping-rule numbers above come from the first smoke pass
(some of it with other agents running); the ladder row is from a repeat after eht-imaging's counting wrappers were made
not to keep the Imager alive (a reference cycle that had slowed its warm calls).

These are smoke timings, not results: the machine was shared, repeats are 1, and nothing here is compared.

## Estimated cost of the full runs (for WP-R)

Per ladder pass (all eight budgets, 6375 steps, one repeat), extrapolated from the smoke per-step costs at 1 thread: virgil 11 s at 24 px, about 70 s at 96 px, about 115 s on I-sim-large; eht-imaging 50 s at 24 px, about 280 s at 96 px, about 2100 s on I-sim-large (its cost is dominated by the direct Fourier products, 4500 x 2304 on the large case). With the plan's repeat rule (3 processes x 5 warm repeats, 3 x 3 above 60 s per call, 1 x 1 above 10 min) the matched ladders for I-sim-24, -48, -96 and -ring come to roughly 2 to 2.5 h for both codes together; the per-evaluation, stopping-rule and native rows add about 30 min (virgil's native and default-rule rows take 15 to 30 s per call at 24 px); I-sim-large, a single process per code, adds about 40 min (eht-imaging) plus 5 min (virgil). Total about 3 to 3.5 h, which exceeds the plan's 60 to 90 min; the uncertainty is about 50 percent, the smoke timings were taken on a shared machine. The plan's priority order puts I-sim-96, I-sim-ring and I-sim-large after the 24 and 48 px ladders, so a cut at the budget falls on them.

## Deviations from the plan, and limits

* **`virgil` matched optimiser memory.** `lbfgs_memory=50` equals eht-imaging's `maxcor=50`; the plan names `fit(...,
  method="lbfgsb")` without it. Predeclared in `cases.yml` and unchanged since.
* **Budget tolerance for virgil.** `gtol=0.0` (the smallest accepted); `fit` still derives a floor of `sqrt(eps)` times
  the starting gradient, so a budget can stop early on convergence. `accuracy.ladder[].converged/stop` records this per budget.
* **Step budget semantics.** eht-imaging `maxit` is SciPy `maxiter`; virgil `max_steps` is SciPy `maxiter` as well, and
  virgil also caps function evaluations at 3 x `max_steps`. eht-imaging's `maxfun` is SciPy's default 15000.
* **`native` virgil grid.** `starting_image` sizes its own grid from the data (41 x 41, 0.297 mas, about the same 12 mas
  field); the truth is rendered on that grid for scoring.
* **Truth scoring.** The analytic scene is sampled at pixel centres (not integrated over pixels); both codes are scored
  against the same render.
* **Per-evaluation task.** virgil: jitted `value_and_grad` of the loss over the logits, built from public functions
  (`whitened_residuals`, `TSV.value`) rather than from `fit`'s internal problem; eht-imaging `objfunc` + `objgrad`.
* **Limits.** eht-imaging without NFFT (direct transform in both codes); no real-data case (no OIFITS reader and closure
  phases are derived from complex visibilities); one laptop CPU; simulated data only.
* **Harness.** No change to `common.py` was needed. Rows are written with `write_row(..., name=...)` so that tasks
  and virgil methods do not collide on the default file name (`imaging__<case>__<task>[-method]__<code>__<config>__t<threads>`);
  `settings.task` and `settings.method` identify them. A summariser should group by these.
