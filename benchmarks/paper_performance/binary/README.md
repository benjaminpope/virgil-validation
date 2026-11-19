# Binary fitting: virgil against PMOIRED (WP-B)

Speed, memory and accuracy of virgil and PMOIRED 26.10.1 on the same simulated and real interferometric binary data, on
one Apple-silicon laptop CPU (1 thread primary, 2 at most; no GPU; float64). Plan: `reports/benchmarks-plan.md`
section 4 (kept with the paper repository). Every case, setting and threshold is predeclared in `cases.yml`; changes
after the first measured run go in `CHANGES.md`. Measured numbers come from the matrix run (WP-R), not from this
directory: the smoke timings below only show that every task runs.

## What is here

| File | Role |
| --- | --- |
| `cases.yml` | cases, tasks, settings, gate thresholds, documentation sources |
| `make_data.py`, `manifest.json` | deterministic simulated data (`crosscheck.simulate.observe`) under `~/.cache/virgil-bench/data/binary/`; the committed manifest holds the sha256 of every simulated file and of the two real NN Del files |
| `run_virgil.py`, `run_pmoired.py` | one fresh process per (task, case, config, threads); each writes one row (harness schema); `--worker` is the inner process |
| `compare.py` | `--gate` (both codes against the neutral reference), `--rows DIR` (paired table) |
| `_shared.py` | case and start-value definitions, neutral scoring on the OIFITS files, row plumbing (imports neither code) |
| `test_binary.py` | tests of the scaffolding (`pytest tests binary` from `benchmarks/paper_performance`) |

Independence (vv AGENTS.md): the simulated data and truth come from `crosscheck`; `run_pmoired.py` never imports virgil and
`run_virgil.py` never imports PMOIRED; both score their optimum with `neutral.chi2_v2_cp` (NumPy and astropy only).

## Commands (from the repository root, after `source ~/.cache/virgil-bench/activate.sh`)

Build the data (about 20 s) and check the manifest, then the gate; smoke one task per code (all `pyquick` or `pyslot`):

```
~/.claude/bin/pyslot $BENCH_PYTHON benchmarks/paper_performance/binary/make_data.py
~/.claude/bin/pyquick $BENCH_PYTHON benchmarks/paper_performance/binary/compare.py --gate
~/.claude/bin/pyquick $BENCH_PYTHON benchmarks/paper_performance/binary/run_virgil.py --task T2 --case B-sim-6 --size tiny --out /tmp/rows
~/.claude/bin/pyquick $BENCH_PYTHON benchmarks/paper_performance/binary/run_pmoired.py --task T2 --case B-sim-6 --size tiny --out /tmp/rows
```

`run_all.sh` (WP-R) passes `--wrapper pyslot` and no `--out`; `--size full` is the default. `--task` is one of `gate`,
`T1`, `T2`, `T3`, `pulls`, `import` (an import-only process for the memory baseline); `--config` one of `matched`,
`tight`, `native`, `matched-coarse`; `--threads 2` is the second pass (PMOIRED `multi=2` in T3).

## Tasks, as implemented

| Task | virgil | PMOIRED | Notes |
| --- | --- | --- | --- |
| T1 forward model | jitted `data.standardize_model(model.model(u, v, wavel))`, V2 + closure-phase vector | `oimodels.VmodelOI(oi.data[i], params)` per file | per-call time (batches of 100); PMOIRED's call also fills fields virgil's does not (stated as a limit of the comparison) |
| T2 local fit | `fit(method="lm", gtol)` then `inference.laplace_cov`, timed separately (`iterations.t_fit_*`, `t_laplace_*`) and summed | `OI.doFit(..., ftol, epsfcn=1e-6, maxfev=10000)`; sigma divided by sqrt(chi2_red) as the vv bridge does | start = truth/reference + (0.3, -0.3) mas, flux x 1.3 |
| T3 blind search | `likelihood_grid` (40 x 40 over +-20 mas, flux 0.05) + `best_grid_point`, then the T2 fit and Laplace covariance | `OI.gridFit` on the same 40 x 40 starts (`multi=False`; `multi=2` in the 2-worker pass), its best fit | task-level comparison: exhaustive vectorised grid against multi-start least squares |
| pulls | the T2 fit over 200 files per noise model (read, fit, covariance), timed as one loop | the same | pull = (fit - truth) / sigma |

Both codes use V2 and closure phases only, no bandwidth smearing, no uncertainty rescaling. PMOIRED runs with
`auto=False` (its default `auto=True` gives NaNs on these files, vv ledger P1) and its default `autoPrior` (>= 0 on flux
and size parameters); virgil uses `Uniform(-60, 60)` mas on the offsets, `Uniform(0, 1)` on the flux ratio and
`Uniform(0, 5)` mas on the uniform-disk diameter (as `virgil_bridge`).

## Gate (before any timing)

`compare.py --gate` runs both workers once at full size and writes `gate_results.json`. The model V2 and closure phase at
the truth, on the noise-free files of B-sim-6 and B-sim-ud, must reproduce crosscheck's values to 1e-8 (V2) and 1e-6
degrees (closure phase); chi2 at the truth on the noisy files is compared with `neutral.chi2_v2_cp`.

Result (`gate_results.json`, status `pass`), against the noise-free crosscheck file and `neutral.chi2_v2_cp`:

| Case | Code | max abs V2 diff | max abs CP diff [deg] | chi2 at reference | neutral chi2 | relative difference |
| --- | --- | --- | --- | --- | --- | --- |
| B-sim-6 | virgil | 6.7e-16 | 2.5e-14 | 206.43 (390 rows) | 251.43 | -0.179 (recorded, not gated) |
| B-sim-6 | PMOIRED | 8.9e-16 | 5.7e-14 | 251.43 (301 rows) | 251.43 | -1.1e-16 |
| B-sim-ud | virgil | 8.9e-16 | 2.5e-14 | 231.44 | 282.20 | -0.180 (recorded, not gated) |
| B-sim-ud | PMOIRED | 4.0e-13 | 4.5e-12 | 282.20 | 282.20 | -1.1e-13 |

Thresholds: V2 1e-8, closure phase 1e-6 deg, PMOIRED chi2 1e-6 relative. Both models reproduce the reference
visibilities to rounding error; PMOIRED's chi2 equals the neutral one to machine precision (its 301 rows are the 300 data
plus one prior row, zero at the truth).

virgil documents no switch for independent closure-phase errors at this pin (closure phases of 4 telescopes are whitened
as a correlated group, with a sine-chord residual and periodic penalty rows), so virgil's default is used and the
difference of its chi2 from the neutral one is stored as `likelihood_mismatch_rel` in every virgil T2, T3 and pulls row
(flag `likelihood_mismatch`). That value (about -18 per cent here) is a finding about the definitions, not a failure of the
gate: virgil whitens the four-telescope closure phases as a group (3 independent combinations per exposure plus the
sine-chord and penalty rows, 390 rows for 300 data), which differs from the independent-Gaussian neutral chi2.

## Smoke timings (tiny size; one process each, laptop busy with other sessions; not measurements)

Tiny size is 6 channels, 8 x 8 grid, 5 pull seeds; cold (first call, includes JIT for virgil) and first warm repeat.

| Task, case | virgil cold / warm [s] | PMOIRED cold / warm [s] | Result |
| --- | --- | --- | --- |
| T1 B-sim-600 | 0.08 / 5e-4 | 0.006 / 5.4e-3 | per call |
| T1 B-real-pionier | 0.06 / 1e-4 | 0.004 / 3.1e-3 | per call |
| T2 B-sim-ud | 11.1 / 0.05 | 0.19 / 0.10 | rows valid |
| T2 B-sim-6, tight | 6.3 / 0.02 | 0.03 / 0.03 | rows valid |
| T2 B-real-pionier | 3.4 / 0.01 | 0.03 / 0.03 | both near the reference, `swapped` false (dra, ddec, flux about 0.115, 3.618, 0.53 in both) |
| T3 B-sim-6, native (tiny) | 2.4 / 0.009 | 3.3 / 2.7 | both converge to the truth |
| T3 B-sim-6, matched, full-size virgil | 13.7 / 0.04 | not run (about 650 s per call) | virgil recovers the truth (within 3 sigma) |
| T3 B-sim-6, 2 threads | valid | valid, flagged `mem_children_not_summed` | tiny grid |
| pulls tri / base, 5 seeds (whole loop) | 3.0 / 3.4 | 0.13 / 0.09 | n_failed 0 |
| import-only (peak footprint) | 174 MB | 169 MB | memory baseline |

At the tiny 8 x 8 grid (spacing about 5.7 mas) virgil's T3 can land in a wrong local minimum (it does for the real case
and for the 2-thread B-sim-6 row); this is expected, and is why the full 40 x 40 grid is the benchmark. `threads_not_held`
appeared on some virgil smoke rows because other jobs were running on the laptop; the matrix run must be quiet.

## Estimated full runtime (WP-R budget; single process, 1 thread)

Rough, from the smoke timings (3 processes x 5 warm repeats; 3 x 3 above 60 s; 1 x 1 above 600 s).

| Block | Estimate |
| --- | --- |
| T1, five cases, both codes | under 5 min |
| T2, matched and tight, both codes | under 10 min |
| pulls, 2 noise models x 200 seeds, both codes | about 10 min |
| T3 virgil (40 x 40 matched; native 81 x 81 x 30) | under 5 min |
| T3 PMOIRED matched, 1600 starts, about 650 s per call, 1 x 1 | about 11 min |
| T3 PMOIRED matched-coarse (20 x 20), about 160 s per call, 3 x 3 | about 25 min |
| T3 PMOIRED native (21 x 21), about 3 min per call, 3 x 3 | about 25 min |
| 2-thread pass | about half of the 1-thread T3 time |
| Total, 1-thread pass | about 1.5 h; with the 2-thread pass about 2 h |

## Deviations from the plan and open points

* T3 grid: 40 points per axis from `np.linspace(-20, 20, 40)` in both codes; PMOIRED's `(min, max, step)` rule
  (`n = int((max - min) / step + 1)`) cannot be given exactly 40 points with a round step, so the step is 40/39 x 0.9999
  (`_shared.pmoired_step`, asserted). The plan said "40 x 40 starts"; the same count is kept.
* `native` T3: the plan cites `optimized_likelihood_grid` as documented in `docs/binary_search.md`; that page (virgil
  checkout read, not the pin) shows `likelihood_grid` with a dra x ddec x log-flux cube, which is what is used (81 x 81 x
  30 over +-20 mas and flux 0.01 to 1). No PMOIRED binary-search example with a stated step could be read offline, so the
  native PMOIRED grid uses the `(min, max, step)` form of the `OI.gridFit` docstring with a predeclared 2.0 mas step
  (21 x 21 starts); it is a predeclared choice, not a cited one.
* T2 `native` coincides with `matched` (stated in `cases.yml`); the harness should run `matched` once and report both.
* `pytest` in `benchmarks/paper_performance` collects only `tests/` (its `pytest.ini`); this directory's tests run with
  `pytest tests binary`. A one-line `testpaths = tests binary` in the shared `pytest.ini` would collect both (not edited
  here).
* B-real-pionier: flux ratio taken as companion/primary = 0.599; both codes read the two epoch files unmodified. The
  swapped solution is recorded in `accuracy.swapped`.
