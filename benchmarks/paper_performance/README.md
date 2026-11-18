# Paper performance benchmarks (scaffold)

Speed, memory and accuracy of virgil against PMOIRED (binary fits), Octofitter.jl (orbit fits) and eht-imaging (image
reconstruction), on one Apple-silicon laptop CPU. This directory is the shared scaffold (WP-0); each comparison adds its
own subdirectory (`binary/`, `orbit/`, `imaging/`) with its own `cases.yml`, and a final package runs the matrix and
writes `results/`. Nothing here edits `src/`, `tests/`, `trust/` or `pyproject.toml`.

## Environment

One shared environment outside every worktree: `~/.cache/virgil-bench/venv` (Python 3.12).
Build it once, from any worktree cut from `origin/main`:

```
benchmarks/paper_performance/env/setup_bench.sh
```

It reads `VIRGIL_REF` from that worktree's `.github/workflows/tests.yml` (never hard-coded), installs
`virgil-astro[orbits,pipeline]` at that commit, `pmoired==26.10.1`, ArviZ, NumPyro, AstroPy, SciPy, PyYAML and pytest,
writes the commit to `~/.cache/virgil-bench/VIRGIL_REF` and `env/requirements-bench.lock`, and refuses to run with less
than 6 GB of free disk. Every package uses it by activating:

```
source ~/.cache/virgil-bench/activate.sh     # from inside the worktree; sets BENCH_PYTHON, EHTIM_PYTHON, PYTHONPATH
~/.claude/bin/pyquick $BENCH_PYTHON <script>
```

`PYTHONPATH` points at the current worktree's `src/` (override with `VV_SRC`), so `crosscheck` and `external_bridge`
come from the worktree while virgil and PMOIRED come from the shared environment. `EHTIM_PYTHON` is the existing
`.venv-ehtim` (eht-imaging 1.3.2) of the main checkout; it is never modified.

Pin: virgil `c2be8278a4d6549282b3cf199852101929df0462` (the 0.4.2 release candidate that vv CI pins; its `__version__`
string reads 0.4.1). The paper's `\version` strings (`v0.4.2` in `main.tex`, `v0.4.1` in `numbers.tex`) are listed
beside it; reconciling them is outside this benchmark. The commit and the version string are recorded in every row.

## Protocol in brief

* Predeclare, then run: cases, settings and metrics are committed in each `cases.yml` before any measured run; a later
  change is a new commit with a line in that directory's `CHANGES.md`.
* Report everything: failures, timeouts, out-of-memory kills, rows where virgil is slower or less accurate. No row is
  rerun because of its value; only rows flagged `noisy` or `threads_not_held`, once, keeping both.
* No tuning toward a result. Three configurations: `matched` (same data, model, priors, noise, budget, after a gate
  against a neutral NumPy reference), `tight` (both codes to a tight predeclared tolerance), `native` (each code's
  documented default workflow).
* One fresh process per (code, case, config, thread setting, process repeat), under `/usr/bin/time -l`. Inside the
  process: `t_calib`, `t_import`, `t_load`, `t_init`, `t_cold` (first call, includes JIT), `t_warm[]`. Repeats 3 x 5
  (3 x 3 above 60 s per call; 1 x 1 above 10 min, flag `single_run`).
* Threads: 1 primary, 2 as a second pass for every `native` row and binary T3; never more. Float64, CPU only.
* Peak memory: `peak memory footprint` from `/usr/bin/time -l` (what `pyslot` enforces) and maximum RSS.
* Measured runs go through `PYSLOT_MEM=6G ~/.claude/bin/pyslot /usr/bin/time -l ...` under `caffeinate -i -s`, on AC
  power with Low Power Mode off. Smoke tests use `pyquick`.

## Files

| File | Role |
| --- | --- |
| `common.py` | `timed_process()`, `provenance()`, `machine_state()`, `check_threads()`, `write_row()`, `validate_row()`, `Stopwatch` / `emit_inner()` for workers, repeat plan, noisy flag, manifest verification |
| `neutral.py` | NumPy reference scorers: `chi2_v2_cp`, `astrometry_loglike`, `image_objective` (plus `binary_vis`, `tv2_cost`, `dft_matrix`); imports neither virgil nor an external package; `--selftest` |
| `calib.py`, `calib.jl` | fixed calibration loops (about 1 s) giving `t_calib` |
| `cases.yml` | protocol-wide predeclared settings; per-comparison cases are in each subdirectory |
| `env/setup_bench.sh`, `env/requirements-bench.lock` | the shared environment and its frozen package list |
| `tests/` | tests of the harness and scorers (`pytest.ini` here replaces the repository's addopts) |

### Worker protocol

A worker is a script the harness launches in a fresh process. It times its sections with `common.Stopwatch`
(`t_calib`, `t_import`, `t_load`, `t_init`, `t_cold`, repeated `t_warm`), blocks on JAX outputs with `common.block`, and
ends with `common.emit_inner(sw, n_data=..., n_param=..., iterations=..., accuracy=..., settings=..., provenance=...)`,
which prints one `BENCH_INNER {json}` line. The harness turns that plus the `time -l` numbers into a row with
`common.row_from_process` and writes it with `common.write_row`.

Row schema (JSON, one file per process in `results/raw/`): case, comparison, code, config, threads, size, status
(`ok`, `timeout`, `oom`, `error`, `gate_failed`, `not_run`, `unsupported`), t_process, t_calib, t_import, t_load, t_init,
t_cold, t_warm, peak_footprint_mb, peak_rss_mb, baseline_mb, cpu_ratio, flags, n_data, n_param, iterations, accuracy,
settings, provenance. Non-`ok` statuses carry a `reason` in `settings` or `accuracy`.

## Commands

Self-test of the scorers, tests, and a dummy row (each under `pyquick`):

```
~/.claude/bin/pyquick $BENCH_PYTHON benchmarks/paper_performance/neutral.py --selftest
(cd benchmarks/paper_performance && ~/.claude/bin/pyquick $BENCH_PYTHON -m pytest)
~/.claude/bin/pyquick $BENCH_PYTHON benchmarks/paper_performance/common.py --dummy-row /tmp/dummy-row
```

## Limits (stated in every output)

One laptop CPU, no GPU row; 1 thread primary and 2 at most, nothing on all cores; eht-imaging without NFFT; Octofitter
interferometry on simulated point-source data only; the HD 224974 orbit case is period-informed; PMOIRED and Octofitter
parallel modes not run beyond 2 workers; versions and commits as recorded in each row.

## Results (WP-R, measured run of 2026-11-19)

Outputs: `results/results.csv` (one row per case x task x config x threads x code), `results/raw/*.json` (every process),
`results/performance_table.tex` (tabular only), `results/performance.pdf` / `.png`, `results/summary.json`,
`results/matrix_log.jsonl`, `results/gates/`. Rebuild with `python summarise.py`; `python summarise.py --check` validates
schema and coverage. The run is `./run_all.sh` (`matrix.py`): serial, one process at a time, under `caffeinate`, every measured
process through `pyslot` with `PYSLOT_MEM=6G`, machine idle and on AC power at the start, 7 h cap (it finished at 7.24 h
elapsed including one restart; rows not reached are `not_run`, reason `budget`).

Rows in which each code was faster (paired groups with a time for both codes; ratio = other / virgil of medians):

| Basis | virgil faster | other code faster | not paired / no time |
| --- | --- | --- | --- |
| warm | 27 | 13 | 33 |
| cold | 10 | 32 | 31 |

These counts include the default-method virgil imaging rows paired with the same eht-imaging row. The numbers are what the
runs gave; no row was repeated or dropped because of its value.

### Findings

* I-sim-large gate fails (eht-imaging objective vs neutral, worst relative difference 1.85e-2), and I-sim-large rows were not reached by the budget anyway.
* virgil's chi2 differs from the neutral reference (4-telescope closure-phase whitening): `likelihood_mismatch` is -18 % on binary T2 (B-sim-6) and 2.4e-2 on orbit Level B (independent form; default 0.20). Flagged, not gated.
* O-A-real matched, virgil: `error`. `starting_orbits` seeds are rejected by NumPyro (`Cannot find valid initial parameters`) with the period-informed prior; not repaired. Octofitter ran the same case.
* O-B-sim native, Octofitter (Pigeons, 8 rounds, 1 thread): `timeout` at the 30 min per-call limit.
* eht-imaging `time-to-target` is `not_reached` for I-sim-24/48/96 (it did not come within 1e-3 of the lowest J reached by either code at any budget); best J values are in the ladder rows.
* PMOIRED T3 B-sim-6 matched took about 130-140 s per call (the WP-B estimate was 650 s).

### Flags (all rows)

`threads_not_held` 83 rows (virgil only: binary 13 groups, imaging 13, orbit 3; cpu/wall above 1.2 at 1 thread, JAX/XLA background threads), `likelihood_mismatch` 45, `noisy` 16 rows, `warm_recompiled` 11, `mem_children_not_summed` 4. Flagged rows were rerun once (`matrix.py`) and both rows kept.

### Not run

Budget: the 2-thread native passes for most cases, binary T2 `tight`, T3 `matched-coarse`, imaging `stop_tight`, the I-sim-ring ladder, all I-sim-large rows, O-A-real native, O-B-sim 2-thread. `matched-truthstart`: no documented route for user start values in Octofitter's `initialize!` recorded.

### Run log

The first launch crashed at the first orbit row (a tuple-unpack bug in `matrix.py`) and the orbit gate step used the wrong command line; both were fixed in `matrix.py` and the run resumed (`--resume`) from `results/matrix_log.jsonl` without repeating finished rows. No `cases.yml` was changed. Binary and imaging gates ran before the crash and are unchanged.

### CHANGES entries

`binary/CHANGES.md`, `imaging/CHANGES.md`, `orbit/CHANGES.md`: see those files (orbit: the `likelihood_mismatch` flag on virgil Level B rows, dense mass matrix, Kepler constants).

### Consistency with the paper (flag, no proposed wording)

`main.tex` l.157 says virgil outperforms all other packages in speed (crediting JAX and GPU acceleration). Rows in which virgil is slower than the other code, and the absence of any GPU row, are listed in `results/results.csv` (columns `ratio_warm`, `ratio_cold`; ratios below 1 mean virgil slower).
