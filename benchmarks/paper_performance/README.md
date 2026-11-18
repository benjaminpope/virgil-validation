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
