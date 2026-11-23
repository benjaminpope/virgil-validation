# Paper performance benchmarks (scaffold)

Speed, memory and accuracy of virgil against PMOIRED (binary fits), Octofitter.jl (orbit fits) and eht-imaging (image
reconstruction), on one CPU (run r1: an Apple-silicon laptop; run r2, the reported results: one OzSTAR EPYC node). This directory is the shared scaffold (WP-0); each comparison adds its
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

Run r2 (PR 275 review) pins virgil `09b268521072a83f73bc0a77defa6485db270c7b` (main after #742: opt-in
`enable_compilation_cache`, fewer eager compiles; also reads 0.4.1). The harness turns the cache on through virgil's own
`enable_compilation_cache` (cache cleared before each cold measurement; a separate warm-with-cache column). Run r1 rows
stay at `c2be8278`; every row records its own `virgil_pin`. Every virgil binary and imaging row and the orbit rows are
rerun at the new pin; r1 rows of the other codes are unchanged. CI still pins `c2be8278` in `tests.yml`.

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

## Follow-up stages (rerun, GPU arm, profiling)

* **Rerun of the budget-cut rows.** `matrix.py --run --run-id r2 --missing-from <r2 log> --shard K/N [--add-case O-A-real]`
  runs only the specs the r2 log records as `not_run`, split into `N` balanced shards of whole blocks (per-row estimates;
  each shard writes `matrix_log_r2_s<K>.jsonl` and `DONE_r2_s<K>`). The rows replace the `not_run` rows of the same name.
  `--add-case C` reruns every spec of a case (O-A-real: its virgil rows errored in r2 on a start-window bug, fixed in `orbit/run_virgil.py`; the start rule is now derived from the priors for every orbit case, `orbit/README.md` "Chain starts").
* **GPU arm.** `BENCH_DEVICE=gpu` (set by the OzSTAR GPU stage) keeps only virgil's one-thread specs, runs them with
  `JAX_PLATFORMS=cuda` and records `settings.device = "gpu"` and `settings.gpu` (the model). The GPU rows never enter the CPU
  tables: `summarise.py` writes them to `results/gpu_results.csv` beside the CPU virgil and other-code warm times of the same group.
  Other codes stay on the CPU.
* **Profiling the warm orbit fits** (`orbit/profile_virgil.py`, `orbit/profile_octofitter.jl`, `jaxprof.py`): the time of
  one log-density call, one gradient call and the 4-chain vmapped gradient in each code, the warm call's wall time per leapfrog
  step, an XProf trace (`xplane.pb`) of virgil's warm NUTS call and of a gradient loop (`JAX_PROFILE_DIR`), and a Julia flat
  profile of Octofitter's gradient. Compare "time per step" with "time per gradient": a large ratio means the cost is outside
  the gradient (sampler loop, lock-step chains, dispatch).

## Limits (stated in every output)

One CPU (r1 laptop; r2 one 64-core EPYC node), no GPU row; 1 thread primary and 2 at most, nothing on all cores; eht-imaging without NFFT; Octofitter
interferometry on simulated point-source data only; the HD 224974 orbit case is period-informed; PMOIRED and Octofitter
parallel modes not run beyond 2 workers; versions and commits as recorded in each row.

## Results (WP-R run r2: OzSTAR CPU node, supersedes the laptop run r1 of 2026-11-19)

**Machine changed from the first digest (2026-11-19, Apple M4 laptop, run r1).** This run (r2) is one OzSTAR/NT job (21374978) on node dave130: AMD EPYC 7543 (Milan, 64 cores allocated of 64, 251 GB), governor performance, ~13.0 h elapsed of the 13 h budget, harness load check on the node (idle at start), 1 thread primary and 2 threads where the plan lists them, no GPU. Every row below is from this one machine. virgil at 09b26852 (opt-in `enable_compilation_cache`, v0.4.2), vv at 5a91ffd, JAX/jaxlib 0.12.0.1, numpyro 0.22.0, PMOIRED 26.10.1, eht-imaging 1.3.2, Octofitter 9.0.0 (Julia 1.12.7); Python codes inside the virgil apptainer image. The laptop rows (r1) stay in `results/raw` but no group is built from them: groups with no r2 row are `not_run`.

Columns: fresh = new process, import + load + init + first call (like-for-like cold); cold = first call only (compile included); warm = a second call, same process. Times in seconds, medians. Ratio = other / virgil (above 1: virgil faster).

## Headline rows (1 thread)

| comparison | case | task | config | thr | virgil fresh | virgil cold | virgil warm | other fresh | other cold | other warm | ratio fresh | ratio cold | ratio warm | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| binary | B-sim-6 | T1 | matched | 1 | 4.3 | 0.0814 | 5.68e-05 | 4.97 | 0.00263 | 0.0022 | 1.16 | 0.0324 | 38.7 |  |
| binary | B-sim-6 | T2 | matched | 1 | 6.67 | 2.48 | 0.0138 | 5 | 0.0272 | 0.026 | 0.749 | 0.011 | 1.89 | likelihood_mismatch;threads_not_held |
| binary | B-sim-600 | T2 | matched | 1 | 6.8 | 2.52 | 0.069 | 5.1 | 0.13 | 0.125 | 0.751 | 0.0515 | 1.81 | likelihood_mismatch;threads_not_held |
| binary | B-sim-6 | T3 | matched | 1 | 6.99 | 2.78 | 0.0213 | 397 | 392 | 392 | 56.8 | 141 | 1.84e+04 | likelihood_mismatch;threads_not_held |
| binary | B-real-pionier | T2 | matched | 1 | 6.71 | 2.54 | 0.0136 | 4.95 | 0.0372 | 0.0362 | 0.737 | 0.0146 | 2.65 | likelihood_mismatch;threads_not_held |
| binary | B-real-pionier | T3 | matched | 1 | 7.04 | 2.8 | 0.02 | 253 | 248 | 248 | 36 | 88.8 | 1.24e+04 | likelihood_mismatch;threads_not_held |
| orbit | O-A-sim | matched | matched | 1 | 19.3 | 12.4 | 3.85 | 62.3 | 7.38 | 1.97 | 3.23 | 0.595 | 0.512 | threads_not_held |
| orbit | O-A-sim-pulls | matched | matched | 1 | 162 | 144 | 4 | 153 | 43 | 1.88 | 0.945 | 0.299 | 0.468 | threads_not_held |
| orbit | O-B-sim | matched | matched | 1 | 211 | 194 | 170 | 262 | 152 | 152 | 1.24 | 0.785 | 0.89 | likelihood_mismatch |

## Imaging: time to precision (virgil L-BFGS-B ladder = `virgil`, virgil Levenberg-Marquardt route = `virgil_lm`, vs eht-imaging)

Rung = relative excess of the objective J over J_ref (the lowest J reached by any code on that target). Entry = "t / t_cold": seconds to reach it once the process is up, and the same plus the cold overhead (import + load + init + first call); `—` = not reached within the budget.

| case | route | cold overhead | 10 % | 3 % | 1 % | 0.3 % | 0.1 % |
|---|---|---|---|---|---|---|---|
| I-sim-24 | virgil_lm | 6.85 | 0.17 / 7.02 | 0.17 / 7.02 | 0.276 / 7.13 | 0.984 / 7.84 | 3.43 / 10.3 |
| I-sim-24 | ehtim | 1.8 | 0.249 / 2.05 | 0.521 / 2.32 | — | — | — |
| I-sim-24 | virgil | 5.39 | 0.0859 / 5.47 | 0.169 / 5.56 | 0.596 / 5.98 | — | — |
| I-sim-48 | virgil_lm | 6.79 | 0.72 / 7.51 | 1.03 / 7.82 | 2.66 / 9.45 | 9.35 / 16.1 | 23 / 29.8 |
| I-sim-48 | ehtim | 1.8 | 3.07 / 4.87 | — | — | — | — |
| I-sim-48 | virgil | 5.34 | 0.667 / 6.01 | 1.49 / 6.84 | 4.43 / 9.77 | — | — |
| I-sim-96 | ehtim | 1.82 | 20.3 / 22.1 | — | — | — | — |
| I-sim-96 | virgil | 5.13 | 4.44 / 9.57 | 8.15 / 13.3 | 13.5 / 18.6 | 15.9 / 21.1 | 16.7 / 21.8 |

## Which code was faster (paired groups with times for both)

| Basis | virgil faster | other faster | unpaired / no time |
|---|---|---|---|
| warm | 30 | 8 | 25 |
| cold_first_call | 14 | 22 | 22 |
| fresh | 24 | 17 | 22 |

Geometric mean of ratio (other / virgil), r2 rows, all virgil routes:

| comparison | basis | n | geo-mean | min | max | virgil faster | geo-mean, threads-held rows only (n) |
|---|---|---|---|---|---|---|---|
| binary | fresh | 23 | 4.16 | 0.632 | 124 | 18 | 2.75 (7) |
| binary | cold | 23 | 1.25 | 0.00857 | 300 | 13 | 0.315 (7) |
| binary | warm | 20 | 74.7 | 1.48 | 1.84e+04 | 20 | 143 (7) |
| orbit | fresh | 6 | 2.23 | 0.945 | 4.08 | 5 | 2.45 (3) |
| orbit | cold | 6 | 0.743 | 0.299 | 3.88 | 1 | 1.2 (3) |
| orbit | warm | 6 | 0.799 | 0.468 | 3.82 | 1 | 1.25 (3) |
| imaging | fresh | 20 | 0.296 | 0.107 | 1.24 | 2 | 0.318 (2) |
| imaging | cold | 20 | 0.0744 | 0.00277 | 1.47 | 2 | 0.00871 (2) |
| imaging | warm | 20 | 0.506 | 0.0235 | 10.9 | 9 | 0.926 (2) |

All paired groups: `results/results.csv`, and the digest `reports/benchmarks-results.md` of virgil-paper.

### Findings

- Binary, warm call (ratio other / virgil, 1 thread, matched): T1 21-90 (B-sim-6 38.7, B-sim-60 21.4, B-real-pionier 90.5); T2 1.8-2.7; T3 1.2e4-1.8e4 (PMOIRED 250-390 s per fit at B-sim-6 and B-real-pionier, 880 s at B-sim-60, against virgil 0.02-0.06 s). All 20 binary warm ratios favour virgil (min 1.48, geo-mean 74.7).
- Binary, first call (cold) and fresh process: virgil pays 2.5 s compile on T2/T3 and has about 4.3 s of import + init; T1 and T2 fresh ratios are 1.16 and 0.74-0.75 (PMOIRED faster on T2 fresh); T3 fresh 36-124 (virgil faster).
- Orbit, warm (Octofitter faster in all three matched Level A/B groups): O-A-sim 3.85 s vs 1.97 s (ratio 0.51), O-A-sim-pulls 4.0 vs 1.88 (0.47), O-B-sim 170 vs 152 (0.89). Fresh (process incl. Julia / JAX start-up): ratios 3.2, 0.95, 1.24 (virgil faster or level).
- O-A-real: the virgil rows are `error` in both configs (matched t1, native t2): NumPyro `Cannot find valid initial parameters` in `chain_init_params`. Cause (review, confirmed): the harness start orbit had `P` on the closed bound of the period prior (`starting_orbits` grid includes it), where the Interval transform is infinite; not a virgil fault. Fixed in `orbit/run_virgil.py` (chain starts derived from the priors for every orbit case and kept strictly inside each support, `orbit/README.md` "Chain starts") and checked locally as far as initialisation: every case has finite starts, log density and gradient. The two r2 rows are stale errors until the case is rerun on the node. Octofitter ran the case (matched t1 warm 1.53 s, fresh 61.5 s; native t2 warm 2.50 s, fresh 69.0 s).
- Imaging, time to precision (seconds to J within 10 % of the span to J_ref / including the fresh overhead): I-sim-24 virgil L-BFGS-B 0.086 / 5.5, virgil LM 0.17 / 7.0, eht-imaging 0.25 / 2.05; I-sim-48 0.667 / 6.0, 0.72 / 7.5, 3.07 / 4.9; I-sim-96 4.44 / 9.6 (L-BFGS-B), eht-imaging 20.3 / 22.1. eht-imaging does not reach the 3 % rung on any target within the budget; virgil LM reaches 0.1 % on I-sim-24 (3.4 s) and I-sim-48 (23 s); virgil L-BFGS-B reaches 0.1 % on I-sim-96 (16.7 s). Cold overhead: virgil 5.1-6.9 s, eht-imaging 1.8 s, so for a single short run from a fresh process eht-imaging is ahead on I-sim-24 and I-sim-48 at 10 % only when the overhead is counted (2.05 s vs 5.5 s and 4.9 s vs 6.0 s).
- Which code was faster (groups with both times): warm virgil 30, other 8; first-call cold virgil 14, other 22; fresh process virgil 24, other 17.
- virgil chi2 differs from the neutral reference (closure-phase whitening, 4 telescopes): flag `likelihood_mismatch` on 22 r2 groups (107 rows; binary T2/T3 and orbit Level B); flagged, not gated.
- Threads: 199 `threads_not_held` rows (virgil rows: cpu/wall above 1.2 at 1 thread; orbit virgil rows measured 1.09-1.42 cores). `thread_probe.py` was not run, so the XLA hold flags are not confirmed to bind on this node; both runs of every flagged row were kept. Rows from threads-held groups only: binary warm geo-mean 143 (7 groups), orbit warm 1.25 (3), imaging warm 0.93 (2).
- Timeouts (3 rows): PMOIRED B-sim-60 T3 matched 2 threads (1 of 3 processes), virgil I-sim-96 ladder default-method (1 process), Octofitter O-B-sim native 1 thread (1 process); the other processes of those groups are kept.

### Not run (status not_run, reason budget; 40 matrix entries, 64 groups)

The 13 h budget ended before: binary T2 `tight` and T3 `matched-coarse` (all cases), imaging `stop_tight` (I-sim-24/48/96/ring), the I-sim-ring ladder rows and `native` 2-thread, all I-sim-large rows, orbit O-A-real native 1 thread. Rows not measured in r2 are `not_run`; the r1 laptop values for them are not used.

### Limits

One 64-core EPYC 7543 node, 1 thread primary, 2 threads where listed (nothing on all cores); no GPU row; eht-imaging with the direct transform (no NFFT); Octofitter interferometry on simulated point-source data only; O-A-real period-informed and not scored; JAX/XLA thread hold not confirmed (see above); Python codes inside the virgil apptainer image (JAX 0.12.0.1); PMOIRED and Octofitter parallel modes not beyond 2 workers; imaging precision targets differ from the first digest (time-to-precision rungs).

### Consistency flag: main.tex l.157

l.157 states that virgil outperforms all other packages in speed, crediting JAX and GPU acceleration. On this machine the statement does not hold unqualified: Octofitter is faster warm on all three matched orbit groups (ratios 0.47-0.89); eht-imaging is faster per call and, counting the 5-7 s start-up, ahead at the 10 % precision rung on I-sim-24 and I-sim-48 for a single fresh run; PMOIRED is faster on a fresh process for binary T2 (ratio 0.74-0.75) and on the first call for T1/T2 (compile time); first-call (cold) ratios favour the other code in 22 of 36 paired groups. It does hold, with large margins, for warm binary fits (20 of 20), PMOIRED T3, and imaging time to a tight precision (eht-imaging does not reach 3 % in the budget). The GPU clause has no measured row (CPU only).

### Files of the node run

Outputs: `results/results.csv` (one row per case x task x config x threads x code), `results/raw/*.json` (every process; r1 laptop rows keep their names, r2 node rows end `__r2.json`), `results/performance_table.tex`, `results/performance.pdf` / `.png`, `results/summary.json`, `results/cluster-09b2685+vv5a91ffd/` (node report `node.txt`, `matrix_log_r2.jsonl`, `DONE_r2`). Rebuild with `python summarise.py`; `python summarise.py --check` validates schema and coverage. The run is the OzSTAR stage `bench275` (jobs repo `scripts/bench275`): one 64-CPU EPYC 7543 node, serial `matrix.py --run --run-id r2 --budget-h 13` (about 13 h), results tarred per task. Submissions 21371877 and 21372673 (wrong node type, stopped by the stage guard) and the earlier set-up failures contributed no rows.

### Review follow-up (after the node run)

* I-sim-large gate fails on the virgil side (virgil loss vs the neutral objective, worst relative difference 1.85e-2 at the truth; eht-imaging agrees to 1e-14): virgil whitens the correlated 4-station closure phases.
* Fixed after review: imaging native virgil rows were scored on a pixel scale 1.5 % too small (key bug in `imaging/compare.py`; timings unaffected, accuracy columns need a rerun); orbit O-A-real start period on the prior bound (see Findings; rows need a rerun); `not_run` rows of the virgil default-method route were labelled `ehtim`.
* Open (rerun or design decision needed): Octofitter Level A carries an unused `flux_H`; Octofitter chains are sequential calls; eht-imaging `flux_weight`; no independent gate on virgil's binary and Level B likelihoods; B-real-pionier T3 matched virgil is 259 sigma from the reference; label-swapped PMOIRED optimum on B-sim-60 T3.

## Reading the O-A-sim-pulls row

`t_init` and `t_cold` of O-A-sim-pulls are **sums over the 20 fits** (one per noise seed, in one process); `t_warm` is **one fit** (seed 00).
Per fit, divide `t_init` and `t_cold` by `n_datasets` (rows record `iterations.t_init_per_fit` and `t_cold_per_fit`). In run r2 the 144 s
virgil `t_cold` was 20 fits, 7.2 s per fit against 3.9 s warm: not a 140 s compile. Level A now passes the data as a model argument and
reuses one `MCMC(jit_model_args=True)` across the seeds, so the compile is paid once.

Octofitter's global optimisation and Pathfinder are paid in `t_init` and their payoff (starts near the mode, a dense metric from the
draws) in `t_cold` and `t_warm`: compare warm ratios together with `t_init`, not alone.
