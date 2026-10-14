# Orbit solver v2 prototypes

Scratch-quality scripts behind the numbers in [§9 of the design](../orbit_solver_v2.md). They are not the
solver: they exist so the design's measurements can be rerun, here or on OzSTAR (ozstar_scripts job
`orbit_v2_proto`, which runs `run_task.sh` once per line of `tasks.txt`).

| File | Stage | What |
|---|---|---|
| `exp1_peaks.py` | S1 | per-night K-peak catalogues with covariances (`virgil.epochs.epoch_positions`) |
| `exp4_neff.py` | f | effective-channel fraction per night from the wavelength structure of residuals |
| `exp2_screen.py` | S2 | the combinatorial Thiele-Innes screen over (P, e, tau) |
| `exp3_lm.py` | S3-S4 | direct-to-data scoring, vmapped LM multistart, GN against full Hessian |
| `inject.py` | S7 | residual-transplant injection at the real (u, v, t); flux 0 gives the null |
| `common_data.py` | | the Gl 229 nights (`ESO_BIN_ROOT`), or an injection (`INJ`); closure phases only with `CPONLY=1` |
| `fastcvis.py` | | vectorised complex visibilities for orbital binaries (monkeypatch) |
| `run_task.sh`, `tasks.txt`, `make_tasks.py` | | one run per line: real data or injection, blind or 10-14 d window |
| `*.json` | | the laptop outputs the runs start from: the Gl 229 S1 catalogue (K = 25), its f table, and the 10-14 d screen and LM (whose best mode `inject.py` subtracts) |

Scripts read and write in `$PROTO_WORK` (default: this directory). Local smoke test, capped to 2 threads:

    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" OMP_NUM_THREADS=2 \
      SMOKE=1 PY=~/code/virgil/.venv/bin/python bash run_task.sh 1 /tmp/smoke
