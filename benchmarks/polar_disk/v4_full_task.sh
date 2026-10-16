#!/bin/bash
# One V4 full-run array task: maps SLURM_ARRAY_TASK_ID 0..5 to (instrument, degree) and runs it.
#   sbatch --array=0-5 ... benchmarks/polar_disk/v4_full_task.sh
# Optional: SEED_OFFSET (default 0) for replicate arrays; PD_OUT for the output directory;
# PY for the interpreter (default .venv/bin/python). Run from the repository root.
set -euo pipefail
t=${SLURM_ARRAY_TASK_ID:?set SLURM_ARRAY_TASK_ID (0..5)}
inst=(nrm nrm nrm vlti vlti vlti)
deg=(1 2 3 1 2 3)
[ "$t" -ge 0 ] && [ "$t" -le 5 ] || { echo "task id must be 0..5, got $t" >&2; exit 2; }
exec "${PY:-.venv/bin/python}" benchmarks/polar_disk/v4_injection_recovery.py --config full \
  --instrument "${inst[$t]}" --degree "${deg[$t]}" --seed-offset "${SEED_OFFSET:-0}"
