#!/usr/bin/env bash
# Build the shared benchmark environment ~/.cache/virgil-bench/venv (py3.12) used by every work package.
# Usage: benchmarks/paper_performance/env/setup_bench.sh   (run from any worktree of virgil-validation)
# virgil is installed at VIRGIL_REF, read from the worktree's .github/workflows/tests.yml (never hard-coded).
# Activate:  source ~/.cache/virgil-bench/activate.sh   (sets BENCH_PYTHON, PYTHONPATH, EHTIM_PYTHON)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
BENCH="${VIRGIL_BENCH_HOME:-$HOME/.cache/virgil-bench}"
VENV="$BENCH/venv"
MIN_GB=6

free_gb=$(df -g / | awk 'NR==2{print $4}')
if [ "$free_gb" -lt "$MIN_GB" ]; then echo "free disk ${free_gb} GB < ${MIN_GB} GB: stopping" >&2; exit 3; fi

REF=$(grep -E '^\s*VIRGIL_REF:' "$WT/.github/workflows/tests.yml" | head -1 | awk '{print $2}')
[ -n "$REF" ] || { echo "VIRGIL_REF not found in tests.yml" >&2; exit 4; }
echo "VIRGIL_REF=$REF"

mkdir -p "$BENCH"
[ -d "$VENV" ] || uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" \
  "virgil-astro[orbits,pipeline] @ git+https://github.com/benjaminpope/virgil@$REF" \
  "pmoired==26.10.1" arviz numpyro astropy scipy pyyaml pytest

echo "$REF" > "$BENCH/VIRGIL_REF"
EHTIM="$HOME/code/virgil-validation/.venv-ehtim/bin/python"
cat > "$BENCH/activate.sh" <<ACT
# source this file; other work packages use it instead of building their own environment
export BENCH_PYTHON="$VENV/bin/python"
export EHTIM_PYTHON="\${EHTIM_PYTHON:-$EHTIM}"
export VIRGIL_BENCH_HOME="$BENCH"
# vv sources come from the worktree you run in (set VV_SRC to override)
export PYTHONPATH="\${VV_SRC:-\$(git rev-parse --show-toplevel)/src}\${PYTHONPATH:+:\$PYTHONPATH}"
ACT

uv pip freeze --python "$VENV/bin/python" > "$HERE/requirements-bench.lock"
[ -x "$EHTIM" ] && echo "ehtim env present: $EHTIM" || echo "WARNING: $EHTIM missing; rebuild with scripts/setup_external.sh ehtim" >&2
echo "done; free disk after: $(df -g / | awk 'NR==2{print $4}') GB"
