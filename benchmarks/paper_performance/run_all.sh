#!/usr/bin/env bash
# Serial measured matrix (WP-R). Run only on an idle machine on AC power, in the background:
#   source ~/.cache/virgil-bench/activate.sh
#   cd benchmarks/paper_performance && ./run_all.sh            # writes results/raw/*.json and results/DONE
#   ./run_all.sh --dry-run                                     # list rows, estimates and the 7 h cap; runs nothing
# Refuses to start unless on AC power, Low Power Mode is off and pyslot has no other jobs. Runs under caffeinate.
set -euo pipefail
cd "$(dirname "$0")"
: "${BENCH_PYTHON:?source ~/.cache/virgil-bench/activate.sh first}"
if [[ "${1:-}" == "--dry-run" ]]; then exec "$BENCH_PYTHON" matrix.py --dry-run; fi
pmset -g batt | head -1 | grep -q "AC Power" || { echo "refusing: not on AC power" >&2; exit 1; }
lpm=$(pmset -g | awk '/lowpowermode/ {print $2}'); [[ "${lpm:-0}" == "0" ]] || { echo "refusing: Low Power Mode on" >&2; exit 1; }
if ~/.claude/bin/pyslot --status 2>&1 | grep -q "pid="; then
  echo "refusing: pyslot is not idle" >&2; ~/.claude/bin/pyslot --status >&2; exit 1
fi
mkdir -p results/raw
rm -f results/DONE
exec caffeinate -i -s "$BENCH_PYTHON" matrix.py --run
