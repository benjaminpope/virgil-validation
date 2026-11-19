#!/usr/bin/env bash
# Serial measured matrix (WP-R). Run only on an idle machine on AC power, in the background:
#   source ~/.cache/virgil-bench/activate.sh
#   cd benchmarks/paper_performance && ./run_all.sh            # writes results/raw/*.json and results/DONE
#   ./run_all.sh --dry-run                                     # list rows, estimates and the 7 h cap; runs nothing
#   ./run_all.sh --rerun [--dry-run]                           # PR 275 rerun (run r2: orbit, imaging, binary cold columns,
#                                                              # 4 h cap); writes results/DONE_r2. Same idle/AC/LPM checks.
# Refuses to start unless on AC power, Low Power Mode is off and pyslot has no other jobs. Runs under caffeinate.
set -euo pipefail
cd "$(dirname "$0")"
: "${BENCH_PYTHON:?source ~/.cache/virgil-bench/activate.sh first}"
for a in "$@"; do
  if [[ "$a" == "--dry-run" ]]; then exec "$BENCH_PYTHON" matrix.py --dry-run "$@"; fi
done
pmset -g batt | head -1 | grep -q "AC Power" || { echo "refusing: not on AC power" >&2; exit 1; }
lpm=$(pmset -g | awk '/lowpowermode/ {print $2}'); [[ "${lpm:-0}" == "0" ]] || { echo "refusing: Low Power Mode on" >&2; exit 1; }
if ~/.claude/bin/pyslot --status 2>&1 | grep -q "pid="; then
  echo "refusing: pyslot is not idle" >&2; ~/.claude/bin/pyslot --status >&2; exit 1
fi
mkdir -p results/raw
if [[ " $* " == *" --rerun "* ]]; then rm -f results/DONE_r2; else rm -f results/DONE; fi
exec caffeinate -i -s "$BENCH_PYTHON" matrix.py --run "$@"
