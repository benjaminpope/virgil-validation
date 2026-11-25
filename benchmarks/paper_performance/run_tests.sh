#!/usr/bin/env bash
# Runs every benchmark test directory in its own pytest process: binary/, imaging/ and orbit/ each have
# make_data.py / compare.py / run_virgil.py modules of the same name, so one combined collection collides.
# Usage (from benchmarks/paper_performance, after `source ~/.cache/virgil-bench/activate.sh`):
#   ~/.claude/bin/pyslot bash run_tests.sh
set -u
rc=0
for t in tests binary imaging orbit; do
  echo "== $t"
  "${BENCH_PYTHON:?source ~/.cache/virgil-bench/activate.sh first}" -m pytest "$t" || rc=1
done
exit $rc
