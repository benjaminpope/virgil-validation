#!/usr/bin/env bash
# The evidence the published Trust page is built from: the artefacts of the
# newest finished validation run on main that has an evidence artefact (fast
# tests, on the pinned virgil) and of the newest such weekly or manual run
# (every test, on virgil's main, with virgil's own CI evidence). A run that
# was cancelled or timed out uploads nothing; it is skipped, with a line on
# stderr, for the next older of the last 10 runs. Evidence measured on a
# laptop is not published.
#
#   scripts/fetch_ci_evidence.sh [--push-only] OUTDIR
#
# With --push-only the weekly or manual run is neither looked up nor fetched
# (so a failed one cannot abort the script) and only OUTDIR/push is written.
#
# writes OUTDIR/push/evidence.jsonl, OUTDIR/full/evidence.jsonl and, when
# the full run fetched it, OUTDIR/full/virgil-evidence.jsonl. Needs gh with
# actions: read on this repository.
set -euo pipefail
push_only=0
if [ "${1:-}" = "--push-only" ]; then push_only=1; shift; fi
out=${1:?usage: fetch_ci_evidence.sh OUTDIR}
repo=${GITHUB_REPOSITORY:-benjaminpope/virgil-validation}

runs=${EVIDENCE_RUNS:-10}  # how many finished runs per event to look through

candidates() {  # ids of the last finished runs of tests.yml on main for the given events, newest first
  for event in "$@"; do
    gh run list -R "$repo" -w tests.yml -b main -e "$event" -s completed -L "$runs" \
      --json databaseId,createdAt --jq '.[] | "\(.createdAt) \(.databaseId)"'
  done | sort -r | sed -n "1,${runs}p" | cut -d' ' -f2
}

fetch() {  # fetch RUN's evidence artefact into DIR; 1 if it has none
  local run=$1 dir=$2 name
  name=$(gh api "repos/$repo/actions/runs/$run/artifacts" --jq '.artifacts[] | select(.name | startswith("evidence-")) | .name' | head -n 1)
  [ -n "$name" ] || return 1
  gh run download "$run" -R "$repo" -n "$name" -D "$dir"
  echo "$dir: $name (https://github.com/$repo/actions/runs/$run)"
}

fetch_newest() {  # fetch_newest DIR EVENT...: the newest run with an evidence artefact; 2 if there is no run at all
  local dir=$1 run ids
  shift
  ids=$(candidates "$@")
  [ -n "$ids" ] || return 2
  for run in $ids; do
    if fetch "$run" "$dir"; then return 0; fi
    echo "run $run has no evidence artefact; using an older run" >&2
  done
  echo "none of the last $runs finished runs ($*) has an evidence artefact" >&2
  return 1
}

rc=0; fetch_newest "$out/push" push || rc=$?
[ "$rc" != 2 ] || { echo "no finished validation run on main" >&2; exit 1; }
[ "$rc" = 0 ] || exit 1
[ "$push_only" = 1 ] && exit 0
rc=0; fetch_newest "$out/full" schedule workflow_dispatch || rc=$?
case $rc in
  0) ;;
  2) echo "no weekly or manual run yet" >&2 ;;
  *) exit 1 ;;
esac
