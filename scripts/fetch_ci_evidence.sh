#!/usr/bin/env bash
# The evidence the published Trust page is built from: the artefacts of the
# newest finished validation run on main (fast tests, on the pinned virgil)
# and of the newest weekly or manual run (every test, on virgil's main, with
# virgil's own CI evidence). Evidence measured on a laptop is not published.
#
#   scripts/fetch_ci_evidence.sh OUTDIR
#
# writes OUTDIR/push/evidence.jsonl, OUTDIR/full/evidence.jsonl and, when
# the full run fetched it, OUTDIR/full/virgil-evidence.jsonl. Needs gh with
# actions: read on this repository.
set -euo pipefail
out=${1:?usage: fetch_ci_evidence.sh OUTDIR}
repo=${GITHUB_REPOSITORY:-benjaminpope/virgil-validation}

newest() {  # newest finished run of tests.yml on main for the given events
  for event in "$@"; do
    gh run list -R "$repo" -w tests.yml -b main -e "$event" -s completed -L 1 \
      --json databaseId,createdAt --jq '.[] | "\(.createdAt) \(.databaseId)"'
  done | sort | tail -n 1 | cut -d' ' -f2
}

fetch() {  # fetch RUN's evidence artefact into DIR
  local run=$1 dir=$2 name
  name=$(gh api "repos/$repo/actions/runs/$run/artifacts" --jq '.artifacts[] | select(.name | startswith("evidence-")) | .name' | head -n 1)
  [ -n "$name" ] || { echo "run $run has no evidence artefact" >&2; return 1; }
  gh run download "$run" -R "$repo" -n "$name" -D "$dir"
  echo "$dir: $name (https://github.com/$repo/actions/runs/$run)"
}

push=$(newest push)
full=$(newest schedule workflow_dispatch)
[ -n "$push" ] || { echo "no finished validation run on main" >&2; exit 1; }
fetch "$push" "$out/push"
if [ -n "$full" ]; then fetch "$full" "$out/full"; else echo "no weekly or manual run yet" >&2; fi
