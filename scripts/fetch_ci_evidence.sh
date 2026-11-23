#!/usr/bin/env bash
# The evidence the published Trust page is built from: the artefacts of the
# newest finished validation run on main that has an evidence artefact (fast
# tests, on the pinned virgil) and of the newest such weekly or manual run
# (every test, on virgil's main, with virgil's own CI evidence). A run that
# was cancelled, timed out, lost its runner or crashed has no evidence.jsonl in its artefact, whatever its
# conclusion; it is skipped, with a line on stderr, for the next older of the last 10 runs. Evidence
# measured on a laptop is not published.
#
#   scripts/fetch_ci_evidence.sh [--push-only] OUTDIR
#
# With --push-only the weekly or manual run is neither looked up nor fetched
# (so a failed one cannot abort the script) and only OUTDIR/push is written.
#
# A weekly or manual run older than FULL_MAX_AGE_DAYS (default 8) is ignored, with one line on
# stderr naming it and its age, and no OUTDIR/full is written: such a run measured an older vv and
# virgil, and for tests moved to the weekly run it would be their only, stale, evidence. Those tests
# then show as unchecked.
#
# writes OUTDIR/push/evidence.jsonl, OUTDIR/full/evidence.jsonl and, when
# the full run fetched it, OUTDIR/full/virgil-evidence.jsonl; beside each,
# source.json (run_id, date, vv_commit, url) of the run chosen. Needs gh with
# actions: read on this repository.
set -euo pipefail
push_only=0
if [ "${1:-}" = "--push-only" ]; then push_only=1; shift; fi
out=${1:?usage: fetch_ci_evidence.sh OUTDIR}
repo=${GITHUB_REPOSITORY:-benjaminpope/virgil-validation}

runs=${EVIDENCE_RUNS:-10}  # how many finished runs per event to look through
full_max_age=${FULL_MAX_AGE_DAYS:-8}  # days; the weekly or manual run must be younger than this

# Everything below runs inside `if` or `||`, where set -e is off, so every gh call is checked by hand:
# a failed lookup or download stops the script (it never reads as "no run" or "no artefact").
candidates() {  # ids of the last finished runs of tests.yml on main for the given events, newest first
  local event out all=
  for event in "$@"; do
    out=$(gh run list -R "$repo" -w tests.yml -b main -e "$event" -s completed -L "$runs" \
      --json databaseId,createdAt,headSha --jq '.[] | "\(.createdAt) \(.databaseId) \(.headSha)"') || return 1
    all+=$out$'\n'
  done
  printf '%s' "$all" | sed '/^$/d' | sort -r | sed -n "1,${runs}p"  # "createdAt id headSha"
}

epoch() {  # seconds since 1970 of an ISO UTC timestamp (GNU or BSD date); fails on anything else
  date -u -d "$1" +%s 2>/dev/null || date -u -j -f %Y-%m-%dT%H:%M:%SZ "$1" +%s 2>/dev/null
}

fetch() {  # fetch RUN's evidence artefact into DIR; 1 if it has none (or no evidence.jsonl), 3 if gh failed
  local run=$1 dir=$2 name
  name=$(gh api "repos/$repo/actions/runs/$run/artifacts" \
    --jq 'first(.artifacts[] | select((.name | startswith("evidence-")) and (.expired | not)) | .name)') || return 3
  [ -n "$name" ] || return 1
  rm -rf "$dir"  # gh run download refuses to overwrite, and an older run's file must not survive
  gh run download "$run" -R "$repo" -n "$name" -D "$dir" || { echo "downloading $name of run $run failed" >&2; return 3; }
  # a cancelled run still uploads its artefact (mem.log, no evidence.jsonl): that is no evidence
  [ -f "$dir/evidence.jsonl" ] || { rm -rf "$dir"; return 1; }
  echo "$dir: $name (https://github.com/$repo/actions/runs/$run)"
}

fetch_newest() {  # fetch_newest DIR MAXDAYS EVENT...: the newest run with an evidence artefact, none MAXDAYS old
  # or more (0: no limit); 2 if there is no run, 4 if the run reached is too old, 3 if gh failed
  local dir=$1 max=$2 run created sha ids rc now age
  shift 2
  ids=$(candidates "$@") || { echo "listing the runs of ($*) failed" >&2; return 3; }
  [ -n "$ids" ] || return 2
  now=$(date +%s)
  while read -r created run sha; do
    if [ "$max" != 0 ]; then
      age=$(epoch "$created") || { echo "cannot read the date $created of run $run" >&2; return 3; }
      age=$(( (now - age) / 86400 ))
      if [ "$age" -ge "$max" ]; then  # newest first: every later run is older still
        rm -rf "$dir"
        echo "ignoring run $run ($*): $age days old, limit $max days (FULL_MAX_AGE_DAYS); no evidence taken from it" >&2
        return 4
      fi
    fi
    rc=0; fetch "$run" "$dir" || rc=$?
    case $rc in
      0) printf '{"run_id": %s, "date": "%s", "vv_commit": "%s", "url": "https://github.com/%s/actions/runs/%s"}\n' \
           "$run" "$created" "$sha" "$repo" "$run" > "$dir/source.json"
         return 0 ;;
      1) echo "run $run has no evidence artefact; using an older run" >&2 ;;
      *) echo "looking up the artefacts of run $run failed" >&2; return 3 ;;
    esac
  done <<< "$ids"
  echo "none of the last $runs finished runs ($*) has an evidence artefact" >&2
  return 1
}

rc=0; fetch_newest "$out/push" 0 push || rc=$?
[ "$rc" != 2 ] || { echo "no finished validation run on main" >&2; exit 1; }
[ "$rc" = 0 ] || exit 1
[ "$push_only" = 1 ] && exit 0
rc=0; fetch_newest "$out/full" "$full_max_age" schedule workflow_dispatch || rc=$?
case $rc in
  0) ;;
  4) ;;  # too old: said on stderr, nothing written
  2) echo "no weekly or manual run yet" >&2 ;;
  *) exit 1 ;;
esac
