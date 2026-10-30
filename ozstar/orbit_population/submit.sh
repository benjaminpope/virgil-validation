#!/bin/bash
# Runs on an NT login node; `bin/oz submit <job>` runs it instead of sbatch.
# Pins this submission to fixed commits of the libraries checked out at
# $MYFRED/code/<lib>, so that a branch switch there, or another submission,
# cannot change the code under tasks that are queued or running.
#
#   bash submit.sh [--ref=REF] [--ref-<lib>=REF ...] [--var NAME=VALUE ...]
#                  [--sbatch=FILE] [--force] [sbatch options, e.g. --array=2]
# Any other option goes to sbatch as it is: --array=2, --dependency=afterok:ID (what
# `bin/oz submit <job> --after=ID` sends; the pipeline stages need no code here).
#
#   --ref=REF          branch, tag or commit of the first library in LIBS (default main)
#   --ref-<lib>=REF    the same for any library, e.g. --ref-virgil-validation=1a2b3c4
#   --var NAME=VALUE   passed to the job's environment (sbatch --export); no commas.
#                      e.g. --var SEED_BASE=1000 --var HALO=1
#   --sbatch=FILE      submit another sbatch file of this job (default SBATCH_FILE)
#   --force            submit even if the same tasks, commits and vars are already
#                      queued or running, or the import check fails
#
# Each REF is resolved to a commit after a fetch, and that commit is unpacked
# once into $MYFRED/code/<lib>-<SNAPSHOT_KIND>/<commit>/ (read-only, shared by
# later submissions of the same commit). The job gets
#   PIN_COMMIT, PIN_SRC, PIN_ROOT  the first library's commit, import dir and snapshot
#   PIN_PATH                       every library's import dir, for PYTHONPATH
#   PIN_DESC                       "virgil@1a2b3c4 virgil-validation@5d6e7f8", for logs
#   PIN_TAG                        "1a2b3c4" or "1a2b3c4+5d6e7f8": every commit, for out/<PIN_TAG>/
# Every submission is recorded in logs/submissions.tsv (bin/oz status reads it).
# Delete this file (and the pin block in the sbatch) if the job imports no
# library from a git checkout.
set -eo pipefail   # not -u: see the template sbatch

LIBS="virgil virgil-validation"   # --ref pins virgil (PIN_COMMIT, which the gate run records); --ref-virgil-validation pins the campaign code
SBATCH_FILE=orbit_population_fit.sbatch   # also orbit_population_oracle.sbatch and aggregate.sbatch (--sbatch=FILE)
PATHS="src scripts design pyproject.toml" # virgil-validation's scripts/orbit_population.py, design/orbit_gate_042.md and src/crosscheck; virgil's code
IMPORT_DIR=src             # the snapshot directory that goes on PYTHONPATH
SNAPSHOT_KIND=detection-snapshots   # not the code-only snapshots, which lack scripts/ and design/
ENV="virgil"               # conda env the job activates, for the import check
CHECK_IMPORT="virgil crosscheck yaml scipy"   # modules to import before submitting
# The campaign records the virgil commit in every file and the aggregator refuses to mix
# commits, and the fit, oracle and aggregate submissions must give the same PIN_TAG: both
# commits are required in full, so no branch can move between the three submissions.
job_finish_args() {
  [[ ${refs[0]} =~ ^[0-9a-f]{40}$ && ${refs[1]} =~ ^[0-9a-f]{40}$ ]] ||
    { echo "--ref=<40-hex virgil commit> and --ref-virgil-validation=<40-hex commit> are both required (got '${refs[0]}', '${refs[1]}')" >&2; exit 1; }
}
# Optional job-specific options: define job_option, which returns 0 if it used $1
# (it may add to vars or opts), and job_finish_args, run after all options are read.
# _notebook_template/submit.sh has an example. Below the next line, keep this file
# identical to scripts/_template/submit.sh.
# ------------------------------------------------------------------------------

[[ $LIBS != "<lib>" && $ENV != "<env>" && $CHECK_IMPORT != "<module>" ]] ||
  { echo "set LIBS, ENV and CHECK_IMPORT in submit.sh" >&2; exit 1; }
: "${MYFRED:=/fred/${OZ_PROJECT:-oz440}/$USER}"
cd "$(dirname "$0")"

# ---- arguments
refs=()       # one per library in LIBS, same order
for lib in $LIBS; do refs+=(main); done
vars=()       # NAME=VALUE
opts=()       # passed to sbatch
force=
while (($#)); do
  a=$1; shift
  case $a in
    --ref=*) refs[0]=${a#--ref=} ;;
    --ref-*=*)
      name=${a#--ref-}; name=${name%%=*}; i=0; found=
      for lib in $LIBS; do [[ $lib == "$name" ]] && { refs[i]=${a#*=}; found=1; }; i=$((i + 1)); done
      [[ -n $found ]] || { echo "--ref-$name: not one of LIBS ($LIBS)" >&2; exit 1; } ;;
    --var) vars+=("${1:?--var needs NAME=VALUE}"); shift ;;
    --var=*) vars+=("${a#--var=}") ;;
    --sbatch=*) SBATCH_FILE=${a#--sbatch=} ;;
    --force) force=1 ;;
    *) if declare -F job_option > /dev/null && job_option "$a"; then :; else opts+=("$a"); fi ;;
  esac
done
if declare -F job_finish_args > /dev/null; then job_finish_args; fi
[[ -f $SBATCH_FILE ]] || { echo "no $SBATCH_FILE in $PWD" >&2; exit 1; }
for v in ${vars[@]+"${vars[@]}"}; do
  [[ $v =~ ^[A-Za-z_][A-Za-z0-9_]*= && $v != *,* ]] ||
    { echo "--var $v: want NAME=VALUE, with no commas" >&2; exit 1; }
done

# ---- resolve and unpack each library
commits=(); desc=; pin_path=; tag=
i=0
for lib in $LIBS; do
  repo=$MYFRED/code/$lib
  ref=${refs[i]}
  git -C "$repo" fetch -q origin
  commit=$(git -C "$repo" rev-parse --verify -q "origin/$ref^{commit}" ||
           git -C "$repo" rev-parse --verify -q "$ref^{commit}") ||
    { echo "$lib: unknown ref $ref" >&2; exit 1; }
  snap=$MYFRED/code/$lib-$SNAPSHOT_KIND/$commit
  if [[ ! -d $snap ]]; then
    mkdir -p "$(dirname "$snap")"
    tmp=$(mktemp -d "$(dirname "$snap")/.tmp.XXXXXX")
    # shellcheck disable=SC2086  # PATHS holds several paths on purpose
    git -C "$repo" archive "$commit" $PATHS | tar -x -C "$tmp"
    chmod -R a-w "$tmp"
    # Atomic: a snapshot directory is always complete. If another submission (of any job)
    # finished the same snapshot first, mv fails (the snapshot is read-only), so use theirs.
    if ! mv "$tmp" "$snap" 2> /dev/null; then
      chmod -R u+w "$tmp"; rm -rf "$tmp"
      [[ -d $snap ]] || { echo "$lib: could not create $snap" >&2; exit 1; }
    fi
  fi
  echo "$lib $ref = ${commit:0:7}"
  if ((i == 0)); then first_commit=$commit; first_snap=$snap; fi
  commits+=("$commit")
  desc="${desc:+$desc }$lib@${commit:0:7}"
  tag="${tag:+$tag+}${commit:0:7}"
  pin_path="${pin_path:+$pin_path:}$snap/$IMPORT_DIR"
  i=$((i + 1))
done

# ---- which array tasks this submission asks for: --array, else the sbatch's default
expand_array() {   # "0-3,7,8-12:2%4" -> one index per line; "-" for a job that isn't an array
  local spec=${1%%\%*} part a b step k parts
  [[ -n $spec ]] || { echo -; return; }
  IFS=, read -r -a parts <<< "$spec"
  for part in "${parts[@]}"; do
    step=1
    if [[ $part == *:* ]]; then step=${part#*:}; part=${part%%:*}; fi
    if [[ $part == *-* ]]; then a=${part%-*}; b=${part#*-}; else a=$part; b=$part; fi
    for ((k = a; k <= b; k += step)); do echo "$k"; done
  done
}
array=
for o in ${opts[@]+"${opts[@]}"}; do [[ $o == --array=* ]] && array=${o#--array=}; done
[[ -n $array ]] || array=$(sed -n 's/^#SBATCH[[:space:]]*--array=\([^[:space:]]*\).*/\1/p' "$SBATCH_FILE" | head -n 1)
asked=$(expand_array "$array" | sort -u)
var_desc=$(printf '%s\n' ${vars[@]+"${vars[@]}"} | sort | paste -sd ' ' -)
var_desc=${var_desc:--}   # "-" for none: the ledger is tab-separated, and read collapses empty fields

# ---- one submission of this job at a time, from the duplicate check to the ledger, so
# two at once (a Run button clicked twice) can't both pass the check. mkdir is atomic on
# Lustre, where flock may not be available.
mkdir -p logs
lock=logs/.submit.lock
for ((w = 0; ; w++)); do
  mkdir "$lock" 2> /dev/null && break
  if [[ -n $(find "$lock" -maxdepth 0 -mmin +15 2> /dev/null) ]]; then
    echo "removing a stale lock from $(cat "$lock/owner" 2> /dev/null || echo "an earlier submission")" >&2
    rm -rf "$lock"; continue
  fi
  ((w)) || echo "waiting for another submission of this job to finish..." >&2
  ((w < 120)) || { echo "gave up after 10 min; if no submission is running: rm -r $PWD/$lock" >&2; exit 1; }
  sleep 5
done
trap 'rm -rf "$lock"' EXIT
echo "$(hostname) pid $$ $(date +%T)" > "$lock/owner"

# ---- refuse to race an earlier submission that writes the same outputs: the same
# sbatch file, commits and vars, with array tasks in common that are still queued or running
ledger=logs/submissions.tsv
if [[ -f $ledger ]]; then
  while IFS=$'\t' read -r id l_sbatch _ l_pins l_vars _; do
    [[ $l_sbatch == "$SBATCH_FILE" && $l_pins == "$desc" && $l_vars == "$var_desc" ]] || continue
    live=$(squeue -h -r -j "$id" -o %K 2>/dev/null | sed 's/^N\/A$/-/' | sort -u) || true
    [[ -n $live ]] || continue
    common=$(comm -12 <(echo "$asked") <(echo "$live") | paste -sd , -)
    [[ -n $common ]] || continue
    if [[ -n $force ]]; then
      echo "warning: job $id is still queued or running tasks $common with the same code and vars (--force)" >&2
    else
      echo "refusing: job $id is still queued or running tasks $common of $SBATCH_FILE with $desc${vars[0]+ and $var_desc}." >&2
      echo "Both would write the same outputs. Cancel it first (scancel $id), submit other tasks (--array=...)," >&2
      echo "or add --force to run both." >&2
      exit 1
    fi
  done < "$ledger"
fi

# Jobs submitted some other way (or before submissions.tsv existed) aren't checked; say they're there
name=$(sed -n 's/^#SBATCH[[:space:]]*--job-name=\([^[:space:]]*\).*/\1/p' "$SBATCH_FILE" | head -n 1)
if [[ -n $name ]]; then
  others=$(squeue --me -h -n "$name" -o %i 2>/dev/null | sed 's/_.*//' | sort -u | paste -sd ' ' -) || true
  [[ -z $others ]] || echo "note: already queued or running as $name: $others"
fi

# ---- the env must import the pinned code (a dependency bump on the ref fails here, not in the job)
py=$MYFRED/.conda/envs/$ENV/bin/python
if [[ -n $CHECK_IMPORT && -x $py ]]; then
  mods=$(echo "$CHECK_IMPORT" | tr ' ' ,)
  if ! out=$(JAX_PLATFORMS=cpu PYTHONPATH=$pin_path timeout 300 "$py" -c "import $mods" 2>&1); then
    echo "$out" | tail -n 5 >&2
    if [[ -n $force ]]; then
      echo "warning: the $ENV env can't import $CHECK_IMPORT at $desc (--force)" >&2
    else
      echo "refusing: the $ENV env can't import $CHECK_IMPORT at $desc; the ref probably needs newer" >&2
      echo "dependencies. Rebuild the env (setup_env.sh, with none of its jobs running), or add --force." >&2
      exit 1
    fi
  fi
elif [[ -n $CHECK_IMPORT ]]; then
  echo "warning: no $py; import check skipped" >&2
fi

# ---- submit
export_list=ALL,PIN_COMMIT=$first_commit,PIN_SRC=$first_snap/$IMPORT_DIR,PIN_ROOT=$first_snap
export_list+=,PIN_PATH=$pin_path,"PIN_DESC=$desc",PIN_TAG=$tag
for v in ${vars[@]+"${vars[@]}"}; do export_list+=,$v; done
id=$(sbatch --parsable --export="$export_list" ${opts[@]+"${opts[@]}"} "$SBATCH_FILE")
id=${id%%;*}
printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$SBATCH_FILE" "${array:--}" "$desc" "$var_desc" "$(date +%F_%T)" >> "$ledger"
echo "Submitted batch job $id: $SBATCH_FILE, tasks ${array:-(not an array)}, $desc${vars[0]+, $var_desc}"
echo "Logs in $PWD/logs; check with bin/oz status $(basename "$PWD")"
