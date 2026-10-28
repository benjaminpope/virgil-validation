#!/bin/bash
# Run on the Mac, after `bin/oz push orbit_population`: submit the fit array, the oracle array and
# the aggregate (afterany both), all pinned to the same two commits.
#   run_all.sh <40-hex virgil commit> <40-hex virgil-validation commit>
# Each step is one `bin/oz submit`, which refuses a duplicate of a task still queued or running.
# Nothing here connects except bin/oz.
set -euo pipefail
OZ=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
VIRGIL=${1:?usage: run_all.sh <virgil commit> <virgil-validation commit>}
VV=${2:?usage: run_all.sh <virgil commit> <virgil-validation commit>}
refs=(--ref="$VIRGIL" --ref-virgil-validation="$VV")
cd "$OZ"
last() { cat "results/orbit_population/.last_submit"; }
bin/oz submit orbit_population "${refs[@]}" --sbatch=orbit_population_fit.sbatch
fit=$(last)
bin/oz submit orbit_population "${refs[@]}" --sbatch=orbit_population_oracle.sbatch
oracle=$(last)
bin/oz submit orbit_population "${refs[@]}" --sbatch=aggregate.sbatch --after-any="$fit,$oracle"
echo "queued: fit array $fit (660 tasks), oracle array $oracle (240 tasks), aggregate $(last) after both"
