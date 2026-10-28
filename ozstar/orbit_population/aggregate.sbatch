#!/bin/bash
# Aggregate the orbit population gate under the registered criteria (design/orbit_gate_042.md
# section 6): scripts/orbit_population.py aggregate over out/<pin>/ (fit and oracle tasks).
# Submitted by run_all.sh after both arrays (afterany), with the same --ref and
# --ref-virgil-validation. Writes out/summary_<pin>.json, outside out/<pin>/.
#SBATCH --job-name=orbit-pop-agg
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:30:00          # not measured: reads 900 small JSON files, numpy and scipy
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
set -eo pipefail  # not -u: lmod and conda read unset variables

: "${MYFRED:=/fred/${OZ_PROJECT:-oz440}/$USER}"
: "${PIN_COMMIT:?submit with run_all.sh / bin/oz submit orbit_population --sbatch=aggregate.sbatch (submit.sh)}"
JOB_DIR=$SLURM_SUBMIT_DIR
OUT=$JOB_DIR/out/$PIN_TAG
VV=$(dirname "${PIN_PATH##*:}")
[[ -d $OUT ]] || { echo "no $OUT: nothing ran with these commits" >&2; exit 1; }

module -q load conda
conda activate virgil
export PYTHONPATH=$PIN_PATH${PYTHONPATH:+:$PYTHONPATH}
export JAX_PLATFORMS=cpu MPLBACKEND=Agg

echo "job=${SLURM_JOB_ID} host=$(hostname) pins=\"$PIN_DESC\" files=$(ls "$OUT"/fit_*.json "$OUT"/oracle_*.json 2>/dev/null | wc -l || true)/900 start=$(date)"
tmp=$(mktemp -d "${JOBFS:-${TMPDIR:-/tmp}}/agg.XXXXXX")
python "$VV/scripts/orbit_population.py" aggregate "$OUT" --summary "$tmp/summary.json"
cp "$tmp/summary.json" "$JOB_DIR/out/summary_$PIN_TAG.json.part.$SLURM_JOB_ID"
mv -f "$JOB_DIR/out/summary_$PIN_TAG.json.part.$SLURM_JOB_ID" "$JOB_DIR/out/summary_$PIN_TAG.json"
echo "summary=$JOB_DIR/out/summary_$PIN_TAG.json end=$(date)"
