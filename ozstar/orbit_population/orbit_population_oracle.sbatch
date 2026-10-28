#!/bin/bash
# virgil orbit population gate, oracle runs (init=truth) of the B systems, for M4 and M8: one task per B system.
# A finished system is not rerun, so a task stopped by its time limit is resubmitted with --array=N.
#SBATCH --job-name=orbit-pop-oracle
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=02:00:00          # not measured: the oracle stops after the modes stage, no NUTS
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-239
set -eo pipefail  # not -u: lmod and conda read unset variables

: "${MYFRED:=/fred/${OZ_PROJECT:-oz440}/$USER}"
: "${PIN_COMMIT:?submit with run_all.sh / bin/oz submit orbit_population (submit.sh), which pins virgil and virgil-validation}"
JOB_DIR=$SLURM_SUBMIT_DIR
OUT=$JOB_DIR/out/$PIN_TAG
VV=$(dirname "${PIN_PATH##*:}")
mkdir -p "$OUT"
[[ -f $VV/scripts/orbit_population.py ]] || { echo "missing $VV/scripts/orbit_population.py" >&2; exit 1; }

module -q load conda
conda activate virgil
export PYTHONPATH=$PIN_PATH${PYTHONPATH:+:$PYTHONPATH}
export JAX_PLATFORMS=cpu JAX_COMPILATION_CACHE_DIR=$MYFRED/.cache/jax MPLBACKEND=Agg

echo "job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} host=$(hostname) pins=\"$PIN_DESC\" start=$(date)"
start=$(date +%s)
srun python "$VV/scripts/orbit_population.py" oracle --task "$SLURM_ARRAY_TASK_ID" --out "$OUT"
echo "task=${SLURM_ARRAY_TASK_ID} elapsed=$(( $(date +%s) - start ))s end=$(date)"
