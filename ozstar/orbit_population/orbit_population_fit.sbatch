#!/bin/bash
# virgil orbit population gate (design/orbit_gate_042.md): one task per system, strata A, B, N, D in
# turn (200 + 240 + 100 + 120 = 660 tasks). scripts/orbit_population.py fit --task N; a finished
# system is not rerun, so a task stopped by its time limit is resubmitted with --array=N.
#SBATCH --job-name=orbit-pop-fit
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=04:00:00          # not measured on OzSTAR: the first run records seconds and peak_rss_mb per system; adjust from them
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4        # OrbitPipeline's four vectorized NUTS chains
#SBATCH --mem=16G                # not measured: see peak_rss_mb in the task files
#SBATCH --array=0-659
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
export XLA_FLAGS="--xla_force_host_platform_device_count=4${XLA_FLAGS:+ $XLA_FLAGS}"

echo "job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} host=$(hostname) pins=\"$PIN_DESC\" start=$(date)"
start=$(date +%s)
srun python "$VV/scripts/orbit_population.py" fit --task "$SLURM_ARRAY_TASK_ID" --out "$OUT"
echo "task=${SLURM_ARRAY_TASK_ID} elapsed=$(( $(date +%s) - start ))s end=$(date)"
