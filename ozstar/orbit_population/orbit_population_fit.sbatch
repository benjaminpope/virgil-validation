#!/bin/bash
# virgil orbit population gate (design/orbit_gate_042.md): one task per system, strata A, B, N, D in
# turn (200 + 240 + 100 + 120 = 660 tasks). scripts/orbit_population.py fit --task N; a finished
# system (a failed one too: a failure stays a failure) is not rerun; a task stopped by its time limit wrote no file and is resubmitted with --array=N.
#SBATCH --job-name=orbit-pop-fit
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=04:00:00          # not measured on OzSTAR: the first run records seconds and peak_rss_mb per system; adjust from them
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1              # gate section 8: GPU nodes (CPU JAX jobs abort with LLVM allocation errors)
#SBATCH --cpus-per-task=4        # OrbitPipeline's four vectorized NUTS chains
#SBATCH --mem=16G                # not measured: see peak_rss_mb in the task files
#SBATCH --array=0-659%8           # at most 8 GPUs at once
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
export JAX_COMPILATION_CACHE_DIR=$MYFRED/.cache/jax MPLBACKEND=Agg

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "no GPU on $(hostname)" >&2; exit 1; }
echo "job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} host=$(hostname) pins=\"$PIN_DESC\" start=$(date)"
start=$(date +%s)
srun python "$VV/scripts/orbit_population.py" fit --task "$SLURM_ARRAY_TASK_ID" --out "$OUT"
echo "task=${SLURM_ARRAY_TASK_ID} elapsed=$(( $(date +%s) - start ))s end=$(date)"
