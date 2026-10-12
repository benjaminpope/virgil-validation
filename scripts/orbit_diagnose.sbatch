#!/bin/bash
# scripts/orbit_diagnose.py on Gl 229 (task 0) and iota Peg (task 1): the published orbit
# against the committed MAP on the same data and loss. Submit from a checkout of this
# branch on OzSTAR, with the same virgil (3420ea1) the O1 fits used:
#   VIRGIL_SRC=<path to virgil checkout at 3420ea1> sbatch scripts/orbit_diagnose.sbatch
#SBATCH --job-name=orbit_diagnose
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-1
set -eo pipefail
: "${MYFRED:=/fred/${OZ_PROJECT:-oz440}/$USER}"
: "${VIRGIL_SRC:?path to virgil at 3420ea1 (its src/ is put on PYTHONPATH)}"
: "${FITS_DIR:?directory holding the committed fit_<id>.json (e.g. the O1 out/<pin>/ dir)}"
mkdir -p logs out
NAMES=(gl229 iotapeg)
IDS=(782185b2-0727-42b0-a185-b2072732b047 fac164e1-d9d0-4500-8164-e1d9d0450099)
T=${SLURM_ARRAY_TASK_ID:-0}
module -q load conda
conda activate virgil
export PYTHONPATH=$VIRGIL_SRC/src${PYTHONPATH:+:$PYTHONPATH} JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR=$MYFRED/.cache/jax
python -c "import virgil; print('virgil from', virgil.__file__)"
srun python scripts/orbit_diagnose.py "${NAMES[$T]}" "$MYFRED/oidb/${IDS[$T]}" \
  "$FITS_DIR/fit_${IDS[$T]}.json" "out/diagnose_${NAMES[$T]}.json"
