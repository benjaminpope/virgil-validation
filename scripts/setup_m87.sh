#!/usr/bin/env bash
# Lai et al. 2026 (Nature 659, 317) M87* replication: fetch the public inputs and build
# their tools' environment. Plan and pass/fail criteria: design/m87_replication.md.
#
#   bash scripts/setup_m87.sh data    # Zenodo archives + EHT GitHub data products into $M87_DATA
#   bash scripts/setup_m87.sh basis   # .venv-basis: BASIS 0.1.0 + ClosureInvariants 1.0.0 + eht-imaging
#
# Data go to $M87_DATA (default ~/data/m87_eht), never into the repo. Every archive is
# checked against the MD5 Zenodo publishes; the EHT repositories are cloned at fixed commits.
# All three Zenodo records are MIT-licensed; the EHT data products carry the EHT
# Collaboration's terms (DATA_ATTRIBUTION.md). Nothing here is vendored: the venv installs
# BASIS and ClosureInvariants from the unpacked archives, and they run only through
# src/external_bridge/ workers.
#
# EHT 2026-D01-01 (2018 + 2021 polarimetric, CyVerse) sits behind a browser challenge and is
# not fetched here; the 2021-04-18 fiducial UVFITS comes with the GenDIReCT archive.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATA=${M87_DATA:-$HOME/data/m87_eht}

zenodo() {  # record key md5 dir
  local url="https://zenodo.org/api/records/$1/files/$2/content" out="$DATA/zenodo/$4/$(basename "$2")"
  mkdir -p "$(dirname "$out")"
  [ -f "$out" ] || curl -sSfL -o "$out" "$url"
  [ "$(md5 -q "$out" 2>/dev/null || md5sum "$out" | cut -d' ' -f1)" = "$3" ] || { echo "MD5 mismatch: $out" >&2; exit 1; }
  mkdir -p "$DATA/zenodo/$4/x" && unzip -q -o "$out" -d "$DATA/zenodo/$4/x"
}

ehtrepo() {  # name commit
  local dir="$DATA/eht/$1"
  [ -d "$dir/.git" ] || git clone -q "https://github.com/eventhorizontelescope/$1" "$dir"
  git -C "$dir" fetch -q --depth 1 origin "$2" 2>/dev/null || true
  git -C "$dir" checkout -q "$2"
}

data() {
  zenodo 22764433 samlaihei/GenDIReCT-v1.1.0-paperM87.zip 764aae1d4ed17b24fdcef7b3c5883037 gendirect
  zenodo 21090230 samlaihei/BASIS-release_v0.1.0.zip 32aab34893130a467b15bedae3973c95 basis
  zenodo 15524510 nithyanandan/ClosureInvariants-v1.0.0.zip a2780931ea0e4ad18ad5ac7426bbecfe closureinvariants
  ehtrepo 2019-D01-01 27ae76aacb9067a83388395c3826012b97aac371
  ehtrepo 2024-D01-01 fda41da417419fd9b54505ea3fa70a8cf6fb60ec
  echo "data in $DATA"
}

basis() {
  local venv="$ROOT/.venv-basis"
  uv venv -q --allow-existing --python 3.11 "$venv"
  uv pip install -q --python "$venv/bin/python" "ehtim==1.3.2" "bilby==2.8.2" "dynesty==3.1.0" \
    torch opencv-python-headless uncertainties pandas tqdm matplotlib scipy "numpy<2.3"
  uv pip install -q --python "$venv/bin/python" --no-deps \
    "$DATA"/zenodo/closureinvariants/x/nithyanandan-ClosureInvariants-*/ \
    "$DATA"/zenodo/basis/x/samlaihei-BASIS-*/
  "$venv/bin/python" -c "import BASIS, ClosureInvariants, ehtim, bilby, dynesty; print('basis env ok: bilby', bilby.__version__, 'dynesty', dynesty.__version__)"
}

[ $# -eq 0 ] && set -- data basis
for step in "$@"; do "$step"; done
