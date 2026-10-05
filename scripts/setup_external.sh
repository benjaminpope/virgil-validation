#!/usr/bin/env bash
# Install the external packages that live in their own environments
# (.venv-<name>), each at a pinned version: eht-imaging (GPL-3), MPoL (MIT)
# and fouriever (no licence stated). They are called through
# src/external_bridge/_subprocess.py, never imported here or vendored.
# CANDID (no PyPI release) has scripts/setup_candid.sh.
#   bash scripts/setup_external.sh [ehtim] [mpol] [fouriever]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
pin() {
  case "$1" in
    ehtim) echo "ehtim==1.3.2" ;;
    mpol) echo "mpol==0.3.1" ;;
    fouriever) echo "fouriever==0.4.3" ;;
    *) echo "unknown package: $1" >&2; exit 1 ;;
  esac
}
[ $# -eq 0 ] && set -- ehtim mpol fouriever
for name in "$@"; do
  spec=$(pin "$name")
  uv venv -q --allow-existing --python 3.12 "$ROOT/.venv-$name"
  if [ "$name" = mpol ] && [ "$(uname)" = Linux ]; then
    # CPU PyTorch: the default Linux wheel brings gigabytes of CUDA
    uv pip install -q --python "$ROOT/.venv-$name/bin/python" torch --index-url https://download.pytorch.org/whl/cpu
  fi
  uv pip install -q --python "$ROOT/.venv-$name/bin/python" "$spec"
  echo "$spec in $ROOT/.venv-$name"
done
