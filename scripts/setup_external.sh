#!/usr/bin/env bash
# Install the external packages that live in their own environments
# (.venv-<name>), each at a pinned version: eht-imaging (GPL-3), MPoL (MIT),
# fouriever (no licence stated) and orbitize! (BSD-3-Clause; a C extension
# and a long dependency list: astropy, h5py, ptemcee, rebound, dynesty...). They are called through
# src/external_bridge/_subprocess.py, never imported here or vendored.
# CASA (casatools, BSD-3-Clause-like NRAO licence) for ALMA measurement sets: `alma`, in
# .venv-alma on Python 3.11 (casatools wheels), with measures data from the casadata
# package inside the venv (casasiteconfig.py, found through CASASITECONFIG by alma_worker.py).
# CANDID (no PyPI release) has scripts/setup_candid.sh.
#   bash scripts/setup_external.sh [ehtim] [mpol] [fouriever] [orbitize] [alma]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
pin() {
  case "$1" in
    ehtim) echo "ehtim==1.3.2" ;;
    mpol) echo "mpol==0.3.1" ;;
    fouriever) echo "fouriever==0.4.3" ;;
    orbitize) echo "orbitize==3.4.0" ;;
    alma) echo "casatools==6.7.0.31 casadata==2025.9.22 casaconfig==1.5.2 visread==0.0.5 numpy" ;;
    *) echo "unknown package: $1" >&2; exit 1 ;;
  esac
}
[ $# -eq 0 ] && set -- ehtim mpol fouriever orbitize alma
for name in "$@"; do
  spec=$(pin "$name")
  py=3.12
  [ "$name" = alma ] && py=3.11
  uv venv -q --allow-existing --python "$py" "$ROOT/.venv-$name"
  if [ "$name" = mpol ] && [ "$(uname)" = Linux ]; then
    # CPU PyTorch: the default Linux wheel brings gigabytes of CUDA
    uv pip install -q --python "$ROOT/.venv-$name/bin/python" torch --index-url https://download.pytorch.org/whl/cpu
  fi
  # shellcheck disable=SC2086
  uv pip install -q --python "$ROOT/.venv-$name/bin/python" $spec
  if [ "$name" = alma ]; then
    site=$("$ROOT/.venv-alma/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
    cat > "$ROOT/.venv-alma/casasiteconfig.py" <<EOF
import os
_d = "$site/casadata/__data__"
measurespath = _d
datapath = [_d]
measures_auto_update = False
data_auto_update = False
nologfile = True
logfile = os.devnull
EOF
  fi
  echo "$spec in $ROOT/.venv-$name"
done
