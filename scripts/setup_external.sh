#!/usr/bin/env bash
# Install the external packages that live in their own environments
# (.venv-<name>), each at a pinned version: eht-imaging (GPL-3), MPoL (MIT),
# fouriever (no licence stated) and orbitize! (BSD-3-Clause; a C extension
# and a long dependency list: astropy, h5py, ptemcee, rebound, dynesty...). They are called through
# src/external_bridge/_subprocess.py, never imported here or vendored.
# CASA (casatools, BSD-3-Clause-like NRAO licence) for ALMA measurement sets: `alma`, in
# .venv-alma on Python 3.11 (casatools wheels), with measures data from the casadata
# package inside the venv (casasiteconfig.py, found through CASASITECONFIG by alma_worker.py).
# `mpol` also clones MPoL-dev/examples at a pinned commit into .external/mpol-examples (the IM Lup
# multi-EB example's own Net is what tests/test_alma_mpol.py calls); MPOL_EXAMPLES overrides the
# location in the tests, MPOL_PYTHON the interpreter.
# The `mpol` env also runs the example's own sgd.py and loaddata.py (OzSTAR reference run), so it
# carries torch (CUDA where the host has a GPU: MPOL_TORCH_INDEX overrides the wheel index), asdf,
# tensorboard, matplotlib and visread 0.0.5 (installed without its dependencies: the example's
# loaddata and sgd import only visread.process, .scatter and .visualization, none of which needs CASA).
# MPoL's export (casa_analysis/src/export_asdf.py) needs casatools 6.7.0.31 *with* visread, and that
# lives in the `alma` env, not `mpol`: casatools wheels pin the Python (3.11 here) and are 300 MB of
# CASA nobody needs for training. Run export_asdf.py with .venv-alma/bin/python (plus asdf, which `alma` installs).
# CANDID (no PyPI release) has scripts/setup_candid.sh.
#   bash scripts/setup_external.sh [ehtim] [mpol] [fouriever] [orbitize] [alma]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
pin() {
  case "$1" in
    ehtim) echo "ehtim==1.3.2" ;;
    mpol) echo "mpol==0.3.1 asdf tensorboard matplotlib" ;;
    fouriever) echo "fouriever==0.4.3" ;;
    orbitize) echo "orbitize==3.4.0" ;;
    alma) echo "casatools==6.7.0.31 casatasks==6.7.0.31 casadata==2025.9.22 casaconfig==1.5.2 visread==0.0.5 asdf numpy" ;;
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
    # MPOL_TORCH_INDEX: "cpu" (default without a GPU) or a CUDA tag such as cu126 (default with nvidia-smi).
    # The default Linux wheel brings gigabytes of CUDA libraries, so the CPU index is used on GPU-less hosts.
    tag=${MPOL_TORCH_INDEX:-$(command -v nvidia-smi >/dev/null 2>&1 && echo cu126 || echo cpu)}
    uv pip install -q --python "$ROOT/.venv-$name/bin/python" torch --index-url "https://download.pytorch.org/whl/$tag"
  fi
  # shellcheck disable=SC2086
  uv pip install -q --python "$ROOT/.venv-$name/bin/python" $spec
  if [ "$name" = mpol ]; then
    # visread's CASA dependency is only for reading measurement sets, which `alma` does
    uv pip install -q --python "$ROOT/.venv-$name/bin/python" --no-deps visread==0.0.5
  fi
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
  if [ "$name" = mpol ]; then
    ex="$ROOT/.external/mpol-examples"
    [ -d "$ex/.git" ] || git clone -q https://github.com/MPoL-dev/examples "$ex"
    git -C "$ex" checkout -q 5a7c3f0b373f62341ca4baf5869f95e916bbabb3
  fi
  echo "$spec in $ROOT/.venv-$name"
  # the resolved versions, for the record (the OzSTAR job copies this into its output)
  "$ROOT/.venv-$name/bin/python" -m pip freeze 2>/dev/null | grep -iE "^(mpol|torch|torchkbnufft|visread|asdf|casatools|tensorboard|matplotlib|numpy)==" || \
    uv pip freeze --python "$ROOT/.venv-$name/bin/python" | grep -iE "^(mpol|torch|torchkbnufft|visread|asdf|casatools|tensorboard|matplotlib|numpy)=="
done
