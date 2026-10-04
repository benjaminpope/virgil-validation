#!/usr/bin/env bash
# Install CANDID (Mérand, Gallenne et al.; github.com/amerand/CANDID) at a
# pinned commit into its own environment, .venv-candid. CANDID has no PyPI
# release and states no licence, so it is installed and called, never
# vendored. src/external_bridge/candid.py runs it in a subprocess.
set -euo pipefail
CANDID_SHA=c255e908f721a99c8ecdd5576f949a6fe0efee83
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC="$ROOT/.external/candid-src"
if [ ! -d "$SRC/.git" ]; then
  git clone -q https://github.com/amerand/CANDID "$SRC"
fi
git -C "$SRC" fetch -q origin
git -C "$SRC" checkout -q "$CANDID_SHA"
uv venv -q --allow-existing --python 3.12 "$ROOT/.venv-candid"
uv pip install -q --python "$ROOT/.venv-candid/bin/python" "$SRC"
echo "CANDID $CANDID_SHA in $ROOT/.venv-candid"
