#!/usr/bin/env bash
# Install CANDID (Mérand, Gallenne et al.; github.com/amerand/CANDID) at a
# pinned commit into its own environment, .venv-candid. CANDID has no PyPI
# release and states no licence, so it is installed and called, never
# vendored. src/external_bridge/candid_bridge.py runs it in a subprocess
# (candid_worker.py). The clone is reset to the pinned commit and cleaned,
# so the installed code is exactly that commit.
set -euo pipefail
CANDID_SHA=c255e908f721a99c8ecdd5576f949a6fe0efee83
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC="$ROOT/.external/candid-src"
if [ ! -d "$SRC/.git" ]; then
  git clone -q https://github.com/amerand/CANDID "$SRC"
fi
git -C "$SRC" fetch -q origin
git -C "$SRC" checkout -q --force --detach "$CANDID_SHA"
git -C "$SRC" clean -q -f -d -x
uv venv -q --allow-existing --python 3.12 "$ROOT/.venv-candid"
uv pip install -q --reinstall-package candid --python "$ROOT/.venv-candid/bin/python" "$SRC"
git -C "$SRC" clean -q -f -d -x  # the build left files behind; the install is a copy
echo "CANDID $CANDID_SHA in $ROOT/.venv-candid"
