"""Copy orbit-alias results into the docs, regenerate the page and build the site.

    python scripts/pull_back_orbit_results.py RESULTS_DIR [--no-build] [--strict]

RESULTS_DIR is anywhere above the per-system output directories of reanalyse_orbits.py
(e.g. ozstar_scripts/results/orbit_aliases/<tag>): every directory under it holding a
bands.json and named after a registered system is used. Copies bands.json, meta.json
and the PNGs to docs/assets/orbit_aliases/<system>/, runs scripts/build_aliases_page.py,
then `.venv/bin/zensical build --clean` (add --strict as CI does). Needs no JAX.
"""

import argparse
import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import build_aliases_page as B  # noqa: E402

FILES = ("bands.json", "meta.json", "sky.png", "corner.png", "residuals.png")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=pathlib.Path)
    ap.add_argument("--no-build", action="store_true", help="copy and regenerate the page, skip zensical")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--assets", type=pathlib.Path, default=B.ASSETS)
    a = ap.parse_args(argv)
    known = set(B._registry())
    found = {}
    for bj in sorted(a.results.rglob("bands.json")):  # later (sorted) tags overwrite earlier ones
        if bj.parent.name in known:
            found[bj.parent.name] = bj.parent
    if not found:
        sys.exit(f"no <system>/bands.json under {a.results} for any of {sorted(known)}")
    for name, src in found.items():
        dst = a.assets / name
        dst.mkdir(parents=True, exist_ok=True)
        for f in FILES:
            if (src / f).exists():
                shutil.copy2(src / f, dst / f)
        print(f"copied {name} from {src}")
    B.OUT.write_text(B.render(a.assets))
    print(f"regenerated {B.OUT.relative_to(REPO)}")
    if a.no_build:
        return
    cmd = [str(REPO / ".venv" / "bin" / "zensical"), "build", "--clean"] + (["--strict"] if a.strict else [])
    if not pathlib.Path(cmd[0]).exists():
        sys.exit(f"{cmd[0]} not found: uv pip install --python .venv/bin/python -e '.[docs]'")
    sys.exit(subprocess.call(cmd, cwd=REPO))


if __name__ == "__main__":
    main()
