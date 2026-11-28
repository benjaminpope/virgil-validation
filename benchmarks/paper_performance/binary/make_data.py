"""Deterministic data for the binary-fitting benchmark (virgil vs PMOIRED).

Simulated data and truth come from ``crosscheck`` only (plan section 1, Independence): this module imports neither
virgil nor PMOIRED. Files go to ``~/.cache/virgil-bench/data/binary/``; ``manifest.json`` (sha256 of every simulated
file, and of the two real NN Del files) is committed, and the runners verify it before use.

    python make_data.py [--size tiny|full] [--check]

``--check`` only verifies existing files against the manifest. ``--write-manifest`` (re)writes it after a full build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

DATA = common.BENCH_HOME / "data" / "binary"
MANIFEST = HERE / "manifest.json"
REAL_DIR = Path.home() / "data" / "eso_binaries" / "trackb" / "nn_del" / "oifits"

# the 4-UT layout of the existing PMOIRED fit tests (east, north, metres)
UTS = np.array([[-9.925, -20.335], [14.887, 30.502], [44.915, 66.183], [103.306, 43.999]])
HA = np.linspace(-3.0, 3.0, 5)
DEC = -50.0
SIGMA_V2 = 0.03
SIGMA_CP = 1.0
TRUTH = {"dra": 6.0, "ddec": -4.0, "flux": 0.10}
UD_TRUTH = 1.0  # mas, B-sim-ud primary
N_PULLS = 200
N_PULLS_TINY = 5
SEEDS = {"B-sim-6": 11, "B-sim-60": 12, "B-sim-600": 13, "B-sim-ud": 14}
PULL_SEED0 = {"triangle": 1000, "baseline": 2000}
CASES = {  # case -> (channels, uniform-disk diameter of the primary)
    "B-sim-6": (6, 0.0), "B-sim-60": (60, 0.0), "B-sim-600": (600, 0.0), "B-sim-ud": (6, UD_TRUTH),
}
SIM_FILE = {"B-sim-6": "sim6", "B-sim-60": "sim60", "B-sim-600": "sim600", "B-sim-ud": "simud"}


def wavelengths(n):
    return np.linspace(1.5e-6, 2.4e-6, n)


def scene_vis(ud=0.0, dra=None, ddec=None, flux=None):
    """Truth visibility function (crosscheck.sky): primary at the origin plus a point companion."""
    from crosscheck import sky

    dra = TRUTH["dra"] if dra is None else dra
    ddec = TRUTH["ddec"] if ddec is None else ddec
    flux = TRUTH["flux"] if flux is None else flux

    def vis(u, v, w):
        prim = sky.vis_uniform_disk(u, v, w, ud) if ud else sky.vis_point(u, v, w)
        return (prim + flux * sky.vis_point(u, v, w, dra, ddec)) / (1.0 + flux)

    return vis


def write_sim(path, n_ch, ud, seed, noise="triangle", clean=False):
    from crosscheck import simulate

    rng = None if clean else np.random.default_rng(seed)
    return simulate.observe(
        path, scene_vis(ud), UTS, hour_angles_h=HA, wavelengths=wavelengths(n_ch), dec_deg=DEC,
        sigma_v2=SIGMA_V2, sigma_cp_deg=SIGMA_CP, rng=rng, phase_noise=noise,
    )


def pull_name(noise, k):
    return f"pulls_{'tri' if noise == 'triangle' else 'base'}/s{k:03d}.fits"


def planned_files(size="full"):
    """{relative name: writer thunk} for the files of this size."""
    out = {}
    cases = ["B-sim-6"] if size == "tiny" else list(CASES)
    for c in cases:
        n_ch, ud = CASES[c]
        out[f"{SIM_FILE[c]}.fits"] = lambda p, n_ch=n_ch, ud=ud, c=c: write_sim(p, n_ch, ud, SEEDS[c])
    for c in ("B-sim-6", "B-sim-ud"):
        n_ch, ud = CASES[c]
        out[f"{SIM_FILE[c]}_clean.fits"] = lambda p, n_ch=n_ch, ud=ud: write_sim(p, n_ch, ud, 0, clean=True)
    n = N_PULLS_TINY if size == "tiny" else N_PULLS
    for noise in ("triangle", "baseline"):
        for k in range(n):
            out[pull_name(noise, k)] = lambda p, k=k, noise=noise: write_sim(
                p, 6, 0.0, PULL_SEED0[noise] + k, noise=noise)
    return out


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def real_files(epoch_date="2017-07-20"):
    """The two NN Del PIONIER files of one epoch, resolved through ``oidb/trackb/nn_del.json`` (its ``dp_ids`` and
    ``files`` block). Returns (paths, sha256 from the JSON, epoch dict)."""
    jpath = HERE.parents[2] / "oidb" / "trackb" / "nn_del.json"
    d = json.loads(jpath.read_text())
    ep = next(e for e in d["epochs"] if e["date"] == epoch_date)
    by_id = {f["dp_id"]: f for f in d["files"]}
    files = [by_id[i] for i in ep["dp_ids"]]
    return [REAL_DIR / f["file"] for f in files], [f["sha256"] for f in files], ep


def build(size="full"):
    DATA.mkdir(parents=True, exist_ok=True)
    for name, thunk in planned_files(size).items():
        p = DATA / name
        p.parent.mkdir(parents=True, exist_ok=True)
        thunk(p)
    return DATA


def manifest_dict(size="full"):
    files = {name: sha256(DATA / name) for name in planned_files(size)}
    paths, shas, ep = real_files()
    return {
        "note": "sha256 of every simulated file (sizes full); real NN Del files are read from ~/data, unmodified",
        "files": files,
        "real": {f"~/data/eso_binaries/trackb/nn_del/oifits/{p.name}": s for p, s in zip(paths, shas)},
    }


def verify(names=None):
    """Names whose sha256 differs from the committed manifest (empty list = fine). ``names`` None checks the files
    present in the data directory that the manifest lists; real files are always checked when present."""
    man = json.loads(MANIFEST.read_text())
    bad = []
    for name, digest in man["files"].items():
        if names is not None and name not in names:
            continue
        p = DATA / name
        if not p.exists():
            if names is not None:
                bad.append(name)
            continue
        if sha256(p) != digest:
            bad.append(name)
    return bad


def verify_real():
    paths, shas, _ = real_files()
    return [p.name for p, s in zip(paths, shas) if not p.exists() or sha256(p) != s]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--size", choices=["tiny", "full"], default="full")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write-manifest", action="store_true")
    a = ap.parse_args()
    if not a.check:
        build(a.size)
        print(f"wrote {len(planned_files(a.size))} files under {DATA}")
    if a.write_manifest:
        if a.size != "full":
            sys.exit("write the manifest from a full build")
        MANIFEST.write_text(json.dumps(manifest_dict(), indent=1, sort_keys=True) + "\n")
        print("manifest written")
    names = set(planned_files(a.size))
    bad = verify(names) + [f"real:{n}" for n in verify_real()]
    print("manifest check:", "OK" if not bad else f"MISMATCH {bad[:5]}")
    sys.exit(1 if bad and not a.write_manifest else 0)


if __name__ == "__main__":
    main()
