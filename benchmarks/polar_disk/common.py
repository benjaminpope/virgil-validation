"""Shared helpers for the PolarDisk benchmarks (stage V of virgil's radial-profile plan):
provenance (virgil commit, software versions, machine) and result files.

Run every script with the repository's environment, from the repository root:
``.venv/bin/python benchmarks/polar_disk/v1_forward.py``.
Results go to ``$PD_OUT`` (default ``benchmarks/polar_disk/results``).
"""

import json
import os
import pathlib
import platform
import sys
from importlib import metadata

import jax

jax.config.update("jax_enable_x64", True)

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = pathlib.Path(os.environ.get("PD_OUT", pathlib.Path(__file__).parent / "results"))
PACKAGES = ("virgil-astro", "jax", "jaxlib", "numpy", "scipy", "numpyro", "frank", "vis-r", "astro-frap",
            "pmoired", "oimodeler")
# stage-V pin: origin/polar-p4 of benjaminpope/virgil (PR #343), P1-P4 stacked
EXPECTED_VIRGIL = "791f9fef65eb49ebc387c5c365a13c7af2342341"


def _version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def provenance():
    from evidence.meta import virgil_source, own_commit

    src = virgil_source()
    return {
        "virgil": {"version": _version("virgil-astro"), **src},
        "virgil_pin_ok": src.get("commit") == EXPECTED_VIRGIL,
        "versions": {p: _version(p) for p in PACKAGES},
        "python": platform.python_version(),
        "machine": f"{platform.system()} {platform.machine()}, {os.cpu_count()} CPUs",
        "jax_backend": jax.default_backend(),
        "validation_commit": own_commit(),
    }


def save(name, payload):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps({"provenance": provenance(), **payload}, indent=1, default=float) + "\n")
    return path
