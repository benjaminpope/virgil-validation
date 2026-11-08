"""Describe the code a test run used, for the header of an evidence file."""

import datetime
import json
import os
import platform
import socket
import subprocess
import sys
from importlib import metadata

PACKAGES = ("virgil-astro", "jax", "numpy", "scipy", "dLux", "pmoired", "astropy", "frank", "vis-r", "astro-frap",
            "oimodeler")


def _version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _tracked_dirs(path):
    """Top-level folders of the checkout that hold tracked files (None if git fails)."""
    try:
        out = subprocess.check_output(["git", "-C", path, "ls-files"], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None
    return {line.split("/", 1)[0] for line in out.splitlines() if "/" in line}


def _dirty(path, ignore=()):
    """Whether a git checkout has uncommitted changes (None if it cannot be told).

    Any edit to a tracked file counts. An untracked file counts only inside a
    top-level folder that already holds tracked files (``trust/``, ``src/``,
    ``tests/``, ...): CI leaves its own untracked material beside the tests
    (the virgil checkout, downloaded artefacts, ``evidence.jsonl``), which is
    not part of the code under test. ``ignore`` lists paths, relative to the
    checkout, that never count: exactly those files, not folders."""
    tracked = _tracked_dirs(path)
    try:
        out = subprocess.check_output(["git", "-C", path, "status", "--porcelain", "--untracked-files=all"],
                                      text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None
    for line in out.splitlines():
        name = line[3:].strip('"')
        if name in ignore:
            continue
        if line.startswith("??") and (tracked is None or name.split("/", 1)[0] not in tracked):
            continue
        return True
    return False


def virgil_source():
    """The virgil commit under test: from the installer's record (PEP 610)
    for a git install, or the checkout's HEAD for a local editable one, which
    is then also marked dirty if it has uncommitted changes."""
    try:
        dist = metadata.distribution("virgil-astro")
    except metadata.PackageNotFoundError:
        return {"commit": None, "url": None}
    info = json.loads(dist.read_text("direct_url.json") or "{}")
    url = info.get("url")
    commit = info.get("vcs_info", {}).get("commit_id")
    dirty = False
    if url and url.startswith("file://"):
        path = url[len("file://"):]
        dirty = _dirty(path)
        try:
            commit = subprocess.check_output(
                ["git", "-C", path, "rev-parse", "HEAD"], text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
    return {"commit": commit, "url": url, "dirty": dirty}


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def own_commit():
    root = ROOT
    try:
        return subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def candid():
    """The CANDID the tests run (it lives in its own environment, so it is
    not among the installed packages): see external_bridge.candid_bridge."""
    try:
        from external_bridge import candid_bridge
    except ImportError:
        return None
    return candid_bridge.provenance()


def external():
    """Versions of the packages that run in their own environments."""
    try:
        from external_bridge import _subprocess
    except ImportError:
        return None
    return {name: _subprocess.version(name) for name in ("ehtim", "mpol", "fouriever", "orbitize")}


def _evidence_path(argv=None):
    """The ``--evidence PATH`` of this pytest run, read from the command line
    (the plugin may also pass it to ``run_header`` directly)."""
    argv = sys.argv if argv is None else argv
    for i, arg in enumerate(argv):
        if arg == "--evidence" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--evidence="):
            return arg.split("=", 1)[1]
    return None


def _relative_to_root(path):
    """``path`` as git names it inside the checkout, or None if it is outside."""
    if not path:
        return None
    rel = os.path.relpath(os.path.abspath(path), ROOT)
    return None if rel.startswith("..") else rel.replace(os.sep, "/")


def jax_info():
    """JAX version, backend, devices and the x64 flag as set when the header is taken."""
    try:
        import jax
        return {"version": jax.__version__, "backend": jax.default_backend(),
                "devices": [str(d) for d in jax.devices()],
                "jax_enable_x64": bool(jax.config.read("jax_enable_x64"))}
    except Exception:  # no JAX, or no usable backend: the header must still be written
        return None


def run_header(evidence=None):
    """The header line of an evidence file.

    ``runner`` is only the runner's claim (it reads ``GITHUB_ACTIONS``, which
    anyone can set); ``github_run_id`` is the raw environment value.
    ``validation_dirty`` ignores only the evidence output file itself
    (``evidence``, default the command line's ``--evidence``), so a hand-edited
    file under ``trust/evidence/`` makes the run dirty."""
    ignore = tuple(p for p in (_relative_to_root(evidence or _evidence_path()),) if p)
    return {
        "record": "run",
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "virgil": {"version": _version("virgil-astro"), **virgil_source()},
        "validation_commit": own_commit(),
        # uncommitted changes, other than the evidence file being written
        "validation_dirty": _dirty(ROOT, ignore=ignore),
        "versions": {p: _version(p) for p in PACKAGES},
        "candid": candid(),
        "external": external(),
        "python": platform.python_version(),
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "jax": jax_info(),
        "env": {k: os.environ.get(k) for k in ("XLA_FLAGS", "OMP_NUM_THREADS", "JAX_NUM_THREADS")},
        "host": socket.gethostname(),
        "slurm": {k: os.environ[k] for k in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID")
                  if k in os.environ},
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "runner": "github-actions" if os.environ.get("GITHUB_ACTIONS") else "local",
        "run_url": (
            f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}"
            f"/actions/runs/{os.environ['GITHUB_RUN_ID']}"
            if os.environ.get("GITHUB_RUN_ID")
            else None
        ),
    }
