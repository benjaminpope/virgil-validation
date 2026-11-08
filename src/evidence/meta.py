"""Describe the code a test run used, for the header of an evidence file."""

import datetime
import json
import os
import platform
import socket
import subprocess
from importlib import metadata
from pathlib import Path

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
        out = subprocess.check_output(["git", "-C", path, "ls-files", "-z"], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None
    return {name.split("/", 1)[0] for name in out.split("\0") if "/" in name}


def _status_entries(path):
    """``(status, path)`` for each entry of ``git status --porcelain -z``
    (None if git fails). NUL-separated, so paths are never quoted; a rename or
    copy gives its new path, and its old path is dropped."""
    try:
        out = subprocess.check_output(["git", "-C", path, "status", "--porcelain", "-z", "--untracked-files=all"],
                                      text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None
    fields = out.split("\0")
    entries = []
    i = 0
    while i < len(fields):
        field = fields[i]
        i += 1
        if len(field) < 4:  # the empty tail after the last NUL, or a malformed entry
            continue
        status, name = field[:2], field[3:]
        if "R" in status or "C" in status:
            i += 1  # the next field is the old path
        entries.append((status, name))
    return entries


def _dirty(path, ignore=()):
    """Whether a git checkout has uncommitted changes (None if it cannot be told).

    Any edit to a tracked file counts. An untracked file counts only inside a
    top-level folder that already holds tracked files (``trust/``, ``src/``,
    ``tests/``, ...): CI leaves its own untracked material beside the tests
    (the virgil checkout, downloaded artefacts, ``evidence.jsonl``), which is
    not part of the code under test. ``ignore`` lists paths, relative to the
    checkout, that never count: exactly those files, not folders."""
    tracked = _tracked_dirs(path)
    entries = _status_entries(path)
    if entries is None:
        return None
    for status, name in entries:
        if name in ignore:
            continue
        if status == "??" and (tracked is None or name.split("/", 1)[0] not in tracked):
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


def _relative_to_root(path, root=None):
    """``path`` as git names it inside the checkout, or None if it is outside.
    Symlinks are resolved on both sides first."""
    if not path:
        return None
    root = Path(ROOT if root is None else root).resolve()
    target = Path(path).resolve()
    if not target.is_relative_to(root):
        return None
    return target.relative_to(root).as_posix()


def jax_info():
    """JAX version, backend, devices and the x64 flag as set when the header is taken."""
    try:
        import jax
        return {"version": jax.__version__, "backend": jax.default_backend(),
                "devices": [str(d) for d in jax.devices()],
                "jax_enable_x64": bool(jax.config.read("jax_enable_x64"))}
    except ImportError:
        return None
    except Exception as e:  # no usable backend: the header must still be written, saying why
        return {"error": f"{type(e).__name__}: {e}"}


def run_header(evidence=None):
    """The header line of an evidence file.

    ``runner`` is only the runner's claim (it reads ``GITHUB_ACTIONS``, which
    anyone can set); ``github_run_id`` is the raw environment value.
    ``validation_dirty`` ignores only the evidence output file itself
    (``evidence``, the plugin's ``--evidence`` path; None exempts nothing), so a hand-edited
    file under ``trust/evidence/`` makes the run dirty."""
    ignore = tuple(p for p in (_relative_to_root(evidence),) if p)
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
