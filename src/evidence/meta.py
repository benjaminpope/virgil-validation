"""Describe the code a test run used, for the header of an evidence file."""

import datetime
import json
import os
import platform
import subprocess
from importlib import metadata

PACKAGES = ("virgil-astro", "jax", "numpy", "scipy", "dLux", "pmoired", "astropy")


def _version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _dirty(path, ignore=()):
    """Whether a git checkout has uncommitted changes (None if it cannot be told)."""
    try:
        out = subprocess.check_output(["git", "-C", path, "status", "--porcelain"], text=True,
                                      stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None
    return any(not line[3:].startswith(ignore) for line in out.splitlines())


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


def run_header():
    return {
        "record": "run",
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "virgil": {"version": _version("virgil-astro"), **virgil_source()},
        "validation_commit": own_commit(),
        # uncommitted changes, other than evidence files being written
        "validation_dirty": _dirty(ROOT, ignore=("trust/evidence/",)),
        "versions": {p: _version(p) for p in PACKAGES},
        "candid": candid(),
        "external": external(),
        "python": platform.python_version(),
        "runner": "github-actions" if os.environ.get("GITHUB_ACTIONS") else "local",
        "run_url": (
            f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}"
            f"/actions/runs/{os.environ['GITHUB_RUN_ID']}"
            if os.environ.get("GITHUB_RUN_ID")
            else None
        ),
    }
