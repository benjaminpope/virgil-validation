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


def virgil_source():
    """The virgil commit under test: from the installer's record (PEP 610)
    for a git install, or the checkout's HEAD for a local editable one."""
    try:
        dist = metadata.distribution("virgil-astro")
    except metadata.PackageNotFoundError:
        return {"commit": None, "url": None}
    info = json.loads(dist.read_text("direct_url.json") or "{}")
    url = info.get("url")
    commit = info.get("vcs_info", {}).get("commit_id")
    if commit is None and url and url.startswith("file://"):
        path = url[len("file://"):]
        try:
            commit = subprocess.check_output(
                ["git", "-C", path, "rev-parse", "HEAD"], text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
    return {"commit": commit, "url": url}


def own_commit():
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    try:
        return subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def candid_commit():
    """The CANDID commit scripts/setup_candid.sh checked out (CANDID runs in
    its own environment, so it is not among the installed packages)."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    try:
        return subprocess.check_output(
            ["git", "-C", os.path.join(root, ".external", "candid-src"), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_header():
    return {
        "record": "run",
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "virgil": {"version": _version("virgil-astro"), **virgil_source()},
        "validation_commit": own_commit(),
        "versions": {p: _version(p) for p in PACKAGES} | {"candid": candid_commit()},
        "python": platform.python_version(),
        "runner": "github-actions" if os.environ.get("GITHUB_ACTIONS") else "local",
        "run_url": (
            f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}"
            f"/actions/runs/{os.environ['GITHUB_RUN_ID']}"
            if os.environ.get("GITHUB_RUN_ID")
            else None
        ),
    }
