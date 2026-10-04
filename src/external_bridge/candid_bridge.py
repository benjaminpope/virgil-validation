"""Call CANDID (Gallenne et al. 2015; github.com/amerand/CANDID) in its
own environment.

CANDID has no PyPI release and states no licence, so it is installed by
scripts/setup_candid.sh at a pinned commit into .venv-candid and run in a
subprocess (candid_worker.py), with JSON in and out. Set CANDID_PYTHON to
use another interpreter.
"""

import json
import pathlib
import subprocess

from . import _subprocess

ROOT = _subprocess.ROOT


def python():
    return _subprocess.python("candid")


def available():
    """Whether the configured interpreter (a path or a command on PATH)
    can import CANDID."""
    return _subprocess.available("candid")


def provenance():
    """What CANDID the tests run: the interpreter, CANDID's version string
    and file, and the pinned commit when it is the managed environment of
    scripts/setup_candid.sh with a clean checkout (otherwise None)."""
    code = "import candid, json; print(json.dumps([candid.__version__, candid.__file__]))"
    try:
        out = subprocess.run([python(), "-c", code], capture_output=True, text=True)
    except OSError:
        return None
    if out.returncode != 0:
        return None
    version, path = json.loads(out.stdout.strip().splitlines()[-1])
    managed = ROOT / ".venv-candid"
    commit = None
    if pathlib.Path(path).resolve().is_relative_to(managed.resolve()):
        src = ROOT / ".external" / "candid-src"
        try:
            head = subprocess.check_output(["git", "-C", str(src), "rev-parse", "HEAD"], text=True).strip()
            dirty = subprocess.check_output(["git", "-C", str(src), "status", "--porcelain"], text=True).strip()
            commit = None if dirty else head
        except (OSError, subprocess.CalledProcessError):
            commit = None
    return {"python": python(), "version": version, "file": path, "commit": commit}


def run(task, timeout=3600):
    """Run one worker task (a dict with "task" and its arguments) and return
    its result. Raises with CANDID's output if the worker fails."""
    return _subprocess.run("candid", "candid_worker.py", task, timeout=timeout)
