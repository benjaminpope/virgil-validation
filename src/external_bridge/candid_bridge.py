"""Call CANDID (Gallenne et al. 2015; github.com/amerand/CANDID) in its
own environment.

CANDID has no PyPI release and states no licence, so it is installed by
scripts/setup_candid.sh at a pinned commit into .venv-candid and run in a
subprocess (candid_worker.py), with JSON in and out. Set CANDID_PYTHON to
use another interpreter.
"""

import json
import os
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKER = pathlib.Path(__file__).with_name("candid_worker.py")


def python():
    return os.environ.get("CANDID_PYTHON", str(ROOT / ".venv-candid" / "bin" / "python"))


def available():
    """Whether the configured interpreter (a path or a command on PATH)
    can import CANDID."""
    try:
        out = subprocess.run([python(), "-c", "import candid"], capture_output=True)
    except OSError:
        return False
    return out.returncode == 0


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
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = pathlib.Path(tmp, "task.json"), pathlib.Path(tmp, "result.json")
        src.write_text(json.dumps({k: (str(v) if k == "path" else v) for k, v in task.items()}))
        env = {**os.environ, "MPLBACKEND": "Agg"}
        out = subprocess.run(
            [python(), str(WORKER), str(src), str(dst)],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
        if out.returncode != 0:
            raise RuntimeError(f"CANDID worker failed:\n{out.stdout[-3000:]}\n{out.stderr[-3000:]}")
        return json.loads(dst.read_text())
