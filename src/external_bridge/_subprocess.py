"""Run a worker script in another package's own environment, JSON in and out.

Packages with heavy, conflicting or unlicensed dependencies (CANDID,
eht-imaging, MPoL, fouriever) live in their own virtual environments,
.venv-<name>, made by scripts/setup_candid.sh and scripts/setup_external.sh.
Each has a worker script here that imports only that package, NumPy and the
standard library. ``<NAME>_PYTHON`` overrides the interpreter.
"""

import json
import os
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]


def python(name):
    return os.environ.get(f"{name.upper()}_PYTHON", str(ROOT / f".venv-{name}" / "bin" / "python"))


def available(name, module=None):
    """Whether the configured interpreter (a path or a command on PATH)
    can import the package."""
    try:
        out = subprocess.run([python(name), "-c", f"import {module or name}"], capture_output=True)
    except OSError:
        return False
    return out.returncode == 0


def version(name, distribution=None):
    code = f"import importlib.metadata as m; print(m.version({(distribution or name)!r}))"
    try:
        out = subprocess.run([python(name), "-c", code], capture_output=True, text=True)
    except OSError:
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def run(name, worker, task, timeout=3600):
    """Run ``worker`` (a script in this directory) under ``name``'s
    interpreter with ``task`` (a JSON-serialisable dict) and return its
    JSON result. Raises with the worker's output if it fails."""
    script = pathlib.Path(__file__).with_name(worker)
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = pathlib.Path(tmp, "task.json"), pathlib.Path(tmp, "result.json")
        src.write_text(json.dumps(task, default=str))
        env = {**os.environ, "MPLBACKEND": "Agg"}
        out = subprocess.run(
            [python(name), str(script), str(src), str(dst)],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
        if out.returncode != 0:
            raise RuntimeError(f"{name} worker failed:\n{out.stdout[-3000:]}\n{out.stderr[-3000:]}")
        return json.loads(dst.read_text())
