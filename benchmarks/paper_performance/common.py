"""Timing harness, provenance, row schema and environment checks shared by every paper-performance benchmark.

Protocol: ``reports/benchmarks-plan.md`` sections 1-3 (kept with the paper; the rules are repeated in README.md).

Outer side (this module, called by ``run_all.sh`` or a runner):
    ``timed_process(cmd, ...)``  launch one fresh process under ``/usr/bin/time -l`` (optionally through
                                 ``pyslot``), parse wall, user, sys, peak memory footprint and RSS, and pick up the
                                 worker's inner timings from its ``BENCH_INNER {json}`` line.
    ``provenance()``             commits, package versions, machine, thread environment seen by the process.
    ``machine_state()``          AC power, Low Power Mode, load average, pyslot occupancy.
    ``check_threads()``          ``threads_not_held`` flag from ``(user+sys)/wall``.
    ``write_row()``              validate and write one JSON row in ``results/raw/``.
    ``validate_row()``           the row schema (also used by ``summarise.py --check``).

Inner side (worker processes):
    ``Stopwatch``                ``perf_counter`` section timer; ``emit_inner()`` prints the ``BENCH_INNER`` line.
    ``block(x)``                 ``block_until_ready`` on any JAX outputs inside ``x`` (no-op without JAX).

Nothing here imports virgil or an external package at module level.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BENCH_HOME = Path(os.environ.get("VIRGIL_BENCH_HOME", Path.home() / ".cache" / "virgil-bench"))
RAW = HERE / "results" / "raw"
TIME_BIN = "/usr/bin/time"
PYSLOT = Path.home() / ".claude" / "bin" / "pyslot"
# BENCH_PLATFORM=slurm: the measured matrix runs inside one exclusive Slurm job on a Linux compute node (the ozstar_scripts
# stage bench275). There is no pyslot (the job owns the node), no macOS ``time -l`` (rusage comes from ``os.wait4``) and no
# AC / Low Power Mode (the machine check reads the node's own load and records the node).
ON_NODE = os.environ.get("BENCH_PLATFORM") == "slurm"
INNER_TAG = "BENCH_INNER "

STATUSES = ("ok", "timeout", "oom", "error", "gate_failed", "not_run", "unsupported")
COMPARISONS = ("binary", "orbit", "imaging", "dummy")
CONFIGS = ("matched", "tight", "native", "matched-coarse", "matched-truthstart", "dummy")
FLAGS = (
    "noisy", "threads_not_held", "single_run", "mem_children_not_summed", "warm_recompiled",
    "corrections_on", "likelihood_mismatch", "swapped",
)
PACKAGES = (
    "virgil-astro", "jax", "jaxlib", "numpyro", "pmoired", "ehtim", "numpy", "scipy", "arviz", "astropy",
)
THREAD_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)

# ----------------------------------------------------------------------------------------------------------------
# environment: threads
# ----------------------------------------------------------------------------------------------------------------


# Extra XLA CPU flags that stop JAX using more than the stated cores (run 1 showed (user+sys)/wall up to 4.75 at "1
# thread": the Eigen switch does not cover XLA's parallel code generation, the XNNPACK thread pool or the concurrent
# scheduler). The set below is what ``thread_probe.py`` found to hold ``(user+sys)/wall`` under 1.2 (see README).
XLA_HOLD_FLAGS = ("--xla_cpu_parallel_codegen_split_count=1 --xla_cpu_use_xnnpack=false "
                  "--xla_cpu_enable_concurrency_optimized_scheduler=false")


def xla_flags(n: int, hold: bool = True) -> str:
    """The ``XLA_FLAGS`` string for ``n`` threads; ``hold`` adds ``XLA_HOLD_FLAGS`` (1 thread only)."""
    eigen = "true" if n > 1 else "false"
    flags = f"--xla_cpu_multi_thread_eigen={eigen} intra_op_parallelism_threads={n}"
    if hold and n == 1:
        flags += " " + XLA_HOLD_FLAGS
    return flags


def thread_env(n: int = 1) -> dict[str, str]:
    """Environment variables that hold a process to ``n`` threads (plan section 3; ``n`` is 1 or 2, never more)."""
    if n not in (1, 2):
        raise ValueError("benchmarks run on 1 thread (primary) or 2 (second pass), never more")
    env = {v: str(n) for v in THREAD_VARS}
    env["XLA_FLAGS"] = xla_flags(n)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["JULIA_NUM_THREADS"] = str(n)
    env["PYTHONHASHSEED"] = "0"
    return env


def thread_env_seen() -> dict[str, str | None]:
    """The thread-related environment as the current process sees it (stored in every row)."""
    return {k: os.environ.get(k) for k in (*THREAD_VARS, "XLA_FLAGS", "JAX_PLATFORMS", "JULIA_NUM_THREADS")}


def check_threads(cpu_ratio: float | None, threads: int) -> list[str]:
    """``(user+sys)/wall`` above 1.2 in a 1-thread row, or 2.3 in a 2-thread row, gives ``threads_not_held``."""
    if cpu_ratio is None:
        return []
    limit = 1.2 if threads == 1 else 2.3
    return ["threads_not_held"] if cpu_ratio > limit else []


# ----------------------------------------------------------------------------------------------------------------
# machine state
# ----------------------------------------------------------------------------------------------------------------


def _run(cmd, timeout=10):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:  # noqa: BLE001 - machine probing must never break a run
        return ""


def _node_state() -> dict:
    """Slurm node: one exclusive job, load low. ``ok`` is False if the 1-minute load is above ``BENCH_MAX_LOAD`` (default 2:
    the node is ours alone, so anything higher is a leftover process of ours or an unexpectedly shared node) or if the job
    does not hold the whole node."""
    load = os.getloadavg()
    mem_kb = 0
    for ln in Path("/proc/meminfo").read_text().splitlines():
        if ln.startswith("MemTotal:"):
            mem_kb = int(ln.split()[1])
    ncpu = os.cpu_count() or 0
    held = int(os.environ.get("SLURM_CPUS_ON_NODE", "0") or 0)
    problems = []
    if load[0] > float(os.environ.get("BENCH_MAX_LOAD", "2")):
        problems.append(f"node load {load[0]:.1f} above {os.environ.get('BENCH_MAX_LOAD', '2')}")
    if not os.environ.get("SLURM_JOB_ID"):
        problems.append("not inside a Slurm job")
    elif held and ncpu and held < ncpu:
        problems.append(f"job holds {held} of {ncpu} cpus (not --exclusive)")
    return {
        "platform": "slurm", "node": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_cpus_on_node": held, "cpu_count": ncpu, "mem_total_gb": round(mem_kb / 1e6, 1), "load_avg": list(load),
        "on_ac": None, "low_power_mode": None, "pyslot_busy": False, "problems": problems, "ok": not problems,
    }


def machine_state() -> dict:
    """AC power, Low Power Mode, load average, pyslot occupancy. ``ok`` is True only when a measured run may start."""
    if ON_NODE:
        return _node_state()
    batt = _run(["pmset", "-g", "batt"])
    on_ac = "AC Power" in batt
    lpm = None
    m = re.search(r"lowpowermode\s+(\d)", _run(["pmset", "-g"]))
    if m:
        lpm = int(m.group(1))
    slots = _run([str(PYSLOT), "--status"]) if PYSLOT.exists() else ""
    busy = bool(re.search(r"^\s+slot\d+ pid=", slots, re.M)) or bool(re.search(r"^\s+queue #", slots, re.M))
    load = os.getloadavg()
    problems = []
    if not on_ac:
        problems.append("not on AC power")
    if lpm != 0:
        problems.append(f"Low Power Mode not off ({lpm})")
    if busy:
        problems.append("pyslot busy")
    return {
        "on_ac": on_ac, "low_power_mode": lpm, "pyslot_busy": busy, "load_avg": list(load),
        "problems": problems, "ok": not problems,
    }


def chip() -> str:
    if ON_NODE:
        for ln in Path("/proc/cpuinfo").read_text().splitlines():
            if ln.startswith("model name"):
                return ln.split(":", 1)[1].strip()
    return _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip() or platform.processor()


# ----------------------------------------------------------------------------------------------------------------
# provenance
# ----------------------------------------------------------------------------------------------------------------


def _git(args, cwd=ROOT):
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def virgil_ref() -> str | None:
    """The virgil commit the shared environment was built at (written by ``env/setup_bench.sh``)."""
    p = BENCH_HOME / "VIRGIL_REF"
    return p.read_text().strip() if p.exists() else None


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_hash(manifest_path) -> str | None:
    """sha256 of a ``manifest.json`` file (the data manifest committed beside each ``make_data.py``)."""
    p = Path(manifest_path)
    return file_sha256(p) if p.exists() else None


def verify_manifest(manifest_path, data_dir) -> list[str]:
    """Return the files whose sha256 differs from ``manifest.json`` (``{relative path: sha256}`` under ``files``)."""
    man = json.loads(Path(manifest_path).read_text())
    bad = []
    for rel, digest in man["files"].items():
        p = Path(data_dir) / rel
        if not p.exists() or file_sha256(p) != digest:
            bad.append(rel)
    return bad


def provenance(extra_packages=(), julia: dict | None = None, data_manifest: str | None = None) -> dict:
    """Everything a row needs to be reproduced. Reads versions from package metadata; imports nothing heavy."""
    pkgs = {p: _version(p) for p in (*PACKAGES, *extra_packages)}
    out = {
        "vv_commit": _git(["rev-parse", "HEAD"]),
        "vv_dirty": bool(_git(["status", "--porcelain", "--untracked-files=no"])),
        "virgil_pin": virgil_ref(),
        "versions": pkgs,
        "python": platform.python_version(),
        "macos": platform.mac_ver()[0],
        "chip": chip(),
        "cpu_count": os.cpu_count(),
        "thread_env": thread_env_seen(),
        "data_manifest_sha256": data_manifest,
    }
    if julia:
        out["julia"] = julia
    return out


# ----------------------------------------------------------------------------------------------------------------
# inner side
# ----------------------------------------------------------------------------------------------------------------


class Stopwatch:
    """Section timer for a worker: ``with sw("t_cold"): ...``; ``sw.times`` holds seconds, ``sw.lists`` the repeats."""

    def __init__(self):
        self.times: dict[str, float] = {}
        self.lists: dict[str, list[float]] = {}
        self._t0 = time.perf_counter()

    def __call__(self, name: str, repeat: bool = False):
        return _Section(self, name, repeat)

    def total(self) -> float:
        return time.perf_counter() - self._t0


class _Section:
    def __init__(self, sw, name, repeat):
        self.sw, self.name, self.repeat = sw, name, repeat

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        dt = time.perf_counter() - self.t
        if self.repeat:
            self.sw.lists.setdefault(self.name, []).append(dt)
        else:
            self.sw.times[self.name] = dt
        return False


def block(x):
    """``block_until_ready`` on every JAX array in ``x`` (a pytree); returns ``x``. Without JAX, a no-op."""
    try:
        import jax

        jax.tree_util.tree_map(lambda a: a.block_until_ready() if hasattr(a, "block_until_ready") else a, x)
    except ImportError:
        pass
    return x


def rusage_peak_mb() -> float:
    """Peak resident set size from ``resource.getrusage`` (bytes on macOS), MB; the in-process cross-check."""
    import resource

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3


def emit_inner(sw: Stopwatch, **payload) -> None:
    """Print the single ``BENCH_INNER`` line the outer harness parses: timings plus whatever the worker adds."""
    out = {**sw.times, **sw.lists, "t_inner_total": sw.total(), "rusage_peak_mb": rusage_peak_mb(),
           "thread_env": thread_env_seen(), **payload}
    print(INNER_TAG + json.dumps(out, default=float), flush=True)


# ----------------------------------------------------------------------------------------------------------------
# outer side: one fresh process under /usr/bin/time -l
# ----------------------------------------------------------------------------------------------------------------

_TIME_RE = {
    "real": re.compile(r"^\s*([\d.]+)\s+real\s+([\d.]+)\s+user\s+([\d.]+)\s+sys", re.M),
    "rss": re.compile(r"^\s*(\d+)\s+maximum resident set size", re.M),
    "footprint": re.compile(r"^\s*(\d+)\s+peak memory footprint", re.M),
}


def parse_time_l(text: str) -> dict:
    """Parse ``/usr/bin/time -l`` (macOS) output. Memory values are bytes in the output and returned in MB."""
    out: dict = {"wall": None, "user": None, "sys": None, "peak_rss_mb": None, "peak_footprint_mb": None}
    m = _TIME_RE["real"].search(text)
    if m:
        out["wall"], out["user"], out["sys"] = (float(g) for g in m.groups())
    m = _TIME_RE["rss"].search(text)
    if m:
        out["peak_rss_mb"] = int(m.group(1)) / 1e6
    m = _TIME_RE["footprint"].search(text)
    if m:
        out["peak_footprint_mb"] = int(m.group(1)) / 1e6
    out["cpu_ratio"] = (out["user"] + out["sys"]) / out["wall"] if out["wall"] else None
    return out


def parse_inner(stdout: str) -> dict | None:
    """The last ``BENCH_INNER`` line of a worker's stdout, decoded; None if absent."""
    found = None
    for line in stdout.splitlines():
        if line.startswith(INNER_TAG):
            found = json.loads(line[len(INNER_TAG):])
    return found


# ----------------------------------------------------------------------------------------------------------------
# JAX persistent compilation cache (virgil workers only; other codes ignore the variables)
# ----------------------------------------------------------------------------------------------------------------
# Mode, from ``BENCH_JAX_CACHE_MODE``: "cold" (default) clears the cache directory of this row before the process, so a
# cold measurement stays cold (the cache is still on, so repeated programs inside the process are read back); "populated"
# keeps the directory a preceding cold process of the same row filled (the "warm with cache" measurement); "off" sets
# nothing. The directory lives under the bench cache, one per row key.

JAX_CACHE_MODES = ("cold", "populated", "off")


def jax_cache_mode() -> str:
    m = os.environ.get("BENCH_JAX_CACHE_MODE", "cold")
    if m not in JAX_CACHE_MODES:
        raise ValueError(f"BENCH_JAX_CACHE_MODE {m!r} not in {JAX_CACHE_MODES}")
    return m


def jax_cache_dir(key: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)[:120]
    return BENCH_HOME / "jax_cache" / safe


def jax_cache_env(key: str | None, mode: str | None = None) -> tuple[dict, str]:
    """(environment variables, mode) for one process of the row ``key``; clears the directory in ``cold`` mode."""
    import shutil

    mode = mode or jax_cache_mode()
    if mode == "off" or not key:
        return {}, "off"
    d = jax_cache_dir(key)
    if mode == "cold":
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    return {"BENCH_JAX_CACHE_DIR": str(d)}, mode


def enable_jax_cache() -> str | None:
    """Worker side: call right after ``import virgil``, before any jitted call. Turns the persistent cache on through
    virgil's own ``enable_compilation_cache`` (virgil >= the commit of PR 742) and falls back to setting JAX's config with
    the same thresholds (zero) when the installed virgil lacks it or is not imported. No-op without the variable."""
    d = os.environ.get("BENCH_JAX_CACHE_DIR")
    if not d:
        return None
    import sys

    v = sys.modules.get("virgil")
    fn = getattr(v, "enable_compilation_cache", None)
    if fn is not None:
        fn(d)
        return d
    import jax

    jax.config.update("jax_compilation_cache_dir", d)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    return d


def _wait4_run(argv, env, cwd, timeout):
    """Linux: run ``argv`` and return (stdout, stderr, returncode, wall, user, sys, peak_rss_mb, timed_out) from ``os.wait4``
    (the rusage of the process and the descendants it waited for; ``ru_maxrss`` is in kB). Output goes through temp files so
    nothing but the child is scheduled while it runs."""
    import tempfile

    with tempfile.TemporaryFile("w+b") as fo, tempfile.TemporaryFile("w+b") as fe:
        t0 = time.perf_counter()
        p = subprocess.Popen(argv, stdout=fo, stderr=fe, env=env, cwd=cwd, start_new_session=True)
        timed_out = False
        while True:
            pid, st, ru = os.wait4(p.pid, os.WNOHANG)
            if pid:
                break
            if time.perf_counter() - t0 > timeout:
                timed_out = True
                os.killpg(p.pid, 9)
                pid, st, ru = os.wait4(p.pid, 0)
                break
            time.sleep(0.02)
        wall = time.perf_counter() - t0
        fo.seek(0)
        fe.seek(0)
        out, err = fo.read().decode(errors="replace"), fe.read().decode(errors="replace")
    rc = None if timed_out else os.waitstatus_to_exitcode(st)
    return out, err, rc, wall, ru.ru_utime, ru.ru_stime, ru.ru_maxrss / 1e3, timed_out


def timed_process(cmd, *, threads: int = 1, timeout: float = 1800, wrapper: str | None = None,
                  pyslot_mem: str = "6G", env: dict | None = None, cwd=None, cache_key: str | None = None) -> dict:
    """Run ``cmd`` once in a fresh process under ``/usr/bin/time -l`` and return its measurements.

    ``wrapper``: None (the caller is already inside pyquick/pyslot), ``"pyslot"`` (as ``run_all.sh`` does, with
    ``PYSLOT_MEM=pyslot_mem``). The thread environment is set from ``threads``. ``cache_key`` names the JAX persistent
    cache directory of this row (see ``jax_cache_env``; the mode comes from ``BENCH_JAX_CACHE_MODE``). The result has
    ``status`` (ok, timeout, oom, error), ``t_process`` (outer wall minus inner total when the worker reports one), the
    parsed ``time -l`` fields, the worker's ``inner`` dict, ``jax_cache`` (the mode), and the stdout/stderr tails.
    """
    cmd = [str(c) for c in cmd]
    cache_env, cache_mode = jax_cache_env(cache_key)
    full_env = {**os.environ, **thread_env(threads), **cache_env, **(env or {})}
    argv = [TIME_BIN, "-l", *cmd]
    if ON_NODE and wrapper == "pyslot":
        wrapper = None   # the Slurm job owns the node: there is no pyslot
    if wrapper == "pyslot":
        full_env["PYSLOT_MEM"] = pyslot_mem
        argv = [str(PYSLOT), *argv]
    elif wrapper is not None:
        raise ValueError(f"unknown wrapper {wrapper!r}")
    t0 = time.perf_counter()
    status = "ok"
    if ON_NODE:
        stdout, stderr, rc, w, u, sy, rss, timed_out = _wait4_run(cmd, full_env, cwd, timeout)
        status = "timeout" if timed_out else "ok"
        wall_outer = time.perf_counter() - t0
        parsed = {"wall": w, "user": u, "sys": sy, "peak_rss_mb": rss, "peak_footprint_mb": rss,
                  "cpu_ratio": (u + sy) / w if w else None}
    else:
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=full_env, cwd=cwd)
            stdout, stderr, rc = p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired as e:
            stdout = (e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout) or ""
            stderr = (e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr) or ""
            rc, status = None, "timeout"
        wall_outer = time.perf_counter() - t0
        parsed = parse_time_l(stderr)
    inner = parse_inner(stdout)
    if status == "ok":
        if rc in (137, -9) or "memory footprint exceeded" in stderr:
            status = "oom"
        elif rc == 124 and "exceeded" in stderr:
            status = "timeout"
        elif rc != 0:
            status = "error"
        elif inner is None:
            status = "error"
    t_process = None
    if inner and parsed["wall"] is not None:
        t_process = max(parsed["wall"] - inner.get("t_inner_total", 0.0), 0.0)
    flags = check_threads(parsed["cpu_ratio"], threads)
    return {
        "status": status, "returncode": rc, "wall_outer": wall_outer, "t_process": t_process, **parsed,
        "inner": inner, "flags": flags, "jax_cache": cache_mode, "stdout_tail": stdout[-2000:], "stderr_tail": stderr[-2000:],
    }


def repeat_plan(t_single_call: float) -> dict:
    """Repeats from the length of one call (plan section 3): 3 x 5; 3 x 3 above 60 s; 1 x 1 above 10 min."""
    if t_single_call > 600:
        return {"processes": 1, "warm": 1, "flags": ["single_run"]}
    if t_single_call > 60:
        return {"processes": 3, "warm": 3, "flags": []}
    return {"processes": 3, "warm": 5, "flags": []}


def warm_stats(values) -> dict:
    """Median and min-max of the warm repeats (all raw values stay in the row)."""
    v = [float(x) for x in values]
    if not v:
        return {"median": None, "min": None, "max": None}
    return {"median": statistics.median(v), "min": min(v), "max": max(v)}


def flag_noisy(rows: list[dict], language_key: str = "language") -> list[dict]:
    """Add ``noisy`` to rows whose ``t_calib`` exceeds 1.3x the session median for that language."""
    by_lang: dict[str, list[float]] = {}
    for r in rows:
        if r.get("t_calib") is not None:
            by_lang.setdefault(r.get(language_key, "python"), []).append(r["t_calib"])
    med = {k: statistics.median(v) for k, v in by_lang.items()}
    for r in rows:
        t = r.get("t_calib")
        if t is not None and t > 1.3 * med[r.get(language_key, "python")] and "noisy" not in r["flags"]:
            r["flags"].append("noisy")
    return rows


def ratio_with_spread(other: dict, virgil: dict) -> dict:
    """Ratio other/virgil of medians, with the range implied by the min-max of both (plan section 3)."""
    if None in (other.get("median"), virgil.get("median")) or virgil["median"] == 0:
        return {"ratio": None, "lo": None, "hi": None}
    return {
        "ratio": other["median"] / virgil["median"],
        "lo": other["min"] / virgil["max"],
        "hi": other["max"] / virgil["min"],
    }


# ----------------------------------------------------------------------------------------------------------------
# rows
# ----------------------------------------------------------------------------------------------------------------

_REQUIRED = {
    "case": str, "comparison": str, "code": str, "config": str, "threads": int, "size": str, "status": str,
    "t_process": (int, float, type(None)), "t_calib": (int, float, type(None)), "t_import": (int, float, type(None)),
    "t_load": (int, float, type(None)), "t_init": (int, float, type(None)), "t_cold": (int, float, type(None)),
    "t_warm": list, "peak_footprint_mb": (int, float, type(None)), "peak_rss_mb": (int, float, type(None)),
    "baseline_mb": (int, float, type(None)), "cpu_ratio": (int, float, type(None)), "flags": list,
    "n_data": (int, type(None)), "n_param": (int, type(None)), "iterations": dict, "accuracy": dict,
    "settings": dict, "provenance": dict,
}


def new_row(*, case, comparison, code, config, threads=1, size="full", status="ok", **fields) -> dict:
    """A row with every key of the schema present (None / empty by default)."""
    row = {
        "case": case, "comparison": comparison, "code": code, "config": config, "threads": threads, "size": size,
        "status": status, "t_process": None, "t_calib": None, "t_import": None, "t_load": None, "t_init": None,
        "t_cold": None, "t_warm": [], "peak_footprint_mb": None, "peak_rss_mb": None, "baseline_mb": None,
        "cpu_ratio": None, "flags": [], "n_data": None, "n_param": None, "iterations": {}, "accuracy": {},
        "settings": {}, "provenance": {},
    }
    row.update(fields)
    return row


def validate_row(row: dict) -> list[str]:
    """Schema errors for one row (empty list = valid). Used by ``write_row`` and ``summarise.py --check``."""
    errs = []
    for k, typ in _REQUIRED.items():
        if k not in row:
            errs.append(f"missing key {k}")
        elif not isinstance(row[k], typ) or (typ is int and isinstance(row[k], bool)):
            errs.append(f"{k}: bad type {type(row[k]).__name__}")
    if errs:
        return errs
    if row["status"] not in STATUSES:
        errs.append(f"status {row['status']!r} not in {STATUSES}")
    if row["comparison"] not in COMPARISONS:
        errs.append(f"comparison {row['comparison']!r} not in {COMPARISONS}")
    if row["threads"] not in (1, 2):
        errs.append("threads must be 1 or 2")
    for f in row["flags"]:
        if f not in FLAGS:
            errs.append(f"unknown flag {f!r}")
    if any(not isinstance(x, (int, float)) for x in row["t_warm"]):
        errs.append("t_warm must hold numbers")
    if row["status"] == "ok" and row["t_cold"] is None:
        errs.append("status ok needs t_cold")
    if row["status"] in ("not_run", "gate_failed", "unsupported", "error") and "reason" not in row["settings"] \
            and "reason" not in row["accuracy"]:
        errs.append(f"status {row['status']} needs a reason in settings or accuracy")
    return errs


def row_from_process(res: dict, **identity) -> dict:
    """Build a row from a ``timed_process`` result and the identity keys (case, comparison, code, config, ...).

    Timings come from the worker's inner dict (``t_calib``, ``t_import``, ``t_load``, ``t_init``, ``t_cold``,
    ``t_warm`` plus ``n_data``, ``n_param``, ``iterations``, ``accuracy``, ``settings``, ``version`` extras).
    """
    inner = res.get("inner") or {}
    flags = list(res["flags"]) + list(inner.get("flags", []))
    return new_row(
        status=res["status"], t_process=res["t_process"], t_calib=inner.get("t_calib"),
        t_import=inner.get("t_import"), t_load=inner.get("t_load"), t_init=inner.get("t_init"),
        t_cold=inner.get("t_cold"), t_warm=list(inner.get("t_warm", [])),
        peak_footprint_mb=res["peak_footprint_mb"], peak_rss_mb=res["peak_rss_mb"],
        baseline_mb=inner.get("baseline_mb"), cpu_ratio=res["cpu_ratio"], flags=flags,
        n_data=inner.get("n_data"), n_param=inner.get("n_param"), iterations=inner.get("iterations", {}),
        accuracy=inner.get("accuracy", {}),
        settings={**inner.get("settings", {}), "run": os.environ.get("BENCH_RUN_ID", "r1"),
                  "jax_cache": res.get("jax_cache", "off"),
                  **({"reason": res["stderr_tail"][-300:]} if res["status"] != "ok" else {})},
        provenance=inner.get("provenance", provenance()), **identity,
    )


def write_row(row: dict, out_dir=None, name: str | None = None) -> Path:
    """Validate and write one row as JSON in ``results/raw/`` (or ``out_dir``). Raises ValueError on a schema error."""
    errs = validate_row(row)
    if errs:
        raise ValueError("invalid row: " + "; ".join(errs))
    d = Path(out_dir) if out_dir else RAW
    d.mkdir(parents=True, exist_ok=True)
    stem = name or f"{row['comparison']}__{row['case']}__{row['code']}__{row['config']}__t{row['threads']}"
    n = len(list(d.glob(stem + "__p*.json")))
    path = d / f"{stem}__p{n}.json"
    path.write_text(json.dumps(row, indent=1, default=float) + "\n")
    return path


def read_rows(raw_dir=None) -> list[dict]:
    d = Path(raw_dir) if raw_dir else RAW
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]


def check_rows(raw_dir=None) -> dict[str, list[str]]:
    """Schema errors per file in ``results/raw/`` (the core of ``summarise.py --check``)."""
    d = Path(raw_dir) if raw_dir else RAW
    return {p.name: e for p in sorted(d.glob("*.json")) if (e := validate_row(json.loads(p.read_text())))}


def check_environment() -> dict:
    """Quick report of what a work package needs: interpreters, disk, virgil pin. For ``README`` troubleshooting."""
    free = int(_run(["df", "-g" if sys.platform == "darwin" else "-BG", "/"]).splitlines()[-1].split()[3].rstrip("G"))
    return {
        "bench_python": os.environ.get("BENCH_PYTHON"), "ehtim_python": os.environ.get("EHTIM_PYTHON"),
        "virgil_pin": virgil_ref(), "free_disk_gb": free, "machine": machine_state(),
        "julia": _run(["julia", "--version"]).strip() or None,
    }


def _dummy_worker() -> None:
    """A tiny worker showing the inner protocol: calibration, import, load, cold, warm, one BENCH_INNER line."""
    sw = Stopwatch()
    sys.path.insert(0, str(HERE))
    from calib import calibrate

    with sw("t_calib"):
        calibrate(n_loop=200_000, n_rep=1)
    with sw("t_import"):
        import numpy as np
    with sw("t_load"):
        a = np.linspace(0.0, 1.0, 200_000)
    with sw("t_cold"):
        float(np.sum(a * a))
    for _ in range(3):
        with sw("t_warm", repeat=True):
            float(np.sum(a * a))
    emit_inner(sw, n_data=a.size, n_param=1, iterations={"calls": 4}, accuracy={"sum": float(np.sum(a * a))},
               settings={"dummy": True}, provenance=provenance())


def dummy_row(out_dir=None, wrapper=None) -> Path:
    """Run the dummy worker through ``timed_process`` and write a valid row (used by the tests and the README)."""
    res = timed_process([sys.executable, str(Path(__file__).resolve()), "--dummy-worker"], threads=1, timeout=100,
                        wrapper=wrapper)
    row = row_from_process(res, case="dummy", comparison="dummy", code="numpy", config="dummy", threads=1, size="tiny")
    return write_row(row, out_dir)


if __name__ == "__main__":
    if "--dummy-worker" in sys.argv:
        _dummy_worker()
    elif "--dummy-row" in sys.argv:
        i = sys.argv.index("--dummy-row")
        print(dummy_row(sys.argv[i + 1] if len(sys.argv) > i + 1 else None))
    else:
        print(json.dumps(check_environment(), indent=1))
