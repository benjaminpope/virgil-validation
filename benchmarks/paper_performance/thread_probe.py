"""Does a process stay on the stated thread count? Run on an IDLE machine before the timed rerun.

Each flag set runs a JAX compile + a few warm calls of a small FFT/matmul program in a fresh process (via
``timed_process``-style env from common.thread_env) and reports peak thread count, CPU seconds / wall seconds, and wall.
A set holds 1 thread if cpu/wall stays near 1 and the peak thread count is low and stable.

  ~/.claude/bin/pyslot $BENCH_PYTHON thread_probe.py            # prints a table, writes results/thread_probe.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import common

HERE = Path(__file__).resolve().parent
WORKER = r"""
import os, resource, subprocess, sys, time, json
import jax, jax.numpy as jnp
@jax.jit
def f(x):
    for _ in range(6):
        x = jnp.fft.ifft2(jnp.fft.fft2(x) * 0.99).real + 0.01 * (x @ x.T) / x.shape[0]
    return x.sum()
x = jnp.ones((256, 256))
t0 = time.perf_counter(); r0 = resource.getrusage(resource.RUSAGE_SELF)
f(x).block_until_ready()
for _ in range(20): f(x).block_until_ready()
wall = time.perf_counter() - t0; r1 = resource.getrusage(resource.RUSAGE_SELF)
cpu = (r1.ru_utime + r1.ru_stime) - (r0.ru_utime + r0.ru_stime)
nthr = int(subprocess.run(["ps", "-M", "-p", str(os.getpid())], capture_output=True, text=True).stdout.count("\n")) - 1
print("PROBE", json.dumps({"wall": wall, "cpu": cpu, "cpu_over_wall": cpu / wall, "threads": nthr}))
"""

SETS = {
    "legacy": lambda n: f"--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads={n}",
    "hold": lambda n: common.xla_flags(n, hold=True),
}


def probe(name: str, n: int = 1) -> dict:
    env = {**os.environ, **common.thread_env(n)}
    env["XLA_FLAGS"] = SETS[name](n)
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-c", WORKER], env=env, capture_output=True, text=True, timeout=300)
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("PROBE ")), None)
    out = json.loads(line[6:]) if line else {"error": p.stderr[-300:]}
    return {"set": name, "n": n, "proc_wall": time.perf_counter() - t0, **out}


def main():
    loadavg = os.getloadavg()[0]
    rows = [probe(s) for s in SETS for _ in range(3)]
    for r in rows:
        print(r)
    res = {"loadavg_1m_at_start": loadavg, "rows": rows,
           "note": "valid only if loadavg_1m_at_start is low (machine idle)"}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "thread_probe.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
