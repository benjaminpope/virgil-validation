"""Evaluate ``TruncatedCone`` at fixed parameters and several ``n_rings``, in whichever virgil
is installed in the interpreter running this script (the pinned old commit a1dae85 in
``.venv-old``, or the new one). Writes an .npz of complex visibilities and per-call times.

    .venv-old/bin/python benchmarks/polar_disk/cone_worker.py OUT.npz IN.npz
"""

import sys
import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
from importlib import metadata  # noqa: E402

import virgil  # noqa: E402

CONES = {
    "plain": dict(tip=4.0, alpha=35.0, s0=3.0, length=5.0, width=1.0, tilt=40.0, pa=60.0, ratio=1.0, flux=1.0, dra=0.3, ddec=-0.2),
    "modulated": dict(tip=4.0, alpha=35.0, s0=3.0, length=5.0, width=1.0, tilt=40.0, pa=60.0, ratio=1.0, flux=1.0, dra=0.3, ddec=-0.2,
                      az_amps=(0.4, 0.2), az_pas=(120.0, 30.0)),
}


def main(out, inp):
    d = np.load(inp)
    u, v, wl = d["u"], d["v"], float(d["wl"])
    res = {"virgil_version": np.array(metadata.version("virgil-astro"))}
    for name, kw in CONES.items():
        for n in d["n_rings"]:
            cone = virgil.models.TruncatedCone(n_rings=int(n), **kw)
            f = jax.jit(lambda uu, vv, c=cone: c.model(uu, vv, wl))
            val = np.asarray(f(u, v))
            t0 = time.perf_counter()
            for _ in range(5):
                f(u, v).block_until_ready()
            res[f"{name}_{int(n)}"] = val
            res[f"{name}_{int(n)}_time"] = np.array((time.perf_counter() - t0) / 5)
    np.savez(out, **res)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
