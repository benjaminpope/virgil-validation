"""BASIS (Lai et al. 2026) through its own environment: thin wrappers over ``basis_worker`` (``.venv-basis``).

Each function runs one task of ``basis_worker.py`` with ``_subprocess.run("basis", ...)`` and returns its JSON.
Imports neither BASIS nor virgil.
"""

from external_bridge import _subprocess

WORKER = "basis_worker.py"


def available():
    """Whether ``.venv-basis`` (``scripts/setup_m87.sh basis``) can import BASIS."""
    return _subprocess.available("basis", "BASIS")


def parameters(models=None):
    """BASIS's parameter table of the model list (default xsringauss, gauss, gauss)."""
    return _subprocess.run("basis", WORKER, {"task": "parameters", **({"models": models} if models else {})})


def invariants(uvfits, params=(), noise_frac=0.0, floor=0.0, noise_factor=1.0, seed=0, models=None, double=False,
               timeout=7200):
    """Closure-invariant data, sigma, triangle labels, alignment check and the model invariants at each parameter
    dict in ``params`` (BASIS names such as ``"0_Rp"``; unspecified ones keep BASIS's defaults). ``double`` sets
    torch's default dtype to float64 first (BASIS's models are float32 by default)."""
    task = {"task": "invariants", "uvfits": str(uvfits), "params": [dict(p) for p in params], "noise_frac": noise_frac,
            "floor": floor, "noise_factor": noise_factor, "seed": seed, "double": bool(double)}
    if models:
        task["models"] = models
    return _subprocess.run("basis", WORKER, task, timeout=timeout)


def render_check(params=None, dim=128, fov=225.0):
    """Orientation of BASIS's ``sky_map`` against its analytic visibilities and the floor Gaussian's side."""
    return _subprocess.run("basis", WORKER, {"task": "render_check", "params": params or {}, "dim": dim, "fov": fov})


def posterior_renders(result_json, out_npz, n=200, dim=128, fov=225.0, seed=0, transform=None, timeout=3600):
    """Render ``n`` posterior samples of a bilby result JSON to ``out_npz`` (``native``, ``sky`` arrays)."""
    task = {"task": "posterior_renders", "result": str(result_json), "out": str(out_npz), "n": n, "dim": dim,
            "fov": fov, "seed": seed}
    if transform:
        task["transform"] = transform
    return _subprocess.run("basis", WORKER, task, timeout=timeout)


def dump_uvfits(uvfits, out_npz):
    """eht-imaging's reading of ``uvfits`` as arrays in ``out_npz`` (time, station indices, u, v, vis, sigma)."""
    return _subprocess.run("basis", WORKER, {"task": "dump_uvfits", "uvfits": str(uvfits), "out": str(out_npz)})
