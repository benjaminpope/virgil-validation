"""Toy stand-ins for the NIRCam steps. Synthetic numbers; the shape of the workflow is the point.

Each step reads its work order (units, decisions, upstream run folders) and writes
``result.json`` with a status and named checks. A real step is the same, with a real fit inside.
"""

import argparse
import hashlib
import json
import pathlib

import numpy as np

N_TRUE = 8


def noise(name: str, n: int) -> np.ndarray:
    """Deterministic pseudo-noise from a name, in [-1, 1]."""
    raw = hashlib.sha256(name.encode()).digest()
    return np.array([raw[i] / 127.5 - 1.0 for i in range(n)])


def truth(unit: dict) -> np.ndarray:
    """A wavefront that drifts slowly in time, differs by filter, and is wrong for a binary."""
    base = noise(unit["filter"], N_TRUE)
    drift = noise("drift", N_TRUE) * float(unit["t_min"]) / 300.0
    binary = 3.0 * noise("binary", N_TRUE) if unit["target_name"] == "TOY-REF-D" else 0.0
    return base + drift + binary + 0.05 * noise(unit["obs_id"], N_TRUE)


def write(out: pathlib.Path, status: str, checks: dict, **numbers) -> None:
    result = {"status": status, "checks": [{"name": k, "status": v} for k, v in checks.items()], **numbers}
    (out / "result.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")


def wavefront(order: dict, out: pathlib.Path, modes: int) -> None:
    unit = order["units"][0]
    coeffs = truth(unit)[:modes]
    # The toy's one idiosyncrasy: TOY-REF-E is a short exposure that cannot constrain six modes.
    converged = not (unit["target_name"] == "TOY-REF-E" and modes > 4)
    rms = float(np.sqrt(np.mean(truth(unit)[modes:] ** 2))) if modes < N_TRUE else 0.0
    np.save(out / "coeffs.npy", coeffs)
    checks = {"fit_converged": "pass" if converged else "fail", "residual_rms": "pass" if rms < 1.5 else "warn"}
    write(out, "pass" if converged else "fail", checks, n_modes=modes, residual_rms=round(rms, 4))


def members(order: dict, stage: str) -> list[np.ndarray]:
    return [np.load(pathlib.Path(m["path"]) / "coeffs.npy") for m in order["inputs"].get(stage, [])]


def basis(order: dict, out: pathlib.Path, components: int) -> None:
    fits = members(order, "wavefront")
    width = min(len(f) for f in fits)
    stack = np.array([f[:width] for f in fits])
    mean = stack.mean(axis=0)
    _, s, vt = np.linalg.svd(stack - mean, full_matrices=False)
    np.save(out / "mean.npy", mean)
    np.save(out / "components.npy", vt[:components])
    explained = float((s[:components] ** 2).sum() / (s**2).sum())
    write(out, "pass", {"enough_members": "pass" if len(fits) >= 3 else "fail"},
          n_members=len(fits), explained_variance=round(explained, 4))


def science(order: dict, out: pathlib.Path, kind: str) -> None:
    unit = order["units"][0]
    base = pathlib.Path(order["inputs"]["basis"][0]["path"])
    mean, comps = np.load(base / "mean.npy"), np.load(base / "components.npy")
    width = len(mean)
    local = np.mean([f[:width] for f in members(order, "wavefront")], axis=0)
    target = truth(unit)[:width]
    resid = (target - mean) - comps.T @ (comps @ (target - mean))
    rms = float(np.sqrt(np.mean(resid**2)))
    local_rms = float(np.sqrt(np.mean((target - local) ** 2)))
    numbers = {"psf_residual_rms": round(rms, 4), "nearest_reference_rms": round(local_rms, 4)}
    if kind == "planet":
        numbers["separation_mas"] = round(400.0 + 20.0 * abs(noise(unit["target_name"], 1)[0]) + 10.0 * rms, 1)
        write(out, "pass", {"injection_recovery": "pass" if rms < 1.0 else "fail"}, **numbers)
    else:
        write(out, "pass", {"residual_structure": "pass" if rms < 1.0 else "warn"}, **numbers)


def score(order: dict, out: pathlib.Path) -> None:
    unit = order["units"][0]
    found = json.loads((pathlib.Path(order["inputs"]["planet_search"][0]["path"]) / "result.json").read_text())
    published = float(unit["published_sep_mas"])  # a sealed column: only this stage is given it
    offset = found["separation_mas"] - published
    write(out, "pass" if abs(offset) < 30 else "fail", {"within_tolerance": "pass" if abs(offset) < 30 else "fail"},
          offset_mas=round(offset, 1))


def catalogue(order: dict, out: pathlib.Path) -> None:
    rows = []
    for stage in ("planet_search", "disk_deconvolution"):
        for m in order["inputs"].get(stage, []):
            result = json.loads((pathlib.Path(m["path"]) / "result.json").read_text())
            rows.append({"order": m["order"], "stage": stage, "status": result["status"],
                         "psf_residual_rms": result["psf_residual_rms"]})
    (out / "catalogue.json").write_text(json.dumps(rows, indent=1) + "\n")
    write(out, "pass", {}, n_results=len(rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["wavefront", "basis", "planet", "disk", "score", "catalogue"])
    p.add_argument("--order", required=True)
    p.add_argument("--modes", type=int, default=6)
    p.add_argument("--components", type=int, default=2)
    a = p.parse_args()
    order = json.loads(pathlib.Path(a.order).read_text())
    out = pathlib.Path(order["out"])
    if a.step == "wavefront":
        wavefront(order, out, a.modes)
    elif a.step == "basis":
        basis(order, out, a.components)
    elif a.step in ("planet", "disk"):
        science(order, out, a.step)
    elif a.step == "score":
        score(order, out)
    else:
        catalogue(order, out)
