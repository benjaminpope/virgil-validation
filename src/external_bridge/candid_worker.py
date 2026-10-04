"""Run inside CANDID's own environment (.venv-candid), not ours.

    python candid_worker.py task.json result.json

The task is a JSON object with "task" and its arguments; the result is
written as JSON. Only CANDID, NumPy and the standard library are used.

Settings common to every task, so CANDID computes what virgil documents:

* observables V² and closure phase (CANDID also uses T3 amplitudes by
  default; virgil does not), or V² alone with "observables": ["v2"];
* no bandwidth smearing: every instrument's channel width set to zero
  (CANDID then averages identical monochromatic models);
* one core, no progress bar, no run-time guard.
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")

import candid  # noqa: E402
import numpy as np  # noqa: E402

candid.CONFIG.update({"Ncores": 1, "long exec warning": None, "progress bar": False})


def _open(task):
    o = candid.Open(task["path"], rmin=task.get("rmin"), rmax=task.get("rmax"))
    o.observables = list(task.get("observables", ["v2", "cp"]))
    o.dwavel = {k: 0.0 for k in o.dwavel}
    return o


def _params(o, p):
    out = {"x": 0.0, "y": 0.0, "f": 0.0, "diam*": 0.0, "alpha*": 0.0, **p}
    for k in o.dwavel:
        out["dwavel;" + k] = 0.0
    return out


def chi2(task):
    """CANDID's reduced chi-squared (mean of squared normalised residuals)
    at each parameter set: x, y (mas), f (percent of the primary), diam*
    (the primary's uniform-disk diameter, mas)."""
    o = _open(task)
    values = [
        float(candid._chi2Func(_params(o, p), o._chi2Data, o.observables, o.instruments))
        for p in task["params"]
    ]
    return {"chi2r": values, "ndata": o.ndata()}


def nsigma(task):
    """candid._nSigmas on given chi-squared ratios."""
    return {"nsigma": [float(candid._nSigmas(r, 1.0, n)) for r, n in task["cases"]]}


def _grid(rmax, step):
    n = int(np.ceil(2 * rmax / step))
    return np.linspace(-rmax, rmax, n)


def chi2map(task):
    """CANDID's chi-squared map at fixed flux ratio fratio (percent). It
    fits the primary's diameter first and maps chi2r of the binary."""
    o = _open(task)
    o.chi2Map(step=task["step"], fratio=task["fratio"], rmin=task["rmin"], rmax=task["rmax"], fig=None)
    axis = _grid(task["rmax"], task["step"])
    return {
        "x": axis.tolist(),
        "y": axis.tolist(),
        "chi2r": o.mapChi2.tolist(),  # [y, x]
        "chi2r_ud": float(o.chi2_UD),
        "diam": float(o.diam),
        "ndata": o.ndata(),
    }


def limits(task):
    """CANDID's 3-sigma detection limits (percent) by the Absil method at
    a fixed primary diameter."""
    o = _open(task)
    o.detectionLimit(
        step=task["step"], diam=float(task["diam"]), fig=None, drawMaps=False,
        rmin=task["rmin"], rmax=task["rmax"], methods=task.get("methods", ["Absil"]),
    )
    axis = _grid(task["rmax"], task["step"])
    return {
        "x": axis.tolist(),
        "y": axis.tolist(),
        "f3s": {m: v.tolist() for m, v in o.allf3s.items()},  # [y, x], percent
        "rmin": task["rmin"],
        "rmax": task["rmax"],
    }


def fitmap(task):
    """CANDID's fitMap: least-squares fits of x, y, f and the primary's
    diameter started on a grid; the best one. CANDID's uncertainties are
    scaled by sqrt(chi2r), with chi2r = chi2 / (N - n_fit + 1); both are
    returned so the caller can undo the scaling."""
    o = _open(task)
    o.fitMap(step=task["step"], rmin=task["rmin"], rmax=task["rmax"], fratio=task.get("fratio", 2.0), fig=None)
    b = o.bestFit
    keys = ["x", "y", "f", "diam*"]
    return {
        "best": {k: float(b["best"][k]) for k in keys},
        "uncer": {k: float(b["uncer"][k]) for k in keys},
        "chi2r": float(b["chi2"]),
        "nsigma": float(b["nsigma"]),
        "ndata": o.ndata(),
    }


def absil_exact(task):
    """CANDID's own Absil criterion, _nSigmas(chi2(f) / chi2(0)) = 3 with
    CANDID's chi-squared and number of data points, solved exactly (Brent)
    at each position instead of by CANDID's bracketing and linear
    interpolation. Separates CANDID's definitions from its numerics."""
    from scipy import optimize

    o = _open(task)
    ndata = sum(c[-1].size for c in o._chi2Data if c[0].split(";")[0] in o.observables)
    base = {"diam*": float(task["diam"])}
    null = candid._chi2Func(_params(o, base), o._chi2Data, o.observables, o.instruments)
    sigma = task.get("sigma", 3.0)

    def excess(f, x, y):
        c = candid._chi2Func(_params(o, {**base, "x": x, "y": y, "f": f}), o._chi2Data, o.observables, o.instruments)
        return candid._nSigmas(c, null, ndata) - sigma

    out = [optimize.brentq(excess, 1e-4, 100.0, args=(x, y), xtol=1e-10) for x, y in task["positions"]]
    return {"f3": out, "ndata": int(ndata)}  # percent


TASKS = {"chi2": chi2, "nsigma": nsigma, "chi2map": chi2map, "limits": limits, "absil_exact": absil_exact, "fitmap": fitmap}

if __name__ == "__main__":
    task = json.load(open(sys.argv[1]))
    result = TASKS[task["task"]](task)
    result["candid_version"] = candid.__version__
    json.dump(result, open(sys.argv[2], "w"))
