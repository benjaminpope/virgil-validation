"""Run inside fouriever's own environment (.venv-fouriever), not ours.

    python fouriever_worker.py task.json result.json

fouriever (J. Kammerer; github.com/kammerje/fouriever) models the
correlation of closure phases that share baselines (Kammerer et al. 2020,
A&A 644, A110): intercorr.add_cpcov writes a CPCOV extension with
C = D^1/2 (T T^T / 3) D^1/2 per observation, and uvfit uses its
pseudo-inverse in the chi-squared. This worker reads our file with
fouriever's reader, adds that covariance, and evaluates util.chi2_bin.
Bandwidth smearing is off (fouriever's default).
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import warnings
from copy import deepcopy

import numpy as np
from scipy.linalg import block_diag

warnings.filterwarnings("ignore")

from fouriever import intercorr, util, uvfit  # noqa: E402


def _load(path, cov, observables=None):
    name = os.path.basename(path)
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        shutil.copy(path, os.path.join(tmp, name))
        if cov:
            intercorr.data(tmp, [name]).add_cpcov(os.path.join(tmp, "cov"))
            u = uvfit.data(os.path.join(tmp, "cov"), [name])
        else:
            u = uvfit.data(tmp, [name])
    if observables is not None:
        with contextlib.redirect_stdout(io.StringIO()):
            u.set_observables(list(observables))
    data = [d for obs in u.data_list for d in obs]
    for d in data:
        d["covflag"] = cov
        if cov:  # as uvfit's own methods set it up, for the observables used
            blocks = {"v2": np.diag(d["dv2"].flatten() ** 2), "cp": d["cpcov"]}
            d["cov"] = block_diag(*[blocks[o] for o in u.observables])
            d["icv"] = u.invert(d["cov"])
    return u, data


def chi2(task):
    """fouriever's chi-squared at each [f, dra, ddec] (f the linear flux
    ratio, positions in mas, East positive), with or without the
    closure-phase covariance; and what it read: the triangle matrix and
    covariances per observation."""
    cov = bool(task["cov"])
    u, data = _load(task["path"], cov)
    values = [float(util.chi2_bin(np.asarray(p, float), data, u.observables, cov=cov)) for p in task["params"]]
    return {
        "chi2": values,
        "observables": list(u.observables),
        "cpmat": data[0]["cpmat"].tolist(),
        "cpcov": [d["cpcov"].tolist() for d in data] if cov else None,
        "dcp": [d["dcp"].tolist() for d in data],
        "ndata": int(sum(d["v2"].size + d["cp"].size for d in data)),
    }


def _ndof(data, observables):
    return int(sum(d[o].size for d in data for o in observables))


def nsigma(task):
    """fouriever's util.nsigma (Absil et al. 2011, eq. 1) at each
    [chi2r_test, chi2r_true, ndof], with SciPy and with mpmath."""
    out = {}
    for mp in (False, True):
        out["mpmath" if mp else "scipy"] = [
            float(util.nsigma(chi2r_test=a, chi2r_true=b, ndof=int(n), use_mpmath=mp)[0]) for a, b, n in task["cases"]]
    return out


def limits_exact(task):
    """fouriever's own Absil and injection criteria (uvfit.lim_absil and
    lim_injection), each solved exactly by Brent's method rather than by
    detlim's grid start and L-BFGS-B, at each [dra, ddec].

    The primary is unresolved, so the uniform disk is held at 0 mas (detlim
    would fit it, to ~0, from V²): the null chi-squared is util.chi2_ud at
    diameter 0. Absil: nsigma(chi2_bin(f) / chi2_null) = sigma on the data.
    Injection (detlim's v2 branch with the diameter held): the companion is
    injected with uvfit.inj_companion, and nsigma(chi2_null(injected) /
    chi2_bin(injected, truth)) = sigma."""
    from scipy.optimize import brentq

    cov, sigma = bool(task["cov"]), float(task["sigma"])
    u, data = _load(task["path"], cov, task.get("observables"))
    obs = list(u.observables)
    ndof = int(task.get("ndof") or _ndof(data, obs))  # fouriever counts every observable
    null = float(util.chi2_ud(np.array([0.0]), data, obs, cov=cov))

    def absil(f, x, y):
        c = float(util.chi2_bin(np.array([f, x, y]), data, obs, cov=cov))
        return util.nsigma(chi2r_test=c / ndof, chi2r_true=null / ndof, ndof=ndof)[0] - sigma

    def injection(f, x, y):
        fit = {"p": np.array([f, x, y]), "model": "bin", "smear": None}
        with contextlib.redirect_stdout(io.StringIO()):
            inj = u.inj_companion(data_list=deepcopy(data), fit_inj=fit)
        c_null = float(util.chi2_ud(np.array([0.0]), inj, obs, cov=cov))
        c_bin = float(util.chi2_bin(np.array([f, x, y]), inj, obs, cov=cov))
        return util.nsigma(chi2r_test=c_null / ndof, chi2r_true=c_bin / ndof, ndof=ndof)[0] - sigma

    out = {"absil": [], "injection": [], "ndof": ndof, "chi2_null": null, "observables": obs}
    for x, y in task["positions"]:
        for name, fn in (("absil", absil), ("injection", injection)):
            out[name].append(float(brentq(fn, 1e-6, 0.5, args=(x, y), xtol=1e-12)))
    return out


def detlim(task):
    """fouriever's public uvfit.detlim, its two limit maps captured from the
    call it makes to plot.detlim (it returns only a figure)."""
    from fouriever import plot

    cov = bool(task["cov"])
    u, _ = _load(task["path"], cov, task.get("observables"))
    captured = {}

    def capture(ffs_absil, ffs_injection, *args, **kwargs):
        captured["absil"], captured["injection"] = np.asarray(ffs_absil), np.asarray(ffs_injection)

    plot.detlim, uvfit.plot.detlim = capture, capture
    with contextlib.redirect_stdout(io.StringIO()):
        u.detlim(sigma=float(task["sigma"]), cov=cov, sep_range=tuple(task["sep_range"]),
                 step_size=float(task["step_size"]))
    grid = util.get_grid(sep_range=tuple(task["sep_range"]), step_size=float(task["step_size"]), verbose=False)[0]
    return {"absil": np.nan_to_num(captured["absil"], nan=-1.0).tolist(),
            "injection": np.nan_to_num(captured["injection"], nan=-1.0).tolist(),
            "dra": np.nan_to_num(grid[0], nan=0.0).tolist(), "ddec": np.nan_to_num(grid[1], nan=0.0).tolist()}


def chi2_grid(task):
    """fouriever's chi2_bin over [dra, ddec] positions at one flux, with or
    without the closure-phase covariance (its chi2map's model, at fixed f)."""
    cov = bool(task["cov"])
    u, data = _load(task["path"], cov)
    f = float(task["flux"])
    return {"chi2": [float(util.chi2_bin(np.array([f, x, y]), data, u.observables, cov=cov))
                     for x, y in task["positions"]],
            "chi2_null": float(util.chi2_ud(np.array([0.0]), data, u.observables, cov=cov)),
            "ndof": _ndof(data, u.observables)}


TASKS = {"chi2": chi2, "nsigma": nsigma, "limits_exact": limits_exact, "detlim": detlim, "chi2_grid": chi2_grid}

if __name__ == "__main__":
    task = json.load(open(sys.argv[1]))
    result = TASKS[task["task"]](task)
    import importlib.metadata as m

    result["fouriever_version"] = m.version("fouriever")
    json.dump(result, open(sys.argv[2], "w"))
