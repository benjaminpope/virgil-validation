"""Run inside eht-imaging's own environment (.venv-ehtim), not ours.

    python ehtim_worker.py task.json result.json

Evaluates eht-imaging's regularisers (ehtim.imaging.imager_utils) and their
hand-written gradients on given images, with its normalisation switched off
(norm_reg=False), so values are the bare sums.
"""

import json
import sys

import numpy as np
from ehtim.imaging import imager_utils as iu


def regularisers(task):
    """For each image (ny, nx): simple entropy against a prior, TV with
    softening epsilon, squared TV; values and gradients with respect to the
    pixel values."""
    out = []
    for im, prior in zip(task["images"], task["priors"]):
        im, prior = np.asarray(im, float), np.asarray(prior, float)
        ny, nx = im.shape
        vec, pvec = im.ravel(), prior.ravel()
        eps = float(task["tv_epsilon"])
        pos = vec > 0  # entropy is defined on positive pixels
        out.append({
            "simple": float(iu.ssimple(vec[pos], pvec[pos], 1.0, norm_reg=False)),
            "simple_grad": _scatter(iu.ssimplegrad(vec[pos], pvec[pos], 1.0, norm_reg=False), pos, ny, nx),
            "tv": float(iu.stv(vec, nx, ny, 1.0, 1.0, norm_reg=False, epsilon=eps)),
            "tv_grad": iu.stvgrad(vec, nx, ny, 1.0, 1.0, norm_reg=False, epsilon=eps).reshape(ny, nx).tolist(),
            "tv2": float(iu.stv2(vec, nx, ny, 1.0, 1.0, norm_reg=False)),
            "tv2_grad": iu.stv2grad(vec, nx, ny, 1.0, 1.0, norm_reg=False).reshape(ny, nx).tolist(),
        })
    return {"results": out}


def _scatter(values, mask, ny, nx):
    full = np.zeros(ny * nx)
    full[mask] = values
    return full.reshape(ny, nx).tolist()


def ft(task):
    """eht-imaging's direct Fourier transform (obs_helpers.ftmatrix) of an
    image (ny, nx; row 0 and column 0 as eht-imaging stores them) with
    pixel size pdim (rad) at spatial frequencies uv (wavelengths), with the
    delta-function pixel response (no pixel smoothing)."""
    from ehtim.observing import obs_helpers, pulses

    im = np.asarray(task["image"], float)
    ny, nx = im.shape
    A = obs_helpers.ftmatrix(float(task["pdim"]), nx, ny, np.asarray(task["uv"], float),
                             pulse=pulses.deltaPulse2D)
    vis = A @ im.ravel()
    return {"re": vis.real.tolist(), "im": vis.imag.tolist()}


def chisq_cphase(task):
    """eht-imaging's closure-phase chi-squared (imager_utils.chisq_cphase,
    (2/N) sum (1 - cos(data - model)) / sigma^2) of an image, for triangles
    given by their first two baselines (u1, v1), (u2, v2) in wavelengths,
    OIFITS convention, closure phases and errors in degrees.

    eht-imaging's transform uses exp(+2 pi i (u x + v y)); for a real image
    that is the OIFITS visibility at (-u, -v), so the baselines are negated
    here, and the triangle closes with (u3, v3) = -(u1 + u2, v1 + v2)."""
    from ehtim.imaging import imager_utils as iu
    from ehtim.observing import obs_helpers, pulses

    im = np.asarray(task["image"], float)
    ny, nx = im.shape
    uv1 = -np.asarray(task["uv1"], float)
    uv2 = -np.asarray(task["uv2"], float)
    uv3 = -uv1 - uv2
    A = [obs_helpers.ftmatrix(float(task["pdim"]), nx, ny, uv, pulse=pulses.deltaPulse2D) for uv in (uv1, uv2, uv3)]
    cp, sigma = np.asarray(task["cp_deg"], float), np.asarray(task["sigma_deg"], float)
    value = iu.chisq_cphase(im.ravel(), A, cp, sigma)
    return {"chisq": float(value), "n": int(cp.size)}


class _ObjectArrayNumpy:
    """NumPy, except that ``array(list_of_arrays, dtype=object)`` is always
    1-D. eht-imaging's Obsdata.tlist and bllist build their groups that way;
    under NumPy 2, when every group has the same length the result is 2-D
    and its records become plain tuples, so closure phases fail with
    ``TypeError: tuple indices must be integers or slices, not str``. That
    happens whenever every time has the same baselines, as in simulated
    data (problem P6). Installed only in ehtim.obsdata, only here."""

    def __getattr__(self, name):
        return getattr(np, name)

    def array(self, obj, *args, **kwargs):
        if kwargs.get("dtype") is object and isinstance(obj, list) and obj and all(
            isinstance(x, np.ndarray) for x in obj
        ):
            out = np.empty(len(obj), dtype=object)
            for i, x in enumerate(obj):
                out[i] = x
            return out
        return np.array(obj, *args, **kwargs)


def _imager(task, init_image, flux_weight):
    """An eht-imaging Imager for complex visibilities given row by row: time
    (one per snapshot and channel, so triangles close within a channel),
    stations t1, t2, (u, v) in wavelengths already in eht-imaging's sign
    convention, vis and sigma.

    Data terms: amplitudes and closure phases, weights alpha_amp and
    alpha_cphase; regularisers: squared TV (tv2) with weight beta_tv2,
    unnormalised (norm_reg=False), and the total flux held at 1 by a
    'flux' term of weight flux_weight. Direct Fourier transform,
    delta-function pixels, log-pixel optimisation."""
    import ehtim as eh
    from ehtim import const_def as ehc
    from ehtim import obsdata as eh_obsdata
    from ehtim.observing import pulses

    eh_obsdata.np = _ObjectArrayNumpy()  # P6 workaround

    rows = task["rows"]
    sites = sorted({r[1] for r in rows} | {r[2] for r in rows})
    tarr = np.zeros(len(sites), dtype=ehc.DTARR)
    tarr["site"] = sites
    tarr["x"] = 1.0 + np.arange(len(sites))  # positions unused: (u, v) are given
    tarr["sefdr"] = tarr["sefdl"] = 1.0
    table = np.zeros(len(rows), dtype=ehc.DTPOL_STOKES)
    for k, (time, t1, t2, u, v, re, im, sigma) in enumerate(rows):
        table[k]["time"], table[k]["tint"] = time, 1.0
        table[k]["t1"], table[k]["t2"] = t1, t2
        table[k]["u"], table[k]["v"] = u, v
        table[k]["vis"] = re + 1j * im
        for s in ("sigma", "qsigma", "usigma", "vsigma"):
            table[k][s] = sigma
    obs = eh.obsdata.Obsdata(0.0, -50.0, 230e9, 1e9, table, tarr, polrep="stokes")
    init = eh.image.Image(np.asarray(init_image, float), float(task["pdim"]), 0.0, -50.0,
                          rf=230e9, pulse=pulses.deltaPulse2D)
    imager = eh.imager.Imager(
        obs, init, prior_im=init, flux=1.0,
        data_term={"amp": float(task["alpha_amp"]), "cphase": float(task["alpha_cphase"])},
        reg_term={"tv2": float(task["beta_tv2"]), "flux": float(flux_weight)},
        maxit=int(task.get("maxit", 2000)), stop=float(task.get("stop", 1e-12)),
        ttype="direct", norm_reg=False, transform=["log"], debias=False,
    )
    return obs, imager


def _counts(obs):
    return int(len(obs.unpack(["amp"]))), int(len(obs.c_phases(count="min")))


def objective(task):
    """eht-imaging's cost and its terms at each given image (ny, nx; row 0
    North, any total flux). Unweighted: chi2_amp and chi2_cphase as
    imager_utils defines them (normalised by the number of data) and the
    'tv2' and 'flux' regulariser values (signs as they enter the cost).
    Weighted: the total cost (Imager.objfunc) and its gradient with respect
    to log pixel values (Imager.objgrad)."""
    import contextlib
    import io

    images = [np.asarray(im, float) for im in task["images"]]
    out = []
    with contextlib.redirect_stdout(io.StringIO()):
        obs, imager = _imager(task, images[0], task.get("flux_weight", 1e4))
        imager.check_params()
        imager.check_limits()
        imager.init_imager()
        for im in images:
            vec = im.ravel()
            chi2 = imager.make_chisq_dict(vec)
            reg = imager.make_reg_dict(vec)
            x = np.log(vec)
            out.append({
                "chi2_amp": float(chi2["amp"]), "chi2_cphase": float(chi2["cphase"]),
                "tv2": float(reg["tv2"]), "flux": float(reg["flux"]),
                "cost": float(imager.objfunc(x)), "grad_log": imager.objgrad(x).reshape(im.shape).tolist(),
            })
        n_amp, n_cp = _counts(obs)
    return {"results": out, "n_amp": n_amp, "n_cphase": n_cp}


def reconstruct(task):
    """An eht-imaging reconstruction (Imager.make_image_I, L-BFGS-B) from
    the image init, run to convergence in stages: for each flux weight in
    flux_weights (a continuation that pins the total flux to 1), restart
    from the last image until a run ends without improving the cost (at
    most max_restarts runs per stage); the stage's best image starts the
    next. Returns the final stage's best image (ny, nx; row 0 North, column
    0 East), its cost, total flux and the norm of the cost's gradient with
    respect to log pixels, and a log of every run."""
    import contextlib
    import io

    flux_weights = task.get("flux_weights", [task.get("flux_weight", 1e4)])
    if not flux_weights:
        raise ValueError("flux_weights must name at least one flux weight")
    image = np.asarray(task["init"], float)
    log = []
    with contextlib.redirect_stdout(io.StringIO()):
        for mu in flux_weights:
            best = None
            for _ in range(int(task.get("max_restarts", 1))):
                obs, imager = _imager(task, image, mu)
                imager.make_image_I(grads=True, show_updates=False)
                trial = imager.out_last().imarr()
                x = np.log(trial.ravel())
                cost = float(imager.objfunc(x))
                gnorm = float(np.linalg.norm(imager.objgrad(x)))
                log.append({"flux_weight": mu, "cost": cost, "grad_log_norm": gnorm, "flux": float(trial.sum())})
                if best is not None and cost >= best[1]:
                    break
                best = (trial, cost, gnorm)
                image = trial
            image, cost, gnorm = best  # the best run of this stage starts the next
        other = [float(imager.objfunc(np.log(np.maximum(np.asarray(im, float).ravel(), 1e-300))))
                 for im in task.get("score", [])]
        n_amp, n_cp = _counts(obs)
    return {"image": image.tolist(), "cost": cost, "grad_log_norm": gnorm, "flux": float(image.sum()),
            "log": log, "scores": other, "n_amp": n_amp, "n_cphase": n_cp}


def p6_probe(task):
    """P6 reproducer, with eht-imaging unpatched: three sites at two times
    (equal-sized time groups); can Obsdata.c_phases run?"""
    import contextlib
    import io

    import ehtim as eh
    from ehtim import const_def as ehc

    tarr = np.zeros(3, dtype=ehc.DTARR)
    tarr["site"] = ["A", "B", "C"]
    tarr["x"] = [1.0, 2.0, 3.0]
    tarr["sefdr"] = tarr["sefdl"] = 1.0
    rows = [(t, a, b) for t in (0.0, 1.0) for a, b in (("A", "B"), ("A", "C"), ("B", "C"))]
    table = np.zeros(len(rows), dtype=ehc.DTPOL_STOKES)
    for k, (time, a, b) in enumerate(rows):
        table[k]["time"], table[k]["tint"], table[k]["t1"], table[k]["t2"] = time, 1.0, a, b
        table[k]["u"], table[k]["v"] = 1e6 * (k + 1), 2e6
        table[k]["vis"], table[k]["sigma"] = 0.5 + 0.1j, 0.01
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            obs = eh.obsdata.Obsdata(0.0, -50.0, 230e9, 1e9, table, tarr, polrep="stokes")
            n = len(obs.c_phases(count="min"))
        return {"ok": True, "n_cphase": int(n), "error": None}
    except Exception as e:  # noqa: BLE001 - report what upstream raises
        return {"ok": False, "n_cphase": 0, "error": f"{type(e).__name__}: {e}"}


TASKS = {"regularisers": regularisers, "ft": ft, "chisq_cphase": chisq_cphase, "objective": objective,
         "reconstruct": reconstruct,
         "p6_probe": p6_probe}

if __name__ == "__main__":
    task = json.load(open(sys.argv[1]))
    result = TASKS[task["task"]](task)
    import importlib.metadata as m

    result["ehtim_version"] = m.version("ehtim")
    json.dump(result, open(sys.argv[2], "w"))
