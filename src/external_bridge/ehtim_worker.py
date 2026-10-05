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


def reconstruct(task):
    """An eht-imaging reconstruction (Imager.make_image_I) from complex
    visibilities given row by row: time (one per snapshot and channel, so
    triangles close within a channel), stations t1, t2, (u, v) in
    wavelengths already in eht-imaging's sign convention, vis and sigma.

    Data terms: amplitudes and closure phases, weights alpha_amp and
    alpha_cphase; regularisers: squared TV (tv2) with weight beta_tv2,
    unnormalised (norm_reg=False), and the total flux held at 1 by a
    'flux' term. Direct Fourier transform, delta-function pixels. Returns
    the image (ny, nx; row 0 North, column 0 East) and the cost."""
    import contextlib
    import io

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
    with contextlib.redirect_stdout(io.StringIO()):
        obs = eh.obsdata.Obsdata(0.0, -50.0, 230e9, 1e9, table, tarr, polrep="stokes")
        init = eh.image.Image(np.asarray(task["init"], float), float(task["pdim"]), 0.0, -50.0,
                              rf=230e9, pulse=pulses.deltaPulse2D)
        imager = eh.imager.Imager(
            obs, init, prior_im=init, flux=1.0,
            data_term={"amp": float(task["alpha_amp"]), "cphase": float(task["alpha_cphase"])},
            reg_term={"tv2": float(task["beta_tv2"]), "flux": float(task.get("flux_weight", 1e4))},
            maxit=int(task.get("maxit", 2000)), stop=float(task.get("stop", 1e-12)),
            ttype="direct", norm_reg=False, transform=["log"], debias=False,
        )
        imager.make_image_I(grads=True, show_updates=False)
        out = imager.out_last()
        cost = float(imager.objfunc(np.log(out.imvec)))
        # the same cost at other images (row 0 North), e.g. virgil's
        other = [float(imager.objfunc(np.log(np.maximum(np.asarray(im, float).ravel(), 1e-300))))
                 for im in task.get("score", [])]
    n_amp = len(obs.unpack(["amp"]))
    n_cp = len(obs.c_phases(count="min"))
    return {"image": out.imarr().tolist(), "cost": cost, "scores": other, "n_amp": int(n_amp), "n_cphase": int(n_cp)}


TASKS = {"regularisers": regularisers, "ft": ft, "chisq_cphase": chisq_cphase, "reconstruct": reconstruct}

if __name__ == "__main__":
    task = json.load(open(sys.argv[1]))
    result = TASKS[task["task"]](task)
    import importlib.metadata as m

    result["ehtim_version"] = m.version("ehtim")
    json.dump(result, open(sys.argv[2], "w"))
