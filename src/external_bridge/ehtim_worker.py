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


TASKS = {"regularisers": regularisers, "ft": ft, "chisq_cphase": chisq_cphase}

if __name__ == "__main__":
    task = json.load(open(sys.argv[1]))
    result = TASKS[task["task"]](task)
    import importlib.metadata as m

    result["ehtim_version"] = m.version("ehtim")
    json.dump(result, open(sys.argv[2], "w"))
