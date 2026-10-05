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


TASKS = {"regularisers": regularisers}

if __name__ == "__main__":
    task = json.load(open(sys.argv[1]))
    result = TASKS[task["task"]](task)
    import importlib.metadata as m

    result["ehtim_version"] = m.version("ehtim")
    json.dump(result, open(sys.argv[2], "w"))
