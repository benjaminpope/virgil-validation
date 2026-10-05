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

import numpy as np
from scipy.linalg import block_diag

warnings.filterwarnings("ignore")

from fouriever import intercorr, util, uvfit  # noqa: E402


def _load(path, cov):
    name = os.path.basename(path)
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        shutil.copy(path, os.path.join(tmp, name))
        if cov:
            intercorr.data(tmp, [name]).add_cpcov(os.path.join(tmp, "cov"))
            u = uvfit.data(os.path.join(tmp, "cov"), [name])
        else:
            u = uvfit.data(tmp, [name])
    data = [d for obs in u.data_list for d in obs]
    for d in data:
        d["covflag"] = cov
        if cov:  # as uvfit's own methods set it up
            d["cov"] = block_diag(np.diag(d["dv2"].flatten() ** 2), d["cpcov"])
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


TASKS = {"chi2": chi2}

if __name__ == "__main__":
    task = json.load(open(sys.argv[1]))
    result = TASKS[task["task"]](task)
    import importlib.metadata as m

    result["fouriever_version"] = m.version("fouriever")
    json.dump(result, open(sys.argv[2], "w"))
