"""Evaluate PMOIRED models on our OIFITS files.

PMOIRED reads the file itself; we compare its model observables at given
parameters with the (noise-free) values we wrote, so any convention
mismatch shows up as a difference.
"""

import contextlib
import io

import numpy as np


def model_minus_data(path, params, setup=None):
    """Largest |model - data| for V^2 and closure phase (degrees, wrapped)
    over every baseline, triangle, epoch and channel, and the largest
    |closure phase| in the file (to judge how sensitive the test is).

    ``setup`` goes to ``OI.setupFit``; ``auto=False`` stops PMOIRED from
    setting a spectral ``wl kernel`` from our widely spaced channels,
    which otherwise makes every model NaN.
    """
    import pmoired
    import pmoired.oimodels as om

    with contextlib.redirect_stdout(io.StringIO()):
        oi = pmoired.OI(str(path), verbose=False)
        oi.setupFit({"obs": ["V2", "T3PHI"], **(setup or {})}, auto=False)
        model = om.VmodelOI(oi.data[0], params)
    data = oi.data[0]
    dv2 = max(
        np.max(np.abs(model["OI_VIS2"][k]["V2"] - data["OI_VIS2"][k]["V2"]))
        for k in data["OI_VIS2"]
    )
    dcp = max(
        np.max(
            np.abs(
                (model["OI_T3"][k]["T3PHI"] - data["OI_T3"][k]["T3PHI"] + 180)
                % 360
                - 180
            )
        )
        for k in data["OI_T3"]
    )
    cp_max = max(np.max(np.abs(data["OI_T3"][k]["T3PHI"])) for k in data["OI_T3"])
    return dv2, dcp, cp_max


def model_samples(path, params, setup=None):
    """PMOIRED's model on every sample of an OIFITS file, with the sample
    coordinates it used, so another code can be evaluated at exactly the
    same points.

    Returns a dict of flat arrays: ``u``, ``v`` (baseline / wavelength,
    in m/um, i.e. units of 1e6 / rad) and ``v2`` for the squared
    visibilities; ``u1``, ``v1``, ``u2``, ``v2_`` and ``t3phi`` (degrees)
    for the closure phases, whose third side is -(u1 + u2).
    """
    import pmoired
    import pmoired.oimodels as om

    with contextlib.redirect_stdout(io.StringIO()):
        oi = pmoired.OI(str(path), verbose=False)
        oi.setupFit({"obs": ["V2", "T3PHI"], **(setup or {})}, auto=False)
        model = om.VmodelOI(oi.data[0], params)
    wl = np.asarray(model["WL"])[None, :]  # um

    def per_wl(x):
        # (epochs,) metres -> (epochs, channels) in m/um
        return (np.asarray(x)[:, None] / wl).ravel()

    vis2 = list(model["OI_VIS2"].values())
    t3 = list(model["OI_T3"].values())
    return {
        "u": np.concatenate([per_wl(b["u"]) for b in vis2]),
        "v": np.concatenate([per_wl(b["v"]) for b in vis2]),
        "v2": np.concatenate([b["V2"].ravel() for b in vis2]),
        "u1": np.concatenate([per_wl(t["u1"]) for t in t3]),
        "v1": np.concatenate([per_wl(t["v1"]) for t in t3]),
        "u2": np.concatenate([per_wl(t["u2"]) for t in t3]),
        "v2_": np.concatenate([per_wl(t["v2"]) for t in t3]),
        "t3phi": np.concatenate([t["T3PHI"].ravel() for t in t3]),
    }


def fit(path, params, free, obs=("V2", "T3PHI"), setup=None):
    """PMOIRED's least-squares fit (``OI.doFit``) of ``params`` to a file,
    varying only ``free``.

    Returns ``best`` and ``sigma`` (dicts over ``free``), ``cov`` (matrix in
    the order of ``free``) and ``chi2_red``. PMOIRED reports uncertainties
    "normalized", i.e. multiplied by sqrt(reduced chi^2); ``sigma`` and
    ``cov`` here undo that, so they are the plain curvature errors that
    virgil's Laplace covariance also gives. ``sigma_reported`` keeps
    PMOIRED's own numbers.
    """
    import pmoired

    with contextlib.redirect_stdout(io.StringIO()):
        oi = pmoired.OI(str(path), verbose=False)
        oi.setupFit({"obs": list(obs), **(setup or {})}, auto=False)
        oi.doFit(
            dict(params),
            doNotFit=[k for k in params if k not in free],
            verbose=0,
        )
    b = oi.bestfit
    chi2 = float(b["chi2"])
    scale = np.sqrt(chi2) if b.get("normalized uncertainties", False) else 1.0
    cov = np.array([[b["covd"][i][j] for j in free] for i in free]) / scale**2
    return {
        "best": {k: float(b["best"][k]) for k in free},
        "sigma": {k: float(b["uncer"][k]) / scale for k in free},
        "sigma_reported": {k: float(b["uncer"][k]) for k in free},
        "cov": cov,
        "chi2_red": chi2,
    }
