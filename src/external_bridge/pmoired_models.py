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
