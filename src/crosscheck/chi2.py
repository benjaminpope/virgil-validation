"""Chi-squared of a visibility function against one of our OIFITS files.

Read directly with astropy (not with virgil's reader): V² as Gaussian
residuals, closure phases as the chord 2 sin(Δ/2) / σ, the residual
virgil documents for unprojected phases. Closure phases are taken as
independent, which is exact for a three-telescope array (one triangle per
snapshot); use such files for comparisons.
"""

import numpy as np
from astropy.io import fits


def load(path):
    with fits.open(path) as h:
        wl = np.asarray(h["OI_WAVELENGTH"].data["EFF_WAVE"], float)
        v2 = h["OI_VIS2"].data
        out = {
            "u": np.repeat(v2["UCOORD"][:, None], wl.size, 1),
            "v": np.repeat(v2["VCOORD"][:, None], wl.size, 1),
            "wl": np.broadcast_to(wl, v2["VIS2DATA"].shape),
            "v2": np.asarray(v2["VIS2DATA"], float),
            "dv2": np.asarray(v2["VIS2ERR"], float),
        }
        if "OI_T3" in [x.name for x in h]:
            t3 = h["OI_T3"].data
            shape = t3["T3PHI"].shape
            out.update(
                u1=np.repeat(t3["U1COORD"][:, None], wl.size, 1),
                v1=np.repeat(t3["V1COORD"][:, None], wl.size, 1),
                u2=np.repeat(t3["U2COORD"][:, None], wl.size, 1),
                v2_=np.repeat(t3["V2COORD"][:, None], wl.size, 1),
                wl3=np.broadcast_to(wl, shape),
                cp=np.deg2rad(np.asarray(t3["T3PHI"], float)),
                dcp=np.deg2rad(np.asarray(t3["T3PHIERR"], float)),
            )
    return out


def n_data(d):
    return d["v2"].size + (d["cp"].size if "cp" in d else 0)


def chi2(d, vis):
    """Chi-squared of ``vis(u, v, wavel)`` against the loaded data."""
    model_v2 = np.abs(vis(d["u"], d["v"], d["wl"])) ** 2
    total = np.sum(((model_v2 - d["v2"]) / d["dv2"]) ** 2)
    if "cp" in d:
        t3 = (
            vis(d["u1"], d["v1"], d["wl3"])
            * vis(d["u2"], d["v2_"], d["wl3"])
            * np.conj(vis(d["u1"] + d["u2"], d["v1"] + d["v2_"], d["wl3"]))
        )
        chord = 2.0 * np.sin((np.angle(t3) - d["cp"]) / 2.0) / d["dcp"]
        total += np.sum(chord**2)
    return float(total)
