"""Chi-squared of a visibility function against one of our OIFITS files.

Read directly with astropy (not with virgil's reader): V² as Gaussian
residuals, closure phases as the chord 2 sin(Δ/2) / σ, the residual
virgil documents for unprojected phases. Closure phases are taken as
independent by default, which is exact for a three-telescope array (one
triangle per snapshot). With ``correlated=True`` they are not: closure
phases of one snapshot and channel that share baselines are correlated as
in Kammerer et al. (2020, A&A 644, A110, sec. 2.2), with covariance
C = D^1/2 (T T^T / 3) D^1/2 (T the triangle-by-baseline matrix of +1, +1,
-1, D the reported variances) and chi-squared r^T C^+ r.
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
            "v2_mjd": np.asarray(v2["MJD"], float),
            "v2_sta": np.asarray(v2["STA_INDEX"], int),
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
                t3_mjd=np.asarray(t3["MJD"], float),
                t3_sta=np.asarray(t3["STA_INDEX"], int),
            )
    return out


def n_data(d):
    return d["v2"].size + (d["cp"].size if "cp" in d else 0)


def triangle_matrix(stations):
    """T for triangles (a, b, c) given as station triples: rows +1 on ab,
    +1 on bc, -1 on ac (the closure phase phi_ab + phi_bc - phi_ac), over
    the baselines they use, each as a sorted station pair."""
    stations = [tuple(int(s) for s in tri) for tri in stations]
    baselines = sorted({tuple(sorted(p)) for a, b, c in stations for p in ((a, b), (b, c), (a, c))})
    index = {b: k for k, b in enumerate(baselines)}
    T = np.zeros((len(stations), len(baselines)))
    for row, (a, b, c) in enumerate(stations):
        for (i, j), sign in (((a, b), 1.0), ((b, c), 1.0), ((a, c), -1.0)):
            T[row, index[tuple(sorted((i, j)))]] += sign if i < j else -sign
    return T


def chi2(d, vis, correlated=False, chord=True, sine=False, whitened=False):
    """Chi-squared of ``vis(u, v, wavel)`` against the loaded data.

    Closure-phase residuals are chords 2 sin(delta/2) (virgil's) or, with
    ``chord=False``, plain, unwrapped differences between the sum of the
    model's baseline phases and the data; ``correlated`` as in the module
    docstring.

    With ``correlated=True``:

    * ``sine=True`` is virgil's continuous form (since virgil#174): the
      sines sin(delta) are correlated, and each closure phase adds an
      uncorrelated periodic penalty ((1 - cos delta) / sigma)^2;
    * ``whitened=True`` uses the generalised inverse D^-1/2 R^+ D^-1/2
      (residuals divided by sigma, then R's pseudo-inverse) instead of C^+;
      the two agree when a group's errors are equal."""
    model_v2 = np.abs(vis(d["u"], d["v"], d["wl"])) ** 2
    total = np.sum(((model_v2 - d["v2"]) / d["dv2"]) ** 2)
    if "cp" in d:
        # the model closure phase as the sum of the three baseline phases,
        # each in (-pi, pi], so it may lie beyond +-pi; the chord is 2 pi
        # periodic, the plain difference is not wrapped (as in CANDID and
        # fouriever)
        model = (
            np.angle(vis(d["u1"], d["v1"], d["wl3"]))
            + np.angle(vis(d["u2"], d["v2_"], d["wl3"]))
            - np.angle(vis(d["u1"] + d["u2"], d["v1"] + d["v2_"], d["wl3"]))
        )
        delta = model - d["cp"]
        r = 2.0 * np.sin(delta / 2.0) if chord else delta
        if correlated and sine:
            r = np.sin(delta)
            total += np.sum(((1.0 - np.cos(delta)) / d["dcp"]) ** 2)
        if not correlated:
            total += np.sum((r / d["dcp"]) ** 2)
        else:
            for mjd in np.unique(d["t3_mjd"]):
                rows = np.flatnonzero(d["t3_mjd"] == mjd)
                T = triangle_matrix(d["t3_sta"][rows])
                R = T @ T.T / 3.0
                for k in range(r.shape[1]):  # one channel at a time
                    s = d["dcp"][rows, k]
                    if whitened:
                        x = r[rows, k] / s
                        total += x @ np.linalg.pinv(R, rcond=1e-10) @ x
                    else:
                        C = s[:, None] * R * s[None, :]
                        total += r[rows, k] @ np.linalg.pinv(C, rcond=1e-10) @ r[rows, k]
    return float(total)


def residuals(d, vis):
    """The whitened residual vector of ``vis`` against the loaded data,
    independent closure phases (a three-telescope file): (model V² − V²)/σ,
    then the chords 2 sin(Δ/2)/σ. Its squared norm is ``chi2(d, vis)``."""
    model_v2 = np.abs(vis(d["u"], d["v"], d["wl"])) ** 2
    r = [((model_v2 - d["v2"]) / d["dv2"]).ravel()]
    if "cp" in d:
        model = (
            np.angle(vis(d["u1"], d["v1"], d["wl3"]))
            + np.angle(vis(d["u2"], d["v2_"], d["wl3"]))
            - np.angle(vis(d["u1"] + d["u2"], d["v1"] + d["v2_"], d["wl3"]))
        )
        r.append((2.0 * np.sin((model - d["cp"]) / 2.0) / d["dcp"]).ravel())
    return np.concatenate(r)
