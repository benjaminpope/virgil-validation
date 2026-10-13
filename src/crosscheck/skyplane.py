"""Sky-plane comparison of a fitted orbit with a reference orbit.

Orbits are compared where they are observed, as tracks of the companion on the sky at the epochs, not by their
elements: (P, e, i, Omega, omega) have conventions and aliases (Omega and omega each shifted by 180 deg, a mirror
for a bright companion) that give one sky orbit many element sets. The positions come from `crosscheck.orbits`
(our own Kepler solver); the reference's omega is the secondary's, as there.

A label swap of the two stars negates every relative position, which is the flux-ratio flip of the fit: the score
takes the reference or its point reflection, whichever is nearer, and reports which.
"""

import numpy as np

from . import orbits as O

# Variants of the reference tried to diagnose a convention problem: (d_Omega, d_omega, i -> 180 - i)
VARIANTS = {
    "as-is": (0, 0, False),
    "omega+180": (0, 180, False),
    "Omega+180": (180, 0, False),
    "omega+180,Omega+180": (180, 180, False),
    "inc->180-inc": (0, 0, True),
}
SAME_FRACTION = 0.10  # rms offset below this fraction of the mean separation, or
SAME_SIGMA = 3.0  # below this many posterior spreads, is the same sky orbit


def thiele_innes(a, inc, omega, Omega):
    """Thiele-Innes constants (A, B, F, G): A, F are Dec and B, G RA coefficients of (X, Y)."""
    i, w, n = np.deg2rad([inc, omega, Omega])
    ci = np.cos(i)
    return (
        a * (np.cos(w) * np.cos(n) - np.sin(w) * np.sin(n) * ci),
        a * (np.cos(w) * np.sin(n) + np.sin(w) * np.cos(n) * ci),
        a * (-np.sin(w) * np.cos(n) - np.cos(w) * np.sin(n) * ci),
        a * (-np.sin(w) * np.sin(n) + np.cos(w) * np.cos(n) * ci),
    )


def _pos(mjd, period, t_peri, ecc, inc, omega, Omega, a):
    x, y = O.position(mjd, period, t_peri, ecc, inc, omega, Omega, a)
    return np.stack([np.asarray(x), np.asarray(y)], -1)


def ref_pos(ref, mjd, d_omega=0.0, d_Omega=0.0, flip_inc=False):
    inc = 180 - ref["inc"] if flip_inc else ref["inc"]
    return _pos(mjd, ref["P"], ref["t_peri"], ref["ecc"], inc, ref["omega"] + d_omega, ref["Omega"] + d_Omega, ref["a_mas"])


def fit_pos(best, mjd, t_ref):
    return _pos(
        mjd, best["period"], t_ref + best["dt_peri"], best["ecc"], best["inc"], best["omega"], best["Omega"], best["a_mas"]
    )


def _rms(d):
    return float(np.sqrt(np.mean(np.sum(d**2, -1))))


def score(meta, best, t_ref=None, samples=None):
    """Compare a fitted orbit with ``meta["reference"]`` on the sky at ``meta["mjd"]``.

    ``best`` holds period, dt_peri (days from ``t_ref``), ecc, inc, omega, Omega, a_mas; ``samples`` (optional)
    the same keys as arrays, for the posterior spread. Returns rms and max offset (mas), the rms as a fraction of
    the mean separation, the rms over the posterior spread, whether the reference's point reflection matches (the
    flux-ratio flip), the rms of each convention variant, the Thiele-Innes constants and the verdict
    SAME-SKY-ORBIT or DIFFERENT.
    """
    mjd = np.asarray(meta["mjd"], float)
    ref = meta["reference"]
    t_ref = best.get("t_ref", 0.0) if t_ref is None else t_ref
    ours = fit_pos(best, mjd, t_ref)
    variants = {}
    for name, (d_Omega, d_omega, flip_inc) in VARIANTS.items():
        r = ref_pos(ref, mjd, d_omega, d_Omega, flip_inc)
        for sign, tag in ((1, ""), (-1, " +flip")):
            variants[name + tag] = _rms(ours - sign * r)
    sign = 1 if variants["as-is"] <= variants["as-is +flip"] else -1
    r0 = sign * ref_pos(ref, mjd)
    dist = np.hypot(*(ours - r0).T)
    sep = np.hypot(*r0.T)
    rms = float(np.sqrt(np.mean(dist**2)))
    out = dict(
        variant_rms=variants,
        mirror="flip" if sign < 0 else "none",
        flux_flip_matches=sign < 0,
        rms=rms,
        max=float(dist.max()),
        mean_sep=float(np.mean(sep)),
        frac_sep=rms / float(np.mean(sep)),
        best_variant=min(variants, key=variants.get),
        best_variant_rms=min(variants.values()),
        TI_ours_ABFG=np.array(thiele_innes(best["a_mas"], best["inc"], best["omega"], best["Omega"])).tolist(),
        TI_ref_ABFG=(sign * np.array(thiele_innes(ref["a_mas"], ref["inc"], ref["omega"], ref["Omega"]))).tolist(),
        spread=float("nan"),
        offset_over_spread=float("nan"),
    )
    if samples is not None:
        n = len(samples["period"])
        idx = np.linspace(0, n - 1, min(n, 400)).astype(int)
        tracks = np.stack(
            [
                _pos(
                    mjd,
                    samples["period"][k],
                    t_ref + samples["dt_peri"][k],
                    samples["ecc"][k],
                    samples["inc"][k],
                    samples["omega"][k],
                    samples["Omega"][k],
                    samples["a_mas"][k],
                )
                for k in idx
            ]
        )
        spread = float(np.sqrt(np.mean(np.sum(tracks.var(0), -1))))
        out["spread"], out["offset_over_spread"] = spread, rms / max(spread, 1e-12)
    same = out["frac_sep"] < SAME_FRACTION or out["offset_over_spread"] < SAME_SIGMA
    out["verdict"] = "SAME-SKY-ORBIT" if same else "DIFFERENT"
    return out
