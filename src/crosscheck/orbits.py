"""Keplerian relative orbits from the textbook formulae, with SciPy.

The visual-binary convention: the node Omega and position angles are
measured North through East, omega from the ascending node in the direction
of motion, u = omega + f the argument of latitude, and

    dDec = r (cos Omega cos u - sin Omega sin u cos i)
    dRA  = r (sin Omega cos u + cos Omega sin u cos i)

so for i < 90 deg the position angle increases with time. Along the line
of sight, z = r sin u sin i is positive away from the observer: the
ascending node is the one where the companion recedes. Its radial velocity
relative to the primary is then the textbook

    v_z = 2 pi a sin i / (P sqrt(1 - e^2)) (cos u + e cos omega),

positive receding. Kepler's equation M = E - e sin E is solved by Newton's
method in SciPy.
"""

import numpy as np
from scipy import optimize


def eccentric_anomaly(mean, ecc):
    mean = np.atleast_1d(np.asarray(mean, float))
    return np.array([optimize.newton(lambda E, m=m: E - ecc * np.sin(E) - m, m + ecc * np.sin(m),
                                     fprime=lambda E: 1 - ecc * np.cos(E), tol=1e-15, maxiter=100)
                     for m in mean])


def true_anomaly(mean, ecc):
    E = eccentric_anomaly(mean, ecc)
    return 2 * np.arctan2(np.sqrt(1 + ecc) * np.sin(E / 2), np.sqrt(1 - ecc) * np.cos(E / 2))


def sky_position(f, ecc, inc, omega, Omega, a=1.0):
    """(dRA, dDec) at true anomaly f (radians); angles in degrees."""
    i, w, W = np.deg2rad([inc, omega, Omega])
    r = a * (1 - ecc**2) / (1 + ecc * np.cos(f))
    u = w + f
    ddec = r * (np.cos(W) * np.cos(u) - np.sin(W) * np.sin(u) * np.cos(i))
    dra = r * (np.sin(W) * np.cos(u) + np.cos(W) * np.sin(u) * np.cos(i))
    return dra, ddec


def position(t, period, t_peri, ecc, inc, omega, Omega, a):
    """(dRA, dDec) in a's units at times t (days)."""
    mean = 2 * np.pi * (np.asarray(t, float) - t_peri) / period
    return sky_position(true_anomaly(np.mod(mean + np.pi, 2 * np.pi) - np.pi, ecc), ecc, inc, omega, Omega, a)


def position_angle(f, ecc, inc, omega, Omega):
    """Position angle (radians, North through East) at true anomaly f."""
    dra, ddec = sky_position(f, ecc, inc, omega, Omega)
    return np.arctan2(dra, ddec)


def depth(f, ecc, inc, omega, a=1.0):
    """z, positive away from the observer, at true anomaly f (radians)."""
    r = a * (1 - ecc**2) / (1 + ecc * np.cos(f))
    return r * np.sin(np.deg2rad(omega) + f) * np.sin(np.deg2rad(inc))


def radial_velocity(t, period, t_peri, ecc, inc, omega, a):
    """dz/dt of the companion relative to the primary (a's units per day,
    positive receding) at times t (days)."""
    mean = 2 * np.pi * (np.asarray(t, float) - t_peri) / period
    f = true_anomaly(np.mod(mean + np.pi, 2 * np.pi) - np.pi, ecc)
    k = 2 * np.pi * a * np.sin(np.deg2rad(inc)) / (period * np.sqrt(1 - ecc**2))
    return k * (np.cos(np.deg2rad(omega) + f) + ecc * np.cos(np.deg2rad(omega)))


# ---------------------------------------------------------------------------
# Comparing no-RV orbit fits in projected degrees of freedom
# (design/orbit_comparison.md).  Samples are dicts of equal-length arrays with
# keys P (days), e, t_p (days), a (mas), inc, omega, Omega (degrees).
# ---------------------------------------------------------------------------

ELEMENTS = ("P", "e", "t_p", "a", "inc", "omega", "Omega")
RTOL_EIG = 1e-6  # eigenvalue threshold of the pseudo-inverse, relative to the largest


def thiele_innes(a, inc, omega, Omega):
    """Thiele-Innes constants (A, B, F, G), degrees in, a's units out.

    Expanding `sky_position` in cos f and sin f (textbook, e.g. Heintz 1978):
    dDec = A x + F y and dRA = B x + G y with x = cos E - e and
    y = sqrt(1 - e^2) sin E. So A, F are the Dec coefficients and B, G the RA
    ones. This is the only place the constants are computed; never take them
    from a quoted table.
    """
    i, w, W = (np.deg2rad(np.asarray(v, float)) for v in (inc, omega, Omega))
    a = np.asarray(a, float)
    A = a * (np.cos(w) * np.cos(W) - np.sin(w) * np.sin(W) * np.cos(i))
    B = a * (np.cos(w) * np.sin(W) + np.sin(w) * np.cos(W) * np.cos(i))
    F = -a * (np.sin(w) * np.cos(W) + np.cos(w) * np.sin(W) * np.cos(i))
    G = -a * (np.sin(w) * np.sin(W) - np.cos(w) * np.cos(W) * np.cos(i))
    return A, B, F, G


def _kepler_E(mean, ecc):
    """Vectorised Newton solution of M = E - e sin E (starting at pi for high e)."""
    mean = np.mod(mean + np.pi, 2 * np.pi) - np.pi
    E = mean + ecc * np.sin(mean)
    E = np.where(ecc > 0.8, np.where(mean < 0, -np.pi, np.pi), E)
    for _ in range(100):
        step = (E - ecc * np.sin(E) - mean) / (1 - ecc * np.cos(E))
        E = E - step
        if np.max(np.abs(step)) < 1e-14:
            break
    else:
        raise RuntimeError("Kepler's equation did not converge")
    return E


def predict_track(samples, t):
    """Relative positions (dRA, dDec) of every sample at times t (days).

    Returns an array (n_samples, len(t), 2). Built from the Thiele-Innes form,
    so it equals `position` to rounding.
    """
    t = np.asarray(t, float)
    P, e, tp = (np.asarray(samples[k], float)[:, None] for k in ("P", "e", "t_p"))
    A, B, F, G = (v[:, None] for v in thiele_innes(*(np.asarray(samples[k], float)
                                                     for k in ("a", "inc", "omega", "Omega"))))
    E = _kepler_E(2 * np.pi * (t[None, :] - tp) / P, e)
    x, y = np.cos(E) - e, np.sqrt(1 - e**2) * np.sin(E)
    return np.stack([B * x + G * y, A * x + F * y], axis=-1)


def _quad(d, C, rtol=RTOL_EIG):
    """d^T C^+ d with eigenvalues below rtol * max dropped.

    Returns (d2, rank, norm of the part of d outside the retained subspace).
    """
    w, V = np.linalg.eigh(np.atleast_2d(C))
    keep = w > rtol * w.max()
    c = V.T @ d
    d2 = float(np.sum(c[keep] ** 2 / w[keep]))
    return d2, int(keep.sum()), c[~keep]


def _flat(track):
    """(n, N, 2) -> (n, 2N) with the two axes interleaved per epoch."""
    return track.reshape(track.shape[0], -1)


def per_epoch_d2(ours, ref, c_ref=None):
    """Statistic A per epoch. ours, ref: (n, N, 2) predicted positions.

    d^2 = dmu^T (C_ours + C_ref)^-1 dmu ~ chi^2_2 per epoch. `c_ref` overrides the
    reference covariance (N, 2, 2), e.g. zeros for the C_ours-alone variant.
    Returns (d2 (N,), p (N,)).
    """
    dmu = ours.mean(0) - ref.mean(0)
    d2 = np.empty(dmu.shape[0])
    for k in range(dmu.shape[0]):
        C = np.cov(ours[:, k, :].T)
        C = C + (np.cov(ref[:, k, :].T) if c_ref is None else c_ref[k])
        d2[k] = _quad(dmu[k], C)[0]
    from scipy import stats
    return d2, stats.chi2.sf(d2, 2)


def joint_d2(ours, ref, sigma, c_ref=None):
    """Statistic A, joint over epochs, with the eigenvalue-threshold pseudo-inverse.

    Returns a dict: d2, rank r, p (chi^2_r), residual (norm of dmu outside the
    retained subspace, in units of the per-epoch position errors `sigma`, which
    is required: the comparison fails if it exceeds 1, see residual_ok).
    """
    from scipy import stats
    dmu = (ours.mean(0) - ref.mean(0)).ravel()
    C = np.cov(_flat(ours).T)
    C = C + (np.cov(_flat(ref).T) if c_ref is None else c_ref)
    w, V = np.linalg.eigh(C)
    keep = w > RTOL_EIG * w.max()
    c = V.T @ dmu
    d2 = float(np.sum(c[keep] ** 2 / w[keep]))
    perp = V[:, ~keep] @ c[~keep]
    resid = float(np.linalg.norm(perp / np.broadcast_to(np.asarray(sigma, float), dmu.shape)))
    r = int(keep.sum())
    return {"d2": d2, "rank": r, "p": float(stats.chi2.sf(d2, r)), "residual": resid,
            "residual_ok": resid <= 1.0}


def max_track_separation(ours, ref):
    """Max separation of the two median tracks over one period, as a fraction of a.

    Samples are dicts; the grid starts at the median periastron of `ours`.
    """
    P = np.median(ours["P"])
    t = np.median(ours["t_p"]) + np.linspace(0, P, 721)
    mo = np.median(predict_track(ours, t), axis=0)
    mr = np.median(predict_track(ref, t), axis=0)
    return float(np.max(np.hypot(*(mo - mr).T)) / np.median(ours["a"]))


def periastron_anchor(samples, t_mean):
    """The single periastron nearest `t_mean`, from the median P and t_p."""
    P, tp = np.median(samples["P"]), np.median(samples["t_p"])
    return float(tp + P * np.round((t_mean - tp) / P))


def projected_elements(samples, t_mean, anchor=None):
    """theta = (P, e, t_p, A, B, F, G) per sample, shape (n, 7).

    t_p is shifted by whole periods to the periastron nearest one anchor shared
    by the whole comparison (default: `periastron_anchor` of these samples), so
    a posterior straddling a half-period boundary is not split into two modes.
    """
    anchor = periastron_anchor(samples, t_mean) if anchor is None else anchor
    P, tp = np.asarray(samples["P"], float), np.asarray(samples["t_p"], float)
    tp = tp + P * np.round((anchor - tp) / P)
    ABFG = thiele_innes(*(np.asarray(samples[k], float) for k in ("a", "inc", "omega", "Omega")))
    return np.column_stack([P, samples["e"], tp, *ABFG])


def statistic_b(ours, ref, t_mean, e_min=0.1, c_ref=None):
    """Statistic B: d^2 ~ chi^2_7 on the projected elements.

    Not applicable when the median e of either posterior is below `e_min`
    (t_p undefined as e -> 0): use statistic A there.
    """
    from scipy import stats
    if min(np.median(ours["e"]), np.median(ref["e"])) < e_min:
        return {"applicable": False, "d2": np.nan, "rank": 0, "p": np.nan,
                "reason": f"median e < {e_min}: t_p undefined, use statistic A"}
    anchor = periastron_anchor(ours, t_mean)
    to, tr = projected_elements(ours, t_mean, anchor), projected_elements(ref, t_mean, anchor)
    C = np.cov(to.T) + (np.cov(tr.T) if c_ref is None else c_ref)
    d2, r, _ = _quad(to.mean(0) - tr.mean(0), C)
    return {"applicable": True, "d2": d2, "rank": r, "p": float(stats.chi2.sf(d2, r)), "reason": ""}


def fold_samples(omega, Omega, Omega_ref):
    """Fold to Omega in [Omega_ref - 90, Omega_ref + 90), shifting omega with it.

    Returns (omega, Omega, mirror) with mirror True for samples that needed an
    odd number of 180 degree shifts (the mirror mode relative to Omega_ref).
    Angles in degrees; omega is returned in [0, 360).
    """
    omega, Omega = np.asarray(omega, float), np.asarray(Omega, float)
    k = np.floor((Omega - Omega_ref + 90.0) / 180.0)
    return np.mod(omega + 180.0 * k, 360.0), Omega - 180.0 * k, (k % 2 == 1)


def mirror_fraction(Omega, Omega_ref):
    """Posterior weight of the mirror mode, before folding."""
    return float(np.mean(fold_samples(np.zeros_like(np.asarray(Omega, float)), Omega, Omega_ref)[2]))


def independent_draws(marginals, n, rng):
    """Samples drawn independently from {element: (mean, sd)}: the C_ref
    variant that ignores correlations. Labelled unscored by `compare_orbits`."""
    return {k: rng.normal(*marginals[k], size=n) for k in ELEMENTS}


def compare_orbits(ours, ref, t_obs, *, sigma, ref_kind="samples", t_mean=None, rng=None):
    """Run statistics A and B against a reference.

    ref_kind="samples": `ref` holds samples (or a sample set carrying the
    published covariance) and every statistic is scored.
    ref_kind="marginals": `ref` maps element -> (mean, sd) only. Nothing is
    scored. Reports the C_ours-alone d^2 (stricter bound, C_ref = 0) and the
    independent-draw C_ref variant, both with scored=False.
    """
    t_obs = np.asarray(t_obs, float)
    t_mean = float(np.mean(t_obs)) if t_mean is None else t_mean
    if ref_kind == "samples":
        rt = predict_track(ref, t_obs)
        ot = predict_track(ours, t_obs)
        d2, p = per_epoch_d2(ot, rt)
        return {"scored": True, "per_epoch_d2": d2, "per_epoch_p": p,
                "joint": joint_d2(ot, rt, sigma), "B": statistic_b(ours, ref, t_mean),
                "max_separation": max_track_separation(ours, ref)}
    if ref_kind != "marginals":
        raise ValueError(ref_kind)
    rng = np.random.default_rng() if rng is None else rng
    n = len(ours["P"])
    mean = {k: np.full(n, float(np.atleast_1d(ref[k])[0])) for k in ELEMENTS}
    ind = independent_draws(ref, n, rng)
    ot = predict_track(ours, t_obs)
    mt, it = predict_track(mean, t_obs), predict_track(ind, t_obs)
    zero_e = np.zeros((len(t_obs), 2, 2))
    out = {"scored": False}
    for label, track, c_ref in (("ours_alone", mt, zero_e), ("independent_draws", it, None)):
        d2, p = per_epoch_d2(ot, track, c_ref=c_ref)
        cflat = None if c_ref is None else np.zeros((2 * len(t_obs),) * 2)
        out[label] = {"scored": False, "per_epoch_d2": d2, "per_epoch_p": p,
                      "joint": joint_d2(ot, track, sigma, c_ref=cflat)}
    return out
