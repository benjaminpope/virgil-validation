"""Closed-form starting values for parametric star models (NumPy only).

The contest pipeline's star arms (scripts/contest_images.py) fitted their
parametric stars from fixed starts on prior edges, and the fits stalled or
ran away (docs/plan_imaging_contests.md, C2g). These estimators give starts
from the data alone, linearly or by an exact 1-D minimum:

- ``second_moments``: the Gaussian second moments of the brightness from the
  short baselines, by a weighted linear least squares of −ln V² on the
  quadratic form in (u, v). Size, elongation and position angle.
- ``disk_diameter_from_sigma``: the uniform or limb-darkened disk with the
  same second moment.
- ``star_fraction``: the fraction of the flux in a star of known visibility,
  next to an environment of known (or unknown) visibility.

They are starts, not answers: a companion or a near-round star biases the
moments (the PA is then poor), and the fits that follow refine them.
"""

import numpy as np

MAS = np.pi / 180.0 / 3600.0 / 1000.0  # radians per milliarcsecond


def second_moments(u, v, v2, err, v2_min=0.3, min_points=6):
    """Second moments of the brightness from V² on the short baselines.

    ``u``, ``v``: spatial frequencies in cycles per radian (baseline over
    wavelength), u East and v North; ``v2``, ``err``: squared visibilities and
    their errors. For a Gaussian of covariance Σ, V² = exp(−4π² qᵀΣq), so on
    the points with V² ≥ ``v2_min`` a weighted linear least squares of
    −ln V² = 4π²(a u² + 2b uv + c v²) gives Σ (weights from err / V²). With
    fewer than ``min_points`` such points, the ``min_points`` shortest
    baselines with V² > 0 are used instead; when Σ is not positive definite
    (or fewer than three points remain), an isotropic Gaussian is fitted.

    Returns a dict: ``sigma_major`` and ``sigma_minor`` (mas), ``ratio``
    (minor over major), ``pa`` (degrees East of North, in [0, 180)), ``n``
    (points used) and ``isotropic`` (True when the fallback was used)."""
    u, v, v2, err = (np.ravel(np.asarray(x, float)) for x in (u, v, v2, err))
    good = np.isfinite(u) & np.isfinite(v) & np.isfinite(v2) & np.isfinite(err) & (err > 0) & (v2 > 0)
    use = good & (v2 >= v2_min)
    if use.sum() < min_points:
        q = np.hypot(u, v)
        order = np.argsort(np.where(good, q, np.inf))
        use = np.zeros_like(good)
        use[order[: min(min_points, int(good.sum()))]] = True
    u, v, v2, err = u[use], v[use], v2[use], err[use]
    n = int(u.size)
    nan = {"sigma_major": np.nan, "sigma_minor": np.nan, "ratio": np.nan, "pa": np.nan, "n": n, "isotropic": True}
    if n == 0:
        return nan
    y = -np.log(np.clip(v2, 1e-12, 1.0))  # V² > 1 is noise about an unresolved point
    w = v2 / err  # 1 / σ(ln V²)
    design = 4 * np.pi**2 * np.stack([u**2, 2 * u * v, v**2], axis=1)
    if n >= 3:
        coef, *_ = np.linalg.lstsq(design * w[:, None], y * w, rcond=None)
        cov = np.array([[coef[0], coef[1]], [coef[1], coef[2]]])
        values, vectors = np.linalg.eigh(cov)  # ascending
        if values[0] > 0:
            major = vectors[:, 1]  # (East, North)
            pa = float(np.degrees(np.arctan2(major[0], major[1])) % 180.0)
            s_major, s_minor = np.sqrt(values[1]) / MAS, np.sqrt(values[0]) / MAS
            return {"sigma_major": float(s_major), "sigma_minor": float(s_minor), "ratio": float(s_minor / s_major),
                    "pa": pa, "n": n, "isotropic": False}
    q2 = 4 * np.pi**2 * (u**2 + v**2)
    s2 = float(np.sum(w**2 * q2 * y) / np.sum(w**2 * q2**2))
    if not s2 > 0:
        return nan | {"sigma_major": 0.0, "sigma_minor": 0.0, "ratio": 1.0, "pa": 90.0}
    s = np.sqrt(s2) / MAS
    return {"sigma_major": float(s), "sigma_minor": float(s), "ratio": 1.0, "pa": 90.0, "n": n, "isotropic": True}


def disk_diameter_from_sigma(sigma, u_ld=0.0):
    """The diameter of a disk with the linear limb-darkening law
    I(μ) = 1 − u(1 − μ) whose second moment per axis is σ². With s = r/R,
    ⟨x²⟩ = R² [(1 − u)/4 + 2u/15] / [2((1 − u)/2 + u/3)], so a uniform disk
    (u = 0) has D = 4σ; limb darkening concentrates the light, so D grows
    with u for the same σ."""
    u = float(u_ld)
    k = ((1 - u) / 4 + 2 * u / 15) / (2 * ((1 - u) / 2 + u / 3))
    return 2.0 * np.asarray(sigma, float) / np.sqrt(k)


def star_fraction(v_star, v_env, v2, err, plateau=0.2, q=None):
    """The fraction f ∈ [0, 1] of the flux in a star, for
    V² = |f V_s + (1 − f) V_e|² with unit-normalised complex visibilities
    ``v_star`` (V_s) and ``v_env`` (V_e) at the data's points. χ²(f) is a
    quartic in f, so its minimum on [0, 1] is exact: among the end points and
    the real roots of the cubic dχ²/df.

    With ``v_env=None`` (the environment unknown) the environment is taken as
    resolved on the longest ``plateau`` fraction of baselines (``q``, their
    spatial frequencies, is then needed), so f ≈ √⟨V² / |V_s|²⟩ there."""
    v2, err = np.ravel(np.asarray(v2, float)), np.ravel(np.asarray(err, float))
    vs = np.ravel(np.broadcast_to(np.asarray(v_star, complex), v2.shape))
    good = np.isfinite(v2) & np.isfinite(err) & (err > 0)
    if v_env is None:
        if q is None:
            raise ValueError("star_fraction needs the spatial frequencies q when v_env is None")
        q = np.ravel(np.asarray(q, float))
        long = good & (q >= np.quantile(q[good], 1 - plateau))
        ratio = np.mean(v2[long] / np.clip(np.abs(vs[long]) ** 2, 1e-12, None))
        return float(np.clip(np.sqrt(max(ratio, 0.0)), 0.0, 1.0))
    ve = np.ravel(np.broadcast_to(np.asarray(v_env, complex), v2.shape))
    a, b, w = (vs - ve)[good], ve[good], 1.0 / err[good] ** 2
    # m(f) = p2 f² + p1 f + p0, residual m − V²; χ² = Σ w (m − V²)².
    p2, p1, p0 = np.abs(a) ** 2, 2 * np.real(a * np.conj(b)), np.abs(b) ** 2 - v2[good]
    quartic = np.array([np.sum(w * p2 * p2), 2 * np.sum(w * p2 * p1), np.sum(w * (p1 * p1 + 2 * p2 * p0)),
                        2 * np.sum(w * p1 * p0), np.sum(w * p0 * p0)])
    roots = np.roots(np.polyder(quartic)) if np.any(quartic[:-1]) else np.array([])
    candidates = [0.0, 1.0] + [float(r.real) for r in roots if abs(r.imag) < 1e-9 and 0.0 < r.real < 1.0]
    return float(min(candidates, key=lambda f: np.polyval(quartic, f)))
