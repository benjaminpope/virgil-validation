"""Scales, starts and checks for the informed contest fits (NumPy and SciPy only; nothing here imports virgil).

The informed fits of ``scripts/contest_informed.py`` (design/contest_informed_plan.md) need numbers that come from the
data alone and checks that say when a fit failed. They are written here from the physics, so a start is never
virgil's own estimate and a check is testable without a fit:

- ``second_moments``: Gaussian second moments from the short baselines by a weighted linear least squares of -ln V^2
  (ported from the branch ``contest-round3-doctor``, ``crosscheck.starts``; used as a cross-check of ``moment_fit``).
- ``moment_fit`` (plan 4.1): V^2 of a point, an elliptical Gaussian and over-resolved light, from a grid of sizes,
  axis ratios and angles, refined with a robust loss. The Fourier transform of a Gaussian of full width at half
  maximum theta is exp(-pi^2 theta^2 q^2 / (4 ln 2)) (Bracewell, *The Fourier Transform and Its Applications*).
- ``Scales`` and ``scales_of`` (plan 2): extent, reach and pixel, each with its source.
- Stratified geometry (plan 4.2): ``log_steps``, ``inc_pa_grid``, ``angle_steps``.
- ``tiled_map`` and ``map_peaks`` (plan 4.3): a map over the whole reach in tiles, and its separated peaks.
- The checks of plan section 5 as pure functions of plain numbers.

Spatial frequencies are in cycles per radian (baseline over wavelength), ``u`` East and ``v`` North; position angles
are in degrees East of North, of the major axis; sizes are in milliarcseconds.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.optimize import least_squares, nnls

MAS = np.pi / 180.0 / 3600.0 / 1000.0  # radians per milliarcsecond
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))
SYSTEMATIC_FLOOR = 0.03  # V^2 floor of the moment fit only: a start must not chase 1e-4 error bars


# ---------------------------------------------------------------------------------------------------------------
# Moments


def second_moments(u, v, v2, err, v2_min=0.3, min_points=6):
    """Second moments of the brightness from V^2 on the short baselines.

    For a Gaussian of covariance S, V^2 = exp(-4 pi^2 q^T S q), so on the points with V^2 >= ``v2_min`` a weighted
    linear least squares of -ln V^2 = 4 pi^2 (a u^2 + 2 b u v + c v^2) gives S (weights from err / V^2). With fewer than
    ``min_points`` such points the shortest baselines are used; when S is not positive definite an isotropic Gaussian
    is fitted. Returns ``sigma_major`` and ``sigma_minor`` (mas), ``ratio``, ``pa`` (degrees), ``n`` and ``isotropic``."""
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
    y = -np.log(np.clip(v2, 1e-12, 1.0))
    w = v2 / err
    design = 4 * np.pi**2 * np.stack([u**2, 2 * u * v, v**2], axis=1)
    if n >= 3:
        coef, *_ = np.linalg.lstsq(design * w[:, None], y * w, rcond=None)
        cov = np.array([[coef[0], coef[1]], [coef[1], coef[2]]])
        values, vectors = np.linalg.eigh(cov)
        if values[0] > 0:
            major = vectors[:, 1]
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


def gaussian_visibility(u, v, fwhm, ratio, pa):
    """Visibility of a unit-flux elliptical Gaussian centred on the origin: ``fwhm`` (mas) along the major axis at
    ``pa``, ``ratio`` times that along the minor axis. Real, since the source is centro-symmetric."""
    theta = np.radians(pa)
    q_major = u * np.sin(theta) + v * np.cos(theta)
    q_minor = u * np.cos(theta) - v * np.sin(theta)
    quad = (fwhm * MAS) ** 2 * (q_major**2 + (ratio * q_minor) ** 2)
    return np.exp(-np.pi**2 * quad / (4.0 * np.log(2.0)))


def moment_model_v2(u, v, compact, extended, fwhm, ratio, pa, amplitude=1.0):
    """V^2 of ``compact`` flux in a point, ``extended`` in an elliptical Gaussian and the rest over-resolved."""
    vis = compact + extended * gaussian_visibility(u, v, fwhm, ratio, pa)
    return amplitude * vis**2


def moment_fit(u, v, v2, err, theta_range, fit_amplitude=False, floor=SYSTEMATIC_FLOOR, n_size=24,
               ratios=(0.3, 0.45, 0.6, 0.75, 0.9, 1.0), n_pa=12):
    """The visibility-moment fit of plan 4.1: V^2 = A (f_p + f_g G(q))^2 with a point of flux ``f_p``, an elliptical
    Gaussian of flux ``f_g`` and axes (``fwhm``, ``ratio``, ``pa``), and over-resolved flux 1 - f_p - f_g, and A = 1
    unless ``fit_amplitude`` (a data set normalised without part of its flux has V^2 above 1 at zero baseline).

    A grid of sizes (log-spaced over ``theta_range``, mas), ratios and angles fixes the geometry; for each, the two
    fluxes are a non-negative least squares on |V|; the three best are refined with a soft-L1 loss on V^2. The errors
    are quoted errors added in quadrature with ``floor`` (a start must not chase error bars of 1e-4).

    Returns ``compact``, ``extended``, ``resolved``, ``fwhm``, ``ratio``, ``pa``, ``amplitude``, ``chi2_quoted`` and
    ``n`` (chi2 of the fit on the *quoted* errors), ``rms``, ``short_v2`` (the largest V^2 on the shortest quarter of
    the baselines), ``longest_tenth`` (the median |V| on the longest tenth: a second estimate of ``compact``) and
    ``at_upper_edge`` (the size reached the top of the grid: the data bound the extent only from below), or ``None``
    with fewer than six usable points."""
    u, v, v2, err = (np.ravel(np.asarray(x, float)) for x in (u, v, v2, err))
    good = np.isfinite(u) & np.isfinite(v) & np.isfinite(v2) & np.isfinite(err) & (err > 0)
    u, v, v2, err = u[good], v[good], v2[good], err[good]
    if u.size < 6:
        return None
    sigma = np.hypot(err, floor)
    q = np.hypot(u, v)
    amp = np.sqrt(np.clip(v2, 0.0, None))
    sigma_amp = sigma / (2.0 * np.maximum(amp, 0.1))  # |V| = sqrt(V^2): d|V| = dV^2 / (2 |V|), bounded below
    sizes = np.geomspace(theta_range[0], theta_range[1], n_size)
    angles = np.linspace(0.0, 180.0, n_pa, endpoint=False)
    scored = []
    for size in sizes:
        for ratio in ratios:
            for pa in angles if ratio < 1.0 else angles[:1]:
                g = gaussian_visibility(u, v, size, ratio, pa)
                design = np.stack([np.ones_like(g), g], axis=1) / sigma_amp[:, None]
                # f_p + f_g <= 1: nnls with the constraint added as a heavily weighted row of "1 - f_p - f_g >= 0"
                # is not needed: the refinement below bounds it; the grid only ranks geometries.
                (fp, fg), _ = nnls(design, amp / sigma_amp)
                total = fp + fg
                if total > 1.0:
                    fp, fg = fp / total, fg / total
                resid = (amplitude_of(fp, fg, g) - amp) / sigma_amp
                scored.append((float(np.sum(resid**2)), size, ratio, pa, fp, fg))
    scored.sort(key=lambda row: row[0])
    best = None
    lower = [0.0, 0.0, theta_range[0], 0.2, -90.0] + ([0.5] if fit_amplitude else [])
    upper = [1.0, 1.0, theta_range[1], 1.0, 270.0] + ([2.5] if fit_amplitude else [])
    seen = []
    for _, size, ratio, pa, fp, fg in scored:
        if any(abs(np.log(size / s)) < 1e-9 and r == ratio for s, r in seen) or len(seen) >= 3:
            continue
        seen.append((size, ratio))
        x0 = np.array([fp, fg, size, ratio, pa] + ([1.0] if fit_amplitude else []), float)
        x0[:2] = np.clip(x0[:2], 1e-3, 1.0)
        x0[2] = np.clip(x0[2], theta_range[0] * 1.001, theta_range[1] * 0.999)
        x0[3] = np.clip(x0[3], 0.201, 0.999)

        def residual(x):
            fp_, fg_ = x[0], x[1]
            overflow = max(fp_ + fg_ - 1.0, 0.0) * 1e3  # f_p + f_g <= 1, as a steep wall
            a = x[5] if fit_amplitude else 1.0
            model = moment_model_v2(u, v, fp_, fg_, x[2], x[3], x[4], a)
            return np.concatenate([(model - v2) / sigma, [overflow]])

        try:
            sol = least_squares(residual, x0, bounds=(lower, upper), loss="soft_l1", f_scale=1.0, x_scale="jac")
        except (ValueError, FloatingPointError):  # pragma: no cover - a degenerate grid point must not stop the others
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None:
        return None
    fp, fg, fwhm, ratio, pa = (float(x) for x in best.x[:5])
    a = float(best.x[5]) if fit_amplitude else 1.0
    total = fp + fg
    if total > 1.0:
        fp, fg = fp / total, fg / total
    model = moment_model_v2(u, v, fp, fg, fwhm, ratio, pa, a)
    chi2 = float(np.sum(((model - v2) / err) ** 2))
    shortest = q <= np.quantile(q, 0.25)
    longest = q >= np.quantile(q, 0.9)
    return {"compact": fp, "extended": fg, "resolved": float(max(0.0, 1.0 - fp - fg)), "fwhm": fwhm, "ratio": ratio,
            "pa": float(pa % 180.0), "amplitude": a, "chi2_quoted": chi2, "n": int(u.size),
            "rms": float(np.sqrt(np.mean((model - v2) ** 2))), "short_v2": float(np.max(v2[shortest])),
            "longest_tenth": float(np.median(amp[longest])), "at_upper_edge": bool(fwhm >= 0.95 * theta_range[1])}


def amplitude_of(fp, fg, g):
    """|V| of the moment model."""
    return fp + fg * g


# ---------------------------------------------------------------------------------------------------------------
# Scales (plan section 2)


@dataclasses.dataclass
class Scales:
    """The three lengths of an informed fit, each with its source (mas), and the notes about them.

    ``extent`` is how far the resolved light reaches (upper limit of size priors, field of every image),
    ``reach`` is how far a compact companion is searched for, ``pixel`` is for any image. ``warnings`` is a list of
    sentences for the report."""

    resolution: float
    extent: float
    extent_source: str
    reach: float
    reach_source: str
    pixel: float
    pixel_source: str
    npix: int
    npix_wanted: int
    extent_known: bool
    warnings: list = dataclasses.field(default_factory=list)

    def asdict(self):
        return dataclasses.asdict(self)


MAX_PIXELS = 128  # pixels a side of any image of the informed fits


def scales_of(b_max, b_min, moments=None, field=None, field_source=None, tapered=False):
    """The extent, reach and pixel of plan section 2, from the largest and smallest baselines in units of the
    wavelength (cycles per radian) ``b_max``, ``b_min``, the visibility-moment fit ``moments`` of ``moment_fit`` (or
    None), and the contest's field (mas) where it gave one (``tapered`` if it also tapered the data).

    extent E: the given field; otherwise 4 x the extended light's major FWHM, at least 8 resolution elements.
    reach S: half the given field when the contest tapered it, else 3 lambda / B_min.
    pixel: lambda / (4 B_max), the field being E; more than ``MAX_PIXELS`` a side coarsens the pixel, and says so.
    When the data bound the extent only from below (V^2 below 0.03 on every short baseline, or the fitted size at the
    top of its grid) and no field was given, E is the reach and ``extent_known`` is False."""
    resolution = 1.0 / (2.0 * b_max) / MAS
    fringe = 1.0 / b_min / MAS  # lambda / B_min
    warnings = []
    if tapered and field:
        reach, reach_source = 0.5 * field, "half the given field (the contest tapered the data)"
    else:
        reach, reach_source = 3.0 * fringe, "3 x lambda / B_min"
    bounded_below = moments is None or moments["short_v2"] < 0.03 or moments["at_upper_edge"]
    if field:
        extent, source, known = float(field), f"the contest's field ({field_source})", True
    elif moments is not None and not bounded_below:
        extent = max(4.0 * moments["fwhm"], 8.0 * resolution)
        source, known = "4 x the major FWHM of the visibility-moment fit (at least 8 resolution elements)", True
    else:
        extent, known = reach, False
        source = "the reach: the data bound the extent only from below, and no field was given"
        warnings.append("extent_unconstrained: V^2 is below 0.03 on every short baseline or the fitted size is at the top "
                        "of its grid, so the extent is only a lower limit")
    pixel = 0.25 / b_max / MAS
    wanted = int(np.ceil(extent / pixel))
    npix = min(wanted, MAX_PIXELS)
    pixel_source = "lambda / (4 B_max)"
    if wanted > MAX_PIXELS:
        pixel = extent / MAX_PIXELS
        pixel_source = f"coarsened from lambda / (4 B_max): {wanted} pixels a side would be needed"
        warnings.append(f"pixel coarsened: {wanted} pixels a side at lambda / (4 B_max) exceed {MAX_PIXELS}")
    return Scales(resolution, float(extent), source, float(reach), reach_source, float(pixel), pixel_source, int(npix),
                  int(wanted), bool(known), warnings)


# ---------------------------------------------------------------------------------------------------------------
# Stratified geometry (plan 4.2)


def log_steps(centre, low=0.5, high=2.0, n=8):
    """``n`` sizes from ``low`` x ``centre`` to ``high`` x ``centre``, equal in log."""
    return list(float(centre) * np.geomspace(low, high, n))


def angle_steps(n, period=180.0):
    """``n`` angles over a period, starting at zero."""
    return [float(x) for x in np.arange(n) * period / n]


def inc_pa_grid(inc0, pa0, n_inc=4, n_pa=6, inc_range=(10.0, 85.0), period=180.0):
    """The moment values of inclination and position angle first, then an ``n_inc`` x ``n_pa`` grid. Each entry is
    ``(inc, pa)`` in degrees."""
    incs = np.linspace(inc_range[0], inc_range[1], n_inc)
    return [(float(inc0), float(pa0))] + [(float(i), float(p)) for i in incs for p in angle_steps(n_pa, period)]


def inside(x, low, high, log=False, edge=0.01):
    """``x`` moved into the interior of [low, high] (``edge`` of the range, in log for a log prior at each end): a
    start never sits on a bound."""
    lo, hi = (np.log(low), np.log(high)) if log else (low, high)
    pad = edge * (hi - lo)
    y = np.log(max(float(x), 1e-300)) if log else float(x)
    y = float(np.clip(y if np.isfinite(y) else hi, lo + pad, hi - pad))
    return float(np.exp(y)) if log else y


# ---------------------------------------------------------------------------------------------------------------
# Tiled maps (plan 4.3)


def tiled_map(evaluate, half, step, tile=256, max_axis=4096):
    """A map of ``evaluate(dra_axis, ddec_axis) -> (len(dra_axis), len(ddec_axis))`` over [-half, half]^2 at ``step``
    (mas), computed in tiles of ``tile`` points a side so memory stays bounded however wide the reach. Returns
    ``(axis, surface, step)``; an axis longer than ``max_axis`` points is coarsened (and ``step`` says by how much)."""
    n = int(round(2.0 * half / step)) + 1
    if n > max_axis:
        n = max_axis
    axis = np.linspace(-half, half, n)
    step = float(axis[1] - axis[0]) if n > 1 else 0.0
    surface = np.full((n, n), -np.inf, dtype=np.float32)
    for i in range(0, n, tile):
        for j in range(0, n, tile):
            block = np.asarray(evaluate(axis[i:i + tile], axis[j:j + tile]), float)
            surface[i:i + block.shape[0], j:j + block.shape[1]] = block
    return axis, surface, step


def map_peaks(axis, surface, k, separation, exclude=0.0):
    """The ``k`` highest peaks of a map at least ``separation`` apart, none within ``exclude`` of the origin.
    Returns ``[(dra, ddec, value)]`` with ``dra`` the first axis of ``surface``."""
    surface = np.where(np.isfinite(surface), surface, -np.inf)
    if exclude > 0:
        dra, ddec = np.meshgrid(axis, axis, indexing="ij")
        surface = np.where(np.hypot(dra, ddec) < exclude, -np.inf, surface)
    peaks = []
    flat = surface.ravel()
    take = min(flat.size, 200000)
    order = np.argpartition(flat, -take)[-take:]
    for index in order[np.argsort(flat[order])[::-1]]:
        if not np.isfinite(flat[index]):
            break
        i, j = np.unravel_index(index, surface.shape)
        if all(np.hypot(axis[i] - p[0], axis[j] - p[1]) > separation for p in peaks):
            peaks.append((float(axis[i]), float(axis[j]), float(flat[index])))
        if len(peaks) == k:
            break
    return peaks


# ---------------------------------------------------------------------------------------------------------------
# Checks (plan section 5). Each returns (ok, detail). Thresholds are the proposals of design/contest_informed_criteria.md.

BOUND_EDGE = 0.01  # no_bounds: within this fraction of the range of a prior bound
AGREE_FRACTION = 0.01  # starts_agree / nested: -2 ln L within this fraction of N of the best
MIN_AGREE = 2  # starts_agree: at least this many starts at the best
FAMILY_FLUX_LIMIT = 0.2  # family_complete: the image on top may carry at most this fraction of the model's flux
RING_FRACTION = 0.2  # no_collapse: a ring narrower than this fraction of the resolution
COMPANION_ELEMENTS = 2.0  # no_collapse: a companion closer than this many resolution elements to the star
PERIODIC = ("angle", "full_angle")  # prior kinds that wrap: no_bounds does not apply


def no_bounds(values, bounds, edge=BOUND_EDGE):
    """Fails when any fitted parameter lies within ``edge`` of the range of a prior bound (in log for a log prior).

    ``values`` maps a path to a number; ``bounds`` maps a path to ``(low, high, log, kind)``. Periodic angles and
    parameters without a finite bound are skipped. ``detail`` lists the offenders."""
    hits = []
    for path, (low, high, log, kind) in bounds.items():
        if path not in values or kind in PERIODIC or not (np.isfinite(low) and np.isfinite(high)):
            continue
        x = float(np.ravel(values[path])[0])
        lo, hi = (np.log(low), np.log(high)) if log else (low, high)
        y = np.log(max(x, 1e-300)) if log else x
        fraction = (y - lo) / (hi - lo)
        if fraction < edge or fraction > 1.0 - edge:
            hits.append({"parameter": path, "value": x, "low": low, "high": high, "end": "low" if fraction < 0.5 else "high"})
    return not hits, hits


def tolerance(n):
    """The -2 ln L difference below which two fits count as the same optimum: ``AGREE_FRACTION`` of one chi2 per datum."""
    return AGREE_FRACTION * max(int(n), 1)


def nested(rungs, n):
    """Fails when a rung fits worse than a rung it contains. ``rungs`` is a list of
    ``{"rung": name, "minus2lnl": float, "contains": [names]}``; worse means larger -2 ln L by more than
    ``tolerance(n)``. ``detail`` lists the pairs."""
    by_name = {r["rung"]: r for r in rungs}
    bad = []
    for r in rungs:
        for below in r.get("contains", []):
            if below in by_name and r["minus2lnl"] > by_name[below]["minus2lnl"] + tolerance(n):
                bad.append({"rung": r["rung"], "contains": below, "minus2lnl": r["minus2lnl"],
                            "contained_minus2lnl": by_name[below]["minus2lnl"]})
    return not bad, bad


def starts_agree(minus2lnl, n, minimum=MIN_AGREE):
    """Fails when fewer than ``minimum`` of the starts are within ``tolerance(n)`` of the best -2 ln L.
    Returns (ok, {"at_best": count, "starts": count})."""
    values = np.asarray([x for x in minus2lnl if np.isfinite(x)], float)
    if values.size == 0:
        return False, {"at_best": 0, "starts": 0}
    count = int(np.sum(values <= values.min() + tolerance(n)))
    return count >= minimum, {"at_best": count, "starts": int(values.size)}


def family_complete(rest_flux, limit=FAMILY_FLUX_LIMIT):
    """Fails when the image fitted on top of the model carries more than ``limit`` of the model's flux (the family
    leaves out too much). ``rest_flux`` is that flux relative to the model's, or None when the image was not run
    (then the result is not a pass: ``(None, ...)``)."""
    if rest_flux is None or not np.isfinite(rest_flux):
        return None, {"rest_flux": None}
    return bool(rest_flux <= limit), {"rest_flux": float(rest_flux), "limit": limit}


def no_collapse(components, resolution):
    """Fails when a component has shrunk to something else. ``components`` is a list of dicts with a ``kind`` of
    ``"arc"`` (``length``), ``"ring"`` (``width``) or ``"companion"`` (``dra``, ``ddec``, from the star); an arc shorter
    than the resolution, a ring narrower than ``RING_FRACTION`` of it, a companion within ``COMPANION_ELEMENTS``
    resolution elements of the star."""
    bad = []
    for c in components:
        if c["kind"] == "arc" and c["length"] < resolution:
            bad.append({**c, "limit": resolution})
        elif c["kind"] == "ring" and c["width"] <= RING_FRACTION * resolution * (1.0 + BOUND_EDGE):
            bad.append({**c, "limit": RING_FRACTION * resolution})
        elif c["kind"] == "companion" and np.hypot(c["dra"], c["ddec"]) < COMPANION_ELEMENTS * resolution:
            bad.append({**c, "limit": COMPANION_ELEMENTS * resolution})
    return not bad, bad
