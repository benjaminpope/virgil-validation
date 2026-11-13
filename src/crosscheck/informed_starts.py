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
- The checks of plan section 5 as pure functions of plain numbers, and the ladder policy's agreement tests
  (``same_optimum``, ``start_agreement``, ``fit_standard``) and ``informative_triangles``, which take precomputed
  numbers: the calls that need virgil (a Gauss-Newton sigma, -2 ln L, residuals) are in ``scripts/contest_informed.py``.

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
    ``n`` (chi2 of the fit on the *quoted* errors), ``rms``, ``short_v2`` (the largest V^2 - 3 sigma on the shortest quarter
    of the baselines: pure noise on a source resolved everywhere gives about zero, not the noise's maximum),
    ``longest_tenth`` (the median |V| on the longest tenth: a second estimate of ``compact``) and
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
            "rms": float(np.sqrt(np.mean((model - v2) ** 2))), "short_v2": float(np.max(v2[shortest] - 3.0 * err[shortest])),
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
SHORT_V2_MIN = 0.03  # extent_known: V^2 - 3 sigma on the shortest quarter of baselines must reach this
EXTENDED_MIN = 0.05  # extent_known: the moment fit's extended light must be at least this fraction of the flux


def scales_of(b_max, b_min, moments=None, field=None, field_source=None, tapered=False):
    """The extent, reach and pixel of plan section 2, from the largest and smallest baselines in units of the
    wavelength (cycles per radian) ``b_max``, ``b_min``, the visibility-moment fit ``moments`` of ``moment_fit`` (or
    None), and the contest's field (mas) where it gave one (``tapered`` if it also tapered the data).

    extent E: the given field; otherwise 4 x the extended light's major FWHM, at least 8 resolution elements.
    reach S: half the given field when the contest tapered it, else 3 lambda / B_min.
    pixel: lambda / (4 B_max), the field being E; more than ``MAX_PIXELS`` a side coarsens the pixel, and says so.
    When the data bound the extent only from below (V^2 - 3 sigma below ``SHORT_V2_MIN`` on every short baseline, the
    fitted size at the top of its grid, or under ``EXTENDED_MIN`` of the flux in extended light, so that the size is
    not a measurement) and no field was given, E is the reach and ``extent_known`` is False."""
    resolution = 1.0 / (2.0 * b_max) / MAS
    fringe = 1.0 / b_min / MAS  # lambda / B_min
    warnings = []
    if tapered and field:
        reach, reach_source = 0.5 * field, "half the given field (the contest tapered the data)"
    else:
        reach, reach_source = 3.0 * fringe, "3 x lambda / B_min"
    bounded_below = (moments is None or moments["short_v2"] < SHORT_V2_MIN or moments["at_upper_edge"]
                     or moments.get("extended", 1.0) < EXTENDED_MIN)
    if field:
        extent, source, known = float(field), f"the contest's field ({field_source})", True
    elif moments is not None and not bounded_below:
        extent = max(4.0 * moments["fwhm"], 8.0 * resolution)
        source, known = "4 x the major FWHM of the visibility-moment fit (at least 8 resolution elements)", True
    else:
        extent, known = reach, False
        source = "the reach: the data bound the extent only from below, and no field was given"
        warnings.append(f"extent_unconstrained: V^2 - 3 sigma is below {SHORT_V2_MIN} on every short baseline, the fitted size "
                        f"is at the top of its grid, or the extended light is under {EXTENDED_MIN} of the flux, so the "
                        "extent is only a lower limit")
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


def tiled_map(evaluate, half, step, tile=256, max_axis=4096, memo=None):
    """A map of ``evaluate(dra_axis, ddec_axis) -> (len(dra_axis), len(ddec_axis))`` over [-half, half]^2 at ``step``
    (mas), computed in tiles of ``tile`` points a side so memory stays bounded however wide the reach. Returns
    ``(axis, surface, step)``; an axis longer than ``max_axis`` points is coarsened (and ``step`` says by how much).
    ``memo(i, j, compute)``, when given, returns the block of the tile starting at (i, j), calling ``compute()`` for it
    only if it has not been computed before (a caller that keeps tiles on disk resumes a map)."""
    n = int(round(2.0 * half / step)) + 1
    if n > max_axis:
        n = max_axis
    axis = np.linspace(-half, half, n)
    step = float(axis[1] - axis[0]) if n > 1 else 0.0
    surface = np.full((n, n), -np.inf, dtype=np.float32)
    for i in range(0, n, tile):
        for j in range(0, n, tile):
            def compute(i=i, j=j):
                return np.asarray(evaluate(axis[i:i + tile], axis[j:j + tile]), float)

            block = np.asarray(compute() if memo is None else memo(i, j, compute), float)
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
START_AGREE = 6  # starts_agree, first rung: at least this many of the broad starts end at the best optimum
START_MARGIN = 25.0  # starts_agree, higher rungs: the next distinct optimum is worse than the best by at least this in -2 ln L
SAME_DELTA = 2.0  # same optimum, and nested: -2 ln L within this
SAME_SIGMA = 3.0  # same optimum: every parameter within this many combined sigma (or within the practical tolerances below)
SIZE_TOLERANCE = 0.05  # same optimum, practical tolerance: a size within this fraction
ANGLE_TOLERANCE = 5.0  # same optimum, practical tolerance: an angle within this many degrees (modulo its period)
POSITION_ELEMENTS = 0.25  # same optimum, practical tolerance: a position within this fraction of a resolution element
FIT_RMS_MAX = 2.0  # fit_standard: normalised residual rms of the last fit, on V^2 and on informative closure phases
INFORMATIVE_SNR = 3.0  # a closure phase is informative when all three V^2 of its triangle exceed this many sigma
INFORMATIVE_SHARE = 0.05  # closure phases are flagged as carrying no information when fewer than this share of triangles are informative
FAMILY_FLUX_LIMIT = 0.2  # family_complete: the image on top may carry at most this fraction of the model's flux
RING_FRACTION = 0.2  # no_collapse: a ring narrower than this fraction of the resolution
WIDTH_FLOOR = 0.05  # a ring's width prior (FWHM) starts at this fraction of the resolution: below RING_FRACTION, so a collapse is inside the prior
COMPANION_ELEMENTS = 2.0  # no_collapse: a companion closer than this many resolution elements to the star
PERIODIC = ("angle", "full_angle")  # prior kinds that wrap: no_bounds does not apply
SIZE_LEAVES_TOLERANT = ("diam", "fwhm", "sigma", "radius", "r_out", "width", "length")  # parameters compared with the size tolerance
ANGLE_PERIOD = {"pa": 180.0, "inc": 180.0, "az_pas": 360.0}  # angle parameters, and the period they are compared modulo


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


def nested(rungs, delta=SAME_DELTA):
    """Fails when a rung fits worse than a rung it contains. ``rungs`` is a list of
    ``{"rung": name, "minus2lnl": float, "contains": [names]}``; worse means larger -2 ln L by more than ``delta``
    (``SAME_DELTA``, the difference under which two fits are one optimum). ``detail`` lists the pairs."""
    by_name = {r["rung"]: r for r in rungs}
    bad = []
    for r in rungs:
        for below in r.get("contains", []):
            if below in by_name and r["minus2lnl"] > by_name[below]["minus2lnl"] + delta:
                bad.append({"rung": r["rung"], "contains": below, "minus2lnl": r["minus2lnl"],
                            "contained_minus2lnl": by_name[below]["minus2lnl"]})
    return not bad, bad


# ---------------------------------------------------------------------------------------------------------------
# The ladder policy: what the data can tell, when two fits are one optimum, whether starts agree, the fit standard.
# Every function takes numbers the caller computed (the Gauss-Newton sigma and -2 ln L need virgil: they are made in
# scripts/contest_informed.py), so each is testable without a fit.


def informative_triangles(vis, d_vis, legs, v2=True, vis_index=None, phi_index=None, snr=INFORMATIVE_SNR):
    """One boolean per closure phase: all three baselines of its triangle have V^2 above ``snr`` sigma.

    ``vis`` and ``d_vis`` are the visibility observable and its error (V^2 when ``v2``, else |V|, whose V^2 error is
    2 |V| e); ``vis_index`` places each row on its sample (default: row i is sample i); ``legs`` is the three arrays of
    sample indices of each triangle's baselines, and ``phi_index`` selects the closure phases that exist (default: all).
    A closure phase over a baseline with no V^2 is not informative. With no triangles the result is empty."""
    vis, err = np.ravel(np.asarray(vis, float)), np.ravel(np.asarray(d_vis, float))
    legs = np.stack([np.ravel(np.asarray(i)) for i in legs]) if legs is not None and len(legs) else np.zeros((3, 0), int)
    if legs.shape[1] == 0:
        return np.zeros(0, bool)
    rows = np.arange(vis.size) if vis_index is None else np.ravel(np.asarray(vis_index))
    snr_of = np.zeros(int(max(legs.max(), rows.max() if rows.size else 0)) + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr_of[rows] = np.nan_to_num(vis / err if v2 else vis / (2.0 * err))  # V^2 = |V|^2, its error 2 |V| e
    chosen = legs[:, np.arange(legs.shape[1]) if phi_index is None else np.ravel(np.asarray(phi_index))]
    return np.all(snr_of[chosen] > snr, axis=0)


def information_record(n_triangles, n_informative, median_v2, median_cp_error_rad=None):
    """The information check, before any fit: median V^2, median quoted closure-phase error, how many triangles have all
    three V^2 above ``INFORMATIVE_SNR`` sigma, and whether closure phases carry information (at least
    ``INFORMATIVE_SHARE`` of the triangles). Closure phases that carry none are still fitted; the fit is judged on V^2
    and nothing is oriented by them."""
    informative = bool(n_triangles and n_informative >= INFORMATIVE_SHARE * n_triangles)
    return {"median_v2": float(median_v2),
            "median_quoted_cp_error_deg": None if median_cp_error_rad is None else float(np.degrees(median_cp_error_rad)),
            "triangles": int(n_triangles), "triangles_all_v2_above_3_sigma": int(n_informative),
            "closure_phases_informative": informative,
            "judged_on": "V2 and closure phases" if informative else "V2 only: the closure phases carry no information"}


def differs(values, ref, resolution):
    """What of a fit differs from ``ref`` by more than the practical tolerances: a size by more than ``SIZE_TOLERANCE``, an
    angle by more than ``ANGLE_TOLERANCE`` degrees (position angles modulo their period), a position by more than
    ``POSITION_ELEMENTS`` of a resolution element. Returns the list of parameter names."""
    why = []
    for k, x in values.items():
        y, leaf = ref[k], k.rsplit(".", 1)[-1]
        if leaf in SIZE_LEAVES_TOLERANT and abs(x - y) > SIZE_TOLERANCE * abs(y):
            why.append(k)
        elif leaf in ANGLE_PERIOD:
            d = (x - y) % ANGLE_PERIOD[leaf]
            if min(d, ANGLE_PERIOD[leaf] - d) > ANGLE_TOLERANCE:
                why.append(k)
        elif leaf in ("dra", "ddec") and abs(x - y) > POSITION_ELEMENTS * resolution:
            why.append(k)
    return why


def canonical(values):
    """Rings of a disk relabelled in order of radius (their letters are arbitrary); weights relative to the first."""
    letters = sorted({k.split(".")[2] for k in values if ".profile." in k})
    if len(letters) < 2:
        return values
    pre, out = next(k for k in values if ".profile." in k).rsplit(".profile.", 1)[0] + ".profile.", dict(values)
    order = sorted(letters, key=lambda letter: values[f"{pre}{letter}.radius"])
    amp = {letter: values.get(f"{pre}{letter}.weight", 1.0) for letter in letters}
    for new, old in zip(letters, order):
        out |= {f"{pre}{new}.radius": values[f"{pre}{old}.radius"], f"{pre}{new}.sigma": values[f"{pre}{old}.sigma"]}
        out |= {f"{pre}{new}.weight": amp[old] / amp[order[0]]} if new != letters[0] else {}
    return out


def same_optimum(a, b, resolution):
    """``a`` and ``b`` (``{"values", "sigma", "chi2q"}``: fitted values, their 1 sigma from the joint Gauss-Newton curvature, and
    -2 ln L on the quoted errors) are one optimum when their -2 ln L differ by under ``SAME_DELTA`` and every parameter
    agrees within ``SAME_SIGMA`` combined sigma, or, for a size, angle or position, within the practical tolerances of
    `differs` where those are looser. Returns the list of what differs (empty: one optimum)."""
    va, vb = canonical(a["values"]), canonical(b["values"])
    tol, why = set(differs(va, vb, resolution)), (["-2lnL"] if abs(a["chi2q"] - b["chi2q"]) >= SAME_DELTA else [])
    for k in va:
        leaf = k.rsplit(".", 1)[-1]
        period = ANGLE_PERIOD.get(leaf)
        if (leaf in SIZE_LEAVES_TOLERANT or leaf in ANGLE_PERIOD or leaf in ("dra", "ddec")) and k not in tol:
            continue
        d = va[k] - vb[k]
        d = (d + period / 2) % period - period / 2 if period else d
        if abs(d) > SAME_SIGMA * np.hypot(a["sigma"][k], b["sigma"][k]):
            why.append(k)
    return why


BROAD_ORIGINS = ("grid",)  # origins of the broad stratified starts of the first rung
CONTROL_ORIGINS = ("moments", "random")  # starts that are not counted: the moment start and control draws from the priors


def start_agreement(starts, resolution, bounds):
    """Start agreement of one rung. ``starts`` is a list of ``{"origin", "values", "sigma", "chi2q"}`` (``chi2q``: -2 ln L on
    the quoted errors, gains marginalised; ``sigma`` as in `same_optimum`), in the order they were run.

    First rung (every counted start is a broad ``grid`` start): at least ``START_AGREE`` of them at the best optimum. Above:
    the best map peak and the jittered starts (every start when the rung places nothing) are all at the best optimum, and
    the best distinct optimum is worse by at least ``START_MARGIN``. The moment start and control draws are not counted.
    ``bounds`` is ``{path: (low, high, log, kind)}``; parameters whose sigma exceeds half their prior range, or their value,
    are listed as not constrained by the data. Returns a record with ``passes``."""
    group = [dict(f) for f in starts if f["origin"] not in CONTROL_ORIGINS and "values" in f]
    if not group:
        return {"passes": False, "n": 0, "at_best": 0, "best_chi2_quoted": None, "margin_to_distinct_other": None, "others": [],
                "not_constrained_by_the_data": {}, "rule": None}
    best = min(group, key=lambda f: f["chi2q"])
    for f in group:
        f["differs_in"] = same_optimum(f, best, resolution)
    others = [f for f in group if f["differs_in"]]
    margin = min((f["chi2q"] for f in others), default=np.inf) - best["chi2q"]
    broad = all(f["origin"] in BROAD_ORIGINS for f in group)
    first_peak = next((f for f in group if f["origin"] == "map"), None)
    mine = [f for f in group if f["origin"] != "map"] + ([first_peak] if first_peak is not None else [])  # the best peak and its jitters
    ok = len(group) - len(others) >= START_AGREE if broad else all(not f["differs_in"] for f in mine) and margin >= START_MARGIN
    free = {k: float(best["sigma"][k]) for k in best["values"]
            if k in bounds and (best["sigma"][k] > 0.5 * (bounds[k][1] - bounds[k][0]) or best["sigma"][k] > abs(best["values"][k]))}
    return {"passes": bool(ok), "rule": f"{START_AGREE} of the broad starts" if broad else "best peak and jitters, next optimum by the margin",
            "n": len(group), "at_best": len(group) - len(others), "best_chi2_quoted": float(best["chi2q"]),
            "margin_to_distinct_other": float(margin),
            "others": [{"origin": f["origin"], "chi2_quoted": float(f["chi2q"]), "differs_in": f["differs_in"][:6]} for f in others],
            "not_constrained_by_the_data": free}


def fit_standard(rms_v2, rms_cp=None, n_triangles=0, limit=FIT_RMS_MAX):
    """The fit standard after the last fit, with the error terms free: the normalised residual rms (residual over the effective
    error) on V^2, and on the informative closure phases when there are any, is at most ``limit``. A fit that misses it
    converged, perhaps, but is **incomplete**: the model leaves structure the data show. A missing or non-finite rms is a
    failure, never a pass. Returns ``(ok, detail)``; ``detail["verdict"]`` is ``"complete"`` or ``"incomplete"``."""
    judged = {"V2": rms_v2}
    if n_triangles and rms_cp is not None:
        judged["closure phase (informative triangles)"] = rms_cp
    missing = [k for k, v in judged.items() if v is None or not np.isfinite(v)]
    over = {k: float(v) for k, v in judged.items() if k not in missing and v > limit}
    ok = not missing and not over
    return ok, {"verdict": "complete" if ok else "incomplete", "limit": limit, "rms": {k: (None if v is None else float(v)) for k, v in judged.items()},
                "informative_triangles": int(n_triangles), "over_the_limit": over, "not_measured": missing}


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
