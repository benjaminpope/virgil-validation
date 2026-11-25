"""Ring features of an image: centre, diameter, width, orientation, asymmetry, central fraction.

Written from the equations of Lai et al. 2026, Nature 659, 317, Methods "Image-domain feature
extraction" (eqs 4–9), which follow the EHT ring extraction (EHT Collaboration 2019, ApJL 875, L4,
section 9; REX in eht-imaging). Neither their code nor virgil's was read to write this.

Conventions: image rows run North (row 0) to South, columns East (column 0) to West; angles are
East of North. Polar resampling is bilinear (``scipy.ndimage.map_coordinates``, order 1).

- Centre (eq 4): the trial centre minimising ``σ_r / ⟨r_pk⟩``, where ``r_pk(θ)`` is the radius of
  peak brightness along each angle (eq 5).
- Diameter: ``2⟨r_pk⟩_θ``; its ``_std`` is ``2σ_r``.
- Width (eq 6): mean over angle of the FWHM of ``I(r, θ) − I_floor``, ``I_floor = ⟨I(2⟨r_pk⟩, θ)⟩_θ``.
- Orientation (eq 7) and asymmetry (eq 8) from ``∫ I(r, θ) e^{iθ} dθ`` at each radius in
  ``⟨r_pk⟩ ± w/2``: the angle's circular mean (``_std``: circular SD) and the mean of
  ``|∫ I e^{iθ} dθ| / ∫ I dθ``. A ring ``1 + a cos(θ − η)`` has ``A = a/2``.
- Central fraction (eq 9): mean brightness over the disk ``r < f_c_radius · ⟨r_pk⟩`` over
  ``⟨I(⟨r_pk⟩, θ)⟩_θ``.

``floor_factor`` (2) and ``fc_radius`` (0.2) are Lai et al.'s; EHT Paper IV uses 50 μas and 5 μas,
which for a 21 μas ring are about 2.5 and 0.25.
"""

import numpy as np
from scipy import ndimage, optimize


def polar(image, pixel_scale, centre, r, theta):
    """``I(r, θ)`` on the grids ``r`` (same unit as ``pixel_scale``) and ``theta`` (radians East of
    North), about ``centre`` = (East, North) offset from the image centre. Shape (len(r), len(theta))."""
    ny, nx = image.shape
    east = centre[0] + r[:, None] * np.sin(theta)[None, :]
    north = centre[1] + r[:, None] * np.cos(theta)[None, :]
    rows = (ny - 1) / 2 - north / pixel_scale
    cols = (nx - 1) / 2 - east / pixel_scale
    return ndimage.map_coordinates(image, [rows, cols], order=1, mode="constant", cval=0.0)


def _peak_radius(profile, r):
    """Radius of the maximum of each column, refined by a parabola through the three samples."""
    k = np.argmax(profile, axis=0)
    out = r[k].astype(float)
    inner = (k > 0) & (k < len(r) - 1)
    cols = np.nonzero(inner)[0]
    y0, y1, y2 = profile[k[cols] - 1, cols], profile[k[cols], cols], profile[k[cols] + 1, cols]
    curv = y0 - 2 * y1 + y2
    good = curv < 0
    shift = np.where(good, 0.5 * (y0 - y2) / np.where(good, curv, -1.0), 0.0)
    out[cols] += np.clip(shift, -0.5, 0.5) * (r[1] - r[0])
    return out


def _fwhm(profile, r, k):
    """Full width at half the value at index ``k`` of a 1D profile, by linear interpolation of
    the half-level crossings on each side; NaN when a side never drops below half."""
    half = 0.5 * profile[k]
    left = np.nonzero(profile[:k] < half)[0]
    right = np.nonzero(profile[k:] < half)[0]
    if half <= 0 or not len(left) or not len(right):
        return np.nan
    i, j = left[-1], k + right[0]
    rl = r[i] + (half - profile[i]) * (r[i + 1] - r[i]) / (profile[i + 1] - profile[i])
    rr = r[j - 1] + (half - profile[j - 1]) * (r[j] - r[j - 1]) / (profile[j] - profile[j - 1])
    return rr - rl


def _circular(angles):
    z = np.mean(np.exp(1j * angles))
    return np.angle(z), np.sqrt(-2 * np.log(max(abs(z), 1e-300)))


def ring_features(image, pixel_scale, *, n_theta=360, oversample=4, r_min=None, search=None,
                  search_step=None, floor_factor=2.0, fc_radius=0.2):
    """Ring features of ``image`` (2D array, pixel size ``pixel_scale``, any angular unit).

    ``r_min`` excludes a central peak from the radius search (default 2 pixels); ``search`` is the
    half-width of the square of trial centres about the image centre (default 1/8 of the field) at
    ``search_step`` (default one pixel, on at most 90 angles), polished with Nelder–Mead. Angles whose peak falls on
    the edge of the radius range count as failures in the centre cost (a guard the equations leave
    implicit). Returns a dict of floats:
    ``centre`` (East, North), ``diameter``, ``width``, ``pa`` (degrees East of North, in [0, 360)),
    ``asymmetry``, ``central_fraction``, with ``_std`` for diameter, width, pa and asymmetry.
    """
    image = np.asarray(image, float)
    ny, nx = image.shape
    ps = float(pixel_scale)
    half = 0.5 * min(ny, nx) * ps
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    dr = ps / oversample
    r_min = 2 * ps if r_min is None else r_min
    r = np.arange(0.0, half, dr)
    search = half / 4 if search is None else search
    step = ps if search_step is None else search_step
    use = r >= r_min
    coarse = theta[:: max(1, n_theta // 90)]

    def rpk(c, th=theta):
        return _peak_radius(polar(image, ps, c, r[use], th), r[use])

    edge = r[use][[0, -1]]

    def cost(c, th=theta):
        # A peak on the edge of the radius range is not a ring crossing (a centre on a bright spot
        # gives r_pk = r_min at every angle and sigma_r = 0): each such angle costs 1.
        p = rpk(c, th)
        bad = (p <= edge[0] + 0.5 * dr) | (p >= edge[1] - 0.5 * dr)
        good = p[~bad]
        if len(good) < 2:
            return 2.0
        return np.std(good) / np.mean(good) + np.mean(bad)

    grid = np.arange(-search, search + 0.5 * step, step)
    trials = [(cost((x, y), coarse), x, y) for x in grid for y in grid]
    _, x0, y0 = min(trials)
    res = optimize.minimize(cost, [x0, y0], method="Nelder-Mead",
                            options={"xatol": 1e-3 * ps, "fatol": 1e-9, "initial_simplex":
                                     [[x0, y0], [x0 + step / 2, y0], [x0, y0 + step / 2]]})
    centre = tuple(float(v) for v in (res.x if res.fun <= cost((x0, y0)) else (x0, y0)))

    peaks = rpk(centre)
    rbar = float(np.mean(peaks))
    prof = polar(image, ps, centre, r, theta)  # (n_r, n_theta), r from 0
    floor = float(np.mean(polar(image, ps, centre, np.array([floor_factor * rbar]), theta)))
    widths = []
    for t in range(n_theta):
        k = int(np.argmin(np.abs(r - peaks[t])))
        widths.append(_fwhm(prof[:, t] - floor, r, k))
    widths = np.asarray(widths)
    w = float(np.nanmean(widths))

    band = (r >= rbar - w / 2) & (r <= rbar + w / 2)
    if not band.any():
        band = np.abs(r - rbar) == np.min(np.abs(r - rbar))
    m1 = np.mean(prof[band] * np.exp(1j * theta)[None, :], axis=1)
    m0 = np.mean(prof[band], axis=1)
    pa, pa_std = _circular(np.angle(m1))
    asym = np.abs(m1) / m0

    inner = r < fc_radius * rbar
    disk = np.sum(prof[inner] * r[inner, None]) / np.sum(r[inner, None] * np.ones_like(theta)[None, :]) \
        if inner.sum() > 1 else float(np.mean(prof[0]))
    ring = float(np.mean(polar(image, ps, centre, np.array([rbar]), theta)))

    return {
        "centre": centre,
        "diameter": 2 * rbar, "diameter_std": 2 * float(np.std(peaks)),
        "width": w, "width_std": float(np.nanstd(widths)),
        "pa": float(np.degrees(pa) % 360), "pa_std": float(np.degrees(pa_std)),
        "asymmetry": float(np.mean(asym)), "asymmetry_std": float(np.std(asym)),
        "central_fraction": float(disk / ring),
        "floor": floor,
    }


def ring_image(n, pixel_scale, radius, sigma, *, a=0.0, pa=0.0, centre=(0.0, 0.0), oversample=8):
    """Reference image: a Gaussian-profile ring ``exp(−(r − radius)² / 2σ²) (1 + a cos(θ − pa))``
    (``pa`` in degrees East of North) about ``centre`` (East, North), each pixel averaged over
    ``oversample²`` sub-pixels. Its peak radius is ``radius`` up to the ``r``-independent profile,
    its ``A`` is ``a/2`` and its orientation is ``pa``."""
    sub = (np.arange(n * oversample) + 0.5) / oversample - 0.5  # pixel index of sub-samples
    c = (n - 1) / 2
    north = (c - sub)[:, None] * pixel_scale - centre[1]
    east = (c - sub)[None, :] * pixel_scale - centre[0]
    rr = np.hypot(east, north)
    th = np.arctan2(east, north)
    img = np.exp(-0.5 * ((rr - radius) / sigma) ** 2) * (1 + a * np.cos(th - np.radians(pa)))
    return img.reshape(n, oversample, n, oversample).mean(axis=(1, 3))
