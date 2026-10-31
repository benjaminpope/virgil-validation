"""Held-out synthetic scenes and coverages for the Toeplitz re-examination (``design/toeplitz_reexam_042.md``, G2).

numpy only. None of these was used to develop virgil's Toeplitz term or its tests (those used random log-normal
images on uniform discs of uv points, and IM Lup's R_mpol solution). Every random draw comes from
``seed(scene, role)``, a hash of names fixed in the criteria file, so no seed can be chosen by trying several.
"""

import hashlib

import numpy as np

from crosscheck import radio_exact as rx
from crosscheck.radio_innovations import MAS, pixel_positions


def seed(scene, role):
    """The registered seed rule: the first 8 hex digits of sha256("toeplitz-reexam-042/<scene>/<role>")."""
    return int(hashlib.sha256(f"toeplitz-reexam-042/{scene}/{role}".encode()).hexdigest()[:8], 16)


def image(kind, npix, pixel_mas, rng):
    """A non-negative brightness, unit sum. ``ringed_disc``: an inclined disc with three Gaussian rings and two gaps;
    ``sources_on_halo``: five point sources spanning 1e4 in brightness (one in a corner pixel) on a faint Gaussian halo;
    ``edge_filled``: a log-normal random field filling the whole grid, edges included (every lag in use)."""
    east, north = (x.reshape(npix, npix) for x in pixel_positions(npix, pixel_mas))
    half = npix * pixel_mas / 2
    if kind == "ringed_disc":
        inc, pa = np.deg2rad(rng.uniform(30, 60)), np.deg2rad(rng.uniform(0, 180))
        x = east * np.cos(pa) - north * np.sin(pa)
        y = (east * np.sin(pa) + north * np.cos(pa)) / np.cos(inc)
        r = np.hypot(x, y) / half
        img = np.exp(-((r / 0.6) ** 2))
        for r0, a in zip((0.15, 0.35, 0.55), (1.0, 0.6, 0.4), strict=True):
            img += a * np.exp(-0.5 * ((r - r0) / 0.02) ** 2)
        for r0 in (0.25, 0.45):
            img *= 1 - 0.9 * np.exp(-0.5 * ((r - r0) / 0.015) ** 2)
        img[r > 0.9] = 0.0
    elif kind == "sources_on_halo":
        img = 1e-4 * np.exp(-0.5 * (np.hypot(east, north) / (0.3 * half)) ** 2)
        rows = list(rng.integers(0, npix, 4)) + [0]
        cols = list(rng.integers(0, npix, 4)) + [npix - 1]
        for r, c, a in zip(rows, cols, 10.0 ** np.linspace(0, -4, 5), strict=True):
            img[r, c] += a
    elif kind == "edge_filled":
        k = np.fft.fftfreq(npix)
        kk = np.hypot(*np.meshgrid(k, k))
        field = np.real(np.fft.ifft2(np.fft.fft2(rng.normal(size=(npix, npix))) * np.exp(-((kk / 0.08) ** 2))))
        img = np.exp(3.0 * field / field.std())
    else:
        raise ValueError(kind)
    img = np.clip(img, 0.0, None)
    return img / img.sum()


def coverage(n_ant, n_times, freqs_hz, reach, pixel_mas, rng):
    """Earth-rotation tracks: ``n_ant`` antennas at log-uniform radii (an ALMA-like compact core and long arms), baselines
    rotated through ``n_times`` hour angles over 8 h at declination -30 deg, scaled so the longest reaches ``reach`` of the
    pixel Nyquist limit ``1 / (2 pixel)`` at the highest frequency. Returns ``(u, v, ddid)``, one ddid per frequency."""
    r = np.exp(rng.uniform(np.log(15.0), np.log(1500.0), n_ant))
    a = rng.uniform(0, 2 * np.pi, n_ant)
    x, y = r * np.cos(a), r * np.sin(a)
    i, j = np.triu_indices(n_ant, 1)
    bx, by = x[i] - x[j], y[i] - y[j]
    h = np.linspace(-np.pi / 3, np.pi / 3, n_times)
    dec = np.deg2rad(-30.0)
    us = (np.outer(np.cos(h), bx) - np.outer(np.sin(h), by)).ravel()
    vs = (np.outer(np.sin(dec) * np.sin(h), bx) + np.outer(np.sin(dec) * np.cos(h), by)).ravel()
    q_nyq = 1.0 / (2.0 * pixel_mas * MAS)
    scale = reach * q_nyq / np.max(np.hypot(us, vs)) / max(freqs_hz)
    u = np.concatenate([us * scale * f for f in freqs_hz])
    v = np.concatenate([vs * scale * f for f in freqs_hz])
    ddid = np.repeat(np.arange(len(freqs_hz)), len(us))
    return u, v, ddid


def make_case(spec):
    """The data set of one registered scene (``spec``: a scene entry of the criteria file). Returns ``(case, truth)``: the
    ``radio_exact.Case`` with simulated data, and the true values (``b``, flux, gain and offset of every observation, the
    per-ddid true error scale). Noise: Gaussian, variance ``s_d^2 / w_j`` in each of Re and Im."""
    name = spec["name"]
    rng = np.random.default_rng(seed(name, "truth"))
    npix, pix = spec["npix"], spec["pixel_mas"]
    b = image(spec["image"], npix, pix, rng)
    n_obs, n_ddid = spec["n_obs"], spec["n_ddid"]
    freqs = 230e9 * (1 + 0.04 * np.arange(n_ddid) / max(n_ddid - 1, 1))
    u, v, ddid = coverage(spec["n_ant"], spec["n_times"], freqs, spec["reach"], pix, rng)
    obs_of_ddid = np.arange(n_ddid) % n_obs
    obs = obs_of_ddid[ddid]
    weight = 10 ** rng.normal(3.0, 0.3, len(u))
    flux = 0.25
    gain = np.concatenate([[1.0], 1 + 0.05 * rng.normal(size=n_obs - 1)])
    bound = spec.get("offset_bound_mas") or 0.0
    offset = np.vstack([[0.0, 0.0], rng.uniform(-0.3 * bound, 0.3 * bound, (n_obs - 1, 2))])
    s_true = 10 ** rng.uniform(-0.15, 0.15, n_ddid)
    blank = rx.Case(u, v, np.zeros(len(u), complex), weight, ddid, obs, npix, pix)
    m = rx.model(blank, b, flux, gain, offset)
    sd = s_true[ddid] / np.sqrt(weight)
    vis = m + sd * (rng.normal(size=len(u)) + 1j * rng.normal(size=len(u)))
    case = rx.Case(u, v, vis, weight, ddid, obs, npix, pix)
    return case, dict(b=b, flux=flux, gain=gain, offset=offset, sigma=s_true)


def points(case, truth, spec, sigma_opt=None):
    """The registered evaluation points (design section 4): ``optimum`` (the truth; fitted error scales at ``sigma_hat``, so
    ``dL/ds = 0``: the cancellation case), ``moved`` (b x exp(0.3 z) renormalised, flux x 1.15, gains x (1 + 0.05 z), offsets
    uniform in half the bound, error scales x 1.1) and, with free offsets, ``edge`` (the moved point with every offset
    component at +-0.9 of the bound). Each is ``dict(eta, flux, gain, offset[, sigma])`` with gain and offset of every
    observation, the reference first."""
    name, bound = spec["name"], spec.get("offset_bound_mas") or 0.0
    eta = np.log(np.maximum(truth["b"], 1e-300))
    base = dict(eta=eta, flux=truth["flux"], gain=truth["gain"].copy(), offset=truth["offset"].copy())
    if sigma_opt is not None:
        base["sigma"] = np.asarray(sigma_opt, float)
    out = {"optimum": base}
    rng = np.random.default_rng(seed(name, "moved"))
    moved = dict(base, eta=eta + 0.3 * rng.normal(size=eta.shape), flux=1.15 * truth["flux"],
                 gain=np.concatenate([[1.0], truth["gain"][1:] * (1 + 0.05 * rng.normal(size=case.n_obs - 1))]))
    moved["offset"] = np.vstack([[0.0, 0.0], rng.uniform(-0.5 * bound, 0.5 * bound, (case.n_obs - 1, 2))])
    if sigma_opt is not None:
        moved["sigma"] = 1.1 * base["sigma"]
    out["moved"] = moved
    if spec["offsets"] == "spread":
        rng = np.random.default_rng(seed(name, "edge"))
        out["edge"] = dict(moved, offset=np.vstack([[0.0, 0.0], 0.9 * bound * rng.choice([-1.0, 1.0], (case.n_obs - 1, 2))]))
    return out
