"""A parametric model ladder for the 2004 contest's data1 and data2 (virgil only).

    python scripts/contest_parametric.py --dataset data2 --out DIR [--smoke]

The 2004 Interferometric Imaging Beauty Contest's blind sets came with no
description. Knowing that data2 is "a limb darkened star with one or more
spots" (the pre-submission information of the other 2004 entries, in the
local paper notes) and may have a companion is *domain knowledge*: this fit is
an **informed** one, badged as such, not blind. The published truth
(scripts/score_2004.py) is used only to score the winner and to fill the
recovery table at the end. It never sets a start, a prior or a choice here.

**The ladder**, fitted in order, each rung to the same data:

1. ``ud``: a uniform disk.
2. ``ld``: a limb-darkened disk (linear law).
3. ``eld``: an elliptical limb-darkened disk (``EllipticalLimbDarkenedDisk``).
4. ``eld+spot``: plus one bright spot, a uniform disk (or a Gaussian,
   ``--spot-shape``) at a free position on the star with a free flux.
5. ``eld+spot+comp``: plus a companion (a uniform disk of free diameter: a
   point source is its small-diameter limit, and the data decide), found
   first by a ``linear_flux_grid`` search over ±λ/B_min with the primary masked.
6. ``eld+2spot+comp``: plus a second spot, run only if the best rung still
   has χ²/N above ``SECOND_SPOT_IF`` (the evidence wants it only when the
   residuals show structure).

One diagnostic rung is added, ``eld+comp`` (no spot): the companion's flux
is a tenth of the star's, so against a companion-free star every earlier rung
is dominated by it, and the spot search of rung 4 would chase its residual.
The spot rungs therefore search with the companion in the model, and
``eld+spot+comp`` is also started from the ``eld+comp`` fit.

**The model.** ``GainSystem``, a virgil ``System`` (star, spots, companion,
weights are fluxes relative to the star's) times one real gain g on the
visibilities, V = g Σ f_i V_i / Σ f_i, so V² scales by g² and closure phases
are untouched. The 2004 rules allowed calibration errors, and data2's V² sit
7–15 % above any unit-sum model (vv#104, scripts/score_2004.py): the data
normalise by the star and companion only, leaving out the spot's brightening.
Every rung has g (log-uniform on [0.5, 2], a scale), so the ladder compares
like with like. g is not tied to the spot's flux; the implied g is compared
with the fitted spot flux in the report (a consistency check, not a prior).
T3AMP is in the file, which virgil's likelihood does not read; it is *not*
fitted, and its χ²/N under the same g (T3AMP scales as g³) is an independent
check, reported per rung.

**Priors** (Jeffreys: log-uniform scales and fluxes, uniform positions and
angles). The diameter is log-uniform on [0.5, 8] beams; u on [0, 1]; the
axis ratio on [0.2, 1]; the PA on [-90°, 270°] (periodic, so no plausible PA
is near a bound); the spot's diameter on [0.1, 2] beams and its flux (relative
to the star) on [1e-3, 1]; the spot's position uniform in the *star's frame*
(x along the major axis, y across it, in units of the radii, on [-1.2, 1.2]),
so that it is a position on the star; the companion's offsets uniform over
±λ/B_min, its flux log-uniform on [1e-4, 1] and its diameter on [0.02, 2]
beams. Only bright spots are
representable (a System's weights are non-negative).

**Starts** (``crosscheck.starts``): the size, axis ratio and PA from the V²
second moments, then a diameter scan from 0.5× to 2× that diameter to pick the
visibility null's branch, then jittered multistarts moved at least 5 % of each
prior's range from its bounds (``contest_images.inside``). Spot and
companion starts are the peaks of linear flux maps (a spot diameter of half a
beam) with the model's other parts in place, off the star for the companion
and on it for the spot: the twelve highest peaks are ranked by the χ² they
actually give (a linear map's SNR is not monotonic in it, and against an
imperfect primary its highest peaks are residual structure) and the best
three are the starts.

**Choosing.** BIC = χ² + k ln N (N = 325 V² and closure phases), lowest wins;
the Laplace evidence is reported beside it: log Z = -χ²/2 + Σ ln(1/range) +
(k/2) ln 2π − ½ ln det JᵀJ, with J the Jacobian of the whitened residuals in
the coordinates where each prior is flat (log for scales), plus ln n! for n
identical spots. The terms common to all rungs (the Gaussian normalization
of the data) are left out. The ellipse's PA range counts as 180° (its two
equivalent modes). It is a Gauss–Newton Laplace approximation: a parameter
the data do not constrain makes it unreliable, and that is flagged by an
on-bounds mark.

**Uncertainties.** NUTS on the winner (numpyro_model, ``--chains`` chains
started at the MAP); the Laplace σ is reported beside it.

**Scoring.** The winner is rendered on score_2004's grid (301 pixels of
0.08 mas, East left, North up), scored with its σ/peak against the truth
rebuilt from the published model (as for the image arms), and the fitted
parameters are compared with the truth's in the output table only. The
published data2 σ/peak: BSMEM 0.116, WISARD 0.163, MIRA 0.532, VLBMEM 0.798;
our best image 0.250.

**Outputs** (``--out``), ``2004_<dataset>_parametric``:

- ``.json``: the ladder (χ²/N per observable, k, BIC, log Z, parameters, σ),
  the choice, the NUTS summary, the score and the recovery table;
- ``.npz``: ``ref_image``/``ref_fov`` (the winner, as the other arms store
  it), the NUTS samples and the Laplace covariance;
- ``.txt``: the report printed at the end.

Heavy: run on OzSTAR (ozstar_scripts job contest_imaging, ``--parametric``).
``--smoke`` is a short check that it runs (two starts, three steps, short
NUTS, no second spot).
"""

import argparse
import json
import math
import pathlib
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro.distributions as dist  # noqa: E402

import virgil.models as vm  # noqa: E402
from virgil.fitting import fit  # noqa: E402
from virgil.grid_fit import linear_flux_grid  # noqa: E402
from virgil.imaging import beam  # noqa: E402
from virgil.likelihood import numpyro_model, whitened_residuals  # noqa: E402
from virgil.oidata import OIData  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import contest_images as ci  # noqa: E402
import score_2004  # noqa: E402
from crosscheck import starts as linear_starts  # noqa: E402

N_STARTS, N_STARTS_SMOKE = 16, 2
SECOND_SPOT_IF = 1.1  # χ²/N of the best rung above which a second spot is tried
GAIN_RANGE = (0.5, 2.0)
MAX_AXIS = 201  # samples per axis of the linear maps (fewer in smoke runs)
SPOT_NAMES = ("spot", "spot2")
PEAKS = 12  # peaks of a linear map that are ranked by their actual χ² (the map's SNR is not monotonic in it)
COMP_DIAM = 0.3  # beams: a companion's starting diameter (the fit frees it)
KEEP = 3  # ... and the best of them kept as starts
SPOT_SEPARATION = 1.0  # minor beams: distinct peaks of a linear map are at least this far apart
SQRT_8LN2 = 2.0 * math.sqrt(2.0 * math.log(2.0))
SCORE_NPIX = score_2004.NPIX
SCORE_FOV = SCORE_NPIX * score_2004.PIXEL


# --- the model --------------------------------------------------------------


class GainSystem(vm.System):
    """A ``System`` whose visibilities are multiplied by a real ``gain`` (the
    data's normalisation, see the module docstring). Its parts are addressed
    as in a System (``star.diam``, ``comp.flux``), and ``gain`` is a path."""

    gain: jax.Array

    def __init__(self, gain=1.0, **named):
        super().__init__(**named)
        self.gain = jnp.asarray(gain, dtype=float)

    def model(self, u, v, wavel):
        return self.gain * super().model(u, v, wavel)

    def model_on_grid(self, u, v, wavel, grid):
        return self.gain * super().model_on_grid(u, v, wavel, grid)


def star_axes(p):
    """(a, b, sin pa, cos pa): the star's semi-axes and the major axis's
    direction (PA East of North), as in score_2004's elliptical frame."""
    a = 0.5 * p["diam"]
    ratio = p.get("ratio", 1.0)
    pa = jnp.radians(p.get("pa", 0.0))
    return a, a * ratio, jnp.sin(pa), jnp.cos(pa)


def sky_offset(p, x, y):
    """(ΔRA, ΔDec) in mas of the point (x, y) in the star's frame (units of
    its semi-axes, x along the major axis)."""
    a, b, s, c = star_axes(p)
    along, across = x * a, y * b
    return along * s + across * c, along * c - across * s


def star_frame(p, dra, ddec):
    """(x, y): the inverse of ``sky_offset``."""
    a, b, s, c = star_axes(p)
    return (dra * s + ddec * c) / a, (dra * c - ddec * s) / b


def spec(shape, n_spots=0, comp=False, spot_shape="disk"):
    return {"shape": shape, "n_spots": n_spots, "comp": comp, "spot_shape": spot_shape}


def build(p, rung):
    """The ``GainSystem`` for the parameters ``p`` (a dict keyed like the
    priors) of ``rung`` (a ``spec``)."""
    if rung["shape"] == "ud":
        star = vm.UniformDisk(p["diam"])
    elif rung["shape"] == "ld":
        star = vm.LimbDarkenedDisk(p["diam"], u=jnp.reshape(p["u"], (1,)))
    else:
        star = vm.EllipticalLimbDarkenedDisk(p["diam"], ratio=p["ratio"], pa=p["pa"], u=jnp.reshape(p["u"], (1,)))
    parts = {"star": star}
    for name in SPOT_NAMES[: rung["n_spots"]]:
        dra, ddec = sky_offset(p, p[f"{name}_x"], p[f"{name}_y"])
        if rung["spot_shape"] == "disk":
            parts[name] = vm.UniformDisk(p[f"{name}_diam"], flux=p[f"{name}_flux"], dra=dra, ddec=ddec)
        else:
            parts[name] = vm.GaussianDisk(p[f"{name}_diam"] / SQRT_8LN2, flux=p[f"{name}_flux"], dra=dra, ddec=ddec)
    if rung["comp"]:
        parts["comp"] = vm.UniformDisk(p["comp_diam"], flux=p["comp_flux"], dra=p["comp_dra"], ddec=p["comp_ddec"])
    return GainSystem(p["gain"], **parts)


class Hyper:
    """Data-derived scales for the priors and starts: the beam, the moments'
    starting diameter and the field λ/B_min (mas)."""

    def __init__(self, data):
        res = beam(data)
        self.bmin, self.bmaj = float(res.minor_mas), float(res.major_mas)
        self.moments = ci.moments_of(data)
        sigma = self.moments["sigma_major"]
        self.d0 = float(linear_starts.disk_diameter_from_sigma(sigma, 0.5)) if np.isfinite(sigma) and sigma > 0 else 3 * self.bmaj
        u, v, wavel, _, _ = ci.v2_table(data)
        self.half = 1.0 / float(np.min(np.hypot(u, v) / wavel)) / linear_starts.MAS


def make_priors(h, rung):
    pri = {"gain": dist.LogUniform(*GAIN_RANGE), "diam": dist.LogUniform(0.5 * h.bmin, 8.0 * h.bmaj)}
    if rung["shape"] in ("ld", "eld"):
        pri["u"] = dist.Uniform(0.0, 1.0)
    if rung["shape"] == "eld":
        pri["ratio"] = dist.Uniform(0.2, 1.0)
        pri["pa"] = dist.Uniform(*ci.PA_PRIOR)
    for name in SPOT_NAMES[: rung["n_spots"]]:
        pri[f"{name}_x"] = dist.Uniform(-1.2, 1.2)
        pri[f"{name}_y"] = dist.Uniform(-1.2, 1.2)
        pri[f"{name}_diam"] = dist.LogUniform(0.1 * h.bmin, 2.0 * h.bmin)
        pri[f"{name}_flux"] = dist.LogUniform(1e-3, 1.0)
    if rung["comp"]:
        pri["comp_dra"] = dist.Uniform(-h.half, h.half)
        pri["comp_ddec"] = dist.Uniform(-h.half, h.half)
        pri["comp_flux"] = dist.LogUniform(1e-4, 1.0)
        pri["comp_diam"] = dist.LogUniform(0.02 * h.bmin, 2.0 * h.bmin)
    return pri


# --- observables ------------------------------------------------------------


def read_t3amp(path):
    """T3AMP and its error with the baselines (m) and wavelength (m), unflagged."""
    from astropy.io import fits

    with fits.open(path) as hdus:
        d = hdus["OI_T3"].data
        ok = ~d["FLAG"].ravel() if "FLAG" in d.columns.names else np.ones(len(d), bool)
        wavel = float(np.ravel(hdus["OI_WAVELENGTH"].data["EFF_WAVE"])[0])
        cols = [np.asarray(d[k], float).ravel()[ok] for k in ("U1COORD", "V1COORD", "U2COORD", "V2COORD", "T3AMP", "T3AMPERR")]
    return (*cols, wavel)


def t3amp_chi2_red(model, t3amp):
    u1, v1, u2, v2, amp, err, wavel = t3amp
    w = jnp.asarray(wavel)
    bis = model.model(u1, v1, w) * model.model(u2, v2, w) * jnp.conj(model.model(u1 + u2, v1 + v2, w))
    return float(np.mean(((np.abs(np.asarray(bis)) - amp) / err) ** 2))


def blocks(model, data, t3amp=None):
    """χ²/N on the quoted errors per observable (V², closure phase, T3AMP
    which the likelihood does not read) and over the V² and closure phases
    together, with their counts."""
    r = np.asarray(whitened_residuals(model, data))
    n_vis, n_ind = int(data.vis.size), int(data.n_independent)
    if r.size != int(data.n_residuals):
        raise RuntimeError("unexpected residual layout")
    out = {"v2": float(np.mean(r[:n_vis] ** 2)), "cp": float(np.mean(r[n_vis:n_ind] ** 2)),
           "total": float(np.sum(r[:n_ind] ** 2) / n_ind), "chi2": float(np.sum(r[:n_ind] ** 2)),
           "n": n_ind, "n_v2": n_vis, "n_cp": n_ind - n_vis}
    out["t3amp"] = t3amp_chi2_red(model, t3amp) if t3amp is not None else None
    return out


# --- evidence ---------------------------------------------------------------


def flat_axes(priors):
    """(keys, low, high, log): each prior's range in its flat coordinate."""
    keys = list(priors)
    lo, hi, log = [], [], []
    for k in keys:
        p = priors[k]
        is_log = isinstance(p, dist.LogUniform)
        a, b = float(p.low), float(p.high)
        lo.append(np.log(a) if is_log else a)
        hi.append(np.log(b) if is_log else b)
        log.append(is_log)
    return keys, np.array(lo), np.array(hi), np.array(log)


def laplace(scene, priors, values, data, rung):
    """Gauss–Newton Laplace approximation at ``values``: the log evidence
    (up to the data's constant), σ of every parameter, and whether any sits
    within 1 % of a prior bound."""
    keys, lo, hi, log = flat_axes(priors)
    n_ind = int(data.n_independent)
    z0 = np.array([np.log(values[k]) if lg else values[k] for k, lg in zip(keys, log)], float)
    is_log = jnp.asarray(log)

    def unpack(z):
        x = jnp.where(is_log, jnp.exp(z), z)
        return {k: x[i] for i, k in enumerate(keys)}

    def resid(z):
        return whitened_residuals(scene(**unpack(z)), data)[:n_ind]

    jac = np.asarray(jax.jacfwd(resid)(jnp.asarray(z0)))
    fisher = jac.T @ jac
    vals, vecs = np.linalg.eigh(fisher)
    floor = max(vals.max(), 1e-300) * 1e-12
    vals = np.clip(vals, floor, None)
    cov_z = (vecs / vals) @ vecs.T
    width = hi - lo
    if "pa" in keys:
        width[keys.index("pa")] = 180.0  # the ellipse's two equivalent modes
    chi2 = float(np.sum(np.asarray(resid(jnp.asarray(z0))) ** 2))
    log_z = (-0.5 * chi2 - np.sum(np.log(width)) + 0.5 * len(keys) * np.log(2 * np.pi) - 0.5 * np.sum(np.log(vals))
             + math.lgamma(rung["n_spots"] + 1))
    sigma_z = np.sqrt(np.diag(cov_z))
    x0 = np.where(log, np.exp(z0), 1.0)
    sigma = sigma_z * np.where(log, x0, 1.0)
    on_bounds = [k for k, z, a, b in zip(keys, z0, lo, hi) if z < a + 0.01 * (b - a) or z > b - 0.01 * (b - a)]
    return {"log_z": float(log_z), "sigma": dict(zip(keys, map(float, sigma))), "cov_z": cov_z, "on_bounds": on_bounds,
            "min_eig_ratio": float(vals.min() / vals.max())}


# --- fitting ----------------------------------------------------------------


def fit_starts(scene, priors, data, starts, max_steps=None, t3amp=None, label=""):
    """Fit from each start (``fit`` with ``init``) and keep the lowest χ².
    Returns the best record (values, blocks, convergence) and every start's χ²/N."""
    kwargs = {} if max_steps is None else {"max_steps": max_steps}
    best, chi2s = None, []
    for i, init in enumerate(starts):
        t0 = time.time()
        try:
            result = fit(scene, priors, data, init=init, **kwargs)
        except Exception as err:  # a failed start is recorded, not fatal
            print(f"  {label} start {i}: failed ({type(err).__name__}: {err})", flush=True)
            chi2s.append(float("nan"))
            continue
        values = {k: float(np.asarray(result.values[k])) for k in priors}
        model = scene(**values)
        b = blocks(model, data, t3amp)
        chi2s.append(b["total"])
        converged = bool(np.ravel(result.info.get("converged", True))[0])
        print(f"  {label} start {i}: χ²/N={b['total']:.3f} ({time.time() - t0:.0f} s, converged {converged})", flush=True)
        if best is None or b["chi2"] < best["blocks"]["chi2"]:
            best = {"values": values, "blocks": b, "converged": converged}
    if best is None:
        raise RuntimeError(f"every start of {label} failed")
    finite = [c for c in chi2s if np.isfinite(c)]
    best["spread"] = float(max(finite) - min(finite))
    best["start_chi2"] = chi2s
    return best


def linear_peaks(data, template, name, axis, allowed, h, n_iter=3, top=3):
    """Peaks of a linear flux map of the part ``name`` of ``template`` (flux
    solved for at each (dra, ddec) of ``axis`` × ``axis``, where ``allowed`` is
    true): up to ``top`` separated by ``SPOT_SEPARATION`` beams, each
    (dra, ddec, flux, snr), the best SNR first."""
    samples = {f"{name}.dra": axis, f"{name}.ddec": axis, f"{name}.flux": np.array([1e-3])}
    grid = linear_flux_grid(template, data, samples, n_iter=n_iter)
    flux, snr = np.asarray(grid.flux), np.asarray(grid.snr)
    dra, ddec = np.meshgrid(axis, axis, indexing="ij")
    snr = np.nan_to_num(np.where(allowed(dra, ddec) & (flux > 0), snr, 0.0))
    peaks = []
    for _ in range(top):
        i, j = np.unravel_index(np.argmax(snr), snr.shape)
        if snr[i, j] <= 0:
            break
        peaks.append((float(axis[i]), float(axis[j]), float(flux[i, j]), float(snr[i, j])))
        snr = np.where(np.hypot(dra - axis[i], ddec - axis[j]) < SPOT_SEPARATION * h.bmin, 0.0, snr)
    return peaks


def search_axis(h):
    step = max(h.bmin / 3, 2 * h.half / (MAX_AXIS - 1))
    return np.arange(-h.half, h.half + 1e-9, step)


def with_rung_values(values, rung, extra=None):
    return build({**values, **(extra or {})}, rung)


def rank_by_chi2(data, rung, values, candidates):
    """``candidates`` (dicts of the added part's parameters) best first by the
    χ² of ``rung``'s model at ``values`` plus the candidate (a linear map's
    SNR is not monotonic in the χ² gain, and against an imperfect primary
    its highest peaks are residual structure)."""
    chi2 = [blocks(build({**values, **c}, rung), data)["chi2"] for c in candidates]
    return [candidates[i] for i in np.argsort(chi2)]


def spot_candidates(data, h, rung, values, name, top=KEEP):
    """Spot starts as (x, y, diam, flux): the linear map's peaks on the star
    (star-frame radius <= 1.2) of the model ``values`` (``rung``, with its
    parts in place) plus a spot of half a beam."""
    diam = 0.5 * h.bmin
    here = {**values, f"{name}_x": 0.0, f"{name}_y": 0.0, f"{name}_diam": diam, f"{name}_flux": 1e-3}
    with_spot = dict(rung, n_spots=rung["n_spots"] + 1)
    template = build(here, with_spot)

    def on_star(dra, ddec):
        x, y = star_frame(values, dra, ddec)
        return np.hypot(np.asarray(x), np.asarray(y)) <= 1.2

    peaks = linear_peaks(data, template, name, search_axis(h), on_star, h, top=PEAKS)
    out = []
    for dra, ddec, flux, _ in peaks:
        x, y = star_frame(values, dra, ddec)
        out.append({f"{name}_x": float(x), f"{name}_y": float(y), f"{name}_diam": diam,
                    f"{name}_flux": float(np.clip(flux, 2e-3, 0.9))})
    return rank_by_chi2(data, with_spot, values, out)[:top]


def comp_candidates(data, h, rung, values, top=2):
    """Companion starts as dicts: the linear map's peaks outside the star
    (plus half a beam; the primary is masked) of ``values``' model."""
    here = {**values, "comp_dra": 0.0, "comp_ddec": 0.0, "comp_flux": 1e-3, "comp_diam": COMP_DIAM * h.bmin}
    template = build(here, dict(rung, comp=True))

    def off_star(dra, ddec):
        a, b, s, c = (np.asarray(t) for t in star_axes(values))
        a, b = a + 0.5 * h.bmaj, b + 0.5 * h.bmaj
        along, across = dra * s + ddec * c, dra * c - ddec * s
        return (along / a) ** 2 + (across / b) ** 2 >= 1.0

    peaks = linear_peaks(data, template, "comp", search_axis(h), off_star, h, top=PEAKS)
    cands = [{"comp_dra": dra, "comp_ddec": ddec, "comp_flux": float(np.clip(flux, 2e-4, 0.9)), "comp_diam": COMP_DIAM * h.bmin} for dra, ddec, flux, _ in peaks]
    ranked = rank_by_chi2(data, dict(rung, comp=True), values, cands)[:top]
    return ranked, peaks


def jitter_for(priors):
    table = {"gain": ("log", 0.05), "diam": ("log", 0.2), "u": ("add", 0.3), "ratio": ("add", 0.15), "pa": ("add", 40.0)}
    for name in SPOT_NAMES:
        table |= {f"{name}_x": ("add", 0.3), f"{name}_y": ("add", 0.3), f"{name}_diam": ("log", 0.3), f"{name}_flux": ("log", 0.5)}
    table |= {"comp_dra": ("add", 0.5), "comp_ddec": ("add", 0.5), "comp_flux": ("log", 0.4), "comp_diam": ("log", 0.5)}
    return {k: v for k, v in table.items() if k in priors}


def make_starts(centres, priors, n, rng):
    """About ``n`` starts: every centre once, then jittered copies spread over
    them, all inside the priors."""
    centres = [{k: v for k, v in c.items() if k in priors} for c in centres]
    out = [ci.jittered_starts(c, priors, 1, rng, {})[0] for c in centres]
    per = max(1, (n - len(out)) // len(centres) + 1)
    for c in centres:
        out += ci.jittered_starts(c, priors, per + 1, rng, jitter_for(priors))[1:]
    return out[: max(n, len(centres))]


def run_ladder(data, h, n_starts, max_steps, rng, t3amp=None, spot_shape="disk", smoke=False, second_spot=True):
    """Fit every rung. Returns {name: record}; each record holds the rung's
    spec, priors, best values, blocks, k, BIC, Laplace evidence and σ."""
    n = int(data.n_independent)
    records = {}

    def finish(name, label, rung, priors, best):
        scene = lambda **p: build(p, rung)  # noqa: E731
        lap = laplace(scene, priors, best["values"], data, rung)
        k = len(priors)
        b = best["blocks"]
        records[name] = {"label": label, "rung": rung, "priors": priors, "k": k, "bic": b["chi2"] + k * math.log(n),
                         "log_z": lap["log_z"], "sigma": lap["sigma"], "on_bounds": lap["on_bounds"], "cov_z": lap["cov_z"],
                         "min_eig_ratio": lap["min_eig_ratio"], **best}
        print(f"{label}: χ²/N {b['total']:.3f} (V² {b['v2']:.3f}, closure {b['cp']:.3f}, T3AMP "
              f"{'n/a' if b['t3amp'] is None else format(b['t3amp'], '.3f')}) k={k} BIC={records[name]['bic']:.1f} "
              f"log Z={lap['log_z']:.1f} on bounds {lap['on_bounds']}", flush=True)
        return records[name]

    def rung_fit(name, label, rung, centres):
        priors = make_priors(h, rung)
        scene = lambda **p: build(p, rung)  # noqa: E731
        starts = make_starts(centres, priors, n_starts, rng)
        print(f"\n{label}: {len(starts)} starts", flush=True)
        return finish(name, label, rung, priors, fit_starts(scene, priors, data, starts, max_steps, t3amp, name))

    # 1. uniform disk: scan the diameter from 0.5x to 2x the moments' value
    rung = spec("ud")
    pri = make_priors(h, rung)
    diams = [ci.inside(pri["diam"], h.d0 / 1.2 * f) for f in np.logspace(np.log10(0.5), np.log10(2.0), 9)]
    chi = [blocks(build({"gain": 1.0, "diam": d}, rung), data)["total"] for d in diams]
    rung_fit("ud", "uniform disk", rung, [{"gain": 1.0, "diam": diams[int(np.nanargmin(chi))]}])
    # 2. limb-darkened disk
    prev = records["ud"]["values"]
    rung_fit("ld", "limb-darkened disk", spec("ld"), [{**prev, "u": 0.5}])
    # 3. elliptical, started from the moments (and the PA a right angle away)
    prev = records["ld"]["values"]
    m = h.moments
    ratio0 = float(np.clip(m["ratio"] if np.isfinite(m["ratio"]) else 0.8, 0.3, 0.95))
    pa0 = float(m["pa"]) if np.isfinite(m["pa"]) else 90.0
    rung_fit("eld", "elliptical LD disk", spec("eld"),
             [{**prev, "ratio": ratio0, "pa": pa0}, {**prev, "ratio": ratio0, "pa": (pa0 + 90.0) % 180.0}])
    eld = spec("eld", spot_shape=spot_shape)
    base = records["eld"]["values"]
    # the diagnostic rung: companion without a spot
    comps, peaks = comp_candidates(data, h, spec("eld"), base)
    print(f"\ncompanion search (±{h.half:.1f} mas, primary masked): {len(peaks)} linear-map peaks ranked by χ²; starts {comps}", flush=True)
    comp_records = None
    if comps:
        comp_records = rung_fit("eld+comp", "elliptical LD + companion", spec("eld", comp=True, spot_shape=spot_shape), [{**base, **c} for c in comps])
    # 4. one spot, searched with the companion in the model when it was found
    ref = comp_records if comp_records is not None else records["eld"]
    ref_rung = ref["rung"]
    spots = spot_candidates(data, h, ref_rung, ref["values"], SPOT_NAMES[0])
    print(f"spot search: starts {spots}", flush=True)
    spot_rung = spec("eld", 1, spot_shape=spot_shape)
    if spots:
        rung_fit("eld+spot", "elliptical LD + spot", spot_rung, [{**base, **s} for s in spots])
    # 5. spot and companion
    full = spec("eld", 1, True, spot_shape)
    centres = []
    for s in spots[:2]:
        for c in comps[:2]:
            centres.append({**ref["values"], **s, **c})
            if "eld+spot" in records:
                centres.append({**records["eld+spot"]["values"], **c})
    if comp_records is not None:
        centres += [{**comp_records["values"], **s} for s in spots]
    if centres:
        rung_fit("eld+spot+comp", "elliptical LD + spot + companion", full, centres)
    # 6. a second spot, only if the residuals ask for it
    top = min(records.values(), key=lambda r: r["bic"])
    if second_spot and not smoke and top["blocks"]["total"] > SECOND_SPOT_IF and top["rung"]["n_spots"] == 1:
        two = spec("eld", 2, top["rung"]["comp"], spot_shape)
        more = spot_candidates(data, h, top["rung"], top["values"], SPOT_NAMES[1])
        if more:
            rung_fit("eld+2spot+comp", "elliptical LD + 2 spots" + (" + companion" if two["comp"] else ""), two,
                     [{**top["values"], **s} for s in more])
    return records


def choose(records):
    """The lowest-BIC rung, the lowest-evidence-free comparison, and each
    rung's Δ against the winner."""
    best = min(records, key=lambda k: records[k]["bic"])
    z_best = max(records, key=lambda k: records[k]["log_z"])
    return best, z_best


# --- uncertainties ----------------------------------------------------------


def run_nuts(record, data, warmup, samples, chains, seed=0):
    from numpyro.diagnostics import summary
    from numpyro.infer import MCMC, NUTS, init_to_value

    rung, priors = record["rung"], record["priors"]
    scene = lambda **p: build(p, rung)  # noqa: E731
    model = numpyro_model(scene, priors, data)
    mcmc = MCMC(NUTS(model, init_strategy=init_to_value(values=record["values"])), num_warmup=warmup, num_samples=samples,
                num_chains=chains, chain_method="sequential", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed))
    grouped = mcmc.get_samples(group_by_chain=True)
    stats = summary({k: v for k, v in grouped.items() if k in priors}, group_by_chain=True)
    flat = {k: np.asarray(v).reshape(-1) for k, v in mcmc.get_samples().items() if k in priors}
    out = {}
    for k, v in flat.items():
        lo, med, hi = np.percentile(v, [16, 50, 84])
        out[k] = {"median": float(med), "lo": float(lo), "hi": float(hi), "std": float(np.std(v)),
                  "r_hat": float(stats[k]["r_hat"]) if chains > 1 else None, "n_eff": float(stats[k]["n_eff"])}
    return out, flat


# --- truth, rendering, scoring ----------------------------------------------


def truth_parameters():
    """The published truth as this model's parameters (score_2004's rebuilt
    model, its unpublished nuisances at their fitted values). For the output
    table only."""
    star, companion = score_2004.truth_parts(score_2004.FITTED["u_ld"], spot_contrast=score_2004.FITTED["spot_contrast"])
    ratio = score_2004.FITTED["ratio"]
    excess = float(star.sum() - 1.0)
    sx, sy = score_2004.offset(150.0, 2.5)
    bx, by = score_2004.offset(90.0, 10.0)
    return {"diam": 5.5, "ratio": 0.7, "pa": 120.0, "u": score_2004.FITTED["u_ld"], "spot_dra": sx, "spot_ddec": sy,
            "spot_diam": 1.0, "spot_flux": excess, "comp_dra": bx, "comp_ddec": by, "comp_flux": ratio, "comp_diam": 0.5,
            "gain": (1.0 + excess + ratio) / (1.0 + ratio)}


def recovery_table(best, sigma, truth):
    """Fitted against true parameters, in the model's own parameters (the
    spot's offset, not its star-frame coordinates). Each row: name, fit, σ,
    truth, (fit − truth)/σ (the PA modulo 180°, in the nearest branch)."""
    p = dict(best)
    rows = []
    for name in SPOT_NAMES[:2]:
        if f"{name}_x" in p:
            dra, ddec = (float(t) for t in sky_offset(p, p[f"{name}_x"], p[f"{name}_y"]))
            p[f"{name}_dra"], p[f"{name}_ddec"] = dra, ddec
    for k in ("diam", "ratio", "pa", "u", "gain", "spot_dra", "spot_ddec", "spot_diam", "spot_flux", "comp_dra", "comp_ddec", "comp_flux", "comp_diam"):
        if k not in p or k not in truth:
            continue
        s = sigma.get(k)
        diff = p[k] - truth[k]
        if k == "pa":
            diff = (diff + 90.0) % 180.0 - 90.0
        rows.append({"name": k, "fit": float(p[k]), "sigma": s, "truth": float(truth[k]), "diff": float(diff),
                     "pull": float(diff / s) if s else None})
    return rows


def spot_positions_sigma(record, sigma):
    """σ of the spot's and companion's sky offsets from the Laplace
    covariance by linear propagation (the star-frame to sky map's gradient)."""
    out = dict(sigma)
    keys = list(record["priors"])
    _, _, _, log = flat_axes(record["priors"])
    v = record["values"]
    scale = np.where(log, [v[k] for k in keys], 1.0)
    cov = record["cov_z"] * np.outer(scale, scale)  # in the parameters themselves
    for name in SPOT_NAMES[: record["rung"]["n_spots"]]:
        need = ["diam", "ratio", "pa", f"{name}_x", f"{name}_y"]
        if not all(k in keys for k in need):
            continue

        def f(vec):
            q = dict(zip(need, vec))
            return jnp.stack(sky_offset(q, q[f"{name}_x"], q[f"{name}_y"]))

        idx = [keys.index(k) for k in need]
        jac = np.asarray(jax.jacfwd(f)(jnp.array([v[k] for k in need])))
        c = jac @ cov[np.ix_(idx, idx)] @ jac.T
        out[f"{name}_dra"], out[f"{name}_ddec"] = float(np.sqrt(c[0, 0])), float(np.sqrt(c[1, 1]))
    return out


def score_entry(image, fov, data_path):
    """σ/peak of ``image`` against the truth, as score_2004.main: with the
    published nuisances (u = U_LD), and with the nuisances fitted to the data
    (``FITTED``). Returns a dict."""
    u, v, v2, err = score_2004.read_v2(data_path)
    star, companion = score_2004.truth_parts(score_2004.U_LD)
    ratio, _ = score_2004.fit_ratio(star, companion, u, v, v2, err)
    scores, ref = score_2004.score(image, fov, ratio, score_2004.U_LD)
    out = {"published_truth": {k: {"sigma_over_peak": s[0], "sigma": s[1], "shift": s[3]} for k, s in scores.items()},
           "ref_peak": float(ref.max()), "ratio_fitted": ratio}
    fitted = score_2004.truth_image(score_2004.FITTED["ratio"], score_2004.FITTED["u_ld"], score_2004.FITTED["spot_contrast"])
    box = np.zeros_like(fitted, bool)
    rows, cols = np.nonzero(fitted > 0)
    box[rows.min() - 10 : rows.max() + 11, cols.min() - 10 : cols.max() + 11] = True
    moved, shift = score_2004.aligned(score_2004.resample(image, fov, score_2004.NPIX + 2 * score_2004.MAX_SHIFT), fitted)
    sp, s, _ = score_2004.sigma_over_peak(moved, fitted, box)
    out["fitted_truth"] = {"sigma_over_peak": sp, "sigma": s, "shift": list(shift)}
    return out


# --- report -----------------------------------------------------------------


def ladder_rows(records):
    order = list(records)
    best = min(records, key=lambda k: records[k]["bic"])
    z0 = max(r["log_z"] for r in records.values())
    rows = []
    for name in order:
        r = records[name]
        b = r["blocks"]
        rows.append({"rung": name, "label": r["label"], "k": r["k"], "chi2_red": b["total"], "v2": b["v2"], "cp": b["cp"],
                     "t3amp": b["t3amp"], "chi2": b["chi2"], "bic": r["bic"], "dbic": r["bic"] - records[best]["bic"],
                     "log_z": r["log_z"], "dlog_z": r["log_z"] - z0, "converged": r["converged"],
                     "spread": r["spread"], "on_bounds": r["on_bounds"], "winner": name == best})
    return rows


def format_report(label, rows, best, z_best, truth_rows, score, nuts, h, records, data_name, elapsed):
    n = records[best]["blocks"]["n"]
    lines = [f"{label}: informed parametric ladder on 2004 {data_name} (N = {n} V² and closure phases), {elapsed:.0f} s",
             f"beam {h.bmin:.2f} x {h.bmaj:.2f} mas, moments' disk start {h.d0:.2f} mas, field ±{h.half:.1f} mas", "",
             "χ²/N on the quoted errors, raw, per observable (T3AMP is not fitted: it is a check under the same gain):",
             f"{'rung':16s} {'k':>2s} {'χ²/N':>8s} {'V²':>8s} {'closure':>8s} {'T3AMP':>8s} {'BIC':>9s} {'ΔBIC':>8s} {'logZ':>9s} {'ΔlogZ':>8s}"]
    for r in rows:
        t3 = "n/a" if r["t3amp"] is None else f"{r['t3amp']:.3f}"
        mark = "  <- BIC winner" if r["winner"] else ("  <- evidence winner" if r["rung"] == z_best else "")
        flag = f"  (on bounds: {', '.join(r['on_bounds'])})" if r["on_bounds"] else ""
        lines.append(f"{r['rung']:16s} {r['k']:2d} {r['chi2_red']:8.3f} {r['v2']:8.3f} {r['cp']:8.3f} {t3:>8s} {r['bic']:9.1f} "
                     f"{r['dbic']:8.1f} {r['log_z']:9.1f} {r['dlog_z']:8.1f}{mark}{flag}")
    lines += ["", f"winner by BIC: {best}; by Laplace evidence: {z_best}"]
    if score:
        pub = score["published_truth"]
        lines += ["", "data2 σ/peak (lower is better; published: BSMEM 0.116, WISARD 0.163, MIRA 0.532, VLBMEM 0.798; our best image 0.250):"]
        for name, s in pub.items():
            lines.append(f"  against the published truth, {name:9s}: σ/peak = {s['sigma_over_peak']:.3f} (shift {s['shift']})")
        ft = score["fitted_truth"]
        lines.append(f"  against the truth with fitted nuisances: σ/peak = {ft['sigma_over_peak']:.3f} (shift {ft['shift']})")
    if truth_rows:
        lines += ["", "parameter recovery (the truth is the rebuilt published model, for this table only):",
                  f"{'parameter':10s} {'fit':>10s} {'σ':>10s} {'truth':>10s} {'fit − truth':>12s} {'pull':>7s}"]
        for r in truth_rows:
            s = "-" if r["sigma"] is None else f"{r['sigma']:.3g}"
            pull = "-" if r["pull"] is None else f"{r['pull']:.1f}"
            lines.append(f"{r['name']:10s} {r['fit']:10.4g} {s:>10s} {r['truth']:10.4g} {r['diff']:12.3g} {pull:>7s}")
    if nuts:
        lines += ["", "NUTS (median, 16–84 %, r_hat, n_eff):"]
        for k, v in nuts.items():
            rh = "-" if v["r_hat"] is None else f"{v['r_hat']:.3f}"
            lines.append(f"  {k:10s} {v['median']:10.4g} ({v['lo']:.4g}, {v['hi']:.4g}) r_hat {rh} n_eff {v['n_eff']:.0f}")
    return "\n".join(lines)


def run(dataset, data_dir, out_dir, smoke=False, n_starts=None, nuts=(500, 1000, 4), spot_shape="disk", seed=0):
    t0 = time.time()
    path = data_dir / f"2004-{dataset}.fits"
    data = OIData(str(path))
    t3amp = read_t3amp(path)
    h = Hyper(data)
    n_starts = n_starts or (N_STARTS_SMOKE if smoke else N_STARTS)
    max_steps = 3 if smoke else None
    rng = np.random.default_rng(seed)
    global MAX_AXIS
    MAX_AXIS = 41 if smoke else 201
    label = f"2004_{dataset}_parametric"
    print(f"{label}: beam {h.bmin:.2f} x {h.bmaj:.2f} mas, moments {h.moments}, field ±{h.half:.1f} mas", flush=True)
    records = run_ladder(data, h, n_starts, max_steps, rng, t3amp, spot_shape, smoke)
    rows = ladder_rows(records)
    best, z_best = choose(records)
    win = records[best]
    sigma = spot_positions_sigma(win, win["sigma"])
    samples, nuts_summary = {}, None
    warmup, n_samples, chains = (10, 10, 1) if smoke else nuts
    try:
        nuts_summary, samples = run_nuts(win, data, warmup, n_samples, chains, seed)
        sigma = {**sigma, **{k: v["std"] for k, v in nuts_summary.items()}}
        sigma = spot_positions_sigma_from_samples(win, samples, sigma)
    except Exception as err:  # the Laplace σ stand in
        print(f"NUTS failed ({type(err).__name__}: {err}); Laplace σ used", flush=True)
    model = build(win["values"], win["rung"])
    image = np.asarray(model.render(SCORE_NPIX, SCORE_FOV))
    score = truth_rows = None
    if dataset == "data2":
        score = score_entry(image, SCORE_FOV, path)
        truth_rows = recovery_table(win["values"], sigma, truth_parameters())
    text = format_report(label, rows, best, z_best, truth_rows, score, nuts_summary, h, records, dataset, time.time() - t0)
    print("\n" + text, flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{label}.txt").write_text(text + "\n")
    record = {"dataset": dataset, "smoke": smoke, "ladder": rows, "winner": best, "evidence_winner": z_best,
              "values": win["values"], "sigma": sigma, "score": score, "recovery": truth_rows, "nuts": nuts_summary,
              "moments": {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in h.moments.items()},
              "params": {name: r["values"] for name, r in records.items()}}
    (out_dir / f"{label}.json").write_text(json.dumps(record, indent=1, default=float))
    np.savez(out_dir / f"{label}.npz", ref_image=image, ref_fov=SCORE_FOV, image=image, fov=SCORE_FOV, cov_z=win["cov_z"],
             **{f"sample_{k}": v for k, v in samples.items()}, **{f"param_{k}": v for k, v in win["values"].items()})
    return record


def spot_positions_sigma_from_samples(record, samples, sigma):
    """σ of the spot offsets from the NUTS samples (propagated sample by sample)."""
    out = dict(sigma)
    for name in SPOT_NAMES[: record["rung"]["n_spots"]]:
        need = ["diam", "ratio", "pa", f"{name}_x", f"{name}_y"]
        if all(k in samples for k in need):
            q = {k: samples[k] for k in need}
            dra, ddec = (np.asarray(t) for t in sky_offset(q, q[f"{name}_x"], q[f"{name}_y"]))
            out[f"{name}_dra"], out[f"{name}_ddec"] = float(np.std(dra)), float(np.std(ddec))
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("data1", "data2"), required=True)
    parser.add_argument("--data-dir", default="~/data/imaging_contests/2004", help="holds 2004-data1.fits and 2004-data2.fits")
    parser.add_argument("--out", default="contest_parametric")
    parser.add_argument("--spot-shape", choices=("disk", "gauss"), default="disk")
    parser.add_argument("--n-starts", type=int, default=None, help=f"starts per rung (default {N_STARTS}, {N_STARTS_SMOKE} with --smoke)")
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="two starts, three steps, short NUTS, no second spot")
    return parser.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    run(a.dataset, pathlib.Path(a.data_dir).expanduser(), pathlib.Path(a.out).expanduser(), a.smoke, a.n_starts,
        (a.warmup, a.samples, a.chains), a.spot_shape, a.seed)


if __name__ == "__main__":
    main()
