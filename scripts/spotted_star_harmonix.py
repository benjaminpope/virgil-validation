"""A harmonix (spherical-harmonic surface) model of the 2004 contest's data2.

    python scripts/spotted_star_harmonix.py --data ~/data/imaging_contests/2004/2004-data2.fits \\
        --out DIR --config spots1|spots2|map [--ydeg L] [--smoke]

The 2004 Interferometric Imaging Beauty Contest's data2 (NPOI, 550 nm, about
195 V² and 130 closure phases) came with one piece of pre-submission
information: "a limb darkened star with one or more spots", and possibly a
companion. This script fits exactly that, as a parametric model, using only
that information. The published truth (scripts/score_2004.py) is for scoring
only: it never sets a prior, a start or a setting here.

**The star.** A harmonix surface (Dholakia & Pope 2025) seen equator-on
(inc = 90°, obl = 0). With one epoch rotation is unobservable, so only the
projected map matters, and the visible hemisphere is longitudes -90° to 90°.
Linear limb darkening u ~ U(0, 1); radius log-uniform. Maps are given as
harmonix's ``data`` (the l >= 1 coefficients, with Y00 = 1).

**Ellipticity.** A harmonix star projects as a circle. ``EllipticalHarmonix``
evaluates its visibility at affinely transformed spatial frequencies: the
outline is compressed by ``ratio`` along the minor axis, the major axis at
``pa`` (degrees East of North), exactly as virgil's
``EllipticalLimbDarkenedDisk`` (``undo_elliptical_transf_spat_freq``, the
Fourier similarity theorem). virgil's helper also rotates the frame so the
major axis points North; we rotate back, so the map keeps its sky
orientation (map x East, y North) and ``pa``/``ratio`` describe the outline
only. Then ratio = 1 is ``HarmonixModel`` exactly, for any ``pa``, and a
spot's latitude and longitude are not degenerate with ``pa``.
ratio ~ U(0.2, 1), pa ~ U(0°, 180°).

**The companion.** Found as in scripts/contest_images.py: a primary
elliptical limb-darkened disk (``fit_primary``), then a linear flux map
(the ``companion_search`` approach: virgil.grid_fit.linear_flux_grid, kept
at SNR >= 5), over ±λ/B_min and off the primary's disk (``find_companion``
says why ``companion_search`` itself is not used). It is then fitted jointly with the harmonix star, ``System(star, comp)``:
position uniform within a beam of the search's peak, flux ratio log-uniform
(``FLUX_FLOOR`` to 1).

**Maps.** Three configurations:

- ``spots1``, ``spots2``: one or two jaxoplanet ``ylm_spot`` spots (degree
  ``--ydeg``, default 12). jaxoplanet's ``contrast`` is 1 - (spot intensity /
  photosphere): positive is dark, negative bright (its profile is -1 inside
  the spot, so the map is 1 - contrast there). We fit the spot's intensity
  relative to the photosphere, ``level = 1 - contrast``, log-uniform on
  [1/20, 20] (a scale, Jeffreys): below 1 a dark spot, above 1 a bright one,
  both allowed. Size (angular radius on the sphere, radians) log-uniform on
  [0.05, 1.2]; spots smaller than about pi / ydeg are smoothed by the
  expansion, so there size and level are degenerate. Latitude
  ``virgil.priors.IsotropicLatitude`` (-pi/2, pi/2), longitude uniform over
  the visible hemisphere (-pi/2, pi/2).
- ``map``: every l >= 1 coefficient up to ``--ydeg`` (default 6), with a
  Gaussian prior of variance C_l = sigma² (1 + l)^-2, falling with degree.
  sigma is chosen by an outer scan over a log-uniform grid (``SIGMAS``),
  keeping the one of largest Laplace evidence,
  log Z = -χ²/2 - Σ c²/(2 C_l) - ½ log det(I + C^½ JᵀJ C^½), with J the
  Jacobian of the whitened residuals with respect to the coefficients at
  each σ's MAP (the other parameters held at their MAP, as
  virgil.imaging.log_evidence does). A scan rather than MacKay's fixed
  point: the fixed point assumes a linear model, and the map's evidence
  can have several local maxima once a spot is resolved; on a log-uniform
  hyperprior (Jeffreys, sigma is a scale) the grid's maximum is the MAP
  sigma. A maximum on the edge of the grid is reported.

All priors are bounded, log-uniform for scales and fluxes and uniform for
positions and angles (Ben's Jeffreys-prior rule).

**Starts.** The radius starts from a uniform-disk χ² scan, as in virgil's
harmonix tutorial; each configuration is started from several position
angles (0°, 60°, 120°, ratio 0.8) and spot positions (latitudes ±0.5,
longitudes ±0.6 rad, bright and dark), and the best χ² is kept.

**Outputs** (``--out``), per configuration, ``2004_data2_harmonix_<config>``:

- ``.npz``: ``ref_image``/``ref_fov`` (the best fit, star plus companion,
  rendered on SCORE_NPIX pixels of 0.08 mas, East left and North up, as
  scripts/score_2004.py reads it), ``star_image``, χ² against the *quoted*
  errors (``chi2``, ``n_data``, ``chi2_red``), MacKay's error scale
  (``error_scale``: s² = χ² / (N - γ), γ the effective number of
  parameters, as virgil.imaging.error_scale; 1 for flat priors on
  measured parameters), the parameter values (``param_<name>``), every
  start's χ² and convergence (``start_chi2``, ``start_steps``,
  ``start_grad_norm``), and for ``map`` the σ scan (``sigmas``,
  ``log_evidence``);
- ``.txt``: the summary printed at the end.

Score with ``python scripts/score_2004.py OUT/2004_data2_harmonix_<config>.npz``.
Heavy: run on OzSTAR (ozstar_scripts job contest_imaging, ``--harmonix``);
``--smoke`` is a short check that the script runs (one start, three steps).
"""

import argparse
import itertools
import pathlib
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro.distributions as dist  # noqa: E402

try:
    from harmonix.harmonix import Harmonix  # noqa: E402
    from jaxoplanet.starry import Surface, Ylm  # noqa: E402
    from jaxoplanet.starry.ylm import ylm_spot  # noqa: E402
except ImportError as err:  # pragma: no cover - reported, not tested
    Harmonix = None
    _HARMONIX_ERROR = err
else:
    _HARMONIX_ERROR = None

import virgil.models as vm  # noqa: E402
from virgil._geometry import (  # noqa: E402
    apply_elliptical_transf_coord,
    apply_elliptical_transf_spat_freq,
    undo_elliptical_transf_coord,
    undo_elliptical_transf_spat_freq,
)
from virgil.fitting import fit  # noqa: E402
from virgil.likelihood import whitened_residuals  # noqa: E402
from virgil.priors import IsotropicLatitude  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from contest_images import COMPANION_SNR, FLUX_FLOOR, companion_map, fit_primary  # noqa: E402

CONFIGS = ("spots1", "spots2", "map")
DEFAULT_YDEG = {"spots1": 12, "spots2": 12, "map": 6}
START_PAS = (15.0, 75.0, 135.0)  # degrees, away from the prior's bounds (a start on a bound is stuck)
START_RATIO = 0.8
START_U = 0.5
SPOT_POSITIONS = tuple(itertools.product((0.5, -0.5), (0.6, -0.6)))  # (lat, lon) radians
SPOT_LEVELS = (3.0, 0.3)  # bright and dark starts
SPOT_SIZE0 = 0.3  # radians
LEVEL_RANGE = (1.0 / 20.0, 20.0)
SIZE_RANGE = (0.05, 1.2)
SIGMAS = np.geomspace(0.01, 10.0, 9)
SCORE_PIXEL = 0.08  # mas, scripts/score_2004.PIXEL
SCORE_NPIX = 321  # 25.7 mas across: the companion at 10 mas is inside


def require_harmonix():
    if Harmonix is None:
        raise ImportError(
            f"harmonix is not importable ({_HARMONIX_ERROR}); install it with pip install 'harmonix>=0.1.0'"
        )


class EllipticalHarmonix(vm.HarmonixModel):
    """A harmonix star with an elliptical outline: the visibility at the
    spatial frequencies of virgil's ``EllipticalLimbDarkenedDisk`` (outline
    compressed by ``ratio`` along the minor axis, major axis at ``pa``
    degrees East of North), with the map kept in sky orientation (see the
    module docstring). Unit flux; weight 1 in a System."""

    ratio: jax.Array
    pa: jax.Array

    def __init__(self, source, ratio=1.0, pa=0.0):
        super().__init__(source, observation_time=0.0)
        self.ratio = jnp.asarray(ratio, dtype=float)
        self.pa = jnp.asarray(pa, dtype=float)

    def _frequencies(self, u, v):
        ut, vt = undo_elliptical_transf_spat_freq(u, v, self.pa, self.ratio)
        return apply_elliptical_transf_spat_freq(ut, vt, self.pa, 1.0)  # rotate back: map stays on the sky

    def model(self, u, v, wavel):
        ut, vt = self._frequencies(u, v)
        return super().model(ut, vt, wavel)

    def _image(self, xx, yy, pixel_scale_mas):
        xt, yt = undo_elliptical_transf_coord(xx, yy, self.pa, jnp.maximum(self.ratio, 1e-9))
        xt, yt = apply_elliptical_transf_coord(xt, yt, self.pa, 1.0)
        return super()._image(xt, yt, pixel_scale_mas)


def template_star(ydeg, u1=START_U, radius=1.0):
    """An unspotted harmonix star of degree ``ydeg``, equator-on, linear limb
    darkening; parameters are swapped in with ``star_from``."""
    require_harmonix()
    y = jnp.zeros((ydeg + 1) ** 2).at[0].set(1.0)
    surface = Surface(y=Ylm.from_dense(y, normalize=False), inc=jnp.pi / 2, obl=0.0, u=(float(u1),))
    return Harmonix(surface, float(radius))


def star_from(template, radius, u1, ratio, pa, data):
    """``template`` with these parameters: radius (mas), linear limb darkening
    ``u1`` (also on the surface, which renders), map ``data`` (l >= 1)."""
    u1 = jnp.asarray(u1, dtype=float)
    source = eqx.tree_at(
        lambda s: (s.radius, s.u, s.data, s.surface.u),
        template,
        (jnp.asarray(radius, dtype=float), jnp.reshape(u1, (1,)), jnp.asarray(data, dtype=float), (u1,)),
    )
    return EllipticalHarmonix(source, ratio, pa)


def spot_map(spot_fn, spots):
    """The l >= 1 coefficients (Y00 = 1) of a map with ``spots``, each
    (level, size, lat, lon): level is the spot's intensity relative to the
    photosphere, so jaxoplanet's contrast is 1 - level (positive dark,
    negative bright). Spots add."""
    y = None
    for level, size, lat, lon in spots:
        yi = spot_fn(1.0 - level, size, lat, lon).todense()
        y = yi if y is None else y + yi.at[0].add(-1.0)
    return y[1:] / y[0]


def degrees_of(ydeg):
    """The degree l of each l >= 1 coefficient, in harmonix's order."""
    return np.concatenate([np.full(2 * ell + 1, ell) for ell in range(1, ydeg + 1)])


def map_prior_scales(ydeg, sigma):
    """Prior standard deviations of the l >= 1 coefficients: C_l = σ² (1 + l)^-2."""
    return sigma / (1.0 + degrees_of(ydeg))


def chi2_of(model, data):
    """χ² on the quoted errors and the number of independent data (V² plus
    closure phases; the whitened residuals hold two terms per closure
    phase)."""
    r = whitened_residuals(model, data)
    return float(jnp.sum(r**2)), int(data.n_independent)


def uniform_disk_radius(data, low, high, n=300):
    """Radius (mas) of the best uniform disk on a log grid, as in virgil's
    harmonix tutorial (a starting value only)."""
    diams = np.geomspace(low, high, n)
    chi2 = jax.jit(lambda d: jnp.sum(whitened_residuals(vm.UniformDisk(d), data) ** 2))
    values = np.array([float(chi2(d)) for d in diams])
    return 0.5 * float(diams[np.argmin(values)])


class Scene:
    """The star (with ``n_spots`` spots, or a free map) and, if found, a
    point companion; ``__call__(**params)`` builds the virgil model."""

    def __init__(self, config, ydeg, companion):
        self.config, self.ydeg, self.companion = config, ydeg, companion
        self.template = template_star(ydeg)
        self.n_spots = {"spots1": 1, "spots2": 2}.get(config, 0)
        self.spot_fn = ylm_spot(ydeg) if self.n_spots else None

    def __call__(self, **p):
        if self.n_spots:
            spots = [(p[f"spot{i}_level"], p[f"spot{i}_size"], p[f"spot{i}_lat"], p[f"spot{i}_lon"])
                     for i in range(self.n_spots)]
            data = spot_map(self.spot_fn, spots)
        else:
            data = p["data"]
        star = star_from(self.template, p["radius"], p["u"], p["ratio"], p["pa"], data)
        if not self.companion:
            return star
        return vm.System(star=star, comp=vm.PointSource(p["comp_flux"], dra=p["comp_dra"], ddec=p["comp_ddec"]))


def base_priors(radius_range, companion, beam_mas):
    priors = {"radius": dist.LogUniform(*radius_range), "u": dist.Uniform(0.0, 1.0),
              "ratio": dist.Uniform(0.2, 1.0), "pa": dist.Uniform(0.0, 180.0)}
    if companion:
        dra, ddec = companion["dra"], companion["ddec"]
        priors |= {"comp_dra": dist.Uniform(dra - beam_mas, dra + beam_mas),
                   "comp_ddec": dist.Uniform(ddec - beam_mas, ddec + beam_mas),
                   "comp_flux": dist.LogUniform(FLUX_FLOOR, 1.0)}
    return priors


def spot_priors(n_spots):
    out = {}
    for i in range(n_spots):
        out |= {f"spot{i}_level": dist.LogUniform(*LEVEL_RANGE), f"spot{i}_size": dist.LogUniform(*SIZE_RANGE),
                f"spot{i}_lat": IsotropicLatitude(-jnp.pi / 2, jnp.pi / 2),
                f"spot{i}_lon": dist.Uniform(-jnp.pi / 2, jnp.pi / 2)}
    return out


def base_starts(radius0, companion):
    starts = []
    for pa in START_PAS:
        s = {"radius": radius0, "u": START_U, "ratio": START_RATIO, "pa": pa}
        if companion:
            s |= {"comp_dra": companion["dra"], "comp_ddec": companion["ddec"],
                  "comp_flux": float(np.clip(companion["flux"], 2 * FLUX_FLOOR, 0.5))}
        starts.append(s)
    return starts


def spot_starts(n_spots):
    """Spot starting values: every position (pair) with bright, or bright and
    dark, spots."""
    out = []
    if n_spots == 1:
        for (lat, lon), level in itertools.product(SPOT_POSITIONS, SPOT_LEVELS):
            out.append({"spot0_level": level, "spot0_size": SPOT_SIZE0, "spot0_lat": lat, "spot0_lon": lon})
    else:
        for (a, b), levels in itertools.product(itertools.combinations(SPOT_POSITIONS, 2),
                                                ((SPOT_LEVELS[0],) * 2, SPOT_LEVELS)):
            s = {}
            for i, ((lat, lon), level) in enumerate(zip((a, b), levels)):
                s |= {f"spot{i}_level": level, f"spot{i}_size": SPOT_SIZE0, f"spot{i}_lat": lat, f"spot{i}_lon": lon}
            out.append(s)
    return out


def residual_function(scene, data, values):
    """Whitened residuals as a function of a flat parameter vector, and that
    vector at ``values``."""
    from jax.flatten_util import ravel_pytree

    theta, unravel = ravel_pytree({k: jnp.asarray(v, dtype=float) for k, v in values.items()})

    def residuals(t):
        return whitened_residuals(scene(**unravel(t)), data)

    return residuals, theta, unravel


def mackay_error_scale(residuals, theta, prior_precision, n=None, iters=200):
    """MacKay's re-estimate of the error-bar scale at a MAP ``theta``:
    s² = χ² / (N - γ), γ = tr[βJᵀJ (βJᵀJ + P)⁺], β = 1/s², with P the prior
    precision in ``theta``'s coordinates (0 for flat priors, which each count
    as one measured parameter) and N the number of independent data (default:
    the residuals'). The same fixed point as virgil.imaging.error_scale, for a
    parametric model. Returns (s, γ)."""
    r = residuals(theta)
    chi2, n = float(jnp.sum(r**2)), int(r.size if n is None else n)
    jac = np.asarray(jax.jacfwd(residuals)(theta))
    a = jac.T @ jac
    p = np.diag(np.asarray(prior_precision, dtype=float))
    s2, gamma = chi2 / n, 0.0
    for _ in range(iters):
        beta = 1.0 / s2
        gamma = float(np.trace(np.linalg.pinv(beta * a + p) @ (beta * a)))
        new = chi2 / max(n - gamma, 1.0)
        if abs(new - s2) <= 1e-12 * s2:
            break
        s2 = new
    return float(np.sqrt(s2)), gamma


def map_log_evidence(scene, data, values, scales):
    """Laplace log evidence of the map's coefficients at the MAP ``values``
    (other parameters held fixed): -χ²/2 - Σ(c/s)²/2 - ½ log det(I + S JᵀJ S)."""
    fixed = {k: v for k, v in values.items() if k != "data"}
    c = jnp.asarray(values["data"], dtype=float)

    def residuals(coeffs):
        return whitened_residuals(scene(**fixed, data=coeffs), data)

    r = residuals(c)
    jac = np.asarray(jax.jacfwd(residuals)(c))
    s = np.asarray(scales)
    m = np.eye(s.size) + (s[:, None] * (jac.T @ jac)) * s[None, :]
    _, logdet = np.linalg.slogdet(m)
    return float(-0.5 * jnp.sum(r**2) - 0.5 * np.sum((np.asarray(c) / s) ** 2) - 0.5 * logdet)


def run_starts(scene, priors, data, starts, max_steps=None):
    """Fit from each start; returns every start's record and the best by χ²."""
    records = []
    for i, init in enumerate(starts):
        t0 = time.time()
        kwargs = {} if max_steps is None else {"max_steps": max_steps}
        result = fit(scene, priors, data, init=init, **kwargs)
        values = {k: np.asarray(result.values[k]) for k in priors}
        chi2, n = chi2_of(scene(**values), data)
        info = result.info
        records.append(dict(values=values, chi2=chi2, n=n, steps=int(np.ravel(info.get("steps", -1))[0]),
                            grad_norm=float(np.ravel(info.get("grad_norm", np.nan))[0]),
                            converged=bool(np.ravel(info.get("converged", True))[0]), seconds=time.time() - t0))
        print(f"  start {i}: χ²/N={chi2 / n:.3f} steps={records[-1]['steps']} "
              f"grad={records[-1]['grad_norm']:.2e} ({records[-1]['seconds']:.0f} s)", flush=True)
    best = min(records, key=lambda r: r["chi2"])
    return records, best




def find_companion(data):
    """The contest pipeline's primary (``fit_primary``: an elliptical
    limb-darkened disk) and companion map (``companion_map``: a linear flux
    map over ±λ/B_min, emission only, with the primary's disk left out; it
    began here and moved into scripts/contest_images.py), a companion kept at
    SNR >= ``COMPANION_SNR``. Returns the companion (or None), the beam, the
    starting field and the search's peak."""
    from virgil.imaging import beam, starting_image

    resolution = beam(data)
    start = starting_image(data, star=True, oversample=4.0, hole_mas=0.5 * resolution.minor_mas)
    field = float(start.env.pixel_scale_mas * np.shape(start.env.log_brightness)[0])
    primary, _, _ = fit_primary(data, resolution, "ellipse", start)
    peak = companion_map(data, resolution, primary)
    kept = peak["snr"] >= COMPANION_SNR
    print(f"companion search (±{peak['half']:.1f} mas, disk < {peak['hole']:.2f} mas excluded): peak SNR "
          f"{peak['snr']:.2f} at ({peak['dra']:.3g}, {peak['ddec']:.3g}) mas, flux {peak['flux']:.3g}: "
          f"{'kept' if kept else 'below threshold'}", flush=True)
    companion = {k: peak[k] for k in ("dra", "ddec", "flux")} if kept else None
    return companion, resolution, field, peak


def run(config, data_path, out_dir, ydeg=None, smoke=False, max_starts=None):
    from virgil.oidata import OIData

    require_harmonix()
    ydeg = ydeg or DEFAULT_YDEG[config]
    if smoke:
        ydeg = min(ydeg, 3)
    max_steps = 3 if smoke else None
    label = f"2004_data2_harmonix_{config}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = OIData(str(data_path))

    companion, resolution, field, peak = find_companion(data)
    radius_range = (0.025 * resolution.minor_mas, 0.25 * field)
    radius0 = uniform_disk_radius(data, 2 * radius_range[0], 2 * radius_range[1])
    radius0 = float(np.clip(radius0, 1.05 * radius_range[0], 0.95 * radius_range[1]))
    print(f"{label}: ydeg={ydeg} beam {resolution.major_mas:.2f}x{resolution.minor_mas:.2f} mas, field {field:.1f} mas, "
          f"uniform-disk radius {radius0:.3f} mas, companion {companion}", flush=True)

    scene = Scene(config, ydeg, companion)
    priors = base_priors(radius_range, companion, resolution.major_mas)
    extra = {}
    if scene.n_spots:
        priors |= spot_priors(scene.n_spots)
        starts = [b | s for b in base_starts(radius0, companion) for s in spot_starts(scene.n_spots)]
        if smoke:
            starts = starts[:1]
        if max_starts:
            starts = starts[:max_starts]
        records, best = run_starts(scene, priors, data, starts, max_steps)
        precision = {k: np.zeros(np.shape(v)) for k, v in best["values"].items()}
    else:
        sigmas = SIGMAS[[2, 6]] if smoke else SIGMAS
        ncoef = (ydeg + 1) ** 2 - 1
        records, scan, warm = [], [], None
        for sigma in sigmas:
            scales = map_prior_scales(ydeg, sigma)
            p = priors | {"data": dist.Normal(jnp.zeros(ncoef), jnp.asarray(scales))}
            starts = [b | {"data": np.zeros(ncoef)} for b in base_starts(radius0, companion)]
            if warm is not None:
                starts = [warm] + starts
            if smoke:
                starts = starts[:1]
            if max_starts:
                starts = starts[:max_starts]
            print(f" sigma={sigma:.3g}", flush=True)
            recs, best_s = run_starts(scene, p, data, starts, max_steps)
            log_z = map_log_evidence(scene, data, best_s["values"], scales)
            print(f" sigma={sigma:.3g}: best χ²/N={best_s['chi2'] / best_s['n']:.3f} log Z={log_z:.2f}", flush=True)
            for r in recs:
                r["sigma"] = float(sigma)
            records += recs
            scan.append((float(sigma), log_z, best_s))
            warm = best_s["values"]
        i_best = int(np.argmax([s[1] for s in scan]))
        sigma, _, best = scan[i_best]
        edge = i_best in (0, len(scan) - 1)
        extra = {"sigmas": np.array([s[0] for s in scan]), "log_evidence": np.array([s[1] for s in scan]),
                 "sigma": sigma, "sigma_on_edge": edge,
                 "scan_chi2": np.array([s[2]["chi2"] for s in scan])}
        scales = map_prior_scales(ydeg, sigma)
        precision = {k: np.zeros(np.shape(v)) for k, v in best["values"].items()} | {"data": 1.0 / scales**2}

    values = best["values"]
    residuals, theta, _ = residual_function(scene, data, values)
    from jax.flatten_util import ravel_pytree

    prec_vec, _ = ravel_pytree({k: jnp.asarray(precision[k], dtype=float) for k in values})
    scale, gamma = mackay_error_scale(residuals, theta, prec_vec, n=int(data.n_independent))

    model = scene(**values)
    fov = SCORE_NPIX * SCORE_PIXEL
    image = np.asarray(model.render(SCORE_NPIX, fov))
    star = model.star if companion else model
    star_image = np.asarray(star.render(SCORE_NPIX, fov))
    chi2, n = best["chi2"], best["n"]
    out = dict(config=config, ydeg=ydeg, smoke=smoke, ref_image=image, ref_fov=fov, image=image, fov=fov,
               star_image=star_image, chi2=chi2, n_data=n, chi2_red=chi2 / n, error_scale=scale, gamma=gamma,
               companion_found=companion is not None, companion_peak=np.array([peak[k] for k in ("dra", "ddec", "flux", "snr")]),
               start_chi2=np.array([r["chi2"] for r in records]), start_steps=np.array([r["steps"] for r in records]),
               start_grad_norm=np.array([r["grad_norm"] for r in records]),
               start_converged=np.array([r["converged"] for r in records]),
               best_steps=best["steps"], best_grad_norm=best["grad_norm"], best_converged=best["converged"],
               elapsed=time.time() - t0, **extra)
    out |= {f"param_{k}": np.asarray(v) for k, v in values.items()}
    np.savez(out_dir / f"{label}.npz", **out)

    lines = [f"{label}: {len(records)} fits, ydeg={ydeg}{' (smoke)' if smoke else ''}, {out['elapsed']:.0f} s",
             f"chi2/N = {chi2 / n:.3f} on the quoted errors (chi2={chi2:.1f}, N={n}); "
             f"MacKay error scale s={scale:.3f} (gamma={gamma:.1f} effective parameters)",
             f"best start: steps={best['steps']} grad_norm={best['grad_norm']:.2e} converged={best['converged']}; "
             f"start chi2/N: {np.array2string(out['start_chi2'] / n, precision=2, max_line_width=200)}"]
    for k, v in values.items():
        if k == "data":
            lines.append(f"  data: {v.size} coefficients, rms {float(np.sqrt(np.mean(v**2))):.3g}")
        else:
            lines.append(f"  {k:12s} {float(v):.4g}")
    if extra:
        lines.append("sigma scan (sigma, log Z, chi2/N): " + ", ".join(
            f"({s:.3g}, {z:.1f}, {c / n:.2f})" for s, z, c in zip(extra["sigmas"], extra["log_evidence"], extra["scan_chi2"])))
        lines.append(f"chosen sigma={extra['sigma']:.3g}" + (" (on the grid's edge)" if extra["sigma_on_edge"] else ""))
    lines.append(f"score: python scripts/score_2004.py {out_dir / (label + '.npz')}")
    text = "\n".join(lines)
    print(text, flush=True)
    (out_dir / f"{label}.txt").write_text(text + "\n")
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="~/data/imaging_contests/2004/2004-data2.fits", help="the data2 OIFITS file")
    parser.add_argument("--out", default="spotted_star_harmonix")
    parser.add_argument("--config", choices=CONFIGS, required=True)
    parser.add_argument("--ydeg", type=int, default=None,
                        help="maximum degree (default 12 for spots, 6 for map)")
    parser.add_argument("--max-starts", type=int, default=None, help="use only the first N starts (per sigma for map)")
    parser.add_argument("--smoke", action="store_true", help="ydeg <= 3, one start, three steps: check that it runs")
    args = parser.parse_args(argv)
    if args.ydeg is not None and args.ydeg < 1:
        parser.error("--ydeg must be at least 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    run(args.config, pathlib.Path(args.data).expanduser(), pathlib.Path(args.out).expanduser(), args.ydeg,
        args.smoke, args.max_starts)


if __name__ == "__main__":
    main()
