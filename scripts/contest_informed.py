"""Informed fits of the imaging contests' data: the model family the contest paper describes, fitted from the data.

    python scripts/contest_informed.py --list
    python scripts/contest_informed.py --task N --model FAMILY [--data DIR] [--out DIR] [--starts 32] [--smoke]

This is **not** a blind reconstruction, and its results must never be quoted as one. It uses what a contest paper
said the object *is* (a spotted star with a companion; a thin disk; a star, a disk rim, a planet and an outer disk;
a ringed disk with companions) to choose a parametric model. It uses nothing the paper said about the *values*: no diameter, position, flux ratio or angle
from a paper is a start, a bound or a constant here. Every start is drawn from priors whose scales come from the
data's own baselines, so the published parameters stay a test of the fit (they are compared afterwards, by the
campaign's sealed stage).

A family is a ladder of nested models, simplest first (FAMILIES). Each rung is fitted from many random starts drawn
from its priors, plus the best fit of the rung below; the best start of each rung is kept and the rungs are compared
by BIC on the quoted errors. Writes into --out:

After the ladder, a simple non-parametric image (a Gaussian-process field) is fitted on top of the chosen model,
held fixed, to show what resolved light the parametric model leaves out (<label>_<family>_rest.npz).

Spectral information is used in two ways. When the data have more than one channel, every component but the
reference gets its own spectral index, so the model is chromatic wherever the object can be. And where a file
carries OI_FLUX spectra they are fitted with the visibilities (scale and linear trend marginalised).

- <label>_<family>.json: per rung, the best parameters, chi2, N, k, BIC, how many starts reached the best chi2
  (a small number means the optimum is hard to find or not unique), and the chosen rung;
- result.json: status and checks for the campaign (`converged`, `starts_agree`), with chi2/N of the chosen rung.

Task numbers and files are those of scripts/contest_images.py (TASKS).
"""

import argparse
import ast
import json
import pathlib
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import numpyro.distributions as dist  # noqa: E402

import virgil  # noqa: E402
import virgil.models as vm  # noqa: E402
from virgil.fitting import fit  # noqa: E402
from virgil.grid_fit import likelihood_grid  # noqa: E402
from virgil.oidata import OIData  # noqa: E402
from virgil.spectra import PowerLaw  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
REACH = 1.5  # positions are searched out to this many fringe periods of the shortest baseline (see prior_of)
PEAKS = 6  # likelihood-map peaks refined per new component
TIME_LIMIT = 300.0  # seconds per start; a start that runs out is recorded as not converged
AGREE = 1.01  # a start "reached the best" when its chi2 is within 1% of it
MIN_AGREE = 2  # fewer starts than this at the best chi2: the optimum was found once, by luck


def tasks():
    """TASKS of contest_images.py, read without importing it (that module sets up a whole imaging run)."""
    tree = ast.parse((REPO / "scripts" / "contest_images.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "TASKS" for t in node.targets):
            return eval(compile(ast.Expression(node.value), "contest_images.py", "eval"), {"dict": dict})
    raise KeyError("TASKS")


def scales(datasets):
    """(resolution, field) in mas from the data's own baselines: lambda / 2 B_max and lambda / B_min."""
    mas = 180.0 / np.pi * 3600e3
    longest, shortest = 0.0, np.inf
    for d in datasets:
        b = np.hypot(np.asarray(d.u, float), np.asarray(d.v, float)).ravel()
        wavel = np.asarray(d.wavel, float).ravel()
        b = b[b > 0]
        longest = max(longest, float(b.max() / wavel.min()))
        shortest = min(shortest, float(b.min() / wavel.max()))
    return mas / (2.0 * longest), mas / shortest


# Each family is a ladder of rungs, simplest first: (name, {component: (maker, {parameter path: prior kind})}).
# A maker is a virgil class, or "RingedDisk" (make_component). Prior kinds are scale-free words that `prior_of`
# turns into distributions from the data's resolution and field; no number here comes from a paper. The first
# component of a rung is the flux reference (flux 1); every other component's flux is relative to it.
POINT = ("PointSource", {})
SMALL_STAR = ("UniformDisk", {"diam": "small"})
LD_STAR = ("LimbDarkenedDisk", {"diam": "size", "u": "limb"})
ELLIPTICAL_STAR = ("EllipticalLimbDarkenedDisk", {"diam": "size", "ratio": "ratio", "pa": "angle", "u": "limb"})
COMPANION = ("PointSource", {"flux": "faint", "dra": "far", "ddec": "far"})
SMALL_DISK = ("UniformDisk", {"diam": "small", "flux": "faint", "dra": "far", "ddec": "far"})
SPOT = ("UniformDisk", {"diam": "small", "flux": "faint", "dra": "near", "ddec": "near"})
SURFACE_SPOT = ("UniformDisk", {"diam": "small", "flux": "faint", "dra": "far", "ddec": "far"})
HALO = ("GaussianDisk", {"sigma": "size", "flux": "bright"})
ELLIPSE = ("EllipticalGaussian", {"fwhm": "size", "ratio": "ratio", "pa": "angle", "flux": "bright"})
OFFSET_ELLIPSE = ("EllipticalGaussian", {"fwhm": "size", "ratio": "ratio", "pa": "angle", "flux": "bright", "dra": "near", "ddec": "near"})
RIM = ("ModulatedGaussianRim", {"diam": "size", "fwhm": "size", "inc": "inclination", "pa": "angle",
                                "az_amps": "modulation", "az_pas": "full_angle", "flux": "bright"})
ARC = ("GaussianArc", {"radius": "size", "width": "size", "length": "size", "pa": "full_angle", "flux": "bright"})


def rings(*names):
    """An inclined disk of Gaussian rings; each ring after the first has a weight relative to the first."""
    params = {"inc": "inclination", "pa": "angle", "flux": "bright"}
    for i, name in enumerate(names):
        params[f"profile.{name}.radius"] = "size"
        params[f"profile.{name}.sigma"] = "size"
        if i:
            params[f"profile.{name}.weight"] = "bright"
    return ("RingedDisk", params)


class Scene:
    """A rung built by a function of named parameters, for models a flat template cannot express: a stellar surface
    in spherical harmonics, a disk with azimuthal structure. ``kinds`` maps parameter names to prior kinds, ``make``
    turns ``{name: value}`` into a virgil model, ``components`` names its parts, and ``positions`` names the parts
    that are placed by a likelihood map when the rung adds them."""

    def __init__(self, kinds, make, components, positions=()):
        self._kinds, self.make, self.components, self.positions = kinds, make, tuple(components), tuple(positions)

    def kinds(self, wavel0=None):
        if not wavel0:
            return dict(self._kinds)
        out = {}
        for name, kind in self._kinds.items():
            if name.endswith(".flux"):
                out[name + ".ratio"], out[name + ".index"] = kind, "colour"
            else:
                out[name] = kind
        return out


def flux_of(p, name, wavel0):
    """A component's flux from the fitted parameters: a number, or a power law when the model is chromatic."""
    if f"{name}.flux.ratio" in p:
        return PowerLaw(p[f"{name}.flux.ratio"], p[f"{name}.flux.index"], wavel0)
    return p[f"{name}.flux"]


def harmonic_star(degree, envelope=False, companion=False):
    """A limb-darkened star whose surface brightness is a spherical-harmonic map of the given degree (harmonix), so
    structure sits on the stellar disk by construction and is resolved only as far as the degree allows. Degree 0 is
    a plain limb-darkened disk. The star is seen once, so its axis is fixed (equator-on, no obliquity): the
    coefficients describe the visible hemisphere. Optionally a Gaussian envelope, and a point companion."""
    n = (degree + 1) ** 2 - 1
    kinds = {"star.radius": "size", "star.u": "limb", **{f"star.y{i}": "harmonic" for i in range(n)}}
    components = ["star"]
    if envelope:
        kinds |= {"envelope.sigma": "size", "envelope.flux": "bright"}
        components.append("envelope")
    if companion:
        kinds |= {"companion.flux": "faint", "companion.dra": "far", "companion.ddec": "far"}
        components.append("companion")

    def make(p, wavel0=None):
        import jax.numpy as jnp
        from harmonix.harmonix import Harmonix
        from jaxoplanet.starry import Surface, Ylm

        y = jnp.concatenate([jnp.ones(1), jnp.stack([jnp.asarray(p[f"star.y{i}"], float).reshape(()) for i in range(n)])]) if n else jnp.ones(1)
        surface = Surface(y=Ylm.from_dense(y), inc=jnp.pi / 2.0, obl=0.0, period=1.0, u=[p["star.u"]])
        parts = {"star": vm.HarmonixModel(Harmonix(surface, p["star.radius"]), observation_time=0.0)}
        if envelope:
            parts["envelope"] = vm.GaussianDisk(sigma=p["envelope.sigma"], flux=flux_of(p, "envelope", wavel0))
        if companion:
            parts["companion"] = vm.PointSource(flux=flux_of(p, "companion", wavel0), dra=p["companion.dra"], ddec=p["companion.ddec"])
        return vm.System(**parts)

    return Scene(kinds, make, components, positions=("companion",) if companion else ())


def modulated_disk(order, star=True):
    """An inclined disk with a power-law radial profile between an inner and an outer radius, whose brightness varies
    around the disk with azimuthal harmonics up to ``order`` (0 is axisymmetric), and optionally a central star.
    An inclined disk is brighter on one side, so there is no reason to expect it to be symmetric."""
    kinds = {"disk.r_out": "size", "disk.inner": "fraction", "disk.index": "slope", "disk.inc": "inclination", "disk.pa": "angle"}
    for m in range(1, order + 1):
        kinds |= {f"disk.c{m}": "wave", f"disk.s{m}": "wave"}
    if star:
        kinds["disk.flux"] = "bright"

    def make(p, wavel0=None):
        import jax.numpy as jnp

        profile = vm.RadialPowerLaw(r_in=p["disk.inner"] * p["disk.r_out"], r_out=p["disk.r_out"], index=p["disk.index"])
        waves = vm.Harmonics(c=jnp.stack([jnp.asarray(p[f"disk.c{m}"], float).reshape(()) for m in range(1, order + 1)]),
                             s=jnp.stack([jnp.asarray(p[f"disk.s{m}"], float).reshape(()) for m in range(1, order + 1)])) if order else None
        disk = vm.MultipoleDisk(profile, harmonics=waves, inc=p["disk.inc"], pa=p["disk.pa"],
                                flux=flux_of(p, "disk", wavel0) if star else 1.0)
        return vm.System(star=vm.PointSource(), disk=disk) if star else vm.System(disk=disk)

    return Scene(kinds, make, ["star", "disk"] if star else ["disk"])


FAMILIES = {
    # Lawson et al. 2004, data1: a star inside an asymmetric shell. The star is a point at the centre of every rung.
    "star_in_shell": [
        ("star+halo", {"star": POINT, "halo": HALO}),
        ("star+shell", {"star": POINT, "shell": RIM}),
        ("star+shell+halo", {"star": POINT, "shell": RIM, "halo": HALO}),
    ],
    # Lawson et al. 2004, data2: a binary whose primary is a limb-darkened elliptical disk with a bright spot.
    "spotted_star_binary": [
        ("star", {"star": ELLIPTICAL_STAR}),
        ("star+companion", {"star": ELLIPTICAL_STAR, "companion": SMALL_DISK}),
        ("star+companion+spot", {"star": ELLIPTICAL_STAR, "companion": SMALL_DISK, "spot": SPOT}),
    ],
    # Lawson et al. 2006: a geometrically thin disk. Inclined, so one side is brighter: the ladder adds azimuthal
    # structure, which a centro-symmetric model cannot have (its closure phases are all zero or pi).
    "thin_disk": [
        ("disk", modulated_disk(0, star=False)),
        ("star+disk", modulated_disk(0)),
        ("star+disk, one azimuthal harmonic", modulated_disk(1)),
        ("star+disk, two azimuthal harmonics", modulated_disk(2)),
    ],
    # Cotton et al. 2008: an AGB star. A stellar surface in spherical harmonics, inside an envelope.
    "evolved_star_with_surface_structure": [
        ("star", harmonic_star(0)),
        ("star+envelope", harmonic_star(0, envelope=True)),
        ("surface degree 1+envelope", harmonic_star(1, envelope=True)),
        ("surface degree 2+envelope", harmonic_star(2, envelope=True)),
        ("surface degree 3+envelope", harmonic_star(3, envelope=True)),
    ],
    # Cotton et al. 2008: an AGN. An unresolved core and elongated emission that need not be centred on it.
    "agn": [
        ("core+dust", {"core": POINT, "dust": OFFSET_ELLIPSE}),
        ("core+dust+extended", {"core": POINT, "dust": OFFSET_ELLIPSE, "extended": OFFSET_ELLIPSE}),
    ],
    # Malbet et al. 2010: a red supergiant's structured surface, and a much fainter companion far outside the disk.
    # The surface is a spherical-harmonic map on the stellar disk; only the companion is free to sit elsewhere.
    "structured_supergiant_with_companion": [
        ("star", harmonic_star(0)),
        ("star+companion", harmonic_star(0, companion=True)),
        ("surface degree 1+companion", harmonic_star(1, companion=True)),
        ("surface degree 2+companion", harmonic_star(2, companion=True)),
        ("surface degree 3+companion", harmonic_star(3, companion=True)),
    ],
    # Sanchez-Bermudez et al. 2018: a star, an inner disk with a bright rim, a planet, and an outer disk.
    "star_rim_planet_disk": [
        ("star+outer", {"star": SMALL_STAR, "outer": ELLIPSE}),
        ("star+rim+outer", {"star": SMALL_STAR, "rim": RIM, "outer": ELLIPSE}),
        ("star+rim+outer+planet", {"star": SMALL_STAR, "rim": RIM, "outer": ELLIPSE, "planet": COMPANION}),
    ],
    # The 2022 contest: high-contrast tests. A star, possibly extended light, and a faint companion.
    "star_with_faint_companion": [
        ("star", {"star": SMALL_STAR}),
        ("star+halo", {"star": SMALL_STAR, "halo": HALO}),
        ("star+halo+companion", {"star": SMALL_STAR, "halo": HALO, "companion": COMPANION}),
    ],
    # The 2024 contest, Obj1: a Wolf-Rayet pinwheel of several turns. virgil has no spiral component, so each turn is
    # approximated by one arc; the arcs are not tied to a common pitch. A stand-in until a spiral exists in virgil.
    "pinwheel_arcs": [
        ("star+envelope", {"star": POINT, "envelope": HALO}),
        ("star+envelope+arc", {"star": POINT, "envelope": HALO, "arc": ARC}),
        ("star+envelope+two arcs", {"star": POINT, "envelope": HALO, "arc": ARC, "arc_b": ARC}),
        ("star+envelope+three arcs", {"star": POINT, "envelope": HALO, "arc": ARC, "arc_b": ARC, "arc_c": ARC}),
    ],
    # The 2024 contest, Obj2: an inclined disk with several rings and gaps, and point-like companions.
    "ringed_disk_with_companions": [
        ("star+one ring", {"star": POINT, "disk": rings("a")}),
        ("star+two rings", {"star": POINT, "disk": rings("a", "b")}),
        ("star+three rings", {"star": POINT, "disk": rings("a", "b", "c")}),
        ("star+three rings+companion", {"star": POINT, "disk": rings("a", "b", "c"), "companion": COMPANION}),
    ],
}
# A value for the constructor before fitting; every free parameter is then set from a prior draw.
PLACEHOLDER = {"diam": 1.0, "fwhm": 1.0, "sigma": 1.0, "radius": 1.0, "width": 1.0, "length": 1.0, "ratio": 0.9, "pa": 0.0,
               "u": 0.5, "inc": 30.0, "az_amps": 0.1, "az_pas": 0.0, "flux": 0.1, "weight": 1.0, "dra": 0.0, "ddec": 0.0}


def prior_of(kind, res, field):
    return {
        "size": dist.LogUniform(res / 5.0, field),  # anything from well under the resolution to the whole field
        "small": dist.LogUniform(res / 20.0, 4.0 * res),  # barely resolved or unresolved
        "ratio": dist.Uniform(0.3, 1.0),
        "angle": dist.Uniform(0.0, 180.0),
        "full_angle": dist.Uniform(0.0, 360.0),
        "inclination": dist.Uniform(0.0, 80.0),
        "limb": dist.Uniform(0.0, 1.0),
        "harmonic": dist.Normal(0.0, 0.3),  # a spherical-harmonic coefficient of the surface, relative to the mean
        "wave": dist.Uniform(-0.7, 0.7),  # an azimuthal harmonic of a disk's brightness, relative to its mean
        "fraction": dist.Uniform(0.02, 0.95),
        "slope": dist.Uniform(-3.0, 1.0),
        "modulation": dist.Uniform(0.0, 1.0),
        # A component's spectrum relative to the reference component: flux ratio x (lambda / lambda_0) ** index.
        "colour": dist.Uniform(-8.0, 8.0),
        "faint": dist.LogUniform(1e-4, 1.0),  # relative to the star, which is fixed at flux 1
        "bright": dist.LogUniform(1e-3, 10.0),
        # Anywhere the data can place a source. lambda / B_min is the fringe period of the shortest baseline, not an
        # edge: longer baselines still locate a source beyond it. The blind images' field rule also grows to 1.5x.
        "far": dist.Uniform(-REACH * field, REACH * field),
        "near": dist.Uniform(-3.0 * res, 3.0 * res),  # a feature on or beside the star (a few resolution elements)
    }[kind]


VECTORS = ("u", "az_amps", "az_pas")  # parameters that are arrays of one coefficient here


def chromatic(rung):
    """The rung with every fitted flux given its own spectral index: ``flux`` becomes ``flux.ratio`` (at the data's
    median wavelength) and ``flux.index``. Used whenever the data have more than one channel, so each component
    may have its own colour, as the contests' polychromatic objects do."""
    out = {}
    for name, (maker, params) in rung.items():
        new = {}
        for path, kind in params.items():
            if path == "flux":
                new["flux.ratio"], new["flux.index"] = kind, "colour"
            else:
                new[path] = kind
        out[name] = (maker, new)
    return out


def make_component(maker, values, wavel0=None):
    """A virgil component from ``{parameter path: number}``."""
    values = dict(values)
    if "flux.ratio" in values:
        values["flux"] = PowerLaw(values.pop("flux.ratio"), values.pop("flux.index"), wavel0)
    if maker != "RingedDisk":
        return getattr(vm, maker)(**values)
    parts = {}
    for path, value in values.items():
        if path.startswith("profile."):
            _, ring, what = path.split(".")
            parts.setdefault(ring, {})[what] = value
    profile = vm.RadialSum(**{ring: (vm.GaussianRing(radius=v["radius"], sigma=v["sigma"]), v.get("weight", 1.0))
                              for ring, v in parts.items()})
    return vm.MultipoleDisk(profile, inc=values["inc"], pa=values["pa"], flux=values["flux"])


def build(rung, res, field, values=None, wavel0=None):
    """(template System, {path: prior}) of one rung, with any given ``values`` ({path: number}) set on the template.
    ``wavel0`` is the reference wavelength of chromatic fluxes (see `chromatic`)."""
    parts, priors = {}, {}
    for name, (maker, params) in rung.items():
        kwargs = {}
        for p in params:
            leaf = p.rsplit(".", 1)[-1]
            value = (values or {}).get(f"{name}.{p}", 0.0 if p == "flux.index" else PLACEHOLDER["flux" if p == "flux.ratio" else leaf])
            kwargs[p] = np.array([value]) if leaf in VECTORS else value
        parts[name] = make_component(maker, kwargs, wavel0)
        for p, kind in params.items():
            priors[f"{name}.{p}"] = prior_of(kind, res, field)
    return vm.System(**parts), priors


def seeds_from_map(name, rung, datasets, res, field, carry, wavel0=None):
    """Starts for a newly added component, from a likelihood map over its position and flux with the rest of the
    scene held at the best fit of the rung below. The map's highest, well separated peaks are returned as
    ``{path: value}`` (virgil's documented companion search; nothing is assumed about where the component is)."""
    kinds = rung[name][1]
    half = (REACH * field) if kinds.get("dra") == "far" else 3.0 * res
    n = int(min(121, 2 * round(2.0 * half / res) + 1))  # about half a resolution element per step
    axis = np.linspace(-half, half, n)
    fluxes = 10.0 ** np.linspace(-3.5, 0.0, 12)
    template, _ = build(rung, res, field, carry, wavel0)
    flux_path = f"{name}.flux.ratio" if "flux.ratio" in kinds else f"{name}.flux"
    grid = {f"{name}.dra": axis, f"{name}.ddec": axis, flux_path: fluxes}
    data = datasets[0] if len(datasets) == 1 else datasets
    loglike = np.asarray(likelihood_grid(template, data=data, grid=grid))
    best_flux = np.argmax(loglike, axis=2)
    surface = np.max(loglike, axis=2)
    seeds = []
    for flat in np.argsort(surface, axis=None)[::-1]:
        i, j = np.unravel_index(flat, surface.shape)
        if all(np.hypot(axis[i] - q[f"{name}.dra"], axis[j] - q[f"{name}.ddec"]) > 2.0 * res for q in seeds):
            seed = {f"{name}.dra": float(axis[i]), f"{name}.ddec": float(axis[j]), flux_path: float(fluxes[best_flux[i, j]])}
            if "flux.index" in kinds:
                seed[f"{name}.flux.index"] = 0.0  # the map is made grey; the fit then frees the colour
            seeds.append(seed)
        if len(seeds) == PEAKS:
            break
    return seeds


def draw(priors, rng):
    """One start: each parameter drawn from its own prior."""
    out = {}
    for path, p in priors.items():
        if isinstance(p, dist.Normal):
            value = float(rng.normal(float(p.loc), float(p.scale)))
        elif isinstance(p, dist.LogUniform):
            value = float(np.exp(rng.uniform(np.log(float(p.low)), np.log(float(p.high)))))
        else:
            value = float(rng.uniform(float(p.low), float(p.high)))
        out[path] = np.array([value]) if path.rsplit(".", 1)[-1] in VECTORS else value
    return out


def model_of(rung, values, res, field, wavel0=None):
    """The virgil model of a rung at ``{parameter: value}`` (for a template rung, of its chromatic form if ``wavel0``)."""
    if isinstance(rung, Scene):
        return rung.make(values, wavel0)
    return build(chromatic(rung) if wavel0 else rung, res, field, values, wavel0)[0]


def map_seeds(base, datasets, res, field):
    """Peaks of a likelihood map for one more point source beside ``base`` (a fitted model, held fixed)."""
    half = REACH * field
    n = int(min(121, 2 * round(2.0 * half / res) + 1))
    axis, fluxes = np.linspace(-half, half, n), 10.0 ** np.linspace(-3.5, 0.0, 12)
    template = vm.System(base=base, extra=vm.PointSource(flux=1e-3))
    data = datasets[0] if len(datasets) == 1 else datasets
    loglike = np.asarray(likelihood_grid(template, data=data, grid={"extra.dra": axis, "extra.ddec": axis, "extra.flux": fluxes}))
    surface, best_flux = np.max(loglike, axis=2), np.argmax(loglike, axis=2)
    seeds = []
    for flat in np.argsort(surface, axis=None)[::-1]:
        i, j = np.unravel_index(flat, surface.shape)
        if all(np.hypot(axis[i] - q[0], axis[j] - q[1]) > 2.0 * res for q in seeds):
            seeds.append((float(axis[i]), float(axis[j]), float(fluxes[best_flux[i, j]])))
        if len(seeds) == PEAKS:
            break
    return seeds


def fit_rung(name, rung, datasets, res, field, rng, starts, carry, max_steps, below=(), wavel0=None, previous=None):
    data = datasets[0] if len(datasets) == 1 else datasets
    inits = []
    if isinstance(rung, Scene):
        priors = {k: prior_of(kind, res, field) for k, kind in rung.kinds(wavel0).items()}
        target = lambda **p: rung.make(p, wavel0)  # noqa: E731
        # A component with a position that this rung adds is placed by a likelihood map beside the fit below.
        for comp in [c for c in rung.positions if c not in below]:
            if previous is not None and carry:
                for dra, ddec, flux in map_seeds(model_of(previous, carry, res, field, wavel0), datasets, res, field):
                    seed = {f"{comp}.dra": dra, f"{comp}.ddec": ddec,
                            (f"{comp}.flux.ratio" if wavel0 else f"{comp}.flux"): flux}
                    if wavel0:
                        seed[f"{comp}.flux.index"] = 0.0
                    inits.append({**draw(priors, rng), **carry, **seed})
        # New surface or azimuthal coefficients start at zero from the fit below: the simpler model is the start.
        if carry:
            grown = {k: 0.0 for k, kind in rung.kinds(wavel0).items() if kind in ("harmonic", "wave") and k not in carry}
            inits.append({**draw(priors, rng), **grown, **carry})
    else:
        target, priors = build(rung, res, field, None, wavel0)
        # Starts: for each component this rung adds that has a position, the peaks of its likelihood map with the
        # rest of the scene at the best fit below; then random draws from the priors.
        for comp in [c for c in rung if c not in below and "dra" in rung[c][1]]:
            for seed in seeds_from_map(comp, rung, datasets, res, field, carry, wavel0):
                inits.append({**draw(priors, rng), **carry, **seed})
        if carry and not inits:
            inits.append({**draw(priors, rng), **carry})
    inits += [draw(priors, rng) for _ in range(starts)]
    runs, n = [], 0
    for init in inits:
        init = {k: (np.array([float(np.ravel(v)[0])]) if k.rsplit(".", 1)[-1] in VECTORS and not isinstance(rung, Scene)
                    else float(np.ravel(v)[0])) for k, v in init.items() if k in priors}
        try:
            r = fit(target, priors, data, init=init, max_steps=max_steps, time_limit=TIME_LIMIT)
        except Exception as err:  # noqa: BLE001 a start that fails is recorded, not hidden
            runs.append({"chi2": float("inf"), "error": f"{type(err).__name__}: {err}"})
            continue
        chi2, n = float(np.sum(r.info["chi2"])), int(np.sum(r.info["ndata"]))
        runs.append({"chi2": chi2 if np.isfinite(chi2) else float("inf"), "converged": bool(r.info.get("converged")),
                     "values": {k: [float(x) for x in np.ravel(v)] for k, v in r.values.items()}})
    best = min(runs, key=lambda x: x["chi2"])
    k = len(priors)
    agree = sum(x["chi2"] <= AGREE * best["chi2"] for x in runs)
    errors = sorted({x["error"] for x in runs if "error" in x})
    return {"rung": name, "k": k, "n": n, "chi2": best["chi2"], "chi2_red": best["chi2"] / max(n, 1),
            "bic": best["chi2"] + k * float(np.log(max(n, 1))), "converged": best.get("converged", False),
            "starts": len(inits), "starts_at_best": int(agree), "failed_starts": sum("error" in x for x in runs),
            "errors": errors[:3], "values": best.get("values", {})}


# Families whose object is a disk also get a non-parametric disk: (azimuthal degree, name of a point companion or None).
GP_DISKS = {"thin_disk": (2, None), "star_in_shell": (2, None), "star_rim_planet_disk": (1, "planet"),
            "ringed_disk_with_companions": (1, "companion")}


def gp_disk(family, chosen, datasets, res, field, wavel0, rng, starts, max_steps, out_path):
    """A multipolar Gaussian-process disk in place of the parametric one: a central star and one inclined
    `MultipoleDisk` whose radial profile is a Gaussian process in radius and whose azimuthal harmonics (to the
    family's degree) are Gaussian processes in radius too, so rings, gaps and lopsidedness need not be named in
    advance. A point companion the ladder found keeps its parametric form and starts where the ladder put it.
    The hyperparameters are fixed, not fitted: the profile varies by e^2 over three resolution elements between
    half a resolution element and the edge of the field. It has too many latent parameters for BIC to mean
    anything, so it is reported beside the ladder and never chosen by it."""
    from virgil import gp_priors

    degree, companion = GP_DISKS[family]
    r_in, r_out = res / 2.0, REACH * field / 2.0
    ell = 3.0 * res
    values = {k: v[0] for k, v in chosen["values"].items()}

    def scene(inc, pa, flux):
        disk = vm.MultipoleDisk(
            vm.RadialGP(r_in, r_out, sigma=2.0, ell=ell),
            harmonics=vm.Harmonics(degree=degree, prior=vm.MultipoleKernel(ell_r=ell, ell_theta=8.0, sigma=0.5, n=10, r_in=r_in, r_out=r_out)),
            inc=inc, pa=pa, n_r=48, flux=PowerLaw(flux, 0.0, wavel0) if wavel0 else flux)
        parts = {"star": vm.PointSource(), "disk": disk}
        if companion and f"{companion}.dra" in values:
            ratio = values.get(f"{companion}.flux.ratio", values.get(f"{companion}.flux", 1e-2))
            parts[companion] = vm.PointSource(dra=values[f"{companion}.dra"], ddec=values[f"{companion}.ddec"],
                                               flux=PowerLaw(ratio, values.get(f"{companion}.flux.index", 0.0), wavel0) if wavel0 else ratio)
        return vm.System(**parts)

    flux_path = "disk.flux.ratio" if wavel0 else "disk.flux"
    template = scene(30.0, 0.0, 0.5)
    priors = gp_priors(template) | {flux_path: prior_of("bright", res, field), "disk.inc": prior_of("inclination", res, field),
                                    "disk.pa": prior_of("angle", res, field)}
    if wavel0:
        priors["disk.flux.index"] = prior_of("colour", res, field)
    if companion and f"{companion}.dra" in values:
        priors |= {f"{companion}.dra": prior_of("far", res, field), f"{companion}.ddec": prior_of("far", res, field),
                   (f"{companion}.flux.ratio" if wavel0 else f"{companion}.flux"): prior_of("faint", res, field)}
    data = datasets[0] if len(datasets) == 1 else datasets
    runs = []
    for _ in range(max(2, starts // 2)):
        start = scene(float(rng.uniform(0.0, 70.0)), float(rng.uniform(0.0, 180.0)), float(np.exp(rng.uniform(np.log(0.05), np.log(3.0)))))
        try:
            r = fit(start, priors, data, max_steps=max_steps, time_limit=2.0 * TIME_LIMIT)
        except Exception as err:  # noqa: BLE001
            runs.append((float("inf"), None, f"{type(err).__name__}: {err}"))
            continue
        runs.append((float(np.sum(r.info["chi2"])), r, None))
    chi2, best, _ = min(runs, key=lambda x: x[0])
    if best is None:
        return {"error": runs[0][2]}
    n = int(np.sum(best.info["ndata"]))
    fov = 2.0 * r_out * 1.1
    np.savez_compressed(out_path, image=np.asarray(best.model.render(npix=256, fov_mas=fov)), fov_mas=fov)
    scalars = {k: float(np.ravel(v)[0]) for k, v in best.values.items() if np.size(v) == 1}
    return {"chi2_red": chi2 / max(n, 1), "converged": bool(best.info.get("converged")), "azimuthal_degree": degree,
            "starts": len(runs), "starts_at_best": int(sum(c <= AGREE * chi2 for c, _, _ in runs)),
            "r_in_mas": r_in, "r_out_mas": r_out, "length_mas": ell, "values": scalars, "file": pathlib.Path(out_path).name}


def residual_image(model, datasets, res, field, out_path):
    """What resolved light is left once the parametric model is taken out: a simple non-parametric image fitted on
    top of the chosen model, which is held fixed. The image is a Gaussian-process field (virgil's documented image
    model) on a grid of half-resolution pixels across the field, with a correlation length of two resolution
    elements; only it and its share of the flux are fitted. Returns its numbers and saves the image."""
    from virgil.fields import GaussianField
    from virgil.imaging import image_priors

    scale = res / 2.0
    npix = int(min(64, max(16, round(field / scale))))
    extra = vm.Image(GaussianField(np.zeros((npix, npix)), sigma=2.0, length_mas=2.0 * res), scale, flux=0.05)
    scene = vm.System(base=model, rest=extra)
    data = datasets[0] if len(datasets) == 1 else datasets
    r = fit(scene, image_priors(scene) | {"rest.flux": dist.LogUniform(1e-4, 2.0)}, data, time_limit=2.0 * TIME_LIMIT)
    image = np.asarray(r.model.rest.render(npix=npix, fov_mas=npix * scale))
    share = float(np.ravel(r.values["rest.flux"])[0])
    np.savez_compressed(out_path, image=image, pixel_scale_mas=scale, flux=share)
    n = int(np.sum(r.info["ndata"]))
    return {"chi2_red": float(np.sum(r.info["chi2"])) / max(n, 1), "flux_relative_to_model": share, "npix": npix,
            "pixel_scale_mas": scale, "field_mas": npix * scale, "converged": bool(r.info.get("converged")), "file": pathlib.Path(out_path).name}


def run(task, family, data_dir, out_dir, starts=32, smoke=False, grey=False, use_spectra=True, residual=True, no_gp=False):
    spec = tasks()[task]
    label = f"{spec['label']}_{family}"
    paths = [pathlib.Path(data_dir) / f for f in spec["files"]]
    if spec.get("wavel"):  # the same channel selection as the blind images of this task
        import tempfile

        sys.path.insert(0, str(REPO / "scripts"))
        import contest_images

        tmp = pathlib.Path(tempfile.mkdtemp())
        paths = [contest_images.select_channels(p, spec["wavel"], tmp) for p in paths]
    datasets, spectra = [], 0
    for path in paths:
        from astropy.io import fits

        with fits.open(path) as hdus:
            has_flux = any(h.name == "OI_FLUX" for h in hdus)
        if has_flux and use_spectra:
            # The contest's flux spectra are fitted with the visibilities. They are uncalibrated, so their scale and
            # a linear trend in wavelength are marginalised: what is left constrains the components' relative colours.
            d = OIData(str(path), extras="flux")
            values = np.asarray(d.extras[0].values, float)
            level = float(np.median(values[np.isfinite(values) & (values > 0)])) if np.any(np.isfinite(values) & (values > 0)) else np.nan
            if np.isfinite(level):
                d = d.with_flux_scale(scale=(level, level), per="dataset", poly_order=1)
                spectra += 1
            else:  # a spectrum with no positive finite value cannot set its own scale: fit the visibilities alone
                d = OIData(str(path))
        else:
            d = OIData(str(path))
        datasets.append(d)
    channels = len(np.unique(np.round(np.concatenate([np.ravel(np.asarray(d.wavel, float)) for d in datasets]) * 1e9)))
    wavel0 = float(np.median(np.concatenate([np.ravel(np.asarray(d.wavel, float)) for d in datasets]))) if channels > 1 and not grey else None
    res, field = scales(datasets)
    rng = np.random.default_rng(7000 + task)
    t0, rungs, carry, below, previous = time.time(), [], {}, (), None
    for name, plain in FAMILIES[family]:
        rung = plain if isinstance(plain, Scene) else (chromatic(plain) if wavel0 else plain)
        got = fit_rung(name, rung, datasets, res, field, rng, 3 if smoke else starts, carry, 50 if smoke else None, below,
                       wavel0, previous)
        carry = {k: v[0] for k, v in got["values"].items() if len(v) == 1}
        below, previous = (rung.components if isinstance(rung, Scene) else tuple(rung)), plain
        rungs.append(got)
        print(f"{label} {name}: chi2/N={got['chi2_red']:.3f} k={got['k']} BIC={got['bic']:.1f} "
              f"starts at best {got['starts_at_best']}/{got['starts']}")
    chosen = min(rungs, key=lambda x: x["bic"])
    # chi2/N is on the quoted errors, which several contests made very small. The typical quoted error says what a
    # given chi2/N means in the data's own units: residuals of about sqrt(chi2/N) times these.
    gp = None
    if family in GP_DISKS and not no_gp:
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        gp = gp_disk(family, chosen, datasets, res, field, wavel0, rng, 3 if smoke else starts, 50 if smoke else None,
                     pathlib.Path(out_dir) / f"{label}_gp_disk.npz")
        if gp and "chi2_red" in gp:
            print(f"{label} multipolar GP disk: chi2/N={gp['chi2_red']:.3f} starts at best {gp['starts_at_best']}/{gp['starts']}")
    errors = {"median_vis_error": float(np.nanmedian(np.concatenate([np.ravel(np.asarray(d.d_vis, float)) for d in datasets]))),
              "median_phase_error_deg": float(np.nanmedian(np.concatenate([np.ravel(np.asarray(d.d_phi, float)) for d in datasets])))}
    errors["implied_vis_residual"] = float(np.sqrt(chosen["chi2_red"]) * errors["median_vis_error"])
    errors["implied_phase_residual_deg"] = float(np.sqrt(chosen["chi2_red"]) * errors["median_phase_error_deg"])
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rest = None
    if residual and np.isfinite(chosen["chi2"]):
        try:
            best = model_of(dict(FAMILIES[family])[chosen["rung"]], {k: v[0] for k, v in chosen["values"].items()}, res, field, wavel0)
            rest = residual_image(best, datasets, res, field, out_dir / f"{label}_rest.npz")
            print(f"{label} residual image: chi2/N {chosen['chi2_red']:.3g} -> {rest['chi2_red']:.3g}, flux {rest['flux_relative_to_model']:.3g} of the model's")
        except Exception as err:  # noqa: BLE001 the parametric result stands without it
            rest = {"error": f"{type(err).__name__}: {err}"}
    report = {"label": label, "task": task, "family": family, "knowledge": "informed: model family from the contest paper; no value",
              "files": spec["files"], "resolution_mas": res, "field_mas": field, "channels": channels,
              "chromatic": bool(wavel0), "reference_wavelength_m": wavel0, "flux_spectra_fitted": spectra, "virgil": getattr(virgil, "__version__", None),
              "quoted_errors": errors, "residual_image": rest, "gp_disk": gp, "rungs": rungs, "chosen": chosen["rung"], "elapsed_s": time.time() - t0, "smoke": smoke}
    (out_dir / f"{label}.json").write_text(json.dumps(report, indent=1) + "\n")
    agree = chosen["starts_at_best"] >= MIN_AGREE
    checks = {"converged": "pass" if chosen["converged"] else "fail", "starts_agree": "pass" if agree else "warn"}
    status = "fail" if not chosen["converged"] else ("warn" if not agree or chosen["chi2_red"] > 1.5 else "pass")
    (out_dir / "result.json").write_text(json.dumps(
        {"status": status, "checks": checks, "chosen": chosen["rung"], "chi2_red": chosen["chi2_red"],
         "starts_at_best": chosen["starts_at_best"], "knowledge": "informed"}, indent=1) + "\n")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--task", type=int)
    ap.add_argument("--model", choices=sorted(FAMILIES))
    ap.add_argument("--data", default="~/data/imaging_contests")
    ap.add_argument("--out", default="contest_informed")
    ap.add_argument("--starts", type=int, default=32, help="random starts per rung")
    ap.add_argument("--smoke", action="store_true", help="three starts of 50 steps: check that it runs")
    ap.add_argument("--grey", action="store_true", help="one flux per component at every wavelength (the default gives each a spectral index when the data have more than one channel)")
    ap.add_argument("--no-gp-disk", action="store_true", dest="no_gp", help="skip the multipolar Gaussian-process disk of the disk families")
    ap.add_argument("--no-residual-image", action="store_true", dest="no_residual", help="skip the non-parametric image fitted on top of the chosen model")
    ap.add_argument("--no-spectra", action="store_true", dest="no_spectra", help="do not fit the files' OI_FLUX spectra")
    a = ap.parse_args(argv)
    if a.list:
        for name, rungs in FAMILIES.items():
            print(name + ": " + " -> ".join(r[0] for r in rungs))
        return 0
    if a.task is None or a.model is None:
        ap.error("--task and --model are required")
    run(a.task, a.model, pathlib.Path(a.data).expanduser(), pathlib.Path(a.out).expanduser(), a.starts, a.smoke, a.grey, not a.no_spectra, not a.no_residual, a.no_gp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
