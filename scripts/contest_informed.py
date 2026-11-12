"""Informed fits of the imaging contests' data: the model family the contest paper describes, fitted from the data.

    python scripts/contest_informed.py --list
    python scripts/contest_informed.py --task N --model FAMILY [--data DIR] [--out DIR] [--starts 8] [--budget N] [--smoke]

This is **not** a blind reconstruction, and its results must never be quoted as one. It uses what a contest paper
said the object *is* (a spotted star with a companion; a thin disk; a star, a disk rim, a planet and an outer disk;
a ringed disk with companions) to choose a parametric model. It uses nothing the paper said about the *values*: no diameter, position, flux ratio or angle
from a paper is a start, a bound or a constant here. Every scale, start and bound comes from the data's own baselines
(or a file entrants were given), so the published parameters stay a test of the fit (they are compared afterwards, by
the campaign's sealed stage). The method is design/contest_informed_plan.md; the checks' thresholds are
design/contest_informed_criteria.md, whose last commit every report records.

A family is a ladder of nested models, simplest first (FAMILIES), ending in the last rung plus over-resolved flux
(`Resolved`). Per rung:

- **Scales** (plan 2): the extent E (size limit, field of every image), the reach S (how far a compact companion is
  searched for) and the pixel, each written to the report with its source. E is the contest's field where it gave
  one, else 4 x the major FWHM of the visibility-moment fit, at least 8 resolution elements.
- **Starts** (plan 4), each recorded with its origin: `moments` (the visibility-moment fit), `grid` (stratified
  geometry: sizes, inclination x angle, arc angles and lengths), `map` (peaks of a tiled likelihood map for a new
  compact component), `below` (the rung below with the new component at negligible flux) and `random` (control draws
  from the priors). The report counts the starts at the best by origin.
- **Errors** (plan 3): every start is fitted first on errors inflated by sqrt(chi2/N) of the moment fit (a smooth
  landscape), the best are then refitted with virgil's fitted noise terms (`vis_error_rel`, `phi_error`), so the
  likelihood carries its normalisation. Rungs are compared by BIC on that likelihood; chi2 on the quoted errors is
  reported beside it and never chooses. Fluxes have no cap (LogUniform(1e-4, 1e4) ratios); colours are fitted grey
  first, then the spectral indices are freed from zero in Uniform(-4, 4).
- **Checks** (plan 5) in result.json: `no_bounds`, `nested`, `starts_agree`, `family_complete`, `no_collapse`,
  `extent_known`; the mandate stage requires them.

After the ladder, a simple non-parametric image (a Gaussian-process field) is fitted on top of the chosen model,
held fixed, to show what resolved light the parametric model leaves out (<label>_<family>_rest.npz).

Spectral information is used in two ways. When the data have more than one channel, every component but the
reference gets its own spectral index, so the model is chromatic wherever the object can be. And where a file
carries OI_FLUX spectra they are fitted with the visibilities (scale and linear trend marginalised).

Writes into --out: <label>_<family>.json (the report) and result.json (status and checks for the campaign).

Task numbers and files are those of scripts/contest_images.py (TASKS).
"""

import argparse
import ast
import dataclasses
import json
import pathlib
import sys
import time

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import numpyro.distributions as dist  # noqa: E402

import virgil  # noqa: E402
import virgil.models as vm  # noqa: E402
from virgil import imaging, likelihood  # noqa: E402
from virgil.fitting import fit  # noqa: E402
from virgil.grid_fit import linear_flux_grid  # noqa: E402
from virgil.oidata import OIData  # noqa: E402
from virgil.spectra import PowerLaw  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from crosscheck import informed_starts as IS  # noqa: E402
from crosscheck import preregistration  # noqa: E402

CRITERIA = "design/contest_informed_criteria.md"
PEAKS = 6  # likelihood-map peaks refined per new component
REFIT = 12  # at most this many starts per rung are refitted at fitted noise (the best by tempered chi2)
REFIT_BAND = 1.5  # ... and only those whose tempered chi2 is within this factor of the best
TIME_LIMIT = 300.0  # seconds per start; a start that runs out is recorded as not converged
RANDOM_STARTS = 8  # control draws from the priors per rung (plan 4.5)
NEGLIGIBLE = 1e-3  # flux ratio of a component added at negligible flux (the prior reaches 1e-4)
NOISE = {"vis_error_rel": dist.LogUniform(1e-4, 3.0), "phi_error": dist.LogUniform(1e-4, 3.0)}  # plan 3; phi in radians
VIS_SCALE = ("2004_data2",)  # datasets normalised without part of their flux (scripts/score_2004.py): V^2 above 1 at zero baseline
SIZE_LEAVES = ("diam", "fwhm", "sigma", "radius", "r_out")  # the principal sizes: scanned by the stratified starts
RING_LEAVES = ("diam", "radius", "r_out")  # a component with one of these and a width leaf is a ring: the width is a width


def tasks():
    """TASKS of contest_images.py, read without importing it (that module sets up a whole imaging run)."""
    tree = ast.parse((REPO / "scripts" / "contest_images.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "TASKS" for t in node.targets):
            return eval(compile(ast.Expression(node.value), "contest_images.py", "eval"), {"dict": dict})
    raise KeyError("TASKS")


# ---------------------------------------------------------------------------------------------------------------
# Data and scales (plan section 2 and 4.1)


def v2_table(datasets):
    """Every dataset's V^2 as flat arrays: u, v in cycles per radian, V^2 and its error (from |V| and its error for
    amplitude data). Ported from contest_images.v2_table of the branch contest-round3-doctor."""
    cols = []
    for d in datasets:
        u, v = np.ravel(np.asarray(d.u, float)), np.ravel(np.asarray(d.v, float))
        wavel = np.ravel(np.asarray(d.wavel, float))
        wavel = np.broadcast_to(wavel, u.shape) if wavel.size in (1, u.size) else wavel[: u.size]
        vis, err = np.ravel(np.asarray(d.vis, float)), np.ravel(np.asarray(d.d_vis, float))
        if vis.size == 0:
            continue
        index = np.arange(vis.size) if d.vis_index is None else np.ravel(np.asarray(d.vis_index))
        if not d.v2_flag:
            vis, err = vis**2, 2 * np.abs(vis) * err
        cols.append((u[index] / wavel[index], v[index] / wavel[index], vis, err))
    if not cols:
        return tuple(np.zeros(0) for _ in range(4))
    return tuple(np.concatenate(c) for c in zip(*cols))


def baseline_limits(datasets):
    """(resolution, fringe) in mas: lambda / 2 B_max and lambda / B_min over the unflagged samples, from virgil's documented helpers."""
    resolution = min(float(imaging.nyquist_pixel_scale(d)) for d in datasets)
    fringe = min(float(imaging.field_of_view(d, largest_mas=1e9)) for d in datasets)
    return resolution, fringe


def divided(data, factor):
    """A copy of ``data`` with its visibilities and their errors divided by ``factor`` (virgil has no visibility-scale
    parameter; the data are normalised without part of their flux, and this is the same as multiplying the model)."""
    return eqx.tree_at(lambda d: (d.vis, d.d_vis), data, (data.vis / factor, data.d_vis / factor))


def measure(datasets, spec, scale_vis=False):
    """The visibility-moment fit and the three scales of a task: ``(Scales, moments, datasets)``. When the contest
    gave a field (``spec["field"]``) it is E; a tapered field (``spec["prior"]``) also halves the reach. With
    ``scale_vis`` the moment fit also has a free amplitude on V^2; one above 1 means the data were normalised without
    part of their flux, and the data are divided by it (named in the report, for those datasets only)."""
    resolution, fringe = baseline_limits(datasets)
    u, v, v2, err = v2_table(datasets)
    theta = (resolution / 5.0, 2.0 * fringe)
    vis_scale = None
    moments = IS.moment_fit(u, v, v2, err, theta, fit_amplitude=scale_vis)
    if scale_vis and moments is not None:
        a = moments["amplitude"]
        vis_scale = {"amplitude_v2": a, "applied": bool(a > 1.02), "why": "V^2 of the moment fit at zero baseline exceeds 1: "
                     "the data were normalised without part of the flux (scripts/score_2004.py)"}
        if vis_scale["applied"]:
            datasets = [divided(d, a if d.v2_flag else float(np.sqrt(a))) for d in datasets]
            moments = IS.moment_fit(*v2_table(datasets), theta)
    b_max, b_min = 1.0 / (2.0 * resolution * IS.MAS), 1.0 / (fringe * IS.MAS)
    field = spec.get("field")
    sc = IS.scales_of(b_max, b_min, moments, field=field, field_source=spec.get("field_source"),
                      tapered=bool(spec.get("prior") and spec["prior"][0] == "gauss"))
    return sc, moments, datasets, vis_scale


def temper_factor(moments):
    """sqrt(chi2/N) of the moment fit on the quoted errors, at least 1: the factor by which the first pass inflates the errors."""
    if moments is None:
        return 1.0
    return float(max(1.0, np.sqrt(moments["chi2_quoted"] / max(moments["n"], 1))))


# ---------------------------------------------------------------------------------------------------------------
# Families. Each family is a ladder of rungs, simplest first: (name, {component: (maker, {parameter path: prior kind})}).
# A maker is a virgil class, or "RingedDisk" (make_component). Prior kinds are scale-free words that `prior_of`
# turns into distributions from the data's scales; no number here comes from a paper. The first component of a rung
# is the flux reference (flux 1); every other component's flux is relative to it.
POINT = ("PointSource", {})
SMALL_STAR = ("UniformDisk", {"diam": "small"})
LD_STAR = ("LimbDarkenedDisk", {"diam": "size", "u": "limb"})
ELLIPTICAL_STAR = ("EllipticalLimbDarkenedDisk", {"diam": "size", "ratio": "ratio", "pa": "angle", "u": "limb"})
COMPANION = ("PointSource", {"flux": "planet", "dra": "far", "ddec": "far"})
SMALL_DISK = ("UniformDisk", {"diam": "small", "flux": "flux", "dra": "far", "ddec": "far"})
SPOT = ("UniformDisk", {"diam": "small", "flux": "flux", "dra": "near", "ddec": "near"})
SURFACE_SPOT = ("UniformDisk", {"diam": "small", "flux": "flux", "dra": "far", "ddec": "far"})
HALO = ("GaussianDisk", {"sigma": "size", "flux": "flux"})
ELLIPSE = ("EllipticalGaussian", {"fwhm": "size", "ratio": "ratio", "pa": "angle", "flux": "flux"})
OFFSET_ELLIPSE = ("EllipticalGaussian", {"fwhm": "size", "ratio": "ratio", "pa": "angle", "flux": "flux", "dra": "near", "ddec": "near"})
RIM = ("ModulatedGaussianRim", {"diam": "size", "fwhm": "width", "inc": "inclination", "pa": "angle",
                                "az_amps": "modulation", "az_pas": "full_angle", "flux": "flux"})
ARC = ("GaussianArc", {"radius": "size", "width": "size", "length": "size", "pa": "full_angle", "flux": "flux"})
RESOLVED = ("Resolved", {"flux": "flux"})  # over-resolved light: visibility zero on every baseline


def rings(*names):
    """An inclined disk of Gaussian rings; each ring after the first has a weight relative to the first."""
    params = {"inc": "inclination", "pa": "angle", "flux": "flux"}
    for i, name in enumerate(names):
        params[f"profile.{name}.radius"] = "size"
        params[f"profile.{name}.sigma"] = "ring_width"
        if i:
            params[f"profile.{name}.weight"] = "flux"
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

    def with_resolved(self):
        """The same scene with over-resolved flux added (a System of this one and a `Resolved`)."""
        inner = self

        def make(p, wavel0=None):
            return vm.System(base=inner.make(p, wavel0), diffuse=vm.Resolved(flux=flux_of(p, "diffuse", wavel0)))

        return Scene(self._kinds | {"diffuse.flux": "flux"}, make, (*self.components, "diffuse"), self.positions)


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
        kinds |= {"envelope.sigma": "size", "envelope.flux": "flux"}
        components.append("envelope")
    if companion:
        kinds |= {"companion.flux": "planet", "companion.dra": "far", "companion.ddec": "far"}
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
        kinds["disk.flux"] = "flux"

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


def ladder(family):
    """The rungs of a family: its own, then the last one with over-resolved flux added (plan 3: every family gets an
    optional `Resolved` component, in a rung of its own)."""
    rungs = list(FAMILIES[family])
    name, last = rungs[-1]
    more = last.with_resolved() if isinstance(last, Scene) else {**last, "diffuse": RESOLVED}
    return rungs + [(name + "+resolved", more)]


def prior_of(kind, sc):
    """The prior of a kind of parameter, from the scales ``sc`` (`crosscheck.informed_starts.Scales`)."""
    res, extent, reach = sc.resolution, sc.extent, sc.reach
    return {
        "size": dist.LogUniform(res / 5.0, extent),  # anything from well under the resolution to the whole extent
        "small": dist.LogUniform(res / 20.0, 4.0 * res),  # barely resolved or unresolved
        # a ring's width (FWHM) may be well under the resolution; the floor sits below `no_collapse`'s RING_FRACTION, so a
        # ring that has collapsed is inside the prior, not at its edge
        "width": dist.LogUniform(IS.WIDTH_FLOOR * res, extent),
        "ring_width": dist.LogUniform(IS.WIDTH_FLOOR * res / IS.FWHM_PER_SIGMA, extent / IS.FWHM_PER_SIGMA),  # the same, as a sigma
        "ratio": dist.Uniform(0.3, 1.0),
        "angle": dist.Uniform(0.0, 180.0),
        "full_angle": dist.Uniform(0.0, 360.0),
        "inclination": dist.Uniform(0.0, 89.0),  # edge-on is allowed
        "limb": dist.Uniform(0.0, 1.0),
        "harmonic": dist.Normal(0.0, 0.3),  # a spherical-harmonic coefficient of the surface, relative to the mean
        "wave": dist.Uniform(-0.7, 0.7),  # an azimuthal harmonic of a disk's brightness, relative to its mean
        "fraction": dist.Uniform(0.02, 0.95),
        "slope": dist.Uniform(-3.0, 1.0),
        "modulation": dist.Uniform(0.0, 1.0),
        # A component's spectrum relative to the reference component: flux ratio x (lambda / lambda_0) ** index.
        "colour": dist.Uniform(-4.0, 4.0),
        "flux": dist.LogUniform(1e-4, 1e4),  # relative to the reference component: no cap
        "planet": dist.LogUniform(1e-4, 0.1),  # a planet or faint companion: at most a tenth of the reference
        "far": dist.Uniform(-reach, reach),  # a compact companion is searched for out to the reach
        "near": dist.Uniform(-extent / 2.0, extent / 2.0),  # a part of the object: within half the extent
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


def flat_kinds(rung, wavel0=None):
    """{parameter path: prior kind} of a rung (a template dict, or a `Scene`), chromatic if ``wavel0``."""
    if isinstance(rung, Scene):
        return rung.kinds(wavel0)
    rung = chromatic(rung) if wavel0 else rung
    return {f"{name}.{p}": kind for name, (_, params) in rung.items() for p, kind in params.items()}


def build_priors(rung, sc, wavel0=None):
    return {path: prior_of(kind, sc) for path, kind in flat_kinds(rung, wavel0).items()}


def model_of(rung, values, wavel0=None):
    """The virgil model of a rung (a template dict or a `Scene`) at ``{parameter: value}``, chromatic if ``wavel0``."""
    if isinstance(rung, Scene):
        return rung.make(values, wavel0)
    rung = chromatic(rung) if wavel0 else rung
    parts = {}
    for name, (maker, params) in rung.items():
        kwargs = {}
        for p in params:
            leaf = p.rsplit(".", 1)[-1]
            value = (values or {}).get(f"{name}.{p}", 0.0 if p == "flux.index" else PLACEHOLDER["flux" if p == "flux.ratio" else leaf])
            kwargs[p] = np.array([value]) if leaf in VECTORS else value
        parts[name] = make_component(maker, kwargs, wavel0)
    return vm.System(**parts)


def components_of(rung):
    return tuple(rung.components if isinstance(rung, Scene) else rung)


def contains(rich, poor):
    """Whether ``rich`` contains ``poor``: every component of ``poor`` is in ``rich`` (same maker, every parameter),
    or, for a `Scene`, every parameter. A richer rung can then start from the one below with the new part at
    negligible flux, and cannot fit worse (the `nested` check)."""
    if isinstance(rich, Scene) != isinstance(poor, Scene):
        return False
    if isinstance(rich, Scene):
        return set(poor.kinds()) <= set(rich.kinds())
    return all(name in rich and rich[name][0] == maker and set(params) <= set(rich[name][1]) for name, (maker, params) in poor.items())


# ---------------------------------------------------------------------------------------------------------------
# Starts (plan 4)


def leaf_of(path):
    return path.rsplit(".", 1)[-1]


def comp_of(path):
    return path.split(".", 1)[0]


def draw(priors, kinds, sc, rng):
    """One random start: each parameter drawn from its own prior (a planet is redrawn until it is two resolution
    elements from the star)."""
    for _ in range(100):
        out = {}
        for path, p in priors.items():
            if isinstance(p, dist.Normal):
                out[path] = float(rng.normal(float(p.loc), float(p.scale)))
            elif isinstance(p, dist.LogUniform):
                out[path] = float(np.exp(rng.uniform(np.log(float(p.low)), np.log(float(p.high)))))
            else:
                out[path] = float(rng.uniform(float(p.low), float(p.high)))
        if all(np.hypot(out[f"{c}.dra"], out[f"{c}.ddec"]) >= IS.COMPANION_ELEMENTS * sc.resolution
               for c in {comp_of(p) for p, k in kinds.items() if leaf_of(p) == "dra" and k == "far"} if f"{c}.dra" in out
               and kinds.get(f"{c}.flux", kinds.get(f"{c}.flux.ratio")) == "planet"):
            break
    return out


def bounds_of(priors, kinds):
    """{path: (low, high, log, kind)} of the priors: what `no_bounds` checks the fitted values against."""
    out = {}
    for path, p in priors.items():
        if isinstance(p, dist.LogUniform):
            out[path] = (float(p.low), float(p.high), True, kinds[path])
        elif isinstance(p, dist.Uniform):
            out[path] = (float(p.low), float(p.high), False, kinds[path])
        else:
            out[path] = (-np.inf, np.inf, False, kinds[path])
    return out


def centre_of(rung, kinds, priors, sc, m):
    """The start at the visibility-moment values (plan 4.1): the extended light's major FWHM, axis ratio and angle give
    each size, ratio, angle and (as arccos of the ratio) inclination; the compact and over-resolved fractions give the
    fluxes. Parameters the moments say nothing about start in the middle of their priors, a surface or modulation
    from zero. Every value is moved inside its prior."""
    bounds = bounds_of(priors, kinds)
    theta = m["fwhm"] if m else float(np.sqrt(sc.resolution * sc.extent))
    ratio0 = float(np.clip(m["ratio"], 0.31, 0.99)) if m else 0.7
    pa0 = m["pa"] if m else 90.0
    inc0 = float(np.degrees(np.arccos(ratio0)))
    compact = max(m["compact"], 0.02) if m else 1.0
    ext_ratio = max((m["extended"] if m else 0.5) / compact, 1e-2)
    res_ratio = max((m["resolved"] if m else 0.1) / compact, NEGLIGIBLE)
    leaves = {}
    for path in kinds:
        leaves.setdefault(comp_of(path), []).append(leaf_of(path))
    extended = [c for c in leaves if any(leaf_of(p) in SIZE_LEAVES and k == "size" for p, k in kinds.items() if comp_of(p) == c)
                and not any(k == "planet" for p, k in kinds.items() if comp_of(p) == c)]
    makers = {c: (rung[c][0] if isinstance(rung, dict) else None) for c in leaves}
    out = {}
    for path, kind in kinds.items():
        comp, leaf = comp_of(path), leaf_of(path)
        ring = any(l in RING_LEAVES for l in leaves[comp])
        if kind in ("size", "small", "width", "ring_width"):
            if "profile" in path:  # a ring of a disk: its radius and width
                index = 0.5 * (1.0 + (ord(path.split(".")[-2]) - ord("a")) * 0.5) if leaf == "radius" else 0.0
                value = theta * index if leaf == "radius" else max(sc.resolution, theta / 8.0)
            elif leaf in ("width", "fwhm", "sigma") and ring and leaf != "diam":
                value = max(sc.resolution, theta / 6.0)
            elif leaf == "fwhm":
                value = theta
            elif leaf == "sigma":
                value = theta / IS.FWHM_PER_SIGMA
            elif leaf == "radius" and makers.get(comp) == "GaussianArc":
                value = theta
            elif leaf == "radius":
                value = theta / 2.0
            elif kind == "small":
                value = float(np.sqrt(float(priors[path].low) * float(priors[path].high)))
            else:
                value = theta
        elif leaf == "ratio":
            value = ratio0
        elif leaf in ("pa", "az_pas") and kind in ("angle", "full_angle"):
            value = pa0 if leaf == "pa" else 0.0
        elif leaf == "inc":
            value = inc0
        elif leaf in ("az_amps", "harmonic") or kind in ("harmonic", "wave", "modulation"):
            value = 0.0
        elif kind == "flux":
            if comp == "diffuse":
                value = res_ratio
            elif comp in extended:
                value = ext_ratio / max(len(extended), 1)
            else:
                value = 1e-2
        elif kind == "planet":
            value = 3e-3
        elif kind in ("far", "near"):
            value = 0.0
        else:  # limb darkening, fractions, slopes: the middle of the prior
            lo, hi, log, _ = bounds[path]
            value = float(np.sqrt(lo * hi)) if log else 0.5 * (lo + hi)
        lo, hi, log, _ = bounds[path]
        out[path] = IS.inside(value, lo, hi, log) if np.isfinite(lo) and np.isfinite(hi) else float(value)
    return out


def stratified_starts(rung, kinds, priors, sc, m, centre):
    """Stratified geometry starts (plan 4.2): sizes 0.5 to 2 x the moment scale in 8 steps, one component at a time;
    inclination x angle (the moment values, then a 4 x 6 grid) for every inclined or elliptical component together;
    for thin, long parts (an arc) the angle in 12 steps and the length in three. Everything else stays at ``centre``."""
    bounds = bounds_of(priors, kinds)
    out = []

    def make(changes):
        s = dict(centre)
        for path, value in changes.items():
            lo, hi, log, _ = bounds[path]
            s[path] = IS.inside(value, lo, hi, log)
        return s

    leaves = {}
    for path in kinds:
        leaves.setdefault(comp_of(path), {})[leaf_of(path)] = path
    for path, kind in kinds.items():  # sizes, one at a time (each ring of a disk has its own radius and width)
        if leaf_of(path) in SIZE_LEAVES and kind in ("size", "width", "ring_width"):
            for size in IS.log_steps(centre[path]):
                out.append(make({path: size}))
    inclined = [ls for ls in leaves.values() if "inc" in ls and "pa" in ls]
    elliptic = [ls for ls in leaves.values() if "inc" not in ls and "ratio" in ls and "pa" in ls and kinds[ls["pa"]] == "angle"]
    if inclined or elliptic:
        inc0, pa0 = float(np.degrees(np.arccos(np.clip(m["ratio"], 0.31, 0.99)))) if m else 45.0, (m["pa"] if m else 90.0)
        for inc, pa in IS.inc_pa_grid(inc0, pa0)[1:]:
            changes = {}
            for ls in inclined:
                changes |= {ls["inc"]: inc, ls["pa"]: pa}
            for ls in elliptic:
                changes |= {ls["ratio"]: max(0.31, float(np.cos(np.radians(inc)))), ls["pa"]: pa}
            out.append(make(changes))
    for ls in leaves.values():
        if "length" in ls and "pa" in ls:  # a thin, long part
            for pa in IS.angle_steps(12, 360.0):
                out.append(make({ls["pa"]: pa}))
            for length in np.geomspace(sc.resolution * 2.0, sc.extent, 3):
                out.append(make({ls["length"]: float(length)}))
    return out


def seeded(start, seed, priors, kinds):
    """``start`` with the values of a map ``seed`` put in, each moved inside its prior: a peak on the edge of the map, a
    flux under 1e-4 or over a planet's cap must not start a fit on or outside the prior's support."""
    bounds, out = bounds_of(priors, kinds), dict(start)
    for path, value in seed.items():
        lo, hi, log, _ = bounds[path]
        out[path] = IS.inside(value, lo, hi, log) if np.isfinite(lo) and np.isfinite(hi) else float(value)
    return out


def on_peaks(start, index, peaks, priors, kinds, inherited, keep=()):
    """``start`` for a rung above the first: the parameters it inherits from the rung below (``inherited``, except those
    in ``keep``) at the rung below's values, and each new positioned component's ``dra``/``ddec`` at a map peak (the
    ``index``-th, cycling, so the starts spread over the best peaks). With no peaks for a component its position is left."""
    out = {**start, **{k: v for k, v in inherited.items() if k in priors and k not in keep}}
    for found in peaks.values():
        if found:
            seed = found[index % len(found)]
            out = seeded(out, {k: v for k, v in seed.items() if leaf_of(k) in ("dra", "ddec")}, priors, kinds)
    return out


FLUX_KINDS = ("flux", "planet")  # prior kinds of a flux (ratio to the reference component), a ring weight included


def embed_below(rung, below_rung, kinds, priors, start, wavel0=None):
    """``start`` with ``rung`` made to reproduce ``below_rung``: every flux or ring weight the rung adds starts at
    ``NEGLIGIBLE`` (a new ring, a new companion, the over-resolved light), and when the rung adds a reference component
    (one with no flux parameter: its flux is 1 and the rest are ratios to it, as when a star is added to a disk) the
    fluxes of the old components start at the top of their priors, so that the reference is the negligible part.
    New parameters are found from the parameter names, not the component names: a ring of an existing disk is new too."""
    bounds = bounds_of(priors, kinds)
    out = dict(start)

    def put(path, value):
        lo, hi, log, _ = bounds[path]
        out[path] = IS.inside(value, lo, hi, log)

    old = set(flat_kinds(below_rung, wavel0))
    for path, kind in kinds.items():
        if path not in old and kind in FLUX_KINDS:
            put(path, NEGLIGIBLE)
    old_comps = set(components_of(below_rung))
    gained = [c for c in components_of(rung) if c not in old_comps
              and not any(comp_of(p) == c and k in FLUX_KINDS for p, k in kinds.items())]
    if gained:
        for path, kind in kinds.items():
            if path in old and kind == "flux" and "profile" not in path and leaf_of(path) in ("flux", "ratio"):
                put(path, 1e4)
    return out


def thin(starts, budget, rng):
    """At most ``budget`` of the stratified starts, evenly spaced through the list (None keeps all)."""
    if budget is None or len(starts) <= budget:
        return starts
    keep = np.unique(np.round(np.linspace(0, len(starts) - 1, budget)).astype(int))
    return [starts[i] for i in keep]


def map_seeds(name, rung, tempered, sc, carry, kinds, smoke=False):
    """Peaks of a tiled likelihood map for a component the rung adds (plan 4.3): the rung below at its best fit, the
    new component's position scanned over the whole reach (or half the extent for a part of the object) at half a
    resolution element, in tiles so memory stays bounded. The score is the improvement in chi2 of a linear flux
    (virgil's `linear_flux_grid`, which solves the flux at each position; datasets are combined by inverse variance),
    on the tempered errors, so a companion carrying under a hundredth of the flux shows beside a poor surface model.
    Returns ``(seeds, record)`` with ``seeds = [{path: value}]``."""
    comp_kinds = {leaf_of(p): k for p, k in kinds.items() if comp_of(p) == name}
    half = sc.reach if comp_kinds.get("dra") == "far" else sc.extent / 2.0
    step = sc.resolution / 2.0
    if smoke:
        step = max(step, 2.0 * half / 60.0)
    exclude = IS.COMPANION_ELEMENTS * sc.resolution if comp_kinds.get("flux") == "planet" else 0.0
    if isinstance(rung, Scene):
        base = model_of(rung, carry)  # the caller passes the rung below here
        template = vm.System(base=base, **{name: vm.PointSource(flux=1e-3)})
    else:
        template = model_of(rung, carry)
    flux_path = f"{name}.flux"

    def evaluate(dra, ddec):
        num, den = 0.0, 0.0
        for d in tempered:
            g = linear_flux_grid(template, d, {f"{name}.dra": dra, f"{name}.ddec": ddec, flux_path: np.array([1e-3])})
            w = 1.0 / np.asarray(g.flux_error) ** 2  # one axis per coordinate; the flux is solved for, not scanned
            num, den = num + np.nan_to_num(np.asarray(g.flux) * w), den + np.nan_to_num(w)
        flux = num / np.where(den > 0, den, np.inf)
        return np.where(flux > 0, np.nan_to_num(num**2 / np.where(den > 0, den, np.inf)), 0.0)

    axis, surface, actual = IS.tiled_map(evaluate, half, step)
    peaks = IS.map_peaks(axis, surface, PEAKS, 2.0 * sc.resolution, exclude)
    seeds = []
    for dra, ddec, score in peaks:
        # the flux at the peak: the map's own linear solution there
        num = den = 0.0
        for d in tempered:
            g = linear_flux_grid(template, d, {f"{name}.dra": np.array([dra]), f"{name}.ddec": np.array([ddec]), flux_path: np.array([1e-3])})
            w = 1.0 / np.ravel(np.asarray(g.flux_error))[0] ** 2
            num, den = num + np.nan_to_num(np.ravel(np.asarray(g.flux))[0] * w), den + np.nan_to_num(w)
        flux = float(num / den) if den > 0 else 1e-3
        seeds.append({f"{name}.dra": dra, f"{name}.ddec": ddec, flux_path: max(flux, 1e-4)})
    return seeds, {"component": name, "half_mas": float(half), "step_mas": float(actual), "requested_step_mas": float(step),
                   "points": int(axis.size) ** 2, "peaks": [(p[0], p[1], p[2]) for p in peaks]}


# ---------------------------------------------------------------------------------------------------------------
# Fitting (plan 3)


def noise_priors(datasets):
    """The noise terms fitted with the parameters: `vis_error_rel` and `phi_error` for the observables a dataset has,
    one set per dataset when there are several."""
    sets = []
    for d in datasets:
        terms = {}
        if np.size(d.vis):
            terms["vis_error_rel"] = NOISE["vis_error_rel"]
        if np.size(d.phi):
            terms["phi_error"] = NOISE["phi_error"]
        sets.append(terms)
    return sets[0] if len(datasets) == 1 else sets


def noise_names(datasets):
    """The keys of the fitted noise terms in ``FitResult.values``, with the dataset each belongs to."""
    sets = noise_priors(datasets)
    if len(datasets) == 1:
        return {f"noise.{t}": (0, t) for t in sets}
    return {f"noise[{i}].{t}": (i, t) for i, terms in enumerate(sets) for t in terms}


def as_data(datasets):
    return datasets[0] if len(datasets) == 1 else datasets


def loglike_total(model, datasets, noise_values):
    """ln L of ``model`` on every dataset, with the fitted noise terms (``{noise key: value}``): virgil's normalised
    Gaussian likelihood, so it carries sum ln sigma and rungs can be compared by BIC."""
    names = noise_names(datasets)
    total = 0.0
    for i, d in enumerate(datasets):
        kwargs = {t: float(noise_values[k]) for k, (j, t) in names.items() if j == i and k in noise_values}
        total += float(likelihood.model_loglike(model, d, **kwargs))
    return total


def chi2_quoted(model, datasets):
    """chi2 of ``model`` on the quoted errors, and the number of independent data (reported beside the likelihood)."""
    chi2 = sum(float(np.sum(np.asarray(likelihood.whitened_residuals(model, d)) ** 2)) for d in datasets)
    return chi2, int(sum(int(d.n_independent) for d in datasets))


def shape_init(values, rung):
    """Starting values as `fit` wants them: arrays for the one-coefficient vectors of a template rung."""
    out = {}
    for k, v in values.items():
        v = float(np.ravel(v)[0])
        out[k] = np.array([v]) if leaf_of(k) in VECTORS and not isinstance(rung, Scene) else v
    return out


def target_of(rung, wavel0=None):
    """What `fit` takes as a model: a template System for a template rung, a function of the parameters for a `Scene`."""
    if isinstance(rung, Scene):
        return lambda **p: rung.make(p, wavel0)
    return model_of(rung, {}, wavel0)


def fit_start(rung, priors, data, init, wavel0=None, noise=None, max_steps=None):
    """One fit: ``(FitResult, None)`` or ``(None, error text)`` (a start that fails is recorded, not hidden)."""
    try:
        return fit(target_of(rung, wavel0), priors, data, noise=noise, init=shape_init(init, rung), max_steps=max_steps, time_limit=TIME_LIMIT), None
    except Exception as err:  # noqa: BLE001
        return None, f"{type(err).__name__}: {err}"


def scalars(values):
    return {k: float(np.ravel(v)[0]) for k, v in values.items() if np.size(v) == 1}


def final_record(result, rung, datasets, wavel0, sc):
    """Everything about one refit at fitted noise: -2 ln L, chi2 on the quoted errors, noise values, values."""
    values = scalars(result.values)
    noise_values = {k: v for k, v in values.items() if k.startswith("noise")}
    model = result.model[0] if isinstance(result.model, (list, tuple)) else result.model
    ll = loglike_total(model, datasets, noise_values)
    chi2, n = chi2_quoted(model, datasets)
    return {"minus2lnl": -2.0 * ll if np.isfinite(ll) else float("inf"), "chi2_quoted": chi2, "n": n, "noise": noise_values,
            "values": {k: v for k, v in values.items() if not k.startswith("noise")},
            "converged": bool(result.info.get("converged")), "at_bound": list(result.info.get("at_bound") or [])}


def fit_rung(name, rung, below_rung, ctx, carry, rng):
    """One rung, grey: the starts of plan 4, a tempered first pass, and the best refitted at fitted noise."""
    datasets, tempered, sc, m = ctx["datasets"], ctx["tempered"], ctx["scales"], ctx["moments"]
    kinds, priors = flat_kinds(rung), build_priors(rung, sc)
    data, tdata = as_data(datasets), as_data(tempered)
    centre = centre_of(rung, kinds, priors, sc, m)
    maps = []
    new = [c for c in components_of(rung) if not below_rung or c not in components_of(below_rung)]
    positioned = [c for c in new if (c in rung.positions if isinstance(rung, Scene) else "dra" in rung[c][1])]
    # The likelihood map comes first: on a rung above the first, a new component's position is never the star's or a random
    # draw, so every start places it at a map peak and takes the parameters it inherits from the rung below.
    map_starts, peaks = [], {}
    for comp in positioned:
        if carry and below_rung is not None:
            seeds, record = map_seeds(comp, below_rung if isinstance(rung, Scene) else rung, tempered, sc, carry, kinds, ctx["smoke"])
            maps.append(record)
            peaks[comp] = seeds[:2 if ctx["smoke"] else PEAKS]
            for seed in peaks[comp]:
                map_starts.append(("map", seeded({**centre, **{k: v for k, v in carry.items() if k in priors}}, seed, priors, kinds)))
    inherited = carry if below_rung is not None else {}
    starts = [("moments", on_peaks(centre, 0, peaks, priors, kinds, inherited))]
    strata = []
    for i, s in enumerate(thin(stratified_starts(rung, kinds, priors, sc, m, centre), ctx["budget"], rng)):
        stratified = {k for k in s if s[k] != centre[k]}  # what the stratum varies is not overwritten by the rung below
        strata.append(("grid", on_peaks(s, i, peaks, priors, kinds, inherited, keep=stratified)))
    if carry and below_rung is not None:
        # the rung below, new parts at negligible flux (a part that is positioned goes where the map's best peak is)
        below = embed_below(rung, below_rung, kinds, priors, {**centre, **{k: v for k, v in carry.items() if k in priors}})
        if map_starts:
            below.update({k: v for k, v in map_starts[0][1].items() if leaf_of(k) in ("dra", "ddec")})
        starts.append(("below", below))
    randoms = [("random", on_peaks(draw(priors, kinds, sc, rng), i, peaks, priors, kinds, inherited)) for i in range(ctx["random"])]
    starts = starts + strata + map_starts + randoms
    # tempered first pass
    runs = []
    for origin, init in starts:
        r, error = fit_start(rung, priors, tdata, init, max_steps=ctx["max_steps"])
        if r is None:
            runs.append({"origin": origin, "error": error, "chi2_tempered": float("inf")})
            continue
        chi2 = float(np.sum(r.info["chi2"]))
        runs.append({"origin": origin, "chi2_tempered": chi2 if np.isfinite(chi2) else float("inf"), "values": scalars(r.values),
                     "converged": bool(r.info.get("converged"))})
    ok = [x for x in runs if "values" in x]
    # refit the best at fitted noise
    best_t = min((x["chi2_tempered"] for x in ok), default=float("inf"))
    chosen = sorted((x for x in ok if x["chi2_tempered"] <= REFIT_BAND * best_t), key=lambda x: x["chi2_tempered"])[: ctx["refit"]]
    # the start from the rung below is always refitted (it is what makes a richer rung no worse than the one it contains)
    chosen += [x for x in ok if x["origin"] == "below" and x not in chosen]
    noise = noise_priors(datasets)
    # fitted noise starts where the rung below left it (0.1 for the first rung)
    start_noise = {k: float((ctx.get("noise0") or {}).get(k, 0.1)) for k in noise_names(datasets)}
    refits = []
    for x in chosen:
        r, error = fit_start(rung, priors, data, {**x["values"], **start_noise}, noise=noise, max_steps=ctx["max_steps"])
        if r is None:
            x["error"] = error
            continue
        rec = final_record(r, rung, datasets, None, sc)
        x["refit"] = rec
        refits.append(x)
    if not refits:
        raise RuntimeError(f"{name}: no start survived: {sorted({x['error'] for x in runs if 'error' in x})[:3]}")
    best = min(refits, key=lambda x: x["refit"]["minus2lnl"])
    final = best["refit"]
    tol = IS.tolerance(final["n"])
    at_best = [x for x in refits if x["refit"]["minus2lnl"] <= final["minus2lnl"] + tol]
    k = len(priors) + len(noise_names(datasets))
    origins = sorted({x["origin"] for x in runs})
    return {"rung": name, "k": k, "n": final["n"], "minus2lnl": final["minus2lnl"], "bic": final["minus2lnl"] + k * float(np.log(max(final["n"], 1))),
            "chi2_quoted": final["chi2_quoted"], "chi2_red": final["chi2_quoted"] / max(final["n"], 1), "noise": final["noise"],
            "converged": final["converged"], "at_bound": final["at_bound"], "values": final["values"],
            "starts": len(runs), "starts_refit": len(refits), "starts_at_best": len(at_best),
            "failed_starts": sum("error" in x for x in runs), "errors": sorted({x["error"] for x in runs if "error" in x})[:3],
            "starts_by_origin": {o: sum(x["origin"] == o for x in runs) for o in origins},
            "starts_at_best_by_origin": {o: sum(x["origin"] == o for x in at_best) for o in origins},
            "only_seeded_starts_reach_the_best": bool(at_best) and not any(x["origin"] == "random" for x in at_best),
            "minus2lnl_of_refit_starts": [x["refit"]["minus2lnl"] for x in refits], "maps": maps,
            "bounds": {k_: list(v) for k_, v in bounds_of(priors, kinds).items()}}


def chromatic_rung(grey, rung, below_rung, below_final, ctx, wavel0):
    """The rung with spectral indices freed (plan 3): from the grey best with every index at zero, and from the rung
    below's chromatic fit with the new parts at negligible flux, so the rung cannot fit worse than the one below."""
    datasets, sc = ctx["datasets"], ctx["scales"]
    kinds, priors = flat_kinds(rung, wavel0), build_priors(rung, sc, wavel0)
    data, noise = as_data(datasets), noise_priors(datasets)

    def from_grey(values, fluxes=None, index=0.0):
        out = {}
        for path in priors:
            if path.endswith(".flux.ratio"):
                base = path[: -len(".ratio")]
                out[path] = (fluxes or {}).get(base, values.get(base, 0.1))
            elif path.endswith(".flux.index"):
                out[path] = index
            else:
                out[path] = values.get(path, 0.0)
        return out

    # Starts of the chromatic stage, each with an origin: the grey best with every index at zero, the rung below's
    # chromatic fit with the new parts at negligible flux, and the grey best with every index at +1 and at -1 (so the
    # stage has more than one start, and `starts_agree` is a test of the chromatic optimum itself).
    # Every start's noise terms are where its own source left them: the grey fit's, or the rung below's for "below".
    starts = [("grey_best", from_grey(grey["values"]), grey["noise"])]
    if below_final is not None:
        old = set(flat_kinds(below_rung, wavel0))
        init = embed_below(rung, below_rung, kinds, priors, from_grey(grey["values"]), wavel0)
        init.update({k: v for k, v in below_final["values"].items() if k in priors and k in old})
        starts.append(("below", init, below_final["noise"]))
    starts += [("index+1", from_grey(grey["values"], index=1.0), grey["noise"]),
               ("index-1", from_grey(grey["values"], index=-1.0), grey["noise"])]
    recs = []
    for origin, init, start_noise in starts:
        r, error = fit_start(rung, priors, data, {**init, **start_noise}, wavel0, noise, ctx["max_steps"])
        if r is None:
            recs.append({"origin": origin, "error": error})
            continue
        recs.append({"origin": origin, **final_record(r, rung, datasets, wavel0, sc)})
    done = [x for x in recs if "minus2lnl" in x]
    if not done:
        return None
    best = dict(min(done, key=lambda x: x["minus2lnl"]))
    best.pop("origin")
    tol = IS.tolerance(best["n"])
    at_best = [x for x in done if x["minus2lnl"] <= best["minus2lnl"] + tol]
    origins = sorted({x["origin"] for x in recs})
    k = len(priors) + len(noise_names(datasets))
    best |= {"k": k, "bic": best["minus2lnl"] + k * float(np.log(max(best["n"], 1))), "chi2_red": best["chi2_quoted"] / max(best["n"], 1),
             "bounds": {k_: list(v) for k_, v in bounds_of(priors, kinds).items()},
             # the chromatic stage's own start statistics (the checks and the report use these for a chromatic rung)
             "starts": len(recs), "starts_refit": len(done), "starts_at_best": len(at_best),
             "failed_starts": len(recs) - len(done), "errors": sorted({x["error"] for x in recs if "error" in x})[:3],
             "starts_by_origin": {o: sum(x["origin"] == o for x in recs) for o in origins},
             "starts_at_best_by_origin": {o: sum(x["origin"] == o for x in at_best) for o in origins},
             "minus2lnl_of_refit_starts": [x["minus2lnl"] for x in done]}
    return best


def as_chromatic(grey, rung, sc, wavel0):
    """The grey fit of a rung in the chromatic form (``<comp>.flux`` becomes ``<comp>.flux.ratio`` with index 0, and the
    bounds are the chromatic ones), for when the chromatic stage failed everywhere: the model, the images and the
    checks of a chromatic run all read chromatic parameters. The grey likelihood, BIC and parameter count stay, and
    ``chromatic_failed`` says so."""
    chromatic_kinds = flat_kinds(rung, wavel0)
    values = {}
    for path, value in grey["values"].items():
        if f"{path}.ratio" in chromatic_kinds:
            values[f"{path}.ratio"], values[f"{path}.index"] = value, 0.0
        else:
            values[path] = value
    bounds = {k: list(v) for k, v in bounds_of(build_priors(rung, sc, wavel0), chromatic_kinds).items()}
    return {"values": values, "bounds": bounds, "chromatic_failed": True}


# ---------------------------------------------------------------------------------------------------------------
# Non-parametric images on top of, or instead of, the ladder


# Families whose object is a disk also get a non-parametric disk: (azimuthal degree, name of a point companion or None).
GP_DISKS = {"thin_disk": (2, None), "star_in_shell": (2, None), "star_rim_planet_disk": (1, "planet"),
            "ringed_disk_with_companions": (1, "companion")}


def fixed_noise(noise_values, datasets):
    """The fitted noise of a rung held fixed (virgil's tied terms: a function of the values returns the constant), in
    the shape `noise_priors` has: one dict for one dataset, a list with one dict per dataset for several, each holding
    only the terms that dataset has."""
    sets = [{} for _ in datasets]
    for key, (i, term) in noise_names(datasets).items():
        if key in noise_values:
            sets[i][term] = (lambda v, c=float(noise_values[key]): c)
    return sets[0] if len(datasets) == 1 else sets


def gp_disk(family, chosen, ctx, wavel0, rng, starts, max_steps, out_path):
    """A multipolar Gaussian-process disk in place of the parametric one: a central star and one inclined
    `MultipoleDisk` whose radial profile is a Gaussian process in radius and whose azimuthal harmonics (to the
    family's degree) are Gaussian processes in radius too, so rings, gaps and lopsidedness need not be named in
    advance. A point companion the ladder found keeps its parametric form and starts where the ladder put it.
    The hyperparameters are fixed, not fitted: the profile varies by e^2 over three resolution elements between
    half a resolution element and the edge of the extent. It has too many latent parameters for BIC to mean
    anything, so it is reported beside the ladder and never chosen by it."""
    from virgil import gp_priors

    datasets, sc = ctx["datasets"], ctx["scales"]
    res = sc.resolution
    degree, companion = GP_DISKS[family]
    r_in, r_out = res / 2.0, sc.extent / 2.0
    ell = 3.0 * res
    values = chosen["values"]

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
    priors = gp_priors(template) | {flux_path: prior_of("flux", sc), "disk.inc": prior_of("inclination", sc), "disk.pa": prior_of("angle", sc)}
    if wavel0:
        priors["disk.flux.index"] = prior_of("colour", sc)
    if companion and f"{companion}.dra" in values:
        priors |= {f"{companion}.dra": prior_of("far", sc), f"{companion}.ddec": prior_of("far", sc),
                   (f"{companion}.flux.ratio" if wavel0 else f"{companion}.flux"): prior_of("planet", sc)}
    data = as_data(datasets)
    runs = []
    for _ in range(max(2, starts // 2)):
        start = scene(float(rng.uniform(0.0, 70.0)), float(rng.uniform(0.0, 180.0)), float(np.exp(rng.uniform(np.log(0.05), np.log(3.0)))))
        try:
            r = fit(start, priors, data, noise=fixed_noise(chosen["noise"], datasets), max_steps=max_steps, time_limit=2.0 * TIME_LIMIT)
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
    return {"chi2_red": chi2 / max(n, 1), "converged": bool(best.info.get("converged")), "azimuthal_degree": degree,
            "starts": len(runs), "starts_at_best": int(sum(c <= IS.tolerance(n) + chi2 for c, _, _ in runs)),
            "r_in_mas": r_in, "r_out_mas": r_out, "length_mas": ell, "values": scalars(best.values), "file": pathlib.Path(out_path).name}


def residual_image(model, ctx, noise_values, out_path):
    """What resolved light is left once the parametric model is taken out: a simple non-parametric image fitted on
    top of the chosen model, which is held fixed. The image is a Gaussian-process field (virgil's documented image
    model) on the pixel rule of plan 2 (lambda / 4 B_max over the extent E; coarsened, with a warning, beyond 128
    pixels a side), with a correlation length of two resolution elements; only it and its share of the flux are fitted,
    at the rung's fitted noise held fixed. Returns its numbers and saves the image."""
    from virgil.fields import GaussianField
    from virgil.imaging import image_priors

    sc, datasets = ctx["scales"], ctx["datasets"]
    npix = max(16, sc.npix)
    scale = sc.extent / npix
    extra = vm.Image(GaussianField(np.zeros((npix, npix)), sigma=2.0, length_mas=2.0 * sc.resolution), scale, flux=0.05)
    scene = vm.System(base=model, rest=extra)
    data = as_data(datasets)
    r = fit(scene, image_priors(scene) | {"rest.flux": dist.LogUniform(1e-4, 2.0)}, data, noise=fixed_noise(noise_values, datasets), time_limit=2.0 * TIME_LIMIT)
    image = np.asarray(r.model.rest.render(npix=npix, fov_mas=npix * scale))
    share = float(np.ravel(r.values["rest.flux"])[0])
    np.savez_compressed(out_path, image=image, pixel_scale_mas=scale, flux=share)
    return {"chi2_red_inflated": float(np.sum(r.info["chi2"])) / max(int(np.sum(r.info["ndata"])), 1), "flux_relative_to_model": share,
            "npix": npix, "pixel_scale_mas": scale, "field_mas": npix * scale, "converged": bool(r.info.get("converged")),
            "file": pathlib.Path(out_path).name}


# ---------------------------------------------------------------------------------------------------------------
# Checks (plan 5) and the run


def collapse_components(rung, values, kinds):
    """The components of the chosen rung that `no_collapse` looks at: arcs (length), rings (width) and companions that
    have a position of their own (a `far` offset: a part of the object that sits on the star is not a companion)."""
    out = []
    comps = components_of(rung)
    for c in comps:
        maker = rung[c][0] if isinstance(rung, dict) else None
        if maker == "GaussianArc" and f"{c}.length" in values:
            out.append({"kind": "arc", "component": c, "length": values[f"{c}.length"]})
        if maker == "ModulatedGaussianRim" and f"{c}.fwhm" in values:
            out.append({"kind": "ring", "component": c, "width": values[f"{c}.fwhm"]})
        if maker == "RingedDisk":
            for path, v in values.items():
                if comp_of(path) == c and path.endswith(".sigma"):
                    out.append({"kind": "ring", "component": path, "width": v * IS.FWHM_PER_SIGMA})
        if kinds.get(f"{c}.dra") == "far" and f"{c}.dra" in values:
            out.append({"kind": "companion", "component": c, "dra": values[f"{c}.dra"], "ddec": values[f"{c}.ddec"]})
    return out


def checks_of(rungs, chosen, ladder_rungs, rest, sc, converged):
    """The six checks of plan 5 and `converged`, each ``{"status", "detail"}``, from the fitted rungs."""
    rung = dict(ladder_rungs)[chosen["rung"]]
    kinds = flat_kinds(rung, "x" if any(k.endswith(".flux.index") for k in chosen["values"]) else None)
    out = {"converged": {"status": "pass" if converged else "fail", "detail": None}}

    def put(name, result):
        ok, detail = result
        out[name] = {"status": "warn" if ok is None else ("pass" if ok else "fail"), "detail": detail}

    bounds = {k: tuple(v) for k, v in chosen["bounds"].items()}
    put("no_bounds", IS.no_bounds(chosen["values"], bounds))
    names = [r["rung"] for r in rungs]
    table = [{"rung": r["rung"], "minus2lnl": r["minus2lnl"],
              "contains": [n for n, below in ladder_rungs if n in names and n != r["rung"] and contains(dict(ladder_rungs)[r["rung"]], below)]}
             for r in rungs]
    put("nested", IS.nested(table, chosen["n"]))
    put("starts_agree", IS.starts_agree(chosen["minus2lnl_of_refit_starts"], chosen["n"]))
    if rest and "error" in rest:  # the image step ran and failed: a failure, not a missing measurement
        out["family_complete"] = {"status": "fail", "detail": {"error": rest["error"]}}
    else:  # skipped by --no-residual-image: nothing measured, so a warn
        put("family_complete", IS.family_complete(None if not rest or "flux_relative_to_model" not in rest else rest["flux_relative_to_model"]))
    put("no_collapse", IS.no_collapse(collapse_components(rung, chosen["values"], kinds), sc.resolution))
    put("extent_known", (sc.extent_known, {"extent_source": sc.extent_source}))
    return out


def load_datasets(spec, data_dir, use_spectra):
    """The task's data as the blind track reads them (same channel selection), with the files' OI_FLUX spectra when asked."""
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
    return datasets, spectra


def run(task, family, data_dir, out_dir, starts=RANDOM_STARTS, smoke=False, grey=False, use_spectra=True, residual=True, no_gp=False,
        budget=None):
    # Refuses before any draw when the criteria file is dirty or untracked, or when there is no git checkout and no
    # CRITERIA_COMMIT: an unpinned fit must not reach the `informed_fit` gate.
    criteria = preregistration.criteria_commit(CRITERIA, REPO)
    spec = tasks()[task]
    label = f"{spec['label']}_{family}"
    datasets, spectra = load_datasets(spec, data_dir, use_spectra)
    channels = len(np.unique(np.round(np.concatenate([np.ravel(np.asarray(d.wavel, float)) for d in datasets]) * 1e9)))
    wavel0 = float(np.median(np.concatenate([np.ravel(np.asarray(d.wavel, float)) for d in datasets]))) if channels > 1 and not grey else None
    sc, moments, datasets, vis_scale = measure(datasets, spec, scale_vis=spec["label"] in VIS_SCALE)
    temper = temper_factor(moments)
    tempered = [d.with_error_scale(temper) for d in datasets]
    ctx = {"datasets": datasets, "tempered": tempered, "scales": sc, "moments": moments, "smoke": smoke, "budget": 4 if smoke and budget is None else budget,
           "random": 2 if smoke else starts, "refit": 2 if smoke else REFIT, "max_steps": 50 if smoke else None}
    print(f"{label}: resolution {sc.resolution:.3g} mas; extent {sc.extent:.3g} ({sc.extent_source}); reach {sc.reach:.3g} ({sc.reach_source}); "
          f"temper x{temper:.3g}; moments {None if moments is None else {k: round(v, 3) for k, v in moments.items() if isinstance(v, float)}}")
    rng = np.random.default_rng(7000 + task)
    t0, rungs, ladder_rungs, carry, below_rung, below_final = time.time(), [], ladder(family), {}, None, None
    for name, rung in ladder_rungs:
        got = fit_rung(name, rung, below_rung, ctx, carry, rng)
        grey_values = dict(got["values"])
        if wavel0:
            chrom = chromatic_rung(got, rung, below_rung, below_final, ctx, wavel0)
            if chrom is None:  # no chromatic start survived: the grey fit stands, written in the chromatic form
                chrom = as_chromatic(got, rung, ctx["scales"], wavel0)
            got = {**got, **chrom, "grey": {k: got[k] for k in ("minus2lnl", "bic", "chi2_quoted", "chi2_red", "values", "noise", "starts",
                                                                  "starts_refit", "starts_at_best", "starts_by_origin",
                                                                  "starts_at_best_by_origin", "minus2lnl_of_refit_starts")}}
        carry, below_rung, below_final = grey_values, rung, got
        ctx["noise0"] = dict(got.get("grey", got)["noise"])
        rungs.append(got)
        print(f"{label} {name}: -2lnL={got['minus2lnl']:.1f} k={got['k']} BIC={got['bic']:.1f} chi2/N (quoted errors)={got['chi2_red']:.3g} "
              f"starts at best {got['starts_at_best']}/{got['starts_refit']} refit of {got['starts']} (by origin {got['starts_at_best_by_origin']})")
    chosen = min(rungs, key=lambda x: x["bic"])
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gp = None
    if family in GP_DISKS and not no_gp:
        gp = gp_disk(family, chosen, ctx, wavel0, rng, 3 if smoke else starts, 50 if smoke else None, out_dir / f"{label}_gp_disk.npz")
        if gp and "chi2_red" in gp:
            print(f"{label} multipolar GP disk: chi2/N={gp['chi2_red']:.3f} starts at best {gp['starts_at_best']}/{gp['starts']}")
    errors = {"median_vis_error": float(np.nanmedian(np.concatenate([np.ravel(np.asarray(d.d_vis, float)) for d in datasets]))),
              "median_phase_error_deg": float(np.nanmedian(np.concatenate([np.ravel(np.asarray(d.d_phi, float)) for d in datasets])))}
    errors["implied_vis_residual"] = float(np.sqrt(chosen["chi2_red"]) * errors["median_vis_error"])
    errors["implied_phase_residual_deg"] = float(np.sqrt(chosen["chi2_red"]) * errors["median_phase_error_deg"])
    rest = None
    if residual and np.isfinite(chosen["minus2lnl"]):
        try:
            best = model_of(dict(ladder_rungs)[chosen["rung"]], chosen["values"], wavel0)
            rest = residual_image(best, ctx, chosen["noise"], out_dir / f"{label}_rest.npz")
            print(f"{label} residual image: flux {rest['flux_relative_to_model']:.3g} of the model's")
        except Exception as err:  # noqa: BLE001 the parametric result stands without it
            rest = {"error": f"{type(err).__name__}: {err}"}
    checks = checks_of(rungs, chosen, ladder_rungs, rest, sc, chosen["converged"])
    report = {"label": label, "task": task, "family": family, "knowledge": "informed: model family from the contest paper; no value",
              "files": spec["files"], "resolution_mas": sc.resolution, "scales": sc.asdict(), "moments": moments, "temper_factor": temper,
              "visibility_scale": vis_scale, "channels": channels, "chromatic": bool(wavel0), "reference_wavelength_m": wavel0,
              "flux_spectra_fitted": spectra, "virgil": getattr(virgil, "__version__", None), "criteria": CRITERIA, "criteria_commit": criteria,
              "noise_terms": {"vis_error_rel": "LogUniform(1e-4, 3)", "phi_error": "LogUniform(1e-4, 3) radians"},
              "quoted_errors": errors, "residual_image": rest, "gp_disk": gp, "rungs": rungs, "chosen": chosen["rung"], "checks": checks,
              "elapsed_s": time.time() - t0, "smoke": smoke, "budget": ctx["budget"]}
    (out_dir / f"{label}.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    states = {c["status"] for c in checks.values()}
    status = "fail" if "fail" in states else ("warn" if "warn" in states else "pass")
    (out_dir / "result.json").write_text(json.dumps(
        {"status": status, "checks": {k: v["status"] for k, v in checks.items()}, "chosen": chosen["rung"], "chi2_red_quoted": chosen["chi2_red"],
         "minus2lnl": chosen["minus2lnl"], "starts_at_best": chosen["starts_at_best"], "knowledge": "informed", "criteria_commit": criteria,
         "smoke": smoke}, indent=1) + "\n")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--task", type=int)
    ap.add_argument("--model", choices=sorted(FAMILIES))
    ap.add_argument("--data", default="~/data/imaging_contests")
    ap.add_argument("--out", default="contest_informed")
    ap.add_argument("--starts", type=int, default=RANDOM_STARTS, help="random control starts per rung")
    ap.add_argument("--budget", type=int, default=None, help="at most this many stratified-geometry starts per rung, evenly spaced (default: all)")
    ap.add_argument("--smoke", action="store_true", help="a few starts of 50 steps and a coarse map: check that it runs")
    ap.add_argument("--grey", action="store_true", help="one flux per component at every wavelength (the default gives each a spectral index when the data have more than one channel)")
    ap.add_argument("--no-gp-disk", action="store_true", dest="no_gp", help="skip the multipolar Gaussian-process disk of the disk families")
    ap.add_argument("--no-residual-image", action="store_true", dest="no_residual", help="skip the non-parametric image fitted on top of the chosen model")
    ap.add_argument("--no-spectra", action="store_true", dest="no_spectra", help="do not fit the files' OI_FLUX spectra")
    a = ap.parse_args(argv)
    if a.list:
        for name in FAMILIES:
            print(name + ": " + " -> ".join(r[0] for r in ladder(name)))
        return 0
    if a.task is None or a.model is None:
        ap.error("--task and --model are required")
    run(a.task, a.model, pathlib.Path(a.data).expanduser(), pathlib.Path(a.out).expanduser(), a.starts, a.smoke, a.grey, not a.no_spectra,
        not a.no_residual, a.no_gp, a.budget)
    return 0


if __name__ == "__main__":
    sys.exit(main())
