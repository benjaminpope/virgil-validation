"""Informed fits of the imaging contests' data: the model family the contest paper describes, fitted from the data.

    python scripts/contest_informed.py --list
    python scripts/contest_informed.py --task N --model FAMILY [--data DIR] [--out DIR] [--budget N] [--smoke] [--image]

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
- **Information** (before any fit): median V^2, the median quoted closure-phase error, how many triangles have all
  three V^2 above 3 sigma. Closure phases that carry no information are still fitted; the fit is then judged on V^2.
- **Nuisance**: one global visibility gain (on log |V|, prior width 0.1) is marginalised in every fit, never estimated
  and never divided out.
- **Starts** (plan 4), each recorded with its origin. Rung 0: `moments` (the visibility-moment fit) and `START_BUDGET`
  broad `grid` starts (stratified geometry: sizes, inclination x angle, arc angles and lengths). A rung above: the rung
  below with what it adds at negligible flux; a new positioned component first gets a tiled likelihood map whose ranked
  peaks (never thresholded) each start a `map` fit, and `jitter` starts move the inherited values narrowly around the best
  peak (a rung that places nothing starts from the rung below and its jitters).
- **Errors** (plan 3): every fit of the ladder is on errors inflated by one factor, sqrt(chi2/N) of the rung-0 fit
  (tempered); rungs are compared by chi2 on those tempered errors with ln N per added parameter. The error terms
  (`vis_error_rel`, `phi_error`) are free only in one last fit of the chosen rung. Fluxes have no cap (LogUniform(1e-4,
  1e4) ratios); a multi-channel data set is chromatic, each component with a spectral index in Uniform(-4, 4).
- **Checks** (plan 5) in result.json: `converged`, `no_bounds`, `nested`, `starts_agree`, `fit_standard` (normalised
  residual rms of the last fit, at the tempered errors, at most 2 on V^2 and on the informative closure phases, else the fit is incomplete),
  `family_complete`, `no_collapse`, `extent_known`; the mandate stage requires them.

After the ladder, and not as a rung: a simple non-parametric image (a Gaussian-process field) is fitted on top of the
final model, held fixed, to show what resolved light the parametric model leaves out (<label>_rest.npz); virgil's CLEAN
probe on top of the final fit (<label>_probe.json, .npz, .png); and, with --image, a Gaussian-process image in place of
the extended light (<label>_image.json).

Spectral information is used in two ways. When the data have more than one channel, every component but the
reference gets its own spectral index, so the model is chromatic wherever the object can be. And with --spectra the
files' OI_FLUX spectra are fitted with the visibilities (scale and linear trend marginalised).

Writes into --out: <label>_<family>.json (the report) and result.json (status and checks for the campaign).

Task numbers and files are those of scripts/contest_images.py (TASKS).
"""

import argparse
import ast
import hashlib
import itertools
import json
import pathlib
import sys
import time

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
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
# The recipe's policy constants (plan 3 and 4). The checks' thresholds are in crosscheck.informed_starts and the criteria file.
GAIN_WIDTH = 0.1  # the global visibility gain's prior width on log |V|: calibration scale is never known better than 10 %
N_PEAKS = 5  # likelihood-map peaks that each start a fit
N_JITTER = 3  # more starts at the best peak, the inherited values moved narrowly
JITTER = {"size": 0.2, "flux": 0.2, "angle": 10.0, "position": 0.1, "colour": 0.2}  # fractions; degrees; resolution elements; index
START_BUDGET = IS.START_OF  # broad stratified starts of the first rung (and starts of a rung that places nothing)
IMAGE_SIGMA, IMAGE_LENGTH, IMAGE_AGREE, IMAGE_SECS = 2.0, 2.0, 0.9, 1800.0  # image step: log-brightness sigma; length in resolution elements; pixel correlation; s per fit
PROBE_CHUNK, PROBE_MAX = 250, 500  # CLEAN iterations per call, and in all
RC_MORE = 75  # exit code of a call that used its time budget: not finished, call again
TIME_LIMIT = 300.0  # seconds per start; a start that runs out is recorded as not converged
NEGLIGIBLE = 1e-3  # flux ratio of a component added at negligible flux (the prior reaches 1e-4)
NOISE = {"vis_error_rel": dist.LogUniform(1e-4, 3.0), "phi_error": dist.LogUniform(1e-4, 3.0)}  # plan 3; phi in radians
VIS_SCALE = ("2004_data2",)  # datasets normalised without part of their flux (scripts/score_2004.py): V^2 above 1 at zero baseline
TURNS_MAX = 8.0  # the prior of a spiral's turns reaches well past several
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
        # Shapes of a thin, long part and of a spiral arm, all from the scales (no number from a paper).
        "thin": dist.LogUniform(res / 5.0 / extent, 1.0),  # an axis ratio: the minor FWHM from a fifth of a resolution element to the major's
        "span": dist.Uniform(-extent / 2.0, extent / 2.0),  # where along its axis a part starts, within half the extent
        "turns": dist.Uniform(0.5, TURNS_MAX),  # turns of a spiral: from half a turn to well past several
        "arm_width": dist.LogUniform(res, extent / 4.0),  # width of a spiral arm: at least a resolution element
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
            kwargs[p] = jnp.array([value]) if leaf in VECTORS else value  # traced values (maps, curvature) must stay traced
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
        ring = any(name in RING_LEAVES for name in leaves[comp])
        if kind == "colour":  # a spectral index: grey to start
            value = 0.0
        elif kind in ("size", "small", "width", "ring_width"):
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
        elif leaf == "ratio" and kind == "ratio":  # an axis ratio (not the ``ratio`` of a chromatic ``flux.ratio``)
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
        if leaf_of(path) in SIZE_LEAVES and kind in ("size", "small", "width", "ring_width"):  # a barely resolved star is a size too: rung 0 always has broad starts
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
    old_comps = set(components_of(below_rung))
    gained = [c for c in components_of(rung) if c not in old_comps
              and not any(comp_of(p) == c and k in FLUX_KINDS for p, k in kinds.items())]
    for path, kind in kinds.items():
        if kind not in FLUX_KINDS:
            continue
        # a reference gained: the old components' fluxes (new parameters, as the old reference had none) go to the top
        # of their priors; every other new flux or weight starts negligible
        old_part = (kind == "flux" and "profile" not in path and leaf_of(path) in ("flux", "ratio") and comp_of(path) in old_comps)
        if gained and old_part:
            put(path, 1e4)
        elif path not in old:
            put(path, NEGLIGIBLE)
    return out


def thin(starts, budget, rng):
    """At most ``budget`` of the stratified starts, evenly spaced through the list (None keeps all)."""
    if budget is None or len(starts) <= budget:
        return starts
    keep = np.unique(np.round(np.linspace(0, len(starts) - 1, budget)).astype(int))
    return [starts[i] for i in keep]


# ---------------------------------------------------------------------------------------------------------------
# The recipe's machinery (plan 3 and 4): what the data can tell, the gain nuisance, a cache, maps and grids that start a rung


def informative(d):
    """One boolean per closure phase of ``d``: all three baselines of its triangle have V^2 above 3 sigma."""
    if d.i_cps1 is None or not np.size(d.phi):
        return np.zeros(0, bool)
    return IS.informative_triangles(d.vis, d.d_vis, (d.i_cps1, d.i_cps2, d.i_cps3), v2=bool(d.v2_flag), vis_index=d.vis_index,
                                    phi_index=d.phi_index)


def information(datasets):
    """The information check, before any fit: ``(record, masks)`` with the record of `IS.information_record` (median V^2,
    median quoted closure-phase error, triangles with all three V^2 above 3 sigma, what the fit will be judged on) and
    one mask of informative closure phases per dataset. Closure phases that carry no information are still fitted, the fit
    is judged on V^2, and nothing is oriented by them."""
    masks = [informative(d) for d in datasets]
    v2 = np.concatenate([np.ravel(np.asarray(d.vis, float)) ** (1 if d.v2_flag else 2) for d in datasets])
    cp_err = np.concatenate([np.ravel(np.asarray(d.d_phi, float)) for d in datasets if np.size(d.phi)] or [np.zeros(0)])
    record = IS.information_record(sum(m.size for m in masks), sum(int(m.sum()) for m in masks),
                                   float(np.median(v2)) if v2.size else float("nan"), float(np.median(cp_err)) if cp_err.size else None)
    return record, masks


def with_gain(d, width=GAIN_WIDTH):
    """``d`` with one global visibility gain marginalised, not estimated: a gain on log |V|, the same for every sample, of prior
    width ``width`` (the calibration scale is never known better than that), free at every rung. A rung that lacks structure
    cannot bias it, and nothing is divided out of the data."""
    gained = d.with_gains(modes=np.ones((1, int(np.size(d.u)))))
    return eqx.tree_at(lambda x: x.gains.widths, gained, np.array([width]))


class Pause(Exception):
    """The time budget of this call is used up: the work so far is in the cache, run again to continue."""


def _key(*parts):
    """A short stable name for a cache file."""
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=repr).encode()).hexdigest()[:16]


class Cache:
    """Fits, grid points and map tiles memoised on disk (``path``), and a wall-clock budget (``budget`` seconds) after which
    no new piece of work starts and `Pause` is raised, so a long ladder can be run in short calls. With no ``path`` nothing is
    stored (and with no ``budget`` nothing pauses)."""

    def __init__(self, path=None, budget=None):
        self.path, self.budget, self.t0 = (pathlib.Path(path) if path else None), budget, time.time()
        if self.path:
            self.path.mkdir(parents=True, exist_ok=True)

    def spent(self):
        if self.budget is not None and time.time() - self.t0 > self.budget:
            raise Pause()

    def memo(self, kind, parts, compute):
        """``compute()`` (a JSON-able result), or the stored result of the same ``kind`` and ``parts``."""
        if self.path is None:
            self.spent()
            return json.loads(json.dumps(compute(), default=float))
        f = self.path / f"{kind}_{_key(*parts)}.json"
        if not f.exists():
            self.spent()
            f.write_text(json.dumps(compute(), default=float))
        return json.loads(f.read_text())

    def block(self, name, compute):
        """The array ``compute()`` returns, kept as ``name``."""
        if self.path is None:
            self.spent()
            return np.asarray(compute(), float)
        f = self.path / f"{name}.npy"
        if not f.exists():
            self.spent()
            np.save(f, np.asarray(compute(), float))
        return np.load(f)

    def arrays(self, name, compute):
        """The ``{key: array}`` that ``compute()`` returns, kept as ``name`` (strings and numbers come back as 0-d arrays)."""
        if self.path is None:
            self.spent()
            return {k: np.asarray(v) for k, v in compute().items()}
        f = self.path / f"{name}.npz"
        if not f.exists():
            self.spent()
            np.savez(f, **{k: np.asarray(v) for k, v in compute().items()})
        with np.load(f) as z:
            return {k: z[k] for k in z.files}


def grey_values(values):
    """Fitted values with every ``<c>.flux.ratio`` as ``<c>.flux`` and the spectral indices dropped: a map is grey in the new part."""
    return {(k[: -len(".ratio")] if k.endswith(".flux.ratio") else k): v for k, v in values.items() if not k.endswith(".flux.index")}


def placed(rung, below):
    """The components a rung adds that a likelihood map places (those with a position of their own)."""
    return [c for c in components_of(rung) if c not in components_of(below)
            and (c in rung.positions if isinstance(rung, Scene) else "dra" in rung[c][1])]


def resolved_primary(rung):
    """Whether the rung's reference component (its first) has a size of its own: then the rung below is poorly known where a new
    component is searched for, and the map lets its parameters move at every position."""
    first = components_of(rung)[0]
    return any(comp_of(p) == first and leaf_of(p) in SIZE_LEAVES and k == "size" for p, k in flat_kinds(rung).items())


def linear_flux(template, comp, data, dra, ddec):
    """The linear flux of ``comp`` and its weight at every (dra, ddec) of a grid, datasets combined by inverse variance
    (virgil's `linear_flux_grid` solves the flux at each position)."""
    num = den = 0.0
    for d in data:
        g = linear_flux_grid(template, d, {f"{comp}.dra": dra, f"{comp}.ddec": ddec, f"{comp}.flux": np.array([1e-3])})
        w = 1.0 / np.asarray(g.flux_error) ** 2  # one axis per coordinate; the flux is solved for, not scanned
        num, den = num + np.nan_to_num(np.asarray(g.flux) * w), den + np.nan_to_num(w)
    return num, den


def profile_evaluator(rung, comp, base, priors, kinds, free, tdata, iters=20):
    """Score (chi2 gained over the rung below) of a new component at each position, with the base's ``free`` parameters and the
    flux re-optimised at every point by a few Levenberg-Marquardt steps from the rung below's values (the base is poorly
    known when the primary is resolved; virgil's `optimized_likelihood_grid` optimises the flux only). Returns
    ``(evaluate(dra_axis, ddec_axis), flux_at(dra, ddec))``. Everything here is grey: ``base`` and ``free`` are grey values."""
    names = [*free, f"{comp}.flux"]
    bounds = bounds_of(priors, kinds)
    lo, hi = jnp.array([bounds[n][0] for n in names]), jnp.array([bounds[n][1] for n in names])
    fixed = {k: v for k, v in base.items() if k not in names and k not in (f"{comp}.dra", f"{comp}.ddec")}

    def resid(x, p):
        vals = {**fixed, **{n: x[i] for i, n in enumerate(names)}, f"{comp}.dra": p[0], f"{comp}.ddec": p[1]}
        m = model_of(rung, vals)
        return jnp.concatenate([likelihood.whitened_residuals(m, d) for d in tdata])

    def lm(x0, p):
        def chi(x):
            return jnp.sum(resid(x, p) ** 2)

        def body(_, s):
            x, lam, c = s
            r, J = resid(x, p), jax.jacfwd(resid)(x, p)
            A = J.T @ J
            step = jnp.linalg.solve(A + lam * jnp.diag(jnp.diag(A)) + 1e-12 * jnp.eye(A.shape[0]), -J.T @ r)
            xn = jnp.clip(x + step, lo, hi)
            cn = chi(xn)
            ok = jnp.isfinite(cn) & (cn < c)
            return jnp.where(ok, xn, x), jnp.where(ok, lam / 3.0, lam * 4.0), jnp.where(ok, cn, c)

        x, _, c = jax.lax.fori_loop(0, iters, body, (x0, 1e-2, chi(x0)))
        return x[-1], c

    solve = jax.jit(jax.vmap(lm))
    template = model_of(rung, base)
    c0 = float(sum(np.sum(np.asarray(likelihood.whitened_residuals(model_of(rung, {**base, f"{comp}.flux": 1e-9}), d)) ** 2) for d in tdata))
    x_free = jnp.array([base[n] for n in free])

    def evaluate(dra, ddec):
        grid_a, grid_d = np.meshgrid(dra, ddec, indexing="ij")
        num, den = linear_flux(template, comp, tdata, dra, ddec)
        flux0 = np.clip(np.nan_to_num(num / np.where(den > 0, den, np.inf)).ravel(), 1e-3, float(hi[-1]))
        pts = np.stack([grid_a.ravel(), grid_d.ravel()], 1)
        out = np.empty(len(pts))
        for i in range(0, len(pts), 256):
            x0 = jnp.concatenate([jnp.tile(x_free, (len(pts[i:i + 256]), 1)), jnp.asarray(flux0[i:i + 256])[:, None]], 1)
            out[i:i + 256] = np.asarray(solve(x0, jnp.asarray(pts[i:i + 256]))[1])
        return np.where(np.isfinite(out), c0 - out, -np.inf).reshape(grid_a.shape)

    def flux_at(dra, ddec):
        num, den = linear_flux(template, comp, tdata, np.array([dra]), np.array([ddec]))
        flux = float(np.clip(np.nan_to_num(num / np.where(den > 0, den, np.inf)).ravel()[0], 1e-3, float(hi[-1])))
        return float(solve(jnp.concatenate([x_free, jnp.array([flux])])[None], jnp.array([[dra, ddec]]))[0][0])

    return evaluate, flux_at


def tiled(cache, tag, half, step, evaluate, tile):
    """`IS.tiled_map` with every tile kept in ``cache`` (a call that pauses resumes the map where it stopped)."""
    return IS.tiled_map(evaluate, half, step, tile=tile, memo=lambda i, j, compute: cache.block(f"tile_{tag}_{i}_{j}", compute))


def map_seeds(comp, rung, base, below_rung, carry, ctx):
    """Ranked peaks of the likelihood map of a component the rung adds (plan 4.3), no threshold: the best ``N_PEAKS`` go on to be
    refitted, whatever their significance. The rung below is held at its fit (``base``: the new rung's values with the new
    part at negligible flux), and the new component's position is scanned over the reach (half the extent for a part of
    the object) at half a resolution element, in tiles so memory stays bounded. The score is the improvement in chi2 of
    a linear flux solved at each position, on the tempered errors, so a companion carrying under a hundredth of the
    flux shows beside a poor surface model. With a resolved primary (`resolved_primary`) the rung below's parameters are
    also re-optimised at each position, coarsely and then on a finer map around the best peaks. The map is grey in the
    new part. Returns ``(seeds, record)``; a seed is ``{path: value}`` with a ``score``."""
    sc, tdata, cache, wavel0 = ctx["scales"], ctx["tempered"], ctx["cache"], ctx.get("wavel0")
    kinds, priors = flat_kinds(rung), build_priors(rung, sc)
    gbase, gcarry = grey_values(base), grey_values(carry)
    ck = {leaf_of(p): k for p, k in kinds.items() if comp_of(p) == comp}
    half = sc.reach if ck.get("dra") == "far" else sc.extent / 2.0
    exclude = max(IS.COMPANION_ELEMENTS * sc.resolution if ck.get("flux") == "planet" else 0.0, ctx.get("exclude", 0.0))
    profile = bool(ctx.get("profile_maps", True)) and resolved_primary(below_rung)
    smoke = ctx.get("smoke", False)
    free = [k for k in gcarry if k in priors and comp_of(k) in components_of(below_rung)]
    step = sc.resolution if profile else sc.resolution / 2.0
    requested = step
    if smoke:
        step = max(step, 2.0 * half / (24 if profile else 60))
    tag = _key(comp, sorted(gbase.items()), half, sc.resolution, profile, exclude, step)
    if profile:
        evaluate, flux_at = profile_evaluator(rung, comp, gbase, priors, kinds, free, tdata)
        axis, surface, actual = tiled(cache, tag, half, step, evaluate, 32)
    else:
        template = model_of(rung, gbase)

        def evaluate(dra, ddec):
            num, den = linear_flux(template, comp, tdata, dra, ddec)
            return np.where(num > 0, num**2 / np.where(den > 0, den, np.inf), 0.0)

        axis, surface, actual = tiled(cache, tag, half, step, evaluate, 128)
    peaks = IS.map_peaks(axis, surface, 20, 2.0 * sc.resolution, exclude)
    key = f"{comp}.flux.ratio" if wavel0 else f"{comp}.flux"
    seeds = []
    for dra, ddec, score in peaks[: ctx.get("n_peaks", N_PEAKS)]:
        if profile:  # a finer map around the peak (a quarter of the coarse step), and the flux re-optimised there
            fine = sc.resolution / 4.0
            ax = np.arange(-4, 5) * fine
            local = np.asarray(evaluate(dra + ax, ddec + ax))
            i, j = np.unravel_index(np.argmax(local), local.shape)
            dra, ddec, score = dra + ax[i], ddec + ax[j], float(local[i, j])
            flux = flux_at(dra, ddec)
        else:
            num, den = linear_flux(template, comp, tdata, np.array([dra]), np.array([ddec]))
            flux = float(np.ravel(num / np.where(den > 0, den, np.inf))[0])
        seeds.append({f"{comp}.dra": float(dra), f"{comp}.ddec": float(ddec), key: float(max(flux, 1e-4)), "score": float(score)})
    return seeds, {"component": comp, "profile": profile, "half_mas": float(half), "step_mas": float(actual), "requested_step_mas": float(requested),
                   "points": int(axis.size) ** 2, "peaks": [(p[0], p[1], p[2]) for p in peaks], "seeds": seeds}


# Grids that start what the moments cannot: an angle, and the shape a rung adds that no map places.
SCAN_FRACTIONS, SCAN_FLUXES = (0.2, 0.5, 0.8), tuple(np.geomspace(0.01, 10.0, 7))  # candidates: a fraction; a flux relative to the reference


def asymmetry(rung, centre, ctx, n_angles=24, amplitudes=(0.25, 0.5, 0.75), skip=()):
    """An angle the moments cannot give (the bright side of an azimuthally modulated component: an amplitude and angle, or the cosine
    and sine terms of harmonic m, or the direction an arc points, a ``full_angle`` ``pa``) is started from a likelihood grid over
    that angle, everything else at its moment start. The amplitude is not known either, so it is stepped over the middle of its
    prior (0 to 1; +-0.7 for a cosine or sine term). ``skip``: values already fitted (a rung below), whose angles are kept.
    Returns ``(changed centre, record)``."""
    wavel0, tempered, cache = ctx.get("wavel0"), ctx["tempered"], ctx["cache"]
    out, rec = dict(centre), {}
    kinds = flat_kinds(rung, wavel0)
    for comp in components_of(rung):
        modes = [(comp, f"{comp}.az_pas", lambda a, t, comp=comp: {f"{comp}.az_amps": a, f"{comp}.az_pas": t})] * (f"{comp}.az_pas" in centre)
        modes += [(f"{comp}.m{m}", f"{comp}.c{m}", lambda a, t, m=m, comp=comp: {f"{comp}.c{m}": 0.7 * a * np.cos(np.radians(t)), f"{comp}.s{m}": 0.7 * a * np.sin(np.radians(t))})
                  for m in range(1, 9) if f"{comp}.c{m}" in centre]
        modes += [(f"{comp}.pa", f"{comp}.pa", lambda a, t, comp=comp: {f"{comp}.pa": t})] * (kinds.get(f"{comp}.pa") == "full_angle")  # where an arc points
        for name, key, setter in [x for x in modes if x[1] not in skip]:
            def compute(setter=setter, key=key, comp=comp):
                grid = []
                for a in (0.0,) if key == f"{comp}.pa" else amplitudes:  # an arc's direction has no amplitude
                    for ang in np.arange(n_angles) * 360.0 / n_angles:
                        cache.spent()
                        grid.append({"amp": a, "angle": float(ang), "chi2": chi2_quoted(model_of(rung, {**centre, **setter(a, float(ang))}, wavel0), tempered)[0]})
                return grid

            grid = cache.memo("asym", (name, sorted(centre.items())), compute)
            best = min(grid, key=lambda g: g["chi2"])
            out |= setter(best["amp"], best["angle"])
            rec[name] = {"best": best, "chi2_range": [best["chi2"], max(g["chi2"] for g in grid)]}
    return out, rec


def shape_scan(rung, base, carry, ctx):
    """Parameters a rung adds that no map places (the radius of a ring, the gap of an annulus, the size and flux of a halo) start from a
    likelihood grid, the rest at the rung below's fit: a size at 8 log steps around its start, a fraction at 0.2, 0.5 and
    0.8, a flux at 7 steps from 0.01 to 10 times the reference. Returns ``(changed base, record or None when nothing is new)``."""
    wavel0, tempered, cache = ctx.get("wavel0"), ctx["tempered"], ctx["cache"]
    kinds = flat_kinds(rung, wavel0)
    new = [p for p, k in kinds.items() if p not in carry and k in ("size", "fraction", "flux")]
    if not new:
        return base, None
    axes = [IS.log_steps(base[p]) if kinds[p] == "size" else SCAN_FRACTIONS if kinds[p] == "fraction" else SCAN_FLUXES for p in new]

    def compute():
        grid = []
        for point in itertools.product(*axes):
            cache.spent()
            c = chi2_quoted(model_of(rung, {**base, **dict(zip(new, point))}, wavel0), tempered)[0]
            grid.append({"values": dict(zip(new, map(float, point))), "chi2": float(c) if np.isfinite(c) else float("inf")})
        return grid

    grid = cache.memo("scan", (new, sorted(base.items())), compute)
    best = min(grid, key=lambda g: g["chi2"])
    return {**base, **best["values"]}, {"scanned": new, "best": best, "chi2_range": [best["chi2"], max(g["chi2"] for g in grid)]}


def jittered(values, kinds, res, rng):
    """Inherited values moved narrowly: sizes and fluxes within 20 %, angles within 10 degrees, positions within a tenth of a
    resolution element, spectral indices within 0.2."""
    out = {}
    for k, v in values.items():
        kind = kinds.get(k)
        if kind in ("angle", "full_angle", "inclination"):
            out[k] = v + rng.uniform(-1, 1) * JITTER["angle"]
        elif kind in ("far", "near"):
            out[k] = v + rng.uniform(-1, 1) * JITTER["position"] * res
        elif kind in ("modulation", "harmonic", "wave"):
            out[k] = v + rng.uniform(-0.1, 0.1)
        elif kind == "colour":
            out[k] = v + rng.uniform(-1, 1) * JITTER["colour"]
        else:
            out[k] = v * rng.uniform(1 - JITTER["size"], 1 + JITTER["size"])
    return out


def rung_start(rung, ctx):
    """``(kinds, priors, centre)`` of a rung: its parameters' kinds, the priors built from the scales, and the data-only start
    (the visibility moments, plan 4.1)."""
    sc, wavel0 = ctx["scales"], ctx.get("wavel0")
    kinds, priors = flat_kinds(rung, wavel0), build_priors(rung, sc, wavel0)
    return kinds, priors, centre_of(rung, kinds, priors, sc, ctx["moments"])


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


def fitc(tag, rung, priors, data, init, ctx, noise=None):
    """One fit, memoised in ``ctx["cache"]`` (a fit that failed is recorded as ``{"error"}``, not hidden). ``tag`` names the rung and
    the data (quoted or tempered errors) so keys cannot collide. Without ``noise`` the record holds the values and chi2 on
    the errors fitted to; with ``noise`` it is `final_record` (the error terms are fitted)."""
    init = {k: float(np.ravel(v)[0]) for k, v in init.items()}
    wavel0 = ctx.get("wavel0")

    def compute():
        r, error = fit_start(rung, priors, as_data(data), init, wavel0, noise, ctx.get("max_steps"))
        if r is None:
            return {"error": error}
        if noise is None:
            chi2 = float(np.sum(r.info["chi2"]))
            return {"values": scalars(r.values), "chi2": chi2 if np.isfinite(chi2) else float("inf"), "converged": bool(r.info.get("converged")),
                    "at_bound": list(r.info.get("at_bound") or [])}
        return final_record(r, rung, ctx["datasets"], wavel0, ctx["scales"])

    return ctx["cache"].memo("fit", (tag, sorted(priors), init, noise is not None, ctx.get("max_steps")), compute)


def fit_ladder(ladder, ctx, rng, scan=False):
    """The ladder (plan 4), every fit on the **tempered** errors (the quoted ones times one factor, fixed by the rung-0 fit) with the
    gain nuisance marginalised; the error terms are fitted once, after the ladder (`final_fit`). Rung 0 starts at the moment
    values and at ``START_BUDGET`` broad stratified starts. Every rung above is started from the rung below, with what it adds
    at negligible flux: a rung that adds a positioned component makes a likelihood map first, starts a fit at each of its
    best ``N_PEAKS`` peaks, and ``N_JITTER`` more at the best peak with the inherited values moved narrowly; a rung that
    places nothing starts from the rung below and ``START_BUDGET - 1`` jitters of it (``scan``: the shape parameters it adds
    first from a grid). Returns one record per rung, each with its best fit on the tempered errors and every start."""
    datasets, sc, wavel0 = ctx["datasets"], ctx["scales"], ctx.get("wavel0")
    n = sum(int(d.n_independent) for d in datasets)
    budget, n_jitter = ctx.get("budget") or START_BUDGET, ctx.get("n_jitter", N_JITTER)
    rungs, carry, below = [], {}, None
    for k, (name, rung) in enumerate(ladder):
        kinds, priors, centre = rung_start(rung, ctx)
        ctx.update(kinds=kinds, priors=priors)
        rec = {"rung": name, "k": len(priors), "maps": [], "scan": None}
        if k == 0:
            first = fitc(f"{name}|quoted", rung, priors, datasets, centre, ctx)  # the rung-0 fit that fixes the tempering
            if "values" not in first:
                raise RuntimeError(f"{name}: the first fit failed: {first['error']}")
            ctx["temper"] = round(float(max(1.0, np.sqrt(first["chi2"] / n))), 3)
            ctx["tempered"] = [d.with_error_scale(ctx["temper"]) for d in datasets]
            centre, rec["asymmetry"] = asymmetry(rung, centre, ctx)
            starts = [("moments", centre)] + [("grid", s) for s in thin(stratified_starts(rung, kinds, priors, sc, ctx["moments"], centre), budget, rng)]
        else:
            inherited = {p: v for p, v in carry.items() if p in priors}
            base = embed_below(rung, below, kinds, priors, seeded(centre, inherited, priors, kinds), wavel0)
            base, rec["asymmetry"] = asymmetry(rung, base, ctx, skip=carry)  # a bright side the rung below lacked
            new, seeds = placed(rung, below), []
            for extra in new[1:]:  # a rung that adds several positioned components: the others at their best peak
                found, record = map_seeds(extra, rung, base, below, carry, ctx)
                rec["maps"].append(record)
                if found:
                    base = seeded(base, {p: v for p, v in found[0].items() if p != "score"}, priors, kinds)
            if new:
                seeds, record = map_seeds(new[0], rung, base, below, carry, ctx)
                rec["maps"].append(record)
            if seeds:
                starts = [("map", seeded(base, {p: v for p, v in s.items() if p != "score"}, priors, kinds)) for s in seeds]
                best = {p: v for p, v in seeds[0].items() if p != "score"}
                starts += [("jitter", seeded(seeded(base, jittered(inherited, kinds, sc.resolution, rng), priors, kinds), best, priors, kinds))
                           for _ in range(n_jitter)]
            else:
                if scan:
                    base, rec["scan"] = shape_scan(rung, base, carry, ctx)
                starts = [("scan" if rec["scan"] else "below", base)]
                starts += [("jitter", seeded(base, jittered(inherited, kinds, sc.resolution, rng), priors, kinds)) for _ in range(budget - 1)]
        fits = []
        for origin, init in starts:
            fits.append({"origin": origin, **fitc(f"{name}|t{ctx['temper']}", rung, priors, ctx["tempered"], init, ctx)})
        ok = [f for f in fits if "values" in f]
        if not ok:
            raise RuntimeError(f"{name}: no start survived: {sorted({f['error'] for f in fits if 'error' in f})[:3]}")
        best = min(ok, key=lambda f: f["chi2"])
        rec |= {"chi2_tempered": best["chi2"], "values": best["values"], "converged": best["converged"], "at_bound": best["at_bound"],
                "best_origin": best["origin"], "starts": len(fits), "failed_starts": len(fits) - len(ok),
                "errors": sorted({f["error"] for f in fits if "error" in f})[:3],
                "starts_by_origin": {o: sum(f["origin"] == o for f in fits) for o in sorted({f["origin"] for f in fits})},
                "start_runs": [{x: f[x] for x in ("origin", "chi2", "values", "converged") if x in f} for f in ok]}
        rungs.append(rec)
        carry, below = dict(best["values"]), rung
        print(f"{name}: tempered chi2 {best['chi2']:.1f} (x{ctx['temper']}) from {len(ok)}/{len(fits)} starts; best from {best['origin']}", flush=True)
    return rungs


def chosen_rung(ladder, rungs, ctx):
    """Index of the last rung that earns its parameters. Each rung's chi2 on the **tempered** errors (the quoted ones times the
    rung-0 factor, the gains marginalised: the errors the ladder was fitted on) is measured at its best fit; the walk up the
    ladder goes on while chi2 falls by more than ln N per added parameter, and a rung that adds nothing (over-resolved light
    that is not there) is not final. Writes ``minus2lnl`` (that chi2: the errors are fixed, so the likelihood's normalisation is a
    constant; it is the scale of the `nested` check, and of the starts' agreement), ``chi2_quoted`` and ``chi2_red`` (on the quoted
    errors, reported beside it), ``n`` and ``bic`` (on ``minus2lnl``) into each record."""
    n = sum(int(d.n_independent) for d in ctx["datasets"])
    for rec, (_, rung) in zip(rungs, ladder):
        model = model_of(rung, rec["values"], ctx.get("wavel0"))
        chi2, quoted = chi2_quoted(model, ctx["tempered"])[0], chi2_quoted(model, ctx["datasets"])[0]
        rec |= {"minus2lnl": chi2, "chi2_quoted": quoted, "n": n, "chi2_red": quoted / n, "bic": chi2 + rec["k"] * float(np.log(n))}
    k = 0
    while k + 1 < len(rungs) and rungs[k]["minus2lnl"] - rungs[k + 1]["minus2lnl"] > (rungs[k + 1]["k"] - rungs[k]["k"]) * np.log(n):
        k += 1
    return k


def final_fit(ladder, rungs, ctx, k):
    """One last fit of rung ``k`` with virgil's error terms free (`vis_error_rel`, `phi_error`), on the quoted errors, from that
    rung's tempered optimum. The only fit with the error terms free; `final_record` says what it found."""
    name, rung = ladder[k]
    priors = build_priors(rung, ctx["scales"], ctx.get("wavel0"))
    start = {**rungs[k]["values"], **{kk: 0.1 for kk in noise_names(ctx["datasets"])}}
    got = fitc(f"{name}|final", rung, priors, ctx["datasets"], start, ctx, noise=noise_priors(ctx["datasets"]))
    if "minus2lnl" not in got:
        raise RuntimeError(f"{name}: the final fit failed: {got.get('error')}")
    return got


def sigma_fn(rung, names, data, wavel0=None):
    """``values -> {name: 1 sigma}`` from the Gauss-Newton curvature J^T J of the whitened residuals of all ``data`` together
    (the gains marginalised), compiled once for a rung. A direction the data do not constrain gets an enormous sigma, never zero."""
    jac = jax.jit(jax.jacfwd(lambda x: jnp.concatenate([likelihood.whitened_residuals(model_of(rung, dict(zip(names, x)), wavel0), d) for d in data])))

    def sigma(values):
        J = np.asarray(jac(jnp.array([float(values[k]) for k in names])))
        A = J.T @ J
        D = np.where(np.diag(A) > 0, np.sqrt(np.diag(A)), 1.0)
        s = np.sqrt(np.abs(np.diag(np.linalg.inv(A / np.outer(D, D) + 1e-9 * np.eye(len(names)))))) / D
        return dict(zip(names, np.where(np.isfinite(s), s, np.inf)))

    return sigma


def start_check(ladder, rungs, ctx):
    """Start agreement of every rung from its saved starts (`IS.start_agreement`: chi2 on the tempered errors, the scale the starts
    were fitted on, gains marginalised; each parameter's sigma from the joint Gauss-Newton curvature at those errors).
    ``{rung name: record}``."""
    datasets, sc, wavel0 = ctx["tempered"], ctx["scales"], ctx.get("wavel0")
    out = {}
    for (name, rung), rec in zip(ladder, rungs):
        kinds, priors = flat_kinds(rung, wavel0), build_priors(rung, sc, wavel0)
        runs = [f for f in rec["start_runs"] if f["origin"] not in IS.CONTROL_ORIGINS]
        scored = []
        if runs:
            sigma = sigma_fn(rung, sorted(runs[0]["values"]), datasets, wavel0)
            for f in runs:
                chi2 = chi2_quoted(model_of(rung, f["values"], wavel0), datasets)[0]
                scored.append({"origin": f["origin"], "values": f["values"], "chi2q": chi2 if np.isfinite(chi2) else float("inf"), "sigma": sigma(f["values"])})
        out[name] = IS.start_agreement(scored, sc.resolution, bounds_of(priors, kinds))
    return out


def residual_stats(model, datasets, noise_values, masks):
    """The fit standard's numbers: the normalised residuals (residual over the error, the gains marginalised, with the error
    terms ``noise_values`` in when given) of ``model`` on every dataset, their rms on V^2 and on the closure phases of the
    informative triangles ``masks`` (and on all of them, beside). The fit standard is judged with no error terms (on the
    tempered datasets): error terms fitted to these residuals set the effective variance to the residuals' own, so the rms
    would be about 1 whatever the model leaves behind."""
    names = noise_names(datasets)
    vis, phase, every = [], [], []
    for i, d in enumerate(datasets):
        terms = {t: float(noise_values[k]) for k, (j, t) in names.items() if j == i and k in noise_values}
        w = np.ravel(np.asarray(likelihood.whitened_residuals(model, d, **terms)))
        nv, nphi = int(np.size(d.vis)), int(np.size(d.phi))
        vis.append(w[:nv])
        phases = w[nv:nv + nphi]
        every.append(phases)
        if masks[i].size == phases.size:
            phase.append(phases[masks[i]])
    def rms(parts):
        return float(np.sqrt(np.mean(np.concatenate(parts) ** 2))) if parts and np.concatenate(parts).size else None

    return {"rms_v2": rms(vis), "rms_cp": rms(phase), "rms_cp_all_triangles": rms(every), "n_v2": int(sum(v.size for v in vis)),
            "n_informative_triangles": int(sum(p.size for p in phase)), "n_triangles": int(sum(e.size for e in every))}


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
    tol = IS.SAME_DELTA
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
            "starts": len(runs), "starts_at_best": int(sum(c <= IS.SAME_DELTA + chi2 for c, _, _ in runs)),
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
# After the ladder: the residual probe, and the image step (neither is a rung)


def flat(model):
    """The leaf components of a (nested) System, by name."""
    return {k: v for n, c in model.components.items() for k, v in (flat(c).items() if isinstance(c, vm.System) else [(n, c)])}


def offsets(model):
    """(dra, ddec) of every component with a position."""
    return [(float(np.ravel(np.asarray(c.dra))[0]), float(np.ravel(np.asarray(c.ddec))[0])) for c in flat(model).values() if hasattr(c, "dra")]


def field_half(model, extent):
    """Half-width of the probe's and the figure's field: half the extent, or 1.3 x the farthest positioned component (a planet
    or companion may lie beyond the extent), whichever is larger."""
    return max(extent / 2.0, 1.3 * max((np.hypot(*o) for o in offsets(model)), default=0.0))


def probe(model, datasets, ctx, tag=""):
    """virgil's CLEAN on top of the fitted ``model``, held fixed, on a support of the extent: where its components cluster says what
    the next rung should add. It runs ``PROBE_CHUNK`` iterations at a time, up to ``PROBE_MAX`` (each chunk is cached, so a
    call that pauses resumes), and stops when the fit stalls. Returns ``(record, components)``; the record's
    ``flux_share_of_model`` is the share of the model's total light that the image carries."""
    sc, cache = ctx["scales"], ctx["cache"]
    pix = sc.resolution / 2.0
    half = field_half(model, sc.extent)
    npix = int(np.ceil(2.0 * half / pix))
    npix += npix % 2
    support = vm.circular_support(npix, pix, half)
    state, done, first = None, 0, None
    for i in range(0, PROBE_MAX, PROBE_CHUNK):
        def compute(state=state):
            r = imaging.clean(as_data(datasets), npix, pix, base=model, support=support, max_iterations=PROBE_CHUNK,
                              init=None if state is None else state["components"])
            return {"components": np.asarray(r.components, float), "chi2_red": np.asarray(r.chi2_red, float), "stop": str(r.stop)}

        state = cache.arrays(f"probe_{tag}_{i}", compute)
        first = float(state["chi2_red"][0]) if first is None else first  # the fitted model alone
        done += max(len(state["chi2_red"]) - 1, 0)
        if str(state["stop"]) != "max_iterations":
            break
    comp, chi2 = np.asarray(state["components"], float), np.asarray(state["chi2_red"], float)
    ys, xs = np.nonzero(comp > 0)
    dra, ddec = (npix / 2.0 - xs) * pix, (npix / 2.0 - ys) * pix  # east left, north up (row 0 is north)
    w = comp[ys, xs]
    order = np.argsort(w)[::-1][:5]
    # CLEAN's pixel fluxes are relative to the base's reference component (flux 1) and the model is normalised by its total flux,
    # so ``comp.sum()`` alone says nothing: the share of the model's light that the image carries is clean / (clean + the base's own total).
    base_total = float(sum(np.ravel(np.asarray(getattr(getattr(c, "flux", 1.0), "ratio", getattr(c, "flux", 1.0))))[0] for c in model.components.values()))
    return {"flux_relative_to_reference": float(comp.sum()), "flux_share_of_model": float(comp.sum() / (comp.sum() + base_total)), "iterations": done,
            "stop": str(state["stop"]), "chi2_red_before": first, "chi2_red_after": float(chi2[-1]), "npix": npix, "pixel_mas": pix,
            "support_radius_mas": half, "flux_weighted_radius_mas": float(np.sum(w * np.hypot(dra, ddec)) / max(w.sum(), 1e-300)),
            "five_largest_components": [{"dra": float(dra[i]), "ddec": float(ddec[i]), "share_of_clean_flux": float(w[i] / w.sum())} for i in order]}, comp


def probe_figure(path, model, comp, rec, title):
    """The fitted model beside the CLEAN components found on top of it, saved to ``path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    half = comp.shape[0] * rec["pixel_mas"] / 2.0
    img = np.asarray(model.render(npix=comp.shape[0], fov_mas=2 * half), float)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.2), dpi=90)
    for a, z, name in ((ax[0], np.clip(img, 0, None) ** 0.5, "fitted model (square root)"), (ax[1], comp, "CLEAN components on the model")):
        a.imshow(z, origin="upper", cmap="gray", extent=[half, -half, -half, half])
        a.set_title(name, fontsize=9)
        a.set_xlabel("RA offset (mas), east left")
        for dra, ddec in offsets(model):  # ring every positioned component: a faint one is invisible at this stretch
            if np.hypot(dra, ddec) > 0:
                a.plot(dra, ddec, "o", mfc="none", mec="c", ms=12)
    fig.suptitle(f"{title}; CLEAN flux {rec['flux_share_of_model']:.3g} of the model's; chi2/N {rec['chi2_red_before']:.3g} to {rec['chi2_red_after']:.3g}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def cp_rms(model, d, mask):
    """Closure-phase rms (degrees) of ``model`` on the informative triangles ``mask`` of ``d``, beside their median quoted error."""
    nv, nphi = int(np.size(d.vis)), int(np.size(d.phi))
    r = np.ravel(np.asarray(d.residuals(d.model(model)), float))[nv:nv + nphi]
    err = np.ravel(np.asarray(d.d_phi, float))
    if not mask.any() or mask.size != r.size:
        return None
    return {"n_triangles": int(mask.sum()), "rms_deg": float(np.degrees(np.sqrt(np.mean(r[mask] ** 2)))),
            "median_quoted_error_deg": float(np.degrees(np.median(err[mask])))}


def image_step(rung, values, ctx, masks, raw, parametric=()):
    """The image step, after the ladder and not a rung: where the family describes structure no parametric component holds, a Gaussian-process
    image (fixed hyperparameters: a log-brightness sigma of ``IMAGE_SIGMA``, a length of ``IMAGE_LENGTH`` resolution elements) takes
    the place of the fitted rung's extended light, confined by `circular_support` to the radius that holds 99 % of that
    light (at most half the extent), at pixel lambda / (4 B_max). Positioned components and over-resolved light stay
    parametric and free. An image has one colour, so each dataset (band) is imaged on its own. Convergence: two fits, one from
    a flat image and one from the parametric model's image, must agree (pixel correlation above ``IMAGE_AGREE`` after smoothing
    to the resolution); the closure-phase rms on informative triangles is reported beside their errors. ``parametric``: components kept
    parametric besides those (a star that is not part of the extended light). ``raw``: the datasets without the gain nuisance."""
    from scipy.fft import dctn
    from scipy.ndimage import gaussian_filter
    from virgil.fields import GaussianField, field_spectrum

    sc, wavel0, cache = ctx["scales"], ctx.get("wavel0"), ctx["cache"]
    kinds, priors = flat_kinds(rung, wavel0), build_priors(rung, sc, wavel0)
    parts = flat(model_of(rung, values, wavel0))
    keep = [c for c, m in parts.items() if f"{c}.dra" in kinds or isinstance(m, vm.Resolved) or c in parametric]
    replaced = {c: m for c, m in parts.items() if c not in keep}
    if not replaced:
        return {"skipped": "every component is positioned or over-resolved: there is no extended light for an image to replace"}, {}
    gone = vm.System(**replaced)
    free = {k: v for k, v in priors.items() if comp_of(k) in keep}
    limit, pix = sc.extent / 2.0, sc.resolution / 2.0
    light = np.asarray(gone.render(npix=256, fov_mas=2.0 * limit), float)
    radius = np.hypot(*np.meshgrid(*[(np.arange(256) - 127.5) * 2.0 * limit / 256] * 2))
    cum = np.cumsum(light.ravel()[np.argsort(radius.ravel())]) / light.sum()
    half = float(np.clip(np.sort(radius.ravel())[np.searchsorted(cum, 0.99)], sc.resolution, limit))
    npix = int(np.ceil(2.0 * half / pix))
    npix += npix % 2
    support = vm.circular_support(npix, pix, half)
    template = np.asarray(gone.render(npix=npix, fov_mas=npix * pix), float)
    amp = np.sqrt(np.maximum(field_spectrum((npix, npix), pix, IMAGE_SIGMA, IMAGE_LENGTH * sc.resolution), 0.0))
    start = {"flat": np.zeros((npix, npix)),  # the latents whose field is the template's log brightness (modes with no variance stay 0)
             "parametric": np.where(amp > 0, dctn(np.log(template / template.max() + 1e-3), type=2, norm="ortho") / np.where(amp > 0, amp, 1.0), 0.0)}
    rec = {"support_radius_mas": half, "npix": npix, "pixel_mas": pix, "kept_parametric": keep, "bands": []}
    images = {}
    for i, (d, r) in enumerate(zip(ctx["tempered"], raw)):
        weight = float(sum(np.ravel(np.asarray(c._weight(float(np.median(np.asarray(r.wavel))))))[0] for c in replaced.values()))  # their light at this band

        def target(latent, **p):
            m = flat(model_of(rung, {**values, **p}, wavel0))
            image = vm.Image(GaussianField(latent, IMAGE_SIGMA, IMAGE_LENGTH * sc.resolution), pix, support=support, flux=weight)
            return vm.System(image=image, **{c: m[c] for c in keep})

        runs = {}
        for kind, latent0 in start.items():
            def compute(d=d, latent0=latent0):
                res = fit(target, {"latent": dist.Normal(np.zeros((npix, npix)), 1.0).to_event(2)} | free, d,
                          init={"latent": latent0, **{k: float(values[k]) for k in free}}, method="lbfgs", time_limit=IMAGE_SECS)
                return {"latent": np.asarray(res.values["latent"]), "free": json.dumps(scalars({k: v for k, v in res.values.items() if k in free})),
                        "chi2": float(np.sum(res.info["chi2"])), "converged": bool(res.info.get("converged"))}

            z = cache.arrays(f"image_{_key(sorted(values.items()), i, kind, npix)}", compute)
            model = target(z["latent"], **json.loads(str(z["free"])))
            runs[kind] = {"image": np.asarray(model.components["image"].brightness), "chi2_red_tempered": float(z["chi2"]) / int(d.n_independent),
                          "converged": bool(z["converged"]), "closure_phase": cp_rms(model, r, masks[i])}
        sm = {k: gaussian_filter(v["image"], sc.resolution / pix / IS.FWHM_PER_SIGMA)[support] for k, v in runs.items()}  # smoothed to the resolution
        corr = float(np.corrcoef(sm["flat"], sm["parametric"])[0, 1])
        rec["bands"].append({"dataset": i, "pixel_correlation_smoothed": corr, "runs_agree": bool(corr > IMAGE_AGREE),
                             "closure_phase_parametric_rung": cp_rms(model_of(rung, values, wavel0), r, masks[i]),
                             **{k: {x: y for x, y in v.items() if x != "image"} for k, v in runs.items()}})
        images.update({f"band{i}_{k}": v["image"] for k, v in runs.items()})
        images[f"band{i}_support"], images[f"band{i}_template"] = support, template
    return rec, images


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


REPORT_ONLY = ("fit_standard",)  # computed and reported with every fit; not a gate (acceptance is an evaluator's judgement), so not in the status


def checks_of(rungs, chosen, ladder_rungs, agreement, stats, rest, sc, final):
    """The checks of plan 5 and `converged`, each ``{"status", "detail"}``, from the fitted rungs. ``agreement`` is
    `start_check`'s ``{rung name: record}``, ``stats`` the `residual_stats` of the final fit, ``final`` the final fit (error
    terms free); ``chosen`` is the chosen rung's record. The checks look at the final fit's values and at the chosen rung's
    starts; `nested` looks at every rung."""
    ladder_rungs = dict(ladder_rungs)
    rung = ladder_rungs[chosen["rung"]]
    kinds = flat_kinds(rung, "x" if any(k.endswith(".flux.index") for k in final["values"]) else None)
    out = {"converged": {"status": "pass" if chosen["converged"] and final["converged"] else "fail",
                         "detail": {"ladder": chosen["converged"], "final_fit": final["converged"]}}}

    def put(name, result):
        ok, detail = result
        out[name] = {"status": "warn" if ok is None else ("pass" if ok else "fail"), "detail": detail}

    put("no_bounds", IS.no_bounds(final["values"], {k: tuple(v) for k, v in chosen["bounds"].items()}))
    names = [r["rung"] for r in rungs]
    table = [{"rung": r["rung"], "minus2lnl": r["minus2lnl"],
              "contains": [n for n, below in ladder_rungs.items() if n in names and n != r["rung"] and contains(ladder_rungs[r["rung"]], below)]}
             for r in rungs]
    put("nested", IS.nested(table))
    mine = agreement[chosen["rung"]]
    put("starts_agree", (mine["passes"], {**mine, "all_rungs": {n: {"passes": a["passes"], "at_best": a["at_best"], "n": a["n"]} for n, a in agreement.items()}}))
    put("fit_standard", IS.fit_standard(stats["rms_v2"], stats["rms_cp"], stats["n_informative_triangles"]))
    if rest and "error" in rest:  # the image step ran and failed: a failure, not a missing measurement
        out["family_complete"] = {"status": "fail", "detail": {"error": rest["error"]}}
    else:  # skipped by --no-residual-image: nothing measured, so a warn
        put("family_complete", IS.family_complete(None if not rest or "flux_relative_to_model" not in rest else rest["flux_relative_to_model"]))
    put("no_collapse", IS.no_collapse(collapse_components(rung, final["values"], kinds), sc.resolution))
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


def plain(x):
    """``json.dumps`` default: numpy numbers, booleans and arrays as the plain Python ones."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    return float(x)


def run(task, family, data_dir, out_dir, smoke=False, grey=False, use_spectra=False, residual=True, no_gp=False, budget=None, scan=False,
        do_probe=True, image=False, profile_maps=True, scale_vis=False, cache_dir=None, time_budget=None):
    """The whole recipe for one task and family (design/contest_informed_plan.md, playbooks/informed.md). In order: the criteria's commit
    is pinned; the data are read; the information check; the scales and the visibility moments; the gain nuisance; the
    ladder, every fit on tempered errors (`fit_ladder`); the rung chosen by chi2 on the tempered errors; one last fit of it with the
    error terms free (`final_fit`); start agreement and the fit standard; the checks; and, after the ladder and not as a rung, the
    non-parametric image on top (`residual_image`, `gp_disk`), the CLEAN probe and, with ``image``, the image step."""
    # Refuses before any draw when the criteria file is dirty or untracked, or when there is no git checkout and no
    # CRITERIA_COMMIT: an unpinned fit must not reach the `informed_fit` gate.
    criteria = preregistration.criteria_commit(CRITERIA, REPO)
    spec = tasks()[task]
    label = f"{spec['label']}_{family}"
    raw, spectra = load_datasets(spec, data_dir, use_spectra)
    info, masks = information(raw)  # before any fit
    channels = len(np.unique(np.round(np.concatenate([np.ravel(np.asarray(d.wavel, float)) for d in raw]) * 1e9)))
    wavel0 = float(np.median(np.concatenate([np.ravel(np.asarray(d.wavel, float)) for d in raw]))) if channels > 1 and not grey else None
    sc, moments, raw, vis_scale = measure(raw, spec, scale_vis=scale_vis and spec["label"] in VIS_SCALE)
    datasets = [with_gain(d) for d in raw]
    ctx = {"datasets": datasets, "scales": sc, "moments": moments, "wavel0": wavel0, "cache": Cache(cache_dir, time_budget), "smoke": smoke,
           "budget": 4 if smoke and budget is None else budget, "n_peaks": 2 if smoke else N_PEAKS, "n_jitter": 1 if smoke else N_JITTER,
           "max_steps": 50 if smoke else None, "profile_maps": profile_maps}
    print(f"{label}: information {info['judged_on']} ({info['triangles_all_v2_above_3_sigma']}/{info['triangles']} triangles informative); "
          f"resolution {sc.resolution:.3g} mas; extent {sc.extent:.3g} ({sc.extent_source}); reach {sc.reach:.3g} ({sc.reach_source}); "
          f"moments {None if moments is None else {k: round(v, 3) for k, v in moments.items() if isinstance(v, float)}}", flush=True)
    rng = np.random.default_rng(7000 + task)
    t0, ladder_rungs = time.time(), ladder(family)
    rungs = fit_ladder(ladder_rungs, ctx, rng, scan)
    for r in rungs:  # the bounds of each rung's priors, for `no_bounds`
        rung = dict(ladder_rungs)[r["rung"]]
        r["bounds"] = {k: list(v) for k, v in bounds_of(build_priors(rung, sc, wavel0), flat_kinds(rung, wavel0)).items()}
    k = chosen_rung(ladder_rungs, rungs, ctx)
    chosen, rung = rungs[k], ladder_rungs[k][1]
    final = final_fit(ladder_rungs, rungs, ctx, k)
    agreement = start_check(ladder_rungs, rungs, ctx)
    for r in rungs:
        r["starts_at_best"] = agreement[r["rung"]]["at_best"]
    model = model_of(rung, final["values"], wavel0)
    stats = residual_stats(model, ctx["tempered"], {}, masks)  # judged at the tempered errors, never at the error terms the last fit chose
    for name, (errs, terms) in {"at_quoted_errors": (datasets, {}), "at_fitted_error_terms": (datasets, final["noise"])}.items():
        stats[name] = {k: v for k, v in residual_stats(model, errs, terms, masks).items() if k.startswith("rms")}  # reported beside it
    print(f"{label}: chosen {chosen['rung']}; final fit -2lnL {final['minus2lnl']:.1f}; normalised residual rms V2 {stats['rms_v2']} "
          f"closure phase {stats['rms_cp']}; starts at best {chosen['starts_at_best']}/{agreement[chosen['rung']]['n']}", flush=True)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gp = None
    if family in GP_DISKS and not no_gp:
        gp = gp_disk(family, final, ctx, wavel0, rng, 3 if smoke else START_BUDGET, 50 if smoke else None, out_dir / f"{label}_gp_disk.npz")
        if gp and "chi2_red" in gp:
            print(f"{label} multipolar GP disk: chi2/N={gp['chi2_red']:.3f} starts at best {gp['starts_at_best']}/{gp['starts']}")
    errors = {"median_vis_error": float(np.nanmedian(np.concatenate([np.ravel(np.asarray(d.d_vis, float)) for d in raw]))),
              "median_phase_error_deg": float(np.nanmedian(np.concatenate([np.ravel(np.asarray(d.d_phi, float)) for d in raw if np.size(d.phi)] or [np.zeros(1)])))}
    errors["implied_vis_residual"] = float(np.sqrt(chosen["chi2_red"]) * errors["median_vis_error"])
    errors["implied_phase_residual_deg"] = float(np.sqrt(chosen["chi2_red"]) * errors["median_phase_error_deg"])
    rest = None
    if residual and np.isfinite(final["minus2lnl"]):
        try:
            rest = residual_image(model, ctx, final["noise"], out_dir / f"{label}_rest.npz")
            print(f"{label} residual image: flux {rest['flux_relative_to_model']:.3g} of the model's")
        except Exception as err:  # noqa: BLE001 the parametric result stands without it
            rest = {"error": f"{type(err).__name__}: {err}"}
    checks = checks_of(rungs, chosen, ladder_rungs, agreement, stats, rest, sc, final)
    probe_record = image_record = None
    if do_probe:
        try:
            probe_record, components = probe(model, datasets, ctx, tag=_key(sorted(final["values"].items())))
            (out_dir / f"{label}_probe.json").write_text(json.dumps(probe_record, indent=1, default=plain) + "\n")
            np.savez_compressed(out_dir / f"{label}_probe.npz", components=components)
            try:
                probe_figure(out_dir / f"{label}_probe.png", model, components, probe_record, f"{label}, {chosen['rung']}")
            except ImportError:  # the figure is a convenience
                pass
            print(f"{label} probe: CLEAN flux {probe_record['flux_share_of_model']:.3g} of the model's, chi2/N {probe_record['chi2_red_before']:.3g} to {probe_record['chi2_red_after']:.3g}")
        except Pause:
            raise
        except Exception as err:  # noqa: BLE001 a probe that fails is recorded, the fit stands
            probe_record = {"error": f"{type(err).__name__}: {err}"}
    if image:
        try:
            image_record, pictures = image_step(rung, final["values"], ctx, masks, raw, image if isinstance(image, tuple) else ())
            (out_dir / f"{label}_image.json").write_text(json.dumps(image_record, indent=1, default=plain) + "\n")
            if pictures:
                np.savez_compressed(out_dir / f"{label}_image.npz", **pictures)
        except Pause:
            raise
        except Exception as err:  # noqa: BLE001
            image_record = {"error": f"{type(err).__name__}: {err}"}
    reported = [{**r, "values_tempered": r["values"], "values": final["values"], "noise": final["noise"], "final_fit": final} if r is chosen else r for r in rungs]
    report = {"label": label, "task": task, "family": family, "knowledge": "informed: model family from the contest paper; no value",
              "files": spec["files"], "resolution_mas": sc.resolution, "scales": sc.asdict(), "moments": moments, "temper_factor": ctx["temper"],
              "moment_temper_factor": temper_factor(moments), "information": info,
              "nuisance": {"visibility_gain_width_log_v": GAIN_WIDTH, "applies_to": "every rung and the final fit, marginalised"},
              "visibility_scale": vis_scale or {"applied": False, "why": "the global visibility gain is marginalised (width 0.1), nothing is divided out"},
              "channels": channels, "chromatic": bool(wavel0), "reference_wavelength_m": wavel0, "flux_spectra_fitted": spectra,
              "virgil": getattr(virgil, "__version__", None), "criteria": CRITERIA, "criteria_commit": criteria,
              "noise_terms": {"vis_error_rel": "LogUniform(1e-4, 3)", "phi_error": "LogUniform(1e-4, 3) radians"}, "error_terms_fitted": "once, after the ladder",
              "quoted_errors": errors, "fit_standard_numbers": stats, "start_agreement": agreement, "residual_image": rest, "gp_disk": gp,
              "probe": probe_record, "image_step": image_record, "rungs": reported, "chosen": chosen["rung"], "checks": checks,
              "elapsed_s": time.time() - t0, "smoke": smoke, "budget": ctx["budget"]}
    (out_dir / f"{label}.json").write_text(json.dumps(report, indent=1, default=plain) + "\n")
    states = {c["status"] for name, c in checks.items() if name not in REPORT_ONLY}
    status = "fail" if "fail" in states else ("warn" if "warn" in states else "pass")
    (out_dir / "result.json").write_text(json.dumps(
        {"status": status, "checks": {name: v["status"] for name, v in checks.items()}, "chosen": chosen["rung"], "chi2_red_quoted": chosen["chi2_red"],
         "minus2lnl": final["minus2lnl"], "starts_at_best": chosen["starts_at_best"], "fit_standard": checks["fit_standard"]["detail"]["verdict"],
         "knowledge": "informed", "criteria_commit": criteria, "smoke": smoke}, indent=1) + "\n")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--task", type=int)
    ap.add_argument("--model", choices=sorted(FAMILIES))
    ap.add_argument("--data", default="~/data/imaging_contests")
    ap.add_argument("--out", default="contest_informed")
    ap.add_argument("--budget", type=int, default=None, help="at most this many stratified-geometry starts on the first rung, evenly spaced (default: 8)")
    ap.add_argument("--smoke", action="store_true", help="a few starts of 50 steps and a coarse map: check that it runs")
    ap.add_argument("--grey", action="store_true", help="one flux per component at every wavelength (the default gives each a spectral index when the data have more than one channel)")
    ap.add_argument("--spectra", action="store_true", help="also fit the files' OI_FLUX spectra (scale and linear trend marginalised)")
    ap.add_argument("--scan", action="store_true", help="start the shape parameters a rung adds that no map places from a likelihood grid")
    ap.add_argument("--no-profile-map", action="store_true", dest="no_profile", help="never re-optimise the rung below at each map position, even with a resolved primary")
    ap.add_argument("--no-probe", action="store_true", dest="no_probe", help="skip the CLEAN probe on top of the final fit")
    ap.add_argument("--image", action="store_true", help="the image step: a Gaussian-process image in place of the extended light, after the ladder")
    ap.add_argument("--no-gp-disk", action="store_true", dest="no_gp", help="skip the multipolar Gaussian-process disk of the disk families")
    ap.add_argument("--no-residual-image", action="store_true", dest="no_residual", help="skip the non-parametric image fitted on top of the chosen model")
    ap.add_argument("--scale-vis", action="store_true", dest="scale_vis", help="also divide the datasets named in VIS_SCALE by their fitted V^2 amplitude (the gain is marginalised without it)")
    ap.add_argument("--cache", default=None, help="keep every fit, grid and map tile here, so a call that pauses resumes")
    ap.add_argument("--time-budget", type=float, default=None, dest="time_budget", help="seconds after which no new fit or tile starts and the call exits with 75 (run it again to continue)")
    a = ap.parse_args(argv)
    if a.list:
        for name in FAMILIES:
            print(name + ": " + " -> ".join(r[0] for r in ladder(name)))
        return 0
    if a.task is None or a.model is None:
        ap.error("--task and --model are required")
    try:
        run(a.task, a.model, pathlib.Path(a.data).expanduser(), pathlib.Path(a.out).expanduser(), a.smoke, a.grey, a.spectra, not a.no_residual, a.no_gp,
            a.budget, a.scan, not a.no_probe, a.image, not a.no_profile, a.scale_vis, a.cache, a.time_budget)
    except Pause:
        print(f"time budget used ({a.time_budget} s): run again with the same --cache to continue", flush=True)
        return RC_MORE
    return 0


if __name__ == "__main__":
    sys.exit(main())
