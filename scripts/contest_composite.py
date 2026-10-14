"""Composite scenes: several Images, each with its own grid, support, offset
and prior, beside an analytic star (virgil design/composite_scenes.md).

    python scripts/contest_images.py --task N --scene "star=point + disk=image[hole=0.5] + planet=stamp"

``--scene`` takes this grammar when it contains ``=``; a plain comma list
(``point,image``) is still contest_images' single-image scene.

    SCENE := PART (" + " PART)*
    PART  := NAME "=" KIND ["[" KEY "=" VALUE ("," KEY "=" VALUE)* "]"]
    KIND  := point | ud | image | stamp

Lengths are in beams (the minor axis of ``virgil.imaging.beam``), so one
scene serves every coverage. At most one central star (``point``, or ``ud``:
a uniform disk with its diameter fitted). Keys of ``image`` and ``stamp``:

    prior  gp (default), or a penalty: mem, tsv, tv, starlet, logsum
    ell    GP correlation lengths, colon-separated: the evidence's grid
           (image default 0.5:1:2; stamp 0.25)
    sigma  GP σ values, colon-separated (image default 1:2:4; stamp 2)
    w      a penalty's weight relative to the other penalties' (default 1)
    field  the field as a multiple of the data-chosen one (image default 1)
    fov    the field in beams (overrides field; stamp default 2)
    pixel  the pixel in beams (default: the data-chosen pixel; stamp 1/8)
    hole   radius of a hole in the support at the image's centre (default
           0.5 for an image on a star; 0 for none)
    rmax   outer radius of the support (default: the whole grid)
    at     stamp position: "search" (default) or DRA:DDEC in mas

The first image takes setup()'s CLEAN start as its GP mean (and its field
and pixel unless given); the others start flat. Fluxes are relative to the
star, or to the first image without one: log-uniform (Jeffreys) between
FLUX_FLOOR and FLUX_CAP. Stamp offsets are uniform within half a beam of
their start.

How the scene is fitted (run_composite):
1. the scene without its stamps, GP hyperparameters at their grids' middles;
2. each GP image's (ℓ, σ) grid in turn, the others held, by the evidence
   with every GP image's latents integrated out and the Occam factor of the
   fluxes, offsets and the star (``log_evidence_all``);
3. the penalties, if any, as one ``Joint`` penalty whose overall factor is
   swept and chosen by the discrepancy principle (the corner if the sweep
   never reaches χ²/N = 1), their ratios fixed by ``w``;
4. each stamp at "search": candidates from ``marginal_flux_map`` (a point
   added to the scene with the GP images' latents marginalized, so that flux
   a GP has absorbed still shows), each fitted with its offset free, the one
   of largest evidence kept if it raises log Z by ln 100;
5. MacKay's error scale over all GP images, and a refit on rescaled errors
   when it is outside 1/1.3-1.3.
The evidence of a scene with penalties is conditional on the penalised
images (held at their fit). Raw χ²/N on the quoted errors is reported first.
"""

import dataclasses
import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

import contest_images as ci
import virgil.models as vm
from virgil.fields import GaussianField
from virgil.fitting import fit
from virgil.imaging import error_scale, image_priors, log_evidence

KINDS = ("point", "ud", "image", "stamp")
PRIORS = ("gp",) + tuple(ci.REGULARISERS)
KEYS = ("prior", "ell", "sigma", "w", "field", "fov", "pixel", "hole", "rmax", "at")
IMAGE_DEFAULTS = {"prior": "gp", "ell": (0.5, 1.0, 2.0), "sigma": (1.0, 2.0, 4.0), "w": 1.0, "field": 1.0,
                  "fov": None, "pixel": None, "hole": None, "rmax": None, "at": None}
STAMP_DEFAULTS = IMAGE_DEFAULTS | {"ell": (0.25,), "sigma": (2.0,), "fov": 2.0, "pixel": 0.125, "hole": 0.0,
                                   "at": "search"}
STAMP_LOG_BAYES = float(np.log(100.0))  # a stamp is kept when it raises log Z by this
STAMP_CANDIDATES = 3  # search peaks tried as stamp positions, at least a beam apart
SEARCH_STEP = 1.0 / 3.0  # search grid step, in beams
JOINT_POINTS = 9  # the penalties' sweep: factors 10^-2 .. 10^2 of their default weights
CHUNK = 64  # search positions per batched Jacobian-vector product


@dataclasses.dataclass(frozen=True)
class Part:
    """One part of a composite scene."""

    name: str
    kind: str
    opts: dict


def _floats(text):
    return tuple(float(x) for x in str(text).split(":"))


def parse_scene(text):
    """``--scene``'s composite grammar (the module docstring) as Parts, star first."""
    parts = []
    for chunk in (c.strip() for c in text.split("+")):
        if not chunk:
            raise ValueError(f"empty part in --scene {text!r}")
        head, _, rest = chunk.partition("[")
        name, eq, kind = (x.strip() for x in head.partition("="))
        if not eq or not name.isidentifier() or kind not in KINDS:
            raise ValueError(f"--scene part {chunk!r}: want NAME=KIND with KIND one of {KINDS}")
        if rest and not rest.endswith("]"):
            raise ValueError(f"--scene part {chunk!r}: unclosed [")
        opts = {}
        for item in filter(None, (x.strip() for x in rest[:-1].split(","))) if rest else ():
            key, eq, value = (x.strip() for x in item.partition("="))
            if not eq or key not in KEYS:
                raise ValueError(f"--scene {name}: option {item!r}; keys are {KEYS}")
            opts[key] = value
        if kind in ("point", "ud") and opts:
            raise ValueError(f"--scene {name}: a {kind} star takes no options")
        defaults = STAMP_DEFAULTS if kind == "stamp" else IMAGE_DEFAULTS
        if kind in ("image", "stamp"):
            out = dict(defaults)
            for key, value in opts.items():
                if key == "prior":
                    if value not in PRIORS:
                        raise ValueError(f"--scene {name}: prior {value!r}, not one of {PRIORS}")
                    out[key] = value
                elif key in ("ell", "sigma"):
                    out[key] = _floats(value)
                elif key == "at":
                    out[key] = value if value == "search" else _floats(value)
                    if out[key] != "search" and len(out[key]) != 2:
                        raise ValueError(f"--scene {name}: at=DRA:DDEC (mas) or at=search")
                else:
                    out[key] = float(value)
            if kind == "image" and out["at"] is not None:
                raise ValueError(f"--scene {name}: at= is for stamps (an image is centred on the star)")
            opts = out
        parts.append(Part(name, kind, opts))
    names = [p.name for p in parts]
    if len(set(names)) != len(names):
        raise ValueError(f"--scene: repeated names {names}")
    stars = [p for p in parts if p.kind in ("point", "ud")]
    if len(stars) > 1:
        raise ValueError("--scene: at most one central star (point or ud)")
    if not any(p.kind == "image" for p in parts):
        raise ValueError("--scene needs at least one image")
    return stars + [p for p in parts if p.kind not in ("point", "ud")]


def scene_tag(parts):
    """The label suffix of a composite scene: _c-<name><kind initial>..."""
    return "_c-" + "-".join(f"{p.name}{p.kind[0]}" + ("" if p.kind in ("point", "ud") or p.opts["prior"] == "gp"
                                                       else p.opts["prior"]) for p in parts)


def _plain(x):
    """JSON for NumPy and JAX scalars and arrays."""
    return np.asarray(x).tolist()


def is_composite(text):
    return text is not None and "=" in text


def odd_grid(fov, pixel, max_pix):
    """Odd side and pixel for a field ``fov`` at ``pixel`` (coarser if the side exceeds ``max_pix``)."""
    n = int(np.ceil(fov / pixel)) | 1
    if n > max_pix:
        n = max_pix - 1 if max_pix % 2 == 0 else max_pix
        pixel = fov / n
    return n, pixel


def support_mask(n, pixel, hole, rmax):
    """The support of an ``n``-pixel grid: radius at least ``hole`` and at most ``rmax`` (mas; None: no limit)."""
    x = (np.arange(n) - (n - 1) / 2) * pixel
    r = np.hypot(x[None, :], x[:, None])
    keep = np.ones((n, n), bool)
    if hole:
        keep &= r >= hole
    if rmax:
        keep &= r <= rmax
    if keep.all():
        return None
    if not keep.any():
        raise ValueError(f"empty support: hole {hole} mas, rmax {rmax} mas on a {n} x {pixel:.3g} mas grid")
    return jnp.asarray(keep)


def resample(image, n):
    """``image`` resampled to ``n`` x ``n`` (bilinear), floored at 10⁻⁶ of its
    peak so that its log is finite (a penalised image's start)."""
    from scipy import ndimage

    image = np.asarray(image, float)
    if image.shape[0] != n:
        image = ndimage.zoom(image, n / image.shape[0], order=1)[:n, :n]
        image = np.pad(image, [(0, n - image.shape[0]), (0, n - image.shape[1])], mode="edge")
    return np.clip(image, 1e-6 * max(float(image.max()), 1e-300), None)


class Joint(eqx.Module):
    """Several penalties as one, ``weight × Σ member.value``: their ratios
    are the members' own weights, and ``l_curve`` sweeps ``weight``. It has
    least-squares residuals when every member does."""

    weight: jax.Array
    members: tuple

    def value(self, model):
        return self.weight * sum(m.value(model) for m in self.members)

    def __getattr__(self, name):
        if name == "residuals" and all(hasattr(m, "residuals") for m in self.members):
            return lambda model: jnp.sqrt(self.weight) * jnp.concatenate(
                [jnp.ravel(m.residuals(model)) for m in self.members])
        raise AttributeError(name)


def penalty_weight(prior, data, w):
    """A penalty's default weight times ``w``: MaxEnt's middle weight, or the
    geometric middle of WEIGHT_RANGES x N (contest_images.regulariser_weights)."""
    weights = ci.regulariser_weights(prior, data)
    return float(w) * float(np.sqrt(weights[0] * weights[-1]))


@dataclasses.dataclass
class Scene:
    """A composite scene being fitted: parts, the model, priors, GP settings."""

    parts: list
    model: object
    priors: dict
    gp: dict  # image name -> (ℓ in mas, σ)
    penalties: dict  # image name -> penalty at its default weight
    others: tuple = ()
    clusters: tuple = ()  # CLEAN's compact clusters (dra, ddec, flux), cut from the first image's mean

    def gp_paths(self):
        return tuple(self.gp)

    def with_field(self, name, ell, sigma):
        """The scene with image ``name``'s GP at (ℓ mas, σ), its latents and mean kept."""
        image = self.model.components[name]
        field = image.log_brightness
        image = eqx.tree_at(lambda im: im.log_brightness, image,
                            GaussianField(field.latent, sigma, ell, mean=field.mean))
        model = vm.System(**(self.model.components | {name: image}))
        return dataclasses.replace(self, model=model, gp=self.gp | {name: (ell, sigma)})


def clean_clusters(s, k=STAMP_CANDIDATES):
    """Up to ``k`` of CLEAN's compact clusters outside the main source
    (contest_images.clean_point_candidates) as (dra, ddec, flux), the flux
    relative to the reference component as the start's; () without CLEAN."""
    cleaned = s.get("cleaned")
    if cleaned is None:
        return ()
    total = max(float(np.sum(np.clip(np.asarray(cleaned.components), 0, None))), 1e-12)
    found = ci.clean_point_candidates(cleaned.components, cleaned.restored(s["restore"]),
                                      float(cleaned.pixel_scale_mas), s["resolution"], k)
    scale = ci.flux_value(s["img0"].flux) / total
    return tuple((c["dra"], c["ddec"], c["flux"] * scale) for c in found)


def cut_clusters(template, pixel, clusters, radius):
    """``template`` (brightness) with each cluster's disk of ``radius`` mas
    lowered to 10⁻³ of its peak, so that a stamp, not the image's mean,
    starts with that flux."""
    n = template.shape[0]
    dra, ddec = ci.pixel_offset(*np.mgrid[:n, :n], n, pixel)
    out = np.array(template, float)
    for ra, de, _ in clusters:
        under = np.hypot(dra - ra, ddec - de) < radius
        out[under] = np.minimum(out[under], 1e-3 * out.max())
    return out


def build_scene(parts, s, smoke=False):
    """The composite scene's model and priors from setup ``s`` (its star, CLEAN
    start, grid and beam). Stamps at "search" are left out (added later)."""
    res, data = s["resolution"], s["data"]
    beam = float(res.minor_mas)
    npts = s["npts"]
    img0, n0, pixel0 = s["img0"], s["npix"], s["pixel"]
    star_part = next((p for p in parts if p.kind in ("point", "ud")), None)
    star = star_part is not None and s["star"]
    comps, priors, gp, penalties = {}, {}, {}, {}
    if star:
        comps[star_part.name] = s["start"].star
        priors |= {k.replace("star.", f"{star_part.name}.", 1): v for k, v in s["star_priors"].items()}
    images = [p for p in parts if p.kind == "image"]
    searched = any(p.kind == "stamp" and p.opts["at"] == "search" for p in parts)
    clusters = clean_clusters(s) if searched else ()
    budget = int(ci.JACOBIAN_BUDGET / max(npts, 1))  # pixels over all images
    first = images[0].name
    for p in parts:
        if p.kind not in ("image", "stamp") or p.kind == "stamp" and p.opts["at"] == "search":
            continue
        o = p.opts
        fov = o["fov"] * beam if o["fov"] else o["field"] * n0 * pixel0
        pixel = o["pixel"] * beam if o["pixel"] else pixel0
        n, pixel = odd_grid(fov, pixel, min(ci.MAX_PIX, int(np.sqrt(budget / len(parts)))))
        hole = o["hole"] if o["hole"] is not None else (0.5 if star and p.kind == "image" else 0.0)
        support = support_mask(n, pixel, hole * beam, (o["rmax"] or 0) * beam or None)
        if p.name == first:
            template = cut_clusters(resample(np.asarray(img0.brightness, float).reshape(n0, n0), n), pixel,
                                    clusters, 0.5 * beam)
        else:
            template = None
        flux = (float(ci.flux_value(img0.flux)) if p.name == first else 0.1 * max(ci.flux_value(img0.flux), 0.1)) \
            if star else (1.0 if p.name == first else 0.1)
        dra, ddec = (0.0, 0.0) if o["at"] in (None, "search") else o["at"]
        if o["prior"] == "gp":
            ell, sigma = o["ell"][len(o["ell"]) // 2] * beam, o["sigma"][len(o["sigma"]) // 2]
            lb = GaussianField(np.zeros((n, n)), sigma, min(ell, n * pixel), mean=template)
            gp[p.name] = (ell, sigma)
        else:
            lb = jnp.log(jnp.asarray(template if template is not None else np.ones((n, n))))
            penalties[p.name] = ci.regulariser(o["prior"], penalty_weight(o["prior"], data, o["w"]), p.name)
        comps[p.name] = vm.Image(lb, pixel, support=support, flux=flux, dra=dra, ddec=ddec)
        if star or p.name != first:
            priors[f"{p.name}.flux"] = dist.LogUniform(ci.FLUX_FLOOR, ci.FLUX_CAP)
            comps[p.name] = eqx.tree_at(lambda im: im.flux, comps[p.name], ci.inside(priors[f"{p.name}.flux"], flux))
        if p.kind == "stamp":
            priors |= offset_priors(p.name, dra, ddec, beam)
    model = vm.System(**comps)
    priors = image_priors(model) | priors
    others = () if star else (ci.Centroid(0.1 * beam, path=first),)
    return Scene(list(parts), model, priors, gp, penalties, others, clusters)


def offset_priors(name, dra, ddec, beam):
    """Uniform (a location) within half a beam of a stamp's start."""
    return {f"{name}.dra": dist.Uniform(dra - 0.5 * beam, dra + 0.5 * beam),
            f"{name}.ddec": dist.Uniform(ddec - 0.5 * beam, ddec + 0.5 * beam)}


def regularisers(sc, factor):
    if not sc.penalties:
        return sc.others
    return (Joint(jnp.asarray(float(factor)), tuple(sc.penalties.values())), *sc.others)


def fit_scene(sc, data, factor=1.0, init=None, **options):
    values = None if init is None else {k: v for k, v in init.items() if k in sc.priors}
    r = fit(sc.model, sc.priors, data, regularisers(sc, factor), init=values, **options)
    return dataclasses.replace(sc, model=r.model), r


def _latent_paths(sc):
    return tuple(f"{name}.log_brightness.latent" for name in sc.gp)


def _flux_path(sc):
    """Some scalar leaf of the scene (for the residuals alone)."""
    return next(f"{n}.flux" for n, c in sc.model.components.items() if isinstance(c, vm.Image))


def log_evidence_all(sc, result, data):
    """Laplace log evidence of a composite fit: every GP image's latents
    integrated out together (``log_evidence(path=<all GP images>)``), then
    the Occam factor of the other parameters (fluxes, offsets, the star) in
    the coordinates in which their Jeffreys priors are flat, with the
    curvature marginalized over the latents (as contest_images.log_evidence_full
    does for one image). Penalised images are held at their fit. Without a
    GP image the first term is log_evidence's with no latents, -½χ² minus
    the likelihood's normalization."""
    from virgil.imaging import _log_norm, _residual_jacobian  # private: log_evidence's own pieces

    model = result.model
    if sc.gp:
        log_z = float(log_evidence(result, data, path=sc.gp_paths()))
    else:
        r, _ = _residual_jacobian(model, data, _flux_path(sc))
        log_z = float(-0.5 * r @ r - _log_norm(model, data))
    rest = {k: v for k, v in sc.priors.items() if isinstance(v, (dist.Uniform, dist.LogUniform))}
    if not rest:
        return log_z, 0.0
    if sc.gp:
        _, jac_z = _residual_jacobian(model, data, _latent_paths(sc))
        u, sv, _ = np.linalg.svd(np.asarray(jac_z).reshape(jac_z.shape[0], -1), full_matrices=False)
    else:
        u, sv = np.zeros((len(r), 0)), np.zeros(0)
    columns, ranges = [], []
    for name, prior in rest.items():
        _, jac = _residual_jacobian(model, data, name)
        lo, hi, log = ci._interior(prior, 0.0)
        if log:
            jac = jac * float(np.ravel(model.get(name))[0])
        columns.append(np.asarray(jac).reshape(jac.shape[0], -1))
        ranges.append(hi - lo)
    jac = np.concatenate(columns, axis=1)
    projected = u.T @ jac
    curvature = jac.T @ jac - projected.T @ ((sv**2 / (1.0 + sv**2))[:, None] * projected)
    curvature += np.diag(12.0 / np.square(ranges))
    _, logdet = np.linalg.slogdet(curvature)
    occam = float(-np.sum(np.log(ranges)) + 0.5 * len(ranges) * np.log(2 * np.pi) - 0.5 * logdet)
    return log_z + occam, occam


def marginal_flux_map(sc, data, positions):
    """Linearized flux and SNR of a point added to ``sc.model`` at each of
    ``positions`` ((n, 2) mas, East and North), against the GP images'
    prior predictive: with their latents z at the prior mean (0, so each
    image at its mean template; fluxes, star and offsets as fitted), the
    whitened residuals are r(z) ≈ r₀ + J z with z ~ N(0, I), so they have
    covariance C = I + J Jᵀ, and a point of flux f and tangent g is

        f = -gᵀC⁻¹r₀ / gᵀC⁻¹g,  SNR = f (gᵀC⁻¹g)^½

    (C⁻¹ by Woodbury on J's SVD; g by forward mode, as linear_flux_grid).
    This is exact for a linear model, where it is also the search on the
    fitted residual r* = C⁻¹r₀. Searching the fitted scene instead, with
    the latents at their fit, finds nothing a GP was free enough to absorb
    (design/composite_scenes.md). With no GP image it is virgil's
    ``linear_flux_grid`` on ``sc.model``. Returns (flux, snr) arrays."""
    from virgil.imaging import _residual_jacobian
    from virgil.likelihood import whitened_residuals

    datasets = data if isinstance(data, list) else [data]
    model = sc.model
    for path in _latent_paths(sc):
        model = model.set(path, jnp.zeros_like(model.get(path)))
    if sc.gp:
        mean, jac = _residual_jacobian(model, data, _latent_paths(sc))
        u, sv, _ = np.linalg.svd(np.asarray(jac).reshape(jac.shape[0], -1), full_matrices=False)
        shrink = sv**2 / (1.0 + sv**2)
    else:
        mean = np.concatenate([np.asarray(whitened_residuals(model, d), float) for d in datasets])
        u, shrink = np.zeros((mean.size, 0)), np.zeros(0)

    def residuals(f, dra, ddec):
        m = vm.System(**model.components, cand=vm.PointSource(f, dra=dra, ddec=ddec))
        return jnp.concatenate([whitened_residuals(m, d) for d in datasets])

    @jax.jit
    def tangents(dra, ddec):
        return jax.vmap(lambda a, b: jax.jvp(lambda f: residuals(f, a, b), (0.0,), (1.0,))[1])(dra, ddec)

    pos = np.asarray(positions, float)
    flux, snr = np.zeros(len(pos)), np.zeros(len(pos))
    for start in range(0, len(pos), CHUNK):
        chunk = pos[start : start + CHUNK]
        pad = np.concatenate([chunk, np.repeat(chunk[-1:], CHUNK - len(chunk), 0)])  # one compiled shape
        g = np.asarray(tangents(jnp.asarray(pad[:, 0]), jnp.asarray(pad[:, 1])), float)[: len(chunk)]
        cg = g - (g @ u) * shrink @ u.T  # (I + J Jᵀ)⁻¹ g, by Woodbury on J's SVD
        gg = np.einsum("ij,ij->i", g, cg)
        with np.errstate(invalid="ignore", divide="ignore"):  # g = 0 at the phase centre
            f = -(cg @ mean) / gg
        flux[start : start + len(chunk)], snr[start : start + len(chunk)] = f, f * np.sqrt(gg)
    return flux, snr


def search_candidates(sc, data, s, k=STAMP_CANDIDATES):
    """The ``k`` highest-SNR positive peaks of ``marginal_flux_map``, at least a
    beam apart, over the largest image's field, outside the star's hole, and
    then CLEAN's clusters (``sc.clusters``) not within a beam of those (at
    the map's flux there, or CLEAN's when that is not positive). Returns
    [(dra, ddec, flux, snr)]."""
    beam = float(s["resolution"].minor_mas)
    half = 0.5 * max(im.pixel_scale_mas * im.log_brightness.shape[0] for im in sc.model.parts
                     if isinstance(im, vm.Image))
    axis = np.arange(-half, half + 1e-9, SEARCH_STEP * beam)
    dra, ddec = (x.ravel() for x in np.meshgrid(axis, axis))
    keep = np.hypot(dra, ddec) >= (0.5 * beam if s["star"] else 0.0)
    clusters = np.array([c[:2] for c in sc.clusters], float).reshape(-1, 2)
    pos = np.concatenate([np.stack([dra[keep], ddec[keep]], 1), clusters])
    flux, snr = marginal_flux_map(sc, data, pos)
    grid = len(pos) - len(clusters)
    peaks = []
    for i in np.argsort(-np.nan_to_num(snr[:grid], nan=-np.inf)):
        if not flux[i] > 0:
            break
        if all(np.hypot(*(pos[i] - (a, b))) >= beam for a, b, _, _ in peaks):
            peaks.append((float(pos[i, 0]), float(pos[i, 1]), float(flux[i]), float(snr[i])))
        if len(peaks) == k:
            break
    for j, (a, b, f_clean) in enumerate(sc.clusters):
        if all(np.hypot(a - x, b - y) >= beam for x, y, _, _ in peaks):
            f, z = float(flux[grid + j]), float(snr[grid + j])
            peaks.append((float(a), float(b), f if f > 0 else float(f_clean), z))
    return peaks


def add_stamp(sc, part, s, dra, ddec, flux):
    """``sc`` with ``part`` (a stamp) at (dra, ddec) mas, its flux started at ``flux``."""
    o, beam = part.opts, float(s["resolution"].minor_mas)
    n, pixel = odd_grid(o["fov"] * beam, o["pixel"] * beam, ci.MAX_PIX)
    support = support_mask(n, pixel, o["hole"] * beam, (o["rmax"] or 0) * beam or None)
    prior = dist.LogUniform(ci.FLUX_FLOOR, ci.FLUX_CAP)
    start = ci.inside(prior, max(flux, ci.FLUX_FLOOR))  # the map's flux is relative to the reference component's
    if o["prior"] == "gp":
        ell, sigma = o["ell"][0] * beam, o["sigma"][0]
        lb = GaussianField(np.zeros((n, n)), sigma, ell)
        gp = sc.gp | {part.name: (ell, sigma)}
        penalties = sc.penalties
    else:
        lb = jnp.zeros((n, n))
        gp = sc.gp
        penalties = sc.penalties | {part.name: ci.regulariser(o["prior"], penalty_weight(o["prior"], s["data"], o["w"]),
                                                               part.name)}
    image = vm.Image(lb, pixel, support=support, flux=start, dra=dra, ddec=ddec)
    model = vm.System(**sc.model.components, **{part.name: image})
    priors = sc.priors | image_priors(vm.System(**{part.name: image})) | {f"{part.name}.flux": prior} | offset_priors(
        part.name, dra, ddec, beam)
    return dataclasses.replace(sc, model=model, priors=priors, gp=gp, penalties=penalties)


def select_gp(sc, s, data, factor, smoke, options, record):
    """Each GP image's (ℓ, σ) grid in turn by the evidence, the others held."""
    beam = float(s["resolution"].minor_mas)
    result = None
    for part in sc.parts:
        if part.name not in sc.gp or part.kind != "image":
            continue
        o = part.opts
        ells = o["ell"][len(o["ell"]) // 2 :][:1] if smoke else o["ell"]
        sigmas = o["sigma"][len(o["sigma"]) // 2 :][:1] if smoke else o["sigma"]
        image = sc.model.components[part.name]
        cap = image.pixel_scale_mas * image.log_brightness.shape[0]
        grid, init = {}, None if result is None else result.values
        for ell in ells:
            for sigma in sigmas:
                trial = sc.with_field(part.name, min(ell * beam, cap), sigma)
                trial, r = fit_scene(trial, data, factor, init, **options)
                init = ci.off_bounds(r.values, trial.priors)
                log_z, _ = log_evidence_all(trial, r, data)
                grid[ell, sigma] = (log_z, trial, r)
                print(f"{part.name}: ℓ={ell:g} beams σ={sigma:g}: log Z {log_z:.2f} χ²/N {ci.chi2_red(r.model, data):.4g}",
                      flush=True)
        key = max(grid, key=lambda k: grid[k][0] if np.isfinite(grid[k][0]) else -np.inf)
        record[part.name] = {"grid": {f"{a:g}/{b:g}": v[0] for (a, b), v in grid.items()}, "ell_beams": key[0],
                             "sigma": key[1]}
        _, sc, result = grid[key]
    return sc, result


def select_factor(sc, data, smoke, options, record):
    """The penalties' common factor, by the discrepancy principle on a sweep
    (strong to weak), else the corner; their ratios stay fixed."""
    from virgil.imaging import LCurve

    factors = np.logspace(2.0, -2.0, 3 if smoke else JOINT_POINTS)
    penalty = Joint(jnp.asarray(1.0), tuple(sc.penalties.values()))
    curve = ci.sweep(sc.model, sc.priors, data, penalty, factors, sc.others, **options)
    ok = ci.finite_points(curve)
    if not ok:
        record["penalties"] = {"factors": curve.weights.tolist(), "chosen": 1.0, "how": "none finite",
                               "members": list(sc.penalties)}
        print(f"penalties {list(sc.penalties)}: no finite fit in the sweep; factor 1", flush=True)
        return 1.0
    curve = LCurve(curve.weights[ok], curve.chi2[ok], curve.chi2_red[ok], curve.penalty[ok],
                   [curve.results[i] for i in ok])
    chosen = curve.discrepancy()
    how = "discrepancy"
    if chosen is None:
        chosen, how = (curve.corner(), "corner") if len(ok) >= 3 else (float(curve.weights[-1]), "weakest")
    record["penalties"] = {"factors": curve.weights.tolist(), "chi2_red": np.asarray(curve.chi2_red).tolist(),
                           "chosen": chosen, "how": how, "members": list(sc.penalties)}
    print(f"penalties {list(sc.penalties)}: factor {chosen:.3g} ({how})", flush=True)
    return chosen


def run_composite(task, data_dir, out_dir, parts, smoke=False, label=None, field=None, informed=False, settings=None):
    """Fit the composite scene ``parts`` (parse_scene) to one task (or a
    contest_bench spec with ``settings``) and write <label>.npz/.txt/.png
    in contest_images' format (``ref_image`` on the common reference grid)."""
    t0 = time.time()
    star_part = next((p for p in parts if p.kind in ("point", "ud")), None)
    star_model = {"point": "point", "ud": "disk", None: "point"}[star_part and star_part.kind]
    opts = dict(star=star_part is not None, star_model=star_model, tag=scene_tag(parts))
    if settings is not None:  # contest_bench: the arm's fixed configuration (its star from the scene)
        settings = dict(settings) | {"star": opts["star"], "star_model": star_model}
        s, settings = ci.member_setup(task, data_dir, None, smoke, settings)
        growth = []
    else:  # setup's hole only shapes the CLEAN start; build_scene gives each image its own
        s, growth = ci.grown_setup(task, data_dir, False, opts.pop("star"), "clean", smoke, field, **opts)
        s["ref_fov"] = ci.reference_fov(task, data_dir)
    label = label or s["label"]  # setup appends the scene tag
    data = s["data"]
    options = {"max_steps": 50} if smoke else {}
    record = {"star_rejected": s.get("star_rejected", ""), "growth": growth}
    sc = build_scene(parts, s, smoke)
    factor = 1.0
    sc, result = fit_scene(sc, data, factor, **options)
    if sc.gp:
        sc, result = select_gp(sc, s, data, factor, smoke, options, record)
    if sc.penalties:
        factor = select_factor(sc, data, smoke, options, record)
        sc, result = fit_scene(sc, data, factor, result.values, **options)
    log_z, occam = log_evidence_all(sc, result, data)
    record["stamps"] = {}
    for part in (p for p in parts if p.kind == "stamp" and p.opts["at"] == "search"):
        peaks = search_candidates(sc, data, s)
        print(f"{part.name}: search peaks (dra, ddec, flux, SNR) {[tuple(round(x, 3) for x in p) for p in peaks]}",
              flush=True)
        tried = []
        for dra, ddec, flux, snr in peaks:
            trial = add_stamp(sc, part, s, dra, ddec, flux)
            trial, r = fit_scene(trial, data, factor, result.values, **options)
            lz, _ = log_evidence_all(trial, r, data)
            tried.append((lz, trial, r, (dra, ddec, flux, snr)))
            print(f"{part.name} at ({dra:.3g}, {ddec:.3g}) mas: log Z {lz} (base {log_z}) "
                  f"χ²/N {ci.chi2_red(r.model, data):.4g}", flush=True)
        best = max(tried, key=lambda t: t[0] if t[0] is not None and np.isfinite(t[0]) else -np.inf, default=None)
        kept = best is not None and log_z is not None and best[0] is not None and best[0] - log_z >= STAMP_LOG_BAYES
        record["stamps"][part.name] = {"peaks": peaks, "log_z": [t[0] for t in tried], "base_log_z": log_z,
                                       "kept": bool(kept)}
        if kept:
            log_z, sc, result = best[0], best[1], best[2]
            st = sc.model.components[part.name]
            record["stamps"][part.name] |= {"dra": float(st.dra), "ddec": float(st.ddec),
                                            "flux": ci.flux_value(st.flux)}
    scale, data_s = 1.0, data
    if sc.gp:
        scale = float(error_scale(result.model, data, path=sc.gp_paths()))
        if not 1 / 1.3 < scale < 1.3:
            datasets = data if isinstance(data, list) else [data]
            data_s = [d.with_error_scale(scale) for d in datasets]
            data_s = data_s if isinstance(data, list) else data_s[0]
            sc, result = fit_scene(sc, data_s, factor, result.values, **options)
    log_z, occam = log_evidence_all(sc, result, data_s)
    final = result.model
    raw = ci.chi2_red(final, data)  # quoted errors, first
    write_outputs(task, label, s, sc, final, raw, scale, log_z, occam, record, out_dir, informed, settings, t0)
    return final


def write_outputs(task, label, s, sc, final, raw, scale, log_z, occam, record, out_dir, informed, settings, t0):
    import matplotlib.pyplot as plt

    data, res = s["data"], s["resolution"]
    ref_fov = s["ref_fov"]
    q_max = ci.data_q_max(data)
    n_ref = ci.reference_npix(ref_fov, q_max)
    names = [n for n in final.names if isinstance(final.components[n], vm.Image)]
    ref = ci.render_scene(final, n_ref, ref_fov, q_max)
    fluxes = {n: ci.flux_value(getattr(c, "flux", 1.0)) for n, c in final.components.items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = dict(ref_image=ref, ref_fov=ref_fov, best_chi2_red=raw, raw_chi2_red=raw, error_scale=scale,
                  best_log_z=np.nan if log_z is None else log_z, occam=np.nan if occam is None else occam,
                  star=bool(s["star"]), star_rejected=record["star_rejected"], informed=bool(informed),
                  fluxes=json.dumps(fluxes), record=json.dumps(record, default=_plain),
                  parts=json.dumps([dataclasses.asdict(p) for p in sc.parts], default=str),
                  beam=np.array([res.major_mas, res.minor_mas, res.pa_deg]))
    for n in names:
        im = final.components[n]
        arrays[f"image_{n}"] = np.asarray(im.brightness)
        arrays[f"pixel_{n}"] = float(im.pixel_scale_mas)
        arrays[f"offset_{n}"] = np.array([float(im.dra), float(im.ddec)])
    tmp = out_dir / f".{label}.npz.part"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(out_dir / f"{label}.npz")
    text = (f"raw chi2/N={raw:.4g} (quoted errors) task={task if isinstance(task, int) else s['label']} label={label}\n"
            f"{ci.informed_line(informed)}\nscene={[f'{p.name}={p.kind}' for p in sc.parts]} star={s['star']} "
            f"star_rejected={record['star_rejected']!r}\n"
            f"error_scale={scale:.4g} log_z={log_z} occam={occam}\nfluxes={fluxes}\n"
            f"gp={ {k: (round(v[0], 4), v[1]) for k, v in sc.gp.items()} } penalties={list(sc.penalties)}\n"
            f"record={json.dumps(record, default=_plain)}\n"
            f"residuals={ci.residual_diagnostics(final, data)}\nelapsed={time.time() - t0:.0f}s\n")
    (out_dir / f"{label}.txt").write_text(text)
    print(text)
    fig, axes = plt.subplots(1, len(names) + 1, figsize=(4.4 * (len(names) + 1), 4.4))
    extent = [ref_fov / 2, -ref_fov / 2, -ref_fov / 2, ref_fov / 2]
    axes[0].imshow(ref, origin="upper", extent=extent, cmap="inferno")
    axes[0].set(title=f"{label}\nχ²/N {raw:.3g}", xlabel="ΔRA (mas)", ylabel="ΔDec (mas)")
    for ax, n in zip(axes[1:], names):
        im = final.components[n]
        half = 0.5 * im.pixel_scale_mas * im.log_brightness.shape[0]
        ax.imshow(np.asarray(im.brightness), origin="upper", cmap="inferno",
                  extent=[im.dra + half, im.dra - half, im.ddec - half, im.ddec + half])
        ax.set(title=f"{n}: flux {fluxes[n]:.3g}", xlabel="ΔRA (mas)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{label}.png", dpi=90)
    plt.close(fig)
