"""virgil.radio on the ALMA files of ``external_bridge.alma_worker`` (the only
module here that imports virgil for radio data).

``load_npz`` turns W2's one-file-per-DATA_DESC_ID ``.npz`` files into a
``ComplexVisData``; the scene builders pair numpy-made truth images with the
virgil ``Image`` that fits them. Nothing here is an instrument default: the
caller passes the pixel scale, the dish and so on.
"""

import os
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
C = 299792458.0
ARCSEC = np.pi / 180.0 / 3600.0

# casatools finds its measures data through this file, written by
# scripts/setup_external.sh alma. Set it before the first casatools import
# (virgil.radio imports casatools lazily) so that a second casatools, the one
# of the virgil venv, shares .venv-alma's measures data.
_site = ROOT / ".venv-alma" / "casasiteconfig.py"
if "CASASITECONFIG" not in os.environ and _site.exists():
    os.environ["CASASITECONFIG"] = str(_site)


def load_npz(zs, ddids=None):
    """A ``ComplexVisData`` from W2's per-ddid dicts (or ``.npz`` paths).

    The reader of W2 has already conjugated ``DATA``; ``re``/``im`` (or a complex ``vis``, as the
    full-data files of the OzSTAR read have) are V. ``ddid`` and ``obs`` are the MS's own ids, as
    virgil's ``from_arrays`` wants them. ``ddids`` keeps only those ddids; the optional columns
    (``time``, ``ant1``, ``ant2``, ``chan``) are passed on only when every file has them.
    """
    from virgil.radio import ComplexVisData

    zs = [dict(np.load(z)) if isinstance(z, (str, pathlib.Path)) else z for z in zs]
    if ddids is not None:
        zs = [z for z in zs if int(z["ddid"]) in set(ddids)]
    cat = lambda k: np.concatenate([np.asarray(z[k]) for z in zs])  # noqa: E731
    n = [len(z["u"]) for z in zs]
    vis = np.concatenate([np.asarray(z["vis"]) if "vis" in z else np.asarray(z["re"], float) + 1j * np.asarray(z["im"], float) for z in zs])
    opt = {k: cat(k) for k in ("time", "ant1", "ant2", "chan") if all(k in z for z in zs)}
    return ComplexVisData.from_arrays(
        cat("u"), cat("v"), vis.astype(complex), cat("weight").astype(float), cat("freq"),
        ddid=np.repeat([int(z["ddid"]) for z in zs], n), obs=np.repeat([int(z["obsid"]) for z in zs], n), **opt,
    )


def ddid_files(directory):
    """The ``ddid_NN.npz`` files of a directory, sorted."""
    files = sorted(pathlib.Path(directory).expanduser().glob("ddid_*.npz"))
    if not files:
        raise FileNotFoundError(f"no ddid_*.npz in {directory}")
    return files


def load_dir(directory, ddids=None):
    """``load_npz`` of every ``ddid_NN.npz`` in ``directory``."""
    return load_npz(ddid_files(directory), ddids)


def image_scene(pixel_scale_mas, site="log_brightness"):
    """``scene(values)`` of a free pixel image, for ``data.term``."""
    from virgil import Image

    return lambda values: Image(values[site], pixel_scale_mas=pixel_scale_mas)


def fixed_image_scene(image, pixel_scale_mas, floor=1e-12):
    """``scene(values)`` of an image held at a numpy brightness (rows North
    first, columns East first), with no free pixels: the nuisance fits."""
    import jax.numpy as jnp

    from virgil import Image

    eta = jnp.log(np.maximum(np.asarray(image, float), floor * np.max(image)))
    return lambda values: Image(eta, pixel_scale_mas=pixel_scale_mas)


def point_scene(sources, pixel_scale_mas=None):
    """``scene(values)`` of fixed point sources ``[(flux, east, north)]``
    (flux relative; ``cal.flux`` carries the total), in mas."""
    from virgil import PointSource, System

    tot = sum(f for f, _, _ in sources)
    return lambda values: System(**{f"p{i}": PointSource(flux=f / tot, dra=e, ddec=n) for i, (f, e, n) in enumerate(sources)})


def ring_and_blob(npix=48, pixel_mas=25.0, radius_mas=300.0, width_mas=60.0, inc_deg=45.0, pa_deg=30.0,
                  blob=(0.2, 150.0, -250.0, 70.0)):
    """The IM Lup stand-in scene in numpy: an inclined Gaussian ring plus an
    asymmetric Gaussian blob (fraction of the flux, East, North, sigma in mas).
    Rows North first, columns East first, centred on ((n-1)/2, (n-1)/2), unit sum."""
    k = ((npix - 1) / 2.0 - np.arange(npix)) * pixel_mas
    east, north = np.meshgrid(k, k)
    pa, inc = np.deg2rad(pa_deg), np.deg2rad(inc_deg)
    major = east * np.sin(pa) + north * np.cos(pa)
    minor = (east * np.cos(pa) - north * np.sin(pa)) / np.cos(inc)
    ring = np.exp(-0.5 * ((np.hypot(major, minor) - radius_mas) / width_mas) ** 2)
    frac, be, bn, bs = blob
    spot = np.exp(-0.5 * ((east - be) ** 2 + (north - bn) ** 2) / bs**2)
    img = (1 - frac) * ring / ring.sum() + frac * spot / spot.sum()
    return img / img.sum()


# ---------------------------------------------------------------------------------------------
# IM Lup, joint MaxEnt fit (P3): configurations R (MPoL's objective) and F (the full model)
# ---------------------------------------------------------------------------------------------
#
# R is MPoL's objective at our optimum: 1028^2 pixels of 5 mas, a 30 mas Gaussian taper, no spectral
# index or primary beam, MPoL's entropy of the *tapered* image in Jy/arcsec^2 (`MPoLEntropy`). F is the
# model we stand behind: 1536^2 pixels, the ALMA primary beam, a free spectral index, virgil's
# `MaxEntropy` with lambda from cross-validation. The design is `design/alma_imlup_p3.md`.

import contextlib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import types  # noqa: E402
import warnings  # noqa: E402

MPOL = dict(npix=1028, pixel_mas=5.0, taper_mas=30.0, prior=1e-4, f_tot=0.253, lam_m=1e-9)  # the example's own numbers (sgd.py)
FULL = dict(npix=1536, pixel_mas=5.0, nu0_hz=239e9, dish_m=12.0, pb_factor=1.13, alpha_free=False)  # C5; alpha fixed at 0 as MPoL (Ben, 2026-10-21)

ALPHA_FIXED_LABEL = "fixed 0 (MPoL)"


def alpha_report(values):
    """``(alpha, label)`` for a summary: the fitted value (a float) and its text, or ``None`` and "fixed 0 (MPoL)"."""
    if "cal.alpha" in values:
        a = float(values["cal.alpha"])
        return a, f"{a:.4f}"
    return None, ALPHA_FIXED_LABEL

SB_OBS = (0, 1, 2, 3, 4)
EPS_FIT, EPS_SCORE = 1e-6, 1e-12  # C7
LN2PI = float(np.log(2 * np.pi))


@contextlib.contextmanager
def ft_eps(eps):
    """virgil's NUFFT at accuracy ``eps`` (float64) inside the block, restored afterwards."""
    from virgil import set_ft_defaults

    old = set_ft_defaults(ft="nufft", eps_f64=eps)
    try:
        yield
    finally:
        set_ft_defaults(**old)


def lam_r(n, lam_m=MPOL["lam_m"]):
    """C4: the strength of MPoL's entropy in units where the data term is -ln L. MPoL's batch loss is
    -ln L_b / (2 n_b) + lam_M S_M, whose stationary point over an epoch is that of -ln L + 2 N lam_M S_M."""
    return 2.0 * n * lam_m


def lam_f_equivalent(lam_r_value, flux=MPOL["f_tot"], pixel_mas=MPOL["pixel_mas"], f_tot=MPOL["f_tot"]):
    """C11: the strength of virgil's `MaxEntropy` (unit-sum image b) with the same pull on the image
    shape as R's penalty at total flux ``flux`` (Jy). With I = F b / dx^2 (Jy/arcsec^2, dx in arcsec),
    sum I ln(I/p) = (F/dx^2) [sum b ln b + ln(F/(dx^2 p))], so up to a term that does not depend on b
    (and that depends on the flux only) S_M = F / (F_tot dx^2) * sum b ln(b/q) for a flat q.
    Hence lam_F = lam_R F / (F_tot dx^2). The taper, which R's entropy sees and F's does not, and the
    primary beam, which F's does, are not in this conversion: it sets a scale, the grid of
    cross-validation decides."""
    dx = pixel_mas / 1000.0
    return lam_r_value * flux / (f_tot * dx**2)


def mpol_entropy_jax(log_brightness, flux, *, pixel_mas, taper_mas, prior, f_tot, floor=1e-30):
    """S_M = (1/F_tot) sum I ln(I / p) of the *tapered* image I (Jy/arcsec^2), in jax.

    The image is ``flux * softmax(log_brightness) / dx^2``; the taper is the circular Gaussian
    convolution of MPoL's ``GaussConvFourier`` (the Fourier multiplier ``exp(-(pi theta q)^2 / (4 ln 2))``
    on the FFT grid). Pixels are floored at ``floor`` before the logarithm, which only matters
    for rounding noise of the FFT far from the disc."""
    import jax
    import jax.numpy as jnp

    n = log_brightness.shape[0]
    dx = pixel_mas / 1000.0  # arcsec
    img = flux * jax.nn.softmax(log_brightness.ravel()).reshape(n, n) / dx**2
    if taper_mas:
        f = jnp.fft.fftfreq(n, d=dx * ARCSEC)  # cycles per radian
        f2 = f[:, None] ** 2 + f[None, :n // 2 + 1] ** 2
        t = jnp.exp(-((jnp.pi * taper_mas * ARCSEC / 1000.0) ** 2) * f2 / (4 * jnp.log(2.0)))
        img = jnp.fft.irfft2(jnp.fft.rfft2(img) * t, s=(n, n))
    img = jnp.maximum(img, floor)
    return jnp.sum(img * jnp.log(img / prior)) / f_tot


def _entropy_term_class():
    import equinox as eqx
    import jax.numpy as jnp

    class MPoLEntropy(eqx.Module):
        """MPoL's entropy as a ``fit`` likelihood term (C3): no residuals, and ``log_norm(values)`` is
        ``lam * S_M`` (computed from the image and ``cal.flux``), which ``fit`` adds to the loss.

        virgil's `MaxEntropy` is a regulariser on the model's unit-sum brightness: it cannot see
        ``cal.flux`` and does not know the taper, so it is not S_M. A regulariser is handed only the
        model; a likelihood term is handed every fitted value, and its normalisation is the one
        place ``fit`` adds a function of those values to the loss. The term has an empty
        residual vector, so it adds nothing to chi^2 or the data count. ``lam`` is a traced array, so
        a different strength does not recompile.
        """

        lam: jnp.ndarray
        pixel_mas: float = eqx.field(static=True)
        taper_mas: float = eqx.field(static=True)
        prior: float = eqx.field(static=True)
        f_tot: float = eqx.field(static=True)
        image_site: str = eqx.field(static=True, default="log_brightness")
        flux_site: str = eqx.field(static=True, default="cal.flux")
        fixed_image: object = None  # a held log_brightness (the self-calibration stage): used instead of the fitted site
        has_log_norm = True

        def __call__(self, values):
            return jnp.zeros((0,))

        def entropy(self, values):
            eta = values[self.image_site] if self.fixed_image is None else self.fixed_image
            return mpol_entropy_jax(eta, values[self.flux_site], pixel_mas=self.pixel_mas,
                                    taper_mas=self.taper_mas, prior=self.prior, f_tot=self.f_tot)

        def log_norm(self, values):
            return self.lam * self.entropy(values)

        def loglike(self, values):
            return -self.log_norm(values)

    return MPoLEntropy


def MPoLEntropy(lam, pixel_mas=MPOL["pixel_mas"], taper_mas=MPOL["taper_mas"], prior=MPOL["prior"], f_tot=MPOL["f_tot"], **kw):  # noqa: N802
    """See ``_entropy_term_class``: MPoL's entropy of the tapered image as a fit term, strength ``lam``
    in units where the data term is -ln L (``lam_r``)."""
    import jax.numpy as jnp

    return _entropy_term_class()(jnp.asarray(float(lam)), pixel_mas, taper_mas, prior, f_tot, **kw)


def grid_scene(npix, pixel_mas, primary_beam=None):
    """``scene(values)``: ``Image(log_brightness [+ the primary beam])`` on the MPoL grid (the half-pixel
    map of ``docs/alma_conventions.md``: ``dra = ddec = -dx/2``). ``primary_beam`` is a dict of the
    arguments of ``virgil.radio.primary_beam`` after ``npix`` and ``pixel_mas``; the beam is computed
    once, from numbers, on first use (outside any trace: `primary_beam` is numpy)."""
    from virgil import Image

    shift = -pixel_mas / 2.0

    cache = []

    def scene(values):
        eta = values["log_brightness"]
        if primary_beam is not None:
            if not cache:
                from virgil.radio import primary_beam as pb

                cache.append(np.asarray(pb(npix, pixel_mas, **primary_beam)))
            eta = eta + cache[0]
        return Image(eta, pixel_scale_mas=pixel_mas, dra=shift, ddec=shift, ft="nufft")

    return scene


def read_shat(path, n_ddid=26):
    """The model-free weight scales ŝ of ``alma_imlup_read`` (``ddid_NN.json``, ``sigma.all.s``) from a
    directory, as an array over ddids."""
    path = pathlib.Path(path).expanduser()
    return np.array([json.loads((path / f"ddid_{d:02d}.json").read_text())["sigma"]["all"]["s"] for d in range(n_ddid)])


def build_problem(data, config, *, lam, npix=None, pixel_mas=5.0, mask=None, use=None, chunk_size=10**6, ref_obs=0,
                  alpha_free=None, offset_bound_mas=None, fixed_log_brightness=None, image_only=False):
    """The fit problem of configuration ``config`` ("R" or "F") on ``data``. F fixes alpha at 0 like MPoL unless ``alpha_free`` (default ``FULL["alpha_free"]``).

    ``data`` is the full set (it fixes the nuisance sites); ``use`` is the set the likelihood is built on
    (default ``data``; the SB stage passes ``data.select(obs=SB_OBS)``); ``mask`` is a boolean per
    point of ``use`` (a cross-validation fold). Returns a namespace of ``scene``, ``nuis``, ``priors``,
    ``init`` (defaults; the caller sets the image, flux and sigma), ``terms``, ``regs`` and ``model``
    for ``virgil.fit``. Priors are the house defaults (C6): log-uniform flux, gains and error scales,
    uniform offsets (and alpha when ``alpha_free``), and a flat prior on the pixels' log-brightness.

    ``fixed_log_brightness`` (config R, an (npix, npix) array) holds the image: it is no free site, the scene and the entropy use it,
    and the fit moves only the nuisances (the self-calibration stage of ``--workflow clean``).

    ``image_only`` (the image-first stage) leaves the gains, offsets and error scales out of the fit: the nuisance model has gains 1, offsets 0
    and s = 1, so the free sites are the image and the flux.

    ``ref_obs`` is the observation whose gain is 1 and offset 0 (the flux gauge; ``resolve_ref_obs``).

    ``offset_bound_mas`` (mas, default None: virgil's own ±1/(2 q_min), about 10^4 mas for IM Lup) sets the offsets' uniform prior to
    ``(-X, X)`` through ``CalibrationNuisance.priors(offset=...)``. A flat prior is flat at any bound, so the objective is unchanged
    unless a fitted offset reaches the bound (``offset_bound_report``), but L-BFGS moves in the logit z of ``x/bound`` and dx/dz is
    about half the bound: a bound near the offsets (say 100 mas) conditions them far better. ``prob.offset_bound_mas`` is the bound in force.
    """
    import jax.numpy as jnp
    import numpyro.distributions as dist
    from virgil import CalibrationNuisance
    from virgil.imaging import MaxEntropy
    from virgil.radio import primary_beam

    config = config.upper()
    off = dict(gains=False, offsets=False, sigmas=False) if image_only else {}
    alpha_free = FULL["alpha_free"] if alpha_free is None else bool(alpha_free)  # F only; R never fits alpha
    use = data if use is None else use
    if config == "R":
        npix = npix or MPOL["npix"]
        nuis = CalibrationNuisance(data, ref_obs=ref_obs, alpha=False, taper_fwhm_mas=MPOL["taper_mas"], **off)
        scene, regs = grid_scene(npix, pixel_mas), []
        if fixed_log_brightness is not None:
            held = jnp.asarray(fixed_log_brightness, float)
            if held.shape != (npix, npix):
                raise ValueError(f"fixed_log_brightness is {held.shape}, not {(npix, npix)}")
            free_scene = scene
            scene = lambda values: free_scene({**values, "log_brightness": held})  # noqa: E731
    elif config == "F":
        if fixed_log_brightness is not None:
            raise ValueError("a held image is defined for config R only")
        npix = npix or FULL["npix"]
        pb = dict(freq_hz=FULL["nu0_hz"], dish_m=FULL["dish_m"], factor=FULL["pb_factor"])
        nuis = CalibrationNuisance(data, ref_obs=ref_obs, alpha=alpha_free, nu0_hz=FULL["nu0_hz"], **off)
        scene = grid_scene(npix, pixel_mas, pb)
        # the pixels are sky x beam; the default image of the entropy is the beam, so the penalty is the
        # entropy of the sky relative to a flat sky (docs: "the default image q ... normalised here")
        regs = [MaxEntropy(float(lam), prior=np.exp(np.asarray(primary_beam(npix, pixel_mas, **pb))))]
    else:
        raise ValueError(f"config must be R or F, not {config!r}")
    term = use.term(scene, nuis, mask=None if mask is None else np.asarray(mask, bool), chunk_size=chunk_size)
    terms = [term] + ([MPoLEntropy(lam, pixel_mas=pixel_mas, fixed_image=None if fixed_log_brightness is None else jnp.asarray(fixed_log_brightness, float))] if config == "R" else [])
    priors = nuis.priors(**({} if offset_bound_mas is None else {"offset": (-float(offset_bound_mas), float(offset_bound_mas))}))
    if fixed_log_brightness is None:
        priors["log_brightness"] = dist.ImproperUniform(dist.constraints.real, (), (npix, npix))
    init = {k: jnp.asarray(v) for k, v in nuis.defaults().items()}
    bound = np.broadcast_to(np.asarray(nuis.offset_bound_mas if offset_bound_mas is None else offset_bound_mas, float), (len(data.obs_ids),)).copy()
    return types.SimpleNamespace(config=config, npix=npix, pixel_mas=pixel_mas, scene=scene, nuis=nuis, priors=priors, init=init,
                                 terms=terms, regs=regs, model=lambda **values: scene(values), lam=float(lam),
                                 offset_bound_mas=float(bound.max()), offset_bound_obs=bound,  # virgil's bound may be one per observation: the largest, and all of them
                                 offset_bound_requested=None if offset_bound_mas is None else float(offset_bound_mas))


OFFSET_EDGE = 0.05  # report warns when a fitted offset is within this fraction of the bound


def offset_bound_report(offsets, bound_mas, requested=None, edge=OFFSET_EDGE, bound_obs=None, ref_obs=0):
    """Is the offsets' uniform bound binding? ``offsets`` is the fitted ``cal.offset`` (mas, (n_obs - 1, 2), the observations but ``ref_obs``;
    None for none). A fitted offset within ``edge`` of its bound means the bound shapes the fit, so the objective is no longer the unbounded one.
    ``bound_obs`` (n_obs,) is the bound of each observation where virgil's is per observation (``prob.offset_bound_obs``; default ``bound_mas``
    for all), and each offset is judged against its own. Returns bound (the largest), requested, the largest |offset|/own bound, the offending
    (obs, axis, mas) entries and a ``warning`` string (None if clear)."""
    off = np.zeros((0, 2)) if offsets is None else np.asarray(offsets, float).reshape(-1, 2)
    obs = np.array([e for e in range(len(off) + 1 if bound_obs is None else len(bound_obs)) if e != ref_obs][:len(off)], int)
    own = np.full(len(off), float(bound_mas)) if bound_obs is None else np.asarray(bound_obs, float)[obs]
    frac = np.abs(off) / own[:, None] if len(off) else np.zeros((0, 2))
    near = [dict(obs=int(obs[i]), axis=["ra", "dec"][int(j)], offset_mas=float(off[i, j])) for i, j in zip(*np.nonzero(frac >= 1.0 - edge))]
    warning = None
    if near:
        warning = (f"!!! OFFSET BOUND BINDING: {len(near)} fitted offset(s) within {edge:.0%} of their bound (±{bound_mas:g} mas at most) "
                   f"({', '.join('obs %d %s %.4g' % (n['obs'], n['axis'], n['offset_mas']) for n in near)}); the bound is changing the objective: "
                   f"raise --offset-bound-mas or drop it !!!")
    return dict(bound_mas=float(bound_mas), bound_obs=None if bound_obs is None else [float(b) for b in bound_obs], requested_mas=requested,
                max_abs_over_bound=float(frac.max()) if frac.size else 0.0, near_bound=near, warning=warning)


def drop_unused_sites(values, prob):
    """``values`` without the ``cal.*`` sites ``prob`` does not have: a warm start or report from a free-alpha F fit
    (``cal.alpha`` saved) must not re-enable or report the alpha that is now fixed at 0."""
    return {k: v for k, v in values.items() if not k.startswith("cal.") or k in prob.init}


def resolve_ref_obs(data, spec="0"):
    """The reference observation (gain 1, offset 0): an index, or ``"auto"``, the short-baseline observation (``SB_OBS``) with the largest total
    weight sum(w), whose amplitude scale is best fixed by the data. Returns ``(index, info)``; ``info`` has the spec and the sum(w) of each SB
    observation, for the log and result.json."""
    obs, w = np.asarray(data.obs), np.asarray(data.weight, float)
    weight = np.array([float(w[obs == o].sum()) for o in SB_OBS])
    if str(spec).lower() == "auto":
        ref = int(SB_OBS[int(np.argmax(weight))])
    else:
        ref = int(spec)
        if not 0 <= ref < len(data.obs_ids):
            raise ValueError(f"--ref-obs {spec!r} is not an observation index (0..{len(data.obs_ids) - 1}) or 'auto'")
    return ref, dict(spec=str(spec), ref_obs=ref, sb_obs=list(SB_OBS), sb_sum_weight=weight)


def offset_beam_bounds(data, n_beams=3.0):
    """A proxy for the per-observation offset bound of ``n_beams`` beams: ``n_beams / q_max`` of each observation, in mas, with the nominal beam
    ``1 / q_max`` (q_max its longest baseline in wavelengths). ``offset_bound_mas`` in force is to be compared with it observation by observation;
    the bound virgil applies, where it has one per observation, is its own (``prob.offset_bound_mas``)."""
    obs = np.asarray(data.obs)
    q = np.hypot(np.asarray(data.u, float), np.asarray(data.v, float))
    qmax = np.array([q[obs == o].max() for o in range(len(data.obs_ids))])
    return n_beams / qmax * (180.0 / np.pi * 3600.0 * 1000.0)


def gauge_to_obs0(gain, offset, flux, ref_obs):
    """The gains, offsets and flux of a fit with reference observation ``ref_obs`` in MPoL's convention (obs 0 has gain 1 and offset 0). The
    model ``flux * g_e V`` is unchanged by ``g -> g / g_0``, ``flux -> flux g_0``, and the offsets' common shift ``o -> o - o_0`` is a translation of
    the whole sky: it is applied here to the table only (the image is not translated). Returns full ``(n_obs,)`` gains, ``(n_obs, 2)`` offsets and the flux."""
    g = np.insert(np.asarray(gain, float), ref_obs, 1.0)
    o = np.insert(np.asarray(offset, float).reshape(-1, 2), ref_obs, 0.0, axis=0)
    return g / g[0], o - o[0], float(flux) * g[0]


def start_values(prob, *, flux=0.25, fwhm_mas=1000.0, shat=None):
    """C9's start: a circular Gaussian of FWHM 1" and flux 0.25 Jy, s_d at ŝ_d (or 1). The bare Gaussian leaves the outer
    pixels at 1e-8 to 1e-15 of the peak, which the fit can never revive (``dead_pixel_report``): ``floor_start`` raises them."""
    import jax.numpy as jnp

    from crosscheck.alma_maxent import gaussian_log_image

    v = dict(prob.init)
    v["log_brightness"] = jnp.asarray(gaussian_log_image(prob.npix, prob.pixel_mas, fwhm_mas))
    v["cal.flux"] = jnp.asarray(flux)
    if shat is not None and "cal.sigma" in v:
        v["cal.sigma"] = jnp.asarray(shat, float)
    return v


def floor_start(values, npix, pixel_mas, floor, log_pb=None):
    """``(log_brightness, cal.flux, info)`` of ``values``' start image with every sky pixel raised to at least ``floor`` Jy/arcsec^2
    (MPoL's entropy prior p = 1e-4 is the level a pixel takes with no data; R_clean does the same). ``image_of`` of the result is
    exactly the floored image, in config F too (``log_pb`` the beam: ``cal.flux`` is then the apparent flux, sky x beam)."""
    image = image_of(values, npix, pixel_mas, log_pb)
    floored = np.maximum(image, floor)
    area = (pixel_mas / 1000.0) ** 2
    pb = 1.0 if log_pb is None else np.exp(np.asarray(log_pb))
    info = dict(floor_jy_arcsec2=float(floor), n_raised=int((image < floor).sum()), flux_before_jy=float(image.sum() * area),
                flux_added_jy=float((floored - image).sum() * area), peak_jy_arcsec2=float(floored.max()))
    return np.log(floored / floored.max()), float((floored * pb).sum() * area), info


DEAD_FRACTION_OF_PEAK = 1e-6  # below this fraction of the peak a pixel's gradient (d loss / d b_i is proportional to I_i) is too small to move it
DEAD_FRACTION_WARN = 0.01


def dead_pixel_report(image, pixel_mas, *, floor=MPOL["prior"]):
    """The pixels the optimiser cannot revive. With ``I = flux softmax(b)`` the gradient on ``b_i`` is proportional to ``I_i``, so a pixel
    far below the peak stays there whatever the entropy term wants (its interior optimum has every pixel above zero, ``dS/dI = ln(I/M)``).
    Returns the fraction of pixels below 1e-6 of the peak and below 1e-3 of the entropy ``floor``, the median brightness beyond 2.4 arcsec
    over the peak, the largest live radius (arcsec from the phase-centre cell; live = at least 1e-6 of the peak) and a ``warning``
    (None if at most 1 % are dead). R_mpol and MPoL's own solution have none dead; R ob100 (18968246_0) had 64 %."""
    from crosscheck.alma_maxent import phase_centre_cell

    img = np.asarray(image, float)
    n = img.shape[0]
    peak = float(img.max())
    dead = img < DEAD_FRACTION_OF_PEAK * peak
    r0, c0 = phase_centre_cell(n)
    rr, cc = np.meshgrid(np.arange(n) - r0, np.arange(n) - c0, indexing="ij")
    radius = np.hypot(rr, cc) * pixel_mas / 1000.0
    frac = float(dead.mean())
    outer = radius > 2.4
    out = {"peak_jy_arcsec2": peak, "frac_below_1e-6_peak": frac, "frac_below_1e-3_floor": float((img < 1e-3 * floor).mean()),
           "median_beyond_2p4_arcsec_over_peak": float(np.median(img[outer]) / peak) if outer.any() else None,
           "largest_live_radius_arcsec": float(radius[~dead].max()) if (~dead).any() else 0.0}
    out["warning"] = None if frac <= DEAD_FRACTION_WARN else (
        f"!!! DEAD PIXELS: {frac:.1%} of the pixels are below {DEAD_FRACTION_OF_PEAK:g} of the peak (live out to {out['largest_live_radius_arcsec']:.2f} arcsec). "
        "The gradient on a softmax logit is proportional to the pixel's brightness, so they cannot grow back and their flux piles into the outermost live pixels: "
        "start from the entropy floor (--start-floor) or treat this fit as an optimiser artefact !!!")
    return out


def mpol_base_jy_arcsec2():
    """MPoL's initial per-pixel brightness in Jy/arcsec^2: ``softplus(-3)`` = ln(1 + e^-3) = 0.04859. ``BaseCube`` fills its
    ``base_cube`` with -3 and maps it through ``torch.nn.Softplus`` (mpol 0.3.1 ``images.py``: ``BaseCube.__init__`` lines 65-76
    for the -3 and the Softplus default, ``forward`` for the mapping), and ``sgd.py:59`` builds it with no ``base_cube``."""
    return float(np.log1p(np.exp(-3.0)))


def mpol_start_values(prob):
    """The start of MPoL's ``sgd.py`` as virgil values (workflow ``mpol``), replacing C9's Gaussian and ŝ:

    - image: uniform ``softplus(-3)`` Jy/arcsec^2 per pixel (``BaseCube``, ``images.py:65-76``; ``sgd.py:59``), i.e. our
      ``flux * softmax(log_brightness) / dx^2`` with equal logits (zero) and ``flux = value * dx^2 * npix^2`` (1.2837 Jy at
      1028^2 x 5 mas, not the 0.253 Jy of ``F_tot``);
    - gains 1: ``sgd.py:49`` ``log10_amp_factors = zeros``;
    - offsets 0: ``sgd.py:56-57`` ``delta_ra``, ``delta_dec = zeros``;
    - s_d = 1: ``sgd.py:53`` ``log10_sigma_factors = zeros`` (s_MPoL = 1) with g = 1 gives s = g s_MPoL = 1. No ŝ.
    """
    import jax.numpy as jnp

    v = dict(prob.init)
    dx = prob.pixel_mas / 1000.0
    v["log_brightness"] = jnp.zeros((prob.npix, prob.npix))
    v["cal.flux"] = jnp.asarray(mpol_base_jy_arcsec2() * dx**2 * prob.npix**2)
    for k, fill in (("cal.gain", 1.0), ("cal.sigma", 1.0), ("cal.offset", 0.0)):
        if k in v:
            v[k] = jnp.full_like(jnp.asarray(v[k], float), fill)
    return v


def run_fit(prob, init, *, max_steps, time_limit=None, eps=EPS_FIT, log=None, gtol=1e-4, precondition=None):
    """L-BFGS through ``virgil.fit`` (C8). Returns ``(values, info, trajectory)``: the fitted values as
    numpy arrays, ``fit``'s diagnostics and one dict (steps, loss, grad_norm, elapsed) per chunk of
    steps. The loss is in -ln L units up to the constant of the priors' normalisation. ``precondition`` (``"diag"``
    or None) is ``virgil.fit``'s opt-in diagonal Gauss-Newton scaling; it needs a virgil that has the argument."""
    import inspect

    from virgil import fit

    kw = {}
    if precondition not in (None, "none"):
        if "precondition" not in inspect.signature(fit).parameters:
            raise RuntimeError(f"precondition={precondition!r} needs a virgil whose fit() has the precondition argument "
                               "(virgil >= the release that includes fit(precondition=...)); the installed one has not")
        kw["precondition"] = precondition

    traj = []

    def progress(d):
        traj.append({k: float(v) for k, v in d.items()})
        if log:
            log(f"  steps {d['steps']:6d}  loss {float(d['loss']):.6f}  |g| {float(d['grad_norm']):.3g}  {float(d['elapsed']):.0f} s")

    with ft_eps(eps), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fit(prob.model, prob.priors, (), regularizers=prob.regs, likelihoods=prob.terms, init=init, method="lbfgs",
                  max_steps=max_steps, time_limit=time_limit, progress=progress, gtol=gtol, **kw)
    values = {k: np.asarray(v) for k, v in res.values.items()}
    info = {k: (v.item() if hasattr(v, "item") else v) for k, v in res.info.items() if np.ndim(v) == 0}
    info.update({k: np.asarray(v).tolist() for k, v in res.info.items() if np.ndim(v) > 0 and k != "precondition_rounds"})
    info["precondition_rounds"] = [{k: float(v) for k, v in r.items()} for r in res.info.get("precondition_rounds", [])]
    return values, info, traj


def run_fit_c8(prob, init, *, max_steps, time_limit=None, eps=EPS_FIT, log=None, gtol=1e-4, on_round=None, max_rounds=50, precondition=None):
    """L-BFGS through ``virgil.fit`` until C8 holds (loss change < 0.1 over >= 100 steps), the step or the time
    budget (``max_steps`` and ``time_limit``, per call) runs out, or the optimiser can go no further.

    ``fit`` stops on its own test: no component of the loss-per-point gradient above ``gtol`` (and 1/1000 of the
    starting gradient, so a warm start gets a tighter test); its ``stop`` is ``None`` when that test passed. On
    IM Lup ``gtol = 1e-4`` passed with the loss still falling by 20 per 200 steps, far from C8. So each time
    ``fit`` stops short of C8 with budget left, this restarts it from its last point (fresh curvature memory),
    whose test is then 1/1000 of the current gradient, until C8 holds. ``on_round(values, info, traj, round)`` is called
    after every round (the caller saves there). Returns ``(values, info, traj, why)``: ``traj`` is the loss history over
    all rounds (steps are cumulative), ``info`` the last round's ``fit`` diagnostics with the totals, and
    ``why`` one of ``c8``, ``max_steps``, ``time_limit``, ``stalled`` (a round that could not move), ``non_finite``,
    ``max_rounds``."""
    import time

    t0, total, traj, values, why, idle = time.time(), 0, [], init, "max_rounds", 0
    pc_rounds, restarts = [], []  # the diagonal scaling's per-L-BFGS-round stats and each restart's steps and wall time
    info = dict(steps=0, rounds=0, stop="time", loss=float("nan"), converged=False)
    for rnd in range(1, max_rounds + 1):
        left = time_limit - (time.time() - t0) if time_limit is not None else None
        if left is not None and left <= 0:
            why = "time_limit"
            break
        t_fit = time.time()
        values, info, part = run_fit(prob, values, max_steps=max_steps - total, time_limit=left, eps=eps, log=log, gtol=gtol, **({} if precondition is None else {"precondition": precondition}))
        steps = int(info["steps"])
        pc_rounds += info.get("precondition_rounds", [])
        restarts.append(dict(steps=steps, wall_s=time.time() - t_fit, loss=float(info["loss"])))
        for t in part:
            traj.append({**t, "steps": t["steps"] + total, "round": rnd})
        if not traj or traj[-1]["steps"] != total + steps:  # a closing record (also for a round of 0 steps)
            traj.append(dict(steps=float(total + steps), loss=float(info["loss"]), grad_norm=float(info.get("grad_norm", np.nan)), elapsed=time.time() - t0, round=rnd))
        total += steps
        info = {**info, "steps": total, "rounds": rnd, "precondition": precondition, "precondition_rounds": pc_rounds, "restarts": restarts}
        stop = info.get("stop")
        finite = np.isfinite(float(info["loss"])) and all(np.all(np.isfinite(v)) for v in values.values())
        if not finite or stop == "non-finite":  # never overwrite the last good save with NaNs
            why = "non_finite"
            break
        if on_round:
            on_round(values, info, traj, rnd)
        verdict = convergence(traj)
        if stop == "stalled":
            why = "stalled"
            break
        if stop == "time":
            why = "time_limit"
            break
        if verdict["converged"] and (verdict["window_steps"] or 0) >= 100:
            why = "c8"
            break
        if total >= max_steps:
            why = "max_steps"
            break
        idle = idle + 1 if steps == 0 else 0
        if idle >= 2:
            why = "stalled"
            break
    return values, info, traj, why


def convergence(traj, info=None, tol=0.1, window=100):
    """C8: converged when the loss changed by less than ``tol`` (-ln L units, i.e. chi^2 by less than
    2 tol) over the last ``window`` steps. ``fit`` reports its progress every chunk of 200 steps, so
    the last record is compared with the latest one at least ``window`` steps earlier (the first record
    if the run was shorter). Returns a dict with the verdict and the numbers it rests on."""
    if len(traj) < 2:
        done = bool(info and info.get("converged"))
        return dict(converged=done, change=None, window_steps=None, reason="fewer than two records" + ("; fit's own test passed" if done else ""))
    last = traj[-1]
    ref = next((t for t in reversed(traj[:-1]) if last["steps"] - t["steps"] >= window), traj[0])
    change = abs(last["loss"] - ref["loss"])
    return dict(converged=bool(change < tol), change=float(change), window_steps=int(last["steps"] - ref["steps"]), reason="loss change over the window")


CHUNK = 10**6  # points per model evaluation: at eps 1e-12 one NUFFT buffer is about 1.3 kB per point (the 20.6 GiB that failed was 1.74e7 points)


def model_chunks(data, sel, scene, jv, nuis, chunk_size=CHUNK):
    """Yield ``(index, subset, model)`` for the points of boolean ``sel``, ``chunk_size`` at a time, so that the
    model of an observation is never evaluated at once (``ComplexVisData.model`` does not chunk). ``index`` holds the
    positions in ``data``; the caller sets the NUFFT accuracy (``ft_eps``)."""
    n = len(data.u)
    idx_all = np.flatnonzero(np.asarray(sel, bool))
    model_scene = scene(jv)
    for i in range(0, len(idx_all), int(chunk_size)):
        idx = idx_all[i:i + int(chunk_size)]
        keep = np.zeros(n, bool)
        keep[idx] = True
        sub = data.subset(keep)
        yield idx, sub, np.asarray(sub.model(model_scene, jv, nuis))


def per_point(data, scene, values, nuis, mask=None, eps=EPS_SCORE, chunk_size=CHUNK):
    """Per-point decomposition of -ln L (the complex Gaussian with variance s_d^2 / w_k per component):
    returns ``(ddid, obs, r2, nll)`` over the unmasked points, ``r2 = w |V - M|^2`` with the quoted weights
    and ``nll = r2 / (2 s_d^2) + 2 ln s_d + ln 2 pi - ln w``. The model is evaluated one observation at a time,
    ``chunk_size`` points at a time (``model_chunks``)."""
    import jax.numpy as jnp

    obs, dd = np.asarray(data.obs), np.asarray(data.ddid)
    sigma = np.asarray(values["cal.sigma"], float) if "cal.sigma" in values else np.ones(int(dd.max()) + 1)
    jv = {k: jnp.asarray(v) for k, v in values.items()}
    keep = np.ones(len(obs), bool) if mask is None else np.asarray(mask, bool)
    out = dict(ddid=[], obs=[], r2=[], nll=[], index=[])
    with ft_eps(eps):
        for e in np.unique(obs):
            sel = (obs == e) & keep
            for idx, sub, m in model_chunks(data, sel, scene, jv, nuis, chunk_size):
                w, d = np.asarray(sub.weight, float), np.asarray(sub.ddid)
                r2 = w * np.abs(np.asarray(sub.vis) - m) ** 2
                out["ddid"].append(d)
                out["obs"].append(np.full(len(d), e))
                out["r2"].append(r2)
                out["nll"].append(0.5 * r2 / sigma[d] ** 2 + 2 * np.log(sigma[d]) + LN2PI - np.log(w))
                out["index"].append(idx)
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in out.items()}


def score(data, scene, values, nuis, mask=None, annulus=None, n_ddid=None, n_annulus=12, eps=EPS_SCORE, chunk_size=CHUNK):
    """-ln L, chi^2 with the quoted weights (s = 1, C10) and with the fitted s, per ddid and (given an
    ``annulus`` per point) per uv annulus, over the unmasked points, at NUFFT accuracy ``eps`` (C7).
    ``chi2`` sums two squares per point, so chi^2/N of the tables is ``chi2 / (2 n)``."""
    pp = per_point(data, scene, values, nuis, mask, eps, chunk_size)
    sigma = np.asarray(values["cal.sigma"], float) if "cal.sigma" in values else np.ones(int(np.asarray(data.ddid).max()) + 1)
    n_ddid = n_ddid or len(sigma)

    def tally(ids, size):
        n = np.bincount(ids, minlength=size)
        return dict(n=n.tolist(), nll=np.bincount(ids, pp["nll"], size).tolist(), chi2_raw=np.bincount(ids, pp["r2"], size).tolist(),
                    chi2_fit=np.bincount(ids, pp["r2"] / sigma[pp["ddid"]] ** 2, size).tolist())

    out = dict(nll=float(pp["nll"].sum()), n=int(len(pp["nll"])), chi2_raw=float(pp["r2"].sum()), by_ddid=tally(pp["ddid"], n_ddid))
    out["chi2_fit"] = float(sum(out["by_ddid"]["chi2_fit"]))
    if annulus is not None:
        out["by_annulus"] = tally(np.asarray(annulus)[pp["index"]].astype(int), n_annulus)
    return out


def residual_dirty_images(data, scene, values, nuis, npix=1024, pixel_mas=3.0, eps=EPS_SCORE, chunk_size=CHUNK):
    """Per observation, the natural-weighted dirty image of the residuals V - M (weights w / s_d^2) on a
    ``npix`` grid of ``pixel_mas`` (``crosscheck.alma.gridded_dirty_image``, Jy per dirty beam; layout of
    crosscheck.alma: row 0 North, column 0 East, centre at ``npix // 2``)."""
    import jax.numpy as jnp

    from crosscheck.alma import gridded_dirty_image

    obs = np.asarray(data.obs)
    sigma = np.asarray(values["cal.sigma"], float) if "cal.sigma" in values else np.ones(int(np.asarray(data.ddid).max()) + 1)
    jv = {k: jnp.asarray(v) for k, v in values.items()}
    out = {}
    with ft_eps(eps):
        for e in np.unique(obs):
            u, v, r, w = [], [], [], []
            for _, sub, m in model_chunks(data, obs == e, scene, jv, nuis, chunk_size):  # the model in chunks, the grid once
                u.append(np.asarray(sub.u))
                v.append(np.asarray(sub.v))
                r.append(np.asarray(sub.vis) - m)
                w.append(np.asarray(sub.weight, float) / sigma[np.asarray(sub.ddid)] ** 2)
            out[int(e)] = gridded_dirty_image(np.concatenate(u), np.concatenate(v), np.concatenate(r), np.concatenate(w), npix, pixel_mas)
    return out


def beam_log(npix, pixel_mas):
    """ln of the primary beam of config F on the grid (numpy)."""
    from virgil.radio import primary_beam

    return np.asarray(primary_beam(npix, pixel_mas, freq_hz=FULL["nu0_hz"], dish_m=FULL["dish_m"], factor=FULL["pb_factor"]))


def image_of(values, npix, pixel_mas, log_pb=None):
    """The sky image in Jy/arcsec^2 (virgil layout) of fitted values. Without ``log_pb`` (config R) the
    pixels are the sky: ``flux * softmax(log_brightness) / dx^2``. In config F the fitted pixels are sky x beam
    (``cal.flux`` is the flux of the apparent image, ``grid_scene``), so pass ``log_pb = beam_log(...)``:
    the sky is ``flux * exp(log_brightness) / sum exp(log_brightness + log_pb) / dx^2``, which times the beam
    sums to ``cal.flux``."""
    eta = np.asarray(values["log_brightness"])
    pb = 0.0 if log_pb is None else np.asarray(log_pb)
    shift = np.max(eta + pb)
    return float(values["cal.flux"]) * np.exp(eta - shift) / np.exp(eta + pb - shift).sum() / (pixel_mas / 1000.0) ** 2


def tapered_image(image, pixel_mas, taper_mas=MPOL["taper_mas"]):
    from crosscheck.alma_maxent import convolve

    return convolve(image, pixel_mas, taper_mas)


# ----------------------------------------------------------- cross-validation folds


def make_folds(data, k=5, n_q=12, n_phi=16, seed=0):
    """C11: ``virgil.radio.dartboard_folds`` on all the data at once; returns ``(fold, annulus)`` as int8."""
    from virgil.radio import dartboard_folds

    fold, annulus = dartboard_folds(data, k=k, n_q=n_q, n_phi=n_phi, seed=seed)
    return np.asarray(fold).astype(np.int8), np.asarray(annulus).astype(np.int8)


def folds_sha256(fold_by_ddid):
    """sha256 of the fold assignment: the int8 folds of the ddids in increasing order, concatenated."""
    h = hashlib.sha256()
    for d in sorted(fold_by_ddid):
        h.update(np.ascontiguousarray(fold_by_ddid[d], np.int8).tobytes())
    return h.hexdigest()


def _uv_sha256(data, sel):
    """sha256 of the (u, v) of the selected points, in order: ties a fold file to the point order it was made for."""
    h = hashlib.sha256()
    for a in (data.u, data.v):
        h.update(np.ascontiguousarray(np.asarray(a)[sel], np.float64).tobytes())
    return h.hexdigest()


def save_folds(path, data, fold, annulus, params):
    """One file: ``fold_NN`` and ``annulus_NN`` (int8, in the order of the ddid's points in its npz file), the
    parameters and the sha256 of the assignment (``folds_sha256``) as a JSON string ``meta``."""
    ids = np.asarray(data.ddid_ids)[np.asarray(data.ddid)]
    arrays, by = {}, {}
    for d in np.unique(ids):
        sel = ids == d
        arrays[f"fold_{d:02d}"], arrays[f"annulus_{d:02d}"] = fold[sel], annulus[sel]
        by[int(d)] = fold[sel]
    uv = {int(d): _uv_sha256(data, ids == d) for d in np.unique(ids)}
    meta = dict(params, sha256=folds_sha256(by), n=int(len(fold)), counts=np.bincount(fold, minlength=params["k"]).tolist(), uv_sha256=uv)
    np.savez_compressed(path, meta=json.dumps(meta), **arrays)
    return meta


def load_folds(path, data):
    """``(fold, annulus, meta)`` of a file written by ``save_folds``, in the order of ``data``'s points."""
    z = np.load(path)
    ids = np.asarray(data.ddid_ids)[np.asarray(data.ddid)]
    fold, annulus = np.zeros(len(ids), np.int8), np.zeros(len(ids), np.int8)
    by = {}
    for d in np.unique(ids):
        sel = ids == d
        fold[sel], annulus[sel] = z[f"fold_{d:02d}"], z[f"annulus_{d:02d}"]
        by[int(d)] = fold[sel]
    meta = json.loads(str(z["meta"]))
    for d in np.unique(ids):
        want = meta.get("uv_sha256", {}).get(str(int(d)))
        if want is not None and _uv_sha256(data, ids == d) != want:
            raise ValueError(f"ddid {d}: the points differ (count or order) from those the fold file was made for")
    if len(np.unique(ids)) == len(set(int(k[5:]) for k in z.files if k.startswith("fold_"))) and folds_sha256(by) != meta["sha256"]:
        raise ValueError("fold file does not match its own checksum")
    return fold, annulus, meta


# ----------------------------------------------------------- MPoL's solution in virgil's parametrisation


def values_from_mpol_export(export, obs_of_ddid):
    """virgil values of an MPoL solution (``mpol_worker.export_checkpoint``'s npz: ``base_image`` in
    Jy/arcsec^2 in this repository's layout, ``amp`` and ``sigma`` (MPoL's s_M) with the fixed observation
    included, ``delta_ra_arcsec``/``delta_dec_arcsec``): ``cal.flux = sum I dx^2``, ``cal.sigma = g s_M``
    per ddid, offsets in mas (``docs/alma_conventions.md``, Part 3)."""
    import jax.numpy as jnp

    base = np.asarray(export["base_image"], float)
    dx = float(export["cell_size_arcsec"])
    amp = np.asarray(export["amp"], float)
    if not np.isclose(amp[0], 1.0, rtol=0, atol=1e-12):
        raise ValueError(f"the reference observation's amplitude must be 1 (cal.gain = amp[1:]); got {amp[0]!r}")
    return dict(
        log_brightness=jnp.asarray(np.log(base)),
        **{"cal.flux": jnp.asarray(base.sum() * dx**2), "cal.gain": jnp.asarray(amp[1:]),
           "cal.offset": jnp.asarray(1000.0 * np.stack([export["delta_ra_arcsec"][1:], export["delta_dec_arcsec"][1:]], axis=-1)),
           "cal.sigma": jnp.asarray(np.asarray(export["sigma"], float) * amp[np.asarray(obs_of_ddid)])},
    )


def r_objective(data, values, *, lam, mask=None, npix=None, pixel_mas=5.0, eps=EPS_SCORE, chunk_size=CHUNK, ref_obs=0, entropy_flux=None):
    """R's objective -ln L + lam S_M at ``values``, in float64 at NUFFT accuracy ``eps`` (the
    normalising constant of -ln L included), and its two parts. -ln L is invariant under the gain gauge (g -> g / g_0, flux -> flux g_0), S_M
    is not (it scales with the image's flux): ``entropy_flux`` evaluates S_M at that flux instead of ``values['cal.flux']``, to compare a fit
    whose reference observation is not 0 with MPoL's in MPoL's gauge (``entropy_fit_gauge`` is then the entropy at the fit's own flux)."""
    import jax.numpy as jnp

    npix = npix or values["log_brightness"].shape[0]
    prob = build_problem(data, "R", lam=lam, npix=npix, pixel_mas=pixel_mas, mask=mask, ref_obs=ref_obs)
    sc = score(data, prob.scene, values, prob.nuis, mask=mask, eps=eps, chunk_size=chunk_size)
    jv = {k: jnp.asarray(v) for k, v in values.items()}
    s_fit = float(prob.terms[1].entropy(jv))
    s = s_fit if entropy_flux is None else float(prob.terms[1].entropy({**jv, "cal.flux": jnp.asarray(entropy_flux)}))
    return dict(nll=sc["nll"], entropy=s, penalty=lam * s, objective=sc["nll"] + lam * s, by_ddid=sc["by_ddid"], n=sc["n"], entropy_fit_gauge=s_fit)


def grad_groups(data, values, *, lam, npix=None, pixel_mas=5.0, eps=EPS_FIT, chunk_size=CHUNK, offset_bound_mas=None, ref_obs=0):
    """How well scaled R's parameters are for L-BFGS, at ``values``: per site (image pixels, flux, gain, offset, sigma), the
    gradient of the loss ``virgil.fit`` minimises with respect to the unconstrained coordinates z it optimises
    (``|g_z|`` max and 2-norm), the median Jacobian ``dx/dz`` of the site's value x (1 for the pixels; for ``cal.offset`` about
    half the offset bound in mas, since z is a logit of ``x/bound``), and the gradient in the site's own units
    ``g_x = g_z / (dx/dz)``. The loss is virgil's ``_Objective.loss`` (a private class, pinned with virgil) at the fit's
    eps; one value-and-gradient costs what an L-BFGS step does."""
    import jax
    import jax.numpy as jnp
    from virgil.fitting import _Objective

    npix = npix or values["log_brightness"].shape[0]
    prob = build_problem(data, "R", lam=lam, npix=npix, pixel_mas=pixel_mas, chunk_size=chunk_size, offset_bound_mas=offset_bound_mas, ref_obs=ref_obs)
    obj = _Objective(prob.model, prob.priors, (), tuple(prob.regs), None, tuple(prob.terms))
    z = obj.init({k: jnp.asarray(v) for k, v in values.items() if k in obj.sites})
    with ft_eps(eps):
        g = jax.grad(obj.loss)(z)
    rows = {}
    for site in obj.sites:
        # elementwise maps: dx_i/dz_i is the gradient of sum(x) with respect to z
        jac = jax.grad(lambda zz, site=site: jnp.sum(obj.constrain({**z, site: zz})[site]))(z[site])
        gz, jz = np.asarray(g[site], float).ravel(), np.asarray(jac, float).ravel()
        gx = gz / jz
        extra = {}
        if site == "cal.sigma":  # dL/d ln s_i = s_i g_x,i
            extra["gx_lns_max"] = float(np.max(np.abs(np.asarray(obj.constrain(z)[site], float).ravel() * gx)))
        rows[site] = dict(n=int(gz.size), **extra, gz_max=float(np.max(np.abs(gz))), gz_norm=float(np.linalg.norm(gz)), dxdz_median=float(np.median(np.abs(jz))),
                          gx_max=float(np.max(np.abs(gx))), gx_norm=float(np.linalg.norm(gx)))
    return rows
