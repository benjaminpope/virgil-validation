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
FULL = dict(npix=1536, pixel_mas=5.0, nu0_hz=239e9, dish_m=12.0, pb_factor=1.13)  # C5
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
        has_log_norm = True

        def __call__(self, values):
            return jnp.zeros((0,))

        def entropy(self, values):
            return mpol_entropy_jax(values[self.image_site], values[self.flux_site], pixel_mas=self.pixel_mas,
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


def build_problem(data, config, *, lam, npix=None, pixel_mas=5.0, mask=None, use=None, chunk_size=10**6, ref_obs=0):
    """The fit problem of configuration ``config`` ("R" or "F") on ``data``.

    ``data`` is the full set (it fixes the nuisance sites); ``use`` is the set the likelihood is built on
    (default ``data``; the SB stage passes ``data.select(obs=SB_OBS)``); ``mask`` is a boolean per
    point of ``use`` (a cross-validation fold). Returns a namespace of ``scene``, ``nuis``, ``priors``,
    ``init`` (defaults; the caller sets the image, flux and sigma), ``terms``, ``regs`` and ``model``
    for ``virgil.fit``. Priors are the house defaults (C6): log-uniform flux, gains and error scales,
    uniform offsets and alpha, and a flat prior on the pixels' log-brightness.
    """
    import jax.numpy as jnp
    import numpyro.distributions as dist
    from virgil import CalibrationNuisance
    from virgil.imaging import MaxEntropy
    from virgil.radio import primary_beam

    config = config.upper()
    use = data if use is None else use
    if config == "R":
        npix = npix or MPOL["npix"]
        nuis = CalibrationNuisance(data, ref_obs=ref_obs, alpha=False, taper_fwhm_mas=MPOL["taper_mas"])
        scene, regs = grid_scene(npix, pixel_mas), []
    elif config == "F":
        npix = npix or FULL["npix"]
        pb = dict(freq_hz=FULL["nu0_hz"], dish_m=FULL["dish_m"], factor=FULL["pb_factor"])
        nuis = CalibrationNuisance(data, ref_obs=ref_obs, alpha=True, nu0_hz=FULL["nu0_hz"])
        scene = grid_scene(npix, pixel_mas, pb)
        # the pixels are sky x beam; the default image of the entropy is the beam, so the penalty is the
        # entropy of the sky relative to a flat sky (docs: "the default image q ... normalised here")
        regs = [MaxEntropy(float(lam), prior=np.exp(np.asarray(primary_beam(npix, pixel_mas, **pb))))]
    else:
        raise ValueError(f"config must be R or F, not {config!r}")
    term = use.term(scene, nuis, mask=None if mask is None else np.asarray(mask, bool), chunk_size=chunk_size)
    terms = [term] + ([MPoLEntropy(lam, pixel_mas=pixel_mas)] if config == "R" else [])
    priors = nuis.priors()
    priors["log_brightness"] = dist.ImproperUniform(dist.constraints.real, (), (npix, npix))
    init = {k: jnp.asarray(v) for k, v in nuis.defaults().items()}
    return types.SimpleNamespace(config=config, npix=npix, pixel_mas=pixel_mas, scene=scene, nuis=nuis, priors=priors, init=init,
                                 terms=terms, regs=regs, model=lambda **values: scene(values), lam=float(lam))


def start_values(prob, *, flux=0.25, fwhm_mas=1000.0, shat=None):
    """C9's start: a circular Gaussian of FWHM 1" and flux 0.25 Jy, s_d at ŝ_d (or 1)."""
    import jax.numpy as jnp

    from crosscheck.alma_maxent import gaussian_log_image

    v = dict(prob.init)
    v["log_brightness"] = jnp.asarray(gaussian_log_image(prob.npix, prob.pixel_mas, fwhm_mas))
    v["cal.flux"] = jnp.asarray(flux)
    if shat is not None and "cal.sigma" in v:
        v["cal.sigma"] = jnp.asarray(shat, float)
    return v


def run_fit(prob, init, *, max_steps, time_limit=None, eps=EPS_FIT, log=None):
    """L-BFGS through ``virgil.fit`` (C8). Returns ``(values, info, trajectory)``: the fitted values as
    numpy arrays, ``fit``'s diagnostics and one dict (steps, loss, grad_norm, elapsed) per chunk of
    steps. The loss is in -ln L units up to the constant of the priors' normalisation."""
    from virgil import fit

    traj = []

    def progress(d):
        traj.append({k: float(v) for k, v in d.items()})
        if log:
            log(f"  steps {d['steps']:6d}  loss {float(d['loss']):.6f}  |g| {float(d['grad_norm']):.3g}  {float(d['elapsed']):.0f} s")

    with ft_eps(eps), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fit(prob.model, prob.priors, (), regularizers=prob.regs, likelihoods=prob.terms, init=init, method="lbfgs",
                  max_steps=max_steps, time_limit=time_limit, progress=progress)
    values = {k: np.asarray(v) for k, v in res.values.items()}
    info = {k: (v.item() if hasattr(v, "item") else v) for k, v in res.info.items() if np.ndim(v) == 0}
    info.update({k: np.asarray(v).tolist() for k, v in res.info.items() if np.ndim(v) > 0})
    return values, info, traj


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


def per_point(data, scene, values, nuis, mask=None, eps=EPS_SCORE):
    """Per-point decomposition of -ln L (the complex Gaussian with variance s_d^2 / w_k per component):
    returns ``(ddid, obs, r2, nll)`` over the unmasked points, ``r2 = w |V - M|^2`` with the quoted weights
    and ``nll = r2 / (2 s_d^2) + 2 ln s_d + ln 2 pi - ln w``. The model is evaluated one observation at a time."""
    import jax.numpy as jnp

    obs, dd = np.asarray(data.obs), np.asarray(data.ddid)
    sigma = np.asarray(values["cal.sigma"], float) if "cal.sigma" in values else np.ones(int(dd.max()) + 1)
    jv = {k: jnp.asarray(v) for k, v in values.items()}
    keep = np.ones(len(obs), bool) if mask is None else np.asarray(mask, bool)
    out = dict(ddid=[], obs=[], r2=[], nll=[], index=[])
    with ft_eps(eps):
        for e in np.unique(obs):
            sel = (obs == e) & keep
            if not sel.any():
                continue
            sub = data.subset(jnp.asarray(sel))
            m = np.asarray(sub.model(scene(jv), jv, nuis))
            w, d = np.asarray(sub.weight, float), np.asarray(sub.ddid)
            r2 = w * np.abs(np.asarray(sub.vis) - m) ** 2
            out["ddid"].append(d)
            out["obs"].append(np.full(len(d), e))
            out["r2"].append(r2)
            out["nll"].append(0.5 * r2 / sigma[d] ** 2 + 2 * np.log(sigma[d]) + LN2PI - np.log(w))
            out["index"].append(np.flatnonzero(sel))
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in out.items()}


def score(data, scene, values, nuis, mask=None, annulus=None, n_ddid=None, n_annulus=12, eps=EPS_SCORE):
    """-ln L, chi^2 with the quoted weights (s = 1, C10) and with the fitted s, per ddid and (given an
    ``annulus`` per point) per uv annulus, over the unmasked points, at NUFFT accuracy ``eps`` (C7).
    ``chi2`` sums two squares per point, so chi^2/N of the tables is ``chi2 / (2 n)``."""
    pp = per_point(data, scene, values, nuis, mask, eps)
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


def residual_dirty_images(data, scene, values, nuis, npix=1024, pixel_mas=3.0, eps=EPS_SCORE):
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
            sub = data.subset(jnp.asarray(obs == e))
            r = np.asarray(sub.vis) - np.asarray(sub.model(scene(jv), jv, nuis))
            w = np.asarray(sub.weight, float) / sigma[np.asarray(sub.ddid)] ** 2
            out[int(e)] = gridded_dirty_image(np.asarray(sub.u), np.asarray(sub.v), r, w, npix, pixel_mas)
    return out


def image_of(values, npix, pixel_mas):
    """The sky image in Jy/arcsec^2 (virgil layout) of fitted values: ``flux * softmax(log_brightness) / dx^2``."""
    b = np.exp(values["log_brightness"] - np.max(values["log_brightness"]))
    return float(values["cal.flux"]) * (b / b.sum()) / (pixel_mas / 1000.0) ** 2


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


def save_folds(path, data, fold, annulus, params):
    """One file: ``fold_NN`` and ``annulus_NN`` (int8, in the order of the ddid's points in its npz file), the
    parameters and the sha256 of the assignment (``folds_sha256``) as a JSON string ``meta``."""
    ids = np.asarray(data.ddid_ids)[np.asarray(data.ddid)]
    arrays, by = {}, {}
    for d in np.unique(ids):
        sel = ids == d
        arrays[f"fold_{d:02d}"], arrays[f"annulus_{d:02d}"] = fold[sel], annulus[sel]
        by[int(d)] = fold[sel]
    meta = dict(params, sha256=folds_sha256(by), n=int(len(fold)), counts=np.bincount(fold, minlength=params["k"]).tolist())
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
    if len(np.unique(ids)) == len(set(int(k[5:]) for k in z.files if k.startswith("fold_"))):
        assert folds_sha256(by) == meta["sha256"], "fold file does not match its own checksum"
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
    return dict(
        log_brightness=jnp.asarray(np.log(base)),
        **{"cal.flux": jnp.asarray(base.sum() * dx**2), "cal.gain": jnp.asarray(amp[1:]),
           "cal.offset": jnp.asarray(1000.0 * np.stack([export["delta_ra_arcsec"][1:], export["delta_dec_arcsec"][1:]], axis=-1)),
           "cal.sigma": jnp.asarray(np.asarray(export["sigma"], float) * amp[np.asarray(obs_of_ddid)])},
    )


def r_objective(data, values, *, lam, mask=None, npix=None, pixel_mas=5.0, eps=EPS_SCORE):
    """R's objective -ln L + lam S_M at ``values``, in float64 at NUFFT accuracy ``eps`` (the
    normalising constant of -ln L included), and its two parts."""
    import jax.numpy as jnp

    npix = npix or values["log_brightness"].shape[0]
    prob = build_problem(data, "R", lam=lam, npix=npix, pixel_mas=pixel_mas, mask=mask)
    sc = score(data, prob.scene, values, prob.nuis, mask=mask, eps=eps)
    s = float(prob.terms[1].entropy({k: jnp.asarray(v) for k, v in values.items()}))
    return dict(nll=sc["nll"], entropy=s, penalty=lam * s, objective=sc["nll"] + lam * s)
