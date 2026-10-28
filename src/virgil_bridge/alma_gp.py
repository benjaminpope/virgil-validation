"""IM Lup as a Gaussian-process image: virgil's ``MultipoleField`` on the MPoL grid, with config R's nuisances.

The brightness is the softmax of a Matern field on a polar grid in the disc frame (``virgil.fields.MultipoleField``, provisional): smooth over
``length_r`` along a radius and ``length_theta * r`` along a ring, so it cannot make chains of dots or isolated dead pixels. Its whitened latents
have a standard-normal prior, which is the regulariser: no taper and no entropy. The geometry is fixed at DSHARP's (Huang et al. 2018, ApJ 869,
L42, "DSHARP II": i = 47.5 deg, PA = 144.5 deg) and the centre at the phase centre. GP imaging of radio interferometer data: Junklewitz et al. 2016
(RESOLVE, A&A 586, A76), Arras et al. 2019 (A&A 627, A134) and 2022 (Nat. Astron. 6, 259); the kernel and its JAX implementation follow Edenhofer et al.
2024 (NIFTy.re, JOSS 9, 6593), as ``virgil.fields``. All five checked on ADS, 2026-10-27.

The problem has the shape of ``alma.build_problem``'s (``scene``, ``nuis``, ``priors``, ``init``, ``terms``, ``model``), so ``alma.run_fit``,
``alma.score`` and ``alma.residual_dirty_images`` take it unchanged. The sites are ``field.latent`` (n_r, n_theta), ``field.sigma`` and the
nuisances' ``cal.*``; the nuisances have no taper (``blur_fwhm_mas=None``), since the field sets the resolution.
"""

import types

import numpy as np

from .alma import CHI2_ENGINES, TOEPLITZ_OFFSET_BOUND_MAS, nuisance_sites, require_blur_fwhm, toeplitz_term

GP = dict(npix=1028, pixel_mas=5.0, r_max_mas=2400.0, n_r=160, n_theta=256, inc_deg=47.5, pa_deg=144.5, nu=1.5, sigma_range=(0.05, 5.0), mean_floor=1e-3)
NODE_PER_LENGTH = 2.0  # the polar grid keeps at least this many nodes per correlation length (radially) when its size is automatic
OUTER_ARCSEC = 1.8  # the flux beyond this radius is reported: the disc is inside it
HELD_SITES = ("cal.gain", "cal.offset", "cal.sigma")


def auto_grid(r_max_mas, length_r, length_theta):
    """``(n_r, n_theta)`` of the polar grid for the wider hyperparameter grid (length_r down to 10 mas, length_theta down to 0.1 rad, r_max beyond 2.4 arcsec): the default 160 x 256 at r_max
    2.4 arcsec (15 mas spacing), scaled with ``r_max_mas``, and never coarser than ``NODE_PER_LENGTH`` nodes per ``length_r`` radially or per ``length_theta`` azimuthally."""
    n_r = max(int(round(GP["n_r"] * r_max_mas / GP["r_max_mas"])), int(np.ceil(NODE_PER_LENGTH * r_max_mas / length_r)))
    n_theta = max(GP["n_theta"], int(np.ceil(2 * np.pi * NODE_PER_LENGTH / length_theta)))
    return n_r, n_theta


def field_scene(npix, pixel_mas, *, r_max_mas=GP["r_max_mas"], inc=GP["inc_deg"], pa=GP["pa_deg"], length_r, length_theta, sigma=1.0, nu=GP["nu"], mean=None,
                mean_floor=GP["mean_floor"]):
    """``scene(values)``: ``Image(MultipoleField(values["field.latent"], ...))`` on the MPoL grid (``alma.grid_scene``'s half-pixel map), the field centred
    on the Image's. ``values["field.sigma"]`` (if present) replaces ``sigma``; the flux is ``cal.flux``. ``mean`` is a template of the image's shape."""
    from virgil import Image
    from virgil.fields import MultipoleField

    shift = -pixel_mas / 2.0

    def scene(values):
        field = MultipoleField(values["field.latent"], image_shape=(npix, npix), r_max_mas=r_max_mas, sigma=values.get("field.sigma", sigma), length_r=length_r,
                               length_theta=length_theta, inc=inc, pa=pa, nu=nu, mean=mean, mean_floor=mean_floor)
        return Image(field, pixel_scale_mas=pixel_mas, dra=shift, ddec=shift, ft="nufft")

    return scene


def _held_class():
    import equinox as eqx

    class HeldSites(eqx.Module):
        """A scalar-chi^2 likelihood term with some sites held: it is called with the fitted values and the held ones, which are not parameters of the fit
        (the nuisances of stage 1). Everything ``virgil.fit`` asks of a term is the wrapped term's."""

        term: object
        held: dict

        def chi2_and_log_norm(self, values):
            return self.term.chi2_and_log_norm({**self.held, **values})

        def derived(self, values):
            return self.term.derived({**self.held, **values})

        @property
        def has_log_norm(self):
            return self.term.has_log_norm

        @property
        def n_data(self):
            return self.term.n_data

        @property
        def chi2_constant(self):
            return self.term.chi2_constant

        @property
        def log_norm_constant(self):
            return self.term.log_norm_constant

        @property
        def loss_constant(self):
            return self.term.loss_constant

    return HeldSites


def build_gp_problem(data, *, length_r, length_theta, npix=GP["npix"], pixel_mas=GP["pixel_mas"], n_r=GP["n_r"], n_theta=GP["n_theta"], r_max_mas=GP["r_max_mas"],
                     inc=GP["inc_deg"], pa=GP["pa_deg"], nu=GP["nu"], mean=None, mean_floor=GP["mean_floor"], sigma=1.0, sigma_range=GP["sigma_range"], ref_obs=0,
                     offset_bound_mas=TOEPLITZ_OFFSET_BOUND_MAS, use=None, mask=None, chi2="toeplitz"):
    """The GP fit problem on ``data`` (``alma.build_problem``'s namespace): free sites ``field.latent`` (standard normal), ``field.sigma`` (log-uniform over
    ``sigma_range``, ``multipole_field_priors``), ``cal.flux`` and the per-observation gains, offsets and per-ddid error scales (``ref_obs`` has gain 1 and offset 0).
    ``length_r`` (mas) and ``length_theta`` (rad) are fixed hyperparameters; ``hold_nuisances`` fixes the nuisances. ``mean`` (an (npix, npix) template, peak > 0) sets the
    field's mean (the log of the template, floored at ``mean_floor`` of its peak). ``chi2`` is ``toeplitz`` (default) or ``direct``; ``use`` and ``mask`` as ``build_problem``."""
    import jax.numpy as jnp
    import numpyro.distributions as dist
    from virgil import CalibrationNuisance
    from virgil.fields import multipole_field_priors

    require_blur_fwhm()

    if chi2 not in CHI2_ENGINES:
        raise ValueError(f"chi2 must be one of {CHI2_ENGINES}, not {chi2!r}")
    use = data if use is None else use
    nuis = CalibrationNuisance(data, ref_obs=ref_obs, alpha=False, blur_fwhm_mas=None)
    scene = field_scene(npix, pixel_mas, r_max_mas=r_max_mas, inc=inc, pa=pa, length_r=length_r, length_theta=length_theta, sigma=sigma, nu=nu, mean=mean, mean_floor=mean_floor)
    m = None if mask is None else np.asarray(mask, bool)
    if chi2 == "toeplitz":
        term, chi2_info = toeplitz_term(use, scene, nuis, npix=npix, pixel_mas=pixel_mas, mask=m, max_offset_mas=float(offset_bound_mas))
    else:
        term, chi2_info = use.term(scene, nuis, mask=m), dict(engine="direct")
    priors, init, bound = nuisance_sites(nuis, data, offset_bound_mas=offset_bound_mas, ref_obs=ref_obs)
    sig = multipole_field_priors(path="field", sigma=sigma_range)["field.sigma"]
    priors = {**priors, "field.latent": dist.Normal(np.zeros((n_r, n_theta)), 1.0), "field.sigma": sig}
    init = {**init, "field.latent": jnp.zeros((n_r, n_theta)), "field.sigma": jnp.asarray(float(np.clip(sigma, *sigma_range)))}
    return types.SimpleNamespace(config="G", npix=npix, pixel_mas=pixel_mas, scene=scene, nuis=nuis, priors=priors, init=init, terms=[term], regs=[], model=lambda **values: scene(values),
                                 length_r=float(length_r), length_theta=float(length_theta), n_r=n_r, n_theta=n_theta, r_max_mas=float(r_max_mas), inc=float(inc), pa=float(pa),
                                 held=False, offset_bound_mas=float(bound.max()), offset_bound_obs=bound, offset_bound_requested=float(offset_bound_mas), chi2_info=chi2_info)


def hold_nuisances(prob, held):
    """``prob`` with the gains, offsets and error scales held at ``held`` (a dict with ``cal.gain``, ``cal.offset`` and ``cal.sigma``, e.g. a previous fit's values): they are no
    parameters, so the fit moves the field, its sigma and the flux. The Toeplitz precompute of ``prob`` is shared, not rebuilt."""
    import jax.numpy as jnp

    missing = [k for k in HELD_SITES if k not in held]
    if missing:
        raise ValueError(f"held nuisances lack {missing}")
    term = _held_class()(prob.terms[0], {k: jnp.asarray(held[k], float) for k in HELD_SITES})
    keep = lambda d: {k: v for k, v in d.items() if not k.startswith("cal.") or k == "cal.flux"}  # noqa: E731
    return types.SimpleNamespace(**{**vars(prob), "terms": [term], "priors": keep(prob.priors), "init": keep(prob.init), "held": True})


def term_loss(term, values):
    """-ln L of a data term at ``values`` up to its constants: 0.5 chi^2 plus the log normalisation, from either engine's own API."""
    import jax.numpy as jnp

    if hasattr(term, "chi2_and_log_norm"):
        chi2, log_norm = term.chi2_and_log_norm(values)
        return 0.5 * chi2 + log_norm
    r, log_norm = term.residuals_and_log_norm(values)
    return 0.5 * jnp.sum(r**2) + log_norm


def eval_seconds(prob, values, repeats=5):
    """Wall seconds of one value-and-gradient evaluation of the data term at ``values`` (compiled once, then the mean of ``repeats``)."""
    import time

    import equinox as eqx
    import jax

    vg = eqx.filter_jit(eqx.filter_value_and_grad(lambda x: term_loss(prob.terms[0], x)))
    v = {k: values[k] for k in prob.init}
    jax.block_until_ready(vg(v))
    t = time.time()
    for _ in range(repeats):
        jax.block_until_ready(vg(v))
    return (time.time() - t) / repeats


def image_of(prob, values):
    """The sky image in Jy/arcsec^2 (rows North first, as ``alma.image_of``) of the fitted ``values``: ``cal.flux`` times the unit-sum brightness over the pixel area."""
    return float(values["cal.flux"]) * np.asarray(prob.scene(values).brightness, float) / (prob.pixel_mas / 1000.0) ** 2


def flux_beyond(image, pixel_mas, radius_arcsec=OUTER_ARCSEC):
    """The flux (Jy) of ``image`` (Jy/arcsec^2) beyond ``radius_arcsec`` of the phase-centre cell."""
    from crosscheck.alma_maxent import phase_centre_cell

    n = image.shape[0]
    r0, c0 = phase_centre_cell(n)
    rr, cc = np.meshgrid(np.arange(n) - r0, np.arange(n) - c0, indexing="ij")
    return float(image[np.hypot(rr, cc) * pixel_mas / 1000.0 > radius_arcsec].sum() * (pixel_mas / 1000.0) ** 2)


def restored_clean(clean_dir, npix=GP["npix"], pixel_mas=GP["pixel_mas"]):
    """``(image, beam, info)``: the restored CLEAN image of ``alma_imlup_clean.py run`` (``image.fits``, Jy/beam) in Jy/arcsec^2 on the ``npix`` x ``pixel_mas`` grid, rows
    North first (``alma_start.regrid``: bilinear and flux-preserving within the field); ``beam`` is ``(bmaj_mas, bmin_mas, pa_deg)`` of its header."""
    import pathlib

    from crosscheck import alma_maxent as cx

    from .alma_start import clean_centre, regrid

    img, _, hdr = cx.read_fits(pathlib.Path(clean_dir) / "image.fits")
    beam = cx.beam_of(hdr)
    if img.shape[0] != img.shape[1] or img.shape[0] % 2 == 0:
        raise ValueError(f"the restored CLEAN image must be odd and square, not {img.shape}")
    px = abs(float(hdr["CDELT1"])) * 3.6e6
    sb = img[::-1] / cx.beam_area_arcsec2(beam[0], beam[1])  # Jy/beam -> Jy/arcsec^2, row 0 North
    out, info = regrid(sb, px, pixel_mas, npix, clean_centre(img.shape[0]))
    return out, beam, dict(info, clean_dir=str(clean_dir), beam_mas_deg=beam, src_pixel_mas=px)
