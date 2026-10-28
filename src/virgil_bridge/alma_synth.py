"""Synthetic truth for the IM Lup arms (Track A of ``design/alma_imlup_p3.md``): a smooth, dot-free image observed at the real uv and weights, and the scores of a reconstruction against it.

The truth is a supplied image (the default recipe: the classic-MEM arm's base image blurred to 15 mas, renormalised to 0.25 Jy), so it has no dots by construction. That recipe takes the truth from a candidate arm, which favours that arm in the NRMSE contest: ``alma_imlup_synth.py make`` records the origin in ``synth_make.json`` and ``synth.json`` (``truth_origin``), and an independent truth (the CLEAN model, an analytic ring) is the cross-check: pass criterion 3 on both. The data are its model
visibilities at the real (u, v) and weights with Gaussian noise of variance ``1 / w`` per component (the quoted weights, ``s = 1``) and no gains, offsets or error scales. A reconstruction is
scored by NRMSE of the images smoothed to a fraction of the CLEAN beam, by ``dots_report`` on its base image and by its flux, so a method that makes dots from smooth data shows it.
"""

import hashlib
import pathlib

import numpy as np

from .alma import CHUNK, MPOL, build_problem, dots_report, ft_eps, load_npz, model_chunks, tapered_image
from .alma_start import CRITERIA, GateRefused, truth_source_sha, truth_spec

TRUTH_BLUR_MAS, TRUTH_FLUX_JY = 15.0, 0.25
BEAM_FRACTIONS = (0.5, 1.0)


def truth_image(base, pixel_mas, blur_mas=TRUTH_BLUR_MAS, flux=TRUTH_FLUX_JY):
    """``base`` (Jy/arcsec^2, row 0 North) blurred to ``blur_mas`` (FFT Gaussian, clipped at 0) and scaled to total flux ``flux`` (Jy)."""
    from crosscheck.alma_maxent import convolve

    img = np.maximum(convolve(np.asarray(base, float), pixel_mas, blur_mas), 0.0) if blur_mas else np.maximum(np.asarray(base, float), 0.0)
    return img * flux / (img.sum() * (pixel_mas / 1000.0) ** 2)


def sha256_file(path):
    """sha256 hex of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def exponential_scale(enclosed=0.95):
    """``x = R / R_c`` at which a 2D exponential disk ``exp(-r / R_c)`` holds the fraction ``enclosed`` of its flux: ``1 - (1 + x) exp(-x) = enclosed``, by bisection."""
    lo, hi = 0.0, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if 1 - (1 + mid) * np.exp(-mid) < enclosed else (lo, mid)
    return 0.5 * (lo + hi)


def ring_gap_image(npix, pixel_mas, recipe):
    """The parametric IM Lup ring-and-gap disk of ``CRITERIA['synth']['truths']['ring_gap']`` (Jy/arcsec^2 up to scale, row 0 North, phase-centre cell ``phase_centre_cell(npix)``): an exponential holding ``dust_radius_enclosed_flux``
    of its flux within ``dust_radius_au``, a Gaussian gap (``gap_depth`` at ``gap_r_mas``, FWHM ``gap_fwhm_au``) and a Gaussian ring (``ring_contrast`` at ``ring_r_mas``, FWHM ``ring_fwhm_au``), in the deprojected radius of the disk
    (``inc_deg``, major axis at ``pa_deg`` East of North). Widths convert at ``distance_pc``."""
    from crosscheck.alma_maxent import phase_centre_cell

    r0, c0 = phase_centre_cell(npix)
    rr, cc = np.meshgrid(np.arange(npix), np.arange(npix), indexing="ij")
    north, east = (r0 - rr) * pixel_mas, (c0 - cc) * pixel_mas  # row 0 North, column 0 East
    pa, inc = np.deg2rad(recipe["pa_deg"]), np.deg2rad(recipe["inc_deg"])
    major = east * np.sin(pa) + north * np.cos(pa)
    minor = (east * np.cos(pa) - north * np.sin(pa)) / np.cos(inc)
    r = np.hypot(major, minor)
    mas_per_au = 1000.0 / recipe["distance_pc"]
    g = lambda r_c, fwhm_au: np.exp(-0.5 * ((r - r_c) / (fwhm_au * mas_per_au / (2 * np.sqrt(2 * np.log(2))))) ** 2)  # noqa: E731
    r_c = recipe["dust_radius_au"] * mas_per_au / exponential_scale(recipe["dust_radius_enclosed_flux"])
    return np.exp(-r / r_c) * (1 - recipe["gap_depth"] * g(recipe["gap_r_mas"], recipe["gap_fwhm_au"])) * (1 + recipe["ring_contrast"] * g(recipe["ring_r_mas"], recipe["ring_fwhm_au"]))


def registered_truth(truth_id, *, npix=None, pixel_mas=None, file=None, clean_dir=None):
    """``(truth, source_sha256, spec)`` of a truth registered in ``CRITERIA`` (a development truth is refused), at the registered blur and flux: the parametric ``ring_gap``; ``mpol_export`` from ``file`` (the export.npz, whose
    sha256 must be the registered one; array ``image``); ``clean_model`` from ``clean_dir`` (``model.fits``, registered sha256). ``npix`` and ``pixel_mas`` default to the registered grid and must equal it."""
    syn = CRITERIA["synth"]
    spec = truth_spec(truth_id)
    npix, pixel_mas = npix or syn["npix"], pixel_mas or syn["pixel_mas"]
    if (npix, pixel_mas) != (syn["npix"], syn["pixel_mas"]):
        raise GateRefused(f"registered truths are on the {syn['npix']} x {syn['pixel_mas']} mas grid, not {npix} x {pixel_mas}")
    want = truth_source_sha(spec)
    if spec["kind"] == "parametric":
        base, sha = ring_gap_image(npix, pixel_mas, spec["recipe"]), want
    elif spec["kind"] == "image":
        if file is None:
            raise GateRefused(f"truth {truth_id!r} needs its file (--truth-file)")
        sha = sha256_file(file)
        if sha != want:
            raise GateRefused(f"{file}: sha256 {sha} is not the registered {want}")
        base = np.asarray(np.load(file)["image"], float)
    else:
        from .alma_start import read_clean, regrid

        if clean_dir is None:
            raise GateRefused(f"truth {truth_id!r} needs --clean-dir")
        sha = sha256_file(pathlib.Path(clean_dir) / "model.fits")
        if want is None or sha != want:
            raise GateRefused(f"{clean_dir}/model.fits: sha256 {sha} is not a registered pin ({want}): pin it in CRITERIA before the synth run")
        model, px_s, _, _ = read_clean(clean_dir)
        from crosscheck.alma_maxent import convolve
        from .alma_start import clean_centre

        sb = convolve(model / (px_s / 1000.0) ** 2, px_s, spec["blur_mas"])
        base, _ = regrid(sb, px_s, pixel_mas, npix, clean_centre(model.shape[0]))
    blur = 0.0 if spec["kind"] == "clean" else spec["blur_mas"]  # the CLEAN model was blurred at its own pixel before regridding
    if base.shape != (npix, npix):
        raise GateRefused(f"truth {truth_id!r} is {base.shape}, not {npix} x {npix}")
    return truth_image(base, pixel_mas, blur, syn["flux_jy"]), sha, spec


def model_visibilities(data, truth, pixel_mas, *, chunk_size=CHUNK, eps=1e-12):
    """The model visibilities of ``truth`` (Jy/arcsec^2) at ``data``'s points: virgil's NUFFT on config R's grid, no blur and no nuisances."""
    import jax.numpy as jnp

    truth = np.asarray(truth, float)
    prob = build_problem(data, "R", lam=0.0, npix=truth.shape[0], pixel_mas=pixel_mas, blur_mas=0.0, image_only=True)
    jv = {k: jnp.asarray(v) for k, v in prob.init.items()}
    jv["log_brightness"] = jnp.asarray(np.log(np.maximum(truth, 1e-12 * truth.max())))
    jv["cal.flux"] = jnp.asarray(truth.sum() * (pixel_mas / 1000.0) ** 2)  # softmax normalises the pixels, so cal.flux sets the total
    out = np.zeros(len(data.u), complex)
    with ft_eps(eps):
        for idx, _, m in model_chunks(data, np.ones(len(data.u), bool), prob.scene, jv, prob.nuis, chunk_size):
            out[idx] = m
    return out


def noisy(model, weight, rng):
    """``model`` plus complex Gaussian noise of variance ``1 / weight`` per component."""
    sd = 1.0 / np.sqrt(np.asarray(weight, float))
    return model + sd * (rng.normal(size=len(model)) + 1j * rng.normal(size=len(model)))


def make_data(files, out_dir, truth, pixel_mas, *, seed=0, chunk_size=CHUNK, noise=True):
    """Write the synthetic ``ddid_NN.npz`` of each of ``files`` (the real ones) to ``out_dir``: the same columns, ``vis`` replaced by the model of ``truth`` plus noise from the weights (a ddid's
    generator is ``default_rng([seed, ddid])``). Returns the list of paths written."""
    import pathlib

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for f in files:
        z = dict(np.load(f))
        d = load_npz([z])
        vis = model_visibilities(d, truth, pixel_mas, chunk_size=chunk_size)
        if noise:
            vis = noisy(vis, z["weight"], np.random.default_rng([int(seed), int(z["ddid"])]))
        z.pop("re", None), z.pop("im", None)
        z["vis"] = vis
        path = out_dir / pathlib.Path(f).name
        np.savez(path, **z)
        written.append(path)
    return written


def smooth(image, pixel_mas, beam, fraction):
    """``image`` convolved with the elliptical Gaussian ``beam = (bmaj_mas, bmin_mas, pa_deg)`` scaled by ``fraction``."""
    from crosscheck.alma_maxent import convolve

    return convolve(image, pixel_mas, fraction * beam[0], fraction * beam[1], beam[2])


def score_image(image, truth, pixel_mas, beam, *, fractions=BEAM_FRACTIONS, taper_mas=MPOL["taper_mas"]):
    """The scores of a reconstruction ``image`` (Jy/arcsec^2, rows North first) against ``truth``, as ``dict`` for ``synth.json``:

    - ``nrmse``: ``{fraction: |S(image) - S(truth)| / |S(truth)|}`` with S the smoothing to that fraction of ``beam`` (``(bmaj, bmin)`` mas and PA deg; 0.5 and 1.0 of the CLEAN beam);
    - ``flux_jy``, ``flux_error`` (relative to the truth's flux);
    - ``dots``: ``dots_report`` of the base image against its ``taper_mas`` blur, with ``n_eff_ratio`` and ``base_dead_in_disk_1e4`` copied to the top level, which are the pass block's inputs."""
    image, truth = np.asarray(image, float), np.asarray(truth, float)
    area = (pixel_mas / 1000.0) ** 2
    nrmse = {}
    for f in fractions:
        st = smooth(truth, pixel_mas, beam, f)
        nrmse[f"{f:g}"] = float(np.linalg.norm(smooth(image, pixel_mas, beam, f) - st) / np.linalg.norm(st))
    dots = dots_report(image, tapered_image(image, pixel_mas, taper_mas), pixel_mas, taper_mas=taper_mas)
    flux, flux_t = float(image.sum() * area), float(truth.sum() * area)
    return dict(nrmse=nrmse, flux_jy=flux, truth_flux_jy=flux_t, flux_error=flux / flux_t - 1.0, n_eff_ratio=dots["n_eff_ratio"], base_dead_in_disk_1e4=dots["base_dead_in_disk_1e4"], dots=dots,
                beam_mas_deg=list(beam), pixel_mas=pixel_mas)
