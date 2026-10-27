"""IM Lup: a Cotton-Schwab multi-scale CLEAN image made with virgil.radio's public API, compared with DSHARP's.

    python scripts/alma_imlup_clean.py run --data DIR --out DIR [--dsharp FITS] [--npix N] [--pixel-mas P] [...]
    python scripts/alma_imlup_clean.py compare --clean DIR --dsharp FITS --out DIR

``run`` repeats the recipe of DSHARP's fiducial IM Lup image (``IMLup_continuum.py`` of the DSHARP release, the last
``tclean_wrapper`` call; Andrews et al. 2018, ApJL 869, L41, Sect. 4): multi-scale deconvolution (Gaussian scales
0, 10, 50, 150, 300, 450 pixels of 3 mas), Briggs ``robust = 0.5``, loop gain 0.1, ``smallscalebias`` 0.6, ``cyclefactor`` 1,
``cycleniter`` 300, ``niter`` 50000, an elliptical mask, threshold 0.05 mJy and the uv taper 33 x 26 mas at PA 137.53 deg.
The defaults are those numbers; every one is an argument. The design and the differences from ``tclean``
are in ``design/alma_clean.md``.

The major cycles are exact: the residual visibilities are ``V - M`` with ``M`` from ``ComplexVisData.model`` (the
nufftax transform of the current model image, in chunks) and the residual dirty image is
``ComplexVisData.dirty_image`` of them. The taper is applied to the visibilities, ``(V - M) T``, which gives the weights
``w f T`` (``f`` the Briggs factor computed without the taper, as in ``tclean``); the tapered PSF is the dirty image of
``T`` itself. The minor cycles are ``crosscheck.alma_clean.multiscale_minor`` (NumPy). The CLEAN image is restored with the
Gaussian fitted to the PSF's main lobe. Only public virgil API is used.
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from crosscheck import alma_clean as ac  # noqa: E402
from crosscheck import alma_maxent as cx  # noqa: E402

RA_DEG, DEC_DEG = (15 + 56 / 60 + 9.20 / 3600) * 15, -(37 + 56 / 60 + 6.42 / 3600)  # the MS's phase centre (docs/alma_conventions.md)
# DSHARP's fiducial recipe (IMLup_continuum.py, last tclean_wrapper call and common_mask)
DSHARP = dict(
    pixel_mas=3.0, imsize=4000, robust=0.5, gain=0.1, bias=0.6, cyclefactor=1.0, cycleniter=300, niter=50000, threshold_jy=5e-5,
    scales_pix=(0, 10, 50, 150, 300, 450), taper=(33.0, 26.0, 137.53),
    mask=dict(ra_deg=(15 + 56 / 60 + 9.18880 / 3600) * 15, dec_deg=-(37 + 56 / 60 + 6.527374 / 3600), semimajor_arcsec=2.75, semiminor_arcsec=1.75, pa_deg=145.0),
)
# IM Lup's disc geometry for the radial profile: Huang et al. 2018b (DSHARP III, ApJL 869, L42), i = 47.5 deg, PA = 144.5 deg
DISC = dict(incl_deg=47.5, pa_deg=144.5)


def say(*a):
    print(*a, flush=True)


def dump(path, obj):
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    path = pathlib.Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=default))
    tmp.replace(path)


# ---------------------------------------------------------------- the grid and FITS


def grid_wcs(npix, pixel_mas, ra_deg=RA_DEG, dec_deg=DEC_DEG):
    """SIN WCS of an array in FITS orientation (row 0 South) whose centre pixel ``(npix - 1) / 2`` is the phase centre.
    ``npix`` is odd, so the centre is a pixel and the PSF peaks on it."""
    from astropy.wcs import WCS

    c = (npix - 1) / 2
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.crval = [ra_deg, dec_deg]
    w.wcs.crpix = [c + 1, c + 1]  # FITS counts from 1; the centre row from the South is the centre row from the North
    w.wcs.cdelt = [-pixel_mas / 3.6e6, pixel_mas / 3.6e6]
    w.wcs.cunit = ["deg", "deg"]
    return w


def write_fits(path, image, pixel_mas, bunit, beam=None, **header):
    """``image`` (row 0 North) as a FITS file on ``grid_wcs``; ``beam`` = (bmaj_mas, bmin_mas, pa_deg) goes in BMAJ/BMIN/BPA."""
    from astropy.io import fits

    hdr = grid_wcs(image.shape[0], pixel_mas).to_header()
    hdr["BUNIT"] = bunit
    if beam is not None:
        hdr["BMAJ"], hdr["BMIN"], hdr["BPA"] = beam[0] / 3.6e6, beam[1] / 3.6e6, beam[2]
    for k, v in header.items():
        hdr[k] = v
    fits.PrimaryHDU(np.asarray(image, np.float32)[::-1], hdr).writeto(path, overwrite=True)


def mask_cell(npix, pixel_mas, m):
    """(row, col) of the mask centre on the grid (row 0 North), from its sky position."""
    x, y = grid_wcs(npix, pixel_mas).all_world2pix([m["ra_deg"]], [m["dec_deg"]], 0)
    return (npix - 1) - float(y[0]), float(x[0])


# -------------------------------------------------------- virgil: imaging and the exact model


def tapered(data, vis, taper):
    """``data`` with its visibilities replaced by ``vis * T`` (``T`` the uv taper; none if ``taper`` is None)."""
    if taper is not None:
        vis = vis * ac.uv_taper(np.asarray(data.u), np.asarray(data.v), *taper)
    return eqx.tree_at(lambda d: d.vis, data, jnp.asarray(vis, dtype=complex))


class Imager:
    """The residual dirty image of ``data`` for a model image, and the PSF, on one grid with one weighting."""

    def __init__(self, data, npix, pixel_mas, robust, taper, uv_cell_npix, chunk_size=10**6):
        from virgil.radio import CalibrationNuisance

        if npix % 2 == 0:
            raise ValueError("npix must be odd (the phase centre is then a pixel, and the PSF peaks on it)")
        self.data, self.npix, self.pixel_mas, self.robust, self.taper, self.chunk = data, int(npix), float(pixel_mas), robust, taper, int(chunk_size)
        self.uv_cell = 1.0 / (uv_cell_npix * pixel_mas * ac.MAS)  # CASA grids the Briggs density on 1 / (imsize cell)
        self.nuis = CalibrationNuisance(data, gains=False, offsets=False, sigmas=False, alpha=False)
        self.vis = np.asarray(data.vis, dtype=complex)

    def _dirty(self, vis):
        d = tapered(self.data, vis, self.taper)
        return d.dirty_image(self.npix, self.pixel_mas, weighting="briggs", robust=self.robust, uv_cell=self.uv_cell, chunk_size=self.chunk)

    def psf(self):
        """(PSF with peak 1, normalisation): the image of the tapered unit visibilities; dividing a dirty image by the
        normalisation puts it in Jy/beam of this PSF."""
        ones = np.ones(len(self.data), complex)
        raw = tapered(self.data, ones, self.taper).dirty_image(
            self.npix, self.pixel_mas, weighting="briggs", robust=self.robust, uv_cell=self.uv_cell, chunk_size=self.chunk
        )
        c = (self.npix - 1) // 2
        return raw / raw[c, c], float(raw[c, c])

    def model_vis(self, image):
        """Visibilities of a signed model image (Jy per pixel) at the data's points: the positive and the negative part
        are each an ``Image`` (virgil's images are non-negative), evaluated chunk by chunk."""
        from virgil import Image

        out = np.zeros(len(self.data), complex)
        for sign, part in ((1.0, np.maximum(image, 0.0)), (-1.0, np.maximum(-image, 0.0))):
            total = float(part.sum())
            if total <= 0:
                continue
            sup = part > 0
            scene = Image(jnp.asarray(np.log(np.where(sup, part / total, 1.0))), pixel_scale_mas=self.pixel_mas, support=jnp.asarray(sup), ft="nufft")
            values = {"cal.flux": total}
            for lo in range(0, len(self.data), self.chunk):
                chunk = jax.tree.map(lambda a: a[lo : lo + self.chunk], self.data)
                out[lo : lo + self.chunk] += sign * np.asarray(chunk.model(scene, values, self.nuis))
        return out

    def residual(self, image, norm):
        """Residual dirty image ``(V - M) * T`` in Jy/beam (``image`` None or all zero: the dirty image)."""
        vis = self.vis if image is None or not np.any(image) else self.vis - self.model_vis(image)
        return self._dirty(vis) / norm


# ------------------------------------------------------------------------ run


def load_data(path, ddids=None):
    import virgil_bridge.alma as va

    return va.load_dir(path, ddids)


def clean(imager, *, scales_pix, mask, gain, bias, threshold, niter, cycleniter, cyclefactor, patch, time_limit, log=say):
    """Cotton-Schwab loop. Returns a dict: model (Jy/pixel), components, residual, psf, beam, history."""
    t0 = time.time()
    psf, norm = imager.psf()
    bmaj, bmin, pa, _ = ac.fit_beam(psf, imager.pixel_mas)
    side = ac.sidelobe_level(psf)
    log(f"PSF: beam {bmaj:.1f} x {bmin:.1f} mas, PA {pa:.1f} deg; highest sidelobe {side:.3f}")
    comps = np.zeros((len(scales_pix),) + psf.shape)
    model = np.zeros(psf.shape)
    history, total = [], 0
    stop = "niter"
    while True:
        resid = imager.residual(model, norm)
        peak = float(np.abs(resid[mask]).max())
        history.append(dict(major=len(history), iterations=total, peak_residual_jy=peak, model_flux_jy=float(model.sum()), wall_s=time.time() - t0))
        log(f"major {len(history) - 1}: {total} iterations, peak residual {peak * 1e3:.4f} mJy/beam, model {model.sum():.5f} Jy, {time.time() - t0:.0f} s")
        if peak < threshold:
            stop = "threshold"
            break
        if total >= niter:
            break
        if time_limit and time.time() - t0 > time_limit:
            stop = "time_limit"
            break
        cyc = max(threshold, min(max(cyclefactor * side, 0.05), 0.8) * peak)  # tclean's minpsffraction/maxpsffraction
        inc, n, _ = ac.multiscale_minor(resid, psf, scales_pix, mask, gain=gain, niter=min(cycleniter, niter - total), threshold=cyc, bias=bias, patch=patch)
        if n == 0:
            stop = "no_progress"
            break
        comps += inc
        total += n
        model = ac.model_image(comps, scales_pix)
    return dict(model=model, components=comps, residual=resid, psf=psf, beam=(bmaj, bmin, pa), history=history, stop=stop, iterations=total, norm=norm, sidelobe=side)


def virgil_clean_kwargs(args, scales_pix, mask):
    """Keywords of ``virgil.radio_clean.clean`` (after the data, ``npix`` and ``pixel_scale_mas``). the public keyword is ``scales_mas`` (mas); vv passed ``scales`` (pixels), which virgil never accepted."""
    return dict(
        deconvolver="multiscale", scales_mas=[float(s) * args.pixel_mas for s in scales_pix], weighting="briggs", robust=args.robust,
        uvtaper=None if args.no_taper else tuple(args.taper), niter=args.niter, gain=args.gain, threshold=args.threshold_jy,
        cycleniter=args.cycleniter, cyclefactor=args.cyclefactor, smallscalebias=args.bias, psfpatch=args.patch, mask=mask,
        restoringbeam=args.beam, uv_cell=1.0 / (args.uv_cell_npix * args.pixel_mas * ac.MAS), chunk_size=args.chunk_size,
        time_limit=args.time_limit or None, verbose=True,
    )


def clean_virgil(data, args, scales_pix, mask, log=say):
    """The same loop by ``virgil.radio_clean.clean`` (the feature under test); returns ``clean``'s dict (no components)."""
    from virgil.radio_clean import clean as vclean

    res = vclean(data, args.npix, args.pixel_mas, **virgil_clean_kwargs(args, scales_pix, mask))
    history = [dict(major=i, iterations=h["iterations"], peak_residual_jy=h["peak_residual"], model_flux_jy=h["model_flux"], wall_s=h["wall_s"]) for i, h in enumerate(res.history)]
    return dict(
        model=res.model, components=None, residual=res.residual, psf=res.psf, beam=res.beam, history=history, stop=res.stop,
        iterations=res.iterations, sidelobe=ac.sidelobe_level(res.psf),
    )


def render_png(path, image, pixel_mas, beam, title, unit, half_arcsec=2.5):
    """``image`` (row 0 North, column 0 East, the phase centre at the middle) as a PNG of the central ``2 half_arcsec``: East to the left, the beam ellipse (``beam`` = bmaj, bmin
    mas and PA deg East of North) in the lower left, ``unit`` on the colour bar. Residuals get a symmetric diverging scale, other images an asinh-free linear one from 0."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    n = image.shape[0]
    h = int(round(half_arcsec * 1000 / pixel_mas))
    c = (n - 1) // 2
    crop = np.asarray(image, float)[max(c - h, 0): c + h + 1, max(c - h, 0): c + h + 1]
    half = crop.shape[0] // 2 * pixel_mas / 1000.0
    signed = "residual" in title.lower()
    lim = float(np.percentile(np.abs(crop), 99.9)) or 1.0
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(crop, origin="upper", extent=[half, -half, -half, half], cmap="RdBu_r" if signed else "magma", vmin=-lim if signed else 0.0, vmax=lim if signed else float(crop.max()))
    ax.add_patch(Ellipse((0.8 * half, -0.8 * half), beam[0] / 1000.0, beam[1] / 1000.0, angle=90.0 - beam[2], fill=False, edgecolor="w" if not signed else "k", lw=1.2))
    ax.set(xlabel="East offset (arcsec)", ylabel="North offset (arcsec)", title=title)
    fig.colorbar(im, ax=ax, label=unit, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def cmd_run(args):
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load_data(args.data, args.ddids)
    say(f"data: {len(data):,} points from {args.data}")
    taper = None if args.no_taper else tuple(args.taper)
    imager = Imager(data, args.npix, args.pixel_mas, args.robust, taper, args.uv_cell_npix, args.chunk_size)
    scales_pix = [s * DSHARP["pixel_mas"] / args.pixel_mas for s in args.scales_pix] if args.scales_in_dsharp_pixels else list(args.scales_pix)
    m = DSHARP["mask"]
    r, c = mask_cell(args.npix, args.pixel_mas, m)
    mask = ac.ellipse_mask((args.npix, args.npix), (r, c), m["semimajor_arcsec"] * 1000 / args.pixel_mas, m["semiminor_arcsec"] * 1000 / args.pixel_mas, m["pa_deg"])
    t0 = time.time()
    if args.engine == "virgil":
        res = clean_virgil(data, args, scales_pix, mask)
    else:
        res = clean(
            imager, scales_pix=scales_pix, mask=mask, gain=args.gain, bias=args.bias, threshold=args.threshold_jy, niter=args.niter,
            cycleniter=args.cycleniter, cyclefactor=args.cyclefactor, patch=args.patch, time_limit=args.time_limit,
        )
    bmaj, bmin, pa = res["beam"]
    if args.beam:
        bmaj, bmin, pa = args.beam
        say(f"restoring beam set to {bmaj} x {bmin} mas, PA {pa} deg")
    px = args.pixel_mas
    restored = ac.restore(res["model"], res["residual"], (bmaj / px, bmin / px, pa))
    beam = (bmaj, bmin, pa)
    write_fits(out / "image.fits", restored, px, "Jy/beam", beam, ORIGIN=f"virgil.radio_clean ({args.engine} engine)", OBJECT="IM Lup")
    write_fits(out / "residual.fits", res["residual"], px, "Jy/beam", beam)
    write_fits(out / "model.fits", res["model"], px, "Jy/pixel")
    write_fits(out / "psf.fits", res["psf"], px, "", beam)
    for name, arr in (("image", restored), ("model", res["model"]), ("residual", res["residual"]), ("psf", res["psf"])):
        np.save(out / f"{name}.npy", np.asarray(arr, np.float32))
    np.save(out / "beam.npy", np.asarray(beam, float))  # bmaj_mas, bmin_mas, pa_deg
    for name, arr, unit in (("image", restored, "Jy/beam"), ("model", res["model"], "Jy/pixel"), ("residual", res["residual"], "Jy/beam")):
        render_png(out / f"{name}.png", arr, px, beam, f"IM Lup CLEAN {name} ({args.engine} engine)", unit)
    if res["components"] is not None:
        np.savez_compressed(out / "components.npz", **{f"scale_{s:g}": np.nonzero(cmp) + (cmp[np.nonzero(cmp)],) for s, cmp in zip(scales_pix, res["components"])})
    beam_area_pix = np.pi * bmaj * bmin / (4 * np.log(2)) / px**2
    flux_mask = float(restored[mask].sum() / beam_area_pix)
    summary = dict(
        command="run", args={k: v for k, v in vars(args).items() if k != "func"}, scales_pix=scales_pix, beam_mas_deg=beam, sidelobe=res["sidelobe"], stop=res["stop"],
        iterations=res["iterations"], majors=len(res["history"]), model_flux_jy=float(res["model"].sum()), flux_in_mask_jy=flux_mask,
        peak_jy_per_beam=float(restored.max()), n_points=len(data), wall_s=time.time() - t0, history=res["history"], versions=versions(),
    )
    area_px = px**2 / 1e6  # arcsec^2
    model_sb = res["model"] / area_px
    summary.update(restored_flux_jy=float(restored.sum() / beam_area_pix), dots=va_dots(model_sb, px))
    dump(out / "clean.json", summary)
    dump(out / "result.json", dict(command="run", flux=dict(model_sum_jy=summary["model_flux_jy"], restored_jy=summary["restored_flux_jy"], restored_in_mask_jy=flux_mask), dots=summary["dots"],
                                   stop=res["stop"], iterations=res["iterations"], beam_mas_deg=beam, pixel_mas=px, wall_s=summary["wall_s"], versions=summary["versions"]))
    say(f"done: {res['stop']}, {res['iterations']} iterations, {len(res['history'])} major cycles, model {summary['model_flux_jy']:.5f} Jy, flux in mask {flux_mask:.5f} Jy, peak {summary['peak_jy_per_beam'] * 1e3:.4f} mJy/beam, {summary['wall_s']:.0f} s")
    if args.dsharp:
        compare(out, args.dsharp, out)


def va_dots(model_sb, pixel_mas):
    """``dots_report`` of the CLEAN model (Jy/arcsec^2) against its 30 mas tapering (MPoL's beam): the model is a set of Gaussian components on a pixel grid."""
    import virgil_bridge.alma as va

    return va.dots_report(model_sb, va.tapered_image(model_sb, pixel_mas), pixel_mas)


def versions():
    import importlib.metadata as m
    import subprocess

    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        sha = None
    return dict(python=sys.version.split()[0], jax=jax.__version__, devices=[str(d) for d in jax.devices()], virgil=m.version("virgil-astro"), numpy=np.__version__, vv=sha)


# -------------------------------------------------------------------- compare


def _fits_beam(hdr):
    return cx.beam_of(hdr)


def compare(clean_dir, dsharp_path, out):
    """Compare ``clean_dir/image.fits`` with DSHARP's image; write ``compare.json``, ``summary.md`` and ``compare.png``."""
    clean_dir, out = pathlib.Path(clean_dir), pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ours, w_o, h_o = cx.read_fits(clean_dir / "image.fits")  # FITS orientation, row 0 South
    d, w_d, h_d = cx.read_fits(dsharp_path)
    run = json.loads((clean_dir / "clean.json").read_text()) if (clean_dir / "clean.json").exists() else {}
    bo, bd = _fits_beam(h_o), _fits_beam(h_d)
    px_o, px_d = abs(h_o["CDELT1"]) * 3.6e6, abs(h_d["CDELT1"]) * 3.6e6
    area = lambda b, px: np.pi * b[0] * b[1] / (4 * np.log(2)) / px**2  # noqa: E731  beam area in pixels

    # onto DSHARP's grid (bilinear), valid where our field covers it
    ours_d = cx.resample(ours, w_o, w_d, d.shape)
    cover = cx.resample(np.ones_like(ours), w_o, w_d, d.shape) > 0.999
    # DSHARP's mask centre on its grid
    m = DSHARP["mask"]
    x, y = w_d.all_world2pix([m["ra_deg"]], [m["dec_deg"]], 0)
    centre_d = (float(y[0]), float(x[0]))  # (row, col) in FITS orientation: the rows run South to North, so the geometry below uses the flipped copy
    flip = lambda a: a[::-1]  # noqa: E731  to row 0 North, the convention of crosscheck.alma_clean
    d_n, o_n, cover_n = flip(d), flip(ours_d), flip(cover)
    centre_n = (d.shape[0] - 1 - centre_d[0], centre_d[1])
    xp, yp = w_d.all_world2pix([RA_DEG], [DEC_DEG], 0)  # the phase centre: a sign or conjugation error of the transform rotates the image about it
    phase_n = (d.shape[0] - 1 - float(yp[0]), float(xp[0]))
    mask_d = ac.ellipse_mask(d.shape, centre_n, m["semimajor_arcsec"] * 1000 / px_d, m["semiminor_arcsec"] * 1000 / px_d, m["pa_deg"])
    mask_in = mask_d & cover_n
    # the noise annulus of DSHARP's script (annulus[[mask centre], [1.1 x the mask's semi-major axis, 5.75 arcsec]]), where our field covers it
    r_in, r_out = 1.1 * m["semimajor_arcsec"] * 1000 / px_d, 5.75e3 / px_d
    ring_cover = cover_n.copy()
    exclude = mask_d | ~ring_cover
    rms_o, n_o = ac.annulus_rms(o_n, centre_n, r_in, r_out, exclude=exclude)
    rms_d, _ = ac.annulus_rms(d_n, centre_n, r_in, r_out, exclude=exclude)

    stats = dict(
        flux_in_mask_jy=dict(ours=float(o_n[mask_in].sum() / area(bo, px_d)), dsharp=float(d_n[mask_in].sum() / area(bd, px_d))),
        peak_jy_per_beam=dict(ours=float(o_n[cover_n].max()), dsharp=float(d_n[cover_n].max())),
        beam=dict(ours_mas_mas_deg=bo, dsharp_mas_mas_deg=bd),
        rms_annulus_jy_per_beam=dict(ours=rms_o, dsharp=rms_d, n_pixels=n_o),
        model_flux_jy=run.get("model_flux_jy"),
        difference=cx.difference_stats(o_n, d_n, region=cover_n),
        difference_in_mask=cx.difference_stats(o_n, d_n, region=mask_in),
    )
    # orientation check: the identity beats any reflection of our image against DSHARP's
    corr = {}
    for name, f in (("identity", lambda a: a), ("flip_east_west", lambda a: a[:, ::-1]), ("flip_north_south", lambda a: a[::-1]), ("rot180", lambda a: a[::-1, ::-1])):
        corr[name] = float(np.corrcoef(f(o_n)[mask_in].ravel(), d_n[mask_in].ravel())[0, 1]) if name == "identity" else float(_reflected_corr(o_n, d_n, phase_n, name, mask_in))
    stats["orientation_correlation"] = corr
    stats["orientation_ok"] = bool(corr["identity"] > max(v for k, v in corr.items() if k != "identity"))
    # centroid of the brightness inside the mask, in mas East and North (ours - DSHARP's)
    def centroid(a):
        rr, cc = np.indices(a.shape, dtype=float)
        wgt = np.where(mask_in, np.maximum(a, 0), 0)
        return (-(cc * wgt).sum() / wgt.sum() * px_d, -(rr * wgt).sum() / wgt.sum() * px_d)

    co, cd = centroid(o_n), centroid(d_n)
    stats["centroid_offset_mas_east_north"] = (co[0] - cd[0], co[1] - cd[1])
    # radial profiles (deprojected)
    edges = np.arange(0.0, 3.0, 0.02)
    mid, prof_o, _ = ac.radial_profile(np.where(cover_n, o_n, np.nan), centre_n, px_d / 1000, DISC["incl_deg"], DISC["pa_deg"], edges)
    _, prof_d, _ = ac.radial_profile(np.where(cover_n, d_n, np.nan), centre_n, px_d / 1000, DISC["incl_deg"], DISC["pa_deg"], edges)
    stats["radial_profile_rms_diff_over_peak"] = float(np.sqrt(np.nanmean((prof_o - prof_d) ** 2)) / np.nanmax(prof_d))
    dump(out / "compare.json", dict(stats, clean=run.get("args"), versions=versions()))
    np.savez_compressed(out / "profiles.npz", r_arcsec=mid, ours=prof_o, dsharp=prof_d)
    _figure(out / "compare.png", o_n, d_n, cover_n, centre_n, px_d, mid, prof_o, prof_d, stats)
    _summary(out / "summary.md", stats, run)
    say((out / "summary.md").read_text())


def _reflected_corr(o_n, d_n, centre, name, region):
    """Correlation of DSHARP's image with a reflection of ours about ``centre``, the phase centre: a V -> V* or u -> -u error
    of the transform reflects the image about it, so it moves a disc that is off the phase centre (IM Lup's is ~0.17 arcsec off)
    by twice that offset; pixels the reflection takes outside the image count as zero."""
    r0, c0 = round(centre[0]), round(centre[1])
    h, w = o_n.shape
    rr, cc = np.indices(o_n.shape)
    nr = 2 * r0 - rr if "north" in name or name == "rot180" else rr
    nc = 2 * c0 - cc if "east" in name or name == "rot180" else cc
    ok = (nr >= 0) & (nr < h) & (nc >= 0) & (nc < w)
    refl = np.where(ok, o_n[np.clip(nr, 0, h - 1), np.clip(nc, 0, w - 1)], 0.0)
    return np.corrcoef(refl[region].ravel(), d_n[region].ravel())[0, 1]


def _figure(path, o_n, d_n, cover, centre, px, r, prof_o, prof_d, stats):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    half = int(1.6e3 / px)  # +- 1.6 arcsec
    r0, c0 = int(round(centre[0])), int(round(centre[1]))
    sl = (slice(r0 - half, r0 + half + 1), slice(c0 - half, c0 + half + 1))
    ext = [half * px / 1e3, -half * px / 1e3, -half * px / 1e3, half * px / 1e3]  # East to the left, North up
    vmax = float(np.nanmax(d_n[sl]))
    fig, ax = plt.subplots(2, 3, figsize=(15, 9.5), constrained_layout=True)
    for a, im, t in ((ax[0, 0], o_n, "virgil CLEAN (this work)"), (ax[0, 1], d_n, "DSHARP fiducial")):
        h = a.imshow(np.where(cover, im, np.nan)[sl] * 1e3, origin="upper", extent=ext, vmin=-0.05 * vmax * 1e3, vmax=vmax * 1e3, cmap="inferno")
        a.set_title(t)
        a.set_xlabel("ΔRA (arcsec; East left)")
        a.set_ylabel("ΔDec (arcsec)")
        fig.colorbar(h, ax=a, label="mJy/beam")
    diff = (o_n - d_n)
    lim = 0.1 * vmax * 1e3
    h = ax[0, 2].imshow(np.where(cover, diff, np.nan)[sl] * 1e3, origin="upper", extent=ext, vmin=-lim, vmax=lim, cmap="RdBu_r")
    ax[0, 2].set_title("ours − DSHARP (±10% of peak)")
    fig.colorbar(h, ax=ax[0, 2], label="mJy/beam")
    ax[1, 0].plot(r, prof_o * 1e3, label="virgil CLEAN")
    ax[1, 0].plot(r, prof_d * 1e3, "--", label="DSHARP")
    ax[1, 0].set(xlabel="deprojected radius (arcsec)", ylabel="mean mJy/beam")
    ax[1, 0].legend()
    ax[1, 0].set_title("radial profile")
    ax[1, 1].plot(r, (prof_o - prof_d) * 1e3)
    ax[1, 1].axhline(0, color="k", lw=0.5)
    ax[1, 1].set(xlabel="deprojected radius (arcsec)", ylabel="ours − DSHARP (mJy/beam)", title="profile difference")
    ax[1, 2].axis("off")
    ax[1, 2].text(0, 1, _summary_text(stats), va="top", family="monospace", fontsize=9)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _summary_text(s):
    f, p, b, rms = s["flux_in_mask_jy"], s["peak_jy_per_beam"], s["beam"], s["rms_annulus_jy_per_beam"]
    return "\n".join([
        f"flux in mask  ours {f['ours'] * 1e3:.2f} mJy  DSHARP {f['dsharp'] * 1e3:.2f} mJy",
        f"peak          ours {p['ours'] * 1e3:.3f}  DSHARP {p['dsharp'] * 1e3:.3f} mJy/beam",
        f"beam ours     {b['ours_mas_mas_deg'][0]:.1f} x {b['ours_mas_mas_deg'][1]:.1f} mas @ {b['ours_mas_mas_deg'][2]:.0f}°",
        f"beam DSHARP   {b['dsharp_mas_mas_deg'][0]:.1f} x {b['dsharp_mas_mas_deg'][1]:.1f} mas @ {b['dsharp_mas_mas_deg'][2]:.0f}°",
        f"rms annulus   ours {rms['ours'] * 1e6:.1f}  DSHARP {rms['dsharp'] * 1e6:.1f} µJy/beam",
        f"correlation   {s['difference']['correlation']:.4f} (all), {s['difference_in_mask']['correlation']:.4f} (mask)",
        f"rms diff/peak {s['difference']['rms_over_peak']:.4f}",
    ])


def _summary(path, s, run):
    f, p, b, rms = s["flux_in_mask_jy"], s["peak_jy_per_beam"], s["beam"], s["rms_annulus_jy_per_beam"]
    lines = [
        "# IM Lup: virgil CLEAN against DSHARP's fiducial CLEAN image",
        "",
        "| | virgil CLEAN | DSHARP | ratio |",
        "|---|---|---|---|",
        f"| flux inside the mask (mJy) | {f['ours'] * 1e3:.3f} | {f['dsharp'] * 1e3:.3f} | {f['ours'] / f['dsharp']:.4f} |",
        f"| peak (mJy/beam) | {p['ours'] * 1e3:.4f} | {p['dsharp'] * 1e3:.4f} | {p['ours'] / p['dsharp']:.4f} |",
        f"| beam major × minor (mas), PA (deg) | {b['ours_mas_mas_deg'][0]:.2f} × {b['ours_mas_mas_deg'][1]:.2f}, {b['ours_mas_mas_deg'][2]:.1f} | {b['dsharp_mas_mas_deg'][0]:.2f} × {b['dsharp_mas_mas_deg'][1]:.2f}, {b['dsharp_mas_mas_deg'][2]:.1f} | |",
        f"| rms in the off-source annulus (µJy/beam) | {rms['ours'] * 1e6:.2f} | {rms['dsharp'] * 1e6:.2f} | {rms['ours'] / rms['dsharp']:.3f} |",
        "",
        f"CLEAN model flux {s['model_flux_jy']} Jy. Difference image on DSHARP's grid (where our field covers it): rms/peak {s['difference']['rms_over_peak']:.4f}, max |diff|/peak {s['difference']['max_abs_over_peak']:.4f}, correlation {s['difference']['correlation']:.5f}; inside the mask: rms/peak {s['difference_in_mask']['rms_over_peak']:.4f}, correlation {s['difference_in_mask']['correlation']:.5f}.",
        f"Radial-profile rms difference / peak: {s['radial_profile_rms_diff_over_peak']:.4f}. Centroid offset (ours − DSHARP): {s['centroid_offset_mas_east_north'][0]:.3f} mas East, {s['centroid_offset_mas_east_north'][1]:.3f} mas North.",
        "",
        "Orientation check (correlation inside the mask of DSHARP with our image and with its reflections about the phase centre): "
        + ", ".join(f"{k} {v:.4f}" for k, v in s["orientation_correlation"].items()) + ". The identity must be the largest" + ("." if s.get("orientation_ok", True) else ": **IT IS NOT: the image may be reflected or conjugated.**"),
        "",
        "![comparison](compare.png)",
        "",
        "Expected differences (details in `design/alma_clean.md`): DSHARP's self-calibration used its own CLEAN models, so the data already carry its image (the comparison is of the two CLEANs of the same visibilities, not of independent reductions); `tclean` and our minor cycles differ in detail (scale kernels, scale-selection normalisation, PSF window, stopping); our field is smaller than DSHARP's 12 arcsec; the PSF is bilinearly resampled from our grid to DSHARP's.",
    ]
    if run:
        lines += ["", f"Run: {run.get('iterations')} minor iterations in {run.get('majors')} major cycles ({run.get('stop')}), {run.get('n_points'):,} visibilities, {run.get('wall_s', 0):.0f} s."]
    pathlib.Path(path).write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="CLEAN the visibilities and compare with DSHARP if --dsharp is given")
    p.add_argument("--data", required=True, help="directory of ddid_NN.npz (no default)")
    p.add_argument("--ddids", type=int, nargs="*", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--dsharp", default=None, help="DSHARP's IMLup_continuum.fits: compare after the run")
    p.add_argument("--npix", type=int, default=3001, help="odd; 3001 x 3 mas = 9 arcsec (DSHARP: 4000 pixels, 12 arcsec)")
    p.add_argument("--pixel-mas", type=float, default=DSHARP["pixel_mas"])
    p.add_argument("--uv-cell-npix", type=int, default=DSHARP["imsize"], help="Briggs density cell = 1 / (this * pixel): DSHARP's 4000 pixels")
    p.add_argument("--robust", type=float, default=DSHARP["robust"])
    p.add_argument("--taper", type=float, nargs=3, default=list(DSHARP["taper"]), metavar=("BMAJ_MAS", "BMIN_MAS", "PA_DEG"))
    p.add_argument("--no-taper", action="store_true")
    p.add_argument("--scales-pix", type=float, nargs="+", default=list(DSHARP["scales_pix"]), help="FWHM of the Gaussian scales")
    p.add_argument("--scales-in-dsharp-pixels", action=argparse.BooleanOptionalAction, default=True, help="scales are in units of DSHARP's 3 mas pixels (rescaled to --pixel-mas)")
    p.add_argument("--gain", type=float, default=DSHARP["gain"])
    p.add_argument("--bias", type=float, default=DSHARP["bias"], help="small-scale bias")
    p.add_argument("--threshold-jy", type=float, default=DSHARP["threshold_jy"])
    p.add_argument("--niter", type=int, default=DSHARP["niter"])
    p.add_argument("--cycleniter", type=int, default=DSHARP["cycleniter"])
    p.add_argument("--cyclefactor", type=float, default=DSHARP["cyclefactor"])
    p.add_argument("--patch", type=int, default=256, help="half-width (pixels) of the PSF window subtracted in the minor cycles")
    p.add_argument("--beam", type=float, nargs=3, default=None, metavar=("BMAJ_MAS", "BMIN_MAS", "PA_DEG"), help="restoring beam (default: fitted to the PSF)")
    p.add_argument("--time-limit", type=float, default=0.0, help="seconds; stop after the major cycle that passes it")
    p.add_argument("--chunk-size", type=int, default=10**6)
    p.add_argument("--engine", choices=("virgil", "prototype"), default="virgil", help="virgil.radio_clean.clean (default) or this repo's NumPy minor cycles (crosscheck.alma_clean)")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("compare", help="compare a finished run with DSHARP's image")
    p.add_argument("--clean", required=True, help="the run's directory (image.fits, clean.json)")
    p.add_argument("--dsharp", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=lambda a: compare(a.clean, a.dsharp, a.out))
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
