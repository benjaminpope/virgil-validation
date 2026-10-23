"""Starting images for the IM Lup MaxEnt fit that use no MPoL information: our own CLEAN model, regridded to the fit's grid.

Layout throughout is the repository's: row 0 North, column 0 East (``crosscheck.alma_maxent``); a cell ``(r, c)`` is at
``(north, east) = ((r0 - r) dx, (c0 - c) dx)`` of the cell ``(r0, c0)`` that holds the phase centre. The fit's grid has
``(r0, c0) = phase_centre_cell(npix)``; the CLEAN grid is odd with the phase centre on its middle pixel.
"""

import json
import pathlib

import numpy as np

from crosscheck import alma_maxent as cx

from .alma import MPOL, gauge_to_obs0


def clean_centre(npix):
    """Phase-centre cell of a CLEAN grid: its middle pixel (``alma_imlup_clean.grid_wcs``)."""
    return ((npix - 1) / 2, (npix - 1) / 2)


def read_clean(clean_dir):
    """``(model, pixel_mas, beam, info)`` of ``alma_imlup_clean.py run`` output: ``model.fits`` (Jy/pixel) flipped to row 0 North,
    its pixel (mas), and the restoring beam ``(bmaj_mas, bmin_mas, pa_deg)`` from ``clean.json`` (else ``image.fits``' header)."""
    clean_dir = pathlib.Path(clean_dir)
    model, _, hdr = cx.read_fits(clean_dir / "model.fits")
    if model.shape[0] != model.shape[1] or model.shape[0] % 2 == 0:
        raise ValueError(f"CLEAN model must be odd and square, not {model.shape}")
    beam = None
    if (clean_dir / "clean.json").exists():
        b = json.loads((clean_dir / "clean.json").read_text()).get("beam_mas_deg")
        beam = None if b is None else tuple(float(x) for x in b)
    if beam is None and (clean_dir / "image.fits").exists():
        beam = cx.beam_of(cx.read_fits(clean_dir / "image.fits")[2])
    return model[::-1], abs(float(hdr["CDELT1"])) * 3.6e6, beam, dict(clean_dir=str(clean_dir), shape=list(model.shape), flux_jy=float(model.sum()))


def regrid(sb, src_pixel_mas, dst_pixel_mas, dst_npix, src_centre, dst_centre=None):
    """Resample a surface-brightness image (Jy/arcsec^2) onto another grid of the same phase centre, flux-preserving.

    Bilinear in surface brightness, then scaled so that the new image holds exactly the flux of the source pixels
    inside its footprint: the cut-out of a larger field (3001 x 3 mas to 1028 x 5 mas) and the change of pixel size
    neither add nor lose flux. Returns ``(image, info)``."""
    from scipy.ndimage import map_coordinates

    sb = np.asarray(sb, float)
    r0d, c0d = cx.phase_centre_cell(dst_npix) if dst_centre is None else dst_centre
    r0s, c0s = src_centre
    rr, cc = np.meshgrid(np.arange(dst_npix), np.arange(dst_npix), indexing="ij")
    ys = r0s - (r0d - rr) * dst_pixel_mas / src_pixel_mas  # north = (r0 - r) dx, so the source row is r0s - north / dx_s
    xs = c0s - (c0d - cc) * dst_pixel_mas / src_pixel_mas
    out = map_coordinates(sb, [ys.ravel(), xs.ravel()], order=1, mode="constant", cval=0.0).reshape(dst_npix, dst_npix)
    # the source pixels whose centres lie inside the new field (cell edges at half a new pixel)
    rs, cs = np.meshgrid(np.arange(sb.shape[0]), np.arange(sb.shape[1]), indexing="ij")
    north, east = (r0s - rs) * src_pixel_mas, (c0s - cs) * src_pixel_mas
    h = dst_pixel_mas / 2
    inside = ((north >= (r0d - (dst_npix - 1)) * dst_pixel_mas - h) & (north <= r0d * dst_pixel_mas + h)
              & (east >= (c0d - (dst_npix - 1)) * dst_pixel_mas - h) & (east <= c0d * dst_pixel_mas + h))
    f_src = float(sb[inside].sum()) * (src_pixel_mas / 1000.0) ** 2
    f_raw = float(out.sum()) * (dst_pixel_mas / 1000.0) ** 2
    if f_raw == 0.0 or f_src == 0.0:
        raise ValueError("the regridded image holds no flux: is the source inside the new field?")
    return out * (f_src / f_raw), dict(flux_in_field_jy=f_src, flux_before_scaling_jy=f_raw, scale=f_src / f_raw)


def clean_start(model, src_pixel_mas, beam, *, npix, pixel_mas, smooth_mas=10.0, beam_frac=0.0, floor=MPOL["prior"]):
    """The MaxEnt start from a CLEAN model (Jy/pixel, row 0 North, odd grid): ``(log_brightness, flux_jy, info)``.

    1. A small positivity kernel: the model is convolved with a circular Gaussian of FWHM ``smooth_mas`` (default 10 mas, two
       fit pixels), so that the sparse components resample onto the 5 mas grid without aliasing and no pixel is isolated at the
       floor, where MaxEnt's gradient is too small to move it. The kernel is far below the 30 mas taper and the long baselines'
       resolution (about 15 mas), so it takes no power from the long baselines: do not smooth with the restoring beam
       (``beam_frac`` > 0 does, as an option: it replaces the circular kernel by that fraction of the beam), because the
       image-fixed stage would then lower the long-baseline gains to absorb the missing power.
    2. The image is regridded to ``npix`` x ``pixel_mas`` (``regrid``), flux-preserving within the field.
    3. Negative pixels (multi-scale CLEAN can leave some) and pixels below ``floor`` are raised to ``floor``, Jy/arcsec^2,
       by default MPoL's entropy prior ``p = 1e-4``: the level a pixel takes with no data (the entropy's minimum is p/e).
       That is the smallest start that keeps ln finite without lifting the dark sky: at 1028^2 x 5 mas it adds
       about 2.6 mJy (1 % of IM Lup's flux), against about 1.3 Jy for MPoL's own flat start.
    4. ``cal.flux`` is the flux of the floored image, and ``log_brightness`` is ``ln`` of it, so that
       ``alma.image_of`` returns exactly the floored image.
    """
    px_s = src_pixel_mas
    sb = np.asarray(model, float) / (px_s / 1000.0) ** 2
    if beam_frac > 0:
        if beam is None:
            raise ValueError("beam_frac > 0 needs the restoring beam (clean.json or image.fits in the CLEAN directory)")
        sb = cx.convolve(sb, px_s, beam[0] * beam_frac, beam[1] * beam_frac, beam[2])
    elif smooth_mas > 0:
        sb = cx.convolve(sb, px_s, smooth_mas)
    sb, ginfo = regrid(sb, px_s, pixel_mas, npix, clean_centre(model.shape[0]))
    area = (pixel_mas / 1000.0) ** 2
    negative = float(sb[sb < 0].sum()) * area
    floored = np.maximum(sb, floor)
    flux = float(floored.sum()) * area
    log_brightness = np.log(floored / floored.max())
    info = dict(**ginfo, beam_mas_deg=beam, beam_frac=beam_frac, smooth_mas=smooth_mas, floor_jy_arcsec2=floor, negative_flux_clipped_jy=negative,
                flux_added_by_floor_jy=flux - ginfo["flux_in_field_jy"], npix=npix, pixel_mas=pixel_mas, flux_jy=flux,
                peak_jy_arcsec2=float(floored.max()), n_at_floor=int((sb <= floor).sum()))
    return log_brightness, flux, info


# the long-baseline gains (obs 5, 6) of the image-fixed stage against what the MPoL-initialised fits found: R_mpol's 1.02-1.03 and MPoL's
# own 0.94-0.98 (mpol_disagreements.md, "Gate (a) PASSES"). A gain far outside both says the held image is biased (too smooth or
# too sharp on the long baselines), not that the data changed: reported, never tuned away.
REF_GAIN_RANGE = (0.94, 1.03)
GAIN_MARGIN = 0.05


def selfcal_check(values, ref=REF_GAIN_RANGE, margin=GAIN_MARGIN, ref_obs=0):
    """The image-fixed stage's gains, flux and offsets, and whether any gain is more than ``margin`` outside ``ref``.
    ``values['cal.gain']`` holds the gains of every observation but ``ref_obs`` (whose gain is 1); the reference gains are those of obs 0 = 1,
    so a fit with another reference is renormalised first (``gauge_to_obs0``: g / g_0, flux g_0). ``gain`` is that of obs 1..n-1, and
    ``gain_lb_obs5_6`` of the long-baseline observations 5 and 6, whichever observation is the reference."""
    full, off_full, flux = gauge_to_obs0(values["cal.gain"], values["cal.offset"], values["cal.flux"], ref_obs)
    g = np.asarray(full, float)[1:]
    lo, hi = ref[0] - margin, ref[1] + margin
    off = np.asarray(values["cal.offset"], float)
    return dict(gain=g, gain_min=float(g.min()), gain_max=float(g.max()), gain_lb_obs5_6=g[4:6], reference_range=ref, margin=margin, ref_obs=int(ref_obs),
                gain_outside_reference=bool(np.any((g < lo) | (g > hi))), flux_jy=float(flux), offset_max_abs_mas=float(np.abs(off).max()))
