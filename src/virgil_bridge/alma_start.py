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


# ------------------------------------------------------------------ the cold-start gate

REFERENCE_FLUX_JY = MPOL["f_tot"]  # 0.253 Jy: IM Lup's total flux as MPoL's example takes it (sgd.py:298-302, ``MPOL``); the reference when no CLEAN of ours is given. DSHARP's own value is not sourced here
GATE_FLUX_TOL = 0.10
GATE_DEAD_TOL = 0.01


def cold_start_gate(*, objective_ours, objective_mpol, chi2n_ours, chi2n_mpol, stages_converged, dead_fraction, flux_ours, flux_ref, flux_ref_name,
                    flux_tol=GATE_FLUX_TOL, dead_tol=GATE_DEAD_TOL):
    """The cold-start gate (``design/alma_imlup_p3.md``, 2026-10-23, approved by Ben): five checks, each PASS or FAIL (``NOT EVALUATED`` when its input is
    None), and a verdict that is PASS only if all five pass.

    a. ``objective_ours`` <= ``objective_mpol`` (R's objective, both evaluated by virgil at eps 1e-12);
    b. raw chi^2/N at s = 1, ``chi2n_ours`` <= ``chi2n_mpol`` (MPoL's solution evaluated in virgil);
    c. C8 converged in every stage (``stages_converged``, a list of bools);
    d. the fraction of pixels below 1e-6 of the peak (``dead_fraction``) is under ``dead_tol`` (1 %);
    e. ``flux_ours`` within ``flux_tol`` (10 %) of ``flux_ref``: the CLEAN flux of ours when one is given, else MPoL's 0.253 Jy (``REFERENCE_FLUX_JY``; ``flux_ref_name`` says which)."""
    def item(key, label, ok, value, limit):
        return dict(key=key, label=label, value=value, limit=limit, status="NOT EVALUATED" if ok is None else ("PASS" if ok else "FAIL"))

    both = lambda a, b: None if a is None or b is None else (a, b)  # noqa: E731
    o, c = both(objective_ours, objective_mpol), both(chi2n_ours, chi2n_mpol)
    sc = None if stages_converged is None or len(stages_converged) == 0 else [bool(s) for s in stages_converged]
    rel = None if flux_ours is None or not flux_ref else abs(flux_ours / flux_ref - 1)
    items = [
        item("a", "objective_R at or below MPoL's", None if o is None else o[0] <= o[1], None if o is None else o[0] - o[1], "<= 0"),
        item("b", "raw chi^2/N (s = 1) at or below MPoL's solution in virgil", None if c is None else c[0] <= c[1], None if c is None else c[0] - c[1], "<= 0"),
        item("c", "C8 converged in every stage", None if sc is None else all(sc), None if sc is None else f"{sum(sc)}/{len(sc)} stages", "all"),
        item("d", f"pixels below 1e-6 of the peak under {dead_tol:.0%}", None if dead_fraction is None else dead_fraction < dead_tol, dead_fraction, f"< {dead_tol:g}"),
        item("e", f"total flux within {flux_tol:.0%} of the {flux_ref_name} flux ({flux_ref:.5f} Jy)" if flux_ref else "total flux within 10% of the reference flux",
             None if rel is None else rel <= flux_tol, rel, f"<= {flux_tol:g}"),
    ]
    verdict = "PASS" if all(i["status"] == "PASS" for i in items) else ("INCOMPLETE" if any(i["status"] == "NOT EVALUATED" for i in items) and not any(i["status"] == "FAIL" for i in items) else "FAIL")
    return dict(verdict=verdict, items=items, flux_reference=dict(name=flux_ref_name, flux_jy=flux_ref), flux_ours_jy=flux_ours,
                rule="PASS only if (a)-(e) all pass; design/alma_imlup_p3.md, 2026-10-23, approved by Ben")


def gate_markdown(gate):
    """The gate as markdown lines (a table and the verdict) for ``compare.md``."""
    rows = [f"**Cold-start gate: {gate['verdict']}** ({gate['rule']})", "", "| | check | value | limit | |", "|---|---|---|---|---|"]
    fmt = lambda v: "" if v is None else (v if isinstance(v, str) else f"{v:.6g}")  # noqa: E731
    rows += [f"| {i['key']} | {i['label']} | {fmt(i['value'])} | {i['limit']} | {i['status']} |" for i in gate["items"]]
    return rows


# ------------------------------------------------------------------ a data-driven start: the elliptical Gaussian of CASA's uvmodelfit

GAUSS_FIT_MAX_Q = 4.0e4  # wavelengths: the short baselines (resolution 1/q = 5 arcsec) that see the disc's flux, centroid and overall size


def gaussian_visibility(u, v, flux, east_mas, north_mas, major_mas, minor_mas, pa_deg):
    """V(u, v) = sum I exp(-2 pi i (u x + v y)) (``docs/alma_conventions.md``: x East, y North, u, v in wavelengths) of an elliptical Gaussian of total
    flux ``flux`` centred ``east_mas``, ``north_mas`` from the phase centre, FWHMs ``major_mas`` >= ``minor_mas`` and the major axis at ``pa_deg``
    East of North: ``flux exp(-2 pi i (u x0 + v y0)) exp(-pi^2 (a^2 U_p^2 + b^2 U_q^2) / (4 ln 2))`` with the baseline's components along (U_p) and across (U_q) the major axis."""
    th = np.deg2rad(pa_deg)
    up, uq = u * np.sin(th) + v * np.cos(th), u * np.cos(th) - v * np.sin(th)
    a, b = major_mas * cx.MAS, minor_mas * cx.MAS
    return flux * np.exp(-2j * np.pi * (u * east_mas + v * north_mas) * cx.MAS) * np.exp(-(np.pi**2) * ((a * up) ** 2 + (b * uq) ** 2) / (4 * np.log(2)))


def fit_gaussian(u, v, vis, weight, *, q_max=GAUSS_FIT_MAX_Q, max_points=2_000_000, seed=0):
    """Least-squares fit of an elliptical Gaussian (flux, centroid, FWHMs, position angle) to the visibilities with ``|(u, v)| < q_max`` wavelengths,
    weighted by ``weight`` (1 / sigma^2): the model of CASA's ``uvmodelfit`` (component type G; CASA fits the same visibility model by Levenberg-Marquardt),
    here ``scipy.optimize.least_squares`` from four position angles, the lowest chi^2 kept. At most ``max_points`` (a random subset, ``seed``) are used.
    No gains or offsets: the start of the fit, not a result. Returns the parameters (``flux_jy``, ``east_mas``, ``north_mas``, ``major_mas`` >= ``minor_mas``,
    ``pa_deg`` in [0, 180)) and ``chi2_per_point``, ``n_used``, ``n_in_cut``, ``q_max``.

    Why a Gaussian: among positive images with a given flux, centroid and second moments the Gaussian has the largest (Shannon) differential entropy;
    that is all this claims (the fit matches those moments only as far as the data below ``q_max`` see them, and it is not the fit's own entropy S_M)."""
    from scipy.optimize import least_squares

    u, v, vis, w = (np.asarray(x) for x in (u, v, vis, weight))
    keep = np.flatnonzero(np.hypot(u, v) < q_max)
    if keep.size < 10:
        raise ValueError(f"only {keep.size} visibilities below {q_max:g} wavelengths: nothing to fit a Gaussian to")
    n_in_cut = int(keep.size)
    if keep.size > max_points:
        keep = np.sort(np.random.default_rng(seed).choice(keep, max_points, replace=False))
    u, v, vis, w = u[keep], v[keep], vis[keep].astype(complex), np.sqrt(w[keep].astype(float))
    f0 = max(float(np.average(vis.real, weights=w**2)), 1e-3)

    def model(p):
        return gaussian_visibility(u, v, p[0], p[1], p[2], np.exp(p[3]), np.exp(p[4]), np.rad2deg(p[5]))

    def resid(p):
        r = (model(p) - vis) * w
        return np.concatenate([r.real, r.imag])

    lo = [1e-4, -2e4, -2e4, np.log(5.0), np.log(5.0), -10.0]
    hi = [10.0, 2e4, 2e4, np.log(5e4), np.log(5e4), 10.0]
    best = None
    for pa0 in (0.0, 45.0, 90.0, 135.0):
        p0 = [f0, 0.0, 0.0, np.log(1000.0), np.log(800.0), np.deg2rad(pa0)]
        r = least_squares(resid, p0, bounds=(lo, hi), x_scale=[0.1, 100.0, 100.0, 1.0, 1.0, 1.0])
        if best is None or r.cost < best.cost:
            best = r
    flux, x0, y0, la, lb, th = best.x
    a, b, pa = float(np.exp(la)), float(np.exp(lb)), float(np.rad2deg(th))
    if b > a:  # the major axis is the longer one
        a, b, pa = b, a, pa + 90.0
    return dict(flux_jy=float(flux), east_mas=float(x0), north_mas=float(y0), major_mas=a, minor_mas=b, pa_deg=float(pa % 180.0),
                chi2_per_point=float(2 * best.cost / (2 * len(u))), n_used=int(len(u)), n_in_cut=n_in_cut, q_max=float(q_max),
                source="elliptical Gaussian least-squares fit to the short-baseline visibilities, the model of CASA uvmodelfit (component type G)")


def gaussian_log_image(npix, pixel_mas, east_mas, north_mas, major_mas, minor_mas, pa_deg):
    """ln of the elliptical Gaussian (peak 1) of ``gaussian_visibility`` on the fit's grid (row 0 North, column 0 East, phase centre at
    ``alma_maxent.phase_centre_cell``): the start's ``log_brightness`` before the floor."""
    r0, c0 = cx.phase_centre_cell(npix)
    rr, cc = np.meshgrid(np.arange(npix), np.arange(npix), indexing="ij")
    north, east = (r0 - rr) * pixel_mas - north_mas, (c0 - cc) * pixel_mas - east_mas
    th = np.deg2rad(pa_deg)
    p, q = east * np.sin(th) + north * np.cos(th), east * np.cos(th) - north * np.sin(th)
    return -4 * np.log(2) * ((p / major_mas) ** 2 + (q / minor_mas) ** 2)
