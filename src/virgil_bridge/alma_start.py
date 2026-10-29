"""Starting images for the IM Lup MaxEnt fit that use no MPoL information: our own CLEAN model, regridded to the fit's grid.

Layout throughout is the repository's: row 0 North, column 0 East (``crosscheck.alma_maxent``); a cell ``(r, c)`` is at
``(north, east) = ((r0 - r) dx, (c0 - c) dx)`` of the cell ``(r0, c0)`` that holds the phase centre. The fit's grid has
``(r0, c0) = phase_centre_cell(npix)``; the CLEAN grid is odd with the phase centre on its middle pixel.
"""

import hashlib
import json
import pathlib

import numpy as np
import yaml

import crosscheck.preregistration as prereg
from crosscheck import alma_maxent as cx

from .alma import C8_TOL, C8_WINDOW, MPOL, convergence, convergence_window, gauge_to_obs0


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

REFERENCE_FLUX_JY = 0.2548  # Jy: sum(DSHARP's IM Lup CLEAN image in Jy/beam) * pixel area / beam area = 0.2547964 (design/alma_imlup_p3.md, "Provenance of gate (e)'s reference flux"); MEASURED, not quoted from a paper. The reference when neither --clean-dir nor --dsharp is given; ``clean_image_flux`` recomputes it from the FITS
REFERENCE_FLUX_NAME = "DSHARP CLEAN image flux (measured)"


def clean_image_flux(path):
    """Total flux (Jy) of a CLEAN image in Jy/beam, e.g. DSHARP's ``IMLup_continuum.fits``: ``sum(image) * pixel_area / beam_area``, the beam area
    pi bmaj bmin / (4 ln 2) from the header's BMAJ and BMIN, the pixel area from CDELT1 and CDELT2 (both in arcsec^2). Equivalently the image in
    Jy/arcsec^2 (divided by the beam area) summed over the pixels. Returns ``dict(flux_jy, sum_jy_per_beam, pixel_mas, beam_area_arcsec2, beam_area_pixels, path)``."""
    from astropy.io import fits

    with fits.open(path) as hdul:
        hdr = hdul[0].header
        data = np.asarray(hdul[0].data, float)
    unit = str(hdr.get("BUNIT", "")).strip().lower()
    if unit != "jy/beam":
        raise ValueError(f"{path}: BUNIT is {hdr.get('BUNIT')!r}, not Jy/beam")
    pix_arcsec = (abs(float(hdr["CDELT1"])) * 3600.0, abs(float(hdr["CDELT2"])) * 3600.0)
    beam_area = cx.beam_area_arcsec2(float(hdr["BMAJ"]) * 3.6e6, float(hdr["BMIN"]) * 3.6e6)
    total = float(np.nansum(data))
    return dict(flux_jy=total * pix_arcsec[0] * pix_arcsec[1] / beam_area, sum_jy_per_beam=total, pixel_mas=1000.0 * pix_arcsec[0],
                beam_area_arcsec2=float(beam_area), beam_area_pixels=float(beam_area / (pix_arcsec[0] * pix_arcsec[1])), path=str(path))


CRITERIA_FILE = pathlib.Path(__file__).resolve().parents[2] / "design" / "alma_imlup_gate_criteria.yml"  # pre-registered: pinned in design/criteria_registry.json
_CRITERIA = yaml.safe_load(CRITERIA_FILE.read_text())
GATE_FLUX_TOL = _CRITERIA["cold_start"]["flux_tol"]
GATE_DEAD_TOL = _CRITERIA["cold_start"]["dead_tol"]
GATE_REFERENCE_FLAG = _CRITERIA["cold_start"]["reference_flag"]  # flag our CLEAN flux and DSHARP's differing by more than this


def cold_start_gate(*, objective_ours, objective_mpol, chi2n_ours, chi2n_mpol, stages_converged, dead_fraction, flux_ours, flux_ref, flux_ref_name,
                    flux_tol=GATE_FLUX_TOL, dead_tol=GATE_DEAD_TOL, dead_fraction_base=None, flux_ref_alt=None, flux_ref_alt_name=None):
    """The cold-start gate (``design/alma_imlup_p3.md``, 2026-10-23, approved by Ben): five checks, each PASS or FAIL (``NOT EVALUATED`` when its input is
    None), and a verdict that is PASS only if all five pass.

    a. ``objective_ours`` <= ``objective_mpol`` (R's objective, both evaluated by virgil at eps 1e-12);
    b. raw chi^2/N at s = 1, ``chi2n_ours`` <= ``chi2n_mpol`` (MPoL's solution evaluated in virgil);
    c. C8 converged in every stage (``stages_converged``, a list of bools);
    d. the fraction of pixels of the image tapered to MPoL's beam below 1e-6 of its peak (``dead_fraction``; Ben, 2026-10-27: the base image may break into dots, MPoL's own is judged tapered) is under ``dead_tol`` (1 %); the base image's fraction (``dead_fraction_base``) is only recorded;
    e. ``flux_ours`` within ``flux_tol`` (10 %) of ``flux_ref``: the CLEAN flux of ours when one is given, else the DSHARP CLEAN image's measured 0.2548 Jy (``REFERENCE_FLUX_JY``, ``REFERENCE_FLUX_NAME``; ``flux_ref_name`` says which)."""
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
        item("d", f"tapered-image pixels below 1e-6 of the peak under {dead_tol:.0%}", None if dead_fraction is None else dead_fraction < dead_tol, dead_fraction, f"< {dead_tol:g}"),
        item("e", f"total flux within {flux_tol:.0%} of the {flux_ref_name} flux ({flux_ref:.5f} Jy)" if flux_ref else "total flux within 10% of the reference flux",
             None if rel is None else rel <= flux_tol, rel, f"<= {flux_tol:g}"),
    ]
    alt = None
    if flux_ref_alt and flux_ref:
        diff = flux_ref / flux_ref_alt - 1
        alt = dict(name=flux_ref_alt_name, flux_jy=flux_ref_alt, relative_difference=diff, flag=bool(abs(diff) > GATE_REFERENCE_FLAG),
                   note=f"the two reference fluxes differ by {diff:+.1%}" + (f" (more than {GATE_REFERENCE_FLAG:.0%})" if abs(diff) > GATE_REFERENCE_FLAG else ""))
    verdict = "PASS" if all(i["status"] == "PASS" for i in items) else ("INCOMPLETE" if any(i["status"] == "NOT EVALUATED" for i in items) and not any(i["status"] == "FAIL" for i in items) else "FAIL")
    return dict(verdict=verdict, items=items, flux_reference=dict(name=flux_ref_name, flux_jy=flux_ref), flux_ours_jy=flux_ours, dead_fraction_base=dead_fraction_base,
                flux_reference_alt=alt,
                rule="PASS only if (a)-(e) all pass; design/alma_imlup_p3.md, 2026-10-23, approved by Ben")


def gate_markdown(gate):
    """The gate as markdown lines (a table and the verdict) for ``compare.md``."""
    rows = [f"**Cold-start gate: {gate['verdict']}** ({gate['rule']})", "", "| | check | value | limit | |", "|---|---|---|---|---|"]
    fmt = lambda v: "" if v is None else (v if isinstance(v, str) else f"{v:.6g}")  # noqa: E731
    rows += [f"| {i['key']} | {i['label']} | {fmt(i['value'])} | {i['limit']} | {i['status']} |" for i in gate["items"]]
    alt = gate.get("flux_reference_alt")
    if alt:
        rows += ["", f"Alternative reference {alt['name']} {alt['flux_jy']:.5g} Jy: {alt['note']}."]
    return rows


CRITERIA = _CRITERIA  # the registered file's content: Track A, the synthetic truths, C8 and the replica pass live there too (a changed value changes the file's pin)
MPOL_RAW_CHI2N = _CRITERIA["better_than_mpol"]["mpol_raw_chi2n"]  # MPoL's solution, raw chi^2/(2N) at s = 1 in virgil (the 2026-10-27 round)
BETTER_NEFF, BETTER_DEAD = _CRITERIA["better_than_mpol"]["neff_min"], _CRITERIA["better_than_mpol"]["dead_max"]


class GateRefused(SystemExit):
    """An input the gate will not score (another criteria hash, a development truth, a C8 flag its trajectory contradicts): a ``SystemExit`` with the reason."""


def criteria_hash():
    """sha256 of the registered criteria file (``crosscheck.preregistration``): what every gate input must have been produced under."""
    return prereg.sha256_file(CRITERIA_FILE)


def require_criteria(record, what):
    """Refuse ``record`` (a result.json / synth.json / ... dict) unless its ``criteria_hash`` is the registered file's now."""
    h = (record or {}).get("criteria_hash")
    if h != criteria_hash():
        raise GateRefused(f"{what}: criteria_hash {h!r} is not the current {criteria_hash()}: it was produced under other criteria (or none); rerun it, never edit the hash")


def cells():
    """The ``(truth id, coverage id)`` cells of criterion 3, in ``CRITERIA`` order."""
    return [(t["id"], c["id"]) for t in CRITERIA["synth"]["truths"] for c in CRITERIA["synth"]["coverages"]]


def truth_spec(truth_id):
    """The ``CRITERIA`` entry of a registered truth; a development truth or unknown id is refused."""
    if truth_id in CRITERIA["synth"]["development_truths"]:
        raise GateRefused(f"truth {truth_id!r} is development-only (made from a candidate arm): it is never scored for the gate")
    for t in CRITERIA["synth"]["truths"]:
        if t["id"] == truth_id:
            return t
    raise GateRefused(f"truth {truth_id!r} is not registered in CRITERIA['synth']['truths']")


def truth_source_sha(spec):
    """The registered identity of a truth's source: its pinned file sha256, or for a parametric truth the sha256 of its recipe, blur and grid as canonical JSON. None if a file truth is unpinned."""
    if spec["kind"] == "parametric":
        syn = CRITERIA["synth"]
        blob = dict(recipe=spec["recipe"], blur_mas=spec["blur_mas"], flux_jy=syn["flux_jy"], pixel_mas=syn["pixel_mas"], npix=syn["npix"])
        return hashlib.sha256(json.dumps(blob, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return spec["sha256"]


def coverage_ddids(coverage_id):
    """The ddids a registered coverage fits (None: all of them)."""
    return next(c["ddids"] for c in CRITERIA["synth"]["coverages"] if c["id"] == coverage_id)


def check_synth(record, truth_id, coverage_id, what, config=None, workflow=None):
    """Refuse a synth.json that is stamped with another hash, is not the registered (``truth_id``, ``coverage_id``) cell, has a development or unpinned truth, whose truth file is not the registered one, that scores a
    fit of another ``config`` / ``workflow`` than expected (None: unchecked), a fit whose own ``--ddids`` are not the coverage's, or a fit whose data or ``truth.npy`` are not those ``make`` wrote."""
    require_criteria(record, what)
    spec = truth_spec(truth_id)
    if record.get("development"):
        raise GateRefused(f"{what}: marked development")
    if (record.get("truth_id"), record.get("coverage")) != (truth_id, coverage_id):
        raise GateRefused(f"{what}: is truth {record.get('truth_id')!r} / coverage {record.get('coverage')!r}, not {truth_id!r} / {coverage_id!r}")
    want = truth_source_sha(spec)
    if want is None:
        raise GateRefused(f"{what}: truth {truth_id!r} has no sha256 pinned in CRITERIA: pin it first")
    if record.get("truth_source_sha256") != want:
        raise GateRefused(f"{what}: truth source sha256 {record.get('truth_source_sha256')!r} is not the registered {want}")
    for key, want_v in (("config", config), ("workflow", workflow)):
        if want_v is not None and record.get(key) != want_v:
            raise GateRefused(f"{what}: scores a fit with {key} {record.get(key)!r}, not the expected {want_v!r}")
    got, exp = record.get("fit_ddids"), coverage_ddids(coverage_id)
    if (None if got is None else sorted(got)) != (None if exp is None else sorted(exp)):
        raise GateRefused(f"{what}: the fit used ddids {got!r}, not coverage {coverage_id!r}'s {exp!r}")
    if not (record.get("data_is_truth_dir") and record.get("truth_npy_matches_make")):
        raise GateRefused(f"{what}: the fit's data are not the directory the truth was made in, or truth.npy is not the one synth_make.json hashed")


def c8_recompute(stage):
    """C8 of a saved stage from its own history: Adam from ``adam.loss_history`` (``convergence_window``), L-BFGS from ``trajectory`` (``convergence``), both at ``CRITERIA['c8']``. Refuses a stage whose stored
    ``c8_converged`` differs. Returns the verdict dict with ``name``."""
    tol, window = CRITERIA["c8"]["tol"], CRITERIA["c8"]["window"]
    adam = stage.get("adam") or {}
    if stage.get("method") == "adam" or adam:
        hist = adam.get("loss_history")
        if not hist:
            raise GateRefused(f"stage {stage.get('name')!r}: Adam stage without adam.loss_history: C8 cannot be recomputed (rerun with this vv)")
        v = convergence_window(hist, tol, window)
    else:
        traj = stage.get("trajectory") or []
        if len(traj) < 2:
            return dict(converged=None, change=None, mean_drop=None, name=stage.get("name"), method="lbfgs", reason="fewer than two trajectory records: C8 cannot be recomputed (the fit's own flag is not evidence)")
        v = convergence(traj, None, tol, window)
    if bool(v["converged"]) != bool(stage.get("c8_converged")):
        raise GateRefused(f"stage {stage.get('name')!r}: stored c8_converged={stage.get('c8_converged')} but its own history gives {v['converged']} (change {v['change']}, mean_drop {v.get('mean_drop')})")
    return dict(v, name=stage.get("name"), method="adam" if adam else "lbfgs")


def better_than_mpol(*, heldout_ours, heldout_mpol, chi2n_ours, synth, synth_r, flux_ours, flux_ref, flux_ref_name, stages, fit_config=None, fit_workflow=None):
    """The "better than MPoL" pass of Track A (``design/alma_imlup_p3.md``, Ben 2026-10-28): one arm must do all five, each PASS / FAIL / NOT EVALUATED (None input). Every threshold is ``CRITERIA['better']`` (hashed):
    there is no keyword to change one.

    1. held-out raw chi^2/N below MPoL's on the same folds by more than its uncertainty: ``heldout_ours`` and ``heldout_mpol`` are per-fold lists (``cv-compare``'s ``per_fold``), and ``mean(d) + se(d) < 0`` for the per-fold differences d (se = sd / sqrt(K)); a pair of scalars is compared without an uncertainty and is flagged in the value;
    2. raw chi^2/N at s = 1 (``chi2n_ours``) no worse than MPoL's 2.7062;
    3. on every registered synthetic truth x coverage (``cells``): ``synth`` and ``synth_r`` map ``(truth id, coverage id)`` to the arm's and config R's ``synth.json`` dicts (``check_synth`` refuses any other hash, a development truth or an unpinned or
       other truth file); NRMSE at 0.5 CLEAN beam below R's, N_eff ratio >= 0.7 and base pixels below 1e-4 of the in-disk peak under 1 %, in every cell;
    4. ``flux_ours`` within 10 % of ``flux_ref``;
    5. C8 converged in every stage, recomputed from each stage's saved history (``c8_recompute``); ``stages`` are the fit's stage records; ``change`` and ``mean_drop`` go in the table."""
    b = c = CRITERIA["better_than_mpol"]
    def item(key, label, ok, value, limit):
        return dict(key=key, label=label, value=value, limit=limit, status="NOT EVALUATED" if ok is None else ("PASS" if ok else "FAIL"))

    h, h_txt = None, None
    if heldout_ours is not None and heldout_mpol is not None:
        d = np.atleast_1d(np.asarray(heldout_ours, float) - np.asarray(heldout_mpol, float))
        se = float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
        h = float(d.mean()) + se
        h_txt = f"mean diff {d.mean():+.6g} + se {se:.3g} over {len(d)} fold(s)" + ("" if len(d) > 1 else " (no uncertainty: one value)")
    frac = b["nrmse_beam_fraction"]
    syn_rows, syn_ok = [], None
    if synth is not None and synth_r is not None:
        for t, cv in cells():
            if (t, cv) not in synth or (t, cv) not in synth_r:
                syn_rows.append(f"{t}/{cv}: missing")
                continue
            check_synth(synth[(t, cv)], t, cv, f"synth {t}/{cv}", config=fit_config, workflow=fit_workflow)
            check_synth(synth_r[(t, cv)], t, cv, f"synth-r {t}/{cv}", config="R")
            s, r = synth[(t, cv)], synth_r[(t, cv)]
            ok = s["nrmse"][frac] < r["nrmse"][frac] and s["n_eff_ratio"] >= c["neff_min"] and s["base_dead_in_disk_1e4"] < c["dead_max"]
            syn_rows.append(f"{t}/{cv}: {'ok' if ok else 'FAIL'} (NRMSE {s['nrmse'][frac]:.4f} vs R {r['nrmse'][frac]:.4f}, N_eff {s['n_eff_ratio']:.3f}, dead {s['base_dead_in_disk_1e4']:.2%})")
            syn_ok = ok if syn_ok is None else (syn_ok and ok)
        if any("missing" in x for x in syn_rows) and syn_ok is not False:
            syn_ok = None
    rel = None if flux_ours is None or not flux_ref else abs(flux_ours / flux_ref - 1)
    c8 = None if not stages else [c8_recompute(st) for st in stages]
    c8_txt = None if c8 is None else "; ".join(f"{v['name']} {'NOT EVALUATED' if v['converged'] is None else 'ok' if v['converged'] else 'FAIL'} (change {v['change']}, mean_drop {v.get('mean_drop')})" for v in c8)
    items = [
        item("1", "held-out chi^2/N below MPoL's on the same folds by more than the fold standard error", None if h is None else h < 0, h_txt, "mean + se < 0"),
        item("2", f"raw chi^2/N (s = 1) at or below {b['mpol_raw_chi2n']:g}", None if chi2n_ours is None else chi2n_ours <= b["mpol_raw_chi2n"], chi2n_ours, f"<= {b['mpol_raw_chi2n']:g}"),
        item("3", f"every synthetic truth x coverage: NRMSE at {frac} beam below R's, N_eff ratio >= {b['neff_min']:g}, base dead < {b['dead_max']:.0%}", syn_ok, "; ".join(syn_rows) if syn_rows else None, "all cells, all three"),
        item("4", f"total flux within {b['flux_tol']:.0%} of the {flux_ref_name} flux ({flux_ref:.5f} Jy)" if flux_ref else "total flux within 10% of CLEAN", None if rel is None else rel <= b["flux_tol"], rel, f"<= {b['flux_tol']:g}"),
        item("5", f"C8 converged in every stage (recomputed: tol {CRITERIA['c8']['tol']:g}, window {CRITERIA['c8']['window']})", None if c8 is None or (all(v["converged"] is not False for v in c8) and any(v["converged"] is None for v in c8)) else all(v["converged"] for v in c8), c8_txt, "all"),
    ]
    verdict = "PASS" if all(i["status"] == "PASS" for i in items) else ("INCOMPLETE" if any(i["status"] == "NOT EVALUATED" for i in items) and not any(i["status"] == "FAIL" for i in items) else "FAIL")
    return dict(verdict=verdict, items=items, c8_stages=c8, cells=[list(x) for x in cells()], rule="PASS only if 1-5 all pass; design/alma_imlup_p3.md, Track A, Ben 2026-10-28")


def better_markdown(b):
    """``better_than_mpol``'s result as markdown lines (a table and the verdict)."""
    rows = [f"**Better than MPoL: {b['verdict']}** ({b['rule']})", "", "| | check | value | limit | |", "|---|---|---|---|---|"]
    fmt = lambda v: "" if v is None else (v if isinstance(v, str) else f"{v:.6g}")  # noqa: E731
    return rows + [f"| {i['key']} | {i['label']} | {fmt(i['value'])} | {i['limit']} | {i['status']} |" for i in b["items"]]


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
