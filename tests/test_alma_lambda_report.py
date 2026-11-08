"""The report of the registered lambda rule (``virgil_bridge.alma_lambda_report``, ``lambda-report`` of ``scripts/alma_imlup_maxent.py``) on a synthetic run directory: records shaped like
the stages' own, a ring-and-spiral image, no fit and no virgil. The report must run on a complete run, on a failed stage (reason shown, later sections skipped) and on a run with stages missing."""

import json

import numpy as np
import pytest

import virgil_bridge.alma_lambda as vl
import virgil_bridge.alma_lambda_report as rep

pytestmark = pytest.mark.validates("virgil_bridge.alma_lambda", roots=["self-consistency"], kind="guard")

NPIX, PIX, S2 = 96, 30.0, 1.1


def disc_flux():
    """Jy/pixel: a ring with an m = 2 spiral, row 0 North, phase-centre cell ``npix // 2 - 1``."""
    r0 = NPIX // 2 - 1
    rr, cc = np.indices((NPIX, NPIX))
    y, x = (r0 - rr) * PIX, (r0 - cc) * PIX  # North, East in mas
    r, th = np.hypot(x, y), np.arctan2(x, y)
    f = np.exp(-0.5 * ((r - 500.0) / 150.0) ** 2) * (1 + 0.3 * np.cos(2 * (th - 0.01 * r)))
    return 0.25 * f / f.sum() + 1e-9


def records(**over):
    pixels = np.sqrt(disc_flux())
    w = vl.grid(S2)
    m = np.linspace(2e6, -3e5, len(w))
    pts = [dict(weight=float(a), w_over_s2=float(a) / S2, chi2=1e7 + 1e4 * i, m=float(b), g=1e4 / (1 + i), kkt_w=5e-5, band_limit=0.8) for i, (a, b) in enumerate(zip(w, m))]
    root = int(np.flatnonzero(np.diff(np.sign(m)))[0])
    wc = float(np.sqrt(w[root] * w[root + 1]))
    sigma = dict(stage="sigma", s2=S2, sigma_s=1.0e-4, sigma_rel=1.0e-4 / S2, n_pairs=10**7, raw_chi2_over_N_pairs=S2, lag1_correlation=-0.49, n_lag1=10**7, ref_obs=2, sum_weights=[1.0, 2.0, 5.0, 3.0],
                 flux_jy=0.25, n_vis=10**7, n_real=2 * 10**7, status="ok", registered=True)
    d4 = dict(stage="d4", s2_injected=S2, mean=S2 * (1 + 1e-4), sd=1e-3, se_mean=1e-3 / np.sqrt(20), bias=S2 * 1e-4, bias_rel=1e-4, bias_in_se=0.5, lag1_correlation_real=-0.49, **{"pass": True}, status="ok")
    sweep = dict(stage="sweep", s2=S2, sigma_s=1e-4, n_rows=10**7, n_real=2 * 10**7, points=pts, crossings=[wc / 3, wc], crossings_w_over_s2=[wc / 3 / S2, wc / S2], weight=wc * 1.01, w_over_s2=wc * 1.01 / S2,
                 weight_band=[wc * 1.2, None], m=3.0, g=9e3, g_se=2.0, h=9.1e3, n_secant=2, unconverged_fits=[], status="ok")
    nuis = {"cal.gain": dict(value=[1.0, 0.98, 1.0, 1.02], lower=[0.7] * 4, upper=[1.3] * 4, edge_frac=[0.5, 0.46, 0.5, 0.54], at_edge=[False] * 4),
            "cal.offset": dict(value=[[0.0, 0.0], [1.0, -2.0]], lower=[[-9.0, -9.0]] * 2, upper=[[9.0, 9.0]] * 2, edge_frac=[[0.5, 0.5], [0.44, 0.39]], at_edge=[[False] * 2] * 2)}
    final = dict(stage="final", raw_chi2_over_N=1.05, raw_chi2_over_N_nominal=1.08, s2=S2, sigma_s=1e-4, g=9e3, g_se=2.0, sigma_factors="not fitted (weights rescaled by s2, item 1)", weight=sweep["weight"], w_over_s2=sweep["w_over_s2"],
                 psi_hp=vl.psi_hp(pixels, PIX), kkt_w=3e-5, kkt_w_max=1e-4, band_limit=0.9, band_limit_max=2.0, f_eta_over_f_floor=0.9, nuisances=nuis, ref_obs=2, flux_jy=0.2499, flux_reference_jy=0.25, npix=NPIX,
                 pixel_mas=PIX, status="ok")
    out = dict(sigma=sigma, d4=d4, sweep=sweep, final=final)
    for name, patch in over.items():
        out[name] = None if patch is None else {**out[name], **patch}
    return out, pixels


def make_run(path, **over):
    recs, pixels = records(**over)
    path.mkdir(parents=True, exist_ok=True)
    for name, fname in rep.STAGES:
        if recs[name] is not None:
            (path / fname).write_text(json.dumps(recs[name]))
    np.savez(path / rep.VALUES, pixels=pixels)
    return path


def failed(reason, **kw):
    return {"status": "failed", "reason": reason, **kw}


def test_a_complete_run_gives_every_section_every_check_and_the_figures(tmp_path):
    run = make_run(tmp_path / "run")
    md = rep.write(run)
    assert (run / "report.md").read_text().strip() == md.strip()
    assert "All checks pass" in md and "FAIL" not in md and "not evaluated" not in md
    assert md.index("Raw χ²/N with the quoted weights") < md.index("**s² = ") < md.index("## d4") < md.index("## sweep") < md.index("## final")
    for item in ("D1", "D2", "D3", "D4", "D5", "D6", "| 1 ", "| 6 ", "| 7 "):
        assert item in md
    for text in ("w/s²", "kkt_w", "band limit", "h(w)", "ψ_hp", "cal.gain[2] (reference)", "box lower", "w at s² + 2σ_s", "off the grid", "about 20%"):
        assert text in md
    for f in ("image_half_beam", "image_at_beam", "sweep", "psi_hp"):
        assert (run / "figures" / f"{f}.png").stat().st_size > 5000 and f"figures/{f}.png" in md
    assert "MPoL: no MPoL export given. CLEAN: no CLEAN directory given." in md and "not found" in md  # the missing MPoL and CLEAN images are said, not fatal
    assert "DSHARP's nominal beam" in md


def test_the_thresholds_are_the_modules_constants(tmp_path):
    md = rep.build(make_run(tmp_path / "run"))
    assert f"≤ {vl.SIGMA_REL_MAX:g}" in md and f"< {vl.KKT_W_MAX:g}" in md and f"< {vl.BAND_MAX:g}" in md and f"{vl.D4_REL_MAX:g} s²" in md and f"≤ {vl.MAX_SECANT} secant" in md


@pytest.mark.parametrize("stage, patch, item", [
    ("sigma", dict(sigma_rel=5e-3), "1"), ("sigma", dict(ref_obs=1), "D3"), ("d4", dict(bias=0.1, bias_in_se=9.0, bias_rel=0.09), "D4"), ("sweep", dict(m=50.0), "D2"),
    ("sweep", dict(n_rows=10**6), "D6"), ("sweep", dict(weight=1.0, w_over_s2=1.0 / S2), "D5"), ("sweep", dict(unconverged_fits=[dict(weight=1.0, kkt_w=1e-2)]), "6"),
    ("final", dict(kkt_w=2e-4), "6"), ("final", dict(band_limit=2.5), "6"),
])
def test_each_check_can_fail(tmp_path, stage, patch, item):
    recs, _ = records(**{stage: patch})
    bad = [c for c in rep.checks_of(recs) if c["ok"] is False]
    assert bad and bad[0]["item"] == item and len(bad) == 1
    assert "FAIL:" in rep.build(make_run(tmp_path / "run", **{stage: patch}))


def test_a_nuisance_at_its_box_edge_fails_item_7(tmp_path):
    recs, _ = records()
    nuis = recs["final"]["nuisances"]
    nuis["cal.gain"] = dict(nuis["cal.gain"], edge_frac=[0.5, 0.02, 0.5, 0.5], at_edge=[False, True, False, False])
    bad = [c for c in rep.checks_of(dict(recs, final=dict(recs["final"], nuisances=nuis))) if c["ok"] is False]
    assert [c["item"] for c in bad] == ["7"]


def test_a_failed_stage_shows_its_reason_and_skips_the_rest(tmp_path):
    run = make_run(tmp_path / "run", sigma=failed("sigma_s / s^2 = 0.01 exceeds 0.002: the noise scale is not known well enough", sigma_rel=1e-2, s2=S2))
    md = rep.write(run)
    assert "## sigma — FAILED" in md and "**Reason:** sigma_s / s^2 = 0.01 exceeds 0.002" in md
    for stage in ("d4", "sweep", "final"):
        assert f"## {stage}\n\nSkipped: stage `sigma` did not complete." in md
    assert "Figures" in md and "Skipped" in md and not (run / "figures" / "psi_hp.png").exists()
    assert "FAIL: 1 " in md


def test_a_failed_sweep_keeps_its_grid_and_skips_the_final(tmp_path):
    recs, _ = records()
    run = make_run(tmp_path / "run", sweep=failed("D1: m(w) has no sign change on the grid", crossings=[]))
    md = rep.build(run)
    assert "## sweep — FAILED" in md and "D1: m(w) has no sign change" in md and "| w/s² |" in md  # the fit points are still the failed stage's record
    assert "## final\n\nSkipped: stage `sweep` did not complete." in md and "## sigma\n\n**Raw" in md and "## d4\n\n|" in md
    assert recs["sweep"]["points"]


def test_missing_stages_are_not_an_error(tmp_path):
    (tmp_path / "empty").mkdir()
    assert "Not run (no `sigma.json`)" in rep.build(tmp_path / "empty")
    run = make_run(tmp_path / "two", d4=None, sweep=None, final=None)
    md = rep.build(run)
    assert "## d4\n\nNot run" in md and "## sweep\n\nSkipped" in md and "Not complete" in md


def test_the_blur_puts_the_beam_major_axis_at_its_position_angle():
    """Row 0 is North and column 0 East: a point source blurred by a beam at PA 30 degrees has its major axis at PA 30, whatever the sign convention of the FFT grid."""
    img = np.zeros((NPIX, NPIX))
    img[NPIX // 2 - 1, NPIX // 2 - 1] = 1.0
    for pa in (0.0, 30.0, 115.0):
        e, n, got = rep.moments(rep.blur(img, PIX, (450.0, 150.0, pa)), PIX)
        assert abs(e) < 1e-6 and abs(n) < 1e-6 and abs((got - pa + 90) % 180 - 90) < 1.0
    assert rep.blur(disc_flux(), PIX, (80.0, 80.0, 0.0)).sum() == pytest.approx(disc_flux().sum())  # flux preserved


def test_north_is_up_and_east_is_left():
    img = np.zeros((NPIX, NPIX))
    img[NPIX // 2 - 1 - 10, NPIX // 2 - 1 + 5] = 1.0  # ten rows towards row 0, five columns towards higher column
    e, n, _ = rep.moments(img, PIX)
    assert n == pytest.approx(10 * PIX) and e == pytest.approx(-5 * PIX)  # North positive; columns run towards West
    ex = rep.extent(img.shape, PIX)
    assert ex[0] > 0 > ex[1] and ex[3] > 0 > ex[2]  # imshow: East (+) on the left, North (+) at the top


def test_the_disc_pa_check_reads_a_known_orientation(tmp_path):
    flux = np.zeros((NPIX, NPIX))
    r0 = NPIX // 2 - 1
    for t in np.linspace(-10, 10, 21):  # a line along PA 144.5 East of North
        pa = np.radians(rep.DISC_PA_DEG)
        flux[int(round(r0 - t * np.cos(pa))), int(round(r0 - t * np.sin(pa)))] = 1.0
    assert rep.moments(flux, PIX)[2] == pytest.approx(rep.DISC_PA_DEG, abs=3.0)


def test_mpol_and_clean_images_are_located_and_shown(tmp_path):
    from astropy.io import fits

    mp = tmp_path / "mpol" / "export"
    mp.mkdir(parents=True)
    n5 = 144  # 5 mas pixels
    rr, cc = np.indices((n5, n5))
    r = np.hypot(rr - (n5 // 2 - 1), cc - (n5 // 2 - 1)) * 5.0
    np.savez(mp / "export.npz", image=np.exp(-0.5 * ((r - 500) / 150) ** 2) * 1e-2, cell_size_arcsec=0.005)
    cl = tmp_path / "clean"
    cl.mkdir()
    hdr = fits.Header()
    for k, v in dict(CTYPE1="RA---SIN", CTYPE2="DEC--SIN", CRVAL1=rep.RA_DEG, CRVAL2=rep.DEC_DEG, CRPIX1=NPIX // 2 + 1, CRPIX2=NPIX // 2 + 1, CDELT1=-PIX / 3.6e6, CDELT2=PIX / 3.6e6, BMAJ=50 / 3.6e6, BMIN=40 / 3.6e6, BPA=100.0).items():
        hdr[k] = v
    yy, xx = np.indices((NPIX, NPIX))
    rc = np.hypot(yy - NPIX // 2, xx - NPIX // 2) * PIX
    fits.PrimaryHDU(np.exp(-0.5 * ((rc - 500) / 150) ** 2) * 1e-3, hdr).writeto(cl / "image.fits")
    md = rep.write(make_run(tmp_path / "run"), mpol=tmp_path / "mpol", clean=cl)
    assert "MPoL: " + str(mp / "export.npz") in md and str(cl / "image.fits") in md and "our CLEAN" not in md.split("## Figures")[0]
    assert "Beam: 50.0 × 40.0 mas, PA 100° (our CLEAN)" in md
    assert "| MPoL | " in md and "not found" not in md.split("Orientation check")[1]
    _, note = rep.locate_clean(tmp_path / "nowhere")
    assert "no image.fits" in note and rep.locate_mpol(tmp_path / "nowhere")[0] is None


def test_the_report_is_spliced_into_the_docs_page(tmp_path):
    page = tmp_path / "page.md"
    page.write_text(f"intro\n\n{rep.BEGIN}\n\nPlaceholder\n\n{rep.END}\n\noutro\n")
    run = make_run(tmp_path / "run")
    rep.write(run, page=page)
    text = page.read_text()
    assert text.startswith("intro") and text.rstrip().endswith("outro") and "Placeholder" not in text and "## Pre-registered checks" in text and "# IM Lup: the registered weight rule" not in text
    assert "](page/psi_hp.png)" in text and (tmp_path / "page" / "psi_hp.png").exists()
    with pytest.raises(ValueError, match="gen:lambda-report"):
        rep.splice("no markers", "x")


def test_psi_hp_map_matches_the_registered_statistics():
    pixels = np.sqrt(disc_flux())
    psi, rel = vl.psi_hp_map(pixels, PIX)
    st = vl.psi_hp(pixels, PIX)
    assert st["disc"] == pytest.approx(psi[rel >= 1e-3].std()) and st["hi"] == pytest.approx(psi[rel >= 1e-1].std()) and psi[rel < 1e-3].max() == 0.0


def test_the_command_writes_the_report_and_refuses_nothing_on_a_failed_run(tmp_path):
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    script = pytest.importorskip("alma_imlup_maxent")
    run = make_run(tmp_path / "run", sweep=failed("D1: m(w) has no sign change on the grid"))
    script.main(["lambda-report", "--run", str(run), "--mpol", str(tmp_path / "none")])
    assert "D1: m(w) has no sign change" in (run / "report.md").read_text()


def test_the_docs_page_has_the_results_block_is_in_the_nav_and_links_no_design_notes():
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    page = (root / "docs/imaging/imlup_maxent.md").read_text()
    assert rep.BEGIN in page and rep.END in page and "imaging/imlup_maxent.md" in (root / "mkdocs.yml").read_text()
    assert ".md)" not in "".join(l for l in page.splitlines() if "design/" in l) and "](design/" not in page and "(../../design" not in page
    assert rep.splice(page, "# T\n\nbody").count("body") == 1
