"""The IM Lup CLEAN image: the restoring-beam fit and the multi-scale minor cycles on analytic inputs (NumPy only), and the
Cotton-Schwab loop of ``scripts/alma_imlup_clean.py`` on small synthetic ``ComplexVisData`` (needs ``virgil.radio``).
The design is ``design/alma_clean.md``."""

import pathlib
import sys

import numpy as np
import pytest

from crosscheck import alma_clean as ac
from evidence.plugin import record

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _psf(n=129, bmaj=7.0, bmin=4.0, pa=30.0, side=0.1, cos=True):
    """A Gaussian main lobe plus a ring-like sidelobe pattern of known amplitude (the Gaussian is exact inside the lobe)."""
    c = (n - 1) / 2
    main = ac.gaussian_image((n, n), (c, c), bmaj, bmin, pa)
    rr, cc = np.indices((n, n))
    r = np.hypot(rr - c, cc - c)
    return main - side * np.exp(-(((r - 14.0) / 3.0) ** 2)) * (np.cos(0.5 * (cc - c)) if cos else 1.0)


@pytest.mark.validates("crosscheck.alma_clean", roots=["mathematics"])
@pytest.mark.parametrize("pa", [0.0, 30.0, 95.0, 160.0])
def test_beam_fit_recovers_analytic_gaussian(pa):
    """The fit of an exact Gaussian PSF returns its FWHM and position angle, on an even and an odd grid."""
    for n in (128, 129):
        c = (n - 1) / 2
        psf = ac.gaussian_image((n, n), (c, c), 9.0, 5.0, pa)
        a, b, p, centre = ac.fit_beam(psf, pixel_mas=3.0)
        record(f"beam fit major error n={n} pa={pa}", abs(a / 27.0 - 1))
        assert a == pytest.approx(27.0, rel=2e-3) and b == pytest.approx(15.0, rel=2e-3)
        assert abs((p - pa + 90) % 180 - 90) < 0.2
        assert centre == pytest.approx((c, c), abs=1e-3)


@pytest.mark.validates("crosscheck.alma_clean", roots=["mathematics"])
def test_beam_fit_ignores_sidelobes_and_orients_major_axis():
    a, b, p, _ = ac.fit_beam(_psf(), pixel_mas=1.0)
    assert a == pytest.approx(7.0, rel=0.03) and b == pytest.approx(4.0, rel=0.03) and p == pytest.approx(30.0, abs=3.0)
    assert ac.sidelobe_level(_psf()) == pytest.approx(0.1, rel=0.1)


@pytest.mark.validates("crosscheck.alma_clean", roots=["mathematics"])
def test_minor_cycle_recovers_point_and_gaussian():
    """A point source and a broad Gaussian blob, dirty image = model * PSF exactly: the cycle finds the point at its pixel with
    its flux, the blob at its scale, and drives the residual to the threshold."""
    n = 129
    psf = _psf(n, side=0.02, cos=False)
    c = (n - 1) // 2
    truth = np.zeros((n, n))
    truth[c + 6, c - 9] = 0.8  # a point
    blob = ac.gaussian_image((n, n), (c - 15, c + 12), 12.0, 12.0, 0.0)  # peak-1 blob of FWHM 12 pixels
    truth += 0.5 * blob * 0.02
    import scipy.signal as ss

    dirty = ss.fftconvolve(truth, psf, mode="same")
    mask = ac.ellipse_mask((n, n), (c, c), 50, 50, 0)
    comps, it, peaks = ac.multiscale_minor(dirty, psf, [0, 12], mask, gain=0.2, niter=2000, threshold=1e-3, patch=64)
    model = ac.model_image(comps, [0, 12])
    record("minor cycle flux ratio", model.sum() / truth.sum())
    assert model.sum() == pytest.approx(truth.sum(), rel=0.05)
    assert comps[0][c + 6, c - 9] == pytest.approx(0.8, rel=0.15)  # most of the point's flux is in its own pixel
    assert comps[1].sum() > 0 and np.unravel_index(np.argmax(model * (ac.gaussian_image((n, n), (c - 15, c + 12), 30, 30, 0) > 0.5)), (n, n)) == pytest.approx((c - 15, c + 12), abs=2)
    resid = dirty - ss.fftconvolve(model, psf, mode="same")
    assert np.abs(resid[mask]).max() < 5e-3


@pytest.mark.validates("crosscheck.alma_clean", roots=["mathematics"])
def test_restore_puts_a_point_at_its_pixel_with_unit_peak():
    n = 65
    model = np.zeros((n, n))
    model[20, 40] = 2.0
    out = ac.restore(model, np.zeros((n, n)), (6.0, 4.0, 40.0))
    assert np.unravel_index(np.argmax(out), out.shape) == (20, 40) and out.max() == pytest.approx(2.0, rel=1e-6)
    # flux in Jy: the sum over the restored image is the model flux times the beam area in pixels
    assert out.sum() == pytest.approx(2.0 * np.pi * 6.0 * 4.0 / (4 * np.log(2)), rel=1e-3)


@pytest.mark.validates("crosscheck.alma_clean", roots=["mathematics"])
def test_taper_is_the_transform_of_the_image_plane_gaussian():
    q = np.array([0.0, 1e6, 2e6])
    t = ac.uv_taper(q, 0 * q, 40.0, 40.0, 0.0)  # circular
    assert t[0] == 1.0 and t[1] == pytest.approx(np.exp(-((np.pi * 40 * ac.MAS * 1e6) ** 2) / (4 * np.log(2))))
    # the major axis at PA 90 (East) is along u: the narrow-in-uv direction
    assert ac.uv_taper(1e6, 0.0, 60.0, 20.0, 90.0) < ac.uv_taper(0.0, 1e6, 60.0, 20.0, 90.0)


@pytest.mark.validates("crosscheck.alma_clean", roots=["mathematics"])
def test_position_angle_is_east_of_north_by_hand():
    """Hand-built from sky offsets (East = smaller column, North = smaller row): a lobe elongated towards the North-East is PA 45,
    not 135, for ``fit_beam`` and ``ellipse_mask``, so the BPA and DSHARP's mask (PA 145) are not mirrored."""
    n, c = 65, 32
    rr, cc = np.indices((n, n), dtype=float)
    east, north = c - cc, c - rr
    ne, nw = (east + north) / np.sqrt(2), (east - north) / np.sqrt(2)  # along the NE-SW diagonal, along the NW-SE one
    psf = np.exp(-4 * np.log(2) * ((ne / 9.0) ** 2 + (nw / 5.0) ** 2))
    a, b, pa, _ = ac.fit_beam(psf, pixel_mas=1.0)
    assert (a, b) == pytest.approx((9.0, 5.0), rel=0.01) and pa == pytest.approx(45.0, abs=0.5)
    m45 = ac.ellipse_mask((n, n), (c, c), 12, 4, 45.0)
    m135 = ac.ellipse_mask((n, n), (c, c), 12, 4, 135.0)
    assert m45[c - 6, c - 6] and not m135[c - 6, c - 6]  # a pixel to the North-East
    assert m135[c - 6, c + 6] and not m45[c - 6, c + 6]  # and one to the North-West


@pytest.mark.validates("crosscheck.alma_clean", roots=["mathematics"])
@pytest.mark.parametrize("pa", [30.0, 137.53])
def test_taper_position_angle_against_a_direct_dft(pa):
    """The taper at an oblique PA is the discrete transform of an image-plane Gaussian built by hand from sky offsets (major axis
    East*sin(PA) + North*cos(PA)); the mirror point u -> -u differs."""
    n, px = 257, 4.0
    x = (np.arange(n) - (n - 1) / 2) * px  # offsets in mas
    dra, ddec = np.meshgrid(x, x, indexing="xy")
    p = np.deg2rad(pa)
    along, across = dra * np.sin(p) + ddec * np.cos(p), dra * np.cos(p) - ddec * np.sin(p)
    g = np.exp(-4 * np.log(2) * ((along / 60.0) ** 2 + (across / 25.0) ** 2))
    q = 1e6
    for u, v in ((q * np.sin(p), q * np.cos(p)), (q * np.cos(p), -q * np.sin(p)), (-q * np.sin(p), q * np.cos(p))):
        dft = (g * np.exp(-2j * np.pi * (u * dra * ac.MAS + v * ddec * ac.MAS))).sum() / g.sum()
        assert ac.uv_taper(u, v, 60.0, 25.0, pa) == pytest.approx(dft.real, abs=2e-3)
    assert abs(ac.uv_taper(q * np.sin(p), q * np.cos(p), 60.0, 25.0, pa) - ac.uv_taper(-q * np.sin(p), q * np.cos(p), 60.0, 25.0, pa)) > 0.05


# ------------------------------------------------------------- with virgil

vr = pytest.importorskip("virgil.radio")
import alma_imlup_clean as script  # noqa: E402


def _data(n=3000, seed=2, npix=65, pixel_mas=20.0, truth=None):
    rng = np.random.default_rng(seed)
    q = rng.uniform(1.5e5, 2.4e6, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    u, v = q * np.cos(phi), q * np.sin(phi)
    c = (npix - 1) / 2
    rows, cols = np.mgrid[:npix, :npix]
    dra, ddec = (c - cols) * pixel_mas * ac.MAS, (c - rows) * pixel_mas * ac.MAS
    vis = np.zeros(n, complex)
    for r, cc, f in truth:
        vis += f * np.exp(-2j * np.pi * (u * dra[r, cc] + v * ddec[r, cc]))
    w = np.full(n, 1e8)
    vis = vis + (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(w)
    return vr.ComplexVisData.from_arrays(u, v, vis, w, np.full(n, 2.3e11))


def _run(data, npix=65, pixel_mas=20.0, taper=None, scales=(0, 4), threshold=2e-3, **kw):
    imager = script.Imager(data, npix, pixel_mas, 0.5, taper, npix)
    c = (npix - 1) // 2
    mask = ac.ellipse_mask((npix, npix), (c, c), 28, 28, 0)
    return imager, script.clean(imager, scales_pix=list(scales), mask=mask, gain=0.1, bias=0.6, threshold=threshold, niter=3000, cycleniter=300, cyclefactor=1.0, patch=32, time_limit=0, log=lambda *a: None, **kw)


@pytest.mark.validates("virgil.radio.ComplexVisData.model", "virgil.radio.ComplexVisData.dirty_image", roots=["mathematics"])
def test_cotton_schwab_recovers_point_sources():
    """Two point sources on pixel centres: the model puts their fluxes on those pixels (the exact major cycle removes the PSF
    approximation of the minor cycles) and the restored image peaks on them. The flux pins ``ComplexVisData.model``'s scale (the PSF
    normalisation cancels a scale error of ``dirty_image``); of ``dirty_image`` it pins the geometry: orientation, phase-centre pixel, pixel scale."""
    truth = [(36, 28, 0.20), (25, 40, 0.08)]  # (row, col, Jy)
    data = _data(truth=truth)
    imager, res = _run(data)
    m = res["model"]
    record("CS point sources flux ratio", m.sum() / 0.28)
    assert res["stop"] == "threshold"
    assert m.sum() == pytest.approx(0.28, rel=0.03)
    for r, c, f in truth:
        assert m[r - 1 : r + 2, c - 1 : c + 2].sum() == pytest.approx(f, rel=0.08)
    # the peak of the restored image is on the brighter source, with its flux
    bmaj, bmin, pa = res["beam"]
    img = ac.restore(m, res["residual"], (bmaj / 20.0, bmin / 20.0, pa))
    assert np.unravel_index(np.argmax(img), img.shape) == (36, 28) and img.max() == pytest.approx(0.20, rel=0.08)
    assert res["history"][-1]["peak_residual_jy"] < 2e-3


@pytest.mark.validates("virgil.radio.ComplexVisData.model", "virgil.radio.ComplexVisData.dirty_image", roots=["mathematics"])
def test_cotton_schwab_recovers_a_gaussian_with_the_taper():
    """A Gaussian of 5-pixel FWHM through the uv taper: the restored image has the tapered beam, peak and flux of the
    Gaussian convolved with it, and the model carries the source's flux; the loop converges to the threshold, and the beam has the
    taper's position angle (as for ``dirty_image``, geometry only for that part)."""
    npix, px = 65, 20.0
    c = (npix - 1) / 2
    rows, cols = np.mgrid[:npix, :npix]
    g = ac.gaussian_image((npix, npix), (c + 4, c - 3), 5.0, 5.0, 0.0)
    g *= 0.3 / g.sum()
    truth = [(r, cc, g[r, cc]) for r, cc in zip(*np.nonzero(g > 1e-6 * g.max()))]
    data = _data(n=4000, truth=truth, seed=4)
    taper = (60.0, 40.0, 20.0)
    imager, res = _run(data, taper=taper, scales=(0, 5, 10), threshold=1.5e-3)
    assert res["stop"] == "threshold" and res["history"][-1]["peak_residual_jy"] < 1.5e-3
    assert res["model"].sum() == pytest.approx(0.3, rel=0.06)
    bmaj, bmin, pa = res["beam"]
    img = ac.restore(res["model"], res["residual"], (bmaj / px, bmin / px, pa))
    # Jy/beam of a source with flux F and FWHM s in a beam b: F * area(b) / (area of the convolution = pi (s^2 + b^2)/(4 ln 2))
    expect = 0.3 * bmaj * bmin / np.sqrt((bmaj**2 + 25.0 * px**2) * (bmin**2 + 25.0 * px**2))
    record("CS gaussian peak ratio", img.max() / expect)
    assert img.max() == pytest.approx(expect, rel=0.08)
    # the PSF's main lobe is wider than the untapered one in the taper's direction: the beam exceeds the taper's own size
    assert bmaj > 60.0 * 0.9 and bmin > 40.0 * 0.9
    assert abs((pa - 20.0 + 90) % 180 - 90) < 5.0  # the uv coverage is isotropic: the beam follows the taper (not its mirror, 160 deg)
    assert bmaj / bmin > 1.1
    assert np.unravel_index(np.argmax(img), img.shape) == pytest.approx((int(c + 4), int(c - 3)), abs=1.5)


@pytest.mark.validates("virgil.radio_clean.clean", "virgil.radio_clean.fit_beam", roots=["mathematics"])
def test_virgil_radio_clean_agrees_with_the_independent_loop():
    """``virgil.radio_clean.clean`` and this repo's NumPy loop (the same recipe: Briggs, taper, multi-scale, exact major cycles)
    give the same restored image on synthetic point sources: the two codes share only the visibilities."""
    rc = pytest.importorskip("virgil.radio_clean")
    truth = [(36, 28, 0.20), (25, 40, 0.08)]
    data = _data(truth=truth)
    taper = (60.0, 40.0, 20.0)
    npix, px = 65, 20.0
    c = (npix - 1) // 2
    mask = ac.ellipse_mask((npix, npix), (c, c), 28, 28, 0)
    imager = script.Imager(data, npix, px, 0.5, taper, npix)
    ref = script.clean(imager, scales_pix=[0, 4], mask=mask, gain=0.1, bias=0.6, threshold=2e-3, niter=3000, cycleniter=300, cyclefactor=1.0, patch=32, time_limit=0, log=lambda *a: None)
    res = rc.clean(
        data, npix, px, deconvolver="multiscale", scales_mas=[0.0, 4 * px], weighting="briggs", robust=0.5, uvtaper=taper, threshold=2e-3, niter=3000,
        cycleniter=300, smallscalebias=0.6, psfpatch=32, mask=mask, uv_cell=1.0 / (npix * px * ac.MAS),
    )
    bmaj, bmin, pa = ref["beam"]
    ref_img = ac.restore(ref["model"], ref["residual"], (bmaj / px, bmin / px, pa))
    np.testing.assert_allclose(res.beam, ref["beam"], rtol=1e-3)
    err = np.abs(res.image - ref_img).max() / ref_img.max()
    record("radio_clean vs independent loop, max image difference over peak", err)
    assert err < 0.03
    assert res.model.sum() == pytest.approx(ref["model"].sum(), rel=0.02)


@pytest.mark.validates("virgil.radio_clean.fit_beam", roots=["mathematics"], headline="max_rel_axis_err")
@pytest.mark.parametrize("pa", [0.0, 35.0, 125.0])
def test_virgil_fit_beam_recovers_analytic_gaussian_and_pa_east_of_north(pa):
    """``virgil.radio_clean.fit_beam`` recovers the major/minor FWHM (mas) and position angle East of North from an analytic
    Gaussian main lobe built directly in sky offsets (East = smaller column, North = smaller row), while ignoring negative
    sidelobes below ``psfcutoff=0.35``."""
    rc = pytest.importorskip("virgil.radio_clean")
    px = 3.0
    bmaj_mas, bmin_mas = 27.0, 15.0
    p = np.deg2rad(pa)
    max_axis_err = 0.0
    max_pa_err = 0.0
    for n in (64, 65):
        c = (n - 1) / 2.0
        rr, cc = np.indices((n, n), dtype=float)
        east, north = (c - cc) * px, (c - rr) * px
        along = east * np.sin(p) + north * np.cos(p)
        across = east * np.cos(p) - north * np.sin(p)
        lobe = np.exp(-4.0 * np.log(2.0) * ((along / bmaj_mas) ** 2 + (across / bmin_mas) ** 2))
        r_pix = np.hypot(rr - c, cc - c)
        psf = lobe - 0.12 * np.exp(-(((r_pix - 14.0) / 2.5) ** 2))
        a, b, got_pa = rc.fit_beam(psf, px, psfcutoff=0.35)
        max_axis_err = max(max_axis_err, abs(a / bmaj_mas - 1.0), abs(b / bmin_mas - 1.0))
        max_pa_err = max(max_pa_err, abs((got_pa - pa + 90.0) % 180.0 - 90.0))
    record("max_rel_axis_err", float(max_axis_err))
    record("max_pa_err_deg", float(max_pa_err))
    assert max_axis_err < 2e-3
    assert max_pa_err < 0.2


@pytest.mark.validates("virgil.radio_clean.fit_beam", roots=["mathematics"], kind="control")
def test_control_virgil_fit_beam_rejects_mirrored_pa():
    """An oblique lobe elongated towards PA 35 deg East of North is separated by 70 deg from its East-West mirror (PA 145 deg)."""
    rc = pytest.importorskip("virgil.radio_clean")
    n, px, pa = 65, 3.0, 35.0
    c = (n - 1) / 2.0
    rr, cc = np.indices((n, n), dtype=float)
    east, north = (c - cc) * px, (c - rr) * px
    p = np.deg2rad(pa)
    along, across = east * np.sin(p) + north * np.cos(p), east * np.cos(p) - north * np.sin(p)
    psf = np.exp(-4.0 * np.log(2.0) * ((along / 27.0) ** 2 + (across / 15.0) ** 2))
    _, _, got_pa = rc.fit_beam(psf, px)
    mirror_pa = 180.0 - pa
    miss_deg = abs((got_pa - mirror_pa + 90.0) % 180.0 - 90.0)
    record("mirrored_pa_miss_deg", float(miss_deg))
    assert miss_deg > 60.0


@pytest.mark.validates("virgil.radio_clean.restore", roots=["mathematics"], headline="flux_integral_rel_err")
def test_virgil_restore_convolves_jy_per_pixel_into_jy_per_beam_and_adds_residual():
    """``virgil.radio_clean.restore`` convolves a Jy/pixel model with a peak-1 elliptical Gaussian of ``(bmaj_mas, bmin_mas, pa_deg)``
    and adds the residual unchanged: an isolated point of flux ``F`` peaks at ``F`` Jy/beam, integrates to ``F * Omega_beam / px^2``,
    and agrees pixel-by-pixel with ``crosscheck.alma_clean.restore``."""
    rc = pytest.importorskip("virgil.radio_clean")
    n, px = 65, 5.0
    beam_mas = (35.0, 20.0, 38.0)
    model = np.zeros((n, n))
    flux = 1.7
    model[28, 37] = flux
    zero_res = np.zeros((n, n))
    restored0 = np.asarray(rc.restore(model, zero_res, beam_mas, px))
    ref0 = ac.restore(model, zero_res, (beam_mas[0] / px, beam_mas[1] / px, beam_mas[2]))
    beam_area_pix = np.pi * (beam_mas[0] / px) * (beam_mas[1] / px) / (4.0 * np.log(2.0))
    flux_rel_err = abs(restored0.sum() / (flux * beam_area_pix) - 1.0)
    max_ref_diff = float(np.abs(restored0 - ref0).max() / flux)
    rng = np.random.default_rng(7)
    residual = 0.03 * rng.normal(size=(n, n))
    restored = np.asarray(rc.restore(model, residual, beam_mas, px))
    resid_add_err = float(np.abs((restored - restored0) - residual).max())
    record("flux_integral_rel_err", float(flux_rel_err))
    record("max_diff_vs_crosscheck_over_peak", max_ref_diff)
    record("residual_additivity_abs_err", resid_add_err)
    assert np.unravel_index(np.argmax(restored0), (n, n)) == (28, 37)
    assert restored0.max() == pytest.approx(flux, rel=1e-6)
    assert flux_rel_err < 1e-10
    assert max_ref_diff < 1e-12
    assert resid_add_err < 1e-13


@pytest.mark.validates("virgil.radio_clean.restore", roots=["mathematics"], kind="control")
def test_control_virgil_restore_wrong_pa_misses():
    """Restoring an elongated beam at the mirrored PA ``180 - pa`` changes off-peak lobe intensities by order one."""
    rc = pytest.importorskip("virgil.radio_clean")
    n, px = 65, 5.0
    model = np.zeros((n, n))
    model[32, 32] = 1.0
    right = np.asarray(rc.restore(model, np.zeros((n, n)), (45.0, 15.0, 35.0), px))
    wrong = np.asarray(rc.restore(model, np.zeros((n, n)), (45.0, 15.0, 145.0), px))
    miss = float(np.abs(right - wrong).max())
    record("mirrored_pa_restore_miss", miss)
    assert miss > 0.25


@pytest.mark.validates("virgil.radio_clean.RadioCleanResult", roots=["mathematics", "standards"], headline="max_rel_vis_err")
def test_virgil_radio_clean_result_invariants_and_model_visibilities(tmp_path):
    """``RadioCleanResult`` satisfies ``image == restore(model, residual, beam, px)``, ``model_visibilities(data)`` matches the
    direct DFT of ``model``, ``pb_corrected`` divides by the power beam above ``pblimit`` (NaN outside), and ``write_fits``
    round-trips the four image products (flipped to FITS South-at-row-0 WCS with ``CDELT1 < 0 < CDELT2``) and beam keywords."""
    from astropy.io import fits

    rc = pytest.importorskip("virgil.radio_clean")
    npix, px = 65, 20.0
    truth = [(36, 28, 0.20), (25, 40, 0.08)]
    data = _data(n=1200, truth=truth, npix=npix, pixel_mas=px)
    res = rc.clean(data, npix, px, deconvolver="hogbom", niter=400, gain=0.1, threshold=2e-3)
    assert isinstance(res, rc.RadioCleanResult)
    re_restored = np.asarray(rc.restore(res.model, res.residual, res.beam, res.pixel_scale_mas))
    restore_consistency = float(np.abs(np.asarray(res.image) - re_restored).max())

    # Direct DFT of res.model on sky offsets (East = (c - col)*px, North = (c - row)*px)
    c = (npix - 1) / 2.0
    rr, cc = np.mgrid[:npix, :npix]
    dra = (c - cc) * px * ac.MAS
    ddec = (c - rr) * px * ac.MAS
    u, v = np.asarray(data.u), np.asarray(data.v)
    nz = np.nonzero(np.asarray(res.model) != 0.0)
    direct_vis = np.zeros(len(u), dtype=complex)
    model_np = np.asarray(res.model, float)
    for r, col in zip(*nz):
        direct_vis += model_np[r, col] * np.exp(-2j * np.pi * (u * dra[r, col] + v * ddec[r, col]))
    got_vis = np.asarray(res.model_visibilities(data))
    max_rel_vis_err = float(np.abs(got_vis - direct_vis).max() / np.abs(direct_vis).max())

    # Primary-beam correction and pblimit cutoff
    pb = np.exp(-(((rr - c) ** 2 + (cc - c) ** 2) / (18.0**2)))
    pbcor = np.asarray(res.pb_corrected(pb, pblimit=0.2))
    inside = pb >= 0.2
    pb_err = float(np.abs(pbcor[inside] - np.asarray(res.image)[inside] / pb[inside]).max())
    assert np.all(np.isnan(pbcor[~inside]))

    # FITS round-trip (virgil row 0 is North; standard FITS CDELT2 > 0 puts South at row 0, i.e. arr[::-1])
    prefix = str(tmp_path / "clean_prod")
    res.write_fits(prefix, ra_deg=239.8, dec_deg=-37.8, freq_hz=230e9)
    for suffix, arr in (("image", res.image), ("model", res.model), ("residual", res.residual), ("psf", res.psf)):
        with fits.open(f"{prefix}.{suffix}.fits") as hdul:
            assert hdul[0].header["CDELT1"] < 0.0 and hdul[0].header["CDELT2"] > 0.0
            assert np.allclose(hdul[0].data.squeeze()[::-1], np.asarray(arr), rtol=1e-6, atol=1e-7)
            if suffix == "image":
                assert hdul[0].header["BMAJ"] * 3.6e6 == pytest.approx(res.beam[0], rel=1e-5)
                assert hdul[0].header["BMIN"] * 3.6e6 == pytest.approx(res.beam[1], rel=1e-5)

    record("max_rel_vis_err", max_rel_vis_err)
    record("restore_consistency_abs_err", restore_consistency)
    record("pb_corrected_abs_err", pb_err)
    assert restore_consistency < 1e-12
    assert max_rel_vis_err < 1e-10
    assert pb_err < 1e-12


@pytest.mark.validates("virgil.radio_clean.clean", roots=["self-consistency"], kind="guard")
def test_virgil_clean_keywords_bind_to_the_installed_signature(monkeypatch):
    """The keywords ``clean_virgil`` passes bind to ``virgil.radio_clean.clean`` (a renamed argument fails here, not on the cluster); the scales go in mas."""
    import inspect

    pytest.importorskip("virgil.radio_clean")
    from virgil.radio_clean import clean as vclean

    import alma_imlup_clean as script

    got = {}
    monkeypatch.setattr(script, "cmd_run", lambda args: got.setdefault("args", args))
    script.main(["run", "--data", "d", "--out", "o"])
    args = got["args"]
    kw = script.virgil_clean_kwargs(args, [0, 10.0], np.ones((3, 3), bool))
    assert not any(p.kind is p.VAR_KEYWORD for p in inspect.signature(vclean).parameters.values())  # **kwargs would accept any name
    inspect.signature(vclean).bind(None, args.npix, args.pixel_mas, **kw)  # TypeError on an unknown or missing keyword
    assert kw["scales_mas"] == [0.0, 10.0 * args.pixel_mas]


@pytest.mark.validates("crosscheck.alma_clean", roots=["self-consistency"], kind="guard")
def test_run_writes_npy_pngs_flux_and_dots_for_the_maxent_start_and_the_page(tmp_path):
    """A short ``run`` on small synthetic data writes the restored image, model, residual, PSF and beam as .npy, a PNG of each, and result.json with the model and restored flux and
    ``dots`` (the page and the gate (e) reference read these); clean.json keeps what ``--start`` reads (``model.fits``, ``beam_mas_deg``)."""
    import json

    from test_alma_maxent import synthetic_dir

    import alma_imlup_clean as script

    syn = synthetic_dir(tmp_path / "s")
    out = tmp_path / "o"
    script.main(["run", "--data", str(syn), "--out", str(out), "--npix", "65", "--pixel-mas", "20", "--niter", "30", "--scales-pix", "0", "3", "--cycleniter", "10", "--patch", "32",
                "--uv-cell-npix", "65", "--threshold-jy", "0"])
    for f in ("image.npy", "model.npy", "residual.npy", "psf.npy", "beam.npy", "image.png", "model.png", "residual.png", "image.fits", "model.fits", "clean.json"):
        assert (out / f).stat().st_size > 0, f
    r = json.loads((out / "result.json").read_text())
    assert r["flux"]["model_sum_jy"] == pytest.approx(float(np.load(out / "model.npy").sum()), rel=1e-5) and {"n_eff", "hole_fraction", "sub_taper_power"} <= set(r["dots"])
    assert len(np.load(out / "beam.npy")) == 3 and json.loads((out / "clean.json").read_text())["restored_flux_jy"] == pytest.approx(r["flux"]["restored_jy"])
