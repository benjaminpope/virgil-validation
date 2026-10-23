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


@pytest.mark.validates("virgil.radio_clean.clean", roots=["mathematics"])
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
