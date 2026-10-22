"""The IM Lup page generator (``scripts/alma_imlup_page.py``): its pure functions (the asinh stretch, the sky-to-pixel map and the
orientation marker, the beam ellipse, the radial profile) against closed forms and the FITS WCS, and its handling of missing inputs.
NumPy, matplotlib and astropy only; no virgil, no JAX. The page is ``docs/imaging/imlup.md``."""

import json
import math
import pathlib
import shutil
import sys

import numpy as np
import pytest

from crosscheck import alma_maxent as cx
from evidence.plugin import record

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import alma_imlup_page as P  # noqa: E402

PAGE = ROOT / "docs" / "imaging" / "imlup.md"
GUARD = pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")


# ------------------------------------------------------------------ stretch


@GUARD
def test_asinh_stretch_is_odd_normalised_and_invertible():
    """0 -> 0, vmax -> 1, odd, linear below the softening and logarithmic above it, and ``asinh_inverse`` undoes it (plumbing of the display scale)."""
    x = np.array([-3.0, -0.2, 0.0, 0.01, 0.3, 1.0, 5.0])
    s = P.asinh_stretch(x, 5.0, 0.2)
    assert s[x == 0.0] == 0.0 and P.asinh_stretch(5.0, 5.0, 0.2) == pytest.approx(1.0)
    assert np.allclose(P.asinh_stretch(-x, 5.0, 0.2), -s)
    assert np.allclose(P.asinh_inverse(s, 5.0, 0.2), x)
    # linear for |x| << soft: slope 1/(soft asinh(vmax/soft)); logarithmic for x >> soft: equal ratios give equal steps
    k = 1.0 / (0.2 * math.asinh(25.0))
    assert P.asinh_stretch(1e-4, 5.0, 0.2) == pytest.approx(1e-4 * k, rel=1e-6)
    steps = np.diff(P.asinh_stretch(np.array([2.0, 4.0, 8.0, 16.0]) * 100, 1e4, 0.2))
    assert steps[0] == pytest.approx(steps[1], rel=1e-3) and steps[1] == pytest.approx(steps[2], rel=1e-3)
    record("asinh round trip max error", float(np.max(np.abs(P.asinh_inverse(s, 5.0, 0.2) - x))))


@GUARD
def test_asinh_norm_maps_vmax_to_one_and_is_monotone():
    n = P.asinh_norm(2.0, 0.05)
    v = np.asarray(n(np.array([0.0, 0.01, 0.1, 1.0, 2.0])))
    assert v[0] == 0.0 and v[-1] == pytest.approx(1.0) and np.all(np.diff(v) > 0)


# ------------------------------------------------------------------ orientation


@pytest.mark.validates("crosscheck.alma_maxent", roots=["standards"])
@pytest.mark.parametrize("n", [64, 1028])
def test_sky_to_pixel_agrees_with_the_fits_wcs_of_the_same_grid(n):
    """A sky offset East and North of the phase centre lands on the pixel that the FITS WCS of ``crosscheck.alma_maxent.sky_wcs`` (SIN, RA decreasing with x,
    row 0 South in the file) assigns it, once its rows are flipped to this repository's layout: North is up and East is left in the image."""
    px = 5.0
    g = P.Grid(n // 2 - 1, n // 2 - 1, px / 1000.0)
    w = cx.sky_wcs(n, px, P.RA_DEG, P.DEC_DEG)
    worst = 0.0
    for east, north in [(0.0, 0.0), (0.1325, 0.1074), (-0.1325, -0.1074), (0.2, -0.15)]:
        row, col = P.sky_to_pixel(east, north, g)
        ra = P.RA_DEG + east / 3600.0 / math.cos(math.radians(P.DEC_DEG))
        dec = P.DEC_DEG + north / 3600.0
        x, y = w.all_world2pix(ra, dec, 0)  # FITS: x along the columns, y from the South
        worst = max(worst, abs(col - float(x)), abs(row - (n - 1 - float(y))))
    record(f"sky_to_pixel against the WCS, max pixel error n={n}", worst)
    assert worst < 0.01


@pytest.mark.validates("crosscheck.alma_maxent", roots=["standards"])
def test_the_mask_centre_marker_is_west_and_south_of_the_phase_centre():
    """DSHARP's mask centre (15h56m09.18880s -37d56m06.527374s) against the phase centre (09.20s, 06.42): RA is smaller, so it is West (negative East
    offset), and Dec is more negative, so it is South: about (-0.1325, -0.1074) arcsec."""
    e, n = P.mask_centre_offset()
    assert e == pytest.approx(-0.1325, abs=1e-4) and n == pytest.approx(-0.1074, abs=1e-4)
    g = P.Grid(31.5, 31.5, 0.01)
    row, col = P.sky_to_pixel(e, n, g)
    assert col > g.cc and row > g.cr  # West is to the right (column index up) and South is down (row index up)


@GUARD
def test_extent_puts_east_left_and_north_up_and_round_trips_pixels():
    g = P.Grid(31.0, 32.0, 0.01)
    left, right, bottom, top = P.grid_extent((64, 64), g)
    assert left > right and top > bottom  # East positive on the left, North on top
    assert left == pytest.approx(0.325) and right == pytest.approx(-0.315) and top == pytest.approx(0.315) and bottom == pytest.approx(-0.325)
    for r, c in [(0, 0), (31, 32), (63, 10)]:
        assert P.sky_to_pixel(*P.pixel_to_sky(r, c, g), g) == pytest.approx((r, c))


def _blob(n, grid, east, north, sigma=3.0):
    rr, cc = np.indices((n, n))
    r0, c0 = P.sky_to_pixel(east, north, grid)
    return np.exp(-((rr - r0) ** 2 + (cc - c0) ** 2) / (2 * sigma**2))


@pytest.mark.validates("crosscheck.alma_maxent", roots=["mathematics"])
def test_centroid_finds_the_known_position_and_a_flipped_image_does_not():
    """A blob placed at the mask centre has its centroid there; the same array flipped on both axes (a 180 degree rotation, which is what an East-West and
    North-South mix-up does) puts it about twice the mask-centre offset away. The control for the orientation marker."""
    g = P.Grid(31.0, 31.0, 0.01)
    me = P.mask_centre_offset()
    img = _blob(64, g, *me)
    e, n = P.centroid_en(img, g, me, radius=0.25)
    assert math.hypot(e - me[0], n - me[1]) < 1e-3
    ef, nf = P.centroid_en(img[::-1, ::-1], g, (0.0, 0.0), radius=0.25)  # a rotated image: the blob moves through the centre
    assert math.hypot(ef - me[0], nf - me[1]) > 0.3 * math.hypot(*me) * 2
    record("flipped centroid offset (arcsec)", math.hypot(ef - me[0], nf - me[1]))


@pytest.mark.validates("crosscheck.alma_maxent", roots=["mathematics"])
@pytest.mark.parametrize("pa", [0.0, 35.0, 90.0, 144.5])
def test_beam_ellipse_major_axis_lies_at_the_position_angle(pa):
    """The ellipse drawn in (East, North) data coordinates has its major axis towards (sin PA, cos PA): East of North, as CLEAN headers define the beam."""
    from matplotlib.patches import Ellipse

    e = Ellipse((0, 0), 2.0, 0.5, angle=P.beam_ellipse_angle(pa))
    v = e.get_verts()
    far = v[np.argmax(np.hypot(v[:, 0], v[:, 1]))]
    d = np.array([math.sin(math.radians(pa)), math.cos(math.radians(pa))])
    assert abs(np.dot(far / np.hypot(*far), d)) > 0.999


@pytest.mark.validates("crosscheck.alma_maxent", roots=["mathematics"])
def test_radial_profile_recovers_a_ring_of_known_radius_through_the_deprojection():
    """A ring of radius R in the disc plane, inclined by i with its major axis at PA, appears as an ellipse; the deprojected azimuthal mean peaks at R
    (and the undeprojected radius would not). Exercises the inclination, the position angle and the centre."""
    n, dx = 400, 0.01
    g = P.Grid(199.5, 199.5, dx)
    centre = (-0.1325, -0.1074)
    rr, cc = np.indices((n, n))
    e, no = P.pixel_to_sky(rr, cc, g)
    e, no = e - centre[0], no - centre[1]
    pa, inc, R = math.radians(P.PA_DEG), math.radians(P.INCL_DEG), 1.2
    a, b = e * math.sin(pa) + no * math.cos(pa), e * math.cos(pa) - no * math.sin(pa)
    ring = np.exp(-((np.hypot(a, b / math.cos(inc)) - R) ** 2) / (2 * 0.04**2))
    edges = np.arange(0.0, 2.0, 0.02)
    r, prof, cnt = P.radial_profile(ring, g, edges, centre_en=centre)
    assert r[np.nanargmax(prof)] == pytest.approx(R, abs=0.02)
    assert np.nanmax(prof) > 0.95
    wrong = P.radial_profile(ring, g, edges, centre_en=centre, incl_deg=0.0)[1]  # no deprojection: the ring is smeared over many radii
    assert np.nanmax(wrong) < 0.6
    rot = P.radial_profile(ring, g, edges, centre_en=centre, pa_deg=P.PA_DEG + 90)[1]  # the wrong position angle also smears it
    assert np.nanmax(rot) < 0.6


@GUARD
def test_crop_window_keeps_the_grid():
    g = P.Grid(99.5, 99.5, 0.01)
    img = np.arange(200 * 200, dtype=float).reshape(200, 200)
    c, cg = P.crop_window(img, g, 0.5)
    for east, north in [(0.2, -0.3), (-0.4, 0.1)]:  # the same sky offset reads the same pixel value
        r0, c0 = P.sky_to_pixel(east, north, g)
        r1, c1 = P.sky_to_pixel(east, north, cg)
        assert c[int(round(r1)), int(round(c1))] == img[int(round(r0)), int(round(c0))]
    assert c.shape[0] <= 102


# ------------------------------------------------------------------ placeholders and blocks


@GUARD
def test_pending_block_says_what_is_missing_and_replace_block_touches_only_its_block():
    p = P.pending("our CLEAN", "ozstar_scripts#60 writes it")
    assert "Pending: our CLEAN" in p and "ozstar_scripts#60" in p and p.startswith("!!! warning")
    text = "a\n<!-- gen:x -->\nold\n<!-- /gen:x -->\nb\n<!-- gen:y -->\nkeep\n<!-- /gen:y -->\n"
    new = P.replace_block(text, "x", "NEW")
    assert "NEW" in new and "old" not in new and "keep" in new and new.startswith("a\n") and new.rstrip().endswith("<!-- /gen:y -->")
    assert P.replace_block(new, "x", "NEW") == new  # idempotent
    with pytest.raises(KeyError):
        P.replace_block(text, "z", "n")


@GUARD
def test_missing_inputs_load_as_none():
    assert P.load_fit(None) is None and P.load_fit("/nonexistent/dir") is None
    assert P.load_mpol("/nonexistent") is None and P.load_compare(None) is None
    assert P.load_clean("/nonexistent") is None and P.load_sky_fits(None) is None


@GUARD
def test_gate_result_is_signed_and_tags_come_from_the_recorded_paths():
    g = P.gate_result(-108_232_200.0, -108_232_184.0, 2.70, 2.7062, True)
    assert g["passed"] and g["delta"] == pytest.approx(-16.0)
    assert not P.gate_result(-106_000_000.0, -108_232_184.0, 2.9, 2.7062, False)["passed"]
    assert P.data_column({"args": {"data": "/fred/x/vis/55eb2e9"}}) == "DATA"
    assert P.data_column({"args": {"data": "/fred/x/vis/15fecaa-cd/"}}) == "CORRECTED_DATA"
    assert P.tag_of({"args": {"out": "/fred/x/out/03531b4/R"}}) == "03531b4"


@GUARD
def test_save_small_stays_under_the_limit(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(np.random.default_rng(0).random((400, 400)))
    dpi = P.save_small(fig, tmp_path / "noise.png", max_bytes=300_000, dpi=110)
    plt.close(fig)
    assert (tmp_path / "noise.png").stat().st_size <= 300_000 and dpi <= 110


# ------------------------------------------------------------------ end to end on a synthetic fit and export


def _synthetic(tmp, flip=False):
    """A fit directory and an MPoL export of a blob at DSHARP's mask centre on a 64^2 grid of 10 mas (or its 180 degree rotation)."""
    n, px = 64, 10.0
    g = P.Grid(n // 2 - 1, n // 2 - 1, px / 1000.0)
    img = 3.0 * _blob(n, g, *P.mask_centre_offset(), sigma=2.5) + 1e-4
    if flip:
        img = img[::-1, ::-1]
    fit = tmp / "R"
    fit.mkdir(parents=True)
    np.save(fit / "image.npy", img.astype(np.float32))
    nd = [92103] * 26
    chi = [1.25] * 18 + [3.0] * 8
    res = dict(n=42_421_131, pixel_mas=px, lam=0.0848, objective_R=-108_232_300.0, chi2_raw=2.8 * 2 * 42_421_131, converged=True, stopped_because="c8", workflow="p3",
               gain=[1.01, 0.99, 1.0, 1.02, 0.98, 1.0], offset_mas=[[0.1, 0.2]] * 6, sigma=[1.1] * 18 + [1.8] * 8, chi2_s1=dict(n=nd, chi2_raw_over_n=chi),
               stages=[dict(name="all", steps=100)], wall_s=3600.0, versions=dict(vv="abc1234", virgil="0.3.0", jax="0.12", devices=["cuda:0"]),
               args=dict(data="/fred/vis/15fecaa-cd", out="/fred/out/15fecaa-cd/R"))
    (fit / "result.json").write_text(json.dumps(res))
    mp = tmp / "mpol" / "export"
    mp.mkdir(parents=True)
    amp = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    np.savez(mp / "export.npz", base_image=img, image=img, cell_size_arcsec=px / 1000.0, amp=amp, sigma=np.array([1.1] * 18 + [1.8] * 8), delta_ra_arcsec=np.zeros(7), delta_dec_arcsec=np.zeros(7))
    (mp / "export.json").write_text("{}")
    (mp / "objective.json").write_text(json.dumps(dict(n=42_421_131, objective_R=-108_232_183.7, n_ddid=nd, chi2_raw=[c * 2 * 92103 for c in chi])))
    return fit, tmp / "mpol"


def _run(tmp, *argv):
    page = tmp / "imlup.md"
    shutil.copy(PAGE, page)
    P.main(["--page", str(page), "--assets", str(tmp / "assets"), *map(str, argv)])
    return page.read_text(), tmp / "assets"


@GUARD
def test_every_block_is_pending_when_no_input_exists(tmp_path):
    text, assets = _run(tmp_path)
    for name in ("verdict", "images", "calibration", "fit", "residuals", "cv", "science", "provenance"):
        assert f"<!-- gen:{name} -->" in text and f"<!-- /gen:{name} -->" in text
    assert text.count("Pending:") >= 7 and "gate result" in text
    assert not list(assets.glob("*.png"))


@GUARD
def test_a_synthetic_run_writes_small_figures_the_gate_and_pending_placeholders(tmp_path):
    fit, mpol = _synthetic(tmp_path)
    text, assets = _run(tmp_path, "--r", fit, "--mpol", mpol, "--job-id", "R=123")
    for f in ("images.png", "calibration.png", "chi2.png", "profile.png"):
        assert (assets / f).exists() and (assets / f).stat().st_size < 1_000_000, f
    assert "The gate is reached" in text  # -108,232,300 <= -108,232,184
    assert "R_mpol: warm-started at MPoL's solution | *pending*" in text and "ozstar_scripts#60" in text  # unfinished runs and our CLEAN are marked
    assert "Pending: MPoL's residual dirty images" not in text  # no residual panel at all without any dirty image: the generic placeholder
    assert "Pending: residual dirty images" in text and "Pending: cross-validation" in text and "Pending: spiral pitch" in text
    s = json.loads((assets / "summary.json").read_text())
    assert s["found"] == {"R": True, "R_mpol": False, "R_mpolwf": False}
    # the stale placeholder warning appears only for the DATA column
    assert "Placeholder:" not in text


@GUARD
def test_a_fit_on_the_data_column_is_flagged_as_a_placeholder(tmp_path):
    fit, mpol = _synthetic(tmp_path)
    r = json.loads((fit / "result.json").read_text())
    r["args"]["data"] = "/fred/vis/55eb2e9"
    (fit / "result.json").write_text(json.dumps(r))
    text, _ = _run(tmp_path, "--r", fit, "--mpol", mpol)
    assert "Placeholder: the fit shown was read from the DATA column" in text


@pytest.mark.validates("crosscheck.alma_maxent", roots=["mathematics"])
def test_orientation_check_in_the_page_fails_for_a_rotated_image(tmp_path):
    """The generated centroid table: a disc at DSHARP's mask centre sits within a few mas of the marker; the same disc rotated by 180 degrees is
    hundreds of mas away (the page would show it)."""
    good, mp = _synthetic(tmp_path / "a")
    _run(tmp_path / "a", "--r", good, "--mpol", mp)
    s_good = json.loads((tmp_path / "a" / "assets" / "summary.json").read_text())["image_metrics"]["centroid_offset_mas"]
    bad, mp2 = _synthetic(tmp_path / "b", flip=True)
    _run(tmp_path / "b", "--r", bad, "--mpol", mp2)
    s_bad = json.loads((tmp_path / "b" / "assets" / "summary.json").read_text())["image_metrics"]["centroid_offset_mas"]
    off = lambda d: math.hypot(*d["virgil"])  # noqa: E731
    record("orientation offset, correct image (mas)", off(s_good))
    record("orientation offset, rotated image (mas)", off(s_bad))
    assert off(s_good) < 10 and off(s_bad) > 100
