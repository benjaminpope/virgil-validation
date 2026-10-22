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
# the page's own display geometry against closed forms: it is reference code of this repository, not a function of any package under test
REF = pytest.mark.validates("evidence", roots=["mathematics"], kind="reference")
CONTROL = pytest.mark.validates("evidence", roots=["mathematics"], kind="control")


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


@pytest.mark.validates("crosscheck.alma_maxent", roots=["standards"], kind="reference")
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


@REF
def test_centroid_finds_a_known_offset_position():
    g = P.Grid(31.0, 31.0, 0.01)
    img = _blob(64, g, 0.12, -0.08)
    e, n = P.centroid_en(img, g, (0.0, 0.0), radius=0.25)
    assert math.hypot(e - 0.12, n + 0.08) < 1e-3


def _disc(n, grid, east=0.0, north=0.0, pa=P.PA_DEG, incl=P.INCL_DEG, a=0.9):
    """An inclined Gaussian disc: semi-major axis ``a`` arcsec along position angle ``pa`` (East of North), semi-minor axis a cos(incl)."""
    rr, cc = np.indices((n, n))
    e, no = P.pixel_to_sky(rr, cc, grid)
    e, no = e - east, no - north
    p = math.radians(pa)
    along, across = e * math.sin(p) + no * math.cos(p), e * math.cos(p) - no * math.sin(p)
    return np.exp(-0.5 * ((along / a) ** 2 + (across / (a * math.cos(math.radians(incl)))) ** 2))


@CONTROL
def test_major_axis_pa_recovers_the_disc_and_fails_for_a_mirror_image():
    """The orientation control for a disc on the phase centre, where a reflection through it cannot move the disc: the position angle of the major axis of an
    inclined disc is 144.5 degrees, an East-West or North-South mirror image has 35.5, and a 180 degree rotation (which keeps it) is not caught by this
    number: the page says so and leaves that to the DSHARP comparison."""
    g = P.Grid(99.5, 99.5, 0.03)
    img = _disc(200, g)
    pa = P.major_axis_pa(img, g, radius=2.0)
    assert pa == pytest.approx(P.PA_DEG, abs=0.5)
    assert P.major_axis_pa(img[:, ::-1], g, radius=2.0) == pytest.approx(180 - P.PA_DEG, abs=0.5)  # East-West mirror
    assert P.major_axis_pa(img[::-1, :], g, radius=2.0) == pytest.approx(180 - P.PA_DEG, abs=0.5)  # North-South mirror
    assert P.major_axis_pa(img[::-1, ::-1], g, radius=2.0) == pytest.approx(P.PA_DEG, abs=0.5)  # a 180 degree rotation is invisible to it
    off = P.Grid(99.5, 99.5, 0.03)
    shifted = _disc(200, off, east=0.1, north=0.07)  # measured about the disc's own centroid
    c = P.centroid_en(shifted, off, (0.0, 0.0), radius=2.0)
    assert P.major_axis_pa(shifted, off, centre_en=c, radius=1.5) == pytest.approx(P.PA_DEG, abs=1.0)
    record("PA of the disc minus its mirror image's (deg)", pa - P.major_axis_pa(img[:, ::-1], g, radius=2.0))


@REF
@pytest.mark.parametrize("pa", [0.0, 35.0, 90.0, 144.5])
def test_beam_ellipse_major_axis_lies_at_the_position_angle(pa):
    """The ellipse drawn in (East, North) data coordinates has its major axis towards (sin PA, cos PA): East of North, as CLEAN headers define the beam."""
    from matplotlib.patches import Ellipse

    e = Ellipse((0, 0), 2.0, 0.5, angle=P.beam_ellipse_angle(pa))
    v = e.get_verts()
    far = v[np.argmax(np.hypot(v[:, 0], v[:, 1]))]
    d = np.array([math.sin(math.radians(pa)), math.cos(math.radians(pa))])
    assert abs(np.dot(far / np.hypot(*far), d)) > 0.999


@REF
def test_radial_profile_recovers_a_ring_of_known_radius_through_the_deprojection():
    """A ring of radius R in the disc plane, inclined by i with its major axis at PA, appears as an ellipse; the deprojected azimuthal mean peaks at R
    (and the undeprojected radius would not). Exercises the inclination, the position angle and the centre."""
    n, dx = 400, 0.01
    g = P.Grid(199.5, 199.5, dx)
    centre = (0.05, -0.03)  # the disc centre need not be the phase centre: the profile is taken about the centre it is given
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
    off = P.radial_profile(ring, g, edges, centre_en=(centre[0] - 0.17, centre[1] - 0.17))[1]  # a centre 0.24 arcsec off (about 5 beams) smears the ring
    assert np.nanmax(off) < 0.5


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
    """A fit directory and an MPoL export of an inclined disc on the phase centre, 64^2 grid of 10 mas (or its East-West mirror image)."""
    n, px = 64, 10.0
    g = P.Grid(n // 2 - 1, n // 2 - 1, px / 1000.0)
    img = 3.0 * _disc(n, g, a=0.12) + 1e-4  # IM Lup's disc sits on the MS's phase centre to a few mas
    if flip:
        img = img[:, ::-1]  # an East-West mirror image
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


@CONTROL
def test_orientation_check_in_the_page_fails_for_a_mirror_image(tmp_path):
    """The generated table: a disc on the phase centre has its centroid within a few mas of it and its major axis at 144.5 degrees; the East-West mirror image
    has the same centroid but 35.5 degrees, which the page shows."""
    good, mp = _synthetic(tmp_path / "a")
    text, _ = _run(tmp_path / "a", "--r", good, "--mpol", mp)
    m_good = json.loads((tmp_path / "a" / "assets" / "summary.json").read_text())["image_metrics"]
    bad, mp2 = _synthetic(tmp_path / "b", flip=True)
    _run(tmp_path / "b", "--r", bad, "--mpol", mp2)
    m_bad = json.loads((tmp_path / "b" / "assets" / "summary.json").read_text())["image_metrics"]
    assert math.hypot(*m_good["centroid_offset_mas"]["virgil"]) < 10 and math.hypot(*m_bad["centroid_offset_mas"]["virgil"]) < 15  # one 10 mas cell: the even grid mirrors about its array centre
    assert m_good["major_axis_pa_deg"]["virgil"] == pytest.approx(P.PA_DEG, abs=2.0)
    assert m_bad["major_axis_pa_deg"]["virgil"] == pytest.approx(180 - P.PA_DEG, abs=2.0)
    assert "major-axis PA" in text and "A 180° rotation leaves both numbers unchanged" in text
    record("major-axis PA, disc minus mirror image (deg)", m_good["major_axis_pa_deg"]["virgil"] - m_bad["major_axis_pa_deg"]["virgil"])


# ------------------------------------------------------------------ the gate and the DSHARP profile file


def _compare(tmp, ours, theirs):
    d = tmp / "compare"
    d.mkdir(parents=True, exist_ok=True)
    (d / "compare.json").write_text(json.dumps(dict(objective=dict(ours=dict(objective=ours), mpol_solution_in_virgil=dict(objective=theirs)))))
    return d


@GUARD
def test_the_gate_uses_the_like_for_like_number_when_compare_has_run(tmp_path):
    """Without ``compare`` the page compares our objective with MPoL's own evaluation and says it is provisional; with it, our re-evaluated objective is
    compared with MPoL's solution evaluated in virgil, the cross-code difference is reported separately, and a worse fit is not hidden by a constant offset
    between the two evaluators."""
    fit, mpol = _synthetic(tmp_path)  # our fit -108,232,300 against MPoL's own -108,232,183.7
    t0, _ = _run(tmp_path, "--r", fit, "--mpol", mpol)
    assert "provisional" in t0 and "MPoL's own number" in t0
    # MPoL's solution evaluated in virgil is lower than its own number by 5000 (a different constant): ours -108,232,300 is now ABOVE it
    cmp_ = _compare(tmp_path, -108_232_300.0, -108_237_184.0)
    t1, _ = _run(tmp_path, "--r", fit, "--mpol", mpol, "--compare", cmp_)
    assert "The gate is not reached" in t1 and "provisional" not in t1.split("Cross-code agreement")[0].split("\n", 3)[2]
    assert "Cross-code agreement" in t1 and "-5,000.3" in t1  # -108,237,184 minus -108,232,183.7
    cmp_ok = _compare(tmp_path, -108_232_300.0, -108_232_000.0)
    t2, _ = _run(tmp_path, "--r", fit, "--mpol", mpol, "--compare", cmp_ok)
    assert "The gate is reached" in t2


@GUARD
@pytest.mark.parametrize("header,rows,r_expect,i_expect", [
    ("r_arcsec  I_Jy/beam  dI_Jy/beam", [[0.5, 2.0e-3, 1e-5]], 0.5, 2.0e-3),
    ("r_au  r_arcsec  I_mJy/beam  dI_mJy/beam", [[79.0, 0.5, 2.0, 0.01]], 0.5, 2.0e-3),
    ("radius [au]  Inu [mJy/beam]", [[79.0, 2.0]], 0.5, 2.0e-3),
    ("radius(arcsec) Intensity(uJy/beam)", [[0.5, 2000.0]], 0.5, 2.0e-3),
])
def test_dsharp_profile_units_are_read_from_the_header(tmp_path, header, rows, r_expect, i_expect):
    """The loader reads the radius unit (arcsec or au, over 158 pc) and the brightness unit (Jy, mJy or uJy per beam) from the file's header, so au read as
    arcsec or mJy read as Jy cannot happen silently."""
    f = tmp_path / "IMLup.profile.txt"
    f.write_text("# " + header + "\n" + "\n".join(" ".join(str(x) for x in row) for row in rows) + "\n")
    r, i, di = P.load_profile(f)
    assert r[0] == pytest.approx(r_expect, rel=5e-3) and i[0] == pytest.approx(i_expect, rel=1e-9)


@GUARD
def test_dsharp_profile_without_named_units_is_refused(tmp_path):
    f = tmp_path / "bad.txt"
    f.write_text("# radius brightness\n0.5 2.0\n")
    with pytest.raises(ValueError, match="does not name"):
        P.load_profile(f)
    g = tmp_path / "none.txt"
    g.write_text("0.5 2.0\n")
    with pytest.raises(ValueError, match="no '#' header"):
        P.load_profile(g)
