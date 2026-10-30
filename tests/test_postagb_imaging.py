"""scripts/postagb_imaging.py: the position-angle convention of region_metrics (North up, East left)."""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
try:  # needs virgil main (MultipoleField); the CI pin of the orbit pages predates it, so these tests skip there (extra: postagb)
    import postagb_imaging as P  # noqa: E402
except ImportError:
    P = None
needs_multipole = pytest.mark.skipif(P is None, reason="virgil without MultipoleField (install the postagb extra)")

# Guards our own script's convention, not virgil against an independent result.
pytestmark = pytest.mark.validates("evidence", roots=["standards"], kind="guard")


@needs_multipole
@pytest.mark.parametrize("pa", [20.0, 80.0, 130.0, 170.0])
def test_major_axis_pa_is_east_of_north(pa):
    n, scale = 129, 0.5
    yy, xx = np.mgrid[:n, :n]
    east = -(xx - (n - 1) / 2) * scale  # East is left (image columns decrease towards East)
    north = -(yy - (n - 1) / 2) * scale  # row 0 is North
    t = np.radians(pa)
    along = east * np.sin(t) + north * np.cos(t)
    across = east * np.cos(t) - north * np.sin(t)
    img = np.exp(-0.5 * ((along / 6.0) ** 2 + (across / 2.0) ** 2))
    m = P.region_metrics(img, scale, 0.0, 30.0)
    assert m["major_pa_deg"] == pytest.approx(pa, abs=0.5)


@needs_multipole
def test_centroid_pa_points_to_the_offset_blob():
    n, scale = 129, 0.5
    yy, xx = np.mgrid[:n, :n]
    img = np.exp(-0.5 * (((xx - 64 + 8) ** 2 + (yy - 64 + 8) ** 2) / 4.0))  # up-left on the array: North-East
    m = P.region_metrics(img, scale, 0.0, 30.0)  # up = North, left = East: PA 45 deg
    assert m["centroid_pa_deg"] == pytest.approx(45.0, abs=1.0)


@needs_multipole
def test_bitmap_to_image_recovers_beam_pa_and_colourbar():
    """A synthetic panel (grey colour bar, blue beam ellipse at PA 30 deg East of North) round-trips."""
    import io

    import postagb_literature as L
    from PIL import Image

    x0, x1, y0, y1, cbar = 171, 650, 61, 540, 672
    im = np.full((600, 800, 3), 255, np.uint8)
    ramp = np.linspace(0, 255, y1 - y0 - 2).astype(np.uint8)  # black (value 1) at the top to white (0) at the bottom
    im[y0 + 1:y1 - 1, cbar, :] = ramp[:, None]
    im[y0:y1, x0:x1, :] = 128  # value ~0.5 everywhere
    pa, a, b = np.radians(30.0), 18.0, 9.0  # semi-axes in pixels; the box for the beam is 45 x 60
    yy, xx = np.mgrid[:45, :60]
    dx, dy = xx - 30.0, yy - 22.0
    u = dx * (-np.sin(pa)) + dy * (-np.cos(pa))  # along the major axis (right = West, down = South)
    v = dx * np.cos(pa) + dy * (-np.sin(pa))
    inside = (u / a) ** 2 + (v / b) ** 2 <= 1
    box = im[y0:y0 + 45, x0:x0 + 60]
    box[inside] = (0, 0, 255)
    buf = io.BytesIO()
    Image.fromarray(im).save(buf, "PNG")
    img, scale, beam, _, _ = L.bitmap_to_image(buf.getvalue(), axes=(x0, x1, y0, y1), cbar_x=cbar)
    assert beam["pa_deg"] == pytest.approx(30.0, abs=2.0)
    assert beam["major_mas"] / beam["minor_mas"] == pytest.approx(a / b, rel=0.1)
    assert beam["major_mas"] == pytest.approx(2 * a * scale, rel=0.1)
    assert np.nanmedian(img) == pytest.approx(0.5, abs=0.03)  # the colour-bar inversion (top = 1)


@needs_multipole
@pytest.mark.parametrize("rot", [0.0, 35.0, 120.0])
def test_hole_mask_window_sits_on_the_secondary_as_virgil_renders_it(rot):
    """The pixel the secondary's window is centred on, lit alone in virgil's Image, has the visibility of PointSource(dra, ddec)."""
    from virgil.models import Image, PointSource

    n, scale, sec = 201, 0.1, (4.0, -3.0)
    env = Image(np.zeros((n, n)), scale)

    class Data:  # only what compare_beam needs
        u, v, wavel = np.array([30.0]), np.array([0.0]), np.array([1.6e-6])

    cfg = dict(sec0=sec, secondary=True, ud0=0.01, hole_mas=0.12, mask_mas=1e9)
    mask, _ = P.hole_mask(cfg, env, Data, rot)
    yy, xx = np.nonzero(~mask)
    far = np.hypot(yy - (n - 1) / 2, xx - (n - 1) / 2) > 20  # the secondary's window, not the star's
    iy, ix = int(round(yy[far].mean())), int(round(xx[far].mean()))
    lit = np.full((n, n), 1e-12)
    lit[iy, ix] = 1.0
    rng = np.random.default_rng(0)
    u, v, wl = rng.uniform(-80, 80, 40), rng.uniform(-80, 80, 40), np.full(40, 1.65e-6)
    ref = np.asarray(Image(np.log(lit), scale, rotation_deg=rot).model(u, v, wl))
    pt = np.asarray(PointSource(dra=sec[0], ddec=sec[1]).model(u, v, wl))
    assert np.max(np.abs(np.angle(ref * np.conj(pt)))) < 0.5  # pixel rounding (0.05 mas) is ~0.25 rad at 80 m


@needs_multipole
def test_grid_search_extends_an_edge_pick_once_and_reports_a_surviving_edge():
    calls = []

    def run(vals):  # log Z rises with the value, so the pick is always the upper edge
        calls.append(vals)
        return None, float(vals[0]), 1.0

    cells, axes, best, on_edge, extended = P.grid_search([[1.0, 2.0, 4.0]], run, [(None, lambda v: 2.0 * v)])
    assert axes[0] == [1.0, 2.0, 4.0, 8.0] and best == (8.0,)  # one extension, not chased
    assert on_edge and extended == [(0, "upper", 8.0)]
    assert len(calls) == len(set(calls)) == 4  # each cell is fitted once


@needs_multipole
def test_grid_search_interior_pick_does_not_extend_and_a_declining_extender_leaves_the_edge():
    cells, axes, best, on_edge, extended = P.grid_search([[1.0, 2.0, 4.0]], lambda v: (None, -abs(v[0] - 2.0), 1.0), [(lambda v: v / 2, lambda v: 2 * v)])
    assert best == (2.0,) and not on_edge and extended == []
    # a lower edge that cannot move (the extender returns None, as for a length at one pixel) stays an edge
    cells, axes, best, on_edge, extended = P.grid_search([[1.0, 2.0]], lambda v: (None, -v[0], 1.0), [(lambda v: None, None)])
    assert best == (1.0,) and on_edge and extended == []


@needs_multipole
def test_pixel_floor_lengths_start_at_one_pixel():
    lengths = P.pixel_floor_lengths(0.3, 1.2, (0.5, 1.0, 2.0))
    assert lengths[0] == pytest.approx(0.3) and lengths == sorted(lengths) and lengths[-1] == pytest.approx(2.4)


def test_the_postagb_page_matches_the_stored_results():
    """docs/imaging/postagb.md is regenerated from the compare / image JSON: no number has drifted."""
    import postagb_page

    page = pathlib.Path(__file__).resolve().parents[1] / "docs/imaging/postagb.md"
    text = page.read_text()
    assert postagb_page.build(text) == text
    assert text.count("<!-- gen:") == text.count("<!-- /gen:") > 0


def _previous_result(**over):
    return dict(chi2_per_point=1.1, gp_converged=True, gp_on_edge=False, star_index_source="ORGANIC SIND1",
                f_prim_pct=60.0, f_sec_pct=5.0, **over)


def _reuse_args(**over):
    import argparse

    return argparse.Namespace(smoke=False, reuse_converged=True, target="t", **over)


@needs_multipole
def test_reuse_converged_needs_the_flag_and_a_clean_previous_result(tmp_path):
    import json

    cfg = {"key": "t"}
    (tmp_path / "t_image.json").write_text(json.dumps(_previous_result()))
    assert P.skip_clean_reason(tmp_path, cfg, "gp-aniso", _reuse_args()) == "converged, interior of its grid"
    # controls: each way a previous result is not clean reruns it
    off = _reuse_args()
    off.reuse_converged = False
    assert P.skip_clean_reason(tmp_path, cfg, "gp-aniso", off) is None
    for bad in (dict(gp_converged=False), dict(gp_on_edge=True), dict(chi2_per_point=5.0), dict(f_sec_pct=30.0)):
        (tmp_path / "t_image.json").write_text(json.dumps(_previous_result() | bad))
        assert P.skip_clean_reason(tmp_path, cfg, "gp-aniso", _reuse_args()) is None, bad
    (tmp_path / "t_image.json").write_text(json.dumps(_previous_result()))
    assert P.skip_clean_reason(tmp_path, {"key": "t", "fov_override": 1}, "gp-aniso", _reuse_args()) is None


@needs_multipole
def test_a_reused_result_leaves_a_record_naming_the_kept_file_and_its_hash(tmp_path, monkeypatch):
    import hashlib
    import json

    prev = tmp_path / "t_image.json"
    prev.write_text(json.dumps(_previous_result()))
    monkeypatch.setattr(P, "cfg_of", lambda args: {"key": "t"})
    P.image_one(_reuse_args(), tmp_path, "gp-aniso")
    rec = json.loads((tmp_path / "t_gp-aniso_reused.json").read_text())
    assert rec["reuse_converged"] is True and rec["strategy"] == "gp-aniso"
    assert rec["kept_file"] == "t_image.json" and rec["kept_sha256"] == hashlib.sha256(prev.read_bytes()).hexdigest()
    assert rec["reason"] == "converged, interior of its grid" and rec["utc"]
    assert json.loads(prev.read_text()) == _previous_result()  # the kept result is untouched
