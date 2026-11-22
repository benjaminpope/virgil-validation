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
needs_multipole = pytest.mark.skipif(P is None, reason="needs virgil main (MultipoleField) and virgil-vlti with virgil_vlti.with_night_terms (install the postagb extra)")

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
    for bad in (dict(gp_converged=False), dict(gp_on_edge=True), dict(chi2_per_point=5.0), dict(chi2_per_point=0.1),
                dict(f_sec_pct=30.0), dict(smoke=True), dict(star_index_source="FIXED")):
        (tmp_path / "t_image.json").write_text(json.dumps(_previous_result() | bad))
        assert P.skip_clean_reason(tmp_path, cfg, "gp-aniso", _reuse_args()) is None, bad
    (tmp_path / "t_image.json").write_text(json.dumps({k: v for k, v in _previous_result().items() if k != "gp_on_edge"}))
    assert P.skip_clean_reason(tmp_path, cfg, "gp-aniso", _reuse_args()) is None  # an unknown edge status reruns
    smoke = _reuse_args()
    smoke.smoke = True
    (tmp_path / "t_image.json").write_text(json.dumps(_previous_result()))
    assert P.skip_clean_reason(tmp_path, cfg, "gp-aniso", smoke) is None
    # the disc+res strategy also needs its render guard and the 192-ring grid
    assert P.skip_clean_reason(tmp_path, cfg, "disc+res", _reuse_args()) is None
    full = _previous_result(render_ok=True, d_env_shared=1.0, n_r=192)
    (tmp_path / "t_image.json").write_text(json.dumps(full))
    assert P.skip_clean_reason(tmp_path, cfg, "disc+res", _reuse_args()).endswith("render guard passed")
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


# ---- review options: --companion, --hole-scale, --geometry, --white (reports/postagb-binaries-psgem-review.md)


def _opts(**over):
    import argparse

    base = dict(star_index=None, bg_index=None, noise="night", mean="blob", strategy="gp-polar", target="IW_Car", companion=None,
                companion_start=None, hole_scale=None, geometry=None, white=None)
    return argparse.Namespace(**(base | over))


@needs_multipole
def test_pair_arg_parses_two_numbers_and_refuses_the_rest():
    import argparse

    assert P.pair_arg("0.59,0.84") == (0.59, 0.84) and P.pair_arg("28, 161") == (28.0, 161.0)
    for bad in ("1", "1,2,3", "a,b", "1,nan", ""):
        with pytest.raises(argparse.ArgumentTypeError):
            P.pair_arg(bad)


@needs_multipole
def test_default_options_leave_the_output_names_unchanged():
    assert P.option_suffix(_opts()) == ""
    seen = []
    P.per_strategy(lambda args, out, strat: seen.append(P.TARGETS[args.target]["key"]), _opts(strategy="gp-polar", noise="night"), None)
    assert seen == ["IW_Car_gp-polar_night"]
    seen.clear()
    P.per_strategy(lambda args, out, strat: seen.append(P.TARGETS[args.target]["key"]),
                   _opts(companion="free", companion_start=(1.0, -1.3), hole_scale=0.5, geometry=(28.0, 161.0), white=(0.59, 0.84)), None)
    assert seen == ["IW_Car_gp-polar_night_compfree_holex0.5_geo28-161_white0.59-0.84"]
    assert P.TARGETS["IW_Car"]["key"] == "IW_Car"  # the table is restored


@needs_multipole
def test_options_that_would_do_nothing_are_refused():
    P.check_options(_opts())
    P.check_options(_opts(companion="fixed", companion_start=(0.7, -1.5), geometry=(0.0, 0.0), white=(0.59, 0.84), hole_scale=0.5))
    for bad in (dict(companion="fixed"), dict(companion_start=(1.0, 1.0)),  # the two go together
                dict(white=(0.59, 0.84), noise="current"), dict(white=(0.59, 0.84), noise="nightw"), dict(white=(0.0, 0.84)),
                dict(geometry=(95.0, 0.0)), dict(hole_scale=0.0), dict(hole_scale=-1.0),
                dict(geometry=(28.0, 161.0), strategy="mem"), dict(companion="free", companion_start=(1.0, 1.0), strategy="all")):
        with pytest.raises(SystemExit):
            P.check_options(_opts(**bad))


@needs_multipole
def test_white_replaces_only_the_two_white_scales(monkeypatch):
    cfg = {"key": "PS_Gem"}
    pair = P.noise_widths(cfg)
    assert P.effective_noise_widths(cfg) == pair  # no option: the pair test's widths, untouched
    got = P.effective_noise_widths(cfg, (0.59, 0.84))
    assert got["v2_white"] == 0.59 and got["t3_white"] == 0.84
    assert {k: v for k, v in got.items() if k not in ("v2_white", "t3_white")} == {k: v for k, v in pair.items() if k not in ("v2_white", "t3_white")}
    assert pair["v2_white"] != 0.59  # PS Gem's pair test says 0.31
    seen = {}
    monkeypatch.setattr(P, "night_noise", lambda data, widths, night: seen.update(night=widths) or "night")
    monkeypatch.setattr(P, "vlti_noise", lambda data, widths: seen.update(vlti=widths) or "vlti")
    assert P.modelled_noise("d", cfg, "night", (0.59, 0.84)) == "night" and seen["night"]["v2_white"] == 0.59
    assert P.modelled_noise("d", cfg, "vlti", (0.59, 0.84)) == "vlti" and seen["vlti"]["t3_white"] == 0.84
    P.modelled_noise("d", cfg, "night")
    assert seen["night"] == pair  # the default path


@needs_multipole
def test_geometry_option_replaces_the_scale_stage_and_needs_no_file(tmp_path):
    import json

    cfg = {"key": "PS_Gem"}
    args = _opts(geometry=(28.0, 161.0), smoke=False)
    assert P.resolve_geometry(args, tmp_path, cfg, "gp-polar") == dict(inc_deg=28.0, pa_deg=161.0)  # no PS_Gem_scale_night.json here
    assert P.resolve_geometry(_opts(geometry=(0.0, 0.0), smoke=True), tmp_path, cfg, "gp-polar") == dict(inc_deg=0.0, pa_deg=0.0)
    with pytest.raises(FileNotFoundError):  # without the option the scale stage's file is still required
        P.resolve_geometry(_opts(smoke=False), tmp_path, cfg, "gp-polar")
    (tmp_path / "PS_Gem_scale_night.json").write_text(json.dumps(dict(geometry=dict(inc_deg=71.8, pa_deg=127.6))))
    assert P.resolve_geometry(_opts(smoke=False), tmp_path, cfg, "gp-polar") == dict(inc_deg=71.8, pa_deg=127.6)
    assert P.resolve_geometry(_opts(smoke=True), tmp_path, cfg, "gp-polar") == dict(inc_deg=45.0, pa_deg=90.0)  # the smoke placeholder


@needs_multipole
def test_hole_scale_scales_the_hole_and_the_default_is_unchanged():
    from virgil.models import Image

    env = Image(np.zeros((101, 101)), 0.1)

    class Data:
        u, v, wavel = np.array([30.0]), np.array([0.0]), np.array([1.6e-6])

    cfg = dict(sec0=None, ud0=0.01, ud_fix=0.01, hole_mas=1.2, mask_mas=1e9)
    m1, r1 = P.hole_mask(cfg, env, Data)
    m_half, r_half = P.hole_mask(cfg | dict(hole_scale=0.5), env, Data)
    assert r1 == 1.2 and r_half == pytest.approx(0.6)
    assert (~m_half).sum() < (~m1).sum() and not (m1 & ~m_half).any()  # the half hole blanks a subset of the full one
    assert P.cfg_of(_opts(target="PS_Gem", hole_scale=0.5))["hole_scale"] == 0.5
    assert "hole_scale" not in P.cfg_of(_opts(target="PS_Gem"))


@needs_multipole
def test_companion_priors_fix_or_free_the_position():
    fixed = P.companion_priors(_opts(companion="fixed", companion_start=(0.7, -1.5)))
    free = P.companion_priors(_opts(companion="free", companion_start=(0.7, -1.5)))
    assert set(fixed) == {"comp.flux.ratio", "comp.flux.index"}  # the position has no prior: it is held
    assert set(free) == set(fixed) | {"comp.dra", "comp.ddec"}
    assert float(free["comp.dra"].low) == -3.0 and float(free["comp.dra"].high) == 3.0
    assert (float(fixed["comp.flux.ratio"].low), float(fixed["comp.flux.ratio"].high)) == (1e-3, 0.3)
    comp = P.companion_component(_opts(companion="fixed", companion_start=(0.7, -1.5)))
    assert (float(comp.dra), float(comp.ddec)) == (0.7, -1.5)


@needs_multipole
def test_companion_summary_reports_position_flux_and_the_flux_left_in_the_image():
    from types import SimpleNamespace

    from virgil.models import PointSource
    from virgil.spectra import PowerLaw

    n, pix = 101, 0.1
    model = SimpleNamespace(comp=PointSource(flux=PowerLaw(0.02, -3.0, P.L0), dra=1.0, ddec=-1.0),
                            env=SimpleNamespace(flux=SimpleNamespace(index=2.0)))
    bright = np.full((n, n), 1e-9)
    iy, ix = 50 + 10, 50 - 10  # 1 mas East (left of the centre column) and 1 mas South (below the centre row)
    bright[iy, ix] = 1.0
    out = P.companion_summary(model, bright, pix, f_env=0.04, sd={"comp.flux.index": 0.2, "comp.dra": 0.1, "comp.ddec": 0.2}, tot=1.1)
    assert out["comp_dra"] == 1.0 and out["comp_ddec"] == -1.0
    assert out["comp_sep_mas"] == pytest.approx(np.sqrt(2.0)) and out["comp_pa_deg"] == pytest.approx(135.0)  # South-East
    assert out["comp_flux_pct"] == pytest.approx(100 * 0.02 / 1.1) and out["comp_pos_sd_mas"] == pytest.approx(np.hypot(0.1, 0.2))
    assert out["comp_colour"] == "photospheric"
    assert out["comp_image_left_frac"] == pytest.approx(2.0, rel=1e-3)  # all 0.04 of the environment's flux sits on the point: twice its own 0.02
    bright[iy, ix] = 1e-9
    bright[50 - 10, 50 + 10] = 1.0  # the image's flux moves to the mirror position: none left near the point
    assert P.companion_summary(model, bright, pix, 0.04, {}, 1.1)["comp_image_left_frac"] == pytest.approx(0.0, abs=1e-6)
    assert P.companion_summary(model, bright, pix, 0.04, {}, 1.1)["comp_pos_sd_mas"] is None  # a held position has no sd


@needs_multipole
@pytest.mark.parametrize("lz_mirror, kept", [(-5.0, "mirror"), (-20.0, "start")])
def test_companion_mirror_keeps_the_higher_scene_evidence(monkeypatch, lz_mirror, kept):
    from types import SimpleNamespace

    class Sc:
        def __init__(self, d=None):
            self.d = d or {}

        def set(self, k, v):
            return Sc(self.d | {k: v})

    best = SimpleNamespace(lz=-10.0, model=SimpleNamespace(comp=SimpleNamespace(dra=0.7, ddec=-1.5)))
    fits = []

    def fake_fit(sc, pri, data, opts):
        f = SimpleNamespace(lz=lz_mirror, info={"chi2_red": 1.0}, model=SimpleNamespace(comp=SimpleNamespace(dra=sc.d["comp.dra"], ddec=sc.d["comp.ddec"])))
        fits.append(f)
        return f, {}

    monkeypatch.setattr(P, "polar_scene_fn", lambda *a: (lambda sg, lr, lt: Sc()))
    monkeypatch.setattr(P, "image_priors", lambda sc: {})
    monkeypatch.setattr(P, "fit_cell", fake_fit)
    monkeypatch.setattr(P, "scene_evidence", lambda f, data, pri: (f.lz, {}))
    args = _opts(companion="free", companion_start=(0.7, -1.5), smoke=True)
    chosen, rec = P.companion_mirror({}, None, None, None, {"env.log_brightness": 0, "comp.dra": 1}, None, args, {}, 0.3, 3, (1.0, 0.5, 1.0), best)
    assert (fits[0].model.comp.dra, fits[0].model.comp.ddec) == (-0.7, 1.5)  # started at the mirror image through the primary
    assert chosen is (fits[0] if kept == "mirror" else best) and rec["comp_mirror_better"] is (kept == "mirror")
    assert rec["comp_mirror_log_z_scene"] == lz_mirror and rec["comp_start_log_z_scene"] == -10.0
