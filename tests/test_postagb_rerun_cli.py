"""scripts/postagb_imaging.py: the options of the post-AGB rerun (WP-02 to WP-08): ``--inject``, counts, ``--fstar``, ``--fstar-night``,
``--binary none``, the display blob and the ``compare`` metrics, ``sec_physical``, and the option checks."""

import pathlib
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
try:
    import postagb_imaging as P  # noqa: E402
    from test_postagb_simulate import toy4  # noqa: E402
except ImportError:
    P = None
pytestmark = [
    pytest.mark.skipif(P is None, reason="needs virgil main (MultipoleField) and virgil-vlti (install the postagb extra)"),
    pytest.mark.validates("evidence", roots=["mathematics"], kind="guard"),
]

L0 = 1.65e-6


def ns(**kw):
    base = dict(target="AI_Sco", stage="image", strategy="gp-polar", noise="current", binary="paper", white=None, geometry=None,
                hole_scale=None, companion=None, companion_start=None, companion_box=None, min_length_r=None, m_max=None,
                n_theta=P.POLAR_N_THETA if P else 32, report_corner=False, polar_rmax=None, fstar="free", fstar_night=False,
                binary_v_sigma=None, inject=None, inject_noise=None, inject_fstar_sine=None, seed=0, drop_nights=None)
    return types.SimpleNamespace(**(base | kw))


# --- counts -------------------------------------------------------------------------------------------------------------------------

def test_chi2r_divides_by_the_independent_observables_and_the_json_carries_all_three_counts():
    from virgil.imaging import whitened_residuals

    d = toy4()
    c2, n_res = P.chi2r(P.PointSource(), d)
    r = np.asarray(whitened_residuals(P.PointSource(), d))
    assert n_res == r.size and c2 == pytest.approx(np.sum(r**2) / d.n_independent)
    counts = P.data_counts(d)
    assert set(counts) == {"n_data", "n_independent", "n_residuals"}
    assert counts["n_independent"] < counts["n_residuals"]  # four telescopes: the closure phases are redundant rows


# --- evidence parts -----------------------------------------------------------------------------------------------------------------

def test_evidence_parts_add_up_to_virgil_log_evidence():
    import jax
    from virgil.coverage import ami_grid_record
    from virgil.fields import GaussianField
    from virgil.fitting import fit
    from virgil.oidata import OIData

    scene = P.System(star=P.PointSource(dra=0.5), env=P.Image(GaussianField(np.zeros((8, 8)), 1.5, 24.0), 12.0, flux=0.3))
    data = OIData(ami_grid_record(pitch_m=0.5)).with_model(scene, key=jax.random.PRNGKey(9))
    pri = {k: v for k, v in P.image_priors(scene).items()}
    f = fit(scene, pri, data, max_steps=5)
    p = P.evidence_parts(f, data)
    assert p["log_z"] == pytest.approx(float(P.log_evidence(f, data)), abs=1e-6)
    assert p["log_z"] == pytest.approx(-0.5 * p["chi2_sum"] - p["log_norm"] - 0.5 * p["z2"] - 0.5 * p["logdet"], abs=1e-9)
    assert 0.0 <= p["n_eff"] <= 64


# --- f* fixed -----------------------------------------------------------------------------------------------------------------------

def sparco(f_sec=0.05, bg=True):
    parts = dict(star=P.UniformDisk(diam=0.4, flux=P.PowerLaw(1.0, -4.0, L0)), env=P.Resolved(flux=P.PowerLaw(0.5, 0.9, L0)))
    if bg:
        parts["bg"] = P.Resolved(flux=P.PowerLaw(0.1, 0.0, L0))
    parts["sec"] = P.PointSource(flux=P.PowerLaw(f_sec, 0.0, L0), dra=-1.0, ddec=-1.4)
    return P.System(**parts)


def test_a_fixed_fstar_scene_holds_the_star_fraction_whatever_the_other_parameters_do():
    import numpyro.distributions as dist

    pri = {"env.flux.ratio": dist.Uniform(0.01, 5.0), "bg.flux.ratio": dist.Uniform(0.0, 2.0), "sec.flux.ratio": dist.LogUniform(1e-3, 1.0)}
    sc, pri2 = P.fix_fstar(sparco(), pri, 0.6)
    assert set(pri2) == {"sec.flux.ratio", "share"}  # one column fewer: env and bg ratios are replaced by their share
    assert float(pri2["sec.flux.ratio"].high) == pytest.approx(0.95 * 0.4 / 0.6)  # the secondary's prior stops short of what f* leaves
    assert P.fstar_of(sc) == pytest.approx(0.6, abs=1e-6)
    for f_sec, share in ((0.02, 0.2), (0.3, 0.9)):
        s2 = sc.set("sec.flux.ratio", f_sec).set("share", share)
        assert P.fstar_of(s2) == pytest.approx(0.6, abs=1e-6)
        r = s2.resolved()
        assert float(np.asarray(r.env.flux.ratio)) + float(np.asarray(r.bg.flux.ratio)) + f_sec == pytest.approx((1 - 0.6) / 0.6, abs=1e-6)
    plain = sc.plain()
    assert type(plain) is P.System and P.fstar_of(plain) == pytest.approx(0.6, abs=1e-6)
    # a secondary that takes all the room leaves a floor, and is reported
    assert float(np.asarray(sc.set("sec.flux.ratio", 0.9).remainder())) < 0


def test_fstar_target_reads_paper_free_or_a_number(monkeypatch):
    monkeypatch.setattr(P, "zenodo_header", lambda cfg: ({"SFLU1": 0.413}, None, None))
    assert P.fstar_target(ns(fstar="free"), {}) is None
    assert P.fstar_target(ns(fstar="paper"), {}) == pytest.approx(0.413)
    assert P.fstar_target(ns(fstar="0.55"), {}) == pytest.approx(0.55)


# --- f* per night -------------------------------------------------------------------------------------------------------------------

def test_a_night_offset_moves_the_star_fraction_of_that_night_by_that_many_points():
    d = toy4(n_frames=6, n_ch=1)
    group = np.repeat([0, 1, 2], d.vis.size // 3)
    f_ref = 0.4
    star = P.PointSource(flux=P.PowerLaw(1.0, 0.0, L0))
    pt = P.PointSource(flux=P.PowerLaw(1.5, 0.0, L0), dra=2.0, ddec=0.5)
    off = np.array([0.03, -0.03, 0.0])
    nf = P.NightFlux(star, group, f_ref, flux=star.flux, d=off)
    sc = P.System(star=nf, pt=pt)
    u, v, wl = (np.asarray(x) for x in (d.u, d.v, d.wavel))
    got = np.asarray(sc.model(u, v, wl))
    w = np.exp(off[group] / (f_ref * (1 - f_ref)))
    for k in (0, 1, 2):  # the same visibility as a static star of that night's weight
        i = np.flatnonzero(group == k)[:5]
        ref = P.System(star=P.PointSource(flux=P.PowerLaw(float(w[i[0]]), 0.0, L0)), pt=pt).model(u[i], v[i], wl[i])
        np.testing.assert_allclose(got[i], np.asarray(ref), atol=1e-6)
    zero = np.asarray(P.System(star=P.NightFlux(star, group, f_ref, flux=star.flux), pt=pt).model(u, v, wl))
    np.testing.assert_allclose(zero, np.asarray(P.System(star=star, pt=pt).model(u, v, wl)), atol=1e-6)


def test_the_night_priors_come_from_the_plan(monkeypatch):
    monkeypatch.setattr(P, "night_labels", lambda d: dict(u_night=np.array([0, 0, 1, 2, 2])))
    sc, pri = P.fstar_night_scene(sparco(), {}, None, 0.4, "AI_Sco")
    p = pri["star.d"]
    assert isinstance(sc.star, P.NightFlux) and np.shape(sc.star.d) == (3,)
    one = lambda x: float(np.ravel(np.asarray(x))[0])  # noqa: E731
    assert one(p.high) == pytest.approx(0.06) and one(p.base_dist.scale) == pytest.approx(0.03)
    _, pri = P.fstar_night_scene(sparco(), {}, None, 0.4, "IW_Car")
    assert one(pri["star.d"].high) == pytest.approx(0.03) and one(pri["star.d"].base_dist.scale) == pytest.approx(0.013)


# --- the options and their names ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expect",
    [("organic", ("organic", ())), ("organic+point:1.5,-2,0.03", ("orgpt", (1.5, -2.0, 0.03))), ("npz:/x/a_image.npz", ("npz", "/x/a_image.npz"))],
)
def test_parse_inject(text, expect):
    assert P.parse_inject(text) == expect


@pytest.mark.parametrize("text", ["composite:x", "organic+point:1,2", "organic+point:1,2,-1", "npz:"])
def test_parse_inject_refuses_what_it_does_not_build(text):
    with pytest.raises(SystemExit):
        P.parse_inject(text)


def test_option_names_are_empty_by_default_and_tag_every_new_option():
    assert P.option_suffix(ns()) == ""
    sfx = P.option_suffix(ns(inject="organic+point:1,2,0.03", seed=3, inject_fstar_sine=(2.0, 30.0), m_max=8, n_theta=64, fstar="0.6",
                             polar_rmax="support", drop_nights=(1, 4), binary_v_sigma=0.5, fstar_night=True, report_corner=True))
    assert sfx == "_inj-orgpt-1_2_0.03_nzwhite_s3_fsine2-30_m8_nt64_fs0.6_rsup_drop1-4_vs0.5_fn_corner"
    assert "_nznight_s0" in P.option_suffix(ns(inject="organic", noise="night"))
    assert P.option_suffix(ns(m_max=-1)).endswith("_mall")


@pytest.mark.parametrize(
    "kw",
    [dict(fstar="1.5"), dict(fstar="loose"), dict(fstar_night=True, target="HR4049"), dict(fstar_night=True, binary="motion"),
     dict(binary_v_sigma=0.1), dict(binary="motion", binary_v_sigma=-1.0), dict(m_max=16, n_theta=32), dict(m_max=3, strategy="mem"),
     dict(inject="organic", noise="vlti"), dict(inject="organic", noise="night", inject_noise="white"), dict(inject_fstar_sine=(1.0, 30.0)),
     dict(inject_noise="white")],
)
def test_options_that_would_do_nothing_or_the_wrong_thing_are_refused(kw):
    with pytest.raises(SystemExit):
        P.check_options(ns(**kw))


def test_sensible_combinations_pass():
    P.check_options(ns(fstar="paper", m_max=8, n_theta=64, report_corner=True, polar_rmax="support"))
    P.check_options(ns(fstar_night=True, target="IW_Car", inject="organic", noise="night", inject_noise="night"))
    P.check_options(ns(binary="none", drop_nights=(0,)))


def test_noise_current_refuses_without_the_scale_file(tmp_path):
    cfg = dict(key="no_such", data_key="no_such")
    with pytest.raises(SystemExit, match="needs the scale stage"):
        P.load_scaled(cfg, ns(noise="current", smoke=False, files=None), tmp_path)


def test_resolve_white_takes_a_pair_or_the_stage_file(tmp_path):
    a = ns(white="0.5,0.6")
    P.resolve_white(a)
    assert a.white == (0.5, 0.6) and a.white_json is None
    f = tmp_path / "T_white.json"
    f.write_text('{"v2_white": 0.4, "t3_white": 0.7, "night_widths_used": {"night_gain_baseline": 0.01, "night_t3_offset_deg": 0.5}}')
    b = ns(white=str(f))
    P.resolve_white(b)
    assert b.white == (0.4, 0.7) and b.white_json["night_widths_used"]["night_t3_offset_deg"] == 0.5
    with pytest.raises(SystemExit):
        P.resolve_white(ns(white="neither"))


def test_binary_none_builds_the_scene_without_a_secondary():
    cfg = P.cfg_of(ns(binary="none", target="AI_Sco"))
    assert cfg["secondary"] is False
    assert P.cfg_of(ns(target="AI_Sco")).get("secondary", True) is not False


# --- compare: display blob and metrics ----------------------------------------------------------------------------------------------

def test_the_display_blob_has_unit_flux_and_a_lone_star_is_a_gaussian_of_the_stated_fwhm():
    n, scale, fwhm = 129, 0.1, 1.2
    env = np.random.default_rng(0).random((n, n))
    img = P.display_blob(env, 0.3, [(0.6, 0.0, 0.0), (0.1, -1.0, 1.4)], scale, fwhm)
    assert img.sum() == pytest.approx(1.0, abs=1e-12)
    star = P.display_blob(env, 0.0, [(1.0, 0.0, 0.0)], scale, fwhm)
    c = n // 2
    row = star[c] / star[c, c]
    x = (np.arange(n) - c) * scale
    half_width = np.interp(0.5, row[c : c + 30][::-1], x[c : c + 30][::-1])  # where the profile falls to half its peak
    assert 2 * half_width == pytest.approx(fwhm, abs=0.01)  # full width at half maximum
    off = P.display_blob(env, 0.0, [(1.0, -1.0, 1.4)], scale, fwhm)  # East is left, North is up
    iy, ix = np.unravel_index(off.argmax(), off.shape)
    assert (c - ix) * scale == pytest.approx(-1.0, abs=scale) and (c - iy) * scale == pytest.approx(1.4, abs=scale)


def test_inner_peak_ratio_on_an_image_with_known_peaks():
    n, scale = 101, 0.1
    img = np.zeros((n, n))
    c = n // 2
    img[c, c + 10] = 4.0  # 1.0 mas out: inside 2.2
    img[c + 30, c] = 2.0  # 3.0 mas out: in the 2.5-3.5 ring
    img[c, c + 24] = 9.0  # 2.4 mas: in neither window
    assert P.inner_peak_ratio(img, scale) == pytest.approx(2.0)
    assert P.flux_within(img, scale, 2.0) == pytest.approx(4.0 / 15.0)


def test_sec_physical_is_false_for_the_ai_sco_source_and_true_for_the_orbital_one():
    ok, m = P.sec_physical("AI_Sco", 1.75)
    assert ok is False and 16.0 < m < 19.0  # lit-binaries.md: about 18 Msun on the 977 d period
    ok, m = P.sec_physical("AI_Sco", 0.8)
    assert ok is True and m < P.SEC_MASS_CEILING
    assert P.sec_physical("IW_Car", 1.0) == (None, None) and P.sec_physical("AI_Sco", None) == (None, None)


def test_compare_extras_scores_an_image_against_itself_perfectly():
    from virgil.imaging import Beam

    n, scale = 64, 0.2
    rng = np.random.default_rng(1)
    img = rng.random((n, n)) ** 3
    img /= img.sum()
    bm = Beam(1.0, 1.0, 0.0)
    conv = np.asarray(P.convolve_beam(img, scale, bm))
    yy, xx = np.mgrid[:n, :n]
    mask = np.hypot(xx - (n - 1) / 2, yy - (n - 1) / 2) * scale < 5.0
    ij = dict(f_env_pct=70.0, f_prim_pct=25.0, f_sec_pct=5.0, sec_dra=-1.0, sec_ddec=-1.4)
    hdr = dict(SFLU1=0.25, SFLU2=0.05, SDEX2=-1.0, SDEY2=-1.4)
    m = P.compare_extras(img, img, conv, conv, scale, bm, ij, hdr, {}, mask)
    assert m["ncc_native"] == pytest.approx(1.0) and m["ncc_beam_total"] == pytest.approx(1.0, abs=1e-6)
    assert m["flux_in_2mas"]["ours"] == m["flux_in_2mas"]["reference"] and m["flux_outside_reference_support"] < 0.05


def test_night_v2_chi2_uses_the_unflagged_rows_when_samples_are_flagged(monkeypatch):
    # four V^2 samples, the second flagged: the whitened V^2 block holds samples 0, 2, 3 and the closure rows follow it
    data = types.SimpleNamespace(vis_index=np.array([0, 2, 3]), vis=np.zeros(3))
    monkeypatch.setattr(P, "night_labels", lambda d: {"u_night": np.array([0, 0, 1, 1])})
    monkeypatch.setattr(P, "whitened_residuals", lambda m, d: np.array([1.0, 2.0, 3.0, 100.0, 100.0]))
    assert P.v2_chi2_by_night(None, data) == [1.0, 4.0 + 9.0]  # sample 0 in night 0; samples 2 and 3 in night 1; no closure row
    data.vis_index = None
    data.vis = np.zeros(4)
    monkeypatch.setattr(P, "whitened_residuals", lambda m, d: np.array([1.0, 2.0, 3.0, 4.0, 100.0]))
    assert P.v2_chi2_by_night(None, data) == [5.0, 25.0]  # nothing flagged: as before


def test_white_file_night_widths_are_recorded_and_tagged(tmp_path):
    wj = dict(night_widths_used={k: 0.123 for k in P.NIGHT_KEYS})
    plain = ns(noise="night", white=(1.1, 1.2))
    filed = ns(noise="night", white=(1.1, 1.2), white_json=wj)
    assert P.white_file_night(plain) is None and P.white_file_night(filed) == {k: 0.123 for k in P.NIGHT_KEYS}
    assert P.option_suffix(plain) != P.option_suffix(filed) and P.option_suffix(filed).endswith("_wfile")


def test_field_mas_sets_the_support_and_is_tagged(monkeypatch):
    monkeypatch.setattr(P, "setup", lambda cfg: cfg.setdefault("fov_mas", 0.0))  # the geometry needs the data; only the field keys matter here
    base = dict(P.TARGETS["HR4049"])
    try:
        cfg = P.cfg_of(ns(target="HR4049", field_mas=25.0))
        assert cfg["mask_mas"] == 25.0 and cfg["half_mas"] == 25.0 and cfg["fov_override"] == pytest.approx(50.0 / 0.95)
        assert cfg["cmp_mask_mas"] == 25.0 and cfg["half_cmp_mas"] == 25.0
        small = P.cfg_of(ns(target="HR4049", field_mas=10.0))
        assert small["cmp_mask_mas"] == 10.0 and small["half_cmp_mas"] == 10.0  # capped at the field
    finally:
        P.TARGETS["HR4049"] = base
    assert P.option_suffix(ns(field_mas=25.0)) == "_fld25" and P.option_suffix(ns()) == ""
