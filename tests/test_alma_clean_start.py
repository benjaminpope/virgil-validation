"""The CLEAN start of the IM Lup MaxEnt fit (``--workflow clean``, ``virgil_bridge.alma_start``): regridding, the start image and the
image-fixed (self-calibration) stage. Tiny synthetic data on the CPU, seconds."""

import json
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("virgil.radio")

import virgil_bridge.alma as va  # noqa: E402
import virgil_bridge.alma_start as vs  # noqa: E402
from crosscheck import alma_clean as ac  # noqa: E402
from crosscheck import alma_maxent as cx  # noqa: E402
from evidence.plugin import record  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import alma_imlup_clean as clean_script  # noqa: E402
import alma_imlup_maxent as script  # noqa: E402
from test_alma_maxent import synthetic_dir  # noqa: E402


def _call(*argv):
    script.main([str(a) for a in argv])


def _blob(n, px, centre, east_mas=0.0, north_mas=0.0, fwhm_mas=30.0):
    """Surface brightness (peak 1, row 0 North) of a Gaussian ``east_mas`` East and ``north_mas`` North of the phase-centre cell."""
    rr, cc = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    north, east = (centre[0] - rr) * px, (centre[1] - cc) * px
    return np.exp(-4 * np.log(2) * ((east - east_mas) ** 2 + (north - north_mas) ** 2) / fwhm_mas**2)


def _clean_dir(path, n=51, px=3.0, east_mas=20.0, north_mas=-10.0, flux=0.2, beam=(40.0, 25.0, 60.0)):
    """A directory as ``alma_imlup_clean.py run`` writes it: model.fits (Jy/pixel, FITS orientation) and clean.json."""
    path.mkdir(parents=True, exist_ok=True)
    img = _blob(n, px, vs.clean_centre(n), east_mas, north_mas)
    img *= flux / img.sum()
    clean_script.write_fits(path / "model.fits", img, px, "Jy/pixel")
    (path / "clean.json").write_text(json.dumps(dict(beam_mas_deg=list(beam))))
    return path, img


@pytest.mark.validates("virgil_bridge.alma_start", roots=["mathematics"])
def test_regrid_preserves_flux_places_the_source_and_keeps_east_and_north():
    """A Gaussian 20 mas East and 10 mas South of the phase centre, regridded from 3 mas (odd grid) to 5 mas (the fit's grid): the flux
    is the source's within the field to 1e-12, the centroid is where it was to 0.05 mas, East is still East (columns) and North North (rows)."""
    n_s, px_s, n_d, px_d = 101, 3.0, 64, 5.0
    sb = _blob(n_s, px_s, vs.clean_centre(n_s), 20.0, -10.0)
    out, info = vs.regrid(sb, px_s, px_d, n_d, vs.clean_centre(n_s))
    flux_in = float(sb.sum()) * (px_s / 1000) ** 2  # the whole blob is inside the 5 mas field (64 x 5 mas = 320 mas)
    r0, c0 = cx.phase_centre_cell(n_d)
    rr, cc = np.meshgrid(np.arange(n_d), np.arange(n_d), indexing="ij")
    w = out / out.sum()
    east, north = ((c0 - cc) * px_d * w).sum(), ((r0 - rr) * px_d * w).sum()
    record("flux_rel_error", abs(out.sum() * (px_d / 1000) ** 2 / flux_in - 1))
    record("centroid_error_mas", float(np.hypot(east - 20.0, north + 10.0)))
    assert abs(out.sum() * (px_d / 1000) ** 2 / flux_in - 1) < 1e-12
    assert abs(east - 20.0) < 0.05 and abs(north + 10.0) < 0.05
    assert info["scale"] == pytest.approx(1.0, abs=0.02)  # bilinear on a smooth image: the rescaling is a small correction, not a rescue


@pytest.mark.validates("virgil_bridge.alma_start", roots=["mathematics"])
def test_regrid_cut_out_keeps_the_flux_inside_the_field_and_upsampling_is_exact_on_a_plane():
    """The fit's field is smaller than the CLEAN's: the flux is that of the source pixels inside it (a source outside contributes none).
    Upsampling a plane from 10 mas to 5 mas is bilinear-exact (the coarse-to-fine step), to 1e-12 away from the edge."""
    n_s, px_s = 101, 3.0
    sb = _blob(n_s, px_s, vs.clean_centre(n_s), 0.0, 0.0, fwhm_mas=60.0) + _blob(n_s, px_s, vs.clean_centre(n_s), 120.0, 0.0, fwhm_mas=30.0)
    out, info = vs.regrid(sb, px_s, 5.0, 32, vs.clean_centre(n_s))  # 32 x 5 mas = 160 mas wide: the blob at 120 mas East is outside
    assert abs(out.sum() * 5e-3**2 - info["flux_in_field_jy"]) < 1e-12 * info["flux_in_field_jy"]
    assert info["flux_in_field_jy"] < 0.9 * float(sb.sum()) * (px_s / 1000) ** 2
    n_c, px_c, n_f, px_f = 21, 10.0, 41, 5.0
    c_c, c_f = (10.0, 10.0), (20.0, 20.0)
    rr, cc = np.meshgrid(np.arange(n_c), np.arange(n_c), indexing="ij")
    plane = 1.0 + 0.3 * ((c_c[0] - rr) * px_c) / 100 - 0.2 * ((c_c[1] - cc) * px_c) / 100  # linear in north and east
    up, _ = vs.regrid(plane, px_c, px_f, n_f, c_c, c_f)
    rf, cf = np.meshgrid(np.arange(n_f), np.arange(n_f), indexing="ij")
    exact = 1.0 + 0.3 * ((c_f[0] - rf) * px_f) / 100 - 0.2 * ((c_f[1] - cf) * px_f) / 100
    sl = slice(2, -2)
    # the flux rescaling is a constant factor close to 1 on a plane; compare after undoing it
    assert np.max(np.abs(up[sl, sl] / exact[sl, sl] - up[10, 10] / exact[10, 10])) < 1e-12


@pytest.mark.validates("crosscheck.alma_maxent", roots=["mathematics"], kind="reference")
def test_the_beam_convolution_uses_cleans_own_position_angle():
    """``crosscheck.alma_maxent.convolve`` of a point source with an elliptical beam is ``alma_clean.gaussian_image`` (the kernel of CLEAN's
    ``restore``) at the same PA: the start image is smoothed with the beam CLEAN restored with, not its mirror image."""
    n, px = 65, 3.0
    delta = np.zeros((n, n))
    delta[(n - 1) // 2, (n - 1) // 2] = 1.0
    ref = ac.gaussian_image((n, n), ((n - 1) // 2, (n - 1) // 2), 60.0 / px, 25.0 / px, 50.0)
    ref = ref / ref.sum()  # the periodic convolution of a point source is the kernel, centred on the point
    corr = {pa: float(np.corrcoef(cx.convolve(delta, px, 60.0, 25.0, pa).ravel(), ref.ravel())[0, 1]) for pa in (50.0, -50.0)}
    record("corr_same_pa", corr[50.0])
    assert corr[50.0] > 0.999 and corr[-50.0] < 0.99


@pytest.mark.validates("virgil_bridge.alma_start", roots=["mathematics"])
def test_clean_start_is_positive_floored_flux_consistent_and_reads_back_through_image_of(tmp_path):
    """From a CLEAN directory: the image is what ``image_of`` returns from (log_brightness, flux) to 1e-12, every pixel is at least the floor, the
    flux is the field's flux plus the floor's, the small kernel (default) keeps the long baselines' power (a peak above the beam-smoothed one)."""
    d, img = _clean_dir(tmp_path / "c")
    model, px, beam, info = vs.read_clean(d)
    assert px == pytest.approx(3.0) and beam == (40.0, 25.0, 60.0) and model.shape == (51, 51)
    assert abs(model.sum() / img.sum() - 1) < 1e-6  # float32 FITS
    model = model.copy()
    model[20, 20] = -1e-4  # a negative component, as multi-scale CLEAN can leave
    eta, flux, st = vs.clean_start(model, px, beam, npix=32, pixel_mas=5.0)
    start = va.image_of(dict(log_brightness=eta, **{"cal.flux": flux}), 32, 5.0)
    assert start.min() >= 1e-4 * (1 - 1e-12) and np.all(np.isfinite(eta)) and eta.max() == 0.0
    assert abs(start.sum() * 5e-3**2 / flux - 1) < 1e-12
    assert abs(flux - (st["flux_in_field_jy"] + st["flux_added_by_floor_jy"])) < 1e-12
    assert st["negative_flux_clipped_jy"] <= 0 and st["flux_added_by_floor_jy"] >= 0
    # no beam smoothing by default: sharper than the beam-smoothed start, same flux in the field
    eta_b, flux_b, st_b = vs.clean_start(model, px, beam, npix=32, pixel_mas=5.0, smooth_mas=0.0, beam_frac=1.0)
    record("peak_ratio_small_kernel_over_beam", st["peak_jy_arcsec2"] / st_b["peak_jy_arcsec2"])
    assert st["peak_jy_arcsec2"] > 1.5 * st_b["peak_jy_arcsec2"] and abs(st_b["flux_in_field_jy"] / st["flux_in_field_jy"] - 1) < 1e-6
    with pytest.raises(ValueError):
        vs.clean_start(model, px, None, npix=32, pixel_mas=5.0, beam_frac=1.0)


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_a_held_image_is_no_free_site_and_gives_the_same_objective(tmp_path):
    """``build_problem(fixed_log_brightness=...)``: no ``log_brightness`` prior, and the data term and MPoL's entropy at the same values equal the
    free problem's, to 1e-12 (the held image is used by both)."""
    syn = synthetic_dir(tmp_path / "s")
    data = va.load_dir(syn)
    free = va.build_problem(data, "R", lam=0.5, npix=24)
    rng = np.random.default_rng(0)
    eta = rng.normal(size=(24, 24))
    held = va.build_problem(data, "R", lam=0.5, npix=24, fixed_log_brightness=eta)
    assert "log_brightness" in free.priors and "log_brightness" not in held.priors
    v = {k: np.asarray(x) for k, x in free.init.items()}
    v_free, v_held = {**v, "log_brightness": eta}, v
    with va.ft_eps(1e-9):
        a, b = np.asarray(free.terms[0](v_free)), np.asarray(held.terms[0](v_held))
    assert np.max(np.abs(a - b)) < 1e-12 * np.max(np.abs(a))
    assert abs(float(held.terms[1].entropy(v_held)) / float(free.terms[1].entropy(v_free)) - 1) < 1e-12
    with pytest.raises(ValueError):
        va.build_problem(data, "R", lam=0.5, npix=24, fixed_log_brightness=eta[:10, :10])
    with pytest.raises(ValueError):
        va.build_problem(data, "F", lam=0.5, npix=24, fixed_log_brightness=eta)


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_workflow_clean_runs_selfcal_then_all_from_the_clean_image_and_records_it(tmp_path):
    """``fit --workflow clean`` on synthetic data: stages selfcal (the image held: no ``log_brightness`` moves) then all, the start is the CLEAN
    image (no C9 Gaussian, no MPoL start), result.json holds the start's numbers, the self-calibration check and the usual report."""
    syn = synthetic_dir(tmp_path / "s")
    d, _ = _clean_dir(tmp_path / "c")
    out = tmp_path / "o"
    _call("fit", "--config", "R", "--workflow", "clean", "--clean-dir", d, "--data", syn, "--out", out, "--npix", 32,
          "--max-steps", 5, "--prestage-max-steps", 5, "--no-dirty")
    r = json.loads((out / "result.json").read_text())
    assert r["workflow"] == "clean" and r["complete"] and [s["name"] for s in r["stages"]] == ["selfcal", "all"]
    assert r["n"] == 5600  # all data in both stages
    assert r["clean_start"]["npix"] == 32 and r["clean_start"]["smooth_mas"] == 10.0 and r["clean_start"]["beam_frac"] == 0.0
    chk = r["stages"][0]["selfcal_check"]
    assert set(chk) >= {"gain_min", "gain_max", "gain_lb_obs5_6", "gain_outside_reference", "flux_jy"}
    for k in ("nll", "chi2_raw", "objective_R", "gain", "offset_mas", "sigma", "chi2_s1", "flux_jy"):
        assert k in r and r[k] is not None
    model, px, beam, _ = vs.read_clean(d)
    eta, _, _ = vs.clean_start(model, px, beam, npix=32, pixel_mas=5.0)
    assert np.max(np.abs(np.load(out / "values.npz")["log_brightness"] - eta)) > 0  # released in `all`
    with pytest.raises(SystemExit):
        _call("fit", "--config", "R", "--workflow", "clean", "--data", syn, "--out", tmp_path / "x")  # no --clean-dir
    with pytest.raises(SystemExit):
        _call("fit", "--config", "F", "--workflow", "clean", "--clean-dir", d, "--data", syn, "--out", tmp_path / "y")
    with pytest.raises(SystemExit):
        _call("fit", "--config", "R", "--workflow", "clean", "--sb-only", "--clean-dir", d, "--data", syn, "--out", tmp_path / "z")


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_the_selfcal_stage_holds_the_image_exactly_and_all_starts_from_it(tmp_path, monkeypatch):
    """Spy on ``run_fit_c8``: the selfcal stage's problem has no ``log_brightness`` site, its nuisances move off their start, and the next
    stage starts from exactly the CLEAN start's image (``eta``) and the self-calibrated nuisances, then frees the image."""
    syn = synthetic_dir(tmp_path / "s")
    d, _ = _clean_dir(tmp_path / "c")
    model, px, beam, _ = vs.read_clean(d)
    eta, _, _ = vs.clean_start(model, px, beam, npix=32, pixel_mas=5.0)
    calls, real = [], va.run_fit_c8

    def spy(prob, start, **kw):
        out, *rest = real(prob, start, **kw)
        calls.append(dict(sites=set(prob.priors), start={k: np.asarray(v) for k, v in start.items()}, out={k: np.asarray(v) for k, v in out.items()}))
        return (out, *rest)

    monkeypatch.setattr(va, "run_fit_c8", spy)
    _call("fit", "--config", "R", "--workflow", "clean", "--clean-dir", d, "--data", syn, "--out", tmp_path / "o", "--npix", 32,
          "--max-steps", 5, "--prestage-max-steps", 5, "--no-dirty")
    selfcal, full = calls[0], calls[1]
    assert "log_brightness" not in selfcal["sites"] and "log_brightness" not in selfcal["start"] and "log_brightness" not in selfcal["out"]
    assert any(np.any(selfcal["out"][k] != selfcal["start"][k]) for k in ("cal.gain", "cal.flux"))
    assert "log_brightness" in full["sites"] and np.array_equal(full["start"]["log_brightness"], eta)
    for k in ("cal.gain", "cal.offset", "cal.sigma", "cal.flux"):
        assert np.array_equal(full["start"][k], selfcal["out"][k])
    assert np.max(np.abs(full["out"]["log_brightness"] - eta)) > 0


@pytest.mark.validates("virgil_bridge.alma_start", roots=["self-consistency"], kind="guard")
def test_selfcal_check_is_in_obs0s_gauge_whatever_the_reference_observation():
    """A fit with reference observation 3 (its gain 1) is renormalised to obs 0 = 1 before the range test and before picking obs 5, 6: the same
    physical gains give the same check from the ref-0 and the ref-3 parametrisations."""
    g0 = np.array([1.0, 1.02, 1.03, 0.99, 1.01, 1.02, 1.03])  # obs 0 .. 6 in obs 0's gauge
    off = np.zeros((6, 2))
    ref0 = vs.selfcal_check({"cal.gain": g0[1:], "cal.flux": 0.27, "cal.offset": off})
    k = 3
    g_k = np.delete(g0 / g0[k], k)  # the same model in obs 3's gauge: g / g_3, flux g_3
    refk = vs.selfcal_check({"cal.gain": g_k, "cal.flux": 0.27 * g0[k], "cal.offset": off}, ref_obs=k)
    assert np.allclose(refk["gain"], ref0["gain"]) and np.allclose(refk["gain_lb_obs5_6"], [1.02, 1.03]) and abs(refk["flux_jy"] - 0.27) < 1e-12
    assert not refk["gain_outside_reference"]
    naive = np.delete(g0 / 1.15, 3)  # a gauge 15 % off, read as if obs 0 were the reference: outside the range
    assert vs.selfcal_check({"cal.gain": naive, "cal.flux": 0.27, "cal.offset": off})["gain_outside_reference"]


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_r_objective_compares_in_one_gauge_whatever_the_reference_observation(tmp_path):
    """-ln L is invariant under g -> g / g_k, flux -> flux g_k; S_M is not. ``r_objective(entropy_flux=...)`` gives the ref-k fit the entropy
    of the same image in obs 0's gauge, so its objective equals the ref-0 parametrisation's; without it the entropies differ."""
    syn = synthetic_dir(tmp_path / "s")
    data = va.load_dir(syn)
    n_obs, k, lam = len(data.obs_ids), 3, 0.5
    p0 = va.build_problem(data, "R", lam=lam, npix=24, ref_obs=0)
    rng = np.random.default_rng(3)
    v0 = {key: np.asarray(x) for key, x in p0.init.items()}
    v0["log_brightness"] = rng.normal(size=(24, 24))
    g_full = np.concatenate([[1.0], rng.uniform(0.9, 1.1, n_obs - 1)])
    v0["cal.gain"], v0["cal.flux"] = g_full[1:], np.asarray(0.3)
    vk = dict(v0, **{"cal.gain": np.delete(g_full / g_full[k], k), "cal.flux": np.asarray(0.3 * g_full[k])})
    o0 = va.r_objective(data, v0, lam=lam, pixel_mas=5.0, ref_obs=0)
    ok = va.r_objective(data, vk, lam=lam, pixel_mas=5.0, ref_obs=k)
    fixed = va.r_objective(data, vk, lam=lam, pixel_mas=5.0, ref_obs=k, entropy_flux=0.3)
    assert abs(ok["nll"] / o0["nll"] - 1) < 1e-9
    assert abs(ok["entropy"] - o0["entropy"]) > 1e-4 * abs(o0["entropy"])  # the gauge shows in the entropy
    assert abs(fixed["objective"] / o0["objective"] - 1) < 1e-9 and fixed["entropy_fit_gauge"] == ok["entropy"]


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_offset_bound_report_judges_each_offset_against_its_own_observations_bound():
    """A per-observation bound (virgil's own): an offset at 90 % of a tight bound warns although it is 9 % of the loosest; the offsets' rows
    are the observations but the reference, so with reference 2 the rows name obs 0, 1, 3."""
    bound = np.array([10000.0, 100.0, 10000.0, 10000.0])
    off = np.array([[1.0, 0.0], [96.0, 0.0], [5.0, 0.0]])  # obs 1 is the tight one; rows are obs 0, 1, 3 for reference 2
    r = va.offset_bound_report(off, bound.max(), bound_obs=bound, ref_obs=2)
    assert [(n["obs"], n["axis"]) for n in r["near_bound"]] == [(1, "ra")] and r["warning"] and abs(r["max_abs_over_bound"] - 0.96) < 1e-12
    assert va.offset_bound_report(off, 100.0)["warning"] is not None  # scalar bound unchanged: |96| near 100
    assert va.offset_bound_report(off[:1], 100.0, ref_obs=0)["near_bound"] == []


@pytest.mark.validates("virgil_bridge.alma_start", roots=["self-consistency"], kind="guard")
def test_selfcal_check_flags_a_gain_outside_the_reference_range():
    v = dict(**{"cal.gain": np.array([1.02, 1.03, 0.99, 1.01, 1.02, 1.03]), "cal.flux": 0.27, "cal.offset": np.array([[1.0, -2.0]] * 6)})
    ok = vs.selfcal_check(v)
    assert not ok["gain_outside_reference"] and ok["offset_max_abs_mas"] == 2.0 and list(ok["gain_lb_obs5_6"]) == [1.02, 1.03]
    v["cal.gain"] = np.array([1.02, 1.03, 0.99, 1.01, 0.80, 1.03])  # the long-baseline gain that falls to absorb a smoothed image
    assert vs.selfcal_check(v)["gain_outside_reference"]


# ------------------------------------------------------------------ image-first staging, the offset bound and the reference observation


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_image_only_problem_has_only_the_image_and_flux_free_and_the_same_objective_at_g1(tmp_path):
    """``image_only``: the fit's sites are the image and the flux; its data term equals the full problem's at gains 1, offsets 0, s = 1 (1e-12)."""
    syn = synthetic_dir(tmp_path / "s")
    data = va.load_dir(syn)
    full = va.build_problem(data, "R", lam=0.5, npix=24)
    img = va.build_problem(data, "R", lam=0.5, npix=24, image_only=True)
    assert set(img.priors) == {"log_brightness", "cal.flux"} and set(img.init) == {"cal.flux"}
    eta = np.random.default_rng(1).normal(size=(24, 24))
    v = {k: np.asarray(x) for k, x in full.init.items()}
    with va.ft_eps(1e-9):
        a = np.asarray(full.terms[0]({**v, "log_brightness": eta}))
        b = np.asarray(img.terms[0]({"cal.flux": v["cal.flux"], "log_brightness": eta}))
    assert np.max(np.abs(a - b)) < 1e-12 * np.max(np.abs(a))


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
@pytest.mark.parametrize("workflow, names", [("mpol", ["image", "all"]), ("p3", ["image", "sb", "all"])])
def test_image_first_runs_an_image_stage_then_the_workflows_own_stages(tmp_path, workflow, names):
    syn = synthetic_dir(tmp_path / "s")
    out = tmp_path / "o"
    _call("fit", "--config", "R", "--workflow", workflow, "--image-first", "--data", syn, "--out", out, "--npix", 24, "--max-steps", 4, "--prestage-max-steps", 4, "--no-dirty")
    r = json.loads((out / "result.json").read_text())
    assert [s["name"] for s in r["stages"]] == names and r["complete"] and r["args"]["image_first"] is True
    assert r["stages"][0]["steps"] > 0
    z = np.load(out / "values.npz")
    assert {"cal.gain", "cal.offset", "cal.sigma"} <= set(z.files)  # released in the later stages
    for bad in (["--workflow", "clean", "--clean-dir", tmp_path], ["--init", out]):
        with pytest.raises(SystemExit):
            _call("fit", "--config", "R", "--image-first", *map(str, bad), "--data", syn, "--out", tmp_path / "x", "--npix", 24)


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_ref_obs_auto_picks_the_short_baseline_observation_with_the_largest_weight_and_is_logged(tmp_path):
    syn = synthetic_dir(tmp_path / "s")
    data = va.load_dir(syn)
    w = np.asarray(data.weight)
    ref, info = va.resolve_ref_obs(data, "auto")
    sums = [float(w[np.asarray(data.obs) == o].sum()) for o in va.SB_OBS]
    assert ref == va.SB_OBS[int(np.argmax(sums))] and info["ref_obs"] == ref and np.allclose(info["sb_sum_weight"], sums)
    assert va.resolve_ref_obs(data, "2")[0] == 2 and va.resolve_ref_obs(data, 4)[0] == 4
    for bad in ("7", "-1"):
        with pytest.raises(ValueError):
            va.resolve_ref_obs(data, bad)


@pytest.mark.validates("virgil_bridge.alma", roots=["mathematics"])
def test_gauge_to_obs0_leaves_the_model_amplitudes_unchanged():
    """The model at observation e is flux g_e: after ``gauge_to_obs0`` the products are the same to 1e-15, obs 0 has gain 1 and offset 0, the
    offsets differ from the fit's by the one shift o_0; with ref_obs 0 the result is the fit's own."""
    rng = np.random.default_rng(4)
    gain, offset, flux = rng.uniform(0.9, 1.1, 6), rng.normal(size=(6, 2)), 0.27
    for ref in (0, 3):
        g, o, f = va.gauge_to_obs0(gain, offset, flux, ref)
        full = np.insert(gain, ref, 1.0)
        assert np.allclose(f * g, flux * full, rtol=1e-15) and g[0] == 1.0 and np.all(o[0] == 0)
        assert np.allclose(o - np.insert(offset, ref, 0.0, axis=0), -(np.insert(offset, ref, 0.0, axis=0)[0]), atol=1e-15)
    g0, o0, f0 = va.gauge_to_obs0(gain, offset, flux, 0)
    assert np.allclose(g0[1:], gain) and np.allclose(o0[1:], offset) and f0 == flux


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_fit_with_ref_obs_records_it_the_bound_report_and_the_beam_bounds(tmp_path):
    """``--ref-obs auto --offset-bound-mas 100``: result.json holds the resolved reference observation and its weights, the 3-beam proxy bound of
    every observation against the bound in force; the saved fit reports again (``report``) in the same gauge."""
    syn = synthetic_dir(tmp_path / "s")
    out = tmp_path / "o"
    _call("fit", "--config", "R", "--ref-obs", "auto", "--offset-bound-mas", 100, "--data", syn, "--out", out, "--npix", 24, "--max-steps", 4, "--no-dirty", "--sb-only")
    r = json.loads((out / "result.json").read_text())
    assert r["ref_obs"] == r["ref_obs_info"]["ref_obs"] and r["ref_obs_info"]["spec"] == "auto" and len(r["ref_obs_info"]["sb_sum_weight"]) == 5
    b = r["offset_bound_3beam"]
    assert len(b["per_obs_mas"]) == 7 and b["bound_in_force_mas"] == 100.0 and b["tighter_than_all"] == (100.0 < min(b["per_obs_mas"]))
    assert len(r["gain"]) == 6 and len(r["offset_mas"]) == 6  # n_obs - 1 free, whichever is the reference
    _call("report", "--data", syn, "--out", out, "--no-dirty")
    assert json.loads((out / "result.json").read_text())["ref_obs"] == r["ref_obs"]
    with pytest.raises(SystemExit):  # a warm start is in its own fit's gauge
        _call("fit", "--config", "R", "--ref-obs", "1" if r["ref_obs"] != 1 else "0", "--init", out, "--data", syn, "--out", tmp_path / "w", "--npix", 24)


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_cv_task_refuses_an_init_fitted_with_another_reference_observation(tmp_path):
    """``cv-task`` builds the problem with reference observation 0: an init whose result.json says another one would put its n - 1 gains and
    offsets on the wrong observations, so it is refused before anything is fitted."""
    syn = synthetic_dir(tmp_path / "s")
    folds, init = tmp_path / "folds.npz", tmp_path / "init"
    _call("folds", "--data", syn, "--out", folds, "--k", 2)
    init.mkdir()
    np.savez(init / "values.npz", **{k: np.asarray(v) for k, v in va.build_problem(va.load_dir(syn), "R", lam=0.5, npix=24).init.items()})
    (init / "result.json").write_text(json.dumps(dict(ref_obs=2)))
    with pytest.raises(SystemExit, match="reference observation 2"):
        _call("cv-task", "--config", "R", "--data", syn, "--folds", folds, "--fold", 0, "--init", init, "--lam", 1.0, "--npix", 24, "--max-steps", 2,
              "--out", tmp_path / "cv")


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_release_arm_flag_combinations_clean_start_adam_toeplitz_and_classic_mem(tmp_path):
    """The flags of the release arms are accepted together: the CLEAN start (``--workflow clean``) with Adam, Toeplitz chi^2 and a decaying lr schedule, and the classic-MEM arm
    (``--taper-mas 0``) with Adam and Toeplitz, all on small synthetic data; Adam's lr schedule needs virgil#562, so those runs are skipped without it."""
    import inspect

    import virgil

    ok, err = va.adam_features_probe(virgil.fit)
    if not ok or any(k not in inspect.signature(virgil.fit).parameters for k in va.ADAM_KEYWORDS + ("post_step",)):
        pytest.skip(f"installed virgil predates virgil#562: {err}")
    syn = synthetic_dir(tmp_path / "s")
    d, _ = _clean_dir(tmp_path / "c")
    common = ("--config", "R", "--data", syn, "--npix", 32, "--method", "adam", "--adam-steps", 30, "--adam-lr", 1e-2, "--adam-const-steps", 10, "--adam-lr-decay", 0.7, "--chi2", "toeplitz",
              "--offset-bound-mas", 20, "--adam-snapshot-every", 20, "--no-dirty")
    _call("fit", *common, "--workflow", "clean", "--clean-dir", d, "--lam-m", 1e-9, "--out", tmp_path / "clean")
    r = json.loads((tmp_path / "clean/result.json").read_text())
    assert r["chi2_engine"] == "toeplitz" and [s["name"] for s in r["stages"]] == ["selfcal", "all"] and r["args"]["method"] == "adam"
    _call("fit", *common, "--workflow", "mpol", "--taper-mas", 0, "--lam-m", 1e-9, "--out", tmp_path / "mem")
    r = json.loads((tmp_path / "mem/result.json").read_text())
    assert r["chi2_engine"] == "toeplitz" and r["summary"]["taper_mas"] == 0 and r["trajectory"]


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_lam_m_resolves_to_lam_r_and_a_lam_r_below_1e_4_is_refused():
    """``--lam-m 3e-10`` is lam_R = 2 N lambda_M = 0.025452679 at N = 42,421,131 (the default lambda_M = 1e-9 gives 0.0848...); ``--lam 1e-9`` (a lambda_M given as lam_R, which turns the entropy
    off) and ``--lam`` with ``--lam-m`` are refused."""
    import argparse

    n = 42_421_131
    ns = lambda **k: argparse.Namespace(**{**dict(lam=None, lam_m=None, config="R", lam_r_equiv=None, flux=0.25), **k})  # noqa: E731
    assert script.lam_of(ns(lam_m=3e-10), n) == pytest.approx(0.025452679, rel=1e-6)
    assert script.lam_of(ns(), n) == pytest.approx(va.lam_r(n))
    assert script.lam_of(ns(lam=0.05), n) == 0.05
    with pytest.raises(SystemExit, match="--lam-m"):
        script.lam_of(ns(lam=1e-9), n)
    with pytest.raises(SystemExit, match="exclusive"):
        script.lam_of(ns(lam=0.05, lam_m=1e-9), n)


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_entropy_on_base_is_the_untapered_entropy_while_the_model_keeps_the_kernel(tmp_path):
    """``entropy_on="base"`` (MEM's intrinsic correlation function, Gull 1989): the entropy term is ``mpol_entropy_jax(taper_mas=0)`` of the base image, the data term is that of the default problem (the
    model keeps its 30 mas kernel, so the model visibilities are unchanged), and the default ``"blurred"`` is the entropy of the blurred image, as before; with ``taper_mas=0`` the two coincide."""
    import jax.numpy as jnp

    data = va.load_dir(synthetic_dir(tmp_path / "s"))
    kw = dict(lam=0.5, npix=24)
    blurred, base = va.build_problem(data, "R", **kw), va.build_problem(data, "R", entropy_on="base", **kw)
    v = {k: jnp.asarray(x) for k, x in blurred.init.items()}
    v["log_brightness"] = jnp.asarray(np.random.default_rng(0).normal(size=(24, 24)))
    s_base = float(base.terms[1].entropy(v))
    ref = float(va.mpol_entropy_jax(v["log_brightness"], v["cal.flux"], pixel_mas=blurred.pixel_mas, taper_mas=0.0, prior=va.MPOL["prior"], f_tot=va.MPOL["f_tot"]))
    assert s_base == pytest.approx(ref, rel=1e-10)
    assert float(blurred.terms[1].entropy(v)) == pytest.approx(float(va.mpol_entropy_jax(v["log_brightness"], v["cal.flux"], pixel_mas=blurred.pixel_mas, taper_mas=30.0, prior=va.MPOL["prior"], f_tot=va.MPOL["f_tot"])), rel=1e-10)
    assert s_base != pytest.approx(float(blurred.terms[1].entropy(v)), rel=1e-6)
    with va.ft_eps(1e-9):
        a, b = np.asarray(base.terms[0](v)), np.asarray(blurred.terms[0](v))
    assert np.max(np.abs(a - b)) <= 1e-12 * np.max(np.abs(a))
    assert float(va.build_problem(data, "R", entropy_on="base", taper_mas=0, **kw).terms[1].entropy(v)) == pytest.approx(float(va.build_problem(data, "R", taper_mas=0, **kw).terms[1].entropy(v)), rel=1e-12)
    with pytest.raises(ValueError):
        va.build_problem(data, "R", entropy_on="nope", **kw)
