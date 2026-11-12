"""The informed contest fits (scripts/contest_informed.py) and their quicklook figures.

Priors, ladders and the quicklook are checked without a fit. The fit tests use synthetic data from a known model
observed on a made-up uv coverage (`OIData.with_model`), never a contest file, so they run anywhere. They check
behaviour that must hold however the optimizer is tuned: nested rungs are never worse, the starts record their
origin, the moment fit and the three scales come from the data, and the checks are written to result.json.
"""

import importlib.util
import json
import pathlib
import sys

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import virgil.models as vm  # noqa: E402
from virgil.oidata import OIData  # noqa: E402

from crosscheck import informed_starts as IS  # noqa: E402

pytestmark = pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")

SCRIPTS = pathlib.Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C = load("contest_informed")
Q = load("contest_informed_quicklook")

WAVEL = 1.65e-6


def synthetic(model, n=120, b_min=8.0, b_max=300.0, sigma=0.01, seed=0):
    """V^2 of ``model`` on a log-uniform uv coverage (metres), with Gaussian noise of ``sigma``."""
    rng = np.random.default_rng(seed)
    b = np.exp(rng.uniform(np.log(b_min), np.log(b_max), n))
    theta = rng.uniform(0.0, np.pi, n)
    blank = OIData({"u": b * np.sin(theta), "v": b * np.cos(theta), "wavel": WAVEL, "vis": np.ones(n), "d_vis": np.full(n, sigma)})
    return blank.with_model(model, key=jax.random.PRNGKey(seed))


def scales_for(datasets, field=None):
    sc, moments, datasets, _ = C.measure(datasets, {"field": field, "field_source": "test"} if field else {})
    return sc, moments, datasets


# ---------------------------------------------------------------------------------------------------------------
# The quicklook figure (plan: the extra panels were drawn upside down)


def test_the_quicklook_draws_a_source_north_of_centre_in_the_upper_half_and_east_on_the_left():
    # virgil's render has row 0 = north and column 0 = east: a source at (dra, ddec) = (+3, +4) mas is in the top
    # rows and the left columns of the array, and must be drawn at positive y, on the left of the axes.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    npix, fov = 64, 20.0
    image = np.asarray(vm.System(star=vm.PointSource(dra=3.0, ddec=4.0)).render(npix=npix, fov_mas=fov))
    row, column = np.unravel_index(np.argmax(image), image.shape)
    assert row < npix / 2 and column < npix / 2, "premise: render puts north in row 0 and east in column 0"
    fig, ax = plt.subplots()
    artist = Q.draw_image(ax, image, fov)
    left, right, bottom, top = artist.get_extent()
    x = left + (column + 0.5) * (right - left) / npix
    y = top - (row + 0.5) * (top - bottom) / npix if artist.origin == "upper" else bottom + (row + 0.5) * (top - bottom) / npix
    plt.close(fig)
    assert y > 0 and x > 0
    assert x < fov / 2 and y < fov / 2
    assert ax.get_xlim()[0] > ax.get_xlim()[1], "east is on the left: the x axis is reversed"
    assert x == pytest.approx(3.0, abs=fov / npix) and y == pytest.approx(4.0, abs=fov / npix)


# ---------------------------------------------------------------------------------------------------------------
# Priors (plan 3)


def test_fluxes_have_no_cap_and_only_a_planet_is_limited_to_a_tenth():
    sc = IS.scales_of(1.0 / (2.0 * 1.0 * IS.MAS), 1.0 / (50.0 * IS.MAS), None)
    flux, planet = C.prior_of("flux", sc), C.prior_of("planet", sc)
    assert (float(flux.low), float(flux.high)) == (1e-4, 1e4)
    assert float(planet.low) == 1e-4 and float(planet.high) == 0.1
    colour = C.prior_of("colour", sc)
    assert (float(colour.low), float(colour.high)) == (-4.0, 4.0)
    inc = C.prior_of("inclination", sc)
    assert (float(inc.low), float(inc.high)) == (0.0, 89.0)


def test_sizes_run_from_a_fifth_of_the_resolution_to_the_extent_and_positions_follow_the_reach_and_extent():
    sc = IS.scales_of(1.0 / (2.0 * 2.0 * IS.MAS), 1.0 / (40.0 * IS.MAS), None)
    size = C.prior_of("size", sc)
    assert float(size.low) == pytest.approx(sc.resolution / 5.0) and float(size.high) == pytest.approx(sc.extent)
    far, near = C.prior_of("far", sc), C.prior_of("near", sc)
    assert float(far.high) == pytest.approx(sc.reach) and float(far.low) == pytest.approx(-sc.reach)
    assert float(near.high) == pytest.approx(sc.extent / 2.0)


def test_no_prior_number_comes_from_a_paper_so_every_kind_is_built_from_the_scales():
    sc = IS.scales_of(1.0 / (2.0 * 3.0 * IS.MAS), 1.0 / (60.0 * IS.MAS), None)
    for family in C.FAMILIES:
        for name, rung in C.ladder(family):
            assert C.build_priors(rung, sc), (family, name)
            assert C.build_priors(rung, sc, wavel0=1.6e-6), (family, name)


# ---------------------------------------------------------------------------------------------------------------
# Ladders


def test_every_family_ends_with_a_resolved_rung_that_contains_the_one_below_it():
    for family in C.FAMILIES:
        rungs = C.ladder(family)
        (_, last), (name, more) = rungs[-2], rungs[-1]
        assert name.endswith("+resolved"), family
        assert C.contains(more, last), family
        assert not C.contains(last, more), family


def test_a_resolved_rung_is_a_model_with_light_that_no_baseline_sees():
    rung = dict(C.ladder("star_with_faint_companion"))["star+halo+companion+resolved"]
    values = {"star.diam": 0.5, "halo.sigma": 4.0, "halo.flux": 0.3, "companion.flux": 0.01, "companion.dra": 5.0, "companion.ddec": 1.0,
              "diffuse.flux": 2.0}
    model = C.model_of(rung, values)
    base = C.model_of(dict(C.ladder("star_with_faint_companion"))["star+halo+companion"], {k: v for k, v in values.items() if k != "diffuse.flux"})
    b = np.array([20.0, 90.0, 250.0])
    blank = OIData({"u": b, "v": np.zeros(3), "wavel": WAVEL, "vis": np.ones(3), "d_vis": np.full(3, 0.01)})
    with_resolved, without = np.asarray(blank.with_model(model).vis), np.asarray(blank.with_model(base).vis)
    # more light, all of it over-resolved: the V^2 is the old one times (old flux / new total)^2
    total_before, total_after = 1.0 + 0.3 + 0.01, 1.0 + 0.3 + 0.01 + 2.0
    assert with_resolved == pytest.approx(without * (total_before / total_after) ** 2, rel=1e-6)


def test_a_scene_rung_gets_a_resolved_component_too():
    rung = dict(C.ladder("thin_disk"))["star+disk, two azimuthal harmonics+resolved"]
    assert "diffuse.flux" in rung.kinds() and rung.components[-1] == "diffuse"


def test_chromatic_rungs_give_every_flux_a_ratio_and_a_spectral_index():
    rung = dict(C.ladder("star_rim_planet_disk"))["star+rim+outer+planet"]
    kinds = C.flat_kinds(rung, wavel0=1.6e-6)
    assert kinds["planet.flux.ratio"] == "planet" and kinds["planet.flux.index"] == "colour"
    assert "planet.flux" not in kinds


def test_a_drawn_planet_start_is_at_least_two_resolution_elements_from_the_star():
    sc = IS.scales_of(1.0 / (2.0 * 2.0 * IS.MAS), 1.0 / (40.0 * IS.MAS), None)
    rung = dict(C.ladder("star_with_faint_companion"))["star+halo+companion"]
    kinds, priors = C.flat_kinds(rung), C.build_priors(rung, sc)
    rng = np.random.default_rng(3)
    for _ in range(50):
        s = C.draw(priors, kinds, sc, rng)
        assert np.hypot(s["companion.dra"], s["companion.ddec"]) >= IS.COMPANION_ELEMENTS * sc.resolution


# ---------------------------------------------------------------------------------------------------------------
# Scales and moments from the data


def test_the_three_scales_and_the_moment_fit_come_from_the_data():
    truth = vm.System(star=vm.PointSource(), halo=vm.GaussianDisk(sigma=6.0, flux=1.5))
    data = synthetic(truth, n=200)
    sc, moments, _ = scales_for([data])
    assert moments is not None and moments["extended"] > moments["compact"] * 0.5
    # the contest gave no field: the extent is 4 x the fitted major FWHM (or at least 8 resolution elements) and says so
    assert sc.extent_known and "visibility-moment" in sc.extent_source
    assert sc.extent == pytest.approx(max(4.0 * moments["fwhm"], 8.0 * sc.resolution))
    assert sc.pixel > 0 and sc.reach > 0 and sc.resolution > 0
    assert moments["fwhm"] == pytest.approx(2.0 * np.sqrt(2.0 * np.log(2.0)) * 6.0, rel=0.35)


def test_a_given_field_is_the_extent_and_says_where_it_came_from():
    data = synthetic(vm.System(star=vm.PointSource(), halo=vm.GaussianDisk(sigma=6.0, flux=1.5)))
    sc, _, _ = scales_for([data], field=40.0)
    assert sc.extent == 40.0 and sc.extent_known and "test" in sc.extent_source


def test_the_temper_factor_is_the_misfit_of_the_moment_fit_and_never_below_one():
    assert C.temper_factor(None) == 1.0
    assert C.temper_factor({"chi2_quoted": 10.0, "n": 100}) == 1.0
    assert C.temper_factor({"chi2_quoted": 400.0, "n": 100}) == pytest.approx(2.0)


def test_the_visibility_scale_of_2004_data2_divides_v2_data_and_its_errors_by_the_fitted_amplitude():
    truth = vm.System(star=vm.PointSource(), halo=vm.GaussianDisk(sigma=5.0, flux=0.8))
    data = synthetic(truth, n=200, sigma=0.01)
    scaled_up = C.divided(data, 1.0 / 1.2)  # the data as if normalised without 20% of the flux: V^2 x 1.2
    sc, moments, fixed, vis_scale = C.measure([scaled_up], {}, scale_vis=True)
    assert vis_scale["applied"] and vis_scale["amplitude_v2"] == pytest.approx(1.2, rel=0.1)
    # after the scale the data are back near the truth
    assert float(np.median(np.asarray(fixed[0].vis) / np.asarray(data.vis))) == pytest.approx(1.0, rel=0.1)
    # and without the option nothing is scaled
    _, _, same, none = C.measure([scaled_up], {}, scale_vis=False)
    assert none is None and np.allclose(np.asarray(same[0].vis), np.asarray(scaled_up.vis))


def test_the_visibility_scale_is_applied_to_the_named_dataset_only():
    assert C.VIS_SCALE == ("2004_data2",)


# ---------------------------------------------------------------------------------------------------------------
# Fits on synthetic data


def context(datasets, budget=3, randoms=2, refit=3, max_steps=60):
    sc, moments, datasets = scales_for(datasets)
    temper = C.temper_factor(moments)
    return {"datasets": datasets, "tempered": [d.with_error_scale(temper) for d in datasets], "scales": sc, "moments": moments, "smoke": True,
            "budget": budget, "random": randoms, "refit": refit, "max_steps": max_steps}


@pytest.fixture(scope="module")
def ladder_fit():
    """A star with a faint companion and a halo, fitted up the family `star_with_faint_companion` (smoke-sized)."""
    truth = vm.System(star=vm.UniformDisk(diam=0.4), halo=vm.GaussianDisk(sigma=5.0, flux=0.6),
                      companion=vm.PointSource(flux=0.02, dra=8.0, ddec=-5.0))
    data = synthetic(truth, n=160, b_max=330.0, sigma=0.004, seed=2)
    ctx = context([data])
    rng = np.random.default_rng(1)
    rungs, carry, below = [], {}, None
    for name, rung in C.ladder("star_with_faint_companion")[:3]:
        got = C.fit_rung(name, rung, below, ctx, carry, rng)
        carry, below = dict(got["values"]), rung
        rungs.append((name, rung, got))
    return ctx, rungs


def test_every_start_records_its_origin_and_the_report_counts_the_starts_at_the_best_by_origin(ladder_fit):
    _, rungs = ladder_fit
    for _, _, got in rungs:
        assert got["starts"] == sum(got["starts_by_origin"].values())
        assert set(got["starts_by_origin"]) <= {"moments", "grid", "map", "below", "random"}
        assert got["starts_by_origin"].get("moments") == 1 and got["starts_by_origin"].get("random", 0) >= 1
        assert got["starts_at_best"] == sum(got["starts_at_best_by_origin"].values()) >= 1
    # a rung that adds a component starts from the rung below, and gets a map for the new compact companion
    _, _, last = rungs[-1]
    assert last["starts_by_origin"].get("below") == 1
    assert last["maps"] and last["maps"][0]["component"] == "companion"


def test_the_likelihood_map_covers_the_reach_at_half_a_resolution_element(ladder_fit):
    ctx, rungs = ladder_fit
    record = rungs[-1][2]["maps"][0]
    sc = ctx["scales"]
    assert record["half_mas"] == pytest.approx(sc.reach)
    # the smoke run coarsens the step; the full one asks for half a resolution element
    assert record["requested_step_mas"] >= sc.resolution / 2.0 - 1e-9


def test_a_richer_rung_is_never_worse_than_the_rung_it_contains_beyond_the_tolerance(ladder_fit):
    ctx, rungs = ladder_fit
    n = rungs[-1][2]["n"]
    by_name = {name: (rung, got) for name, rung, got in rungs}
    ok, detail = IS.nested([{"rung": name, "minus2lnl": got["minus2lnl"],
                             "contains": [other for other, (r, _) in by_name.items() if other != name and C.contains(rung, r)]}
                            for name, (rung, got) in by_name.items()], n)
    assert ok, detail


def test_the_bic_uses_the_noise_terms_and_chi2_on_the_quoted_errors_is_reported_beside_it(ladder_fit):
    ctx, rungs = ladder_fit
    got = rungs[-1][2]
    assert set(got["noise"]) == {"noise.vis_error_rel"}  # this data set has V^2 only
    assert got["bic"] == pytest.approx(got["minus2lnl"] + got["k"] * np.log(got["n"]))
    assert got["chi2_quoted"] > 0 and got["chi2_red"] == pytest.approx(got["chi2_quoted"] / got["n"])
    # k counts the parameters and the noise terms
    assert got["k"] == len(got["values"]) + len(got["noise"])


def test_the_checks_are_assembled_into_pass_warn_fail_with_the_six_names(ladder_fit):
    ctx, rungs = ladder_fit
    fitted = [got for _, _, got in rungs]
    checks = C.checks_of(fitted, fitted[-1], [(n, r) for n, r, _ in rungs], None, ctx["scales"], True)
    assert set(checks) == {"converged", "no_bounds", "nested", "starts_agree", "family_complete", "no_collapse", "extent_known"}
    assert all(c["status"] in ("pass", "warn", "fail") for c in checks.values())
    assert checks["family_complete"]["status"] == "warn", "no image was fitted on top: a missing measurement, not a pass"
    assert checks["extent_known"]["status"] == "pass"


def test_the_report_and_result_json_carry_the_new_fields(tmp_path, monkeypatch):
    # run() end to end on a synthetic contest-shaped data set: one small family, smoke settings
    truth = vm.System(star=vm.UniformDisk(diam=0.4), halo=vm.GaussianDisk(sigma=5.0, flux=0.6))
    data = synthetic(truth, n=100, sigma=0.005, seed=4)
    monkeypatch.setattr(C, "tasks", lambda: [{"label": "synthetic", "files": ["synthetic.fits"]}])
    monkeypatch.setattr(C, "load_datasets", lambda spec, data_dir, use_spectra: ([data], 0))
    monkeypatch.setattr(C, "FAMILIES", {"star_with_faint_companion": C.FAMILIES["star_with_faint_companion"][:2]})
    monkeypatch.setenv("CRITERIA_COMMIT", "0" * 40)
    report = C.run(0, "star_with_faint_companion", tmp_path, tmp_path, smoke=True, residual=False, no_gp=True)
    written = json.loads((tmp_path / "synthetic_star_with_faint_companion.json").read_text())
    result = json.loads((tmp_path / "result.json").read_text())
    for key in ("scales", "moments", "temper_factor", "visibility_scale", "criteria_commit", "noise_terms", "checks"):
        assert key in written
    assert set(written["scales"]) >= {"resolution", "extent", "extent_source", "reach", "reach_source", "pixel", "pixel_source"}
    assert {"no_bounds", "nested", "starts_agree", "family_complete", "no_collapse", "extent_known", "converged"} <= set(result["checks"])
    assert result["status"] in ("pass", "warn", "fail")
    assert report["rungs"][0]["starts_by_origin"]


# ---------------------------------------------------------------------------------------------------------------
# Review fixes: what a check may and may not wave through


def hand_made_fit(starts=(0.0, 0.1)):
    rung = dict(C.ladder("star_with_faint_companion"))["star"]
    chosen = {"rung": "star", "minus2lnl": 0.0, "n": 100, "values": {}, "bounds": {}, "minus2lnl_of_refit_starts": list(starts)}
    return rung, chosen, IS.scales_of(1.0 / (2.0 * 1.0 * IS.MAS), 1.0 / (50.0 * IS.MAS), None)


def test_family_complete_fails_when_the_image_step_errored_and_warns_only_when_it_was_skipped():
    rung, chosen, sc = hand_made_fit()
    run = lambda rest: C.checks_of([chosen], chosen, [("star", rung)], rest, sc, True)["family_complete"]
    assert run({"error": "RuntimeError: boom"})["status"] == "fail"
    assert run(None)["status"] == "warn"
    assert run({"flux_relative_to_model": 0.05})["status"] == "pass"
    assert run({"flux_relative_to_model": 2.0})["status"] == "fail"


def test_a_chromatic_rung_reports_the_start_statistics_of_its_own_stage(monkeypatch):
    # three of four chromatic starts agree on the optimum; the grey stage's own counts must not leak into the result
    data = synthetic(vm.System(star=vm.PointSource(), halo=vm.GaussianDisk(sigma=5.0, flux=0.5)), n=40)
    sc, _, datasets = scales_for([data])
    rung = dict(C.ladder("star_with_faint_companion"))["star+halo"]
    grey = {"values": {"star.diam": 0.5, "halo.sigma": 4.0, "halo.flux": 0.4}, "noise": {"noise.vis_error_rel": 0.1}}
    below = {"values": {"star.diam": 0.5}}
    minus2lnl = iter([-100.0, -100.2, -100.1, -50.0])

    def final_record(*args, **kwargs):
        return {"minus2lnl": next(minus2lnl), "chi2_quoted": 120.0, "n": 100, "noise": {}, "values": {}, "converged": True, "at_bound": []}

    monkeypatch.setattr(C, "fit_start", lambda *a, **k: (object(), None))
    monkeypatch.setattr(C, "final_record", final_record)
    ctx = {"datasets": datasets, "scales": sc, "max_steps": 5}
    got = C.chromatic_rung(grey, rung, dict(C.ladder("star_with_faint_companion"))["star"], below, ctx, 1.65e-6)
    assert got["starts"] == 4 and got["starts_refit"] == 4
    assert set(got["starts_by_origin"]) == {"grey_best", "below", "index+1", "index-1"}
    assert got["starts_at_best"] == 3 and got["minus2lnl_of_refit_starts"] == [-100.0, -100.2, -100.1, -50.0]
    assert got["minus2lnl"] == -100.2
    # starts_agree is then run on these: three of four within tolerance pass; a lone optimum does not
    assert IS.starts_agree(got["minus2lnl_of_refit_starts"], 100)[0]
    assert not IS.starts_agree([-100.2, 49.0, -50.0, -10.0], 100)[0]


def test_a_run_without_a_pinned_criteria_commit_is_refused_before_any_data_is_read(tmp_path, monkeypatch):
    def refuse(path, root=None):
        raise C.preregistration.CriteriaError("design/contest_informed_criteria.md: not a git checkout and CRITERIA_COMMIT is not set")

    monkeypatch.setattr(C.preregistration, "criteria_commit", refuse)
    monkeypatch.setattr(C, "load_datasets", lambda *a, **k: pytest.fail("data were read before the criteria were pinned"))
    with pytest.raises(C.preregistration.CriteriaError):
        C.run(0, "star_with_faint_companion", tmp_path, tmp_path, smoke=True)
