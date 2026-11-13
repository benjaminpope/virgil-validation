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

import equinox as eqx
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import virgil.models as vm  # noqa: E402
from virgil.oidata import OIData  # noqa: E402

from crosscheck import informed_starts as IS  # noqa: E402

# These tests simulate data with `OIData.with_model`, fit it with virgil, and check the wiring and the report: the
# data's own model is the oracle, so the root is self-consistency, not a closed-form expression.
pytestmark = pytest.mark.validates("pipeline:contest-imaging", roots=["self-consistency"], kind="guard")

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
    blank = OIData({"u": b * np.sin(theta), "v": b * np.cos(theta), "wavel": WAVEL, "vis": np.ones(n), "d_vis": np.full(n, sigma), "frame": np.zeros(n, int)})
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


def contest_data(seed=0, companion=True):
    """Four telescopes seen at six hour angles: 36 baselines (up to 300 m) and the 24 closure phases of their four triangles, of a
    resolved star with a faint companion, built from numpy and `OIData.with_model` alone (no contest file)."""
    tel = np.array([[0.0, 0.0], [120.0, 20.0], [40.0, 180.0], [-100.0, 140.0]])  # metres
    pairs, triangles = [(0, 1), (1, 2), (0, 2), (1, 3), (2, 3), (0, 3)], [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    u, v, tri, where = [], [], [], {}
    for e, angle in enumerate(np.radians([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])):
        c, s = np.cos(angle), np.sin(angle)
        p = tel @ np.array([[c, -s], [s, c]]).T
        for a, b in pairs:
            where[e, a, b] = len(u)
            u.append(p[b, 0] - p[a, 0])
            v.append(p[b, 1] - p[a, 1])
        tri += [(where[e, a, b], where[e, b, c_], where[e, a, c_]) for a, b, c_ in triangles]
    tri = np.array(tri)
    blank = OIData({"u": np.array(u), "v": np.array(v), "wavel": WAVEL, "vis": np.ones(len(u)), "d_vis": np.full(len(u), 0.01),
                    "phi": np.zeros(len(tri)), "d_phi": np.full(len(tri), 0.02), "i_cps1": tri[:, 0], "i_cps2": tri[:, 1], "i_cps3": tri[:, 2],
                    "frame": np.repeat(np.arange(6), 6)})  # one frame per hour angle: a gain needs each sample's frame
    parts = {"star": vm.UniformDisk(diam=1.0)} | ({"companion": vm.PointSource(flux=0.05, dra=6.0, ddec=-4.0)} if companion else {})
    return blank.with_model(vm.System(**parts), key=jax.random.PRNGKey(seed))


TINY = [("star", {"star": C.SMALL_STAR}), ("star+companion", {"star": C.SMALL_STAR, "companion": C.COMPANION})]  # a two-rung family
TRUTH = {"star.diam": 1.0, "companion.flux": 0.05, "companion.dra": 6.0, "companion.ddec": -4.0}


def context(datasets, budget=4, n_peaks=2, n_jitter=1, max_steps=40):
    """What `run` builds before the ladder: scales, moments, the gain nuisance on every dataset, an in-memory cache."""
    sc, moments, raw = scales_for(datasets)
    return {"datasets": [C.with_gain(d) for d in raw], "scales": sc, "moments": moments, "wavel0": None, "cache": C.Cache(), "smoke": True,
            "budget": budget, "n_peaks": n_peaks, "n_jitter": n_jitter, "max_steps": max_steps, "profile_maps": False}


@pytest.fixture(scope="module")
def recipe(tmp_path_factory):
    """The whole recipe, end to end, on the tiny synthetic data set: information check, gain nuisance, rung-0 starts, a companion map,
    the last error-terms fit and the checks (smoke-sized: a few starts of 50 steps; no CLEAN probe, residual image or image step)."""
    out = tmp_path_factory.mktemp("recipe")
    data = contest_data()
    patch = pytest.MonkeyPatch()
    patch.setattr(C, "tasks", lambda: [{"label": "synthetic", "files": ["synthetic.fits"]}])
    patch.setattr(C, "load_datasets", lambda spec, data_dir, use_spectra: ([data], 0))
    patch.setattr(C, "FAMILIES", {"tiny": TINY})
    patch.setattr(C.preregistration, "criteria_commit", lambda path, root=None: "0" * 40)  # the refusal is tested below
    try:
        report = C.run(0, "tiny", out, out, smoke=True, residual=False, no_gp=True, do_probe=False)
    finally:
        patch.undo()
    return report, out


def test_the_recipe_runs_end_to_end_on_a_tiny_synthetic_dataset(recipe):
    report, out = recipe
    written = json.loads((out / "synthetic_tiny.json").read_text())
    result = json.loads((out / "result.json").read_text())
    for key in ("scales", "moments", "temper_factor", "information", "nuisance", "visibility_scale", "criteria_commit", "noise_terms",
                "fit_standard_numbers", "start_agreement", "checks", "rungs", "chosen"):
        assert key in written, key
    assert set(written["scales"]) >= {"resolution", "extent", "extent_source", "reach", "reach_source", "pixel", "pixel_source"}
    # the information check came first: closure phases are judged because most triangles are resolved
    assert written["information"]["triangles"] == 24 and written["information"]["closure_phases_informative"]
    # the gain is marginalised at width 0.1 and nothing is divided out
    assert written["nuisance"]["visibility_gain_width_log_v"] == C.GAIN_WIDTH == 0.1 and not written["visibility_scale"]["applied"]
    assert written["temper_factor"] >= 1.0 and written["error_terms_fitted"] == "once, after the ladder"
    assert result["status"] in ("pass", "warn", "fail") and result["fit_standard"] in ("complete", "incomplete")
    assert report["rungs"][-1]["rung"].endswith("+resolved"), "the ladder ends in a +resolved rung; the image step is not a rung"
    assert set(result["checks"]) == {"converged", "no_bounds", "nested", "starts_agree", "fit_standard", "family_complete", "no_collapse", "extent_known"}


def test_rung_zero_starts_at_the_moments_and_broad_starts_and_higher_rungs_at_map_peaks_and_jitters(recipe):
    rungs = recipe[0]["rungs"]
    first, second, last = (r["starts_by_origin"] for r in rungs)
    assert first["moments"] == 1 and first["grid"] >= 1 and set(first) == {"moments", "grid"}
    # the rung that adds the companion is started at the peaks of its likelihood map and at jitters of the best one: no star-position or random start
    assert set(second) == {"map", "jitter"} and second["map"] >= 1 and second["jitter"] == 1
    assert set(last) == {"below", "jitter"}  # a rung that places nothing: the rung below and jitters of it
    assert all("random" not in r["starts_by_origin"] for r in rungs)
    assert all(r["starts"] == sum(r["starts_by_origin"].values()) for r in rungs)


def test_the_companion_map_ranks_peaks_and_never_thresholds_them(recipe):
    report, _ = recipe
    record = report["rungs"][1]["maps"][0]
    assert record["component"] == "companion" and record["peaks"]
    assert record["half_mas"] == pytest.approx(report["scales"]["reach"])  # a companion is searched for over the whole reach
    assert record["requested_step_mas"] >= report["resolution_mas"] / 2.0 - 1e-9  # smoke coarsens the step; the full run asks for half a resolution element
    assert len(record["seeds"]) == min(2, len(record["peaks"])), "the best N_PEAKS peaks are refitted whatever their score"
    scores = [s["score"] for s in record["seeds"]]
    assert scores == sorted(scores, reverse=True)


def test_the_error_terms_are_free_only_in_the_one_last_fit(recipe):
    report, _ = recipe
    for r in report["rungs"]:
        assert all(not k.startswith("noise") for run in r["start_runs"] for k in run["values"]), r["rung"]
    chosen = next(r for r in report["rungs"] if r["rung"] == report["chosen"])
    assert set(chosen["noise"]) == {"noise.vis_error_rel", "noise.phi_error"} and chosen["final_fit"]["noise"] == chosen["noise"]


def test_a_richer_rung_is_never_worse_than_the_rung_it_contains_beyond_the_same_optimum_delta(recipe):
    report, _ = recipe
    a, b = report["rungs"][:2]
    assert b["minus2lnl"] <= a["minus2lnl"] + IS.SAME_DELTA
    assert report["checks"]["nested"]["status"] == "pass", report["checks"]["nested"]["detail"]


def test_the_choice_of_rung_uses_chi2_on_the_tempered_errors_and_the_quoted_chi2_is_reported_beside_it(recipe):
    T = recipe[0]["temper_factor"]
    for r in recipe[0]["rungs"]:
        assert r["chi2_quoted"] > 0 and r["chi2_red"] == pytest.approx(r["chi2_quoted"] / r["n"])
        assert r["bic"] == pytest.approx(r["minus2lnl"] + r["k"] * np.log(r["n"]))
        assert r["minus2lnl"] == pytest.approx(r["chi2_quoted"] / T**2, rel=2e-3)  # the ladder's errors are the quoted ones times T


def test_the_checks_are_assembled_into_pass_warn_fail_and_the_fit_standard_follows_the_residual_rms(recipe):
    report, _ = recipe
    checks, stats = report["checks"], report["fit_standard_numbers"]
    assert all(c["status"] in ("pass", "warn", "fail") for c in checks.values())
    assert checks["family_complete"]["status"] == "warn", "no image was fitted on top: a missing measurement, not a pass"
    assert checks["extent_known"]["status"] == "pass"
    # the standard is judged at the tempered errors; the rms at the quoted errors and at the fitted error terms are reported beside it
    assert set(stats["at_quoted_errors"]) >= {"rms_v2", "rms_cp"} and set(stats["at_fitted_error_terms"]) >= {"rms_v2", "rms_cp"}
    assert stats["at_quoted_errors"]["rms_v2"] == pytest.approx(stats["rms_v2"] * report["temper_factor"], rel=1e-3)
    ok, _ = IS.fit_standard(stats["rms_v2"], stats["rms_cp"], stats["n_informative_triangles"])
    assert (checks["fit_standard"]["status"] == "pass") == ok and checks["fit_standard"]["detail"]["verdict"] == ("complete" if ok else "incomplete")
    assert stats["n_informative_triangles"] > 0 and stats["rms_cp"] is not None
    # start agreement is reported for every rung and is the chosen rung's in the checks
    assert set(report["start_agreement"]) == {r["rung"] for r in report["rungs"]}
    assert checks["starts_agree"]["detail"]["passes"] == report["start_agreement"][report["chosen"]]["passes"]


# ---------------------------------------------------------------------------------------------------------------
# The ladder's policy without fits: which starts a rung gets (fitc is replaced by a function that records its start)


def fake_ladder(monkeypatch, peaks):
    """`fit_ladder` on the tiny data with `fitc` returning the start as the fit and `map_seeds` the given ``peaks``."""
    seen = []

    def fitc(tag, rung, priors, data, init, ctx, noise=None):
        seen.append({"tag": tag, "noise": noise, "init": dict(init)})
        return {"values": {k: float(v) for k, v in init.items()}, "chi2": float(ctx["datasets"][0].n_independent), "converged": True, "at_bound": []}

    monkeypatch.setattr(C, "fitc", fitc)
    monkeypatch.setattr(C, "map_seeds", lambda comp, rung, base, below, carry, ctx: (peaks, {"component": comp, "peaks": [], "seeds": peaks}))
    return seen


def test_a_higher_rung_starts_at_map_peaks_and_narrow_jitters_of_the_best_peak_and_inherits_the_rung_below(monkeypatch):
    ctx = context([contest_data()], budget=4, n_peaks=2, n_jitter=3)
    rungs = dict(C.ladder("star_with_faint_companion"))
    ladder = [(n, rungs[n]) for n in ("star", "star+halo", "star+halo+companion")]
    peaks = [{"companion.dra": 1.5, "companion.ddec": -1.0, "companion.flux": 0.02, "score": 9.0},
             {"companion.dra": -2.0, "companion.ddec": 2.5, "companion.flux": 0.01, "score": 4.0}]
    seen = fake_ladder(monkeypatch, peaks)
    got = C.fit_ladder(ladder, ctx, np.random.default_rng(0))
    assert all(s["noise"] is None for s in seen), "no error term is free anywhere in the ladder"
    assert ctx["temper"] == 1.0 and len(ctx["tempered"]) == 1
    zero, halo, companion = got
    assert zero["starts_by_origin"]["moments"] == 1 and 1 <= zero["starts_by_origin"]["grid"] <= 4 and set(zero["starts_by_origin"]) == {"moments", "grid"}
    assert halo["starts_by_origin"] == {"below": 1, "jitter": 3}, "a rung that places nothing: the rung below and budget - 1 jitters of it"
    assert companion["starts_by_origin"] == {"map": 2, "jitter": 3}, "no start from the rung below alone, none at random"
    runs = {origin: [r["values"] for r in companion["start_runs"] if r["origin"] == origin] for origin in ("map", "jitter")}
    assert {(round(v["companion.dra"], 6), round(v["companion.ddec"], 6)) for v in runs["map"]} == {(1.5, -1.0), (-2.0, 2.5)}
    carry = halo["values"]
    for v in runs["jitter"]:
        assert (round(v["companion.dra"], 6), round(v["companion.ddec"], 6)) == (1.5, -1.0)  # at the best peak
        assert abs(v["star.diam"] / carry["star.diam"] - 1.0) <= C.JITTER["size"] + 1e-9  # the inherited values moved narrowly
        assert abs(v["halo.sigma"] / carry["halo.sigma"] - 1.0) <= C.JITTER["size"] + 1e-9
    assert any(v["star.diam"] != carry["star.diam"] for v in runs["jitter"])
    for v in runs["map"]:
        assert v["star.diam"] == pytest.approx(carry["star.diam"]) and v["halo.sigma"] == pytest.approx(carry["halo.sigma"])


def test_a_ladder_with_no_map_peak_starts_the_new_companion_from_the_rung_below_and_jitters(monkeypatch):
    ctx = context([contest_data()], budget=3, n_jitter=2)
    ladder = C.ladder("star_with_faint_companion")[:3]
    seen = fake_ladder(monkeypatch, [])
    got = C.fit_ladder(ladder, ctx, np.random.default_rng(0))
    assert got[-1]["starts_by_origin"] == {"below": 1, "jitter": 2} and seen


def test_the_error_terms_enter_through_the_final_fit_only(monkeypatch):
    ctx = context([contest_data()])
    ladder = C.ladder("star_with_faint_companion")[:2]
    seen = []

    def fitc(tag, rung, priors, data, init, ctx, noise=None):
        seen.append((tag, noise, dict(init)))
        return {"minus2lnl": 1.0, "values": {}, "noise": {}}

    monkeypatch.setattr(C, "fitc", fitc)
    rungs = [{"values": {"star.diam": 0.5}}, {"values": {"star.diam": 0.5, "halo.sigma": 4.0, "halo.flux": 0.3}}]
    C.final_fit(ladder, rungs, ctx, 1)
    (tag, noise, init), = seen
    assert tag.endswith("|final") and set(noise) == {"vis_error_rel", "phi_error"}
    assert set(init) >= {"noise.vis_error_rel", "noise.phi_error", "star.diam", "halo.sigma"}


# ---------------------------------------------------------------------------------------------------------------
# The recipe's pieces


def test_the_information_check_counts_the_informative_triangles_before_any_fit():
    record, masks = C.information([contest_data()])
    assert record["closure_phases_informative"] and masks[0].size == 24 and masks[0].all()
    # quoted errors as large as the signal: no triangle is informative, and the fit is judged on V^2 alone
    d = contest_data()
    noisy = eqx.tree_at(lambda x: x.d_vis, d, np.full(np.size(d.vis), 5.0))
    record, masks = C.information([noisy])
    assert not masks[0].any() and not record["closure_phases_informative"] and "V2 only" in record["judged_on"]
    # V^2 only: there are no triangles at all
    record, masks = C.information([synthetic(vm.System(star=vm.PointSource()), n=20)])
    assert masks[0].size == 0 and not record["closure_phases_informative"] and record["median_quoted_cp_error_deg"] is None


def test_the_gain_is_a_marginalised_nuisance_of_width_a_tenth_and_absorbs_a_calibration_scale():
    d = contest_data()
    g = C.with_gain(d)
    assert g.gains is not None and np.asarray(g.gains.widths) == pytest.approx([C.GAIN_WIDTH])
    truth = C.model_of(dict(C.ladder("star_with_faint_companion"))["star+halo+companion"], {
        "star.diam": 1.0, "halo.sigma": 4.0, "halo.flux": 1e-4, "companion.flux": 0.05, "companion.dra": 6.0, "companion.ddec": -4.0})
    off = eqx.tree_at(lambda x: x.vis, d, d.vis * 1.1)  # the data calibrated 10 % high
    plain = float(np.sum(np.asarray(C.likelihood.whitened_residuals(truth, off)) ** 2))
    gained = float(np.sum(np.asarray(C.likelihood.whitened_residuals(truth, C.with_gain(off))) ** 2))
    assert gained < 0.5 * plain, "a global scale is marginalised, not left to bias the fit"
    exact = float(np.sum(np.asarray(C.likelihood.whitened_residuals(truth, C.with_gain(d))) ** 2))
    assert exact < 3.0 * len(np.ravel(d.vis)) + len(np.ravel(d.phi))  # and nothing is lost when there is no scale error


def test_the_cache_keeps_a_result_and_a_time_budget_stops_new_work(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return {"x": 1.5}

    cache = C.Cache(tmp_path / "c")
    assert cache.memo("fit", ("a", 1), compute) == cache.memo("fit", ("a", 1), compute) == {"x": 1.5} and len(calls) == 1
    cache.memo("fit", ("a", 2), compute)
    assert len(calls) == 2
    assert np.array_equal(cache.block("tile_0", lambda: np.arange(3)), cache.block("tile_0", lambda: np.zeros(3)))
    got = cache.arrays("probe_0", lambda: {"components": np.ones(2), "stop": "done"})
    assert str(cache.arrays("probe_0", lambda: pytest.fail("recomputed"))["stop"]) == "done" and np.array_equal(got["components"], np.ones(2))
    # a budget already spent: work that is stored is still returned, new work raises Pause
    spent = C.Cache(tmp_path / "c", budget=-1.0)
    assert spent.memo("fit", ("a", 1), compute) == {"x": 1.5}
    with pytest.raises(C.Pause):
        spent.memo("fit", ("a", 3), compute)
    with pytest.raises(C.Pause):
        C.Cache(None, budget=-1.0).memo("fit", ("a",), compute)
    assert C.Cache(None).memo("fit", ("a",), compute) == {"x": 1.5}  # no path, no budget: nothing stored, nothing pauses


def test_a_map_is_grey_in_the_new_part_and_its_values_come_from_the_chromatic_form():
    assert C.grey_values({"a.flux.ratio": 0.2, "a.flux.index": 1.0, "a.diam": 0.5, "b.flux": 3.0}) == {"a.flux": 0.2, "a.diam": 0.5, "b.flux": 3.0}


def test_a_primary_with_a_size_of_its_own_is_resolved_and_a_point_source_is_not():
    assert C.resolved_primary(dict(C.ladder("pinwheel_arcs"))["star+envelope"]) is False  # a point-source reference
    assert C.resolved_primary(dict(C.ladder("ringed_disk_with_companions"))["star+one ring"]) is False  # the reference is the star
    assert C.resolved_primary(dict(C.ladder("thin_disk"))["disk"]) is True


def test_a_jitter_moves_sizes_fluxes_angles_and_positions_narrowly_and_deterministically():
    kinds = {"s.diam": "size", "c.flux": "planet", "d.pa": "angle", "c.dra": "far", "c.ddec": "near", "s.c1": "harmonic", "c.flux.index": "colour"}
    values = {"s.diam": 2.0, "c.flux": 0.05, "d.pa": 40.0, "c.dra": 3.0, "c.ddec": -1.0, "s.c1": 0.1, "c.flux.index": 0.5}
    res = 0.5
    for seed in range(20):
        j = C.jittered(values, kinds, res, np.random.default_rng(seed))
        assert abs(j["s.diam"] / 2.0 - 1.0) <= C.JITTER["size"] and abs(j["c.flux"] / 0.05 - 1.0) <= C.JITTER["size"]
        assert abs(j["d.pa"] - 40.0) <= C.JITTER["angle"] and abs(j["c.flux.index"] - 0.5) <= C.JITTER["colour"]
        assert abs(j["c.dra"] - 3.0) <= C.JITTER["position"] * res and abs(j["c.ddec"] + 1.0) <= C.JITTER["position"] * res
        assert abs(j["s.c1"] - 0.1) <= 0.1
    assert C.jittered(values, kinds, res, np.random.default_rng(1)) == C.jittered(values, kinds, res, np.random.default_rng(1))


def test_the_priors_of_a_thin_part_a_span_a_spiral_and_its_arms_come_from_the_scales():
    sc = small_scales()
    thin, span = C.prior_of("thin", sc), C.prior_of("span", sc)
    assert float(thin.low) == pytest.approx(sc.resolution / 5.0 / sc.extent) and float(thin.high) == 1.0
    assert float(span.low) == pytest.approx(-sc.extent / 2.0) and float(span.high) == pytest.approx(sc.extent / 2.0)
    turns, arm = C.prior_of("turns", sc), C.prior_of("arm_width", sc)
    assert (float(turns.low), float(turns.high)) == (0.5, C.TURNS_MAX)
    assert float(arm.low) == pytest.approx(sc.resolution) and float(arm.high) == pytest.approx(sc.extent / 4.0)


def test_the_residual_rms_is_one_for_the_truth_and_large_for_a_model_that_misses_the_companion(monkeypatch):
    monkeypatch.setattr(C, "FAMILIES", {"tiny": TINY})
    d = contest_data()
    _, masks = C.information([d])
    gained = [C.with_gain(d)]
    rungs = dict(C.ladder("tiny"))
    good = C.residual_stats(C.model_of(rungs["star+companion"], TRUTH), gained, {}, masks)
    bad = C.residual_stats(C.model_of(rungs["star"], {"star.diam": 1.0}), gained, {}, masks)
    assert good["n_v2"] == 36 and good["n_informative_triangles"] == good["n_triangles"] == 24
    assert good["rms_v2"] < IS.FIT_RMS_MAX and good["rms_cp"] < IS.FIT_RMS_MAX
    assert bad["rms_v2"] > IS.FIT_RMS_MAX and bad["rms_cp"] > IS.FIT_RMS_MAX
    assert IS.fit_standard(good["rms_v2"], good["rms_cp"], good["n_informative_triangles"])[1]["verdict"] == "complete"
    assert IS.fit_standard(bad["rms_v2"], bad["rms_cp"], bad["n_informative_triangles"])[1]["verdict"] == "incomplete"
    # closure phases of uninformative triangles are not judged
    none = C.residual_stats(C.model_of(rungs["star"], {"star.diam": 1.0}), gained, {}, [np.zeros(24, bool)])
    assert none["rms_cp"] is None and none["rms_cp_all_triangles"] > IS.FIT_RMS_MAX


# ---------------------------------------------------------------------------------------------------------------
# Review fixes: what a check may and may not wave through


def hand_made_fit(**stats):
    """The arguments of `checks_of` for one rung fitted to nothing: every measurement is a plain number the test sets."""
    rung = dict(C.ladder("star_with_faint_companion"))["star"]
    chosen = {"rung": "star", "minus2lnl": 0.0, "n": 100, "values": {}, "bounds": {}, "converged": True}
    agreement = {"star": {"passes": True, "at_best": 6, "n": 8}}
    numbers = {"rms_v2": 1.0, "rms_cp": None, "n_informative_triangles": 0, **stats}
    return rung, chosen, {"values": {}, "converged": True}, agreement, numbers, IS.scales_of(1.0 / (2.0 * 1.0 * IS.MAS), 1.0 / (50.0 * IS.MAS), None)


def test_family_complete_fails_when_the_image_step_errored_and_warns_only_when_it_was_skipped():
    rung, chosen, final, agreement, stats, sc = hand_made_fit()

    def run(rest):
        return C.checks_of([chosen], chosen, [("star", rung)], agreement, stats, rest, sc, final)["family_complete"]

    assert run({"error": "RuntimeError: boom"})["status"] == "fail"
    assert run(None)["status"] == "warn"
    assert run({"flux_relative_to_model": 0.05})["status"] == "pass"
    assert run({"flux_relative_to_model": 2.0})["status"] == "fail"


def test_a_fit_that_misses_the_standard_is_incomplete_and_a_rung_whose_starts_disagree_fails():
    rung, chosen, final, agreement, stats, sc = hand_made_fit()

    def checks(a=agreement, **numbers):
        return C.checks_of([chosen], chosen, [("star", rung)], a, {**stats, **numbers}, None, sc, final)

    assert checks()["fit_standard"]["status"] == "pass" and checks()["fit_standard"]["detail"]["verdict"] == "complete"
    bad = checks(rms_v2=3.0)["fit_standard"]
    assert bad["status"] == "fail" and bad["detail"]["verdict"] == "incomplete" and "V2" in bad["detail"]["over_the_limit"]
    cp = checks(rms_cp=2.5, n_informative_triangles=12)["fit_standard"]
    assert cp["status"] == "fail" and "closure phase (informative triangles)" in cp["detail"]["over_the_limit"]
    assert checks(rms_v2=None)["fit_standard"]["status"] == "fail"  # a missing rms is never a pass
    assert checks({"star": {**agreement["star"], "passes": False, "at_best": 3}})["starts_agree"]["status"] == "fail"
    assert checks()["starts_agree"]["status"] == "pass"


def test_nested_in_the_checks_allows_the_same_optimum_delta():
    rung, chosen, final, agreement, stats, sc = hand_made_fit()
    richer = dict(C.ladder("star_with_faint_companion"))["star+halo"]
    below = {**chosen, "minus2lnl": 100.0}

    def nested(delta):
        above = {**chosen, "rung": "star+halo", "minus2lnl": 100.0 + delta, "converged": True}
        return C.checks_of([below, above], below, [("star", rung), ("star+halo", richer)], {"star": agreement["star"]}, stats, None, sc, final)["nested"]

    assert nested(0.9 * IS.SAME_DELTA)["status"] == "pass"
    assert nested(1.5 * IS.SAME_DELTA)["status"] == "fail"


def test_a_chromatic_rung_reports_the_start_statistics_of_its_own_stage(monkeypatch):
    # three of four chromatic starts agree on the optimum; the grey stage's own counts must not leak into the result
    data = synthetic(vm.System(star=vm.PointSource(), halo=vm.GaussianDisk(sigma=5.0, flux=0.5)), n=40)
    sc, _, datasets = scales_for([data])
    rung = dict(C.ladder("star_with_faint_companion"))["star+halo"]
    grey = {"values": {"star.diam": 0.5, "halo.sigma": 4.0, "halo.flux": 0.4}, "noise": {"noise.vis_error_rel": 0.1}}
    below = {"values": {"star.diam": 0.5}, "noise": {"noise.vis_error_rel": 0.3}}
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
    # three of the four finish within SAME_DELTA of the best (-2 ln L): the start counts that start agreement is made from
    at_best = [m for m in got["minus2lnl_of_refit_starts"] if abs(m - got["minus2lnl"]) <= IS.SAME_DELTA]
    assert len(at_best) == 3


def test_a_run_without_a_pinned_criteria_commit_is_refused_before_any_data_is_read(tmp_path, monkeypatch):
    def refuse(path, root=None):
        raise C.preregistration.CriteriaError("design/contest_informed_criteria.md: not a git checkout and CRITERIA_COMMIT is not set")

    monkeypatch.setattr(C.preregistration, "criteria_commit", refuse)
    monkeypatch.setattr(C, "load_datasets", lambda *a, **k: pytest.fail("data were read before the criteria were pinned"))
    with pytest.raises(C.preregistration.CriteriaError):
        C.run(0, "star_with_faint_companion", tmp_path, tmp_path, smoke=True)


# ---------------------------------------------------------------------------------------------------------------
# Review fixes: rings, embedding the rung below, seeds, noise of several datasets, the chromatic fallback


def small_scales():
    return IS.scales_of(1.0 / (2.0 * 2.0 * IS.MAS), 1.0 / (60.0 * IS.MAS), None)


def test_a_ring_width_prior_reaches_below_the_collapse_limit_so_a_collapsed_ring_is_inside_the_prior():
    sc = small_scales()
    width, sigma = C.prior_of("width", sc), C.prior_of("ring_width", sc)
    assert float(width.low) == pytest.approx(IS.WIDTH_FLOOR * sc.resolution)
    assert float(width.low) < IS.RING_FRACTION * sc.resolution
    assert float(sigma.low) * IS.FWHM_PER_SIGMA == pytest.approx(float(width.low))  # the same bound, as a sigma
    rung = dict(C.ladder("ringed_disk_with_companions"))["star+one ring"]
    kinds = C.flat_kinds(rung)
    assert kinds["disk.profile.a.sigma"] == "ring_width"
    collapsed = {"disk.profile.a.sigma": 0.1 * sc.resolution / IS.FWHM_PER_SIGMA}  # a FWHM of a tenth of the resolution
    resolved = {"disk.profile.a.sigma": 0.5 * sc.resolution / IS.FWHM_PER_SIGMA}
    assert not IS.no_collapse(C.collapse_components(rung, collapsed, kinds), sc.resolution)[0]
    assert IS.no_collapse(C.collapse_components(rung, resolved, kinds), sc.resolution)[0]


def test_the_stratified_starts_scan_the_radius_of_every_ring_of_a_ringed_disk():
    sc = small_scales()
    rung = dict(C.ladder("ringed_disk_with_companions"))["star+two rings"]
    kinds, priors = C.flat_kinds(rung), C.build_priors(rung, sc)
    centre = C.centre_of(rung, kinds, priors, sc, None)
    moved = {p for s in C.stratified_starts(rung, kinds, priors, sc, None, centre) for p in kinds if s[p] != centre[p]}
    assert {"disk.profile.a.radius", "disk.profile.b.radius"} <= moved


def test_a_ring_added_to_a_disk_starts_at_negligible_weight_and_a_star_added_to_a_disk_makes_the_disk_the_whole_object():
    sc = small_scales()
    rungs = dict(C.ladder("ringed_disk_with_companions"))
    rung, below = rungs["star+two rings"], rungs["star+one ring"]
    start = C.embed_below(rung, below, C.flat_kinds(rung), C.build_priors(rung, sc), {})
    assert start["disk.profile.b.weight"] == pytest.approx(C.NEGLIGIBLE) and "disk.flux" not in start  # the disk keeps its flux
    thin = dict(C.ladder("thin_disk"))
    rung, below = thin["star+disk"], thin["disk"]
    start = C.embed_below(rung, below, rung.kinds(), C.build_priors(rung, sc), {})
    assert start["disk.flux"] > 1e3  # the star is the negligible part: the disk's flux relative to it is at the top of its prior


def test_a_map_seed_is_moved_inside_the_prior():
    sc = small_scales()
    rung = dict(C.ladder("star_with_faint_companion"))["star+halo+companion"]
    kinds, priors = C.flat_kinds(rung), C.build_priors(rung, sc)
    seed = {"companion.dra": sc.reach, "companion.ddec": -sc.reach, "companion.flux": 5.0}  # on the edge of the map, over the planet's cap
    start = C.seeded({}, seed, priors, kinds)
    for path, (lo, hi, _, _) in C.bounds_of(priors, kinds).items():
        if path in start:
            assert lo < start[path] < hi, path
    assert start["companion.flux"] < 0.1


def test_the_fitted_noise_of_several_datasets_is_held_per_dataset():
    a = synthetic(vm.System(star=vm.PointSource()), n=20)
    b = synthetic(vm.System(star=vm.PointSource()), n=20, seed=1)
    fixed = C.fixed_noise({"noise[0].vis_error_rel": 0.1, "noise[1].vis_error_rel": 0.2}, [a, b])
    assert isinstance(fixed, list) and len(fixed) == 2
    assert fixed[0]["vis_error_rel"](None) == 0.1 and fixed[1]["vis_error_rel"](None) == 0.2
    one = C.fixed_noise({"noise.vis_error_rel": 0.3}, [a])
    assert one["vis_error_rel"](None) == 0.3


def test_a_grey_fit_kept_in_a_chromatic_run_is_written_in_the_chromatic_form():
    sc = small_scales()
    rung = dict(C.ladder("star_with_faint_companion"))["star+halo"]
    grey = {"values": {"star.diam": 0.5, "halo.sigma": 4.0, "halo.flux": 0.4}}
    got = C.as_chromatic(grey, rung, sc, 1.65e-6)
    assert got["values"]["halo.flux.ratio"] == 0.4 and got["values"]["halo.flux.index"] == 0.0 and "halo.flux" not in got["values"]
    assert "halo.flux.ratio" in got["bounds"] and "halo.flux" not in got["bounds"] and got["chromatic_failed"]


def test_the_chromatic_below_start_carries_the_noise_of_the_rung_below(monkeypatch):
    data = synthetic(vm.System(star=vm.PointSource(), halo=vm.GaussianDisk(sigma=5.0, flux=0.5)), n=40)
    sc, _, datasets = scales_for([data])
    ladder = dict(C.ladder("star_with_faint_companion"))
    seen = {}

    def fit_start(rung, priors, data, init, *args, **kwargs):
        seen.setdefault("inits", []).append(init)
        return None, "stop"

    monkeypatch.setattr(C, "fit_start", fit_start)
    grey = {"values": {"star.diam": 0.5, "halo.sigma": 4.0, "halo.flux": 0.4}, "noise": {"noise.vis_error_rel": 0.1}}
    below = {"values": {"star.diam": 0.5}, "noise": {"noise.vis_error_rel": 0.3}}
    C.chromatic_rung(grey, ladder["star+halo"], ladder["star"], below, {"datasets": datasets, "scales": sc, "max_steps": 5}, 1.65e-6)
    noises = [i["noise.vis_error_rel"] for i in seen["inits"]]
    assert noises == [0.1, 0.3, 0.1, 0.1]  # grey_best, below, index+1, index-1
    assert seen["inits"][1]["halo.flux.ratio"] == pytest.approx(C.NEGLIGIBLE)  # the new halo at negligible flux


def test_the_fit_standard_is_reported_and_not_a_gate_while_every_gating_check_is_produced():
    import yaml

    mandate = yaml.safe_load((pathlib.Path(__file__).parents[1] / "campaigns" / "contests" / "mandate.yaml").read_text())
    requires = set(mandate["stages"]["informed_fit"]["requires"])
    rung, chosen, final, agreement, stats, sc = hand_made_fit(rms_v2=5.0)  # incomplete by the fit standard
    checks = C.checks_of([chosen], chosen, [("star", rung)], agreement, stats, None, sc, final)
    assert checks["fit_standard"]["status"] == "fail" and C.REPORT_ONLY == ("fit_standard",)
    assert "fit_standard" not in requires and requires <= set(checks), "mandate gates only on checks the script produces"
