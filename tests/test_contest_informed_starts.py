"""The scales, moment fit, stratified starts, tiled maps and checks of the informed contest fits.

Everything here is NumPy: ``crosscheck.informed_starts`` does not import virgil. The data are synthetic: a known
source observed on a made-up uv coverage (baselines from 10 m to 330 m, many position angles).
"""

import numpy as np
import pytest

from crosscheck import informed_starts as S

pytestmark = pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")

WAVEL = 1.65e-6


def coverage(n=400, b_min=10.0, b_max=330.0, seed=0):
    """u, v in cycles per radian: baselines drawn log-uniformly in length, uniformly in angle over half a circle."""
    rng = np.random.default_rng(seed)
    b = np.exp(rng.uniform(np.log(b_min), np.log(b_max), n))
    theta = rng.uniform(0.0, np.pi, n)
    return b * np.sin(theta) / WAVEL, b * np.cos(theta) / WAVEL


def observe(u, v, compact, extended, fwhm, ratio, pa, sigma=0.01, seed=1, amplitude=1.0):
    v2 = S.moment_model_v2(u, v, compact, extended, fwhm, ratio, pa, amplitude)
    return v2 + np.random.default_rng(seed).normal(0.0, sigma, v2.size), np.full(v2.size, sigma)


def angular_difference(a, b, period=180.0):
    d = abs(a - b) % period
    return min(d, period - d)


def test_the_gaussian_visibility_is_the_fourier_transform_of_a_gaussian():
    # FWHM theta: V = exp(-pi^2 theta^2 q^2 / (4 ln 2)); at q = 1 / theta it is exp(-pi^2 / (4 ln 2)) = 0.0285.
    theta = 5.0
    q = 1.0 / (theta * S.MAS)
    assert S.gaussian_visibility(q, 0.0, theta, 1.0, 0.0) == pytest.approx(np.exp(-np.pi**2 / (4 * np.log(2))))
    # the major axis lies along the position angle (East of North): a source elongated North-South (PA 0) is
    # narrow in v and wide in u, so its visibility falls slowly along v ... no: it is wide in the sky along v, so
    # it is resolved sooner along v.
    along_major = S.gaussian_visibility(0.0, q, theta, 0.4, 0.0)
    along_minor = S.gaussian_visibility(q, 0.0, theta, 0.4, 0.0)
    assert along_major < along_minor


def test_the_moment_fit_recovers_a_synthetic_ellipse():
    u, v = coverage()
    truth = dict(compact=0.2, extended=0.7, fwhm=6.0, ratio=0.5, pa=60.0)  # 0.1 over-resolved
    v2, err = observe(u, v, **truth)
    fit = S.moment_fit(u, v, v2, err, theta_range=(0.2, 60.0))
    assert fit["fwhm"] == pytest.approx(truth["fwhm"], rel=0.15)
    assert fit["ratio"] == pytest.approx(truth["ratio"], abs=0.1)
    assert angular_difference(fit["pa"], truth["pa"]) < 8.0
    assert fit["compact"] == pytest.approx(truth["compact"], abs=0.06)
    assert fit["resolved"] == pytest.approx(0.1, abs=0.1)
    assert fit["longest_tenth"] == pytest.approx(truth["compact"], abs=0.06)  # the cross-check agrees
    assert not fit["at_upper_edge"] and fit["short_v2"] > 0.03


def test_the_moment_fit_flags_a_source_resolved_on_every_baseline():
    u, v = coverage(b_min=40.0)
    v2, err = observe(u, v, 0.0, 0.3, 80.0, 0.4, 90.0)  # a large source: V^2 ~ 0 on every baseline
    fit = S.moment_fit(u, v, v2, err, theta_range=(0.2, 60.0))
    assert fit["short_v2"] < S.SHORT_V2_MIN or fit["at_upper_edge"]


def test_noise_on_a_source_resolved_everywhere_does_not_pass_for_a_measurement_of_the_extent():
    # quoted sigma 0.012 on ~100 short points: the largest of pure noise is about 0.03, which the old statistic (the
    # largest V^2) took for a signal; V^2 - 3 sigma is about zero
    u, v = coverage(n=800, b_min=40.0)
    v2, err = observe(u, v, 0.0, 0.0, 80.0, 0.4, 90.0, sigma=0.012)  # no light on these baselines at all
    fit = S.moment_fit(u, v, v2, err, theta_range=(0.2, 60.0))
    assert fit["short_v2"] < S.SHORT_V2_MIN
    b = np.hypot(u, v)
    assert not S.scales_of(b.max(), b.min(), fit).extent_known


def test_a_moment_fit_with_no_extended_light_does_not_measure_the_extent():
    u, v = coverage()
    b = np.hypot(u, v)
    fit = {"fwhm": 5.0, "short_v2": 0.5, "at_upper_edge": False, "extended": 0.5 * S.EXTENDED_MIN}  # a compact star, a faint companion
    s = S.scales_of(b.max(), b.min(), fit)
    assert not s.extent_known and s.extent == s.reach and any("extended light" in w for w in s.warnings)
    assert S.scales_of(b.max(), b.min(), {**fit, "extended": 2.0 * S.EXTENDED_MIN}).extent_known


def test_the_tolerance_is_a_percent_of_n_but_never_more_than_ln_n():
    assert S.tolerance(500) == pytest.approx(5.0)  # 1% of N
    assert S.tolerance(20000) == pytest.approx(np.log(20000))  # not 200
    # on a large data set two optima 200 apart are not the same optimum, and a richer rung 200 worse is not nested
    assert not S.starts_agree([1000.0, 1200.0], n=20000)[0]
    ok, bad = S.nested([{"rung": "a", "minus2lnl": 1000.0, "contains": []}, {"rung": "b", "minus2lnl": 1200.0, "contains": ["a"]}], n=20000)
    assert not ok and bad[0]["rung"] == "b"


def test_the_criteria_file_and_the_code_agree_on_every_constant():
    import pathlib
    import re

    text = (pathlib.Path(__file__).resolve().parents[1] / "design" / "contest_informed_criteria.md").read_text()
    table = {m.group(1): float(m.group(2)) for m in re.finditer(r"^\| `([A-Z_0-9]+)` \| ([0-9.]+) \|", text, re.M)}
    assert set(table) == {"BOUND_EDGE", "AGREE_FRACTION", "MIN_AGREE", "FAMILY_FLUX_LIMIT", "RING_FRACTION", "WIDTH_FLOOR",
                          "COMPANION_ELEMENTS", "SHORT_V2_MIN", "EXTENDED_MIN"}
    for name, value in table.items():
        assert getattr(S, name) == pytest.approx(value), name
    assert S.WIDTH_FLOOR * (1.0 + S.BOUND_EDGE) < S.RING_FRACTION  # a collapsed ring is not at the prior's edge


def test_the_moment_fit_finds_an_amplitude_above_one_when_asked():
    u, v = coverage()
    v2, err = observe(u, v, 0.3, 0.7, 4.0, 0.8, 30.0, amplitude=1.14)
    fit = S.moment_fit(u, v, v2, err, theta_range=(0.2, 60.0), fit_amplitude=True)
    assert fit["amplitude"] == pytest.approx(1.14, abs=0.05)
    assert S.moment_fit(u, v, v2, err, theta_range=(0.2, 60.0))["amplitude"] == 1.0


def test_the_moment_fit_needs_enough_points():
    assert S.moment_fit([1.0] * 3, [0.0] * 3, [0.9] * 3, [0.01] * 3, (0.2, 60.0)) is None


def test_the_linear_second_moments_agree_with_the_moment_fit_on_a_resolved_ellipse():
    u, v = coverage(b_max=60.0)
    v2, err = observe(u, v, 0.0, 1.0, 8.0, 0.6, 120.0, sigma=0.005)
    m = S.second_moments(u, v, v2, err, v2_min=0.2)
    # a Gaussian of FWHM theta has sigma = theta / 2.355
    assert m["sigma_major"] * S.FWHM_PER_SIGMA == pytest.approx(8.0, rel=0.1)
    assert m["ratio"] == pytest.approx(0.6, abs=0.1)
    assert angular_difference(m["pa"], 120.0) < 8.0


def test_the_three_scales_and_their_sources():
    u, v = coverage()
    b = np.hypot(u, v)
    fit = {"fwhm": 6.0, "short_v2": 0.5, "at_upper_edge": False}
    s = S.scales_of(b.max(), b.min(), fit)
    assert s.resolution == pytest.approx(0.5 / b.max() / S.MAS)
    assert s.extent == pytest.approx(max(4 * 6.0, 8 * s.resolution)) and s.extent_known and "moment" in s.extent_source
    assert s.reach == pytest.approx(3.0 / b.min() / S.MAS) and "lambda / B_min" in s.reach_source
    assert s.pixel == pytest.approx(0.25 / b.max() / S.MAS) or s.npix == S.MAX_PIXELS
    # a given field decides the extent; a tapered field halves the reach
    g = S.scales_of(b.max(), b.min(), fit, field=30.0, field_source="readme", tapered=True)
    assert g.extent == 30.0 and g.extent_known and g.reach == 15.0 and "half the given field" in g.reach_source
    assert S.scales_of(b.max(), b.min(), fit, field=106.0, field_source="clue").reach == pytest.approx(s.reach)
    # a small fit still gets at least eight resolution elements
    tiny = S.scales_of(b.max(), b.min(), {"fwhm": 0.1, "short_v2": 0.5, "at_upper_edge": False})
    assert tiny.extent == pytest.approx(8 * tiny.resolution)


def test_an_unconstrained_extent_is_the_reach_and_says_so():
    u, v = coverage()
    b = np.hypot(u, v)
    s = S.scales_of(b.max(), b.min(), {"fwhm": 50.0, "short_v2": 0.01, "at_upper_edge": False})
    assert not s.extent_known and s.extent == s.reach and any("extent_unconstrained" in w for w in s.warnings)
    # ... unless the contest gave a field
    assert S.scales_of(b.max(), b.min(), {"fwhm": 50.0, "short_v2": 0.01, "at_upper_edge": True}, field=106.0, field_source="x").extent_known


def test_a_field_needing_more_than_128_pixels_is_coarsened_with_a_warning():
    s = S.scales_of(1.0 / (2 * 1.0 * S.MAS), 1.0 / (3000.0 * S.MAS), field=500.0, field_source="x")  # resolution 1 mas, field 500
    assert s.npix == S.MAX_PIXELS and s.npix_wanted > S.MAX_PIXELS
    assert s.pixel == pytest.approx(500.0 / S.MAX_PIXELS) and any("coarsened" in w for w in s.warnings)
    assert "coarsened" in s.pixel_source


def test_stratified_geometry():
    sizes = S.log_steps(10.0)
    assert len(sizes) == 8 and sizes[0] == pytest.approx(5.0) and sizes[-1] == pytest.approx(20.0)
    assert np.allclose(np.diff(np.log(sizes)), np.diff(np.log(sizes))[0])
    grid = S.inc_pa_grid(40.0, 33.0)
    assert grid[0] == (40.0, 33.0) and len(grid) == 1 + 4 * 6
    assert {p for _, p in grid[1:]} == set(S.angle_steps(6)) and max(i for i, _ in grid) <= 85.0
    assert S.angle_steps(12, 360.0)[:3] == [0.0, 30.0, 60.0]
    assert S.inside(0.0, 0.0, 1.0) == pytest.approx(0.01) and S.inside(1.0, 0.0, 1.0) == pytest.approx(0.99)
    assert S.inside(1e-9, 1e-4, 1e4, log=True) > 1e-4 and S.inside(1e9, 1e-4, 1e4, log=True) < 1e4
    assert S.inside(3.0, 0.0, 10.0) == 3.0


def test_the_tiled_map_finds_a_faint_peak_far_outside_the_old_121_point_grid():
    # the old map had at most 121 points a side, so a step of 2 half / 120; with half = 300 mas and a resolution of
    # 0.4 mas that is 5 mas: a peak a fraction of a resolution element wide is missed. The tile map steps by half the
    # resolution over the whole reach and reports it.
    resolution, half = 0.4, 300.0
    target = (-217.3, 143.1)

    def evaluate(dra, ddec):  # a peak as wide as a resolution element, on a flat floor
        d2 = (dra[:, None] - target[0]) ** 2 + (ddec[None, :] - target[1]) ** 2
        return 1.0e-3 * np.exp(-0.5 * d2 / (0.5 * resolution) ** 2)

    old_axis = np.linspace(-half, half, 121)
    assert max(float(np.max(evaluate(old_axis, old_axis))), 0.0) < 1e-6  # the old grid misses it
    axis, surface, step = S.tiled_map(evaluate, half, 0.5 * resolution, tile=200, max_axis=4000)
    assert step == pytest.approx(0.5 * resolution, rel=0.01) and surface.shape == (axis.size, axis.size)
    peak = S.map_peaks(axis, surface, 3, 2 * resolution)[0]
    assert np.hypot(peak[0] - target[0], peak[1] - target[1]) < 0.5 * resolution
    # tiles do not change the answer
    _, whole, _ = S.tiled_map(evaluate, half, 0.5 * resolution, tile=10**6, max_axis=4000)
    assert np.allclose(surface, whole)


def test_a_map_coarsens_and_says_so_beyond_its_axis_cap():
    axis, surface, step = S.tiled_map(lambda a, b: np.zeros((a.size, b.size)), 100.0, 0.001, tile=64, max_axis=300)
    assert axis.size == 300 and step == pytest.approx(200.0 / 299)


def test_map_peaks_are_separated_and_leave_out_the_origin():
    axis = np.linspace(-10, 10, 201)
    dra, ddec = np.meshgrid(axis, axis, indexing="ij")
    surface = 5.0 * np.exp(-0.5 * (dra**2 + ddec**2)) + 2.0 * np.exp(-0.5 * ((dra - 6) ** 2 + (ddec + 4) ** 2))
    peaks = S.map_peaks(axis, surface, 4, separation=2.0, exclude=1.0)
    assert all(np.hypot(p[0], p[1]) >= 1.0 for p in peaks)
    assert all(np.hypot(a[0] - b[0], a[1] - b[1]) > 2.0 for i, a in enumerate(peaks) for b in peaks[:i])
    assert S.map_peaks(axis, surface, 1, 2.0)[0][:2] == pytest.approx((0.0, 0.0))
    assert np.hypot(peaks[0][0] - 6, peaks[0][1] + 4) < 1.5 or np.hypot(peaks[0][0], peaks[0][1]) < 2.0


# ---- the checks: each fires on a constructed failing case and passes on a good one -------------------------------


BOUNDS = {"a.diam": (0.1, 10.0, True, "size"), "a.ratio": (0.3, 1.0, False, "ratio"), "a.pa": (0.0, 180.0, False, "angle"),
          "b.flux": (1e-4, 1e4, True, "flux")}


def test_no_bounds_fires_on_a_parameter_at_a_bound_and_passes_in_the_interior():
    ok, hits = S.no_bounds({"a.diam": 1.0, "a.ratio": 0.7, "a.pa": 0.0, "b.flux": 3.0}, BOUNDS)
    assert ok and hits == []  # an angle at 0 wraps: not a bound
    for values, path in (({"a.diam": 0.1004, "a.ratio": 0.7, "b.flux": 3.0}, "a.diam"),  # log scale: 0.1004 is within 1%
                         ({"a.diam": 1.0, "a.ratio": 0.998, "b.flux": 3.0}, "a.ratio"),
                         ({"a.diam": 1.0, "a.ratio": 0.7, "b.flux": 9.9e3}, "b.flux")):
        ok, hits = S.no_bounds(values, BOUNDS)
        assert not ok and [h["parameter"] for h in hits] == [path]
    # in log, 0.12 of [0.1, 10] is at 8% of the range, not at the bound; in linear it would also pass
    assert S.no_bounds({"a.diam": 0.12}, BOUNDS)[0]
    # an unbounded parameter is skipped
    assert S.no_bounds({"x": 3.0}, {"x": (-np.inf, np.inf, False, "harmonic")})[0]


def test_nested_fires_when_a_richer_rung_fits_worse_and_passes_otherwise():
    good = [{"rung": "a", "minus2lnl": 100.0, "contains": []}, {"rung": "ab", "minus2lnl": 80.0, "contains": ["a"]}]
    assert S.nested(good, n=500)[0]
    bad = [{"rung": "a", "minus2lnl": 100.0, "contains": []}, {"rung": "ab", "minus2lnl": 130.0, "contains": ["a"]}]
    ok, detail = S.nested(bad, n=500)
    assert not ok and detail[0]["rung"] == "ab" and detail[0]["contains"] == "a"
    within = [{"rung": "a", "minus2lnl": 100.0, "contains": []}, {"rung": "ab", "minus2lnl": 101.0, "contains": ["a"]}]
    assert S.nested(within, n=500)[0]  # 1 < 1% of 500 = 5: the same optimum
    # a rung that does not contain another is never compared with it
    swap = [{"rung": "a", "minus2lnl": 100.0, "contains": []}, {"rung": "b", "minus2lnl": 300.0, "contains": []}]
    assert S.nested(swap, n=500)[0]


def test_starts_agree_fires_when_one_start_found_the_best():
    ok, detail = S.starts_agree([100.0, 100.2, 300.0, 250.0], n=500)  # tolerance 5
    assert ok and detail == {"at_best": 2, "starts": 4}
    ok, detail = S.starts_agree([100.0, 140.0, 300.0], n=500)
    assert not ok and detail["at_best"] == 1
    assert not S.starts_agree([float("inf"), float("nan")], n=10)[0]
    assert S.starts_agree([-50.0, -49.0], n=100)[0]  # a normalised likelihood can be negative


def test_family_complete_fires_when_the_image_on_top_carries_much_of_the_flux():
    assert S.family_complete(0.05)[0] is True
    assert S.family_complete(2.0)[0] is False  # 2004 data1 in the last run: 2.0 of the model's flux
    assert S.family_complete(0.2)[0] is True and S.family_complete(0.21)[0] is False
    assert S.family_complete(None)[0] is None  # not run: neither a pass nor a failure


def test_no_collapse_fires_on_each_collapsed_component_and_passes_on_good_ones():
    res = 0.8
    good = [{"kind": "arc", "length": 5.0}, {"kind": "ring", "width": 0.5}, {"kind": "companion", "dra": 3.0, "ddec": -1.0}]
    assert S.no_collapse(good, res)[0]
    cases = [{"kind": "arc", "length": 0.5}, {"kind": "ring", "width": 0.8 / 5}, {"kind": "ring", "width": 0.8 / 5 * 1.005},
             {"kind": "companion", "dra": 0.6, "ddec": 0.6}]
    for c in cases:
        ok, bad = S.no_collapse([*good, c], res)
        assert not ok and len(bad) == 1 and bad[0]["kind"] == c["kind"], c
