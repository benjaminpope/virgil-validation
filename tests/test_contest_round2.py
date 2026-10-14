"""Round-2 contest features (contests/fit_plan.yml) in scripts/contest_images.py:
regularisers beside MaxEnt with weights spanning the discrepancy principle's
range, and the "informed" provenance of round-2 outputs. Tiny NumPy/virgil
evaluations only; the fits themselves run on OzSTAR."""

import importlib.util
import pathlib
import types

import numpy as np
import pytest

pytest.importorskip("virgil")

ROOT = pathlib.Path(__file__).parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("contest_images", ROOT / "scripts" / "contest_images.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ci = _load()


def _image(n=16, pixel=1.0):
    import jax.numpy as jnp
    import virgil.models as vm

    y, x = np.mgrid[:n, :n] - n / 2
    return vm.Image(jnp.asarray(-(x**2 + y**2) / 8.0), pixel, flux=1.0)


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
@pytest.mark.parametrize("name", ["tv", "tsv", "starlet", "logsum"])
def test_regulariser_weights_span_the_documented_range(name):
    """Non-MaxEnt weights are virgil.ensemble's per-data-point range x N, 12 log-spaced, strong to weak."""
    from virgil.ensemble import _default_weight_ranges

    low, high = _default_weight_ranges()["starlet" if name == "logsum" else name]
    data = types.SimpleNamespace(n_independent=500)
    weights = ci.regulariser_weights(name, data)
    assert len(weights) == 12
    np.testing.assert_allclose(weights[0], high * 500, rtol=1e-9)
    np.testing.assert_allclose(weights[-1], low * 500, rtol=1e-9)
    assert np.all(np.diff(np.log(weights)) < 0)
    np.testing.assert_allclose(np.diff(np.log(weights)), np.diff(np.log(weights))[0], rtol=1e-9)
    assert len(ci.regulariser_weights(name, data, smoke=True)) == 3


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_mem_keeps_its_weights():
    """MaxEnt keeps the round-1 weights, and the Adam fallback is off for it, so round-1 runs are unchanged."""
    w = ci.regulariser_weights("mem", types.SimpleNamespace(n_independent=500))
    np.testing.assert_array_equal(w, ci.WEIGHTS)


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_informed_provenance():
    assert ci.informed_line(True).startswith("provenance=informed")
    assert ci.informed_line(False) == "provenance=blind"


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_pixel_offset_matches_virgil_image():
    """A one-pixel Image and a PointSource at pixel_offset have the same
    visibilities, so CLEAN clusters start point sources in the right place."""
    import jax.numpy as jnp
    import virgil.models as vm

    n, pixel, row, col = 21, 1.0, 8, 4
    lb = np.full((n, n), -40.0)
    lb[row, col] = 0.0
    image = vm.Image(jnp.asarray(lb), pixel, flux=1.0)
    dra, ddec = ci.pixel_offset(row, col, n, pixel)
    u, v, w = np.array([10.0, 0.0, 7.0]), np.array([0.0, 10.0, -5.0]), np.full(3, 1e-6)
    np.testing.assert_allclose(np.asarray(image.model(u, v, w)),
                               np.asarray(vm.PointSource(1.0, dra=float(dra), ddec=float(ddec)).model(u, v, w)), atol=1e-5)


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_point_candidates_skip_the_main_source():
    """Candidates come from compact clusters outside the main source,
    brightest first; the main source's own components are never offered."""
    n, pixel = 41, 1.0
    beam = types.SimpleNamespace(major_mas=3.0, minor_mas=3.0)
    c = np.zeros((n, n))
    c[18:23, 18:23] = 1.0  # the main source, 25 units
    c[5, 30] = 0.5  # a bright dot
    c[35, 8] = 0.2  # a faint dot
    y, x = np.mgrid[:n, :n]
    kernel = np.exp(-0.5 * ((x - n // 2) ** 2 + (y - n // 2) ** 2) / 1.3**2)
    restored = np.real(np.fft.ifft2(np.fft.fft2(c) * np.fft.fft2(np.fft.ifftshift(kernel))))
    found = ci.clean_point_candidates(c, restored, pixel, beam, 5)
    assert [round(f["flux"], 3) for f in found] == [0.5, 0.2]
    assert found[0]["dra"] == pytest.approx(ci.pixel_offset(5, 30, n, pixel)[0])
    assert found[0]["ddec"] == pytest.approx(ci.pixel_offset(5, 30, n, pixel)[1])


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_suffixes_and_no_systematics():
    assert ci.suffix(3, "gains+block_scales", reg="tv") == "_tv_ps3_gain_bscale"
    assert ci.suffix() == ""
    data = object()
    assert ci.with_systematics(data, None) is data


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_systematics_mapping():
    """Each --systematics choice names its nuisances; widths are scanned, or the median alone."""
    assert set(ci.SYSTEMATICS) == set(ci.SYSTEMATIC_SUFFIX)
    assert ci.SYSTEMATICS["gains+block_scales"] == ("gains", "block_scales")
    assert ci.systematic_widths("none") == {} and ci.systematic_widths("block_scales") == {}
    assert ci.systematic_widths("gains+closure_offsets") == {"gains": [0.01, 0.03, 0.1], "closure": [0.01, 0.03, 0.1]}
    assert ci.systematic_widths("gains+block_scales", scan=False) == {"gains": [0.03]}
    assert len({ci.SYSTEMATIC_SUFFIX[k] for k in ci.SYSTEMATICS}) == len(ci.SYSTEMATICS)


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_rescale_applies_scales_to_each_dataset():
    class D:
        def with_error_scale(self, s):
            return ("scaled", s)

    assert ci.rescale(D(), {"vis": 2.0}) == ("scaled", {"vis": 2.0})
    assert ci.rescale([D(), D()], {"vis": 2.0}) == [("scaled", {"vis": 2.0})] * 2


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_scene_options_map_parts_to_setup():
    o = ci.scene_options("ld_ellipse,point,image")
    assert (o["star"], o["star_model"], o["gauss_halo"], o["tag"]) == (True, "ellipse", False, "_ld_ellipse-point-image")
    assert ci.scene_options("ud,gauss_halo,image")["star_model"] == "disk"
    assert ci.scene_options("ud,gauss_halo,image")["gauss_halo"] is True
    o = ci.scene_options("point,image", hole_mas=2.0)
    assert (o["star"], o["star_model"], o["hole_mas"]) == (True, "point", 2.0)
    assert ci.scene_options("image")["star"] is False
    for bad in ("point", "ud,ld_ellipse,image", "ud,point,image", "disk,image", "point,point,image"):
        with pytest.raises(ValueError):
            ci.scene_options(bad)


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_hole_support():
    x = (np.arange(9) - 4.0) * 0.5
    assert ci.hole_support(x, None) is None
    m = ci.hole_support(x, 1.0)
    assert not m[4, 4] and not m[4, 5] and m[4, 7] and m.sum() == 81 - 9


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_joint_spec_merges_files_in_order():
    bands = [f"2008/2008-Contest2_{b}.oifits" for b in "JHK"]
    by_label = ci.joint_spec(["2008_agn_J", "2008_agn_H", "2008_agn_K"])
    by_index = ci.joint_spec([str(ci.task_index(f"2008_agn_{b}")) for b in "JHK"])
    assert by_label == by_index and by_label["files"] == bands
    assert by_label["label"] == "2008_agn_J+2008_agn_H+2008_agn_K" and by_label["field"] == 30.0
    assert len(ci.TASKS) == 27 and all("JHK" not in t["label"] for t in ci.TASKS)
    with pytest.raises(ValueError):
        ci.joint_spec(["2004_data1", "2008_agn_J"])  # field None against 30 mas


def _plan_module():
    from importlib import util

    spec = util.spec_from_file_location("contest_fit_plan", ROOT / "scripts" / "contest_fit_plan.py")
    fp = util.module_from_spec(spec)
    spec.loader.exec_module(fp)
    return fp


def _round2_lines():
    fp = _plan_module()
    plan = fp.load_plan()
    indices = {t["label"]: i for i, t in enumerate(ci.TASKS)}
    assert fp.check_briefs(plan, indices) == []
    notes = []
    return fp.oz_lines_round2(plan, indices, notes), notes


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_every_round2_line_passes_contest_images_validation():
    """Each generated line parses with contest_images' own parser and passes validate (no data read, no fit)."""
    import shlex

    lines, _ = _round2_lines()
    assert len(lines) > 50
    parser = ci.build_parser()
    for line in lines:
        label, cmd = line.split("\t")
        argv = shlex.split(cmd.replace('"$VV_DATA"', "d").replace('"$OUT"', "o"))
        assert argv[0] == "scripts/contest_images.py"
        args = parser.parse_args(argv[1:])
        try:
            ci.validate(parser, args)
        except SystemExit:  # parser.error
            pytest.fail(f"{label}: contest_images rejects {cmd}")
        assert args.label == label and args.informed


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_round2_labels_are_unique_and_round2():
    lines, _ = _round2_lines()
    labels = [line.split("\t")[0] for line in lines]
    assert len(set(labels)) == len(labels) and all(x.endswith("_r2") for x in labels)


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_incompatible_combinations_are_dropped_with_a_note():
    fp = _plan_module()
    entry = dict(year="1", name="x", task="2004_data1", brief=dict(
        scene=[["ud", "point", "image"], ["gauss_halo", "image"], ["gauss_halo"]], prior=["gp2", "mem"],
        systematics="block_scales", bands="grey", sweep={"off": "test"}))
    plan = dict(datasets=[entry], groups={})
    lines, notes = fp.expand_brief(plan, entry, {"2004_data1": 0})
    argvs = [" ".join(a) for _, a in lines]
    assert not any("ud,point" in a for a in argvs) and not any("--scene gauss_halo " in a for a in argvs)
    assert any("--prior gp2" in a and "gauss_halo" not in a for a in argvs)
    assert not any("--systematics" in a for a in argvs)
    kinds = {k for k, _ in notes}
    assert {"dropped", "needs"} <= kinds


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_sweep_spec_is_virgils_default_apart_from_families():
    from virgil.ensemble import EnsembleSpec

    default, smoke = EnsembleSpec(), EnsembleSpec(**ci.SWEEP_SMOKE, families=("tv", "tsv"))
    assert smoke.weight_ranges == default.weight_ranges and smoke.families == ("tv", "tsv")
    assert smoke.n_weights == 3 and smoke.max_npix < default.max_npix
    with pytest.raises(ValueError):
        EnsembleSpec(families=("logsum",))  # not an ensemble family: --sweep-families rejects it too


class _Reg:
    """A stand-in penalty: value = w * P(model), P = model["p"]."""

    def __init__(self, w):
        self.w = float(w)

    def value(self, model):
        return self.w * model["p"]


def _result(chi2, p, stop="converged", converged=True):
    return types.SimpleNamespace(model={"p": p}, values=None, info={
        "converged": converged, "stop": stop, "chi2": [chi2], "ndata": [10]})


def _curve(rows):
    """rows: (weight, chi2, penalty, stop)."""
    import dataclasses

    @dataclasses.dataclass
    class LC:
        weights: object
        results: list
        chi2: object
        chi2_red: object
        penalty: object

    return LC(weights=np.array([r[0] for r in rows]),
              results=[_result(r[1], r[2], r[3], r[3] == "converged") for r in rows],
              chi2=np.array([r[1] for r in rows]), chi2_red=np.array([[r[1] / 10] for r in rows]),
              penalty=np.array([r[2] for r in rows]))


@pytest.fixture
def fake_fit(monkeypatch):
    refits = {}  # weight -> (chi2, p) of the Adam refit

    monkeypatch.setattr(ci, "regulariser", lambda reg, w, path, q: _Reg(w))
    monkeypatch.setattr(ci, "_reference", lambda model: model)
    monkeypatch.setattr(ci, "n_independent", lambda data: 10)

    def fit(start, priors, data, regs, init=None, method=None, max_steps=None):
        c, p = refits[regs[0].w]
        return _result(c, p)

    monkeypatch.setattr(ci, "fit", fit)
    return refits


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_adam_fallback_keeps_the_lower_objective(fake_fit):
    """A refit replaces a stalled L-BFGS point only if chi2 + w*P is lower: lower chi2 alone (less
    regularised) is not enough; converged points are never refitted."""
    # objective of the L-BFGS fits: 100 + 2*5 = 110, 50 + 1*5 = 55, 20 + 0.5*1 = 20.5
    curve = _curve([(2.0, 100.0, 5.0, "stalled"), (1.0, 50.0, 5.0, "limit"), (0.5, 20.0, 1.0, "converged")])
    fake_fit[2.0] = (90.0, 10.0)  # lower chi2 but objective 90 + 20 = 110 is not lower: stays L-BFGS
    fake_fit[1.0] = (45.0, 4.0)  # objective 45 + 4 = 49 < 55: substituted
    patched, methods, pairs = ci.adam_fallback(curve, None, None, object(), "tv", None, None, [])
    assert methods == ["lbfgs", "adam", "lbfgs"]
    np.testing.assert_allclose(patched.chi2, [100.0, 45.0, 20.0])
    np.testing.assert_allclose(patched.penalty, [5.0, 4.0, 1.0])
    assert pairs == [(10.0, 9.0), (5.0, 4.5), None]  # both chi2/N kept as the diagnostic


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_adam_fallback_is_off_for_maxent(fake_fit):
    curve = _curve([(2.0, 100.0, 5.0, "stalled")])
    patched, methods, pairs = ci.adam_fallback(curve, None, None, object(), "mem", None, None, [])
    assert patched is curve and methods == ["lbfgs"] and pairs == [None]


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
@pytest.mark.parametrize("label,scales,expected", [
    ("2006_nostar_clean", (0.0,), "2006_nostar_cleanfinal"),
    ("2006_star_halo_clean", (0.0,), "2006_star_halo_cleanfinal"),
    ("2006_star_clean", (0.0, 1.0), "2006_star_cleanfinal_cleanms"),
])
def test_clean_label_keeps_star_and_halo_suffixes(label, scales, expected):
    assert ci.clean_label({"label": label}, scales) == expected


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_weight_report_with_a_single_dataset():
    """A one-file task's chi2_red may be 1-D (weights,); the report must not raise."""
    curve = types.SimpleNamespace(chi2_red=np.array([3.0, 1.5, 0.9]), discrepancy=lambda: 0.5, corner=lambda: 1.0)
    rep = ci.weight_report(curve, object(), "tv", None, 1)
    assert rep["binding_dataset"] == 0 and rep["chi2_red_by_dataset"] == [1.5] and rep["n_datasets"] == 1


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_off_bounds_moves_values_off_the_prior_edges():
    """A fit can end on a bound; a start there is infinite in the unconstrained coordinates (NaN fit)."""
    import numpyro.distributions as dist

    priors = {"a.flux": dist.LogUniform(1e-4, 1e3), "a.dra": dist.Uniform(-2.0, 2.0)}
    out = ci.off_bounds({"a.flux": 1e-4, "a.dra": 2.0, "env.log_brightness": np.zeros(3)}, priors)
    assert 1e-4 < float(out["a.flux"]) and -2.0 < float(out["a.dra"]) < 2.0
    assert np.shape(out["env.log_brightness"]) == (3,)


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_sweep_warm_starts_inside_the_bounds_and_skips_non_finite_fits(monkeypatch):
    """Round-2 lines 2004_*_tv_point-gauss_halo-image_r2 failed because the strongest weight ended on a
    bound and seeded the next weight with it. The seed must be interior, and a NaN fit must not seed."""
    import numpyro.distributions as dist

    priors = {"g.flux": dist.LogUniform(1e-4, 1e3)}
    seeds, ends = [], iter([1e-4, float("nan"), 0.5])

    def fake(start, pri, data, pen, weights, others, warm_start=True, init=None, **kw):
        seeds.append(init)
        v = next(ends)
        res = types.SimpleNamespace(values={"g.flux": np.asarray(v)})
        return types.SimpleNamespace(chi2=np.array([1.0]), chi2_red=np.array([[1.0]]), penalty=np.array([1.0]),
                                     results=[res])

    monkeypatch.setattr(ci, "l_curve", fake)
    ci.sweep(None, priors, None, None, [1.0, 2.0, 3.0], (), warm_start=True)
    assert seeds[0] is None
    assert 1e-4 < float(seeds[1]["g.flux"])
    assert seeds[2] is seeds[1]  # the NaN fit did not replace the last finite seed


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_non_finite_fits_are_left_out_and_render_as_nan():
    """A fit that went non-finite takes no part in choosing the weight, and
    renders as NaN instead of stopping the run."""
    good = types.SimpleNamespace(values={"a": np.array([1.0, 2.0])})
    bad = types.SimpleNamespace(values={"a": np.array([np.nan, 2.0])})
    curve = types.SimpleNamespace(chi2=np.array([5.0, np.nan, 3.0, 4.0]), results=[good, good, bad, good])
    assert ci.finite_points(curve) == [0, 3]

    class Broken:
        def render(self, npix, fov):
            raise ValueError("Rendered image must contain positive finite flux.")

    assert np.all(np.isnan(ci.safe_render(Broken(), 4, 1.0)))
    assert ci.safe_render(_image(8), 8, 8.0).shape == (8, 8)


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_safe_clean_retries_a_non_finite_refit_at_a_lower_gain(monkeypatch):
    """CLEAN's nnls refit raising on infs/NaNs (2004 data2 TV, job 18579912) is retried once at a quarter of the gain; other errors and a second failure are raised."""
    gains = []

    def flaky(*args, gain=0.1, **kw):
        gains.append(gain)
        if len(gains) == 1:
            raise ValueError("array must not contain infs or NaNs")
        return "cleaned"

    monkeypatch.setattr(ci, "clean", flaky)
    assert ci.safe_clean("data", gain=0.2) == "cleaned"
    assert gains == [0.2, 0.05]

    def always(*args, **kw):
        raise ValueError("array must not contain infs or NaNs")

    monkeypatch.setattr(ci, "clean", always)
    with pytest.raises(ValueError):
        ci.safe_clean("data")

    def other(*args, **kw):
        raise ValueError("something else")

    monkeypatch.setattr(ci, "clean", other)
    with pytest.raises(ValueError, match="something else"):
        ci.safe_clean("data")


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_two_scale_image_is_unit_sum_and_flux_weighted():
    """Image.render is unit-sum per component: the saved gp2 image must weight them by flux, not add them (it summed to 2)."""
    env, ext = np.zeros((4, 4)), np.zeros((4, 4))
    env[1, 1], ext[2, 2] = 1.0, 1.0
    img = ci.two_scale_image(env, ext, 3.0, 1.0)
    np.testing.assert_allclose(img.sum(), 1.0)
    np.testing.assert_allclose([img[1, 1], img[2, 2]], [0.75, 0.25])
