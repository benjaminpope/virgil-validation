"""Starts for the contest pipeline's parametric star models (C2g).

The closed-form estimators (crosscheck.starts, NumPy only) must recover the
size, elongation and position angle of companion-free elongated spotted
stars, and the star fraction of the thin-disk and star-disk-planet phantoms,
from simulated V² alone. The pipeline (scripts/contest_images.py) must never
start a fit on a prior bound, must bound the star fraction instead of letting
CLEAN's flux run away, must only keep a companion that the data support, and
must fall back to imaging without the star when the star start fails. The
virgil parts are a few small fits on the 2004 coverage.
"""

import importlib
import importlib.util
import pathlib
import warnings

import numpy as np
import pytest

from crosscheck import image_metrics, phantoms, sky, starts

ROOT = pathlib.Path(__file__).parents[1]
DATA = pathlib.Path("~/data/imaging_contests").expanduser()
TEMPLATE = DATA / "2004" / "2004-data2.fits"
needs_2004 = pytest.mark.skipif(not TEMPLATE.exists(), reason="contest data not fetched")
needs_ellipse = pytest.mark.skipif(
    importlib.util.find_spec("virgil") is None
    or not hasattr(importlib.import_module("virgil.models"), "EllipticalLimbDarkenedDisk"),
    reason="needs virgil with EllipticalLimbDarkenedDisk (virgil#250)")
LAM, BMAX = 0.55e-6, 60.0  # an NPOI-like coverage, as 2004's: a 1.9 mas beam
BEAM = LAM / BMAX / sky.MAS


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _coverage(rng, n=400):
    """Baselines log-uniform from 3 m to BMAX at uniform angles (m)."""
    r, t = np.exp(rng.uniform(np.log(3.0), np.log(BMAX), n)), rng.uniform(0, np.pi, n)
    return r * np.sin(t), r * np.cos(t)


def _observe(params, rng, coverage):
    """Noisy V² of a phantom: u, v (cycles per radian), V², error."""
    pixel = BEAM / 10
    npix = int(np.ceil(2 * phantoms.extent(params) / pixel)) | 1
    cloud = sky.pixel_image(phantoms.render(params, npix, pixel), pixel)
    bu, bv = coverage
    v2 = np.abs(sky.visibility(cloud, bu, bv, LAM)) ** 2
    err = 0.005 + 0.02 * v2
    return bu / LAM, bv / LAM, v2 + err * rng.standard_normal(v2.shape), err


def _dpa(a, b):
    return (a - b + 90.0) % 180.0 - 90.0


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_moments_recover_an_elliptical_gaussian_exactly():
    rng = np.random.default_rng(1)
    bu, bv = _coverage(rng)
    fwhm, ratio, pa = 4.0, 0.6, 35.0
    v2 = np.abs(sky.vis_elliptical_gaussian(bu, bv, LAM, fwhm, ratio, pa)) ** 2
    m = starts.second_moments(bu / LAM, bv / LAM, v2, 0.01 + 0 * v2)
    assert not m["isotropic"]
    assert m["sigma_major"] == pytest.approx(fwhm / 2.3548200450309493, rel=1e-6)
    assert m["ratio"] == pytest.approx(ratio, rel=1e-6)
    assert _dpa(m["pa"], pa) == pytest.approx(0.0, abs=1e-4)


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_moments_start_the_2004_spotted_stars():
    # The 2004 family ("a limb darkened star with one or more spots"), the
    # companion-free, clearly elongated draws: the moments are a start, so
    # the tolerances are a start's (diameter 25%, PA 20°, ratio 0.2).
    rng = np.random.default_rng(0)
    coverage = _coverage(rng)
    checked = 0
    for seed in range(60):
        p = phantoms.sample("spotted_star", np.random.default_rng(seed), BEAM)
        if p["companion"] or p["axis_ratio"] > 0.8:
            continue
        m = starts.second_moments(*_observe(p, rng, coverage))
        diam = starts.disk_diameter_from_sigma(m["sigma_major"], p["u_ld"])
        assert diam == pytest.approx(p["diam"], rel=0.25), seed
        assert abs(_dpa(m["pa"], p["pa"])) < 20.0, seed
        assert m["ratio"] == pytest.approx(p["axis_ratio"], abs=0.2), seed
        checked += 1
    assert checked >= 6


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_moments_fall_back_with_few_points():
    m = starts.second_moments([1e7, 2e7], [0.0, 1e7], [0.9, 0.8], [0.01, 0.01])
    assert m["isotropic"] and m["n"] == 2 and m["sigma_major"] > 0 and m["ratio"] == 1.0
    assert np.isnan(starts.second_moments([], [], [], [])["sigma_major"])


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
@pytest.mark.parametrize("u_ld", [0.0, 0.3, 0.8])
def test_disk_diameter_matches_the_disks_second_moment(u_ld):
    # ⟨x²⟩ of the linear-law disk by quadrature, against the closed form.
    s = np.linspace(0, 1, 200001)
    intensity = 1 - u_ld * (1 - np.sqrt(1 - s**2))
    radius = 1.7
    x2 = radius**2 * np.trapezoid(intensity * s**3, s) / np.trapezoid(intensity * s, s) / 2
    assert starts.disk_diameter_from_sigma(np.sqrt(x2), u_ld) == pytest.approx(2 * radius, rel=1e-6)
    if u_ld == 0:
        assert starts.disk_diameter_from_sigma(1.0) == pytest.approx(4.0)


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_star_fraction_is_the_exact_minimum():
    rng = np.random.default_rng(3)
    bu, bv = _coverage(rng)
    vs = sky.vis_uniform_disk(bu, bv, LAM, 0.3)
    ve = sky.vis_elliptical_gaussian(bu, bv, LAM, 6.0, 0.5, 20.0, dra=1.0)
    for f in (0.0, 0.05, 0.37, 0.9, 1.0):
        v2 = np.abs(f * vs + (1 - f) * ve) ** 2
        assert starts.star_fraction(vs, ve, v2, 0.01 + 0 * v2) == pytest.approx(f, abs=1e-6)
    v2 = np.abs(0.6 * vs + 0.4 * ve) ** 2 + 0.01 * rng.standard_normal(bu.shape)
    grid = np.linspace(0, 1, 100001)
    chi2 = [np.sum((np.abs(g * vs + (1 - g) * ve) ** 2 - v2) ** 2) for g in grid[::100]]
    assert starts.star_fraction(vs, ve, v2, 0.01 + 0 * v2) == pytest.approx(grid[::100][np.argmin(chi2)], abs=1e-3)


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
@pytest.mark.parametrize("family", ["thin_disk", "star_disk_planet"])
def test_star_fraction_from_the_long_baseline_plateau(family):
    # The 2006 thin disk and the 2018 star-disk-planet phantoms: with the
    # environment unknown, the long-baseline plateau gives f within 0.1.
    rng = np.random.default_rng(4)
    coverage = _coverage(rng)
    for seed in range(6):
        p = phantoms.sample(family, np.random.default_rng(seed), BEAM)
        u, v, v2, err = _observe(p, rng, coverage)
        vs = 1.0 if family == "thin_disk" else sky.vis_uniform_disk(*coverage, LAM, p["star_diam"])
        assert starts.star_fraction(vs, None, v2, err, q=np.hypot(u, v)) == pytest.approx(p["star_frac"], abs=0.1), seed


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_no_start_lies_within_five_percent_of_a_bound():
    import numpyro.distributions as dist

    ci = _load("contest_images")
    priors = {"diam": dist.LogUniform(0.1, 10.0), "ratio": dist.Uniform(0.2, 1.0), "pa": dist.Uniform(*ci.PA_PRIOR),
              "env.flux": dist.LogUniform(ci.FLUX_FLOOR, ci.FLUX_CAP)}
    # A centre on (or beyond) every bound, as the old starts were: PA 0 on
    # U(0, 180), and CLEAN's runaway flux of 1e163.
    centre = {"diam": 10.0, "ratio": 1.0, "pa": -90.0, "env.flux": 1e163}
    draws = ci.jittered_starts(centre, priors, 200, np.random.default_rng(0),
                               {"diam": ("log", 0.3), "ratio": ("add", 0.1), "pa": ("add", 30.0), "env.flux": ("log", 1.0)})
    for k, prior in priors.items():
        lo, hi = float(prior.low), float(prior.high)
        x = np.array([d[k] for d in draws])
        if isinstance(prior, dist.LogUniform):
            lo, hi, x = np.log(lo), np.log(hi), np.log(x)
        pad = 0.05 * (hi - lo)
        assert np.all(x >= lo + pad - 1e-9) and np.all(x <= hi - pad + 1e-9), k
    assert max(d["env.flux"] for d in draws) < ci.FLUX_CAP
    # Every plausible PA (0-180) and its ±30° jitter is well inside the window.
    assert ci.PA_PRIOR[0] + 0.05 * 360 < -30 and 180 + 30 < ci.PA_PRIOR[1] - 0.05 * 360


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_l1_with_zero_and_denormal_entry_pixels():
    rng = np.random.default_rng(5)
    ref = rng.random((9, 9))
    entry = ref.copy()
    entry[:3] = 0.0
    entry[3, :] = 1e-310  # denormal: r/e overflowed before the floor
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        score = image_metrics.l1_score(entry, ref)
        empty = image_metrics.l1_score(np.zeros_like(ref), ref)
    assert np.isfinite(score) and 0 < score < 1
    assert empty == 0.0


@pytest.fixture(scope="module")
def simulated(tmp_path_factory):
    """Simulated 2004 datasets (NumPy, scripts/contest_bench.py): a spotted
    star with no companion, a point star with a companion at (6, -3) mas and
    flux 0.1, a bare point star and a flat-ish thin disk with no star."""
    if not TEMPLATE.exists():
        pytest.skip("contest data not fetched")
    bench = _load("contest_bench")
    out = tmp_path_factory.mktemp("sim")
    clouds = {"binary": sky.mix([sky.point(), sky.point(6.0, -3.0)], [1.0, 0.1]), "bare": sky.point()}
    for family, seed in (("spotted_star", 0), ("thin_disk", 0)):
        p = phantoms.sample(family, np.random.default_rng(seed), 1.9)
        if family == "thin_disk":
            p["star_frac"] = 0.0
        npix, pixel = bench.truth_grid(p, 1.9)
        clouds[family] = sky.pixel_image(phantoms.render(p, npix, pixel), pixel)
    paths = {}
    for name, cloud in clouds.items():
        paths[name] = out / f"{name}.fits"
        bench.simulate_file(TEMPLATE, paths[name], cloud, np.random.default_rng(7))
    return paths


@pytest.mark.validates("pipeline:contest-imaging", "virgil.grid_fit.linear_flux_grid", roots=["self-consistency"], kind="reference")
@needs_2004
def test_companion_found_on_a_binary_and_not_on_a_bare_star(simulated):
    import virgil.models as vm
    from virgil.imaging import beam
    from virgil.oidata import OIData

    ci = _load("contest_images")
    data = OIData(str(simulated["binary"]))
    resolution = beam(data)
    base, priors, record = ci.companion_search(data, resolution, vm.PointSource(), {})
    assert record["kept"] and isinstance(base, vm.System)
    assert float(base.comp.dra) == pytest.approx(6.0, abs=0.3)
    assert float(base.comp.ddec) == pytest.approx(-3.0, abs=0.3)
    assert float(base.comp.flux) == pytest.approx(0.1, rel=0.2)
    assert record["dchi2"] >= ci.COMPANION_DCHI2
    assert {"star.comp.dra", "star.comp.ddec", "star.comp.flux"} <= set(priors)
    bare = OIData(str(simulated["bare"]))
    base, priors, record = ci.companion_search(bare, beam(bare), vm.PointSource(), {})
    assert isinstance(base, vm.PointSource) and not record["kept"] and record["searched"]
    # The guard: against a primary far worse than the no-star ellipse, no search.
    base, _, record = ci.companion_search(data, resolution, vm.PointSource(), {}, primary_chi2=500.0, nostar_chi2=2.0)
    assert isinstance(base, vm.PointSource) and not record["searched"]


def _star_setup(ci, path, **options):
    spec = {"label": "sim", "files": [str(path)]}
    opts = dict(star=True, init="clean", clean_iters=30, oversample=2.0, star_model="point", n_starts=2) | options
    return ci.setup(spec, pathlib.Path("/"), False, 1.0, **opts)


@pytest.mark.validates("pipeline:contest-imaging", "virgil.imaging.clean", roots=["self-consistency"], kind="reference")
@needs_2004
@pytest.mark.parametrize("scene", ["spotted_star", "thin_disk"])
def test_clean_star_flux_is_bounded_or_rejected(simulated, scene):
    # The runaway (C2g): CLEAN with a point base on a resolved star (or on a
    # disk with no star) ran its components' total to 1e163. Now either the
    # star keeps a fraction >= F_MIN with the env/star ratio inside its fixed
    # prior, or the star is rejected with a reason and the setup has no star.
    ci = _load("contest_images")
    s = _star_setup(ci, simulated[scene])
    if s["star"]:
        assert s["star_rejected"] == ""
        flux = ci.flux_value(s["img0"].flux)
        assert ci.FLUX_FLOOR < flux < ci.FLUX_CAP
        assert 1 / (1 + flux) >= ci.F_MIN
        assert float(s["priors"]["env.flux"].high) == ci.FLUX_CAP
    else:
        assert s["star_rejected"] in ("clean_flux", "chi2", "not_converged")
        assert "env.flux" not in s["priors"] and s["label"] == "sim_star"


@pytest.mark.validates("pipeline:contest-imaging", roots=["self-consistency"], kind="reference")
@needs_2004
def test_star_guard_falls_back_to_no_star(simulated, monkeypatch):
    # With the guard's factor at 0 every star start counts as worse than the
    # no-star ellipse: the arm becomes the baseline, under its own label.
    ci = _load("contest_images")
    monkeypatch.setattr(ci, "STAR_GUARD", 0.0)
    s = _star_setup(ci, simulated["bare"], init="moments")
    assert s["star_rejected"] == "chi2" and not s["star"] and s["star_model"] == "none"
    assert s["label"] == "sim_star" and s["ellipse"] is not None
    assert "env.flux" not in s["priors"]


@pytest.mark.validates("pipeline:contest-imaging", "virgil.models.EllipticalLimbDarkenedDisk", roots=["self-consistency"], kind="reference")
@needs_2004
@needs_ellipse
def test_primary_fit_moves_from_its_moment_start(simulated):
    # The old fit_primary started PA 0 on its U(0, 180) prior and LM took no
    # steps. From the moments, the multistart lands near the simulated star.
    from virgil.imaging import beam, starting_image
    from virgil.oidata import OIData

    ci = _load("contest_images")
    truth = phantoms.sample("spotted_star", np.random.default_rng(0), 1.9)
    data = OIData(str(simulated["spotted_star"]))
    resolution = beam(data)
    start = starting_image(data, star=True, hole_mas=0.5 * resolution.minor_mas)
    model, priors, record = ci.fit_primary(data, resolution, "ellipse", start, n_starts=3)
    assert float(priors["pa"].low) == ci.PA_PRIOR[0] and 0.0 <= float(model.pa) < 180.0
    assert record["converged"] and len(record["chi2_starts"]) == 3
    assert float(model.diam) == pytest.approx(truth["diam"], rel=0.3)
    assert abs(_dpa(float(model.pa), truth["pa"])) < 20.0
