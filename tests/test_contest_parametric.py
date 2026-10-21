"""The parametric / combined routes of the contest campaign (scripts/contest_parametric.py).

A tiny known-truth file from our simulator: four VLTI UTs, a 1.2 mas uniform-disk
star and a point companion whose flux ratio falls as a power law in wavelength
(the SPARCO model), on a 15-pixel grid. The parametric fit must recover the
truth, the combined fit must not do worse than it on its own evidence-free
χ², and the route choice follows the BIC with ties to the simpler route.
"""

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

from crosscheck import simulate, sky
from evidence.plugin import record

pytest.importorskip("virgil.imaging")
from virgil.imaging import beam  # noqa: E402
from virgil.oidata import OIData  # noqa: E402

SCRIPTS = pathlib.Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("contest_parametric", SCRIPTS / "contest_parametric.py")
cp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cp)

pytestmark = [pytest.mark.x64, pytest.mark.slow]

UTS = np.array([[-9.925, -20.335], [14.887, 30.502], [44.915, 66.183], [103.306, 43.999]])
DIAM, DRA, DDEC, RATIO, INDEX, WAVEL0 = 1.2, 4.0, 1.5, 0.25, -2.0, 1.9e-6
NPIX, FOV = 15, 15.0


def _scene(u, v, w):
    f = RATIO * (w / WAVEL0) ** INDEX
    return (sky.vis_uniform_disk(u, v, w, DIAM) + f * sky.vis_point(u, v, w, DRA, DDEC)) / (1 + f)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    path = tmp_path_factory.mktemp("param") / "p.fits"
    simulate.observe(path, _scene, UTS, hour_angles_h=np.linspace(-3, 3, 7),
                     wavelengths=np.array([1.5e-6, 1.9e-6, 2.4e-6]), dec_deg=-30.0, sigma_v2=2e-3,
                     sigma_cp_deg=0.3, rng=np.random.default_rng(7))
    return [OIData(str(path))]


@pytest.fixture(scope="module")
def parametric(data):
    res = beam(data)
    w0, bandwidth = cp.wavelengths(data)
    assert bandwidth > cp.SPARCO_BANDWIDTH  # three bands: the flux ratio gets an index
    fit, priors = cp.fit_parametric(data, "point", res, FOV, NPIX, True, w0)
    return res, w0, fit, priors


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"])
def test_parametric_fit_recovers_star_and_chromatic_companion(parametric):
    """Star diameter, companion offset and flux ratio at the fitted wavel0
    (the data's median, 1.9 µm) to a few per cent; the index to 0.5."""
    _, w0, fit, _ = parametric
    comp = fit.model.comp
    ratio = cp.ci.flux_value(comp.flux)
    expected = RATIO * (w0 / WAVEL0) ** INDEX
    record("parametric_diam_error", abs(float(fit.model.star.diam) / DIAM - 1))
    record("parametric_index_error", abs(float(np.ravel(comp.flux.index)[0]) - INDEX))
    assert abs(float(fit.model.star.diam) / DIAM - 1) < 0.1
    assert np.hypot(float(comp.dra) - DRA, float(comp.ddec) - DDEC) < 0.2
    assert abs(ratio / expected - 1) < 0.1
    assert abs(float(np.ravel(comp.flux.index)[0]) - INDEX) < 0.5
    assert float(np.sum(fit.info["chi2_red"])) < 2.0


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="control")
def test_a_point_star_alone_cannot_fit_the_companion(data, parametric):
    """Control: without the companion the raw χ²/N is far above the fit's."""
    res, w0, fit, _ = parametric
    alone, _ = cp.fit_parametric(data, None, res, FOV, NPIX, False, w0)
    assert float(np.sum(alone.info["chi2_red"])) > 5 * float(np.sum(fit.info["chi2_red"]))


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"])
def test_combined_route_keeps_a_good_parametric_fit(data, parametric):
    """With the truth already parametric, the GP residual image adds nothing
    material: the combined raw χ²/N stays near 1 and the companion stays put;
    the report splits V² and closure phases."""
    res, w0, fit, priors = parametric
    r, log_z, hyper, gamma = cp.fit_image(data, cp.parts_of(fit.model), priors, res, FOV, NPIX, True, w0, smoke=True)
    report = cp.chi2_report(r.model, data)
    assert np.isfinite(log_z) and hyper["sigma"] > 0 and gamma > 0
    assert report["chi2_red"] < 3.0
    assert set(report["datasets"][0]) == {"chi2_v2", "chi2_cp", "short_mean"}
    assert report["chi2"] > 0
    assert np.hypot(float(r.model.comp.dra) - DRA, float(r.model.comp.ddec) - DDEC) < 0.5


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_choice_follows_evidence_and_ties_go_to_the_simpler_route():
    assert cp.choose({"parametric": 500.0, "image": 300.0, "combined": 100.0}) == "combined"
    assert cp.choose({"parametric": 500.0, "image": 300.0, "combined": 305.0}) == "image"
    assert cp.choose({"parametric": 108.0, "image": 300.0, "combined": 100.0}) == "parametric"
