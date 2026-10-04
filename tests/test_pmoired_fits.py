"""PMOIRED Stage 2: virgil and PMOIRED fit the same files.

Both packages fit our simulated OIFITS files (30 samples per baseline, where
PMOIRED's models are exact, P3). Their best fits must agree well inside the
errors, and their uncertainties must agree once two documented
differences of definition are accounted for:

1. PMOIRED reports uncertainties "normalized" by sqrt(reduced chi^2);
   virgil's Laplace covariance is the plain curvature. The bridge undoes
   PMOIRED's normalisation.
2. PMOIRED treats every closure phase as independent; virgil whitens the
   closure phases of each snapshot as a correlated group, as they are when
   formed from baseline phases (the realistic case). With 3 telescopes (one
   triangle, nothing to correlate) the two agree; with 4 telescopes
   (4 triangles, 3 independent) PMOIRED's errors are a few per cent smaller.
"""

import numpy as np
import pytest

from crosscheck import simulate

pytest.importorskip("pmoired")
vb = pytest.importorskip("virgil_bridge")
from evidence.plugin import record  # noqa: E402
from external_bridge.pmoired_models import fit as pmoired_fit  # noqa: E402

pytestmark = [pytest.mark.external, pytest.mark.x64]

UTS = np.array(
    [[-9.925, -20.335], [14.887, 30.502], [44.915, 66.183], [103.306, 43.999]]
)
HA = np.linspace(-3, 3, 5)
WL = np.linspace(1.5e-6, 2.4e-6, 6)

# virgil path -> (PMOIRED key, transform virgil value -> PMOIRED value)
MAPS = {
    "binary": (
        {"A,ud": 0.0, "A,f": 1.0, "B,ud": 0.0},
        {"dra": "B,x", "ddec": "B,y", "flux": "B,f"},
    ),
    "resolved_star_companion": (
        {"star,f": 1.0, "comp,ud": 0.0},
        {"star.diam": "star,ud", "comp.dra": "comp,x", "comp.ddec": "comp,y", "comp.flux": "comp,f"},
    ),
}
SCENES = [vb.binary, vb.resolved_star_companion]


def _pmoired(scene, maker, path):
    fixed, keys = MAPS[maker.__name__]
    start = dict(fixed)
    for path_v, key in keys.items():
        start[key] = float(scene.start[path_v])
    free = [keys[p] for p in scene.truth]
    return pmoired_fit(path, start, free), free


def _observe(path, scene, stations, rng=None, mode="baseline"):
    simulate.observe(
        path, scene.vis, stations, hour_angles_h=HA, wavelengths=WL,
        dec_deg=-50.0, sigma_v2=0.02, sigma_cp_deg=1.0, rng=rng,
        phase_noise=mode,
    )


@pytest.mark.validates("virgil.fitting.fit", roots=["pmoired"])
@pytest.mark.parametrize("maker", SCENES, ids=lambda m: m.__name__)
def test_noise_free_fits_reach_the_truth(tmp_path, maker):
    scene = maker()
    path = tmp_path / "f.fits"
    _observe(path, scene, UTS)
    pm, free = _pmoired(scene, maker, path)
    res, _ = vb.fit_scene(scene, vb.load(path))
    truth = vb.flat_truth(scene)
    got_v = vb.flat_values(scene, res.values)
    got_p = np.array([pm["best"][k] for k in free])
    np.testing.assert_allclose(got_v, truth, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(got_p, truth, rtol=1e-4, atol=1e-6)


@pytest.mark.validates("virgil.fitting.fit", "virgil.inference.laplace_cov", roots=["pmoired"])
@pytest.mark.parametrize("maker", SCENES, ids=lambda m: m.__name__)
@pytest.mark.parametrize("mode", ["baseline", "triangle"])
def test_best_fits_agree_on_noisy_data(tmp_path, maker, mode):
    scene = maker()
    path = tmp_path / "f.fits"
    _observe(path, scene, UTS, np.random.default_rng(5), mode)
    pm, free = _pmoired(scene, maker, path)
    res, cov = vb.fit_scene(scene, vb.load(path))
    sigma = np.sqrt(np.diag(cov))
    diff = (np.array([pm["best"][k] for k in free]) - vb.flat_values(scene, res.values)) / sigma
    record("max_abs_dbest_sigma", np.max(np.abs(diff)))
    assert np.all(np.abs(diff) < 0.25), diff


@pytest.mark.validates("virgil.inference.laplace_cov", roots=["pmoired"])
@pytest.mark.parametrize("maker", SCENES, ids=lambda m: m.__name__)
def test_uncertainties_agree_without_closure_redundancy(tmp_path, maker):
    """Three telescopes: one closure phase per snapshot, nothing to
    correlate, so the two packages' curvature errors must agree."""
    scene = maker()
    path = tmp_path / "f.fits"
    _observe(path, scene, UTS[:3], np.random.default_rng(1))
    pm, free = _pmoired(scene, maker, path)
    _, cov = vb.fit_scene(scene, vb.load(path))
    ratio = np.array([pm["sigma"][k] for k in free]) / np.sqrt(np.diag(cov))
    record("max_abs_sigma_ratio_minus_1", np.max(np.abs(ratio - 1)))
    np.testing.assert_allclose(ratio, 1.0, atol=0.04)


@pytest.mark.validates("virgil.inference.laplace_cov", roots=["pmoired"], kind="control")
def test_independent_closure_phases_shrink_pmoired_errors(tmp_path):
    """Four telescopes: PMOIRED counts 4 closure phases per snapshot as
    independent, virgil whitens them as 3 correlated ones, so PMOIRED's
    errors are a few per cent smaller. A difference of definition; the
    pull campaign shows which is calibrated."""
    scene = vb.binary()
    path = tmp_path / "f.fits"
    _observe(path, scene, UTS, np.random.default_rng(1))
    pm, free = _pmoired(scene, vb.binary, path)
    _, cov = vb.fit_scene(scene, vb.load(path))
    ratio = np.array([pm["sigma"][k] for k in free]) / np.sqrt(np.diag(cov))
    record("mean_sigma_ratio", np.mean(ratio))
    assert np.all((ratio > 0.85) & (ratio < 0.995)), ratio


@pytest.mark.slow
@pytest.mark.validates("virgil.inference.laplace_cov", "virgil.fitting.fit", roots=["pmoired", "statistics"], tier="B")
@pytest.mark.parametrize("mode", ["baseline", "triangle"])
def test_pull_campaign_against_pmoired(tmp_path, mode):
    """Both packages fit the same 200 noisy realisations of the binary;
    their pulls (fit - truth) / sigma are recorded. virgil's must be N(0, 1);
    PMOIRED's are reported (expected slightly wide with correlated
    closure-phase noise, because it treats closure phases as independent)."""
    scene = vb.binary()
    rng = np.random.default_rng(23)
    path = tmp_path / "f.fits"
    truth = vb.flat_truth(scene)
    pv, pp = [], []
    n = 200
    for _ in range(n):
        _observe(path, scene, UTS, rng, mode)
        pm, free = _pmoired(scene, vb.binary, path)
        res, cov = vb.fit_scene(scene, vb.load(path), start=scene.truth)
        pv.append((vb.flat_values(scene, res.values) - truth) / np.sqrt(np.diag(cov)))
        pp.append((np.array([pm["best"][k] for k in free]) - truth) / np.array([pm["sigma"][k] for k in free]))
    pv, pp = np.array(pv), np.array(pp)
    record("virgil_pull_sd_mean", pv.std(0).mean())
    record("pmoired_pull_sd_mean", pp.std(0).mean())
    # N = 200: a mean has sd 0.07, an sd has sd 0.05
    assert np.all(np.abs(pv.mean(0)) < 3 / np.sqrt(n))
    band = 3 / np.sqrt(2 * n)
    if mode == "baseline":
        # correlated closure phases, as virgil assumes: calibrated
        assert np.all(np.abs(pv.std(0) - 1) < band), pv.std(0)
    else:
        # independent closure phases: virgil is conservative, not overconfident
        assert np.all((pv.std(0) < 1 + band) & (pv.std(0) > 0.75)), pv.std(0)
    assert np.all(np.isfinite(pp))
