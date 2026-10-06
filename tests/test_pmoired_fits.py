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

IDENT = (lambda x: x, lambda y: y, lambda y: 1.0)
# ratio = cos(incl): virgil's axis ratio against PMOIRED's inclination
RATIO = (
    lambda r: float(np.rad2deg(np.arccos(r))),
    lambda i: float(np.cos(np.deg2rad(i))),
    lambda i: float(-np.sin(np.deg2rad(i)) * np.pi / 180),  # d ratio / d incl
)

# per scene: fixed PMOIRED parameters, and virgil path -> (PMOIRED key,
# (to PMOIRED, back to virgil, derivative of the way back))
MAPS = {
    "binary": (
        {"A,ud": 0.0, "A,f": 1.0, "B,ud": 0.0},
        {"dra": ("B,x", IDENT), "ddec": ("B,y", IDENT), "flux": ("B,f", IDENT)},
    ),
    "resolved_star_companion": (
        {"star,f": 1.0, "comp,ud": 0.0},
        {"star.diam": ("star,ud", IDENT), "comp.dra": ("comp,x", IDENT),
         "comp.ddec": ("comp,y", IDENT), "comp.flux": ("comp,f", IDENT)},
    ),
    "star_envelope": (
        {"star,ud": 0.0, "star,f": 1.0},
        {"env.fwhm": ("env,fwhm", IDENT), "env.ratio": ("env,incl", RATIO),
         "env.pa": ("env,projang", IDENT), "env.flux": ("env,f", IDENT)},
    ),
}
SCENES = [vb.binary, vb.resolved_star_companion, vb.star_envelope]


def _pmoired(scene, maker, path):
    """PMOIRED's fit, returned in virgil's parameters and order: best
    values, sigmas and covariance (through the Jacobian of the mapping)."""
    fixed, keys = MAPS[maker.__name__]
    start = dict(fixed)
    for path_v, (key, (to_pm, _, _)) in keys.items():
        start[key] = to_pm(float(scene.start[path_v]))
    free = [keys[p][0] for p in scene.truth]
    pm = pmoired_fit(path, start, free)
    back = [keys[p][1] for p in scene.truth]
    best = np.array([b[1](pm["best"][k]) for k, b in zip(free, back)])
    jac = np.diag([b[2](pm["best"][k]) for k, b in zip(free, back)])
    cov = jac @ pm["cov"] @ jac
    sigma = np.array([abs(b[2](pm["best"][k])) * pm["sigma"][k] for k, b in zip(free, back)])
    return {"best": best, "sigma": sigma, "cov": cov, "raw": pm}, free


def _observe(path, scene, stations, rng=None, mode="baseline"):
    simulate.observe(
        path, scene.vis, stations, hour_angles_h=HA, wavelengths=WL,
        dec_deg=-50.0, sigma_v2=0.02, sigma_cp_deg=1.0, rng=rng,
        phase_noise=mode,
    )


# Both packages read the file with their own OIFITS reader: agreement
# would fail if virgil misread uv signs, closure-phase orientation or
# errors, so these also validate read_oifits against PMOIRED's reader.
@pytest.mark.validates("virgil.fitting.fit", "virgil.oifits.read_oifits", roots=["pmoired"])
@pytest.mark.parametrize("maker", SCENES, ids=lambda m: m.__name__)
def test_noise_free_fits_reach_the_truth(tmp_path, maker):
    scene = maker()
    path = tmp_path / "f.fits"
    _observe(path, scene, UTS)
    pm, free = _pmoired(scene, maker, path)
    res, _ = vb.fit_scene(scene, vb.load(path))
    truth = vb.flat_truth(scene)
    got_v = vb.flat_values(scene, res.values)
    np.testing.assert_allclose(got_v, truth, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(pm["best"], truth, rtol=1e-4, atol=1e-6)


@pytest.mark.validates("pipeline:pmoired-identical-fits", "virgil.fitting.fit", "virgil.inference.laplace_cov", "virgil.oifits.read_oifits", roots=["pmoired"])
@pytest.mark.parametrize("maker", SCENES, ids=lambda m: m.__name__)
@pytest.mark.parametrize("mode", ["baseline", "triangle"])
def test_best_fits_agree_on_noisy_data(tmp_path, maker, mode):
    scene = maker()
    path = tmp_path / "f.fits"
    _observe(path, scene, UTS, np.random.default_rng(5), mode)
    pm, free = _pmoired(scene, maker, path)
    res, cov = vb.fit_scene(scene, vb.load(path))
    sigma = np.sqrt(np.diag(cov))
    diff = (pm["best"] - vb.flat_values(scene, res.values)) / sigma
    record("max_abs_dbest_sigma", np.max(np.abs(diff)))
    assert np.all(np.abs(diff) < 0.25), diff


@pytest.mark.validates("virgil.inference.laplace_cov", roots=["pmoired"])
@pytest.mark.parametrize("maker", SCENES, ids=lambda m: m.__name__)
def test_uncertainties_agree_without_closure_redundancy(tmp_path, maker):
    """Three telescopes: one closure phase per snapshot, nothing to
    correlate, so the two packages' curvature errors and parameter
    correlations must agree."""
    scene = maker()
    path = tmp_path / "f.fits"
    _observe(path, scene, UTS[:3], np.random.default_rng(1))
    pm, _ = _pmoired(scene, maker, path)
    _, cov = vb.fit_scene(scene, vb.load(path))
    # the bridge's covariance is consistent with its sigmas, in order
    np.testing.assert_allclose(np.sqrt(np.diag(pm["cov"])), pm["sigma"], rtol=1e-6)
    ratio = pm["sigma"] / np.sqrt(np.diag(cov))
    record("max_abs_sigma_ratio_minus_1", np.max(np.abs(ratio - 1)))
    np.testing.assert_allclose(ratio, 1.0, atol=0.04)

    def corr(c):
        d = np.sqrt(np.diag(c))
        return c / np.outer(d, d)

    dcorr = np.max(np.abs(corr(pm["cov"]) - corr(cov)))
    record("max_abs_correlation_difference", dcorr)
    assert dcorr < 0.05, dcorr


@pytest.mark.validates("virgil.inference.laplace_cov", roots=["pmoired"], kind="control")
def test_independent_closure_phases_shrink_pmoired_errors(tmp_path):
    """Four telescopes: PMOIRED counts 4 closure phases per snapshot as
    independent, virgil whitens them as 3 correlated ones, so PMOIRED's
    errors are a few per cent smaller. A difference of definition; the
    pull campaign shows which is calibrated."""
    scene = vb.binary()
    path = tmp_path / "f.fits"
    _observe(path, scene, UTS, np.random.default_rng(1))
    pm, _ = _pmoired(scene, vb.binary, path)
    _, cov = vb.fit_scene(scene, vb.load(path))
    ratio = pm["sigma"] / np.sqrt(np.diag(cov))
    record("mean_sigma_ratio", np.mean(ratio))
    assert np.all((ratio > 0.85) & (ratio < 0.995)), ratio


@pytest.mark.slow
@pytest.mark.validates("pipeline:pmoired-identical-fits", "virgil.inference.laplace_cov", "virgil.fitting.fit", roots=["pmoired", "statistics"], tier="B", kind="regression")
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
        pp.append((pm["best"] - truth) / pm["sigma"])
    pv, pp = np.array(pv), np.array(pp)
    record("virgil_pull_sd_mean", pv.std(0).mean())
    record("pmoired_pull_sd_mean", pp.std(0).mean())
    # N = 200: a mean has sd 0.07, an sd has sd 0.05
    band = 3 / np.sqrt(2 * n)
    for p in (pv, pp):
        assert np.all(np.abs(p.mean(0)) < 3 / np.sqrt(n)), p.mean(0)
    if mode == "baseline":
        # correlated closure phases, as virgil assumes: virgil calibrated;
        # PMOIRED, treating them as independent, overconfident but by a
        # bounded amount (measured ~1.10)
        assert np.all(np.abs(pv.std(0) - 1) < band), pv.std(0)
        assert np.all((pp.std(0) > 1 - band) & (pp.std(0) < 1.25 + band)), pp.std(0)
        assert pp.std(0).mean() > pv.std(0).mean()
    else:
        # independent closure phases, as PMOIRED assumes: PMOIRED
        # calibrated; virgil conservative, never overconfident
        assert np.all(np.abs(pp.std(0) - 1) < band), pp.std(0)
        assert np.all((pv.std(0) < 1 + band) & (pv.std(0) > 0.75)), pv.std(0)
