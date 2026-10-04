"""Long-baseline injection and recovery, from uv geometry to a fit.

Our uv tracks (Thompson, Moran & Swenson) and our OIFITS writer produce the
file; virgil reads it and fits it. Noise-free files must be recovered to
optimiser precision; noisy ones must give pulls (fit - truth) / sigma
distributed as N(0, 1), with sigma virgil's Laplace uncertainty.
"""

import numpy as np
import pytest

from crosscheck import array, simulate

vb = pytest.importorskip("virgil_bridge")
from virgil.coverage import vlti_oidata  # noqa: E402

pytestmark = pytest.mark.x64

# The four VLTI UTs as virgil's coverage module lists them (East, North, m)
UTS = np.array(
    [[-9.925, -20.335], [14.887, 30.502], [44.915, 66.183], [103.306, 43.999]]
)
HA = np.linspace(-3, 3, 7)
WL = np.linspace(1.5e-6, 2.4e-6, 6)
DEC = -50.0


@pytest.mark.validates("virgil.coverage.vlti_oidata", roots=["standards"])
def test_uv_tracks_match_virgil_coverage():
    ha, wl = [-3.0, -1.5, 0.0, 1.5, 3.0], np.array([3.0e-6, 3.5e-6, 4.0e-6])
    data = vlti_oidata(
        stations=UTS, declination_deg=DEC, hour_angles_h=ha, wavelengths_m=wl
    )
    ours = [array.snapshot_uv(UTS, h, DEC, -24.6276) for h in ha]
    u = np.concatenate([o[0] for o in ours])
    v = np.concatenate([o[1] for o in ours])
    # virgil orders (baseline, wavelength); ours (baseline) per snapshot
    np.testing.assert_allclose(np.asarray(data.u)[:: len(wl)], u, atol=1e-9)
    np.testing.assert_allclose(np.asarray(data.v)[:: len(wl)], v, atol=1e-9)


@pytest.mark.parametrize("make", vb.SCENES, ids=lambda f: f.__name__)
@pytest.mark.validates("virgil.oifits.read_oifits", "virgil.oidata.OIData", roots=["standards"])
def test_file_reproduces_virgil_model(tmp_path, make):
    """virgil's reader + model reproduce what we wrote, sample by sample:
    the OIFITS sign of u, v, the T3 orientation and the units agree."""
    scene = make()
    path = tmp_path / "s.fits"
    simulate.observe(
        path, scene.vis, UTS, hour_angles_h=HA, wavelengths=WL, dec_deg=DEC,
        sigma_v2=0.01, sigma_cp_deg=0.5,
    )
    data = vb.load(path)
    observed, _ = data.flatten_data()
    model = data.model(scene.template)
    assert np.max(np.abs(np.asarray(observed) - np.asarray(model))) < 1e-12


@pytest.mark.parametrize("make", vb.SCENES, ids=lambda f: f.__name__)
@pytest.mark.validates("virgil.fitting.fit", roots=["mathematics"])
def test_noise_free_recovery(tmp_path, make):
    scene = make()
    path = tmp_path / "s.fits"
    simulate.observe(
        path, scene.vis, UTS, hour_angles_h=HA, wavelengths=WL, dec_deg=DEC,
        sigma_v2=0.01, sigma_cp_deg=0.5,
    )
    result, _ = vb.fit_scene(scene, vb.load(path))
    truth = vb.flat_truth(scene)
    got = vb.flat_values(scene, result.values)
    np.testing.assert_allclose(got, truth, rtol=1e-6, atol=1e-8)


@pytest.mark.slow
@pytest.mark.parametrize("make", vb.SCENES, ids=lambda f: f.__name__)
@pytest.mark.parametrize("phase_noise", ["baseline", "triangle"])
@pytest.mark.validates("virgil.fitting.fit", "virgil.inference.laplace_cov", "virgil.likelihood.whitened_residuals", roots=["statistics"], tier="B")
def test_noisy_pulls_are_unit_normal(tmp_path, make, phase_noise):
    scene = make()
    rng = np.random.default_rng(11)
    path = tmp_path / "s.fits"
    pulls = []
    n = 60
    for _ in range(n):
        simulate.observe(
            path, scene.vis, UTS, hour_angles_h=HA, wavelengths=WL,
            dec_deg=DEC, sigma_v2=0.02, sigma_cp_deg=1.0, rng=rng,
            phase_noise=phase_noise,
        )
        result, cov = vb.fit_scene(scene, vb.load(path), start=scene.truth)
        got = vb.flat_values(scene, result.values)
        pulls.append((got - vb.flat_truth(scene)) / np.sqrt(np.diag(cov)))
    pulls = np.array(pulls)
    # 60 draws: the mean has sd 0.13 and the sd has sd ~0.09
    assert np.all(np.abs(pulls.mean(0)) < 0.5)
    assert np.all((pulls.std(0) > 0.7) & (pulls.std(0) < 1.35))


@pytest.mark.validates("virgil.inference.laplace_cov", roots=["mathematics"])
def test_laplace_cov_with_array_parameters(tmp_path):
    """Finding 5, fixed in virgil#135: the rim's array-valued az_amps and
    az_pas beside scalar paths give a full, positive-definite covariance."""
    scene = vb.star_rim()
    path = tmp_path / "s.fits"
    simulate.observe(
        path, scene.vis, UTS, hour_angles_h=HA, wavelengths=WL, dec_deg=DEC,
        sigma_v2=0.02, sigma_cp_deg=1.0,
    )
    _, cov = vb.fit_scene(scene, vb.load(path), start=scene.truth)
    n = vb.flat_truth(scene).size
    assert cov.shape == (n, n)
    assert np.all(np.isfinite(cov))
    np.testing.assert_allclose(cov, cov.T, rtol=1e-8, atol=1e-14)
    assert np.all(np.linalg.eigvalsh(cov) > 0)


@pytest.mark.validates("virgil.fitting.fit", roots=["self-consistency"])
@pytest.mark.parametrize("wrap", ["expand", "expand_to_event"])
def test_fit_uses_lm_for_expanded_uniform_priors(tmp_path, wrap):
    """fit documents LM as the default whenever the objective has a
    least-squares form; Uniform(0, 1).expand([1]) is the same prior as an
    array-shaped Uniform, which does get LM (finding 7, fixed in
    virgil#142)."""
    import numpyro.distributions as dist

    from virgil.fitting import fit

    scene = vb.star_rim()
    path = tmp_path / "s.fits"
    simulate.observe(
        path, scene.vis, UTS, hour_angles_h=HA, wavelengths=WL, dec_deg=DEC,
        sigma_v2=0.02, sigma_cp_deg=1.0, rng=np.random.default_rng(3),
    )
    priors = dict(scene.priors)
    amps, pas = dist.Uniform(0, 1).expand([1]), dist.Uniform(0, 360).expand([1])
    if wrap == "expand_to_event":
        amps, pas = amps.to_event(1), pas.to_event(1)
    priors["rim.az_amps"], priors["rim.az_pas"] = amps, pas
    assert fit(scene.template, priors, vb.load(path)).info["method"] == "lm"


@pytest.mark.parametrize("method", ["lm", "lbfgs"])
@pytest.mark.validates("virgil.fitting.fit", roots=["mathematics"])
def test_fit_from_the_exact_optimum_converges(tmp_path, method):
    """A fit started exactly at a zero-residual optimum is converged at
    once (finding 6, fixed in virgil#144)."""
    import warnings

    from virgil.fitting import fit

    scene = vb.binary()
    path = tmp_path / "s.fits"
    simulate.observe(
        path, scene.vis, UTS, hour_angles_h=HA, wavelengths=WL, dec_deg=DEC,
        sigma_v2=0.02, sigma_cp_deg=1.0,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fit(scene.template, scene.priors, vb.load(path), method=method)
    assert not [w for w in caught if "converge" in str(w.message)]
    assert result.info["converged"]
    # L-BFGS tests the starting gradient before stepping; LM may take a
    # step or two of rounding noise (virgil#144's own bounds)
    assert result.info["steps"] == 0 if method == "lbfgs" else result.info["steps"] <= 3


@pytest.mark.validates("virgil.oidata.OIData", roots=["standards"])
def test_all_closure_phases_flagged_leaves_the_visibilities(tmp_path):
    """Finding 8, fixed in virgil#155 and superseded by virgil#158: a file
    whose OI_T3 FLAG is all set warns and is fitted as visibility-only data,
    with exactly the likelihood of the same V² without a T3 table."""
    from astropy.io import fits

    from virgil.likelihood import model_loglike
    from virgil.oidata import OIData

    scene = vb.binary()
    flagged, plain = tmp_path / "flagged.fits", tmp_path / "plain.fits"
    for path, phases in [(flagged, True), (plain, False)]:
        simulate.observe(
            path, scene.vis, UTS, hour_angles_h=[0.0], wavelengths=[2.0e-6],
            dec_deg=DEC, sigma_v2=0.02, sigma_cp_deg=1.0, closure_phases=phases,
        )
    with fits.open(flagged, mode="update") as h:
        h["OI_T3"].data["FLAG"][:] = True
    with pytest.warns(UserWarning, match="closure phase"):
        data = OIData(str(flagged))
    assert not data.has_phases
    probe = scene.template.set("flux", 0.04)
    # exactly: the flagged phases must contribute nothing at all
    assert float(model_loglike(probe, data)) == float(model_loglike(probe, OIData(str(plain))))
