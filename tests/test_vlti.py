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
@pytest.mark.parametrize("make", vb.SCENES[:3], ids=lambda f: f.__name__)
@pytest.mark.parametrize("phase_noise", ["baseline", "triangle"])
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


@pytest.mark.xfail(
    strict=True,
    reason="laplace_cov cannot mix array-valued parameter paths (the rim's "
    "az_amps, az_pas) with scalar ones",
)
def test_laplace_cov_with_array_parameters(tmp_path):
    from virgil.inference import laplace_cov

    scene = vb.star_rim()
    path = tmp_path / "s.fits"
    simulate.observe(
        path, scene.vis, UTS, hour_angles_h=HA, wavelengths=WL, dec_deg=DEC,
        sigma_v2=0.02, sigma_cp_deg=1.0,
    )
    data = vb.load(path)
    values = vb.flat_truth(scene)
    cov = laplace_cov(values, list(scene.truth), data, scene.template)
    assert np.all(np.isfinite(np.diag(np.asarray(cov))))
