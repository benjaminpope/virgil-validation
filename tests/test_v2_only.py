"""Visibility-only data (virgil#158): V² files with no OI_T3.

Our simulator writes V² alone; virgil reads it and fits a uniform-disk
diameter from the visibilities; the reference is the injected truth (our
closed form), PMOIRED's fit of the same file, and the pull statistics.
"""

import numpy as np
import numpyro.distributions as dist
import pytest

from crosscheck import simulate, sky
from evidence.plugin import record

vm = pytest.importorskip("virgil.models")
from virgil.fitting import fit  # noqa: E402
from virgil.inference import laplace_cov  # noqa: E402
from virgil.oidata import OIData  # noqa: E402

pytestmark = pytest.mark.x64

UTS = np.array(
    [[-9.925, -20.335], [14.887, 30.502], [44.915, 66.183], [103.306, 43.999]]
)
HA = np.linspace(-3, 3, 5)
WL = np.linspace(1.5e-6, 2.4e-6, 6)
DIAM = 1.8


def _v2_file(path, rng=None, sigma=0.01):
    simulate.observe(
        path, lambda u, v, w: sky.vis_uniform_disk(u, v, w, DIAM), UTS,
        hour_angles_h=HA, wavelengths=WL, dec_deg=-50.0, sigma_v2=sigma,
        sigma_cp_deg=1.0, rng=rng, closure_phases=False,
    )
    return path


def _virgil_fit(path):
    data = OIData(str(path))
    result = fit(vm.UniformDisk(1.5), {"diam": dist.Uniform(0.1, 10.0)}, data)
    d = float(result.values["diam"])
    cov = np.asarray(laplace_cov(np.array([d]), ["diam"], data, result.model))
    return data, d, float(np.sqrt(cov[0, 0]))


@pytest.mark.validates("virgil.oifits.read_oifits", "virgil.oidata.OIData", roots=["standards"])
def test_a_v2_only_file_is_read(tmp_path):
    from astropy.io import fits

    path = _v2_file(tmp_path / "v2.fits")
    with fits.open(path) as h:
        assert "OI_T3" not in [x.name for x in h]
    data = OIData(str(path))
    assert not data.has_phases
    observed, _ = data.flatten_data()
    assert observed.size == 6 * HA.size * WL.size
    np.testing.assert_allclose(observed, np.asarray(data.model(vm.UniformDisk(DIAM))), atol=1e-12)


@pytest.mark.validates("virgil.fitting.fit", "virgil.models.UniformDisk", roots=["mathematics"])
def test_noise_free_v2_only_fit_recovers_the_diameter(tmp_path):
    path = _v2_file(tmp_path / "v2.fits")
    _, d, _ = _virgil_fit(path)
    assert abs(d - DIAM) < 1e-6 * DIAM


@pytest.mark.external
@pytest.mark.validates("virgil.fitting.fit", "virgil.inference.laplace_cov", roots=["pmoired"])
def test_v2_only_fit_agrees_with_pmoired(tmp_path):
    pytest.importorskip("pmoired")
    from external_bridge.pmoired_models import fit as pmoired_fit

    path = _v2_file(tmp_path / "v2.fits", np.random.default_rng(4))
    _, d, sigma = _virgil_fit(path)
    pm = pmoired_fit(path, {"ud": 1.5}, ["ud"], obs=("V2",))
    dbest = record("dbest_sigma", (pm["best"]["ud"] - d) / sigma)
    ratio = record("sigma_ratio", pm["sigma"]["ud"] / sigma)
    assert abs(dbest) < 0.05
    assert abs(ratio - 1) < 0.03


@pytest.mark.slow
@pytest.mark.validates("virgil.inference.laplace_cov", "virgil.fitting.fit", roots=["statistics"], tier="B")
def test_v2_only_diameter_pulls(tmp_path):
    rng = np.random.default_rng(31)
    n = 200
    pulls = []
    for _ in range(n):
        path = _v2_file(tmp_path / "v2.fits", rng, sigma=0.02)
        _, d, sigma = _virgil_fit(path)
        pulls.append((d - DIAM) / sigma)
    pulls = np.array(pulls)
    record("pull_mean", pulls.mean())
    record("pull_sd", pulls.std())
    assert abs(pulls.mean()) < 3 / np.sqrt(n)
    assert abs(pulls.std() - 1) < 3 / np.sqrt(2 * n)


@pytest.mark.validates("virgil.oifits.read_oifits", "virgil.oidata.OIData", roots=["standards"])
def test_two_telescopes_give_v2_alone(tmp_path):
    """A two-telescope array has no closure phases at all."""
    path = tmp_path / "two.fits"
    simulate.observe(
        path, lambda u, v, w: sky.vis_uniform_disk(u, v, w, DIAM), UTS[:2],
        hour_angles_h=HA, wavelengths=WL, dec_deg=-50.0, sigma_v2=0.01,
        sigma_cp_deg=1.0, closure_phases=False,
    )
    data = OIData(str(path))
    assert not data.has_phases
    assert data.flatten_data()[0].size == HA.size * WL.size
    _, d, _ = _virgil_fit(path)
    assert abs(d - DIAM) < 1e-6 * DIAM
