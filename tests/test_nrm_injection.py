"""Noise-free dLux observables through our OIFITS writer to virgil, which must recover the scene
(pipeline dlux-masking-fit). Split from tests/test_nrm.py: this check goes through
crosscheck.simulate, the checks of crosscheck.nrm itself do not."""

import numpy as np
import pytest

from _nrm_common import PIX, UV, WL
from _nrm_scenes import dlux_and_analytic_images, scene_ids, vb
from crosscheck import nrm, simulate
from evidence.plugin import record

pytestmark = pytest.mark.x64


@pytest.fixture(scope="module")
def images():
    """Calibrator and science images, dLux and analytic, per scene."""
    return dlux_and_analytic_images()


def _scene_ids():
    return scene_ids()


@pytest.mark.slow
@pytest.mark.parametrize("k", range(4), ids=_scene_ids())
@pytest.mark.validates("pipeline:dlux-masking-fit", "virgil.fitting.fit", "virgil.models.System", roots=["dlux"], tier="B")
def test_dlux_injection_recovery(images, tmp_path, k):
    """Noise-free dLux observables, fitted by virgil: the bias left by the
    field edge must be far below realistic errors (1 deg closure phases,
    0.02 in V^2): under 0.1 of those Laplace sigmas (0.05 at most so far,
    for the rim)."""
    scene = vb.masking_scenes()[k]
    cal_d, _ = images["cal"]
    sci_d, _ = images[scene.name]
    vis = nrm.calibrated_vis_fn(sci_d, cal_d, WL, PIX)
    path = tmp_path / "nrm.fits"
    simulate.observe(
        path, vis, nrm.HOLES, hour_angles_h=[0.0], wavelengths=[WL],
        fixed_uv=UV, sigma_v2=0.02, sigma_cp_deg=1.0,
    )
    data = vb.load(path)
    result, cov = vb.fit_scene(scene, data)
    got = vb.flat_values(scene, result.values)
    truth = vb.flat_truth(scene)
    # judged in units of virgil's own uncertainty: a weakly constrained
    # parameter (the rim's fwhm, sigma ~ 80 mas) may be off by a few
    # per cent and still be negligibly biased
    bias = record("max_abs_bias_sigma", np.max(np.abs((got - truth) / np.sqrt(np.diag(cov)))))
    assert bias < 0.1, (got, truth)
