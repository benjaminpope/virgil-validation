"""Calibrated dLux and analytic visibilities against each scene's exact visibility (crosscheck.nrm).
Split from tests/test_nrm.py: the scenes come from virgil_bridge, which the mask and orientation
checks of crosscheck.nrm do not use."""

import numpy as np
import pytest

from _nrm_common import PIX, UV, WL
from _nrm_scenes import dlux_and_analytic_images, scene_ids, vb
from crosscheck import nrm, sky

pytestmark = pytest.mark.x64


@pytest.fixture(scope="module")
def images():
    """Calibrator and science images, dLux and analytic, per scene."""
    return dlux_and_analytic_images()


def _scene_ids():
    return scene_ids()


@pytest.mark.slow
@pytest.mark.parametrize("k", range(4), ids=_scene_ids())
@pytest.mark.validates("crosscheck.nrm", roots=["dlux", "mathematics"], kind="reference", tier="B")
def test_calibrated_visibilities(images, k):
    scene = vb.masking_scenes()[k]
    exact = scene.vis(*UV, WL)
    cal_d, cal_a = images["cal"]
    sci_d, sci_a = images[scene.name]
    v_d = nrm.calibrated_vis_fn(sci_d, cal_d, WL, PIX)(*UV, WL)
    v_a = nrm.calibrated_vis_fn(sci_a, cal_a, WL, PIX)(*UV, WL)
    # the point cloud is the scene
    assert np.max(np.abs(sky.visibility(scene.cloud, *UV, WL) - exact)) < 1e-8
    # light lost beyond a 7.7" field limits both imagers to 1e-4 - 8e-4
    # (largest for the rim, the most extended scene)
    assert np.max(np.abs(v_d - exact)) < 1e-3
    assert np.max(np.abs(v_a - exact)) < 1e-3
