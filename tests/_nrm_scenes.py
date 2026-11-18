"""Shared by tests/test_nrm_calibrated.py and tests/test_nrm_injection.py: the dLux and analytic images
of each masking scene, whose definitions live in virgil_bridge."""

import pytest

from _nrm_common import WL
from crosscheck import nrm, sky

vb = pytest.importorskip("virgil_bridge")


def dlux_and_analytic_images():
    """Calibrator and science images, dLux and analytic, per scene."""
    optics = nrm.dlux_optics(npix=256)
    out = {
        "cal": (
            nrm.image_dlux(sky.point(), WL, optics),
            nrm.image_analytic(sky.point(), WL, npix=256),
        )
    }
    for scene in vb.masking_scenes():
        out[scene.name] = (
            nrm.image_dlux(scene.cloud, WL, optics),
            nrm.image_analytic(scene.cloud, WL, npix=256),
        )
    return out


def scene_ids():
    return [s.name for s in vb.masking_scenes()]
