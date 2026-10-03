"""Aperture masking: dLux images -> calibrated observables -> virgil fits.

Each scene's point cloud is imaged through a NIRISS-like 7-hole mask, both
by dLux and by the closed-form interferogram (Airy envelope x fringes). The
two imagers must agree with each other and, once calibrated by a point
source, with the scene's exact visibility. The dLux observables then go
through our OIFITS writer to virgil, which must recover the scene.
"""

import numpy as np
import pytest

from crosscheck import array, nrm, simulate, sky

vb = pytest.importorskip("virgil_bridge")

pytestmark = pytest.mark.x64
WL = float(np.float32(4.8e-6))  # as OIFITS will store it
PIX = 30.0
UV = array.pupil_uv(nrm.HOLES)


def test_mask_is_non_redundant():
    assert nrm.is_non_redundant(nrm.HOLES, nrm.HOLE_DIAM)


def test_dlux_orientation():
    """dLux images a source offset by (x, y) at larger column and row
    respectively, centred on pixel (n - 1) / 2: the convention nrm.py uses.
    Checked with the calibrated phase slope, which a centroid (biased by
    the field edge) cannot match."""
    optics = nrm.dlux_optics(npix=128)
    cal = nrm.image_dlux(sky.point(), WL, optics)
    sci = nrm.image_dlux(sky.point(40.0, -25.0), WL, optics)
    got = nrm.calibrated_vis_fn(sci, cal, WL, PIX)(*UV, WL)
    want = sky.vis_point(*UV, WL, 40.0, -25.0)
    # a flipped axis would give O(1) errors; the small field leaves ~1e-2
    assert np.max(np.abs(got - want)) < 3e-2


@pytest.fixture(scope="module")
def images():
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


def _scene_ids():
    return [s.name for s in vb.masking_scenes()]


@pytest.mark.slow
@pytest.mark.parametrize("k", range(4), ids=_scene_ids())
def test_calibrated_visibilities(images, k):
    scene = vb.masking_scenes()[k]
    exact = scene.vis(*UV, WL)
    cal_d, cal_a = images["cal"]
    sci_d, sci_a = images[scene.name]
    v_d = nrm.calibrated_vis_fn(sci_d, cal_d, WL, PIX)(*UV, WL)
    v_a = nrm.calibrated_vis_fn(sci_a, cal_a, WL, PIX)(*UV, WL)
    # the point cloud is the scene
    assert np.max(np.abs(sky.visibility(scene.cloud, *UV, WL) - exact)) < 1e-8
    # light lost beyond a 7.7" field limits both imagers to ~1e-4
    assert np.max(np.abs(v_d - exact)) < 5e-4
    assert np.max(np.abs(v_a - exact)) < 5e-4


@pytest.mark.slow
@pytest.mark.parametrize("k", range(4), ids=_scene_ids())
def test_dlux_injection_recovery(images, tmp_path, k):
    """Noise-free dLux observables, fitted by virgil: the bias left by the
    field edge must be far below realistic errors (1 deg closure phases,
    0.02 in V^2), so we compare it with those Laplace sigmas."""
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
    if cov is not None:
        bias = (got - truth) / np.sqrt(np.diag(cov))
        assert np.all(np.abs(bias) < 0.05), bias
    np.testing.assert_allclose(got, truth, rtol=2e-2, atol=1e-3)
