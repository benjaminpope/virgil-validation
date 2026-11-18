"""Aperture masking: dLux images -> calibrated observables -> virgil fits.

Each scene's point cloud is imaged through a NIRISS-like 7-hole mask, both
by dLux and by the closed-form interferogram (Airy envelope x fringes). The
two imagers must agree with each other and, once calibrated by a point
source, with the scene's exact visibility. The dLux observables then go
through our OIFITS writer to virgil, which must recover the scene
(tests/test_nrm_injection.py; it goes through crosscheck.simulate). The calibrated-visibility
comparison on each scene is in tests/test_nrm_calibrated.py (the scenes come from virgil_bridge);
the checks of crosscheck.nrm below use neither, so what they went through is only crosscheck.nrm.
"""

import numpy as np
import pytest

from _nrm_common import PIX, UV, WL
from crosscheck import nrm, sky

pytestmark = pytest.mark.x64


@pytest.mark.validates("crosscheck.nrm", roots=["mathematics"], kind="reference")
def test_mask_is_non_redundant():
    assert nrm.is_non_redundant(nrm.HOLES, nrm.HOLE_DIAM)


@pytest.mark.validates("crosscheck.nrm", roots=["dlux"], kind="reference")
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
