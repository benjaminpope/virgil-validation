"""Re-centring of reconstructed contest images on their flux centroid
(crosscheck.image_metrics.centre_on_centroid), applied before display and
scoring: an off-centre Gaussian comes back centred with its flux kept, and an
already centred image is untouched. NumPy only."""

import numpy as np
import pytest

from crosscheck.image_metrics import centre_on_centroid, centroid


def gaussian(n, cx, cy, sigma=3.0):
    rows, cols = np.indices((n, n))
    return np.exp(-0.5 * ((cols - cx) ** 2 + (rows - cy) ** 2) / sigma**2)


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
@pytest.mark.parametrize("order", [1, 3])
def test_off_centre_gaussian_is_centred_with_flux_kept(order):
    img = gaussian(65, 36.37, 27.81)
    moved, (dx, dy) = centre_on_centroid(img, order=order)
    cx, cy = centroid(moved)
    assert cx == pytest.approx(32.0, abs=0.05) and cy == pytest.approx(32.0, abs=0.05)
    assert moved.sum() == pytest.approx(img.sum(), rel=1e-9)
    assert dx == pytest.approx(32.0 - 36.37, abs=1e-3) and dy == pytest.approx(32.0 - 27.81, abs=1e-3)


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_negative_ringing_does_not_drag_the_centroid():
    img = gaussian(65, 40.0, 24.0) - 0.05 * gaussian(65, 10.0, 55.0, sigma=6.0)
    moved, _ = centre_on_centroid(img)
    cx, cy = centroid(moved)
    assert cx == pytest.approx(32.0, abs=0.05) and cy == pytest.approx(32.0, abs=0.05)


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_centred_image_is_unchanged():
    img = gaussian(65, 32.0, 32.0)
    moved, shift = centre_on_centroid(img)
    assert shift == (0.0, 0.0)
    np.testing.assert_array_equal(moved, img)
