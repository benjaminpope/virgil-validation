"""Gradient CLEAN must stay finite (finding F12, fixed in virgil#190).

Before virgil#190, ``virgil.imaging.clean`` without a base scene returned
a NaN χ² at its first iteration on some grids when the closure phases were
correlated (four or more telescopes). The file is from our simulator: four
VLTI UTs, a binary, seven snapshots. On #174's parent no grid failed; from
its merge until #190, every even size from 34 to 68 at 0.4 mas did. Three
telescopes (uncorrelated closure phases) never failed and are the control.
"""

import numpy as np
import pytest

from crosscheck import simulate, sky

imaging = pytest.importorskip("virgil.imaging")
if not hasattr(imaging, "clean"):
    pytest.skip("virgil without imaging.clean", allow_module_level=True)
from virgil.oidata import OIData  # noqa: E402

pytestmark = pytest.mark.x64

UTS = np.array(
    [[-9.925, -20.335], [14.887, 30.502], [44.915, 66.183], [103.306, 43.999]]
)


def _binary(u, v, w):
    return (sky.vis_point(u, v, w) + 0.3 * sky.vis_point(u, v, w, 4.0, 1.0)) / 1.3


def _data(tmp_path, stations):
    path = tmp_path / f"{len(stations)}t.fits"
    simulate.observe(
        path, _binary, stations, hour_angles_h=np.linspace(-3, 3, 7),
        wavelengths=np.array([1.65e-6]), dec_deg=-30.0, sigma_v2=0.01, sigma_cp_deg=1.0,
    )
    return OIData(str(path))


def _finite(data, npix):
    return bool(np.isfinite(np.ravel(imaging.clean(data, npix, 0.4, max_iterations=3).chi2_red)).all())


@pytest.mark.validates("virgil.imaging.clean", roots=["mathematics"])
def test_clean_is_finite_with_three_telescopes(tmp_path):
    data = _data(tmp_path, UTS[:3])
    assert all(_finite(data, n) for n in (40, 41, 60))


@pytest.mark.validates("virgil.imaging.clean", roots=["mathematics"])
def test_clean_is_finite_with_four_telescopes(tmp_path):
    data = _data(tmp_path, UTS)
    assert all(_finite(data, n) for n in (40, 41, 60))
