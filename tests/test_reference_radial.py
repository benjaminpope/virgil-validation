"""crosscheck.radial, checked before it judges virgil: the composite Gauss-Legendre Hankel
transform against adaptive quadrature. Split from tests/test_polar_disk_brute_force.py, whose
other checks (crosscheck.polar_images, virgil's MultipoleDisk) use crosscheck.radial."""

import numpy as np
import pytest

from crosscheck import radial as R
from evidence.plugin import record

pytestmark = pytest.mark.x64


@pytest.mark.validates("crosscheck.radial", roots=["mathematics"], kind="reference", headline="max_abs_dV")
def test_hankel_truth_matches_adaptive_quadrature():
    f = R.with_gap(R.erf_power(5.0, 14.0, -1.0, 0.3), 9.0, 0.8, 0.9)
    q = np.array([1e5, 1e6, 8e6])
    err = record("max_abs_dV", np.abs(R.hankel(f, q) - R.hankel_quad(f, q)).max())
    assert err < 1e-9
