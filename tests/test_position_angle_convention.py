"""(dra, ddec) <-> (sep, position angle): the angle runs North through East.

virgil documents the position angle as North (0) through East (90): ``(0, 1)`` is
0 deg, ``(1, 0)`` is 90 deg, ``(-1, 0)`` is 270 deg, the inverse of
``(sep sin pa, sep cos pa)``. Every other test goes the forward way, so the slip
``arctan2(ddec, dra)`` (an angle from East; mutant M11) was invisible to the
tests of BinaryModelCartesian. The convention is only defined by virgil's own
documentation here (and the astronomical standard it agrees with); it is on the
sign-off list (trust/signoffs.yml, written by scripts/mutation_trust.py).
"""

import numpy as np
import pytest

vm = pytest.importorskip("virgil.models")

pytestmark = pytest.mark.x64


@pytest.mark.validates("virgil.models.BinaryModelCartesian", roots=["mathematics"])
def test_cartesian_to_angular_reads_the_position_angle_north_through_east():
    cases = [((0.0, 2.0), 0.0), ((3.0, 0.0), 90.0), ((0.0, -4.0), 180.0), ((-5.0, 0.0), 270.0),
             ((1.0, 1.0), 45.0), ((-1.0, 1.0), 315.0), ((1.0, -np.sqrt(3.0)), 150.0)]
    for (dra, ddec), pa in cases:
        angular = vm.BinaryModelCartesian(dra, ddec, 0.1).to_angular()
        assert float(angular.sep) == pytest.approx(np.hypot(dra, ddec), rel=1e-12)
        assert float(angular.pa) == pytest.approx(pa, abs=1e-9), (dra, ddec)
        back = angular.to_cartesian()
        assert (float(back.dra), float(back.ddec)) == pytest.approx((dra, ddec), abs=1e-12)
