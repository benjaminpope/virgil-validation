"""scripts/postagb_imaging.py: the position-angle convention of region_metrics (North up, East left)."""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
pytest.importorskip("virgil")
import postagb_imaging as P  # noqa: E402

# Guards our own script's convention, not virgil against an independent result.
pytestmark = pytest.mark.validates("evidence", roots=["standards"], kind="guard")


@pytest.mark.parametrize("pa", [20.0, 80.0, 130.0, 170.0])
def test_major_axis_pa_is_east_of_north(pa):
    n, scale = 129, 0.5
    yy, xx = np.mgrid[:n, :n]
    east = -(xx - (n - 1) / 2) * scale  # East is left (image columns decrease towards East)
    north = -(yy - (n - 1) / 2) * scale  # row 0 is North
    t = np.radians(pa)
    along = east * np.sin(t) + north * np.cos(t)
    across = east * np.cos(t) - north * np.sin(t)
    img = np.exp(-0.5 * ((along / 6.0) ** 2 + (across / 2.0) ** 2))
    m = P.region_metrics(img, scale, 0.0, 30.0)
    assert m["major_pa_deg"] == pytest.approx(pa, abs=0.5)


def test_centroid_pa_points_to_the_offset_blob():
    n, scale = 129, 0.5
    yy, xx = np.mgrid[:n, :n]
    img = np.exp(-0.5 * (((xx - 64 + 8) ** 2 + (yy - 64 + 8) ** 2) / 4.0))  # up-left on the array: North-East
    m = P.region_metrics(img, scale, 0.0, 30.0)  # up = North, left = East: PA 45 deg
    assert m["centroid_pa_deg"] == pytest.approx(45.0, abs=1.0)
