"""crosscheck.array, checked before it judges virgil: uv from Thompson, Moran & Swenson's
matrices against the baseline projected onto the sky-plane East and North directions, which
are built from the spherical-trigonometry source direction (a different route to the same
geometry). One file per reference, importing only the references its checks use, so that
what a check went through (scripts/trust.py, ``via``) is not the whole file's imports."""

import numpy as np
import pytest

from crosscheck import array
from evidence.plugin import record

UTS4 = np.array([[-9.925, -20.335, 0.0], [14.887, 30.502, 0.0], [44.915, 66.183, 0.0], [103.306, 43.999, 0.0]])


@pytest.mark.parametrize("lat,dec", [(-24.6, -50.0), (-24.6, 10.0), (34.2, 60.0), (0.0, 0.0)])
@pytest.mark.validates("crosscheck.array", roots=["mathematics"], kind="reference")
def test_uv_against_the_sky_plane_projection(lat, dec):
    phi, d = np.deg2rad(lat), np.deg2rad(dec)
    worst = 0.0
    for ha in np.linspace(-4, 4, 9):
        h = np.deg2rad(15 * ha)
        # East, North, Up components of the unit vectors on the sky at the source
        d_dh = np.array([-np.cos(d) * np.cos(h), np.sin(phi) * np.cos(d) * np.sin(h), -np.cos(phi) * np.cos(d) * np.sin(h)])
        east = -d_dh / np.cos(d)  # increasing right ascension is decreasing hour angle
        north = np.array([np.sin(d) * np.sin(h), np.cos(phi) * np.cos(d) + np.sin(phi) * np.sin(d) * np.cos(h),
                          np.sin(phi) * np.cos(d) - np.cos(phi) * np.sin(d) * np.cos(h)])
        u, v = array.snapshot_uv(UTS4, ha, dec, lat)
        b = np.array([UTS4[j] - UTS4[i] for i, j in array.baselines(4)])
        worst = max(worst, np.max(np.abs(u - b @ east)), np.max(np.abs(v - b @ north)))
    record("max_abs_duv_m", worst)
    assert worst < 1e-9
