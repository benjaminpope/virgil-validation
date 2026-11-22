"""crosscheck.orbits, checked before it judges virgil: Kepler's equation is solved, the
radial velocity is the rate of change of depth, and positions equal the Thiele-Innes form
of the textbook visual binary, which uses the eccentric anomaly and never the true anomaly."""

import numpy as np
import pytest

from crosscheck import orbits
from evidence.plugin import record


@pytest.mark.validates("crosscheck.orbits", roots=["mathematics"], kind="reference")
def test_kepler_and_thiele_innes():
    mean = np.linspace(-np.pi, np.pi, 41)
    for e in (0.0, 0.3, 0.7, 0.95):
        E = orbits.eccentric_anomaly(mean, e)
        assert np.max(np.abs(E - e * np.sin(E) - mean)) < 1e-12
    worst = 0.0
    for e, i, w, W in [(0.3, 60.0, 40.0, 110.0), (0.8, 130.0, 250.0, 20.0), (0.0, 0.0, 0.0, 300.0), (0.5, 90.0, 10.0, 75.0)]:
        E = orbits.eccentric_anomaly(mean, e)
        X, Y = np.cos(E) - e, np.sqrt(1 - e**2) * np.sin(E)
        iw, ww, WW = np.deg2rad([i, w, W])
        # Thiele-Innes constants (a = 1): North = A X + F Y, East = B X + G Y
        A = np.cos(ww) * np.cos(WW) - np.sin(ww) * np.sin(WW) * np.cos(iw)
        B = np.cos(ww) * np.sin(WW) + np.sin(ww) * np.cos(WW) * np.cos(iw)
        F = -np.sin(ww) * np.cos(WW) - np.cos(ww) * np.sin(WW) * np.cos(iw)
        G = -np.sin(ww) * np.sin(WW) + np.cos(ww) * np.cos(WW) * np.cos(iw)
        dra, ddec = orbits.sky_position(orbits.true_anomaly(mean, e), e, i, w, W)
        worst = max(worst, np.max(np.abs(ddec - (A * X + F * Y))), np.max(np.abs(dra - (B * X + G * Y))))
    record("max_abs_dpos", worst)
    assert worst < 1e-12
    # face-on circular orbit: prograde means position angle increases (North through East)
    dra, ddec = orbits.sky_position(np.array([0.0, 0.1]), 0.0, 0.0, 0.0, 0.0)
    assert np.arctan2(dra[1], ddec[1]) > np.arctan2(dra[0], ddec[0])


@pytest.mark.validates("crosscheck.orbits", roots=["mathematics"], kind="reference")
def test_radial_velocity_is_the_rate_of_recession():
    """The closed-form RV is the time derivative of z (by central differences
    of our own positions), and the companion recedes through the ascending node."""
    worst = 0.0
    for period, t_peri, e, i, w in [(800.0, 120.0, 0.3, 60.0, 40.0), (40.0, 3.0, 0.75, 130.0, 300.0), (90.0, 0.0, 0.0, 35.0, 0.0)]:
        t, h = np.linspace(0.0, 2 * period, 97), 1e-5 * period
        mean = [np.mod(2 * np.pi * (tt - t_peri) / period + np.pi, 2 * np.pi) - np.pi for tt in (t - h, t + h)]
        z = [orbits.depth(orbits.true_anomaly(m, e), e, i, w, 3.0) for m in mean]
        numeric = (z[1] - z[0]) / (2 * h)
        rv = orbits.radial_velocity(t, period, t_peri, e, i, w, 3.0)
        worst = max(worst, np.max(np.abs(rv - numeric)) / np.max(np.abs(rv)))
    record("max_rel_drv", worst)
    assert worst < 1e-5
    # circular orbit, omega = 0: at periastron (u = 0, the ascending node) z = 0 and rising
    assert orbits.radial_velocity(np.array([0.0]), 90.0, 0.0, 0.0, 35.0, 0.0, 1.0)[0] > 0
    assert orbits.depth(np.array([0.1]), 0.0, 35.0, 0.0)[0] > 0
