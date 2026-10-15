"""Projected-orbit helpers of crosscheck.orbits (design/orbit_comparison.md): Thiele-Innes, the
Thiele-Innes track against the Kepler position, and folding. Numpy only.

The statistics A and B and `compare_orbits` that once lived beside them were deleted: nothing but their own tests used
them, and the sky-plane comparison is `crosscheck.skyplane` (tests/test_skyplane.py)."""

import numpy as np
import pytest

from crosscheck import orbits as co

BASE = dict(P=800.0, e=0.45, t_p=120.0, a=100.0, inc=62.0, omega=40.0, Omega=110.0)


@pytest.mark.validates("crosscheck.orbits", roots=["mathematics"], kind="reference")
@pytest.mark.parametrize("e", [0.0, 0.3, 0.8])
def test_thiele_innes_reproduces_sky_position(e):
    """A, F give dDec and B, G give dRA, exactly as sky_position."""
    f = np.linspace(-3, 3, 13)
    a, inc, w, W = 7.0, 55.0, 130.0, 250.0
    A, B, F, G = co.thiele_innes(a, inc, w, W)
    E = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(f / 2), np.sqrt(1 + e) * np.cos(f / 2))
    x, y = np.cos(E) - e, np.sqrt(1 - e**2) * np.sin(E)
    dra, ddec = co.sky_position(f, e, inc, w, W, a)
    assert np.allclose(A * x + F * y, ddec, atol=1e-12)
    assert np.allclose(B * x + G * y, dra, atol=1e-12)


@pytest.mark.validates("crosscheck.orbits", roots=["mathematics"], kind="reference")
@pytest.mark.parametrize("e", [0.45, 0.85, 0.95])
def test_predict_track_matches_position(e):
    s = {k: np.array([v]) for k, v in {**BASE, "e": e}.items()}
    t = BASE["t_p"] + np.linspace(-0.6, 0.6, 41) * BASE["P"]  # spans periastron
    got = co.predict_track(s, t)[0]
    dra, ddec = co.position(t, BASE["P"], BASE["t_p"], e, BASE["inc"],
                            BASE["omega"], BASE["Omega"], BASE["a"])
    assert np.allclose(got[:, 0], dra, atol=1e-9) and np.allclose(got[:, 1], ddec, atol=1e-9)


@pytest.mark.validates("crosscheck.orbits", roots=["mathematics"], kind="reference")
def test_the_twin_is_one_sky_orbit_and_one_shift_is_the_reflection():
    """(Omega + 180, omega + 180) leaves A, B, F, G unchanged; omega + 180 alone, or Omega + 180 alone, negates them."""
    a, inc, w, W = 7.0, 55.0, 130.0, 250.0
    base = np.array(co.thiele_innes(a, inc, w, W))
    assert np.allclose(co.thiele_innes(a, inc, w + 180.0, W + 180.0), base, atol=1e-12)
    assert np.allclose(co.thiele_innes(a, inc, w + 180.0, W), -base, atol=1e-12)
    assert np.allclose(co.thiele_innes(a, inc, w, W + 180.0), -base, atol=1e-12)


@pytest.mark.validates("crosscheck.orbits", roots=["mathematics"], kind="reference")
def test_fold_and_mirror_fraction():
    rng = np.random.default_rng(0)
    W = np.where(rng.random(5000) < 0.3, 110.0 + 180.0, 110.0) + 5 * rng.standard_normal(5000)
    w = 40.0 + 0 * W + np.where(W > 200, 180.0, 0.0)
    assert co.mirror_fraction(W, 110.0) == pytest.approx(0.3, abs=0.03)
    wf, Wf, _ = co.fold_samples(w, W, 110.0)
    assert np.all(np.abs(Wf - 110.0) < 90) and np.allclose(wf, 40.0)

