"""crosscheck.radio_exact: the budget constants from their formulas. Split from tests/test_toeplitz_reexam.py,
whose other checks of radio_exact build their scene with crosscheck.toeplitz_scenes (which depends on radio_exact)
and import scripts/toeplitz_reexam.py; this one uses neither."""

import math

import pytest

from crosscheck import radio_exact as rx


@pytest.mark.validates("crosscheck.radio_exact", roots=["mathematics"], kind="reference")
def test_the_budget_constants():
    """gamma_n, the FFT factor and the probabilistic bound from their formulas (Higham 2002, Lemma 3.1, Thm 24.2; Higham & Mary 2019)."""
    assert rx.U == 2.0**-53
    assert rx.gamma(10**6) == pytest.approx(1e6 * 2.0**-53, rel=1e-9)
    assert rx.fft_factor(2**20) == pytest.approx(20 * rx.ETA_FFT, rel=1e-9)
    assert rx.gamma_probabilistic(10**6) == pytest.approx(math.sqrt(2 * math.log(2e12)) * 1e3 * 2.0**-53)
    assert rx.gamma_probabilistic(10**6) < rx.gamma(10**6)
