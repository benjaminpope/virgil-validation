"""external_bridge.cv_folds, numpy only: matching asdf points to npz points whatever the channel order, the
profiled score, the tallies. Split from tests/test_alma_mpol_cv.py, whose other checks go through
external_bridge.cv_compare, external_bridge.mpol_worker and virgil_bridge (the checks of cv_folds do not)."""

import numpy as np
import pytest

from external_bridge import cv_folds


def _points(rng, n=60, nchan=4):
    """Keyed points of one ddid: ``time`` unique per row, channels of ascending frequency, random flags."""
    row, ch = np.nonzero(rng.uniform(size=(n, nchan)) > 0.15)
    freqs = 2.3e11 + 4e7 * np.arange(nchan)
    ant = np.stack([rng.integers(0, 20, n), rng.integers(20, 40, n)], 1)
    return dict(freq=freqs[ch], time=1.0e9 + 6.0 * row, ant1=ant[row, 0], ant2=ant[row, 1], u=rng.normal(size=len(row)) * 1e6, v=rng.normal(size=len(row)) * 1e6,
                weight=np.exp(rng.normal(8, 0.3, len(row))))


@pytest.mark.validates("external_bridge.cv_folds", roots=["mathematics"], kind="reference")
def test_match_index_pairs_the_points_whatever_the_order_of_channels_and_rows():
    """The asdf's points (channel-major, frequency descending, as visread flips an ascending MS) are paired with the npz's
    (row-major, ascending) by their keys: each gets the npz point it was made from."""
    rng = np.random.default_rng(0)
    b = _points(rng)
    order = np.lexsort((b["time"], -b["freq"]))  # channel-major, descending frequency: what loaddata returns
    a = {k: v[order] for k, v in b.items()}
    idx, bad = cv_folds.match_index(a, b)
    assert bad == 0
    np.testing.assert_array_equal(idx, order)  # a[i] is b[order[i]]
    shuffled = rng.permutation(len(order))
    idx2, bad2 = cv_folds.match_index({k: v[shuffled] for k, v in a.items()}, b)
    assert bad2 == 0 and np.array_equal(idx2, order[shuffled])


@pytest.mark.validates("external_bridge.cv_folds", roots=["mathematics"], kind="control")
def test_match_index_and_labels_refuse_data_that_are_not_the_same_points():
    """A different count raises; a time or antenna that differs is counted and ``labels_for_ddid`` raises; so does a u that is off by
    1e-3 (a wrong pairing would put points in the wrong fold); missing keys raise instead of assuming the order."""
    rng = np.random.default_rng(1)
    b = _points(rng)
    with pytest.raises(ValueError, match="cannot match"):
        cv_folds.match_index({k: v[:-1] for k, v in b.items()}, b)
    a = {k: v.copy() for k, v in b.items()}
    a["ant1"][3] += 1
    _, bad = cv_folds.match_index(a, b)
    assert bad > 0
    folds = {"fold_07": np.zeros(len(b["u"]), np.int8), "annulus_07": np.zeros(len(b["u"]), np.int8)}
    with pytest.raises(ValueError, match="channel, time or antennas"):
        cv_folds.labels_for_ddid(a, folds, 7, b)
    a = {k: v.copy() for k, v in b.items()}
    a["u"] = a["u"] * (1 + 1e-3)
    with pytest.raises(ValueError, match="differ beyond"):
        cv_folds.labels_for_ddid(a, folds, 7, b)
    cv_folds.labels_for_ddid({k: v.copy() for k, v in b.items()}, folds, 7, b)  # the same points pass
    with pytest.raises(ValueError, match="lack"):
        cv_folds.match_index({k: v for k, v in b.items() if k != "time"}, b)


@pytest.mark.validates("external_bridge.cv_folds", roots=["mathematics"], kind="reference")
def test_profiled_score_is_the_minimum_over_a_global_error_scale():
    """``profiled`` equals the minimum over s of w|r|^2/(2 s^2) + 2 ln s + ln 2 pi - ln w, and s_hat^2 = chi2 / 2n."""
    rng = np.random.default_rng(2)
    n = 500
    w = np.exp(rng.normal(3, 0.5, n))
    r2 = w * rng.exponential(1.0, n) * 3.0
    s_grid = np.linspace(0.5, 6, 200001)
    ll = np.array([np.sum(0.5 * r2 / s**2 + 2 * np.log(s) + cv_folds.LN2PI - np.log(w)) for s in s_grid[::200]])
    nll_s1 = np.sum(0.5 * r2 + cv_folds.LN2PI - np.log(w))
    prof, s_hat = cv_folds.profiled(n, nll_s1, r2.sum())
    assert prof == pytest.approx(ll.min(), rel=1e-6) and prof <= ll.min() + 1e-9
    assert s_hat == pytest.approx(np.sqrt(r2.sum() / (2 * n)))
    assert np.isnan(cv_folds.profiled(0, 0.0, 0.0)[0])


@pytest.mark.validates("external_bridge.cv_folds", roots=["mathematics"], kind="reference")
def test_tally_layout_is_virgils_score_layout():
    t = cv_folds.tally([0, 2, 2, 1], 4, [1.0, 2.0, 3.0, 4.0], [0.5, 1.0, 1.5, 2.0], [5.0, 6.0, 7.0, 8.0])
    assert t == dict(n=[1, 1, 2, 0], nll=[1.0, 4.0, 5.0, 0.0], chi2_raw=[0.5, 2.0, 2.5, 0.0], chi2_fit=[5.0, 8.0, 13.0, 0.0])
    assert cv_folds.add_tallies(t, t)["n"] == [2, 2, 4, 0]
