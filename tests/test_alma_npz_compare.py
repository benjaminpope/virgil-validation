"""``external_bridge.npz_compare.compare_ddid``, the PASS/FAIL behind ``asdf_vs_npz`` (numpy only).

The IM Lup MS has ddids whose channels run down in frequency in the MS (our npz keeps that order) while
visread, hence the asdf, numbers them up; and the DATA and CORRECTED_DATA columns differ by up to 1.6 Jy.
The comparison must match channels independent of order and must fail on a different column."""
import numpy as np
import pytest

from external_bridge.npz_compare import compare_ddid

pytestmark = pytest.mark.validates("external_bridge.mpol_worker", roots=["mpol"], kind="control")


def _ddid(seed=0, nchan=4, nrow=30):
    """An npz-like reading (MS channel order: frequency decreasing) and the asdf-like reading of the same points."""
    rng = np.random.default_rng(seed)
    freq = 2.3e11 - np.arange(nchan) * 1e6          # MS order: decreasing
    ch, rw = np.meshgrid(np.arange(nchan), np.arange(nrow), indexing="ij")
    ch, rw = ch.ravel(), rw.ravel()
    ant1, ant2 = rng.integers(0, 20, nrow), rng.integers(20, 40, nrow)
    time = np.sort(rng.random(nrow)) * 1e3 + 5e9
    uv = rng.normal(0, 1e5, (2, nrow))
    vis = rng.normal(size=ch.size) + 1j * rng.normal(size=ch.size)
    w = rng.uniform(1, 5, ch.size)
    npz = dict(u=uv[0][rw] * freq[ch] / 3e8, v=uv[1][rw] * freq[ch] / 3e8, vis=vis, weight=w, freq=freq[ch],
               ant1=ant1[rw], ant2=ant2[rw], time=time[rw], chan=ch)
    asc = nchan - 1 - ch                              # visread: ascending frequency
    order = np.lexsort((rw, asc))                     # and its own point order
    asdf = {k: np.array(v[order]) for k, v in npz.items() if k != "freq"}
    asdf["chan"] = asc[order]
    return asdf, npz


def test_passes_when_the_channel_order_is_reversed_and_the_points_are_shuffled():
    a, b = _ddid()
    r = compare_ddid(a, b)
    assert r["matching"] == "keys" and r["key_mismatch"] == 0 and r["pass"], r
    assert r["vis_max_abs"] == 0.0


def test_fails_on_a_different_data_column():
    a, b = _ddid()
    a["vis"] = a["vis"] + 0.03                       # the size of the ddid-0 DATA vs CORRECTED_DATA difference
    assert not compare_ddid(a, b)["pass"]


def test_fails_on_a_tiny_vis_difference_above_the_tolerance_and_passes_below_it():
    a, b = _ddid()
    a["vis"] = a["vis"] + 2e-5
    assert not compare_ddid(a, b)["pass"]
    a, b = _ddid()
    a["vis"] = a["vis"] + 2e-6
    assert compare_ddid(a, b)["pass"]


def test_fails_when_a_key_differs():
    a, b = _ddid()
    a["ant2"] = a["ant2"].copy()
    a["ant2"][0] += 1
    r = compare_ddid(a, b)
    assert r["key_mismatch"] > 0 and not r["pass"]


def test_a_conjugated_reading_is_reported_not_passed():
    a, b = _ddid()
    a["vis"] = np.conj(a["vis"])
    r = compare_ddid(a, b)
    assert not r["pass"] and r["vis_conj_max_abs"] < 1e-12
