"""``external_bridge.npz_compare.compare_ddid``, the PASS/FAIL behind ``asdf_vs_npz`` (numpy only, synthetic dicts: unit tests
of our own comparison code, no MPoL, visread or asdf involved).

The IM Lup MS has ddids whose channels run up in frequency in the MS (our npz keeps that order) while visread, hence the asdf,
numbers them in decreasing frequency (visread 0.0.5 ``process_casa.get_channel_sorted_data``); and the DATA and CORRECTED_DATA
columns differ by up to 1.6 Jy. The comparison must match channels by their own frequencies, whichever way either side runs, and
must fail on a different column."""
import numpy as np
import pytest

from external_bridge.npz_compare import compare_ddid

pytestmark = pytest.mark.validates("external_bridge.mpol_worker", roots=["self-consistency"], kind="reference")


def _ddid(seed=0, nchan=4, nrow=30, ms_decreasing=True, asdf_decreasing=True):
    """An npz-like reading (MS channel order) and the asdf-like reading of the same points, each side's channel
    axis running in the given direction, both carrying their own per-point frequencies."""
    rng = np.random.default_rng(seed)
    f_asc = 2.3e11 + np.arange(nchan) * 1e6
    ch, rw = np.meshgrid(np.arange(nchan), np.arange(nrow), indexing="ij")
    ch, rw = ch.ravel(), rw.ravel()
    ant1, ant2 = rng.integers(0, 20, nrow), rng.integers(20, 40, nrow)
    time = np.sort(rng.random(nrow)) * 1e3 + 5e9
    uv = rng.normal(0, 1e5, (2, nrow))
    vis = rng.normal(size=ch.size) + 1j * rng.normal(size=ch.size)
    w = rng.uniform(1, 5, ch.size)
    f_ms = f_asc[::-1] if ms_decreasing else f_asc
    npz = dict(u=uv[0][rw] * f_ms[ch] / 3e8, v=uv[1][rw] * f_ms[ch] / 3e8, vis=vis, weight=w, freq=f_ms[ch],
               ant1=ant1[rw], ant2=ant2[rw], time=time[rw])
    f_as = f_asc[::-1] if asdf_decreasing else f_asc
    idx = np.array([int(np.argmin(np.abs(f_as - f))) for f in f_ms])   # asdf channel index of each MS channel
    order = np.lexsort((rw, idx[ch]))                                  # and the asdf's own point order
    asdf = {k: np.array(v[order]) for k, v in npz.items()}
    asdf["chan"] = idx[ch][order]
    return asdf, npz


@pytest.mark.parametrize("ms_decreasing", [True, False])
@pytest.mark.parametrize("asdf_decreasing", [True, False])
def test_passes_whichever_way_either_side_orders_its_channels(ms_decreasing, asdf_decreasing):
    a, b = _ddid(ms_decreasing=ms_decreasing, asdf_decreasing=asdf_decreasing)
    r = compare_ddid(a, b)
    assert r["matching"] == "keys" and r["key_mismatch"] == 0 and r["pass"], r
    assert r["vis_max_abs"] == 0.0


@pytest.mark.validates("external_bridge.mpol_worker", roots=["self-consistency"], kind="control")
def test_control_a_flipped_channel_axis_is_a_failure_not_a_pass():
    """Two channels' frequencies swapped on one side (a wrong channel pairing) must fail: the u of those points differ by ~u dnu/nu."""
    a, b = _ddid()
    f = np.unique(a["freq"])
    lo, hi = a["freq"] == f[0], a["freq"] == f[-1]
    a["freq"] = a["freq"].copy()
    a["freq"][lo], a["freq"][hi] = f[-1], f[0]
    assert not compare_ddid(a, b)["pass"]


def test_matching_without_frequencies_raises_instead_of_assuming_a_direction():
    a, b = _ddid()
    del a["freq"]
    with pytest.raises(ValueError, match="freq"):
        compare_ddid(a, b)


@pytest.mark.validates("external_bridge.mpol_worker", roots=["self-consistency"], kind="control")
def test_control_a_short_baseline_error_hidden_by_the_longest_baseline_fails():
    """u is relative per point: 10 lambda off on the shortest baseline fails although it is only ~1e-7 of max|u|."""
    a, b = _ddid()
    j = np.argmin(np.abs(a["u"]))   # the shortest baseline: 10 lambda is >> 1e-6 of it, but 1e-7 of max|u|
    a["u"] = a["u"].copy(); a["u"][j] += 10.0
    assert not compare_ddid(a, b)["pass"]


@pytest.mark.validates("external_bridge.mpol_worker", roots=["self-consistency"], kind="control")
def test_control_a_different_data_column_fails():
    a, b = _ddid()
    a["vis"] = a["vis"] + 0.03                       # the size of the ddid-0 DATA vs CORRECTED_DATA difference
    assert not compare_ddid(a, b)["pass"]


def test_tolerance_edge_fails_above_1e5_jy_and_passes_below():
    a, b = _ddid()
    a["vis"] = a["vis"] + 2e-5
    assert not compare_ddid(a, b)["pass"]
    a, b = _ddid()
    a["vis"] = a["vis"] + 2e-6
    assert compare_ddid(a, b)["pass"]


@pytest.mark.validates("external_bridge.mpol_worker", roots=["self-consistency"], kind="control")
def test_control_a_differing_key_fails():
    a, b = _ddid()
    a["ant2"] = a["ant2"].copy()
    a["ant2"][0] += 1
    r = compare_ddid(a, b)
    assert r["key_mismatch"] > 0 and not r["pass"]


@pytest.mark.validates("external_bridge.mpol_worker", roots=["self-consistency"], kind="control")
def test_control_a_conjugated_reading_is_reported_not_passed():
    a, b = _ddid()
    a["vis"] = np.conj(a["vis"])
    r = compare_ddid(a, b)
    assert not r["pass"] and r["vis_conj_max_abs"] < 1e-12
