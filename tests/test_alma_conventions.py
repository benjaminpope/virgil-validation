"""Measurement-set conventions, pinned on a synthetic MS made by CASA.

``casatools.simulator`` builds a small MS from scratch (8 antennas, two
spectral windows, dual linear feeds) and CASA's own predict puts two point
sources into it: 10 mJy at (+300, +100) mas and 5 mJy at (-200, -400) mas
(East, North) from the phase centre. Our reader (written from CASA's MS
definition, src/external_bridge/alma_worker.py) must then give visibilities
whose dirty image peaks at those positions under

    V(u, v) = sum I exp(-2 pi i (u x + v y)),   x East, y North,

which pins the conjugation and the sign of u (docs/alma_conventions.md).
The environment is .venv-alma (scripts/setup_external.sh alma); without it
these tests are skipped.
"""

import numpy as np
import pytest

from crosscheck import alma
from evidence.plugin import record
from external_bridge import _subprocess as sp

pytestmark = [
    pytest.mark.external,
    pytest.mark.skipif(not sp.available("alma", "casatools"), reason="CASA not installed (scripts/setup_external.sh alma)"),
]

MAS = np.pi / 180 / 3600 / 1000
SOURCES = [(0.010, 300.0, 100.0), (0.005, -200.0, -400.0)]  # Jy, East mas, North mas
NPIX, PIXEL = 128, 25.0
VALIDATES = dict(roots=["casa"], tier="B")


def worker(task):
    return sp.run("alma", "alma_worker.py", task)


def load(files):
    return [dict(np.load(f["file"])) for f in files]


def model(z, sources=SOURCES):
    """The two points' visibilities in the documented convention."""
    return sum(f * np.exp(-2j * np.pi * (z["u"] * e + z["v"] * n) * MAS) for f, e, n in sources)


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("alma_clean")
    ms = str(tmp / "clean.ms")
    worker({"task": "simulate_ms", "path": ms})
    return ms, tmp


@pytest.fixture(scope="module")
def clean_read(clean):
    ms, tmp = clean
    return load(worker({"task": "read", "path": ms, "out": str(tmp / "ours")})["files"])


def peak_error(z, conjugate, source):
    vis = z["re"] + 1j * z["im"]
    vis = np.conj(vis) if conjugate else vis
    img = alma.dft_dirty_image(z["u"], z["v"], vis, z["weight"], NPIX, PIXEL)
    _, east, north = source
    c = NPIX // 2 - int(round(east / PIXEL))
    r = NPIX // 2 - int(round(north / PIXEL))
    box = (r - 6, r + 7, c - 6, c + 7)  # about 150 mas: inside one beam
    e, n = alma.peak_offset_mas(img, PIXEL, box)
    return max(abs(e - east), abs(n - north))


@pytest.mark.validates("crosscheck.alma", "external_bridge.alma_worker", **VALIDATES)
def test_dirty_image_peaks_at_the_casa_predicted_sources(clean_read):
    """The bright source is at (+300, +100) mas and the faint one at
    (-200, -400) mas, to one pixel, in both spectral windows."""
    assert len(clean_read) == 2
    worst = 0.0
    for z in clean_read:
        for src in SOURCES:
            worst = max(worst, peak_error(z, conjugate=False, source=src))
    record("worst_peak_error_mas", worst)
    assert worst <= PIXEL
    # the brightest pixel of the whole map is the bright source
    z = clean_read[0]
    img = alma.dft_dirty_image(z["u"], z["v"], z["re"] + 1j * z["im"], z["weight"], NPIX, PIXEL)
    e, n = alma.peak_offset_mas(img, PIXEL)
    assert max(abs(e - 300.0), abs(n - 100.0)) <= PIXEL


@pytest.mark.validates("crosscheck.alma", "external_bridge.alma_worker", **VALIDATES)
def test_visibilities_are_the_conjugate_of_the_data_column_at_plus_uvw(clean_read):
    """CASA's DATA is the complex conjugate of V = sum F exp(-2 pi i (u x + v y))
    at u = +UVW[0] nu / c (equivalently V at (-u, -v)): the `read` default
    (conjugate=True) matches the point-source model to float32 precision."""
    worst = max(np.abs(z["re"] + 1j * z["im"] - model(z)).max() for z in clean_read)
    record("max_abs_vis_error_jy", worst)
    assert worst < 1e-6  # float32 rounding of 15 mJy is 1e-9


@pytest.mark.validates("crosscheck.alma", "external_bridge.alma_worker", kind="control", **VALIDATES)
def test_control_unconjugated_data_puts_the_sources_at_minus_their_position(clean, clean_read, tmp_path):
    """The wrong mapping, V = DATA at +u (conjugate=False), mirrors the
    sky through the phase centre: the bright peak lands at (-300, -100) and the
    visibilities miss the model by about twice their amplitude."""
    ms, _ = clean
    wrong = load(worker({"task": "read", "path": ms, "out": str(tmp_path), "conjugate": False})["files"])
    miss = max(np.abs(z["re"] + 1j * z["im"] - model(z)).max() for z in wrong)
    record("max_abs_vis_miss_jy", miss)
    assert miss > 0.02  # 2 x 15 mJy, against 1e-6 for the correct mapping
    z = wrong[0]
    img = alma.dft_dirty_image(z["u"], z["v"], z["re"] + 1j * z["im"], z["weight"], NPIX, PIXEL)
    e, n = alma.peak_offset_mas(img, PIXEL)
    record("peak_east_mas", e)
    record("peak_north_mas", n)
    assert max(abs(e + 300.0), abs(n + 100.0)) <= PIXEL  # mirrored, so far from (+300, +100)


@pytest.mark.validates("crosscheck.alma", "external_bridge.alma_worker", **VALIDATES)
def test_uv_follow_uvw_and_channel_frequency(clean, clean_read):
    """u = UVW[0] nu / c: the longest baseline of the 8-antenna layout
    (about 1.6 km) at 230 GHz is 1.0-1.3 Mlambda, and u scales with nu."""
    z0, z1 = clean_read
    r = np.hypot(z0["u"], z0["v"]).max()
    assert 1.0e6 < r < 1.4e6
    # same row and channel in both windows: u1/u0 = nu1/nu0 (same antennas and times)
    n = min(len(z0["u"]), len(z1["u"]))
    ok = np.abs(z0["u"][:n]) > 1e3
    ratio = z1["u"][:n][ok] / z0["u"][:n][ok]
    want = z1["freq"][:n][ok] / z0["freq"][:n][ok]
    record("max_ratio_error", np.abs(ratio / want - 1).max())
    assert np.allclose(ratio, want, rtol=1e-6)
    assert z0["chan_width_hz"] == pytest.approx(125e6) and z0["t_int_s"] == pytest.approx(6.0)


@pytest.mark.validates("crosscheck.alma", "external_bridge.alma_worker", roots=["golden:visread"], tier="B")
def test_our_reader_agrees_with_visread(clean, clean_read):
    """visread (the reader MPoL uses; it also conjugates DATA) gives the same
    samples, to float32 precision."""
    ms, tmp = clean
    res = worker({"task": "visread_read", "path": ms, "out": str(tmp / "visread")})
    record("n_files", len(res["files"]))
    ref = load(res["files"])
    for ours, theirs in zip(clean_read, ref):
        assert len(ours["u"]) == len(theirs["u"])
        for k in ("u", "v", "freq", "time"):
            assert np.allclose(ours[k], theirs[k], rtol=0, atol=1e-6 * np.abs(ours[k]).max())
        for k in ("re", "im", "weight"):
            assert np.allclose(ours[k], theirs[k], rtol=1e-6, atol=1e-9)


@pytest.fixture(scope="module")
def noisy_read(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("alma_noisy")
    ms = str(tmp / "noisy.ms")
    worker({"task": "simulate_ms", "path": ms, "noise_jy": 1e-3})
    return load(worker({"task": "read", "path": ms, "out": str(tmp / "ours")})["files"])


@pytest.mark.validates("crosscheck.alma", roots=["statistics"], tier="B")
def test_weights_are_inverse_variance_per_component_and_polarizations_add(noisy_read):
    """With 1 mJy simple noise on each of XX and YY, the polarization-averaged
    visibility has sigma = 1 mJy / sqrt 2 on each of Re and Im, weight w1 + w2 =
    2e6 Jy^-2; the successive-difference estimate of s is 1 within 3
    standard errors, and the scatter about the known model agrees."""
    for z in noisy_read:
        got = alma.successive_difference_sigma(z)
        record("s2", got["s2"])
        record("s2_err", got["s2_err"])
        assert abs(got["s2"] - 1.0) < 3 * got["s2_err"]
        assert np.allclose(z["weight"], 2e6, rtol=1e-6)
        res = z["re"] + 1j * z["im"] - model(z)
        for part in (res.real, res.imag):
            assert part.std() == pytest.approx(1e-3 / np.sqrt(2), rel=0.02)


@pytest.mark.validates("crosscheck.alma", "external_bridge.alma_worker", **VALIDATES)
def test_stride_keeps_every_strideth_row_of_the_cross_correlations(clean, clean_read, tmp_path):
    """16800 cross-correlation rows per window, 4 channels: stride 7 keeps
    rows 0, 7, 14, ... (2400 rows)."""
    ms, _ = clean
    sub = load(worker({"task": "read", "path": ms, "out": str(tmp_path), "ddids": [0], "stride": 7})["files"])[0]
    full = clean_read[0]
    assert len(full["u"]) == 16800 * 4
    assert len(sub["u"]) == 2400 * 4
    keys = lambda z: {(a, b, t) for a, b, t in zip(z["ant1"], z["ant2"], z["time"])}  # noqa: E731
    assert keys(sub) <= keys(full) and len(keys(sub)) == 2400
    assert np.all(sub["ant1"] != sub["ant2"])


@pytest.mark.validates("crosscheck.alma", "external_bridge.alma_worker", **VALIDATES)
def test_samples_are_dropped_for_flags_flag_row_and_zero_weights(tmp_path):
    """A sample is kept only if FLAG_ROW is false, neither parallel-hand
    correlation is flagged and both weights are positive."""
    ms = str(tmp_path / "f.ms")
    worker({"task": "simulate_ms", "path": ms})
    kept = lambda: len(load(worker({"task": "read", "path": ms, "out": str(tmp_path / "o"), "ddids": [0]})["files"])[0]["u"])  # noqa: E731
    before = kept()
    worker({"task": "flag_ms", "path": ms, "ddid": 0})
    # 100 rows x 4 channels (XX flagged) + 400 (FLAG_ROW) + 400 (YY weight 0) + 100 (channel 1)
    # the last block overlaps nothing: rows 300-399 only
    assert before - kept() == 100 * 4 + 100 * 4 + 100 * 4 + 100


@pytest.mark.validates("crosscheck.alma", "external_bridge.alma_worker", **VALIDATES)
def test_a_selection_with_more_than_one_field_raises_unless_a_field_is_given(tmp_path):
    """The reader is for one pointing: a two-field MS needs `field`, which then
    selects its rows only."""
    ms = str(tmp_path / "two.ms")
    worker({"task": "simulate_ms", "path": ms, "two_fields": True})
    with pytest.raises(RuntimeError, match="field"):
        worker({"task": "read", "path": ms, "out": str(tmp_path / "a"), "ddids": [0]})
    n = [sum(f["n"] for f in worker({"task": "read", "path": ms, "out": str(tmp_path / f"f{k}"), "ddids": [0], "field": k})["files"])
         for k in (0, 1)]
    record("rows_field0", n[0])
    assert n[0] == n[1] == 28 * 5 * 60 * 4  # 28 baselines x 300 integrations (30 min each) x 4 channels
