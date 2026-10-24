"""virgil.radio against real CASA (casatools and casatasks), on a small CASA-simulated measurement set (virgil issue #533).

Needs the CASA environment `.venv-alma` (``bash scripts/setup_external.sh alma``: casatools, casatasks, casadata, pinned) and
casatools in virgil's own interpreter (``pip install 'virgil-astro[radio]'``). Without them the module skips; CI installs both
(`.github/workflows/tests.yml`), so the records are in the evidence file wherever it is regenerated with them, for
example on a laptop:

    bash scripts/setup_external.sh alma
    .venv/bin/python -m pytest tests/test_radio_trust_casa.py -m "not slow"

* ``read_ms``: every sample of a flagged and an unflagged measurement set against the raw DATA, WEIGHT, FLAG, FLAG_ROW and UVW columns
  and the ddid channel frequencies read with ``casatools.table``, reduced here by the documented rule (Stokes I by weights, the
  conjugate, u = UVW nu / c). The other reader checks are in ``tests/test_alma_radio.py``.
* ``radio_clean.fit_beam`` and ``crosscheck.alma_clean.fit_beam`` (the lobe-only reference, psfcutoff 0.35) against the restoring
  beam CASA's ``tclean`` fits to its own PSF; ``radio_clean.clean`` against ``tclean`` (Hogbom, natural weights, 500 iterations,
  gain 0.1) on the same data: model, restored peak, PSF and beam. CASA's restored image is shifted by one pixel from its model
  on an odd image size (observed, recorded), so the restored images are compared by peak and flux, not pixel by pixel.
"""

import shutil

import numpy as np
import pytest

radio = pytest.importorskip("virgil.radio")
rc = pytest.importorskip("virgil.radio_clean")

import virgil_bridge.alma as vba  # noqa: E402,F401  (sets CASASITECONFIG before casatools loads)
from crosscheck import alma_clean as ac  # noqa: E402
from evidence.plugin import record  # noqa: E402
from external_bridge import _subprocess as sp  # noqa: E402

C = 299792458.0
NPIX, CELL = 225, 25.0  # odd for virgil's clean; the sources sit on whole pixels at this cell


def _virgil_casatools():
    try:
        import casatools
    except Exception:  # noqa: BLE001
        return None
    return casatools.version_string()


pytestmark = pytest.mark.skipif(not sp.available("alma", "casatasks") or _virgil_casatools() is None,
                                reason="casatools/casatasks missing in .venv-alma or casatools missing in virgil's interpreter")


def _worker(name, task):
    return sp.run("alma", name, task)


@pytest.fixture(scope="module")
def ms(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("trust_casa")
    path = str(tmp / "clean.ms")
    _worker("alma_worker.py", {"task": "simulate_ms", "path": path})
    return path, tmp


@pytest.fixture(scope="module")
def tclean(ms):
    """tclean on a copy (it adds scratch columns to its input)."""
    path, tmp = ms
    copy = str(tmp / "for_tclean.ms")
    shutil.copytree(path, copy)
    res = _worker("casa_clean_worker.py", {"task": "tclean", "ms": copy, "outdir": str(tmp / "tc"), "imsize": NPIX, "cell_mas": CELL,
                                           "niter": 500, "gain": 0.1})
    return {**res, **dict(np.load(res["npz"]))}


@pytest.fixture(scope="module")
def data(ms):
    return radio.ComplexVisData.read_ms(ms[0])


def _reference_samples(npz, ddid_ids):
    """The documented reduction of the raw columns: per row (cross-correlations only) and channel, kept if the row and both
    parallel hands are unflagged with positive weights; V = conj of the weight-averaged hands, weight = their sum,
    u, v = UVW nu / c."""
    rows = []
    cross = npz["ANTENNA1"] != npz["ANTENNA2"]
    for r in np.flatnonzero(cross & ~npz["FLAG_ROW"]):
        d = int(npz["DATA_DESC_ID"][r])
        for ch, nu in enumerate(npz["ddid_freqs"][d]):
            if npz["FLAG"][0, ch, r] or npz["FLAG"][1, ch, r]:
                continue
            w1, w2 = npz["WEIGHT"][0, r], npz["WEIGHT"][1, r]
            if not (w1 > 0 and w2 > 0):
                continue
            v = np.conj((w1 * npz["DATA"][0, ch, r] + w2 * npz["DATA"][1, ch, r]) / (w1 + w2))
            rows.append((npz["TIME"][r], npz["ANTENNA1"][r], npz["ANTENNA2"][r], d, ch, npz["UVW"][0, r] * nu / C,
                         npz["UVW"][1, r] * nu / C, v, w1 + w2, nu))
    arr = {k: np.array([x[i] for x in rows]) for i, k in enumerate(("time", "ant1", "ant2", "ddid", "chan", "u", "v", "vis", "weight", "freq"))}
    return arr


@pytest.mark.validates("virgil.radio.ComplexVisData.read_ms", roots=["casa"], tier="B")
@pytest.mark.parametrize("flagged", [False, True])
def test_read_ms_equals_the_documented_reduction_of_the_raw_casatools_columns(flagged, ms):
    """Every (row, channel) sample of `read_ms` (ddid, antennas, time, channel, u, v, V, weight, frequency) equals the
    reduction of the raw table columns written out above, with and without flags (FLAG on one hand, FLAG_ROW, a zero weight
    on one hand, one flagged channel); u, v to 1e-12 relative, V and weight to 1e-6 (the columns are float32), frequency exactly."""
    path, tmp = ms
    if flagged:
        path2 = str(tmp / "flagged.ms")
        shutil.copytree(path, path2)
        _worker("alma_worker.py", {"task": "flag_ms", "path": path2, "ddid": 0})
        path = path2
    res = _worker("casa_clean_worker.py", {"task": "columns", "ms": path, "out": str(tmp / f"cols{int(flagged)}.npz")})
    ref = _reference_samples(dict(np.load(res["npz"])), None)
    got = radio.ComplexVisData.read_ms(path, datacolumn="DATA")  # pinned: the simulated MS also has CORRECTED_DATA
    assert len(got) == len(ref["u"])
    ids = np.asarray(got.ddid_ids)[np.asarray(got.ddid)]
    order_g = np.lexsort((np.asarray(got.chan), ids, np.asarray(got.ant2), np.asarray(got.ant1), np.asarray(got.time)))
    order_r = np.lexsort((ref["chan"], ref["ddid"], ref["ant2"], ref["ant1"], ref["time"]))
    for k in ("time", "ant1", "ant2", "chan"):
        assert np.array_equal(np.asarray(getattr(got, k))[order_g], ref[k][order_r]), k
    assert np.array_equal(ids[order_g], ref["ddid"][order_r])
    assert np.array_equal(np.asarray(got.freq)[order_g], ref["freq"][order_r])
    worst = {}
    for k in ("u", "v"):
        a, b = np.asarray(getattr(got, k))[order_g], ref[k][order_r]
        worst[k] = np.abs(a - b).max() / np.abs(b).max()
    a, b = np.asarray(got.vis)[order_g], ref["vis"][order_r]
    worst["vis"] = np.abs(a - b).max() / np.abs(b).max()
    a, b = np.asarray(got.weight)[order_g], ref["weight"][order_r]
    worst["weight"] = np.abs(a / b - 1).max()
    tag = "flagged" if flagged else "unflagged"
    for k, x in worst.items():
        record(f"max_rel_{k}_{tag}", x)
    record(f"n_samples_{tag}", len(ref["u"]))
    assert worst["u"] < 1e-12 and worst["v"] < 1e-12 and worst["vis"] < 1e-6 and worst["weight"] < 1e-6
    if flagged:
        full = _reference_samples(dict(np.load(_worker("casa_clean_worker.py", {"task": "columns", "ms": ms[0], "out": str(tmp / "cols_full.npz")})["npz"])), None)
        assert len(ref["u"]) < len(full["u"])  # the flags removed samples


def _close(a, b, rel):
    return abs(a - b) / abs(b) <= rel


@pytest.mark.validates("virgil.radio_clean.clean", "virgil.radio.ComplexVisData.dirty_beam", roots=["casa"], tier="B")
def test_fit_beam_and_the_lobe_only_reference_agree_with_the_beam_tclean_fits(tclean, data):
    """CASA's restoring beam (the Gaussian `tclean` fits to its PSF above psfcutoff 0.35) against virgil's `fit_beam` and the
    repository's lobe-only reference `crosscheck.alma_clean.fit_beam` (psfcutoff 0.35), each applied to CASA's own PSF and,
    for virgil, to its own dirty beam of the same data: axes within 3 %, position angle within 1 degree (the grid is 5 pixels
    per beam; observed differences are recorded). This settles the deferred item: the lobe-only reference is the CASA fit
    to the stated tolerance, not exactly."""
    bmaj, bmin, pa = tclean["beam_mas_deg"]
    pa %= 180
    psf_virgil = np.asarray(data.dirty_beam(NPIX, CELL))
    fits = {
        "virgil_on_casa_psf": rc.fit_beam(tclean["psf"], CELL),
        "virgil_on_own_psf": rc.fit_beam(psf_virgil, CELL),
        "lobe_reference_on_casa_psf": tuple(float(x) for x in ac.fit_beam(tclean["psf"], CELL)[:3]),
    }
    for name, (a, b, p) in fits.items():
        dpa = abs((p - pa + 90) % 180 - 90)
        record(f"{name}_major_rel_diff", a / bmaj - 1)
        record(f"{name}_minor_rel_diff", b / bmin - 1)
        record(f"{name}_pa_diff_deg", dpa)
        assert _close(a, bmaj, 0.03) and _close(b, bmin, 0.03) and dpa < 1.0, (name, (a, b, p), (bmaj, bmin, pa))
    # virgil and the independent lobe-only reference are the same estimator on the same PSF: tight, not the CASA tolerance
    va, vb, vp = fits["virgil_on_casa_psf"]
    ra, rb, rp = fits["lobe_reference_on_casa_psf"]
    record("virgil_vs_reference_major_rel_diff", va / ra - 1)
    record("virgil_vs_reference_minor_rel_diff", vb / rb - 1)
    assert _close(va, ra, 1e-3) and _close(vb, rb, 1e-3) and abs((vp - rp + 90) % 180 - 90) < 0.05
    record("casa_bmaj_mas", bmaj)
    record("casa_bmin_mas", bmin)
    record("max_abs_psf_difference", np.abs(psf_virgil - tclean["psf"]).max())
    assert np.abs(psf_virgil - tclean["psf"]).max() < 5e-3  # CASA grids; virgil sums exactly


@pytest.mark.validates("virgil.radio_clean.clean", roots=["casa"], tier="B", kind="control")
def test_control_the_bounding_box_fit_of_the_lobe_misses_the_tclean_beam(tclean):
    """The beam tolerance separates the lobe-only fit from the earlier bounding-box fit (ledger C3): a Gaussian fitted to every
    pixel of the lobe's bounding box (including those below the cutoff) misses CASA's minor axis by more than 3 %, while the
    lobe-pixel fit above is within 0.1 %."""
    from scipy.ndimage import label
    from scipy.optimize import least_squares

    psf = tclean["psf"]
    peak = np.unravel_index(np.argmax(psf), psf.shape)
    lab, _ = label(psf >= 0.35)
    rows, cols = np.where(lab == lab[peak])
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    sub = psf[r0:r1, c0:c1]
    best = None
    for pa0 in (0.0, 45.0, 90.0, 135.0):
        fit = least_squares(lambda p: (ac.gaussian_image(sub.shape, (p[0], p[1]), p[2], p[3], p[4]) - sub).ravel(),
                            [peak[0] - r0, peak[1] - c0, 8.0, 7.0, pa0])
        if best is None or fit.cost < best.cost:
            best = fit
    a, b = sorted(abs(best.x[2:4]) * CELL, reverse=True)
    bmaj, bmin, _ = tclean["beam_mas_deg"]
    miss = max(abs(a / bmaj - 1), abs(b / bmin - 1))
    record("control_miss", miss)
    assert miss > 0.03


@pytest.mark.validates("virgil.radio_clean.clean", roots=["casa"], tier="B")
def test_clean_reproduces_the_tclean_model_peak_flux_and_residual(tclean, data):
    """Hogbom CLEAN, natural weights, 500 iterations, gain 0.1, no mask, on the CASA-simulated set (two point sources of 10 and
    5 mJy on whole pixels): virgil's model equals tclean's to 1 % of the brighter source pixel by pixel, the total model flux
    to 1e-3, the restored peak to 1e-3 (both 10 mJy/beam), and both residuals are below 1e-6 Jy/beam. The restored images are
    not compared pixelwise: CASA's lies one pixel from its own model on an odd grid (recorded)."""
    res = rc.clean(data, NPIX, CELL, niter=500, gain=0.1)
    model_diff = np.abs(res.model - tclean["model"]).max() / tclean["model"].max()
    flux = res.model.sum() / tclean["model"].sum() - 1
    peak = res.image.max() / tclean["image"].max() - 1
    shift = np.array(np.unravel_index(np.argmax(tclean["image"]), tclean["image"].shape)) - np.array(
        np.unravel_index(np.argmax(tclean["model"]), tclean["model"].shape))
    record("max_model_difference_over_peak", model_diff)
    record("total_flux_rel_diff", flux)
    record("restored_peak_rel_diff", peak)
    record("tclean_image_minus_model_peak_shift_rows", float(shift[0]))
    record("tclean_image_minus_model_peak_shift_cols", float(shift[1]))
    record("max_residual_virgil", np.abs(res.residual).max())
    record("max_residual_tclean", np.abs(tclean["residual"]).max())
    assert model_diff < 0.01 and abs(flux) < 1e-3 and abs(peak) < 1e-3
    assert np.abs(res.residual).max() < 1e-6 and np.abs(tclean["residual"]).max() < 1e-6
    assert res.model.max() == pytest.approx(0.010, rel=2e-3)  # the injected brighter source


@pytest.mark.validates("virgil.radio_clean.clean", roots=["casa"], tier="B", kind="control")
def test_control_a_flipped_model_misses_the_tclean_model(tclean, data):
    """The model comparison pins orientation: the same CLEAN model mirrored East-West or North-South differs from tclean's by
    at least the brighter source's peak, so the 1 % test above could not pass with a flipped axis."""
    res = rc.clean(data, NPIX, CELL, niter=500, gain=0.1)
    for name, wrong in (("flip_east", res.model[:, ::-1]), ("flip_north", res.model[::-1])):
        miss = np.abs(wrong - tclean["model"]).max() / tclean["model"].max()
        record(f"control_miss_{name}", miss)
        assert miss > 0.5
