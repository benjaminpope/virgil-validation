"""The contest imaging benchmark: metrics, phantoms and simulated data.

The metrics (crosscheck.image_metrics) must give their perfect values for a
perfect entry, undo a whole-pixel shift, and under the V²-only rule accept an
inverted entry. The simulator (scripts/contest_bench.py, NumPy only) must
agree with virgil's own model of the same image: noiseless data are fitted
exactly, noisy data at χ²/N ≈ 1, and the inverted image is clearly worse
(the closure phases fix the orientation). That ties the benchmark's
conventions (Fourier sign, row 0 North, column 0 East) to virgil's.
"""

import importlib.util
import pathlib

import numpy as np
import pytest

from crosscheck import image_metrics, phantoms

ROOT = pathlib.Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("contest_bench", ROOT / "scripts" / "contest_bench.py")
contest_bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contest_bench)
DATA = pathlib.Path("~/data/imaging_contests").expanduser()


def _phantom(family="spotted_star", seed=1, beam=2.0):
    params = phantoms.sample(family, np.random.default_rng(seed), beam)
    npix, pixel = contest_bench.truth_grid(params, beam)
    return params, phantoms.render(params, npix, pixel), pixel


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
@pytest.mark.parametrize("family", phantoms.FAMILIES)
def test_phantoms_are_unit_sum_and_inside_their_extent(family):
    params, image, pixel = _phantom(family)
    assert image.sum() == pytest.approx(1.0)
    assert np.all(image >= 0)
    east, north = image_metrics.offsets(image.shape[0], pixel)
    radius = np.hypot(east, north)
    assert image[radius > phantoms.extent(params)].sum() < 0.02


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_metrics_perfect_shifted_and_inverted_entries():
    _, ref, pixel = _phantom()
    fov = ref.shape[0] * pixel
    best = image_metrics.score(ref, fov, ref, pixel, beam_mas=2.0)
    assert best["lawson"] == pytest.approx(0.0, abs=1e-9)
    assert best["rms_e6"] == pytest.approx(0.0, abs=1e-6)
    assert best["l1"] == pytest.approx(1.0, abs=1e-9)
    assert best["ncc"] == pytest.approx(1.0, abs=1e-12)
    shifted = image_metrics.score(np.roll(ref, (2, -3), axis=(0, 1)), fov, ref, pixel, beam_mas=2.0)
    assert shifted["shift"] == (-2, 3) and shifted["ncc"] == pytest.approx(1.0, abs=1e-9)
    inverted = ref[::-1, ::-1]
    with_cp = image_metrics.score(inverted, fov, ref, pixel, beam_mas=2.0)
    v2_only = image_metrics.score(inverted, fov, ref, pixel, beam_mas=2.0, v2_only=True)
    assert v2_only["orientation"] == "inverted" and v2_only["ncc"] == pytest.approx(1.0, abs=1e-9)
    assert with_cp["ncc"] < v2_only["ncc"]
    # A worse entry scores worse on every metric.
    blurred = image_metrics.score(image_metrics.gaussian_filter(ref, 3.0), fov, ref, pixel, beam_mas=2.0)
    assert blurred["lawson"] > 0.05 and blurred["l1"] < 0.95 and blurred["ncc"] < 0.99


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_l1_scale_is_the_optimum():
    rng = np.random.default_rng(2)
    ref = rng.random((9, 9))
    entry = ref * (1 + 0.1 * rng.standard_normal(ref.shape))
    best = image_metrics.l1_score(entry, ref)
    for a in (0.9, 0.97, 1.03, 1.1):
        assert best >= 1 - np.sum(np.abs(a * entry - ref)) / ref.sum() - 1e-12


@pytest.mark.validates("pipeline:contest-imaging", "virgil.models.Image", roots=["mathematics"], kind="reference")
@pytest.mark.skipif(not (DATA / "2004" / "2004-data2.fits").exists(), reason="contest data not fetched")
def test_simulated_data_agree_with_virgil(tmp_path):
    import virgil.models as vm
    from virgil.likelihood import whitened_residuals
    from virgil.oidata import OIData

    from crosscheck import sky

    params, image, pixel = _phantom(beam=1.9)
    template = DATA / "2004" / "2004-data2.fits"
    cloud = sky.pixel_image(image, pixel)
    model = vm.Image.from_brightness(image, pixel)
    inverted = vm.Image.from_brightness(image[::-1, ::-1], pixel)
    chi2 = {}
    for noise in (False, True):
        out = tmp_path / f"sim_{noise}.fits"
        contest_bench.simulate_file(template, out, cloud, np.random.default_rng(3), noise=noise)
        data = OIData(str(out))
        n = data.n_independent
        chi2[noise] = float(np.sum(np.asarray(whitened_residuals(model, data)) ** 2)) / n
        if noise:
            chi2["inverted"] = float(np.sum(np.asarray(whitened_residuals(inverted, data)) ** 2)) / n
    assert chi2[False] < 0.05
    # The simulator's correlated closure noise (shared baseline phases, see
    # contest_bench.closure_noise) is not exactly virgil's assumed covariance,
    # so χ²/N lands within ~30% of 1 (2004: 1.1-1.4; 2008: 0.8-0.9; 2024 PIONIER: 1.0).
    assert 0.7 < chi2[True] < 1.5
    assert chi2["inverted"] > 3 * chi2[True]


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
@pytest.mark.skipif(not (DATA / "2008" / "2008-Contest1_H.oifits").exists(), reason="contest data not fetched")
def test_closure_noise_has_the_quoted_variance():
    from astropy.io import fits

    with fits.open(DATA / "2008" / "2008-Contest1_H.oifits") as h:
        t3 = h["OI_T3"].data
        err = np.asarray(t3["T3PHIERR"], float)
        draws = np.array([contest_bench.closure_noise(t3, err, np.random.default_rng(s)) for s in range(400)])
    ratio = draws.std(axis=0) / err
    assert np.median(ratio) == pytest.approx(1.0, abs=0.05)
    assert np.all((ratio > 0.8) & (ratio < 1.2))
