"""The 2004 data2 scorer (scripts/score_2004.py): the rebuilt truth's
geometry, the flux-ratio fit, and the metric's zero for a perfect entry.
NumPy only; the truth is used for scoring, never for imaging."""

import importlib.util
import os
import pathlib

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "score_2004", pathlib.Path(__file__).parents[1] / "scripts" / "score_2004.py"
)
score_2004 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(score_2004)


DATA2 = (
    pathlib.Path(os.environ.get("CONTEST_DATA", "~/data/imaging_contests")).expanduser() / "2004" / "2004-data2.fits"
)
needs_data2 = pytest.mark.skipif(not DATA2.exists(), reason="contest data not fetched")


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_truth_geometry_companion_east_and_unit_sum():
    star, companion = score_2004.truth_parts()
    # B has unit sum; A's *unspotted* flux does, so the spot's excess is left
    # out of the visibility normalisation and A.sum() > 1.
    assert companion.sum() == pytest.approx(1.0)
    unspotted, _ = score_2004.truth_parts(spot_contrast=1.0)
    assert unspotted.sum() == pytest.approx(1.0)
    assert star.sum() > 1.0
    x, y = score_2004.grid()
    assert np.sum(companion * x) == pytest.approx(10.0, abs=0.05)  # 10 mas east (PA 90°)
    assert np.sum(companion * y) == pytest.approx(0.0, abs=0.05)
    # A's spot (PA 150°: south-east) pulls A's light that way.
    assert np.sum(star * x) > 0 and np.sum(star * y) < 0


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_flux_ratio_recovered_from_noiseless_v2():
    star, companion = score_2004.truth_parts()
    rng = np.random.default_rng(3)
    u, v = rng.uniform(-6e7, 6e7, (2, 60))  # wavelengths: up to ~33 m at 550 nm
    v2 = (np.abs(score_2004.visibilities(star, u, v) + 0.2 * score_2004.visibilities(companion, u, v)) / 1.2) ** 2
    ratio, chi2 = score_2004.fit_ratio(star, companion, u, v, v2, np.full(v2.size, 1e-3))
    assert ratio == pytest.approx(0.2, rel=0.02)


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_a_perfect_entry_scores_zero():
    ref = score_2004.truth_image(0.1)
    box = np.ones_like(ref, bool)
    assert score_2004.sigma_over_peak(ref, ref, box)[0] == pytest.approx(0.0, abs=1e-12)
    assert score_2004.sigma_over_peak(np.roll(ref, 5, axis=1), ref, box)[0] > 0.1


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_alignment_keeps_emission_a_shift_brings_into_view():
    # The truth on a wider canvas, translated 38 pixels east: aligning it back
    # must recover the truth exactly, companion included (no cropping, no wrap).
    ref = score_2004.truth_image(0.1)
    pad = score_2004.MAX_SHIFT
    wide = score_2004.shifted(np.pad(ref, pad), (0, -38))
    moved, shift = score_2004.aligned(wide, ref)
    assert shift == (0, 38)
    np.testing.assert_allclose(moved, ref, atol=1e-15)
    # Zero-filled translation: flux leaving an edge is dropped, not wrapped.
    edge = np.zeros((5, 5))
    edge[2, 4] = 1.0
    assert score_2004.shifted(edge, (0, 1)).sum() == 0.0


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_visibility_normalisation_leaves_out_the_spot_excess():
    ratio = 0.11
    star, companion = score_2004.truth_parts(spot_contrast=5.0)
    vis = score_2004.truth_vis_image(ratio, spot_contrast=5.0)
    # V(0) = (1 + excess + B/A) / (1 + B/A) > 1; the scoring image has unit sum.
    assert vis.sum() == pytest.approx((star.sum() + ratio) / (1 + ratio))
    assert vis.sum() > 1.05
    assert score_2004.truth_image(ratio, spot_contrast=5.0).sum() == pytest.approx(1.0)
    v0 = score_2004.visibilities(vis, np.zeros(1), np.zeros(1))[0]
    assert v0.real == pytest.approx(vis.sum())


@needs_data2
@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_rebuilt_truth_fits_data2_v2():
    # The data's V² are normalised without the spot's excess (V²(0) ~ 1.14): the
    # rebuilt truth then fits at V² χ²/N ~ 1.3 (50 with a unit-sum image, once
    # blamed on calibration errors; 2.8 with the spot clipped to the disk), and
    # its closure phases fix the orientation.
    u, v, v2, err = score_2004.read_v2(DATA2)
    f = score_2004.FITTED
    img = score_2004.truth_vis_image(f["ratio"], f["u_ld"], f["spot_contrast"])
    assert score_2004.v2_chi2(img, u, v, v2, err) < 2.0
    unit_sum = score_2004.truth_image(f["ratio"], f["u_ld"], f["spot_contrast"])
    assert score_2004.v2_chi2(unit_sum, u, v, v2, err) > 5.0 * score_2004.v2_chi2(img, u, v, v2, err)
    cp, cp_inverted = score_2004.closure_chi2(img, score_2004.read_t3(DATA2))
    assert cp < 1.0 and cp_inverted > 100.0


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_spot_is_an_unclipped_added_disk():
    # The spot sits 2.5 mas out where A's limb is at 2.45 mas: about half of it
    # is off the star, and it adds (contrast - 1) x the limb brightness 1 - u.
    u_ld, contrast = 0.5, 3.7
    with_spot, _ = score_2004.truth_parts(u_ld, spot_contrast=contrast)
    without, _ = score_2004.truth_parts(u_ld, spot_contrast=1.0)
    extra = with_spot - without
    x, y = score_2004.grid()
    sx, sy = score_2004.offset(150.0, 2.5)
    on_spot = (x - sx) ** 2 + (y - sy) ** 2 < 0.5**2
    assert np.all(extra[~on_spot] == 0.0)
    np.testing.assert_allclose(extra[on_spot], (contrast - 1.0) * (1.0 - u_ld) * without.max())  # the centre is 1/norm
    outside = on_spot & (without == 0)
    assert outside.sum() > 0.3 * on_spot.sum()  # light beyond the limb


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_analytic_visibilities_match_a_fine_pixel_model():
    # The closed form agrees with a 0.02 mas pixel DFT of the same model.
    npix, pixel = 1201, 0.02
    star, companion = score_2004.truth_parts(npix=npix, pixel=pixel)
    ratio = 0.112
    rng = np.random.default_rng(1)
    u, v = rng.uniform(-6e7, 6e7, (2, 12))
    pix = (
        score_2004.visibilities(star, u, v, npix, pixel) + ratio * score_2004.visibilities(companion, u, v, npix, pixel)
    ) / (1 + ratio)
    np.testing.assert_allclose(score_2004.model_vis(u, v, ratio), pix, atol=3e-3)
    assert score_2004.model_vis(np.zeros(1), np.zeros(1), ratio)[0].real == pytest.approx(
        score_2004.truth_vis_image(ratio).sum(), rel=2e-3
    )


@needs_data2
@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_analytic_truth_fits_data2_near_one():
    u, v, v2, err = score_2004.read_v2(DATA2)
    f = score_2004.FITTED
    m = np.abs(score_2004.model_vis(u, v, f["ratio"], f["u_ld"], f["spot_contrast"])) ** 2
    assert np.mean(((m - v2) / err) ** 2) < 1.5
