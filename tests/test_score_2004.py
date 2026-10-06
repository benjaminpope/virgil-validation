"""The 2004 data2 scorer (scripts/score_2004.py): the rebuilt truth's
geometry, the flux-ratio fit, and the metric's zero for a perfect entry.
NumPy only; the truth is used for scoring, never for imaging."""

import importlib.util
import pathlib

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "score_2004", pathlib.Path(__file__).parents[1] / "scripts" / "score_2004.py"
)
score_2004 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(score_2004)


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_truth_geometry_companion_east_and_unit_sum():
    star, companion = score_2004.truth_parts()
    assert star.sum() == pytest.approx(1.0) and companion.sum() == pytest.approx(1.0)
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
