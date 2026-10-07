"""virgil.metrics against the published figures of merit, written here in
NumPy (crosscheck.image_metrics), not from virgil's code.

* ncc: Pearson's r of the pixels; 1 for a positive affine copy.
* l1_score: 1 - min_a Σ|a e - r| / Σ r, minimized by brute force over every
  breakpoint of the piecewise-linear objective.
* lawson_sigma_over_peak: Lawson et al. (2004), Eq. 2.
* rms_convolved: the rms of the unit-sum difference, bare and after both
  images are convolved with a pixel-sampled Gaussian beam.
* resample: exact separable overlap rebinning (old pixels uniform).
* align: a blob moved by whole pixels comes back, with the documented sign
  (dra positive East, ddec positive North).
* score: the same numbers through the one-call interface, and the 180°
  ambiguity of V²-only data.
"""

import numpy as np
import pytest

from crosscheck import image_metrics as ours
from evidence.plugin import record

metrics = pytest.importorskip("virgil.metrics")
from virgil.imaging import Beam  # noqa: E402

pytestmark = pytest.mark.x64
SCALE = 0.7  # mas per pixel


def blob(n=32, x0=0.0, y0=0.0, sx=2.5, sy=1.6, extra=True):
    """A lopsided test scene: an elliptical Gaussian plus a fainter offset
    one, so that nothing is symmetric under a 180° turn."""
    y, x = np.mgrid[:n, :n] - (n - 1) / 2
    img = np.exp(-0.5 * (((x - x0) / sx) ** 2 + ((y - y0) / sy) ** 2))
    if extra:
        img += 0.4 * np.exp(-0.5 * (((x - x0 - 5) / 1.5) ** 2 + ((y - y0 + 3) / 1.5) ** 2))
    return img


@pytest.fixture(scope="module")
def pair():
    rng = np.random.default_rng(11)
    truth = blob()
    image = np.clip(truth + 0.05 * rng.normal(size=truth.shape), 0, None) * 3.7
    return image, truth


@pytest.mark.validates("virgil.metrics.ncc", roots=["mathematics"])
def test_ncc_is_pearson_r(pair):
    image, truth = pair
    got, want = float(metrics.ncc(image, truth)), ours.ncc(image, truth)
    record("abs_dncc", abs(got - want))
    assert got == pytest.approx(want, abs=1e-12)
    assert float(metrics.ncc(2.0 * truth + 0.3, truth)) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.validates("virgil.metrics.l1_score", roots=["mathematics"])
def test_l1_score_is_the_minimum_over_flux_scale(pair):
    image, truth = pair
    negative = image - 0.02  # some pixels negative: documented to be set to zero
    for e in (image, negative):
        got, want = float(metrics.l1_score(e, truth)), ours.l1_score(e, truth)
        record("abs_dl1", abs(got - want))
        assert got == pytest.approx(want, abs=1e-12)
    assert float(metrics.l1_score(5.0 * truth, truth)) == pytest.approx(1.0, abs=1e-12)
    assert float(metrics.l1_score(np.zeros_like(truth), truth)) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.validates("virgil.metrics.lawson_sigma_over_peak", roots=["literature", "mathematics"])
def test_lawson_sigma_over_peak_is_eq_2(pair):
    image, truth = pair
    got, want = float(metrics.lawson_sigma_over_peak(image, truth)), ours.lawson_sigma_over_peak(image, truth)
    record("rel_dlawson", abs(got / want - 1))
    assert got == pytest.approx(want, rel=1e-12)
    assert float(metrics.lawson_sigma_over_peak(3 * truth, truth)) == pytest.approx(0.0, abs=1e-14)


@pytest.mark.validates("virgil.metrics.rms_convolved", roots=["mathematics"])
def test_rms_without_a_beam(pair):
    image, truth = pair
    got, want = float(metrics.rms_convolved(image, truth)), ours.rms(image, truth)
    assert got == pytest.approx(want, rel=1e-12)
    got_rel = float(metrics.rms_convolved(image, truth, relative=True))
    record("rel_drms", abs(got_rel / ours.rms(image, truth, relative=True) - 1))
    assert got_rel == pytest.approx(ours.rms(image, truth, relative=True), rel=1e-12)


@pytest.mark.validates("virgil.metrics.rms_convolved", roots=["mathematics"])
def test_rms_after_a_gaussian_beam(pair):
    """Both images convolved with the beam's Gaussian (FWHM 3 and 2 pixels,
    major axis at PA 30°), sampled on the pixels and convolved linearly
    (zeros outside the field). virgil agrees to rounding, so it convolves
    the same way."""
    image, truth = pair
    beam = Beam(3.0 * SCALE, 2.0 * SCALE, 30.0)
    kernel = ours.gaussian_kernel(31, SCALE, beam.major_mas, beam.minor_mas, beam.pa_deg)
    want = ours.rms(image, truth, kernel)
    got = float(metrics.rms_convolved(image, truth, pixel_scale_mas=SCALE, beam=beam))
    record("rel_drms_beam", abs(got / want - 1))
    assert got == pytest.approx(want, rel=1e-10)
    # The orientation matters: the beam turned by 90° gives a different number.
    turned = ours.rms(image, truth, ours.gaussian_kernel(31, SCALE, 3 * SCALE, 2 * SCALE, 120.0))
    assert abs(got - want) < 0.2 * abs(turned - want)


@pytest.mark.parametrize("new_scale,npix", [(1.3, 20), (0.45, 50), (0.7, 24)], ids=["coarser", "finer", "crop"])
@pytest.mark.validates("virgil.metrics.resample", roots=["mathematics"])
def test_resample_is_exact_overlap_rebinning(new_scale, npix):
    image = blob(30)
    got = np.asarray(metrics.resample(image, SCALE, npix, new_scale))
    want = ours.resample(image, SCALE, npix, new_scale)
    err = record("max_abs_dflux", np.max(np.abs(got - want)) / image.sum())
    assert err < 1e-12
    if new_scale * npix >= SCALE * 30:  # the new field covers the old: flux conserved
        assert got.sum() == pytest.approx(image.sum(), rel=1e-12)


@pytest.mark.validates("virgil.metrics.align", roots=["mathematics"])
def test_align_undoes_a_whole_pixel_shift():
    truth = blob(40)
    moved = ours.shift(truth, 1, 2)  # content 1 pixel South and 2 West
    shifted, (dra, ddec) = metrics.align(moved, truth, 5 * SCALE, pixel_scale_mas=SCALE)
    record("abs_dshift_mas", max(abs(dra - 2 * SCALE), abs(ddec - 1 * SCALE)))
    # moving it back means moving it East and North: both positive
    assert dra == pytest.approx(2 * SCALE, abs=1e-6) and ddec == pytest.approx(1 * SCALE, abs=1e-6)
    np.testing.assert_allclose(np.asarray(shifted), ours.shift(moved, -1, -2), atol=1e-9)


@pytest.mark.validates("virgil.metrics.score", roots=["mathematics"])
def test_score_reports_the_same_numbers(pair):
    image, truth = pair
    s = metrics.score(image, truth, pixel_scale_mas=SCALE)
    assert s["ncc"] == pytest.approx(ours.ncc(image, truth), abs=1e-12)
    assert s["l1"] == pytest.approx(ours.l1_score(image, truth), abs=1e-12)
    assert s["lawson"] == pytest.approx(ours.lawson_sigma_over_peak(image, truth), rel=1e-12)
    assert s["rms"] == pytest.approx(ours.rms(image, truth), rel=1e-12)
    assert s["rotated"] is False or not s["rotated"]


@pytest.mark.validates("virgil.metrics.score", roots=["mathematics"])
def test_score_v2_only_keeps_the_better_of_a_180_degree_turn():
    truth = blob(32, x0=3.0, y0=-2.0)
    turned = truth[::-1, ::-1]
    assert ours.ncc(turned, truth) < 0.5  # the scene is not point-symmetric
    s = metrics.score(turned, truth, pixel_scale_mas=SCALE, v2_only=True)
    record("ncc_v2_only", s["ncc"])
    assert bool(s["rotated"]) and s["ncc"] == pytest.approx(1.0, abs=1e-12)
    plain = metrics.score(turned, truth, pixel_scale_mas=SCALE)
    assert not bool(plain["rotated"]) and plain["ncc"] == pytest.approx(ours.ncc(turned, truth), abs=1e-12)


# ------------------------------------------------- our own references first


@pytest.mark.validates("crosscheck.image_metrics", roots=["mathematics"], kind="reference")
def test_our_metric_references(pair):
    from scipy import optimize, signal

    image, truth = pair
    # l1: the brute-force breakpoint minimum is SciPy's bounded minimum
    cost = lambda a: np.abs(a * image - truth).sum()  # noqa: E731
    best = optimize.minimize_scalar(cost, bounds=(0, 1), method="bounded", options={"xatol": 1e-12}).fun
    assert ours.l1_score(image, truth) == pytest.approx(1 - best / truth.sum(), abs=1e-9)
    # convolution: SciPy's 'same' convolution, zeros outside
    k = ours.gaussian_kernel(9, SCALE, 2.0, 1.4, 30.0)
    np.testing.assert_allclose(ours.convolve(truth, k), signal.convolve2d(truth, k, mode="same"), atol=1e-14)
    # resampling: the identity on the same grid, and flux conserved going coarser
    np.testing.assert_allclose(ours.resample(truth, SCALE, 32, SCALE), truth, atol=1e-14)
    assert ours.resample(truth, SCALE, 16, 2 * SCALE).sum() == pytest.approx(truth.sum(), rel=1e-14)
    # shift: integer moves and back
    np.testing.assert_array_equal(ours.shift(ours.shift(truth, 2, -3), -2, 3)[3:-3, 3:-3], truth[3:-3, 3:-3])
