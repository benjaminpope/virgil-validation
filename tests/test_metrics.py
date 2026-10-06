"""virgil.metrics (the imaging contests' image-recovery scores) against the
independent scorer crosscheck.image_metrics, direct evaluations of each
published formula, brute-force minimisers and exact resampling sums.

Images: synthetic scenes (a star with a companion and an arc, asymmetric so
that shifts and the 180° rotation matter) and the benchmark phantoms
(crosscheck.phantoms), all on small grids. Conventions learnt from virgil's
documentation: row 0 North, column 0 East; the rotation for V²-only data is
about the array centre; ``score`` aligns only if given ``max_shift_mas``.

Definition differences are findings F17-F19 (README, trust/ledger.yml):
strict xfails that flip when virgil changes.
"""

import importlib.util
import pathlib

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize_scalar

from crosscheck import image_metrics as cm, phantoms
from virgil_bridge import metrics as vb

ROOT = pathlib.Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("contest_bench", ROOT / "scripts" / "contest_bench.py")
contest_bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contest_bench)

PIXEL = 0.2
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def scene(n=64, pixel=PIXEL):
    """An asymmetric unit-sum scene, row 0 North, column 0 East: a bright
    star, a companion 3 mas East and 1.5 mas North, a faint arc."""
    c = (n - 1) / 2.0
    east = (c - np.arange(n))[None, :] * pixel
    north = (c - np.arange(n))[:, None] * pixel
    star = np.exp(-0.5 * ((east / 0.8) ** 2 + (north / 0.6) ** 2))
    comp = 0.3 * np.exp(-0.5 * (((east - 3.0) / 0.5) ** 2 + ((north - 1.5) / 0.5) ** 2))
    r = np.hypot(east + 1.0, north + 2.0)
    arc = 0.05 * np.exp(-0.5 * ((r - 2.0) / 0.4) ** 2) * (north < -1.0)
    img = star + comp + arc
    return img / img.sum()


def phantom(family, seed=1, beam=2.0):
    params = phantoms.sample(family, np.random.default_rng(seed), beam)
    npix, pixel = contest_bench.truth_grid(params, beam)
    return phantoms.render(params, npix, pixel), pixel


def entries(truth):
    """Entries of a range of quality: identity, flux scale, convolution,
    a perturbed copy (a clipped blur plus a lump)."""
    rng = np.random.default_rng(0)
    return {
        "identity": truth,
        "scaled": 7.5 * truth,
        "blurred": gaussian_filter(truth, 1.5),
        "perturbed": np.maximum(gaussian_filter(truth, 1.0) + 0.02 * truth.max() * rng.random(truth.shape) * (gaussian_filter(truth, 3.0) > 0.01 * truth.max()), 0),
    }


CASES = [("scene", None)] + [(f, None) for f in phantoms.FAMILIES]


def truth_for(name):
    return (scene(), PIXEL) if name == "scene" else phantom(name)


# ----------------------------------------------------------------- ncc


@pytest.mark.validates("virgil.metrics.ncc", roots=["mathematics"])
@pytest.mark.parametrize("name", [c[0] for c in CASES])
def test_ncc_is_pearson_and_matches_crosscheck(name, metric):
    truth, _ = truth_for(name)
    for label, entry in entries(truth).items():
        ours = np.corrcoef(entry.ravel(), truth.ravel())[0, 1]
        virgil = vb.ncc(entry, truth)
        assert virgil == pytest.approx(ours, abs=1e-12), label
        assert virgil == pytest.approx(cm.ncc(entry, truth), abs=1e-12), label
    assert vb.ncc(truth, truth) == pytest.approx(1.0, abs=1e-12)
    # Independent of flux scale and offset of the entry, as documented.
    assert vb.ncc(3.0 * truth + 0.01, truth) == pytest.approx(1.0, abs=1e-12)
    metric("ncc_blurred_difference", abs(vb.ncc(entries(truth)["blurred"], truth) - cm.ncc(entries(truth)["blurred"], truth)))


# ----------------------------------------------------------------- l1


def brute_l1(entry, truth):
    """1 - min over a >= 0 of sum|a e - r| / sum r, by a bounded scalar search
    (the objective is convex and piecewise linear in a)."""
    f = lambda a: np.sum(np.abs(a * entry - truth))
    top = 10.0 * truth.sum() / max(entry[entry > 0].sum(), 1e-300)
    res = minimize_scalar(f, bounds=(0.0, top), method="bounded", options={"xatol": 1e-13 * top})
    return 1.0 - min(res.fun, f(0.0)) / truth.sum()


@pytest.mark.validates("virgil.metrics.l1_score", roots=["mathematics"])
@pytest.mark.parametrize("name", [c[0] for c in CASES])
def test_l1_matches_crosscheck_and_bruteforce_minimiser(name, metric):
    truth, _ = truth_for(name)
    for label, entry in entries(truth).items():
        virgil = vb.l1_score(entry, truth)
        assert virgil == pytest.approx(cm.l1_score(entry, truth), abs=1e-12), label
        assert virgil == pytest.approx(brute_l1(entry, truth), abs=1e-6), label
    assert vb.l1_score(truth, truth) == pytest.approx(1.0, abs=1e-12)
    # Flux scale of either image does not matter.
    blurred = gaussian_filter(truth, 1.5)
    assert vb.l1_score(40 * blurred, 0.3 * truth) == pytest.approx(vb.l1_score(blurred, truth), abs=1e-12)
    metric("l1_blurred", vb.l1_score(blurred, truth))


def _with_negatives():
    truth = scene()
    entry = gaussian_filter(truth, 1.0) - 0.05 * truth.max() * np.random.default_rng(3).random(truth.shape)
    assert (entry < 0).sum() > 100
    return entry, truth


@pytest.mark.validates("virgil.metrics.l1_score", roots=["mathematics"])
def test_l1_clips_negative_pixels_as_documented():
    """Documented: negative pixels of the image are set to zero, so virgil
    equals the scorer (and brute force) evaluated on the clipped image."""
    entry, truth = _with_negatives()
    clipped = np.maximum(entry, 0.0)
    assert vb.l1_score(entry, truth) == pytest.approx(cm.l1_score(clipped, truth), abs=1e-12)
    assert vb.l1_score(entry, truth) == pytest.approx(brute_l1(clipped, truth), abs=1e-6)


@pytest.mark.validates("virgil.metrics.l1_score", roots=["mathematics"], kind="finding")
@pytest.mark.xfail(strict=True, reason="F18: virgil clips negative pixels; the L1 minimum of the unclipped image differs")
def test_l1_of_unclipped_image_with_negative_pixels():
    entry, truth = _with_negatives()
    assert vb.l1_score(entry, truth) == pytest.approx(brute_l1(entry, truth), abs=1e-6)


# ------------------------------------------------------------- lawson


def lawson_eq2(entry, truth, box=None):
    """Lawson et al. (2004) Eq. 2 as recorded in docs/contest_scoring_sources.md:
    sigma = [sum p_ref (p_i - p_ref)^2 / sum p_ref]^(1/2) over the box, both
    images unit sum in the box; the score is sigma / peak(reference)."""
    box = np.ones(truth.shape, bool) if box is None else box
    e = np.where(box, entry, 0.0)
    r = np.where(box, truth, 0.0)
    e, r = e / e.sum(), r / r.sum()
    return np.sqrt(np.sum(r * (e - r) ** 2) / np.sum(r)) / r.max()


@pytest.mark.validates("virgil.metrics.lawson_sigma_over_peak", roots=["literature"])
@pytest.mark.parametrize("name", [c[0] for c in CASES])
def test_lawson_is_eq2_and_matches_crosscheck(name, metric):
    truth, _ = truth_for(name)
    for label, entry in entries(truth).items():
        virgil = vb.lawson(entry, truth)
        assert virgil == pytest.approx(lawson_eq2(entry, truth), rel=1e-12, abs=1e-14), label
        assert virgil == pytest.approx(cm.lawson_sigma_over_peak(entry, truth)[0], rel=1e-12, abs=1e-14), label
    assert vb.lawson(truth, truth) == 0.0
    assert vb.lawson(5 * truth, truth) == pytest.approx(0.0, abs=1e-14)  # flux scale is normalised out
    metric("lawson_blurred", vb.lawson(entries(truth)["blurred"], truth))


def _halo_entry():
    """Truth with compact support; an entry carrying 30% of its flux outside
    the box that holds the truth's emission."""
    truth = np.zeros((64, 64))
    truth[20:40, 20:40] = scene(20)
    entry = gaussian_filter(truth, 0.8)
    entry[:, :12] += 0.3 * entry.sum() / (64 * 12)
    box = np.zeros(truth.shape, bool)
    box[16:44, 16:44] = True
    return entry, truth, box


@pytest.mark.validates("virgil.metrics.lawson_sigma_over_peak", roots=["literature"])
def test_lawson_normalises_over_the_whole_image():
    """Pinned: unit sum over the whole image, no box (the documented
    signature has none)."""
    entry, truth, _ = _halo_entry()
    assert vb.lawson(entry, truth) == pytest.approx(lawson_eq2(entry, truth), rel=1e-12)


@pytest.mark.validates("virgil.metrics.lawson_sigma_over_peak", roots=["literature"], kind="finding")
@pytest.mark.xfail(strict=True, reason="F19: flux outside the truth's box counts in virgil's normalisation, not in Eq. 2 over the box")
def test_lawson_with_flux_outside_the_comparison_box():
    entry, truth, box = _halo_entry()
    assert vb.lawson(entry, truth) == pytest.approx(lawson_eq2(entry, truth, box), rel=1e-6)


# ---------------------------------------------------------------- rms


def gaussian_fft(image, pixel, major, minor, pa_deg):
    """Convolution with an elliptical Gaussian of the given FWHMs (mas), major
    axis at position angle pa (North through East), by multiplying the
    zero-padded transform by exp(-2 pi^2 f^T S f). Covariance in (East, North);
    pixels run towards West (columns) and South (rows), and the quadratic form
    is unchanged by flipping both axes."""
    sM, sm = major / FWHM_PER_SIGMA, minor / FWHM_PER_SIGMA
    pa = np.radians(pa_deg)
    d = np.array([np.sin(pa), np.cos(pa)])  # (East, North)
    dp = np.array([np.cos(pa), -np.sin(pa)])
    cov = sM**2 * np.outer(d, d) + sm**2 * np.outer(dp, dp)
    n = image.shape[0]
    pad = 2 * n
    fc = np.fft.fftfreq(pad, pixel)[None, :]  # along columns = -East
    fr = np.fft.fftfreq(pad, pixel)[:, None]  # along rows = -North
    q = cov[0, 0] * fc**2 + cov[1, 1] * fr**2 + 2 * cov[0, 1] * fc * fr
    big = np.zeros((pad, pad))
    big[:n, :n] = image
    return np.fft.irfft2(np.fft.rfft2(big) * np.exp(-2 * np.pi**2 * q)[:, : pad // 2 + 1], s=(pad, pad))[:n, :n]


def rms_ref(entry, truth, kernel):
    e, r = kernel(entry / entry.sum()), kernel(truth / truth.sum())
    return np.sqrt(np.mean((e - r) ** 2))


@pytest.mark.validates("virgil.metrics.rms_convolved", roots=["literature"])
@pytest.mark.parametrize("name", [c[0] for c in CASES])
@pytest.mark.parametrize("fwhm", [1.0, 2.0, 4.0])
def test_rms_convolved_matches_scipy_and_crosscheck(name, fwhm, metric):
    truth, pixel = truth_for(name)
    entry = entries(truth)["perturbed"]
    # The beam is a FWHM in mas; the reference convolves with scipy at 8 sigma.
    sigma = fwhm / pixel / FWHM_PER_SIGMA
    ref = rms_ref(entry, truth, lambda x: gaussian_filter(x, sigma, truncate=8.0, mode="constant"))
    virgil = vb.rms_convolved(entry, truth, pixel, vb.circular_beam(fwhm))
    # virgil's kernel is finite (odd grid): 1e-4 to 3e-4 at 4 mas on this field.
    assert virgil == pytest.approx(ref, rel=1e-3)
    # crosscheck truncates its kernel at scipy's default 4 sigma (1e-4..1e-3 effect).
    assert virgil == pytest.approx(cm.rms_convolved(entry, truth, fwhm / pixel) * 1e-6, rel=3e-2)  # its 4-sigma kernel and renormalisation after zero-padded edges

    metric("rms_rel_diff_vs_scipy8", abs(virgil - ref) / ref)


@pytest.mark.validates("virgil.metrics.rms_convolved", roots=["literature"])
@pytest.mark.parametrize("pa", [0.0, 90.0, 30.0, 135.0])
def test_rms_convolved_elliptical_beam_position_angle(pa):
    """Beam major axis at PA North through East: PA 0 along rows (North),
    90 along columns (East-West); any PA against an exact Gaussian transform."""
    truth, pixel = scene(), PIXEL
    entry = entries(truth)["perturbed"]
    ref = rms_ref(entry, truth, lambda x: gaussian_fft(x, pixel, 2.4, 1.0, pa))
    virgil = vb.rms_convolved(entry, truth, pixel, vb.beam(2.4, 1.0, pa))
    assert virgil == pytest.approx(ref, rel=2e-4)


@pytest.mark.validates("virgil.metrics.rms_convolved", roots=["literature"])
def test_rms_convolved_normalisation_and_relative():
    truth, pixel = scene(), PIXEL
    entry = entries(truth)["blurred"]
    b = vb.circular_beam(2.0)
    base = vb.rms_convolved(entry, truth, pixel, b)
    assert vb.rms_convolved(9 * entry, 0.1 * truth, pixel, b) == pytest.approx(base, rel=1e-12)
    peak = gaussian_filter(truth, 2.0 / pixel / FWHM_PER_SIGMA, truncate=8.0, mode="constant").max()
    assert vb.rms_convolved(entry, truth, pixel, b, relative=True) == pytest.approx(base / peak, rel=2e-5)
    # No beam: the plain rms of the two unit-sum images.
    assert vb.rms_convolved(entry, truth, pixel) == pytest.approx(np.sqrt(np.mean((entry / entry.sum() - truth) ** 2)), rel=1e-12)


@pytest.mark.validates("virgil.metrics.rms_convolved", roots=["literature"], kind="finding")
@pytest.mark.xfail(strict=True, raises=ValueError, reason="F17: docs say pixel_scale_mas is needed only with a beam; arrays without a beam raise")
def test_rms_convolved_array_without_beam_needs_no_pixel_scale():
    truth = scene()
    from virgil.metrics import rms_convolved

    assert float(rms_convolved(entries(truth)["blurred"], truth)) > 0


# ----------------------------------------------------------- resample


@pytest.mark.validates("virgil.metrics.resample", roots=["mathematics"])
def test_resample_is_exact_area_average_and_conserves_flux():
    rng = np.random.default_rng(5)
    img = rng.random((32, 32))
    # Coarser by 2 on the same centre: exactly the sum of each 2x2 block.
    blocks = img.reshape(16, 2, 16, 2).sum(axis=(1, 3))
    np.testing.assert_allclose(vb.resample(img, 0.2, 16, 0.4), blocks, atol=1e-12)
    # Padding with empty sky (same pixels, wider field) keeps every value; flux conserved.
    wide = vb.resample(img, 0.2, 40, 0.2)
    np.testing.assert_allclose(wide[4:-4, 4:-4], img, atol=1e-12)
    assert wide.sum() == pytest.approx(img.sum(), rel=1e-12)
    # Finer by 2: each pixel's flux spreads over four, total conserved, block sums returned.
    fine = vb.resample(img, 0.2, 64, 0.1)
    assert fine.sum() == pytest.approx(img.sum(), rel=1e-12)
    np.testing.assert_allclose(fine.reshape(32, 2, 32, 2).sum(axis=(1, 3)), img, atol=1e-12)
    # Cropping drops flux outside the new field.
    assert vb.resample(img, 0.2, 16, 0.2).sum() == pytest.approx(img[8:24, 8:24].sum(), rel=1e-12)


@pytest.mark.validates("virgil.metrics.resample", roots=["mathematics"])
def test_resample_agrees_with_crosscheck_on_a_smooth_image(metric):
    """crosscheck.image_metrics.resample point-samples (bilinear after a
    smoothing) and renormalises to unit sum; virgil integrates exactly. The
    two definitions agree where the image is smooth on the pixel scale."""
    n = 64
    x = (np.arange(n) - (n - 1) / 2) * PIXEL
    img = np.exp(-0.5 * (x[None, :] ** 2 + x[:, None] ** 2) / 1.8**2)
    img /= img.sum()
    virgil = vb.resample(img, PIXEL, 32, 2 * PIXEL)
    ours = cm.resample(img, n * PIXEL, 32, 2 * PIXEL)
    assert virgil.sum() == pytest.approx(1.0, rel=1e-9)
    diff = np.abs(virgil / virgil.sum() - ours).max() / ours.max()
    metric("resample_vs_crosscheck_peak_fraction", diff)
    assert diff < 0.02


# -------------------------------------------------------------- align


@pytest.mark.validates("virgil.metrics.align", roots=["mathematics"])
@pytest.mark.parametrize("shift", [(2, -3), (0, 4), (-3, 0), (1, 1)])
def test_align_recovers_whole_pixel_shifts_with_crosscheck_signs(shift):
    truth = scene()
    entry = cm.shifted(truth, shift)  # entry moved by (rows, columns) pixels
    moved, (dra, ddec) = vb.align(entry, truth, 6 * PIXEL, PIXEL)
    np.testing.assert_allclose(moved, truth, atol=1e-9)
    # crosscheck's integer shift is the correction in (rows, columns); rows run
    # South and columns West, so Delta RA = -columns, Delta Dec = -rows (pixels).
    _, (di, dj) = cm.align(np.pad(entry, 6), truth, 6)
    assert (dra, ddec) == pytest.approx((-dj * PIXEL, -di * PIXEL), abs=1e-6)
    assert (di, dj) == (-shift[0], -shift[1])


@pytest.mark.validates("virgil.metrics.align", roots=["mathematics"])
def test_align_subpixel_shift_recovered():
    """A known sub-pixel translation (an exact Fourier shift of a smooth
    image) is recovered to a tenth of a pixel."""
    truth = scene()
    k = np.fft.fftfreq(64)
    dr, dc = 0.4, -0.3
    entry = np.fft.ifft2(np.fft.fft2(truth) * np.exp(-2j * np.pi * (k[:, None] * dr + k[None, :] * dc))).real
    _, (dra, ddec) = vb.align(entry, truth, 3 * PIXEL, PIXEL)
    # The entry sits dr rows South and dc columns West of the truth; the
    # correction is Delta Dec = +dr pixels (North) and Delta RA = +dc pixels
    # (East, as columns run West) with the sign of the shift applied.
    assert dra == pytest.approx(dc * PIXEL, abs=0.1 * PIXEL)
    assert ddec == pytest.approx(dr * PIXEL, abs=0.1 * PIXEL)


# --------------------------------------------------------------- score


@pytest.mark.validates("virgil.metrics.score", roots=["mathematics"])
@pytest.mark.parametrize("name", ["scene", "spotted_star", "spiral"])
def test_score_matches_crosscheck_components_after_alignment(name, metric):
    truth, pixel = truth_for(name)
    for label, entry in {"blurred": gaussian_filter(truth, 1.5), "perturbed": entries(truth)["perturbed"]}.items():
        shifted = cm.shifted(entry, (2, -3))
        virgil = vb.score(shifted, truth, pixel, max_shift_mas=5 * pixel, beam=vb.circular_beam(2.0))
        assert not virgil["rotated"]
        assert (virgil["shift_dra_mas"], virgil["shift_ddec_mas"]) == pytest.approx((-3 * pixel, 2 * pixel), abs=0.03 * pixel)  # plus the sub-pixel refinement of an asymmetric NCC peak
        for key, ref in {
            "ncc": cm.ncc(entry, truth),
            "l1": cm.l1_score(entry, truth),
            "lawson": cm.lawson_sigma_over_peak(entry, truth)[0],
        }.items():
            assert virgil[key] == pytest.approx(ref, rel=1e-2, abs=1e-6), (label, key)
        plain = np.sqrt(np.mean((entry / entry.sum() - truth / truth.sum()) ** 2))
        assert virgil["rms"] == pytest.approx(plain, rel=1e-2)
        assert virgil["rms_convolved"] == pytest.approx(cm.rms_convolved(entry, truth, 2.0 / pixel) * 1e-6, rel=1e-2)
        metric(f"score_{label}_lawson_diff", abs(virgil["lawson"] - cm.lawson_sigma_over_peak(entry, truth)[0]))


@pytest.mark.validates("virgil.metrics.score", roots=["mathematics"])
def test_score_aligns_only_when_asked():
    """Documented: max_shift_mas=None does not align (crosscheck.score always
    aligns); the shifted entry is then scored where it lies."""
    truth = scene()
    shifted = cm.shifted(truth, (2, -3))
    unaligned = vb.score(shifted, truth, PIXEL)
    assert (unaligned["shift_dra_mas"], unaligned["shift_ddec_mas"]) == (0.0, 0.0)
    assert unaligned["ncc"] == pytest.approx(cm.ncc(shifted, truth), abs=1e-12)
    assert unaligned["lawson"] == pytest.approx(cm.lawson_sigma_over_peak(shifted, truth)[0], rel=1e-12)
    assert unaligned["lawson"] > 0.1
    aligned = vb.score(shifted, truth, PIXEL, max_shift_mas=5 * PIXEL)
    ours = cm.score(shifted, 64 * PIXEL, truth, PIXEL, beam_mas=2.0, max_shift=5)
    assert aligned["lawson"] == pytest.approx(0.0, abs=1e-7) and ours["lawson"] == pytest.approx(0.0, abs=1e-7)
    assert aligned["ncc"] == pytest.approx(1.0, abs=1e-12) and ours["ncc"] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.validates("virgil.metrics.score", roots=["mathematics"])
def test_score_v2_only_keeps_the_better_of_image_and_its_180_rotation():
    truth = scene()  # asymmetric, so the rotation is a different image
    entry = truth[::-1, ::-1]
    assert vb.ncc(entry, truth) < 0.99
    with_phase = vb.score(entry, truth, PIXEL)
    assert not with_phase["rotated"] and with_phase["ncc"] < 0.99
    v2 = vb.score(entry, truth, PIXEL, v2_only=True)
    assert v2["rotated"] and v2["ncc"] == pytest.approx(1.0, abs=1e-12)
    assert v2["lawson"] == pytest.approx(0.0, abs=1e-9)
    ours = cm.score(entry, 64 * PIXEL, truth, PIXEL, beam_mas=2.0, v2_only=True, max_shift=0)
    assert ours["orientation"] == "inverted" and ours["ncc"] == pytest.approx(1.0, abs=1e-12)
    # An entry that is already right is not rotated.
    assert not vb.score(truth, truth, PIXEL, v2_only=True)["rotated"]
    # Rotation and a shift together (the shift sign survives the rotation).
    both = vb.score(cm.shifted(entry, (2, 1)), truth, PIXEL, v2_only=True, max_shift_mas=4 * PIXEL)
    assert both["rotated"] and both["ncc"] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.validates("virgil.metrics.score", roots=["mathematics"])
def test_score_resamples_a_coarser_entry_onto_the_truth_grid():
    truth = scene(64, PIXEL)
    entry = vb.resample(truth, PIXEL, 32, 2 * PIXEL)  # exact area average of the truth
    out = vb.score(entry, truth, 2 * PIXEL, truth_pixel_scale_mas=PIXEL)
    # The coarse entry is blurrier than the truth, so imperfect but well correlated.
    assert 0.95 < out["ncc"] < 1.0
    ref = vb.resample(entry, 2 * PIXEL, 64, PIXEL)
    assert out["ncc"] == pytest.approx(cm.ncc(ref, truth), abs=1e-9)
