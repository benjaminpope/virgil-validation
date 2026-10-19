"""The ALMA cross-checks (src/crosscheck/alma.py) on NumPy-made data with
known answers: they check our own reference code, not virgil."""

import numpy as np
import pytest

from crosscheck import alma
from evidence.plugin import record

MAS = alma.MAS
NPIX, PIXEL = 64, 10.0


def uv_points(m=400, seed=0, umax=2e5):
    rng = np.random.default_rng(seed)
    return rng.uniform(-umax, umax, (2, m))


def point_vis(u, v, sources):
    return sum(f * np.exp(-2j * np.pi * (u * e + v * n) * MAS) for f, e, n in sources)


def pixel_of(east, north, npix=NPIX, pixel=PIXEL):
    return npix // 2 - int(round(north / pixel)), npix // 2 - int(round(east / pixel))


@pytest.mark.validates("crosscheck.alma", roots=["mathematics"], kind="reference")
def test_dft_dirty_image_of_a_point_source_peaks_at_its_position_with_its_flux():
    """V = F exp(-2 pi i (u x + v y)) inverts to F at (x, y): +East is column
    0 side, +North is row 0 side, and the peak value is the flux."""
    u, v = uv_points()
    for east, north in [(60.0, 100.0), (-120.0, 30.0), (0.0, -200.0)]:
        vis = point_vis(u, v, [(0.7, east, north)])
        img = alma.dft_dirty_image(u, v, vis, np.ones_like(u), NPIX, PIXEL)
        r, c = np.unravel_index(np.argmax(img), img.shape)
        assert (r, c) == pixel_of(east, north)
        assert img[r, c] == pytest.approx(0.7, rel=1e-12)
        assert alma.peak_offset_mas(img, PIXEL) == (east, north)
    record("peak_value_error", abs(img.max() - 0.7))


@pytest.mark.validates("crosscheck.alma", roots=["mathematics"], kind="reference")
def test_dft_dirty_image_is_linear_weighted_and_conjugation_mirrors_it():
    u, v = uv_points(seed=1)
    w = np.random.default_rng(2).uniform(0.5, 2, len(u))
    a, b = point_vis(u, v, [(1.0, 50, 20)]), point_vis(u, v, [(0.5, -80, -60)])
    ia = alma.dft_dirty_image(u, v, a, w, NPIX, PIXEL)
    ib = alma.dft_dirty_image(u, v, b, w, NPIX, PIXEL)
    iab = alma.dft_dirty_image(u, v, a + b, w, NPIX, PIXEL)
    assert np.allclose(iab, ia + ib, atol=1e-14)
    # conjugating V is reflecting through the centre: pixel (r, c) -> (n-r, n-c) modulo the grid
    ic = alma.dft_dirty_image(u, v, np.conj(a), w, NPIX, PIXEL)
    idx = (NPIX - np.arange(NPIX)) % NPIX
    assert np.allclose(ic[1:, 1:], ia[np.ix_(idx[1:], idx[1:])], atol=1e-13)  # row and column 0 have no partner
    # a constant weight scale changes nothing; chunking changes nothing
    assert np.allclose(alma.dft_dirty_image(u, v, a, 7 * w, NPIX, PIXEL, chunk=33), ia, atol=1e-14)


@pytest.mark.validates("crosscheck.alma", roots=["mathematics"], kind="reference")
def test_gridded_dirty_image_follows_the_dft_when_points_sit_on_cells():
    """Points at cell centres have no gridding error: the FFT route equals the DFT."""
    n, pix = 64, 10.0
    du = 1.0 / (n * pix * MAS)
    rng = np.random.default_rng(3)
    iu, iv = rng.integers(-20, 21, (2, 300))
    u, v = iu * du, iv * du
    vis = point_vis(u, v, [(1.0, 70, -40), (0.4, -90, 110)])
    w = rng.uniform(0.5, 2, len(u))
    d = alma.dft_dirty_image(u, v, vis, w, n, pix)
    g = alma.gridded_dirty_image(u, v, vis, w, n, pix)
    record("max_abs_difference", np.abs(d - g).max())
    assert np.allclose(d, g, atol=1e-12)


@pytest.mark.validates("crosscheck.alma", roots=["mathematics"], kind="reference")
def test_gridded_dirty_image_is_tapered_by_a_sinc_off_cell():
    """Off-cell points are moved to the nearest cell: the image peak stays at
    the source and its height is reduced, never increased."""
    n, pix = 64, 10.0
    u, v = uv_points(2000, seed=4, umax=0.4 / (pix * MAS))
    vis = point_vis(u, v, [(1.0, 80, 30)])
    g = alma.gridded_dirty_image(u, v, vis, np.ones_like(u), n, pix)
    assert alma.peak_offset_mas(g, pix) == (80.0, 30.0)
    assert 0.9 < g.max() <= 1.0 + 1e-12


def pair_data(sigma, n_base=40, n_t=20, seed=0, dt=6.0, drift=0.0, wscale=1.0):
    """Noise-only visibilities on n_base baselines, one channel, one scan, n_t
    consecutive integrations, per-component sigma, weights wscale / sigma^2."""
    rng = np.random.default_rng(seed)
    a1, a2 = np.repeat(np.arange(n_base), n_t), np.repeat(np.arange(n_base) + 100, n_t)
    t = np.tile(np.arange(n_t) * dt, n_base)
    sig = np.full(len(t), sigma)
    return {
        "re": (rng.normal(size=len(t)) * sig + drift * t).astype(np.float32),
        "im": (rng.normal(size=len(t)) * sig).astype(np.float32),
        "weight": (wscale / sig**2).astype(np.float32), "ant1": a1, "ant2": a2,
        "chan": np.zeros(len(t), np.int16), "scan": np.ones(len(t), np.int32), "time": t, "t_int_s": dt,
    }


@pytest.mark.validates("crosscheck.alma", roots=["statistics"], kind="reference")
def test_successive_difference_sigma_is_one_when_the_weights_are_right():
    got = alma.successive_difference_sigma(pair_data(0.3, n_base=300, n_t=20))
    record("s2", got["s2"])
    assert got["n_pairs"] == 300 * 10  # runs of 20 adjacent samples give 10 pairs each
    assert abs(got["s2"] - 1.0) < 3 * got["s2_err"]
    assert got["s2_err"] < 0.02


@pytest.mark.validates("crosscheck.alma", roots=["statistics"], kind="reference")
def test_successive_difference_sigma_detects_weights_that_are_too_large_or_small():
    """Weights 4 times too large (sigma underestimated by 2) give s^2 = 4."""
    got = alma.successive_difference_sigma(pair_data(0.3, n_base=300, wscale=4.0))
    assert got["s2"] == pytest.approx(4.0, abs=4 * 4 * got["s2_err"] / 1.0)
    assert got["s"] == pytest.approx(2.0, rel=0.05)
    got = alma.successive_difference_sigma(pair_data(0.3, n_base=300, wscale=0.25))
    assert got["s"] == pytest.approx(0.5, rel=0.05)


@pytest.mark.validates("crosscheck.alma", roots=["statistics"], kind="reference")
def test_successive_difference_pairs_need_the_same_baseline_scan_and_adjacent_times():
    d = pair_data(0.3, n_base=4, n_t=4)
    # a gap larger than 1.5 t_int splits a baseline's run: times 0, 6, | 30, 36 -> two pairs of two
    d["time"] = np.tile([0.0, 6.0, 30.0, 36.0], 4)
    assert alma.successive_difference_sigma(d)["n_pairs"] == 8
    # a change of scan splits it too
    d = pair_data(0.3, n_base=4, n_t=4)
    d["scan"] = np.tile([1, 1, 2, 2], 4).astype(np.int32)
    assert alma.successive_difference_sigma(d)["n_pairs"] == 8
    # odd run of 5 adjacent samples: 2 non-overlapping pairs
    d = pair_data(0.3, n_base=1, n_t=5)
    assert alma.successive_difference_sigma(d)["n_pairs"] == 2
    # different baselines never pair
    d = pair_data(0.3, n_base=6, n_t=1)
    assert alma.successive_difference_sigma(d)["n_pairs"] == 0


@pytest.mark.validates("crosscheck.alma", roots=["statistics"], kind="reference")
def test_successive_difference_sigma_is_biased_up_by_a_changing_sky():
    """A slow drift between integrations adds to |dV|^2: s^2 > 1, the bias the
    docs bound with the short-baseline data."""
    flat = alma.successive_difference_sigma(pair_data(0.3, n_base=300))
    drifting = alma.successive_difference_sigma(pair_data(0.3, n_base=300, drift=0.05))
    # Re changes by 0.05 * 6 = 0.3 between neighbours: |dV|^2 gains 0.09 over 4 sigma^2 = 0.36
    assert drifting["s2"] - flat["s2"] == pytest.approx(0.09 / 0.36, abs=0.04)


@pytest.mark.validates("crosscheck.alma", roots=["mathematics"], kind="reference")
def test_image_visibilities_of_pixels_are_the_sum_of_their_point_sources():
    """Each pixel is a point at its centre: the separable sum equals the
    sum of the closed-form point-source visibilities, at off-centre pixels
    on a non-square grid."""
    rng = np.random.default_rng(3)
    image = rng.uniform(size=(7, 9))
    x = (np.arange(9) - 2.5) * 11.0 * MAS  # East offsets, radians
    y = (np.arange(7) - 1.0) * 7.0 * MAS
    u, v = uv_points(50, seed=1, umax=3e5)
    got = alma.image_visibilities(image, x, y, u, v, chunk=16)
    want = sum(image[r, c] * np.exp(-2j * np.pi * (u * x[c] + v * y[r])) for r in range(7) for c in range(9))
    assert np.abs(got - want).max() < 1e-13 * image.sum()
    record("image_visibilities_error", np.abs(got - want).max())


@pytest.mark.validates("crosscheck.alma", roots=["mathematics"], kind="reference")
def test_adjoint_image_is_the_gradient_of_the_visibility_sum():
    """d/dI[r, c] of F = Re sum_k conj(a_k) V_k(I) is the adjoint sum with z = conj(a),
    checked against central differences of ``image_visibilities``."""
    rng = np.random.default_rng(5)
    image = rng.uniform(size=(6, 5))
    x = (np.arange(5) - 2.0) * 9.0 * MAS
    y = (np.arange(6) - 3.0) * 9.0 * MAS
    u, v = uv_points(40, seed=2, umax=4e5)
    a = rng.normal(size=40) + 1j * rng.normal(size=40)

    def f(img):
        return np.sum((np.conj(a) * alma.image_visibilities(img, x, y, u, v)).real)

    got = alma.adjoint_image(np.conj(a), x, y, u, v, chunk=16)
    h = 1e-6
    fd = np.zeros_like(image)
    for r in range(6):
        for c in range(5):
            d = np.zeros_like(image)
            d[r, c] = h
            fd[r, c] = (f(image + d) - f(image - d)) / (2 * h)
    assert np.abs(got - fd).max() < 1e-8 * np.abs(fd).max()
    record("adjoint_image_error", np.abs(got - fd).max())
