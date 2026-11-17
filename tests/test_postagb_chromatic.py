"""scripts/postagb_imaging.py: the chromatic test of design/gp_polar_hyperparameters.md §11, the ``chrom`` calibration arm's truth
(``chromatic_truth``) and ``--chrom``'s fitted ring (``chromatic_ring``, ``chromatic_summary``)."""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
try:  # needs virgil main (MultipoleField), as tests/test_postagb_imaging.py
    import postagb_imaging as P  # noqa: E402
except ImportError:
    P = None
pytestmark = [
    pytest.mark.skipif(
        P is None, reason="virgil without MultipoleField (install the postagb extra)"
    ),
    # guards our own construction of the chromatic truth and ring: the index gradient and star offset are checked against their
    # definitions, and the grey limit against PowerLaw; no standard or external code is involved
    pytest.mark.validates(
        "postagb_imaging.chromatic", roots=["mathematics"], kind="guard"
    ),
]

R, W, D, SIND1 = 4.0, 1.0, 0.6, -2.8
U = np.array([20.0, 35.0, 50.0, 20.0, 35.0, 50.0])
V = np.array([5.0, -10.0, 15.0, 5.0, -10.0, 15.0])


def grey_truth():
    ring = P.MultipoleDisk(
        P.GaussianRing(R, W),
        harmonics=P.Harmonics(re=0.1, im=0.1),
        inc=40.0,
        pa=30.0,
        n_r=48,
        flux=P.PowerLaw(0.4, D, P.L0),
    )
    star = P.UniformDisk(diam=0.4, flux=P.PowerLaw(1.0, SIND1, P.L0))
    return P.System(star=star, env=ring)


def test_chromatic_truth_has_the_stated_index_gradient_and_star_offset():
    truth, info = P.chromatic_truth(
        grey_truth(), dict(radius_mas=R, width_mas=W), SIND1
    )
    alpha = truth.env.spectral_index
    assert float(alpha(R - W) - alpha(R + W)) == pytest.approx(P.CAL_CHROM_DALPHA)
    assert float(alpha(R)) == pytest.approx(D)  # the grey ring's own index at the centre
    assert float(truth.star.flux.index) == pytest.approx(SIND1 + P.CAL_CHROM_STAR)
    assert info["star_index_fit"] == SIND1


def test_chromatic_truth_matches_the_grey_one_at_the_reference_wavelength_only():
    grey = grey_truth()
    chrom, _ = P.chromatic_truth(grey, dict(radius_mas=R, width_mas=W), SIND1)
    wl = np.array([P.L0] * 3 + [1.52e-6] * 3)
    vg, vc = np.asarray(grey.model(U, V, wl)), np.asarray(chrom.model(U, V, wl))
    np.testing.assert_allclose(vc[:3], vg[:3], rtol=1e-5, atol=1e-7)
    assert np.max(np.abs(vc[3:] - vg[3:])) > 1e-3


def test_a_flat_chromatic_ring_is_the_grey_power_law():
    """With zero slope the chromatic MultipoleDisk is PowerLaw(flux, index): the F_lambda convention is shared."""
    grey = grey_truth()
    flat, _ = P.chromatic_truth(
        grey, dict(radius_mas=R, width_mas=W), SIND1, d_alpha=0.0, d_star=0.0
    )
    wl = np.array([1.52e-6, 1.6e-6, 1.68e-6, 1.76e-6, 1.52e-6, 1.76e-6])
    np.testing.assert_allclose(
        np.asarray(flat.model(U, V, wl)),
        np.asarray(grey.model(U, V, wl)),
        rtol=1e-5,
        atol=1e-7,
    )


def test_chromatic_ring_priors_reach_its_parameters_and_scale_with_the_width():
    cfg = dict(half_mas=8.0, img_star_index=SIND1, env_index0=0.5)
    geo = dict(inc_deg=50.0, pa_deg=100.0, radius_mas=0.2, width_mas=1.5)
    rim, pri = P.chromatic_ring(cfg, geo, unit=1.2)
    assert float(rim.profile.radius) == 0.5  # clipped to the prior's floor
    scene = P.System(
        star=P.UniformDisk(diam=0.4, flux=P.PowerLaw(1.0, SIND1, P.L0)), rim=rim
    )
    for k in pri:  # every prior names a parameter of the scene
        scene.set(k, 0.0 if "slope" in k else 1.0)
    slope = pri["rim.spectral_index.slope"]
    assert float(slope.high) == pytest.approx(P.CHROM_DALPHA_MAX / (2 * 1.5))
    star = pri["star.flux.index"]
    assert (float(star.low), float(star.high)) == pytest.approx(
        (SIND1 - P.CHROM_STAR_WINDOW, SIND1 + P.CHROM_STAR_WINDOW)
    )
    _, pri_narrow = P.chromatic_ring(
        cfg, geo | dict(width_mas=0.2), unit=1.2
    )  # width below half a beam: the slope bound uses half a beam
    assert float(pri_narrow["rim.spectral_index.slope"].high) == pytest.approx(
        P.CHROM_DALPHA_MAX / 1.2
    )


def test_chromatic_summary_says_whether_the_fit_uses_colour():
    cfg = dict(half_mas=8.0, img_star_index=SIND1, env_index0=0.5)
    rim, _ = P.chromatic_ring(
        cfg, dict(inc_deg=50.0, pa_deg=100.0, radius_mas=3.0, width_mas=1.0), unit=1.2
    )
    env = grey_truth().env.set("flux", P.PowerLaw(0.5, D, P.L0))
    star = P.UniformDisk(diam=0.4, flux=P.PowerLaw(1.0, SIND1, P.L0))
    m = P.System(star=star, env=env, rim=rim.set("flux", 0.0001))
    s = P.chromatic_summary(m, cfg)
    assert not s["chrom_used"] and s["chrom_star_shift"] == 0.0
    m2 = m.set("rim.flux", 0.2).set("rim.spectral_index.slope", -0.5)  # 2 * 0.5 * 1 = 1 across +-w
    s2 = P.chromatic_summary(m2, cfg)
    assert s2["chrom_rim_d_alpha"] == pytest.approx(1.0) and s2["chrom_used"]
    assert P.chromatic_summary(m.set("star.flux.index", SIND1 - 0.6), cfg)[
        "chrom_used"
    ]
