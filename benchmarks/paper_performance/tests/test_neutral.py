"""neutral.py: the NumPy reference scorers, against closed forms and the crosscheck chi-squared."""

import numpy as np
import pytest

import neutral


def test_selftest_passes():
    assert neutral._selftest() == 0


def test_chi2_matches_crosscheck_on_a_simulated_file(tmp_path):
    """On a small simulated binary file the wrapped independent-Gaussian chi2 equals crosscheck's plain-difference one."""
    pytest.importorskip("astropy")
    from crosscheck import chi2, simulate

    stations = np.array([[0, 0, 0], [30, 5, 0], [-10, 45, 0]], float)
    wl = np.linspace(1.5e-6, 2.4e-6, 4)
    x, y, f = 6.0, -4.0, 0.1

    def vis(u, v, w):
        return neutral.binary_vis(u, v, w, x, y, f)

    path = tmp_path / "bin.fits"
    simulate.observe(path, vis, stations, hour_angles_h=[-1.0, 0.0, 1.0], wavelengths=wl, dec_deg=-50.0,
                     sigma_v2=0.03, sigma_cp_deg=1.0, rng=np.random.default_rng(3), phase_noise="triangle")
    d = chi2.load(path)
    ref = chi2.chi2(d, vis, chord=False)  # plain difference, independent errors
    m_v2 = np.abs(vis(d["u"], d["v"], d["wl"])) ** 2
    m_cp = neutral.closure_phase(vis(d["u1"], d["v1"], d["wl3"]), vis(d["u2"], d["v2_"], d["wl3"]),
                                 vis(-d["u1"] - d["u2"], -d["v1"] - d["v2_"], d["wl3"]))
    got = neutral.chi2_v2_cp(d["v2"], d["dv2"], m_v2, d["cp"], d["dcp"], m_cp)
    assert got["n_v2"] == d["v2"].size and got["n_cp"] == d["cp"].size
    assert got["chi2"] == pytest.approx(ref, rel=1e-6)


def test_wrap_and_chi2_independent_of_branch():
    a = neutral.chi2_v2_cp([0.5], [0.1], [0.5], [0.1], [0.05], [0.1 + 2 * np.pi])
    assert a["chi2"] == pytest.approx(0.0, abs=1e-18)


def test_astrometry_loglike_peaks_at_truth():
    d = np.array([1.0, 2.0])
    best = neutral.astrometry_loglike(d, d, d, d, [0.1, 0.1], [0.1, 0.1], [0.2, 0.2])
    worse = neutral.astrometry_loglike(d, d, d + 0.05, d, [0.1, 0.1], [0.1, 0.1], [0.2, 0.2])
    assert best > worse


def test_tv2_virgil_minus_ehtim_is_the_two_edges():
    b = np.random.default_rng(1).uniform(size=(6, 6))
    diff = neutral.tv2_cost(b, "virgil") - neutral.tv2_cost(b, "ehtim")
    assert diff == pytest.approx(np.sum(b[0] ** 2) + np.sum(b[:, 0] ** 2))
    with pytest.raises(ValueError):
        neutral.tv2_cost(b, "other")


def test_image_objective_flux_invariant_and_positive():
    rng = np.random.default_rng(2)
    n, pix = 6, 0.5
    truth = rng.uniform(0.1, 1, (n, n))
    u, v = rng.uniform(-40, 40, (2, 9))
    data = {"npix": n, "pixel_mas": pix, "u_amp": u, "v_amp": v, "wavel_amp": np.full(9, 2e-6),
            "sigma_amp": np.full(9, 0.02)}
    x, y = neutral.pixel_grid(n, pix)
    data["amp"] = np.abs(neutral.dft_matrix(u, v, data["wavel_amp"], x, y) @ (truth / truth.sum()).ravel())
    j0 = neutral.image_objective(truth, data, beta=10.0)
    assert neutral.image_objective(5 * truth, data, beta=10.0) == pytest.approx(j0, rel=1e-12)
    other = truth.copy()
    other[2, 2] *= 2
    assert neutral.image_objective(other, data, beta=0.0) > 0.0
