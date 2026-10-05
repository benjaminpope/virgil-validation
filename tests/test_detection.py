"""Detection statistics, injection limits and the log-uniform flux evidence,
against our own chi-squared (crosscheck.chi2, a three-telescope file read
with astropy) and SciPy.

* injection_limits (Gallenne et al. 2015, the injection method): the flux f
  at which nsigma(chi2_null(data + signal(f)) / chi2_null(data)) = sigma,
  solved here by Brent's method with our own chi-squared.
* detection_statistics: delta_chi2 (the best improvement over the grid
  positions, flux >= 0), log_bayes_factor (a log-sum-exp of the likelihood
  ratio with trapezoid prior weights in each axis's index) and max_snr (the
  largest best flux over its Laplace error), each rebuilt on the same grid.
* local_nsigma: the one-sided Gaussian tail of half a chi2_1, sqrt(delta_chi2).
* linear_flux_grid with LogUniform(f_min, f_max): the evidence ratio
  Z = int N(f; f_hat, s^2) p(f) df / N(0; f_hat, s^2), p = 1/(f ln(f_max/f_min)),
  and the posterior mean and sd, by adaptive quadrature in ln f (not
  virgil's fixed Gauss-Legendre rule).
* gaussian_null: simulated data minus the null prediction, over the errors,
  are standard normal (scaled by error_scale).
* flux_to_contrast and the magnitude helpers: their formulae.
"""

import jax
import numpy as np
import pytest
from scipy import integrate, optimize, special, stats

from crosscheck import chi2 as ours, simulate, sky
from evidence.plugin import record

vm = pytest.importorskip("virgil.models")
det = pytest.importorskip("virgil.detection")
lim = pytest.importorskip("virgil.limits")
from virgil.grid_fit import LogUniform, linear_flux_grid  # noqa: E402
from virgil.oidata import OIData  # noqa: E402

pytestmark = pytest.mark.x64

UTS3 = np.array([[-9.925, -20.335], [14.887, 30.502], [103.306, 43.999]])


def binary_vis(f, x, y):
    def vis(u, v, w):
        return (sky.vis_point(u, v, w) + f * sky.vis_point(u, v, w, x, y)) / (1 + f)

    return vis


@pytest.fixture(scope="module", params=[0.0, 0.03], ids=["null", "companion"])
def dataset(request, tmp_path_factory):
    path = tmp_path_factory.mktemp("det") / "d.fits"
    simulate.observe(path, binary_vis(request.param, 6.0, -4.0), UTS3, hour_angles_h=np.linspace(-3, 3, 7),
                     wavelengths=np.linspace(1.5e-6, 2.4e-6, 6), dec_deg=-50.0, sigma_v2=0.01, sigma_cp_deg=0.5,
                     rng=np.random.default_rng(8))
    return request.param, OIData(str(path)), ours.load(path)


def significance(ratio, ndof):
    return -special.ndtri_exp(stats.chi2.logsf(ndof * ratio, ndof) - np.log(2.0))


@pytest.mark.validates("virgil.limits.injection_limits", roots=["mathematics"])
def test_injection_limits_by_root_finding(dataset):
    kind, data, d = dataset
    if kind != "null":
        pytest.skip("limits are for companion-free data")
    xs, ys = np.array([-8.0, 5.0]), np.array([-7.0, 6.0])
    got = np.asarray(lim.injection_limits(data, vm.BinaryModelCartesian, {"dra": xs, "ddec": ys, "flux": [1e-3]}, sigma=3.0))
    ndof = ours.n_data(d)
    null = binary_vis(0.0, 0.0, 0.0)
    base = ours.chi2(d, null)
    worst = 0.0
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            def excess(f):
                # inject: the data plus the companion's signal, then score the null model
                injected = dict(d)
                sig = binary_vis(f, x, y)
                v2s = np.abs(sig(d["u"], d["v"], d["wl"])) ** 2 - np.abs(null(d["u"], d["v"], d["wl"])) ** 2
                t3 = sig(d["u1"], d["v1"], d["wl3"]) * sig(d["u2"], d["v2_"], d["wl3"]) * np.conj(
                    sig(d["u1"] + d["u2"], d["v1"] + d["v2_"], d["wl3"]))
                injected["v2"] = d["v2"] + v2s
                injected["cp"] = d["cp"] + np.angle(t3)
                return significance(ours.chi2(injected, null) / base, ndof) - 3.0

            want = optimize.brentq(excess, 1e-6, 1.0, xtol=1e-12)
            worst = max(worst, abs(got[i, j] / want - 1))
    record("max_rel_limit_difference", worst)
    assert worst < 1e-4


GRID = {"dra": np.linspace(-9.0, 9.0, 7), "ddec": np.linspace(-9.0, 9.0, 7), "flux": np.logspace(-4, -0.5, 60)}


def trapezoid_log_weights(n):
    w = np.ones(n)
    w[[0, -1]] = 0.5
    return np.log(w / w.sum())


@pytest.mark.validates("virgil.detection.detection_statistics", roots=["mathematics"])
def test_detection_statistics_rebuilt_on_the_grid(dataset):
    kind, data, d = dataset
    with np.errstate(all="ignore"):
        got = det.detection_statistics(data, vm.BinaryModelCartesian, GRID)
    null = ours.chi2(d, binary_vis(0.0, 0.0, 0.0))
    xs, ys, fs = GRID["dra"], GRID["ddec"], GRID["flux"]
    best, snr, terms = 0.0, -np.inf, []
    lw = trapezoid_log_weights(xs.size)[:, None, None] + trapezoid_log_weights(ys.size)[None, :, None] \
        + trapezoid_log_weights(fs.size)[None, None, :]
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            def c2(f, x=x, y=y):
                return ours.chi2(d, binary_vis(f, x, y))

            grid_c2 = np.array([c2(f) for f in fs])
            terms.append(-0.5 * (grid_c2 - null) + lw[i, j])
            f_hat = optimize.minimize_scalar(c2, bounds=(-0.2, 1.0), method="bounded", options={"xatol": 1e-12}).x
            best = max(best, null - c2(max(f_hat, 0.0)))
            h = 1e-4
            curv = (c2(f_hat + h) - 2 * c2(f_hat) + c2(f_hat - h)) / h**2
            snr = max(snr, f_hat / np.sqrt(2.0 / curv))
    log_b = special.logsumexp(np.concatenate([np.ravel(t) for t in terms]))
    record("abs_ddelta_chi2", abs(float(got["delta_chi2"]) - best))
    record("abs_dlogB", abs(float(got["log_bayes_factor"]) - log_b))
    assert abs(float(got["delta_chi2"]) - best) < 1e-5 * max(1.0, best)
    assert abs(float(got["log_bayes_factor"]) - log_b) < 1e-8 * max(1.0, abs(log_b))
    assert abs(float(got["max_snr"]) - snr) < 1e-3 * max(1.0, abs(snr))


@pytest.mark.validates("virgil.detection.local_nsigma", roots=["mathematics"])
def test_local_nsigma_is_the_half_chi2_tail():
    dc2 = np.array([0.25, 1.0, 4.0, 9.0, 25.0, 100.0])
    got = np.asarray(det.local_nsigma(dc2))
    want = -stats.norm.isf(0.5 * stats.chi2.sf(dc2, 1)) * -1  # one-sided tail of half a chi2_1
    np.testing.assert_allclose(got, want, rtol=1e-9)
    np.testing.assert_allclose(got, np.sqrt(dc2), rtol=1e-9)


@pytest.mark.parametrize("bounds", [(1e-4, 1.0), (1e-3, 0.1)])
@pytest.mark.validates("virgil.grid_fit.linear_flux_grid", roots=["mathematics"])
def test_log_uniform_evidence_by_quadrature(dataset, bounds):
    """The Jeffreys (log-uniform) prior's evidence ratio and posterior from
    the result's own f_hat and sigma_f, by adaptive quadrature in ln f."""
    _, data, _ = dataset
    lo, hi = bounds
    grid = {"dra": np.array([6.0, -3.0]), "ddec": np.array([-4.0, 7.0]), "flux": [1e-3]}
    res = linear_flux_grid(data, vm.BinaryModelCartesian, grid, prior=LogUniform(lo, hi))
    worst = 0.0
    for i in range(2):
        for j in range(2):
            fh, s = float(res.flux[i, j]), float(res.flux_error[i, j])
            norm = np.log(hi / lo)

            def log_ratio(lnf):  # N(f; fh, s)/N(0; fh, s) p(f) df, with p(f) df = dlnf / ln(f_max/f_min)
                f = np.exp(lnf)
                return -0.5 * ((f - fh) ** 2 - fh**2) / s**2 - np.log(norm)

            ln = np.linspace(np.log(lo), np.log(hi), 20001)
            peak = np.max(log_ratio(ln))  # factor out, so nothing overflows
            pts = [float(ln[np.argmax(log_ratio(ln))])]

            # moments weight by f^k = exp(k lnf)
            Z0 = integrate.quad(lambda t: np.exp(log_ratio(t) - peak), np.log(lo), np.log(hi), points=pts, limit=1000,
                                epsrel=1e-13, epsabs=0)[0]
            m1 = integrate.quad(lambda t: np.exp(t + log_ratio(t) - peak), np.log(lo), np.log(hi), points=pts,
                                limit=1000, epsrel=1e-13, epsabs=0)[0] / Z0
            m2 = integrate.quad(lambda t: np.exp(2 * t + log_ratio(t) - peak), np.log(lo), np.log(hi), points=pts,
                                limit=1000, epsrel=1e-13, epsabs=0)[0] / Z0
            sd = np.sqrt(max(m2 - m1**2, 0.0))
            log_Z = np.log(Z0) + peak
            worst = max(worst, abs(float(res.log_bayes_factor[i, j]) - log_Z))
            assert abs(float(res.posterior_mean[i, j]) / m1 - 1) < 1e-6
            assert abs(float(res.posterior_sd[i, j]) / sd - 1) < 1e-5
    record("max_abs_dlogB", worst)
    assert worst < 1e-6


@pytest.mark.parametrize("scale", [1.0, 1.5])
@pytest.mark.validates("virgil.detection.gaussian_null", roots=["statistics"])
def test_gaussian_null_noise_is_standard_normal(dataset, scale):
    _, data, _ = dataset
    scene = vm.BinaryModelCartesian(0.0, 0.0, 0.0)
    simulate_ = det.gaussian_null(data, scene, error_scale=scale)
    pred = np.asarray(data.model(scene))
    n_vis = np.asarray(data.vis).size
    draws = np.concatenate([(np.asarray(simulate_(jax.random.key(k)).vis) - pred[:n_vis]) / np.asarray(data.d_vis)
                            for k in range(40)])
    record("sd_over_scale", float(np.std(draws) / scale))
    assert abs(np.mean(draws)) < 4 * scale / np.sqrt(draws.size)
    assert abs(np.std(draws) / scale - 1) < 4 / np.sqrt(2 * draws.size)
    assert stats.kstest(draws / scale, "norm").pvalue > 1e-3


@pytest.mark.validates("virgil.limits.flux_to_contrast", "virgil.limits.flux_to_delta_mag", roots=["mathematics"])
def test_flux_contrast_and_magnitudes():
    f = np.array([1e-4, 0.01, 0.3, 1.0])
    np.testing.assert_allclose(np.asarray(lim.flux_to_contrast(f)), 1 / f, rtol=1e-15)
    np.testing.assert_allclose(np.asarray(lim.flux_to_delta_mag(f)), -2.5 * np.log10(f), rtol=1e-14)
    np.testing.assert_allclose(np.asarray(lim.delta_mag_to_flux(lim.flux_to_delta_mag(f))), f, rtol=1e-14)
    np.testing.assert_allclose(np.asarray(lim.contrast_to_flux(lim.flux_to_contrast(f))), f, rtol=1e-15)
