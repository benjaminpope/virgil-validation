"""Grid search and detection limits against independent references.

The data: our simulated 3-telescope VLTI files (one closure phase per
snapshot, so closure phases are independent), V² and closure phases with
noise. Our chi-squared (crosscheck.chi2) reads the file with astropy and
uses closed-form binary visibilities; the significance and the limits are
written from Absil et al. (2011) and Ruffio et al. (2018) with SciPy and
mpmath.
"""

import mpmath
import numpy as np
import pytest
from scipy import optimize, special, stats

from crosscheck import chi2 as ours, simulate, sky
from evidence.plugin import record

vm = pytest.importorskip("virgil.models")
from virgil.grid_fit import (  # noqa: E402
    laplace_flux_uncertainty_grid,
    likelihood_grid,
    optimized_flux_grid,
)
from virgil.limits import absil_limits, nsigma, ruffio_upperlimit  # noqa: E402
from virgil.oidata import OIData  # noqa: E402

pytestmark = pytest.mark.x64

UTS3 = np.array([[-9.925, -20.335], [14.887, 30.502], [103.306, 43.999]])
HA = np.linspace(-3, 3, 7)
WL = np.linspace(1.5e-6, 2.4e-6, 6)
DRA, DDEC, FLUX = 6.0, -4.0, 0.03


def binary_vis(dra, ddec, flux):
    def vis(u, v, w):
        return (sky.vis_point(u, v, w) + flux * sky.vis_point(u, v, w, dra, ddec)) / (1 + flux)

    return vis


@pytest.fixture(scope="module", params=["companion", "none"])
def dataset(request, tmp_path_factory):
    flux = FLUX if request.param == "companion" else 0.0
    path = tmp_path_factory.mktemp("g") / "g.fits"
    simulate.observe(
        path, binary_vis(DRA, DDEC, flux), UTS3, hour_angles_h=HA,
        wavelengths=WL, dec_deg=-50.0, sigma_v2=0.01, sigma_cp_deg=0.5,
        rng=np.random.default_rng(8),
    )
    return request.param, OIData(str(path)), ours.load(path)


@pytest.mark.validates("virgil.grid_fit.likelihood_grid", "virgil.likelihood.model_loglike", roots=["mathematics"])
def test_likelihood_grid_is_minus_half_chi2(dataset):
    _, data, d = dataset
    xs = np.linspace(-10, 10, 9)
    grid = np.asarray(
        likelihood_grid(data, vm.BinaryModelCartesian, {"dra": xs, "ddec": xs, "flux": [FLUX]})
    )[..., 0]
    want = np.array([[-0.5 * ours.chi2(d, binary_vis(x, y, FLUX)) for y in xs] for x in xs])
    # equal up to a constant (the normalisation terms)
    diff = (grid - grid.max()) - (want - want.max())
    record("max_abs_dloglike", np.max(np.abs(diff)))
    assert np.max(np.abs(diff)) < 1e-9


@pytest.mark.validates("virgil.limits.nsigma", roots=["mathematics"])
@pytest.mark.parametrize(
    "ratio,ndof", [(1.2, 50), (2.0, 300), (1.05, 2000), (5.0, 400), (1.0, 100)]
)
def test_nsigma_matches_the_chi2_tail(ratio, ndof):
    """Absil et al. (2011): the significance of a chi-squared ratio r is
    the two-sided Gaussian equivalent of P(chi2_ndof > ndof r)."""
    log_p = stats.chi2.logsf(ndof * ratio, ndof)
    want = -special.ndtri_exp(log_p - np.log(2.0))
    got = float(nsigma(ratio, 1.0, ndof))
    record("abs_dsigma", abs(got - want))
    assert abs(got - want) < 1e-6 * max(1.0, want)


@pytest.mark.validates("virgil.limits.nsigma", roots=["mathematics"])
@pytest.mark.parametrize("true", [0.7, 1.8])
def test_nsigma_uses_the_ratio_to_the_true_chi2(true):
    """With chi2r_true != 1 the significance is that of the ratio
    chi2r_test / chi2r_true; ignoring chi2r_true would give another answer
    (a control)."""
    ratio, ndof = 1.2, 200
    want = -special.ndtri_exp(stats.chi2.logsf(ndof * ratio, ndof) - np.log(2.0))
    got = float(nsigma(ratio * true, true, ndof))
    record("abs_dsigma", abs(got - want))
    assert abs(got - want) < 1e-6 * max(1.0, want)
    assert abs(float(nsigma(ratio * true, 1.0, ndof)) - want) > 0.5


@pytest.mark.validates("virgil.inference.laplace_cov", roots=["mathematics"])
def test_laplace_cov_is_the_inverse_hessian_of_half_chi2(dataset):
    """laplace_cov of the binary at (dra, ddec, flux) against the inverse of
    the Hessian of chi2/2 from our own chi-squared, by central differences."""
    which, data, d = dataset
    if which != "companion":
        pytest.skip("with no companion this point is not a minimum, and the Hessian is not positive-definite")
    from virgil.inference import laplace_cov

    x0 = np.array([DRA, DDEC, FLUX])
    got = np.asarray(laplace_cov(x0, ["dra", "ddec", "flux"], data, vm.BinaryModelCartesian))

    def half_chi2(x):
        return 0.5 * ours.chi2(d, binary_vis(*x))

    h = np.array([1e-3, 1e-3, 1e-5])
    hess = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            ei, ej = np.eye(3)[i] * h[i], np.eye(3)[j] * h[j]
            hess[i, j] = (half_chi2(x0 + ei + ej) - half_chi2(x0 + ei - ej)
                          - half_chi2(x0 - ei + ej) + half_chi2(x0 - ei - ej)) / (4 * h[i] * h[j])
    want = np.linalg.inv(0.5 * (hess + hess.T))
    scale = np.sqrt(np.outer(np.diag(want), np.diag(want)))
    worst = float(np.max(np.abs(got - want) / scale))
    record("max_rel_dcov", worst)
    assert worst < 1e-4


@pytest.mark.validates("virgil.grid_fit.optimized_flux_grid", "virgil.grid_fit.laplace_flux_uncertainty_grid", roots=["mathematics"])
def test_best_flux_and_its_curvature_at_fixed_positions(dataset):
    _, data, d = dataset
    xs = np.array([-8.0, DRA, 9.0])
    ys = np.array([DDEC, 7.0])
    samples = {"dra": xs, "ddec": ys, "flux": np.linspace(0.0, 0.1, 6)}
    best = np.asarray(optimized_flux_grid(data, vm.BinaryModelCartesian, samples))
    sigma = np.asarray(laplace_flux_uncertainty_grid(data, vm.BinaryModelCartesian, samples, flux=best))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            def c2(f):
                return ours.chi2(d, binary_vis(x, y, f))

            f0 = optimize.minimize_scalar(c2, bracket=(-0.05, 0.0, 0.1), tol=1e-12).x
            h = 1e-4
            curv = (c2(f0 + h) - 2 * c2(f0) + c2(f0 - h)) / h**2  # d2(chi2)/df2
            s0 = np.sqrt(2.0 / curv)
            assert abs(best[i, j] - f0) < 1e-3 * s0, (x, y, best[i, j], f0)
            assert abs(sigma[i, j] / s0 - 1) < 1e-3, (x, y, sigma[i, j], s0)


@pytest.mark.validates("virgil.limits.absil_limits", roots=["mathematics"])
def test_absil_limits_by_root_finding(dataset):
    """At each position, the flux f at which nsigma(chi2(f)/chi2(0)) = 3,
    with the number of data points as degrees of freedom (as documented)."""
    kind, data, d = dataset
    if kind != "none":
        pytest.skip("limits are for companion-free data")
    xs = np.array([-12.0, -3.0, 5.0, 11.0])
    ys = np.array([-9.0, 2.0, 8.0])
    samples = {"dra": xs, "ddec": ys, "flux": np.logspace(-4, -1, 8)}
    got = np.asarray(absil_limits(data, vm.BinaryModelCartesian, samples, sigma=3.0))
    ndof = ours.n_data(d)
    null = ours.chi2(d, binary_vis(0.0, 0.0, 0.0))

    def significance(f, x, y):
        ratio = ours.chi2(d, binary_vis(x, y, f)) / null
        return -special.ndtri_exp(stats.chi2.logsf(ndof * ratio, ndof) - np.log(2.0))

    worst = 0.0
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            want = optimize.brentq(lambda f: significance(f, x, y) - 3.0, 1e-6, 1.0, xtol=1e-12)
            worst = max(worst, abs(got[i, j] / want - 1))
    record("max_rel_limit_difference", worst)
    assert worst < 1e-3


@pytest.mark.validates("virgil.limits.absil_limits", roots=["mathematics"])
@pytest.mark.parametrize("start", [1e-6, 0.5, 0.95], ids=["far-below", "far-above", "saturated"])
def test_absil_limits_from_a_single_bad_start(dataset, start):
    """The flux axis only sets the start (virgil#191): a single value far
    below or far above the limit, where the significance saturates (a
    flat loss), must still give the root of nsigma = 3. virgil brackets by
    decades, bisects 14 times in log flux and returns the midpoint, so its
    limit is within 10^(1/2^15) - 1 = 7.0e-5 of the root."""
    kind, data, d = dataset
    if kind != "none":
        pytest.skip("limits are for companion-free data")
    xs, ys = np.array([-12.0, 5.0]), np.array([-9.0, 8.0])
    samples = {"dra": xs, "ddec": ys, "flux": [start]}
    got = np.asarray(absil_limits(data, vm.BinaryModelCartesian, samples, sigma=3.0))
    ndof = ours.n_data(d)
    null = ours.chi2(d, binary_vis(0.0, 0.0, 0.0))

    def significance(f, x, y):
        ratio = ours.chi2(d, binary_vis(x, y, f)) / null
        return -special.ndtri_exp(stats.chi2.logsf(ndof * ratio, ndof) - np.log(2.0))

    worst = 0.0
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            want = optimize.brentq(lambda f: significance(f, x, y) - 3.0, 1e-6, 1.0, xtol=1e-12)
            worst = max(worst, abs(got[i, j] / want - 1))
    record("max_rel_limit_difference", worst)
    assert worst < 1e-4


@pytest.mark.validates("virgil.limits.ruffio_upperlimit", roots=["mathematics"])
@pytest.mark.parametrize("mean,sigma", [(0.01, 0.01), (0.0, 0.02), (-0.05, 0.01), (-0.3, 0.01), (0.2, 0.01)])
@pytest.mark.parametrize("percentile", [0.5, 0.84, stats.norm.cdf(2.0), 0.999])
def test_ruffio_upperlimit_is_the_truncated_gaussian_quantile(mean, sigma, percentile):
    """Ruffio et al. (2018, eq. 8): the percentile of N(mean, sigma)
    truncated to flux >= 0, computed with mpmath at 50 digits (so the far
    tail, 30 sigma below zero, is exact)."""
    mpmath.mp.dps = 50
    a = mpmath.mpf(-mean) / sigma
    tail = mpmath.ncdf(-a)  # P(z > a)
    q = mpmath.mpf(1) - mpmath.mpf(percentile)
    target = q * tail
    lo, hi = a, a + 40  # P(z > lo) >= target > P(z > hi)
    for _ in range(300):  # bisection: the tail is monotonic in z
        mid = (lo + hi) / 2
        if mpmath.ncdf(-mid) > target:
            lo = mid
        else:
            hi = mid
    z = (lo + hi) / 2
    want = float(mean + sigma * z)
    got = float(ruffio_upperlimit(mean, sigma, percentile))
    record("rel_difference", abs(got / want - 1))
    assert abs(got / want - 1) < 1e-6


@pytest.mark.slow
@pytest.mark.validates("virgil.grid_fit.optimized_flux_grid", "virgil.grid_fit.laplace_flux_uncertainty_grid", roots=["statistics"], tier="B", kind="regression")
def test_flux_pulls_at_the_true_position(tmp_path):
    """Over 200 noisy realisations, the best flux at the true position, in
    units of its Laplace uncertainty, is N(0, 1)."""
    rng = np.random.default_rng(29)
    n, pulls = 200, []
    path = tmp_path / "p.fits"
    for _ in range(n):
        simulate.observe(
            path, binary_vis(DRA, DDEC, FLUX), UTS3, hour_angles_h=HA, wavelengths=WL,
            dec_deg=-50.0, sigma_v2=0.01, sigma_cp_deg=0.5, rng=rng,
        )
        data = OIData(str(path))
        samples = {"dra": [DRA], "ddec": [DDEC], "flux": [0.0, FLUX, 0.06]}
        best = np.asarray(optimized_flux_grid(data, vm.BinaryModelCartesian, samples))[0, 0]
        sig = np.asarray(laplace_flux_uncertainty_grid(data, vm.BinaryModelCartesian, samples, flux=[[best]]))[0, 0]
        pulls.append((best - FLUX) / sig)
    pulls = np.array(pulls)
    record("pull_mean", pulls.mean())
    record("pull_sd", pulls.std())
    assert abs(pulls.mean()) < 3 / np.sqrt(n)
    assert abs(pulls.std() - 1) < 3 / np.sqrt(2 * n)
