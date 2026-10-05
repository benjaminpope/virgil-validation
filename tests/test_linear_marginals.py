"""Analytic marginalisation of linear parameters (Luger, Foreman-Mackey &
Hogg 2017) in virgil, against dense Gaussians and brute force.

* Calibration gains (virgil.gains, OIData.with_gains): the change in the
  log-likelihood when gains are switched on must be
  log N(r; 0, D + U U^T) - log N(r; 0, D), with r the V² (or |V|) residuals,
  D their variances and U = J tau m one column per mode, J = 2 V²_model for
  V² and |V|_model for amplitudes, the modes as documented: per (frame,
  telescope), per (frame, baseline), per (frame, baseline) shaped
  (lambda_ref / lambda)² with lambda_ref the median wavelength, and
  supplied shapes in sample order. The dense covariance is built here with
  NumPy from our own reading of the file, and evaluated with SciPy.
* linear_flux_grid: the closed-form flux and its error against a
  Gauss-Newton step on our own whitened residual vector; the Gauss-Newton
  refinement against a direct optimiser of our own chi-squared; the log
  Bayes factor against brute-force integration over the flux, exactly in
  the linear model, and against the true chi-squared where the model is
  linear (no companion). This is the evidence under a Gaussian prior on f
  (``prior=(mean, sd)``), pinned as "Gaussian-prior evidence": virgil's
  default priors are Jeffreys/invariant ones (log-uniform for a scale such
  as f), so it is not the default detection statistic.
"""

import numpy as np
import pytest
from astropy.io import fits
from scipy import integrate, optimize, stats

from crosscheck import chi2 as ours, simulate, sky
from evidence.plugin import record

vm = pytest.importorskip("virgil.models")
from virgil.grid_fit import linear_flux_grid  # noqa: E402
from virgil.likelihood import model_loglike  # noqa: E402
from virgil.oidata import OIData  # noqa: E402

pytestmark = pytest.mark.x64

UTS = np.array([[-9.925, -20.335], [14.887, 30.502], [44.915, 66.183], [103.306, 43.999]])
UTS3 = UTS[[0, 1, 3]]
WL = np.linspace(1.6e-6, 2.4e-6, 5)


def truth(u, v, w):
    return (sky.vis_uniform_disk(u, v, w, 1.5) + 0.05 * sky.vis_point(u, v, w, 6.0, -4.0)) / 1.05


def model_vis(u, v, w):
    return (sky.vis_uniform_disk(u, v, w, 1.45) + 0.04 * sky.vis_point(u, v, w, 6.2, -3.9)) / 1.04


MODEL = vm.System(star=vm.UniformDisk(1.45), comp=vm.PointSource(0.04, 6.2, -3.9))


def dense_delta(r, D, U):
    """log N(r; 0, diag(D) + U U^T) - log N(r; 0, diag(D))."""
    zero = np.zeros(r.size)
    return stats.multivariate_normal(zero, np.diag(D) + U @ U.T).logpdf(r) - stats.multivariate_normal(zero, np.diag(D)).logpdf(r)


def mode_columns(mjd, sta, wl, telescope=None, baseline=None, chromatic=None):
    """Our modes on the (row, channel) samples of a visibility table."""
    nrow, nw = wl.shape
    lref = np.median(np.unique(wl))
    cols = []
    for m in np.unique(mjd):
        rows = np.flatnonzero(mjd == m)
        if telescope:
            for s in np.unique(sta[rows]):
                shape = np.zeros((nrow, nw))
                shape[rows[(sta[rows] == s).any(1)], :] = 1.0
                cols.append(telescope * shape.ravel())
        for row in rows:
            if baseline:
                shape = np.zeros((nrow, nw))
                shape[row, :] = 1.0
                cols.append(baseline * shape.ravel())
            if chromatic:
                shape = np.zeros((nrow, nw))
                shape[row, :] = (lref / wl[row]) ** 2
                cols.append(chromatic * shape.ravel())
    return np.array(cols).T


@pytest.fixture(scope="module")
def v2_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("gains") / "v2.fits"
    simulate.observe(path, truth, UTS, hour_angles_h=[-2.0, 0.0, 2.0], wavelengths=WL, dec_deg=-50.0,
                     sigma_v2=0.01, sigma_cp_deg=1.0, rng=np.random.default_rng(3))
    return path


GROUPS = {
    "telescope": dict(telescope=0.02),
    "baseline": dict(baseline=0.01),
    "chromatic": dict(chromatic=0.03),
    "all three": dict(telescope=0.02, baseline=0.01, chromatic=0.03),
}


@pytest.mark.parametrize(
    "groups",
    [pytest.param(g, id=k, marks=pytest.mark.validates("virgil.gains.gain_modes", "virgil.likelihood.model_loglike",
                                                        roots=["mathematics"])) for k, g in GROUPS.items()],
)
def test_gains_match_a_dense_gaussian_on_v2(v2_file, groups):
    d = ours.load(v2_file)
    data = OIData(str(v2_file))
    v2m = np.abs(model_vis(d["u"], d["v"], d["wl"])) ** 2
    r = (d["v2"] - v2m).ravel()
    U = (2 * v2m).ravel()[:, None] * mode_columns(d["v2_mjd"], d["v2_sta"], d["wl"], **groups)
    want = dense_delta(r, (d["dv2"] ** 2).ravel(), U)
    got = float(model_loglike(MODEL, data.with_gains(**groups))) - float(model_loglike(MODEL, data))
    record("abs_dloglike", abs(got - want))
    assert abs(got - want) < 1e-9 * max(1.0, abs(want))


@pytest.mark.validates("virgil.gains.gain_modes", "virgil.likelihood.model_loglike", roots=["mathematics"])
def test_supplied_modes_are_in_sample_order(v2_file):
    """Supplied modes, one value per sample in the order of the file's
    (row, channel) samples (virgil's ``u``)."""
    d = ours.load(v2_file)
    data = OIData(str(v2_file))
    v2m = np.abs(model_vis(d["u"], d["v"], d["wl"])) ** 2
    r = (d["v2"] - v2m).ravel()
    shapes = np.random.default_rng(7).normal(size=(3, r.size)) * 0.02
    want = dense_delta(r, (d["dv2"] ** 2).ravel(), (2 * v2m).ravel()[:, None] * shapes.T)
    got = float(model_loglike(MODEL, data.with_gains(modes=shapes))) - float(model_loglike(MODEL, data))
    assert abs(got - want) < 1e-9 * max(1.0, abs(want))


@pytest.mark.validates("virgil.gains.gain_modes", "virgil.likelihood.model_loglike", roots=["mathematics"])
def test_gains_on_amplitudes_use_the_amplitude_jacobian(tmp_path):
    """For |V| data (OI_VIS VISAMP) the Jacobian is |V|_model."""
    path = tmp_path / "amp.fits"
    simulate.observe_visibilities(path, truth, UTS, hour_angles_h=[-2.0, 0.0, 2.0], wavelengths=WL, dec_deg=-50.0,
                                  sigma=0.01, rng=np.random.default_rng(4))
    with fits.open(path) as h:
        t = h["OI_VIS"].data
        amp, err = t["VISAMP"], t["VISAMPERR"]
        u, v, mjd, sta = t["UCOORD"], t["VCOORD"], t["MJD"], t["STA_INDEX"]
        wl = np.broadcast_to(h["OI_WAVELENGTH"].data["EFF_WAVE"].astype(float), amp.shape)
    am = np.abs(model_vis(u[:, None], v[:, None], wl))
    r = (amp - am).ravel()
    data = OIData(str(path))
    for groups in GROUPS.values():
        U = am.ravel()[:, None] * mode_columns(mjd, sta, wl, **groups)
        want = dense_delta(r, (err**2).ravel(), U)
        got = float(model_loglike(MODEL, data.with_gains(**groups))) - float(model_loglike(MODEL, data))
        assert abs(got - want) < 1e-9 * max(1.0, abs(want))


# ----------------------------------------------------------- linear flux map


def binary_vis(f, x, y):
    def vis(u, v, w):
        return (sky.vis_point(u, v, w) + f * sky.vis_point(u, v, w, x, y)) / (1 + f)

    return vis


XS, YS = np.array([6.0, -3.0]), np.array([-4.0, 7.0])
PRIOR = (0.0, 0.02)


def flux_file(tmp_path, flux, seed=8):
    path = tmp_path / f"flux{flux}.fits"
    simulate.observe(path, binary_vis(flux, 6.0, -4.0), UTS3, hour_angles_h=np.linspace(-3, 3, 7),
                     wavelengths=np.linspace(1.5e-6, 2.4e-6, 6), dec_deg=-50.0, sigma_v2=0.01, sigma_cp_deg=0.5,
                     rng=np.random.default_rng(seed))
    return ours.load(path), OIData(str(path))


def gauss_newton(d, x, y, at=0.0, h=1e-7):
    """g = dr/df at ``at`` (central difference of our residual vector), and
    the residuals there."""
    def r(f):
        return ours.residuals(d, binary_vis(f, x, y))

    return (r(at + h) - r(at - h)) / (2 * h), r(at)


@pytest.mark.parametrize("flux", [0.0, 0.01], ids=["no companion", "companion"])
@pytest.mark.validates("virgil.grid_fit.linear_flux_grid", roots=["mathematics"])
def test_linear_flux_and_gaussian_prior_evidence(tmp_path, flux):
    """Closed form (no refinement): f = -(g.r0)/(g.g), sigma = (g.g)^-1/2,
    and log B by quadrature over f of the linearised likelihood times the
    prior."""
    d, data = flux_file(tmp_path, flux)
    out = linear_flux_grid(data, vm.BinaryModelCartesian, {"dra": XS, "ddec": YS, "flux": [1e-3]}, prior=PRIOR)
    m, sd = PRIOR
    worst_f = worst_b = 0.0
    for i, x in enumerate(XS):
        for j, y in enumerate(YS):
            g, r0 = gauss_newton(d, x, y)
            f_hat, s_hat = -(g @ r0) / (g @ g), (g @ g) ** -0.5

            def integrand(f):
                return np.exp(-0.5 * (np.sum((r0 + g * f) ** 2) - np.sum(r0**2))) * stats.norm.pdf(f, m, sd)

            log_b = np.log(integrate.quad(integrand, m - 15 * sd, m + 15 * sd, points=[f_hat], limit=500)[0])
            worst_f = max(worst_f, abs(float(out["flux"][i, j]) - f_hat) / s_hat,
                          abs(float(out["flux_error"][i, j]) / s_hat - 1))
            worst_b = max(worst_b, abs(float(out["log_bayes_factor"][i, j]) - log_b))
    record("max_flux_difference_in_sigma", worst_f)
    record("max_abs_dlogB_linear_model", worst_b)
    assert worst_f < 1e-5
    assert worst_b < 1e-5


@pytest.mark.validates("virgil.grid_fit.linear_flux_grid", roots=["mathematics"])
def test_gaussian_prior_evidence_is_the_true_one_without_a_companion(tmp_path):
    """With no companion the likelihood is nearly linear in f over the
    posterior (it is not exactly: the binary's V² and closure phases are
    nonlinear in f), so the closed-form log B is close to the integral of
    the true likelihood (measured 1.5e-3)."""
    d, data = flux_file(tmp_path, 0.0)
    out = linear_flux_grid(data, vm.BinaryModelCartesian, {"dra": XS, "ddec": YS, "flux": [1e-3]}, prior=PRIOR)
    m, sd = PRIOR
    worst = 0.0
    for i, x in enumerate(XS):
        for j, y in enumerate(YS):
            null = ours.chi2(d, binary_vis(0.0, x, y))

            def integrand(f):
                return np.exp(-0.5 * (ours.chi2(d, binary_vis(f, x, y)) - null)) * stats.norm.pdf(f, m, sd)

            log_b = np.log(integrate.quad(integrand, m - 15 * sd, m + 15 * sd, limit=500)[0])
            worst = max(worst, abs(float(out["log_bayes_factor"][i, j]) - log_b))
    record("max_abs_dlogB_true", worst)
    assert worst < 0.01


@pytest.mark.validates("virgil.grid_fit.linear_flux_grid", roots=["mathematics"])
def test_gauss_newton_refinement_reaches_the_optimum(tmp_path):
    """A bright companion (0.3): with 5 Gauss-Newton steps, the flux at each
    position is the minimiser of our own chi-squared."""
    d, data = flux_file(tmp_path, 0.3)
    flux = np.asarray(linear_flux_grid(data, vm.BinaryModelCartesian, {"dra": XS, "ddec": YS, "flux": [1e-3]},
                                       n_iter=5)[0])
    worst = 0.0
    for i, x in enumerate(XS):
        for j, y in enumerate(YS):
            best = optimize.minimize_scalar(lambda f: ours.chi2(d, binary_vis(f, x, y)), bounds=(-0.5, 1.0),
                                            method="bounded", options={"xatol": 1e-12}).x
            worst = max(worst, abs(flux[i, j] / best - 1))
    record("max_rel_flux_difference", worst)
    assert worst < 1e-5
