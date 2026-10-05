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

import numpyro.distributions as dist
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
PRIOR_MEAN, PRIOR_SD = 0.0, 0.02
PRIOR = dist.Normal(PRIOR_MEAN, PRIOR_SD)  # a bare (mean, sd) tuple is an error since virgil#234


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
    m, sd = PRIOR_MEAN, PRIOR_SD
    worst_f = worst_b = 0.0
    for i, x in enumerate(XS):
        for j, y in enumerate(YS):
            g, r0 = gauss_newton(d, x, y)
            f_hat, s_hat = -(g @ r0) / (g @ g), (g @ g) ** -0.5

            def integrand(f):
                return np.exp(-0.5 * (np.sum((r0 + g * f) ** 2) - np.sum(r0**2))) * stats.norm.pdf(f, m, sd)

            log_b = np.log(integrate.quad(integrand, m - 15 * sd, m + 15 * sd, points=[f_hat], limit=500)[0])
            worst_f = max(worst_f, abs(float(out.flux[i, j]) - f_hat) / s_hat,
                          abs(float(out.flux_error[i, j]) / s_hat - 1))
            worst_b = max(worst_b, abs(float(out.log_bayes_factor[i, j]) - log_b))
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
    m, sd = PRIOR_MEAN, PRIOR_SD
    worst = 0.0
    for i, x in enumerate(XS):
        for j, y in enumerate(YS):
            null = ours.chi2(d, binary_vis(0.0, x, y))

            def integrand(f):
                return np.exp(-0.5 * (ours.chi2(d, binary_vis(f, x, y)) - null)) * stats.norm.pdf(f, m, sd)

            log_b = np.log(integrate.quad(integrand, m - 15 * sd, m + 15 * sd, limit=500)[0])
            worst = max(worst, abs(float(out.log_bayes_factor[i, j]) - log_b))
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


# ----------------------------------------------------- closure-phase offsets


def closure_offsets_reference(d, s, baseline=None, triangle=None, modes=None):
    """Δ log-likelihood of closure-phase offsets, written from virgil's docs
    (virgil.gains.ClosureOffsets, likelihood._whiten): per frame, the sines
    s = sin Δ divided by σ and projected on an orthonormal basis Q of the
    column space of the triangle matrix T of each channel, are Gaussian
    with covariance Qᵀ R Q per channel (R = T Tᵀ / 3), and offsets common to
    a frame's channels add modes: T e per baseline, one per triangle, or
    given shapes; Δ = log N(y; 0, Σ0 + V Vᵀ) - log N(y; 0, Σ0)."""
    sig = d["dcp"]
    nrow, nw = s.shape
    total = 0.0
    for mjd in np.unique(d["t3_mjd"]):
        rows = np.flatnonzero(d["t3_mjd"] == mjd)
        T = ours.triangle_matrix(d["t3_sta"][rows])
        R = T @ T.T / 3
        left, values, _ = np.linalg.svd(T, full_matrices=False)
        Q = left[:, values > 1e-9 * values.max()]
        k = Q.shape[1]
        P = [Q.T @ np.diag(1 / sig[rows, c]) for c in range(nw)]
        y = np.concatenate([P[c] @ s[rows, c] for c in range(nw)])
        S0 = np.zeros((k * nw, k * nw))
        for c in range(nw):
            S0[c * k:(c + 1) * k, c * k:(c + 1) * k] = Q.T @ R @ Q
        cols = []
        if baseline:
            cols += [np.concatenate([P[c] @ (baseline * T[:, b]) for c in range(nw)]) for b in range(T.shape[1])]
        if triangle:
            for t in range(len(rows)):
                e = np.zeros(len(rows))
                e[t] = triangle
                cols.append(np.concatenate([P[c] @ e for c in range(nw)]))
        for mode in [] if modes is None else modes:
            shape = mode.reshape(nrow, nw)
            cols.append(np.concatenate([P[c] @ shape[rows, c] for c in range(nw)]))
        V = np.array(cols).T
        zero = np.zeros(y.size)
        total += stats.multivariate_normal(zero, S0 + V @ V.T).logpdf(y) - stats.multivariate_normal(zero, S0).logpdf(y)
    return total


@pytest.fixture(scope="module")
def cp_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("offsets") / "cp4.fits"
    simulate.observe(path, binary_vis(0.05, 6.0, -4.0), UTS, hour_angles_h=[-2.0, 0.0, 2.0],
                     wavelengths=np.linspace(1.6e-6, 2.4e-6, 4), dec_deg=-50.0, sigma_v2=0.01, sigma_cp_deg=1.0,
                     rng=np.random.default_rng(5))
    d = ours.load(path)
    vis = binary_vis(0.045, 6.2, -3.9)
    model_cp = (np.angle(vis(d["u1"], d["v1"], d["wl3"])) + np.angle(vis(d["u2"], d["v2_"], d["wl3"]))
                - np.angle(vis(d["u1"] + d["u2"], d["v1"] + d["v2_"], d["wl3"])))
    return d, OIData(str(path)), np.sin(model_cp - d["cp"]), vm.BinaryModelCartesian(6.2, -3.9, 0.045)


@pytest.mark.parametrize(
    "kw",
    [pytest.param(k, id=i, marks=pytest.mark.validates("virgil.gains.closure_offsets", "virgil.likelihood.model_loglike", roots=["mathematics"]))
     for i, k in [("baseline", dict(baseline=0.02)), ("triangle", dict(triangle=0.03)),
                  ("both", dict(baseline=0.02, triangle=0.03))]],
)
def test_closure_offsets_match_a_dense_gaussian(cp_file, kw):
    d, data, s, model = cp_file
    got = float(model_loglike(model, data.with_closure_offsets(**kw))) - float(model_loglike(model, data))
    want = closure_offsets_reference(d, s, **kw)
    record("abs_dloglike", abs(got - want))
    assert abs(got - want) < 1e-9 * max(1.0, abs(want))


@pytest.mark.validates("virgil.gains.closure_offsets", "virgil.likelihood.model_loglike", roots=["mathematics"])
def test_supplied_closure_offset_modes(cp_file):
    """Supplied modes, one value per closure phase of ``data.phi`` (the
    file's (row, channel) order), each within one frame."""
    d, data, s, model = cp_file
    nrow, nw = s.shape
    epochs = np.unique(d["t3_mjd"])
    modes = np.zeros((2, nrow * nw))
    for k, (epoch, scale) in enumerate([(epochs[0], None), (epochs[1], 0.03)]):
        rows = np.flatnonzero(d["t3_mjd"] == epoch)
        shape = np.zeros((nrow, nw))
        shape[rows, :] = np.random.default_rng(1).normal(size=(len(rows), nw)) * 0.02 if scale is None else scale
        modes[k] = shape.ravel()
    got = float(model_loglike(model, data.with_closure_offsets(modes=modes))) - float(model_loglike(model, data))
    want = closure_offsets_reference(d, s, modes=modes)
    assert abs(got - want) < 1e-9 * max(1.0, abs(want))


# --------------------------------------------------------- RV zero points


@pytest.mark.parametrize("jitter", [0.0, 0.8], ids=["no jitter", "jitter"])
@pytest.mark.validates("virgil.orbits.RVData", roots=["mathematics"])
def test_rv_zero_points_match_a_dense_gaussian_and_its_conditional(jitter):
    """Given virgil's Keplerian model m (validated elsewhere), the
    zero-point-marginalised density is N(rv; m + A μ, C + A Λ Aᵀ), with A
    the instrument indicator, C = diag(d_rv² + jitter²), Λ = diag(sd²); the
    zero points' posterior is the Gaussian conditional of w given rv."""
    vo = pytest.importorskip("virgil.orbits")
    pytest.importorskip("jaxoplanet")
    rng = np.random.default_rng(2)
    n = 24
    mjd = np.sort(rng.uniform(59000, 60500, n))
    inst = np.array(["A"] * 10 + ["B"] * 8 + ["C"] * 6)
    orbit = vo.KeplerOrbit(800.0, 120.0, 0.3, 60.0, 40.0, 110.0, 25.0, t_ref=59000.0)
    q, gamma, dist = 0.7, 0.0, 120.0
    d_rv = rng.uniform(0.2, 0.6, n)
    m = np.asarray(vo.RVData(mjd, np.zeros(n), d_rv, instrument=inst).model(orbit, q, gamma, dist))
    offsets = {"A": 5.0, "B": -3.0, "C": 12.0}
    rv = m + np.array([offsets[i] for i in inst]) + rng.normal(size=n) * d_rv
    data = vo.RVData(mjd, rv, d_rv, instrument=inst)
    A = np.array([[1.0 if i == label else 0.0 for label in data.instruments] for i in inst])
    mean, sd = np.array([1.0, -2.0, 3.0]), np.array([20.0, 10.0, 30.0])
    C, Lam = np.diag(d_rv**2 + jitter**2), np.diag(sd**2)
    want = stats.multivariate_normal(m + A @ mean, C + A @ Lam @ A.T).logpdf(rv)
    got = float(data.marginal_loglike(orbit, q, gamma, dist, jitter=jitter, prior=(mean, sd)))
    S_dd, S_wd = C + A @ Lam @ A.T, Lam @ A.T
    post_mean = mean + S_wd @ np.linalg.solve(S_dd, rv - m - A @ mean)
    post_cov = Lam - S_wd @ np.linalg.solve(S_dd, S_wd.T)
    got_mean, got_cov = (np.asarray(x) for x in data.zero_point_posterior(orbit, q, gamma, dist, jitter=jitter,
                                                                          prior=(mean, sd)))
    record("abs_dloglike", abs(got - want))
    assert abs(got - want) < 1e-9 * max(1.0, abs(want))
    np.testing.assert_allclose(got_mean, post_mean, atol=1e-10)
    np.testing.assert_allclose(got_cov, post_cov, atol=1e-10)
