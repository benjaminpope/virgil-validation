"""Independent mathematical and statistical validation of virgil 0.4.2
F22-free likelihood / grid / cross-validation primitives (`is_tied`,
`tied_log_prior`, `group_outlier_mixture`, `group_outlier_probabilities`,
`grid_fit.Gaussian`, `grid_fit.LogUniform`, `LinearFluxGrid`, `fold_masks`,
`save_record`, `load_records`, `aggregate`, `aggregate_files`) and honest
F22-dependent inference routines (`chain_init_params`, `FitResult`,
`gauss_newton_mass`, `GaussNewtonMass`, `optimized_likelihood_grid`, `cv_task`,
`cross_validate`, `svi.variational`, `VariationalResult`).
"""

from __future__ import annotations

import gc
import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro.infer import NUTS
import optax
import pytest
from scipy import integrate, linalg, optimize, special, stats

from crosscheck import chi2 as xchi2, simulate as xsim, sky as xsky, smc_tempering_ref as ref
from evidence.plugin import record

vlike = pytest.importorskip("virgil.likelihood")
vgf = pytest.importorskip("virgil.grid_fit")
vcv = pytest.importorskip("virgil.crossval")
vfit = pytest.importorskip("virgil.fitting")
vsvi = pytest.importorskip("virgil.svi")
vpr = pytest.importorskip("virgil.priors")
from virgil.models import BinaryModelCartesian  # noqa: E402
from virgil.oidata import OIData  # noqa: E402
from virgil.oifits import read_oifits  # noqa: E402

pytestmark = pytest.mark.x64

UTS3 = np.array([[-9.925, -20.335], [14.887, 30.502], [103.306, 43.999]])


@pytest.fixture(scope="module", autouse=True)
def _clear_jax_caches():
    yield
    jax.clear_caches()
    gc.collect()


def _binary_cloud(flux: float, dra: float, ddec: float) -> xsky.Cloud:
    """Point-source binary with unit primary at the origin and relative flux ``flux`` at ``(dra, ddec)``."""
    return xsky.mix([xsky.point(0.0, 0.0), xsky.point(dra, ddec)], [1.0, flux])


# ---------------------------------------------------------------------------
# 0. Reference module self-checks (crosscheck.smc_tempering_ref)
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "crosscheck.smc_tempering_ref",
    roots=["mathematics"],
    kind="reference",
    headline="max_outlier_quad_vs_subset_err",
)
def test_smc_tempering_ref_outlier_flux_and_ridge_identities():
    """Verify every remaining oracle helper in crosscheck.smc_tempering_ref
    against closed-form analytical identities and independent quadrature:
    Jeffreys Beta(1/2, 1/2) subset & K=2 weights, all 3 outputs of Gaussian and
    LogUniform linear-flux priors (including exp1/erf closed forms at f_hat=0),
    all folds and aggregate rules of K-fold ridge CV, analytical nonlinear
    Jacobian pullback for gauss_newton_precision_fd_ref, and analytical
    correlated bivariate Gaussian moments/quantiles for posterior_2d_grid_moments_ref."""
    rng = np.random.default_rng(1024)
    max_sub_err = 0.0
    for k_groups in (1, 5, 10):
        ll = rng.uniform(-80.0, 20.0, size=k_groups)
        ol = rng.uniform(-75.0, 15.0, size=k_groups)
        lz_q, q_q = ref.beta_outlier_mixture_quad_ref(ll, ol, alpha=0.5, beta_param=0.5)
        lz_s, q_s = ref.jeffreys_outlier_mixture_exact_subset_ref(ll, ol)
        max_sub_err = max(max_sub_err, abs(lz_q - lz_s), float(np.max(np.abs(q_q - q_s))))
    # Hand-expanded K = 2 Jeffreys weights: E[(1-e)^2] = 3/8, E[e(1-e)] = 1/8, E[e^2] = 3/8
    ll2 = np.array([-1.2, -0.4])
    ol2 = np.array([-0.7, -2.1])
    L0, L1 = np.exp(ll2)
    B0, B1 = np.exp(ol2)
    z2_hand = 0.375 * L0 * L1 + 0.125 * (B0 * L1 + L0 * B1) + 0.375 * B0 * B1
    q2_hand = np.array(
        [(0.125 * B0 * L1 + 0.375 * B0 * B1) / z2_hand, (0.125 * L0 * B1 + 0.375 * B0 * B1) / z2_hand]
    )
    lz2_s, q2_s = ref.jeffreys_outlier_mixture_exact_subset_ref(ll2, ol2)
    max_sub_err = max(max_sub_err, abs(lz2_s - math.log(z2_hand)), float(np.max(np.abs(q2_s - q2_hand))))
    assert max_sub_err < 1e-13

    # Luger Gaussian prior: log_bayes_factor, posterior_mean, and posterior_sd vs 1D quadrature
    fh, sf, mu0, tau0 = 0.024, 0.005, 0.004, 0.03
    g_closed = ref.linear_flux_gaussian_prior_ref(np.array([fh]), np.array([sf]), mu0, tau0)
    g_weight = lambda f: (
        math.exp(0.5 * (fh / sf) ** 2 - 0.5 * ((f - fh) / sf) ** 2)
        * math.exp(-0.5 * ((f - mu0) / tau0) ** 2)
        / (tau0 * math.sqrt(2.0 * math.pi))
    )
    bf_quad, _ = integrate.quad(g_weight, -0.5, 0.5, epsabs=1e-14, epsrel=1e-13)
    gm1_quad, _ = integrate.quad(lambda f: f * g_weight(f), -0.5, 0.5, epsabs=1e-14, epsrel=1e-13)
    g_mean_q = gm1_quad / bf_quad
    gv2_quad, _ = integrate.quad(
        lambda f: ((f - g_mean_q) ** 2) * g_weight(f), -0.5, 0.5, epsabs=1e-14, epsrel=1e-13
    )
    err_luger = max(
        abs(float(g_closed["log_bayes_factor"][0]) - math.log(bf_quad)),
        abs(float(g_closed["posterior_mean"][0]) - g_mean_q),
        abs(float(g_closed["posterior_sd"][0]) - math.sqrt(gv2_quad / bf_quad)),
    )
    assert err_luger < 1e-12

    # LogUniform linear-flux prior: exact exp1/erf closed forms at f_hat = 0 + linear-f quad at f_hat > 0
    f_min, f_max, sf0 = 1e-3, 0.08, 0.01
    lu0 = ref.linear_flux_loguniform_prior_ref(0.0, sf0, f_min, f_max)
    a_lo = 0.5 * (f_min / sf0) ** 2
    a_hi = 0.5 * (f_max / sf0) ** 2
    j0 = 0.5 * float(special.exp1(a_lo) - special.exp1(a_hi))
    j1 = sf0 * math.sqrt(0.5 * math.pi) * float(special.erf(math.sqrt(a_hi)) - special.erf(math.sqrt(a_lo)))
    j2 = (sf0**2) * (math.exp(-a_lo) - math.exp(-a_hi))
    lu0_mean_exact = j1 / j0
    lu0_sd_exact = math.sqrt(j2 / j0 - lu0_mean_exact**2)
    lu0_bf_exact = math.log(j0 / math.log(f_max / f_min))
    err_lu_ref = max(
        abs(lu0["log_bayes_factor"] - lu0_bf_exact),
        abs(lu0["posterior_mean"] - lu0_mean_exact),
        abs(lu0["posterior_sd"] - lu0_sd_exact),
    )
    lu_pos = ref.linear_flux_loguniform_prior_ref(fh, sf, f_min, f_max)
    span_lu = math.log(f_max / f_min)
    peak_lu = 0.5 * (fh / sf) ** 2
    lu_f_w = lambda f: math.exp(-0.5 * ((f - fh) / sf) ** 2) / (f * span_lu)
    z_lu_f, _ = integrate.quad(lu_f_w, f_min, f_max, points=[fh], epsabs=1e-14, epsrel=1e-13)
    m1_lu_f, _ = integrate.quad(lambda f: f * lu_f_w(f), f_min, f_max, points=[fh], epsabs=1e-14, epsrel=1e-13)
    mean_lu_f = m1_lu_f / z_lu_f
    v2_lu_f, _ = integrate.quad(
        lambda f: ((f - mean_lu_f) ** 2) * lu_f_w(f), f_min, f_max, points=[fh], epsabs=1e-14, epsrel=1e-13
    )
    err_lu_ref = max(
        err_lu_ref,
        abs(lu_pos["log_bayes_factor"] - (peak_lu + math.log(z_lu_f))),
        abs(lu_pos["posterior_mean"] - mean_lu_f),
        abs(lu_pos["posterior_sd"] - math.sqrt(v2_lu_f / z_lu_f)),
    )
    # Boundary-layer LogUniform regimes outside [f_min, f_max] (f_hat / sf = -1000 and (f_hat - f_max) / sf = +300)
    for fh_ex, sf_ex, f_lo, f_hi in [(-0.10, 1e-4, 1e-4, 0.2), (0.50, 1e-3, 1e-4, 0.2)]:
        got_ex = ref.linear_flux_loguniform_prior_ref(fh_ex, sf_ex, f_lo, f_hi)
        f_star = min(max(fh_ex, f_lo), f_hi)
        peak_rel = 0.5 * f_star * (2.0 * fh_ex - f_star) / (sf_ex * sf_ex)
        span_ex = math.log(f_hi / f_lo)
        if fh_ex <= f_lo:
            sc_t = (sf_ex * sf_ex) / (f_lo - fh_ex)
            t_cap = min((f_hi - f_lo) / sc_t, 800.0)
            w_t = lambda t, sc=sc_t, fl=f_lo, sp=span_ex, s=sf_ex: (sc / ((fl + sc * t) * sp)) * math.exp(
                -t - 0.5 * ((sc * t) / s) ** 2
            )
            z0_ex, _ = integrate.quad(w_t, 0.0, t_cap, epsabs=0.0, epsrel=1e-13)
            d1_ex, _ = integrate.quad(lambda t: (sc_t * t) * w_t(t), 0.0, t_cap, epsabs=0.0, epsrel=1e-13)
            dm_ex = d1_ex / z0_ex
            m_ex = f_lo + dm_ex
            z2_ex, _ = integrate.quad(
                lambda t: ((sc_t * t - dm_ex) ** 2) * w_t(t), 0.0, t_cap, epsabs=0.0, epsrel=1e-13
            )
        else:
            sc_t = (sf_ex * sf_ex) / (fh_ex - f_hi)
            t_cap = min((f_hi - f_lo) / sc_t, 800.0)
            w_t = lambda t, sc=sc_t, fh_b=f_hi, sp=span_ex, s=sf_ex: (sc / ((fh_b - sc * t) * sp)) * math.exp(
                -t - 0.5 * ((sc * t) / s) ** 2
            )
            z0_ex, _ = integrate.quad(w_t, 0.0, t_cap, epsabs=0.0, epsrel=1e-13)
            d1_ex, _ = integrate.quad(lambda t: (-sc_t * t) * w_t(t), 0.0, t_cap, epsabs=0.0, epsrel=1e-13)
            dm_ex = d1_ex / z0_ex
            m_ex = f_hi + dm_ex
            z2_ex, _ = integrate.quad(
                lambda t: ((-sc_t * t - dm_ex) ** 2) * w_t(t), 0.0, t_cap, epsabs=0.0, epsrel=1e-13
            )
        bf_ex = peak_rel + math.log(z0_ex)
        sd_ex = math.sqrt(z2_ex / z0_ex)
        err_lu_ref = max(
            err_lu_ref,
            abs(got_ex["log_bayes_factor"] - bf_ex) / max(abs(bf_ex), 1.0),
            abs(got_ex["posterior_mean"] - m_ex) / m_ex,
            abs(got_ex["posterior_sd"] - sd_ex) / sd_ex,
        )
    assert err_lu_ref < 1e-11

    # Ridge K-fold CV across all folds (theta_hat, score, chi2, group_score) + aggregate_cv_ref
    x_mat = rng.normal(size=(12, 3))
    sig = rng.uniform(0.1, 0.3, size=12)
    y = x_mat @ np.array([0.8, -0.5, 0.3]) + rng.normal(0.0, 0.05, size=12)
    fold = np.tile(np.arange(3), 4)
    grps = np.where(np.arange(12) < 6, 0, 1)
    recs = ref.ridge_kfold_cv_ref(x_mat, y, sig, fold, [0.1, 0.4, 2.0], groups=grps, include_log_norm=True)
    err_ridge = 0.0
    for rec in recs:
        lam_v = rec["setting"]
        k_id = rec["fold"]
        tr_m = fold != k_id
        te_m = fold == k_id
        opt = optimize.minimize(
            lambda th, tm=tr_m, lv=lam_v: 0.5 * np.sum(((y[tm] - x_mat[tm] @ th) / sig[tm]) ** 2)
            + 0.5 * lv * np.sum(th**2),
            np.zeros(3),
            jac=lambda th, tm=tr_m, lv=lam_v: -(x_mat[tm].T @ ((y[tm] - x_mat[tm] @ th) / (sig[tm] ** 2)))
            + lv * th,
            method="BFGS",
            options={"gtol": 1e-13},
        )
        r_te = (y[te_m] - x_mat[te_m] @ opt.x) / sig[te_m]
        c2_want = float(np.sum(r_te**2))
        sc_want = 0.5 * c2_want + float(np.sum(np.log(sig[te_m])))
        g0_m = grps[te_m] == 0
        g0_sc = 0.5 * float(np.sum(r_te[g0_m] ** 2)) + float(np.sum(np.log(sig[te_m][g0_m])))
        err_ridge = max(
            err_ridge,
            float(linalg.norm(rec["theta_hat"] - opt.x)),
            abs(rec["chi2"] - c2_want),
            abs(rec["score"] - sc_want),
            abs(rec["group_score"][0] - g0_sc),
        )
    assert err_ridge < 1e-10

    agg_hand = ref.aggregate_cv_ref(
        [
            {"setting": 0.1, "fold": 0, "score": 2.0, "chi2": 4.0},
            {"setting": 0.1, "fold": 1, "score": 4.0, "chi2": 8.0},
            {"setting": 0.1, "fold": 2, "score": 3.0, "chi2": 6.0},
            {"setting": 1.0, "fold": 0, "score": 2.4, "chi2": 4.8},
            {"setting": 1.0, "fold": 1, "score": 3.4, "chi2": 6.8},
            {"setting": 1.0, "fold": 2, "score": 4.4, "chi2": 8.8},
        ],
        larger_is_simpler=True,
    )
    # Setting 0.1 has mean 3.0, sample SD 1.0, SE 1/sqrt(3) = 0.57735; setting 1.0 has mean 3.4 <= 3.57735
    assert agg_hand["best"] == 0 and agg_hand["one_se"] == 1
    assert abs(float(agg_hand["se"][0]) - 1.0 / math.sqrt(3.0)) < 1e-15

    # Analytical nonlinear Jacobian pullback check for gauss_newton_precision_fd_ref
    z_nl = np.array([math.log(1.5), -0.4])
    gn_fd = ref.gauss_newton_precision_fd_ref(
        lambda z: np.array([math.sqrt(2.0) * math.exp(z[0]) - 1.0, z[1] - 2.0 * z[0]]),
        lambda z: -0.5 * (4.0 * z[0] ** 2 + 9.0 * z[1] ** 2),
        z_nl,
    )
    j_exact = np.array([[math.sqrt(2.0) * 1.5, 0.0], [-2.0, 1.0]])
    f_exact = j_exact.T @ j_exact + np.diag([4.0, 9.0])
    inv_exact = linalg.inv(f_exact)
    sd_e = np.sqrt(np.diag(inv_exact))
    cond_exact = float(np.linalg.cond(inv_exact / np.outer(sd_e, sd_e)))
    err_gn_ref = max(
        float(linalg.norm(np.asarray(gn_fd["precision"]) - f_exact)),
        float(linalg.norm(np.asarray(gn_fd["inverse_mass"]) - inv_exact)),
        abs(float(gn_fd["condition"]) - cond_exact),
    )
    assert err_gn_ref < 1e-5  # second-order prior FD at eps_p = 1e-5 has ~1e-6 float64 rounding floor

    # Analytical correlated bivariate Gaussian check for posterior_2d_grid_moments_ref
    mu_2d = np.array([1.2, -0.7])
    cov_2d = np.array([[0.36, -0.18], [-0.18, 0.25]])
    prec_2d = linalg.inv(cov_2d)
    log_z_2d_want = 1.75 + math.log(2.0 * math.pi) + 0.5 * math.log(float(linalg.det(cov_2d)))

    def biv_log_post(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        dx = xx - mu_2d[0]
        dy = yy - mu_2d[1]
        q_form = prec_2d[0, 0] * dx**2 + 2.0 * prec_2d[0, 1] * dx * dy + prec_2d[1, 1] * dy**2
        return 1.75 - 0.5 * q_form

    g2d = ref.posterior_2d_grid_moments_ref(biv_log_post, (-3.6, 6.0), (-4.7, 3.3), n_grid=301)
    z_q = stats.norm.ppf([0.16, 0.50, 0.84])
    err_2d_mom = max(
        abs(float(g2d["log_z"]) - log_z_2d_want),
        float(np.max(np.abs(np.asarray(g2d["mean"]) - mu_2d))),
        float(np.max(np.abs(np.asarray(g2d["cov"]) - cov_2d))),
    )
    err_2d_quant = max(
        float(np.max(np.abs(np.asarray(g2d["quantiles_x"]) - (1.2 + 0.6 * z_q)))),
        float(np.max(np.abs(np.asarray(g2d["quantiles_y"]) - (-0.7 + 0.5 * z_q)))),
    )
    assert err_2d_mom < 1e-9
    assert err_2d_quant < 5e-4  # trapezoid + linear CDF interpolation at dx = 0.032

    record("max_outlier_quad_vs_subset_err", max_sub_err)
    record("luger_quad_err", max(err_luger, err_lu_ref))
    record("ridge_opt_err", max(err_ridge, err_gn_ref, err_2d_mom))


# ---------------------------------------------------------------------------
# 1. virgil.likelihood.{is_tied, tied_log_prior}
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.likelihood.is_tied",
    "virgil.likelihood.tied_log_prior",
    roots=["mathematics"],
    headline="max_tied_lp_err",
)
def test_is_tied_and_tied_log_prior_deduplicate_shared_population_terms():
    """is_tied distinguishes callable tied scale objects from priors and
    constants, and tied_log_prior sums distinct PopulationScale log_priors
    while counting a shared tied term only once."""
    _, scales_c = vpr.hierarchical_scales("err_s", 3, centred=True)
    _, scales_nc = vpr.hierarchical_scales("err_s", 3, centred=False)

    assert vlike.is_tied(scales_c[0]) is True
    assert vlike.is_tied(scales_nc[0]) is True
    assert vlike.is_tied(dist.LogUniform(0.5, 2.0)) is False
    assert vlike.is_tied(dist.Normal(1.0, 0.1)) is False
    assert vlike.is_tied(1.25) is False
    assert vlike.is_tied(None) is False

    vals_c = {
        "err_s_median": jnp.array(1.8),
        "err_s_spread": jnp.array(0.35),
        "err_s_log": jnp.array([-0.25, 0.45, 0.90]),
    }
    vals_nc = {
        "err_s_median": jnp.array(1.8),
        "err_s_spread": jnp.array(0.35),
        "err_s_z": jnp.array([-1.2, 0.3, 1.1]),
    }

    # Same tied scale shared across vis_scale and phi_scale of epoch 0 counts once
    sites_shared = vlike.noise_sites(
        {"vis_scale": scales_c[0], "phi_scale": scales_c[0]}, 1
    )
    got_shared = float(vlike.tied_log_prior(sites_shared, vals_c))
    want_0 = float(scales_c[0].log_prior(vals_c))
    err_shared = abs(got_shared - want_0)

    # Multi-epoch noise mixing shared tied scale on epoch 0, distinct tied scale on epoch 1, and untied prior on epoch 2
    sites_multi = vlike.noise_sites(
        [
            {"vis_scale": scales_c[0], "phi_scale": scales_c[0]},
            {"vis_scale": scales_c[1], "phi_scale": dist.LogUniform(0.5, 2.0)},
            {"vis_scale": dist.LogNormal(0.0, 0.1)},
        ],
        3,
    )
    got_multi = float(vlike.tied_log_prior(sites_multi, vals_c))
    want_01 = want_0 + float(scales_c[1].log_prior(vals_c))
    err_multi = abs(got_multi - want_01)

    # Non-centred scales carry zero population log_prior
    sites_nc = vlike.noise_sites([{"vis_scale": scales_nc[0]}, {"vis_scale": scales_nc[1]}], 2)
    err_nc = abs(float(vlike.tied_log_prior(sites_nc, vals_nc)))

    max_err = max(err_shared, err_multi, err_nc)
    assert max_err < 1e-14
    record("max_tied_lp_err", max_err)


@pytest.mark.validates(
    "virgil.likelihood.tied_log_prior",
    roots=["mathematics"],
    kind="control",
    headline="double_count_error",
)
def test_tied_log_prior_rejects_undeduplicated_double_counting():
    """Control: summing log_prior across sites without deduplicating shared
    tied terms double-counts scales_c[0].log_prior."""
    _, scales_c = vpr.hierarchical_scales("err_s", 2, centred=True)
    vals_c = {
        "err_s_median": jnp.array(2.1),
        "err_s_spread": jnp.array(0.25),
        "err_s_log": jnp.array([0.1, 0.8]),
    }
    sites = vlike.noise_sites({"vis_scale": scales_c[0], "phi_scale": scales_c[0]}, 1)
    got = float(vlike.tied_log_prior(sites, vals_c))
    naive_sum = sum(float(spec.log_prior(vals_c)) for spec, _, _ in sites.values())
    gap = abs(got - naive_sum)
    assert gap > 0.5
    record("double_count_error", gap)


# ---------------------------------------------------------------------------
# 2. virgil.likelihood.{group_outlier_mixture, group_outlier_probabilities} (WF4)
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.likelihood.group_outlier_mixture",
    "virgil.likelihood.group_outlier_probabilities",
    roots=["mathematics"],
    headline="max_jeffreys_mixture_err",
)
def test_group_outlier_mixture_and_probabilities_exact_jeffreys_quadrature():
    """group_outlier_mixture and group_outlier_probabilities match continuous
    Beta(1/2, 1/2) quadrature and exact 2^K combinatorial Beta-function sums to
    < 1e-12, and satisfy exact log-partition derivative and shift identities."""
    rng = np.random.default_rng(2048)
    max_lz_err = 0.0
    max_q_err = 0.0
    max_grad_err = 0.0

    for k_groups in (1, 4, 9, 14):
        ll_np = rng.uniform(-45.0, -5.0, size=k_groups)
        ol_np = ll_np + rng.choice([-12.0, -2.5, 0.4, 8.5, 25.0], size=k_groups)
        ll_j = jnp.asarray(ll_np, dtype=jnp.float64)
        ol_j = jnp.asarray(ol_np, dtype=jnp.float64)

        got_lz = float(vlike.group_outlier_mixture(ll_j, ol_j))
        got_q = np.asarray(vlike.group_outlier_probabilities(ll_j, ol_j))

        # Verify exact underflow-safe shift invariance across -900 nats per group
        got_lz_shifted = float(vlike.group_outlier_mixture(ll_j - 900.0, ol_j - 900.0))
        assert abs((got_lz_shifted + 900.0 * k_groups) - got_lz) < 5e-12
        np.testing.assert_allclose(
            np.asarray(vlike.group_outlier_probabilities(ll_j - 900.0, ol_j - 900.0)),
            got_q,
            atol=1e-14,
        )

        want_lz_q, want_q_q = ref.beta_outlier_mixture_quad_ref(
            ll_np, ol_np, alpha=0.5, beta_param=0.5
        )
        want_lz_s, want_q_s = ref.jeffreys_outlier_mixture_exact_subset_ref(ll_np, ol_np)

        max_lz_err = max(max_lz_err, abs(got_lz - want_lz_q), abs(got_lz - want_lz_s))
        max_q_err = max(
            max_q_err,
            float(np.max(np.abs(got_q - want_q_q))),
            float(np.max(np.abs(got_q - want_q_s))),
        )

        # Exact thermodynamic identity: d(ln Z)/d(ln B_k) = q_k and d(ln Z)/d(ln L_k) = 1 - q_k
        grad_ll, grad_ol = jax.grad(vlike.group_outlier_mixture, argnums=(0, 1))(ll_j, ol_j)
        max_grad_err = max(
            max_grad_err,
            float(np.max(np.abs(np.asarray(grad_ol) - got_q))),
            float(np.max(np.abs(np.asarray(grad_ll) - (1.0 - got_q)))),
        )

    # Scalar outlier_loglikes broadcast + equal-likelihood Beta(1/2, 1/2) symmetry (q_k == 0.5)
    eq_ll = jnp.array([-12.3, -12.3, -12.3, -12.3], dtype=jnp.float64)
    assert float(vlike.group_outlier_mixture(eq_ll, -12.3)) == pytest.approx(
        float(jnp.sum(eq_ll)), abs=1e-13
    )
    np.testing.assert_allclose(
        np.asarray(vlike.group_outlier_probabilities(eq_ll, -12.3)), 0.5, atol=1e-14
    )

    assert max_lz_err < 1e-12
    assert max_q_err < 1e-12
    assert max_grad_err < 1e-13

    max_total = max(max_lz_err, max_q_err, max_grad_err)
    record("max_jeffreys_mixture_err", max_total)
    record("max_prob_err", max_q_err)
    record("max_thermodynamic_grad_err", max_grad_err)


@pytest.mark.validates(
    "virgil.likelihood.group_outlier_mixture",
    "virgil.likelihood.group_outlier_probabilities",
    roots=["mathematics"],
    kind="control",
    headline="beta_1_9_log_z_gap",
)
def test_group_outlier_mixture_rejects_asymmetric_beta_1_9_prior():
    """Control (WF4): group_outlier_mixture integrates the symmetric Jeffreys
    Beta(1/2, 1/2) prior, cleanly rejecting an informative Beta(1, 9) prior."""
    ll = np.array([-10.0, -35.0, -38.0, -40.0, -36.0, -42.0])
    ol = np.full(6, -12.0)
    got_lz = float(vlike.group_outlier_mixture(jnp.asarray(ll), jnp.asarray(ol)))
    got_q = np.asarray(vlike.group_outlier_probabilities(jnp.asarray(ll), jnp.asarray(ol)))

    wrong_lz, wrong_q = ref.beta_outlier_mixture_quad_ref(ll, ol, alpha=1.0, beta_param=9.0)
    lz_gap = abs(got_lz - wrong_lz)
    eq_q = np.asarray(vlike.group_outlier_probabilities(jnp.zeros(4), jnp.zeros(4)))
    assert lz_gap > 3.0
    assert float(np.max(np.abs(got_q - wrong_q))) > 0.05
    assert abs(float(eq_q[0]) - 0.1) > 0.39  # Beta(1/2,1/2) gives 0.5, Beta(1,9) gives 0.1
    record("beta_1_9_log_z_gap", lz_gap)


# ---------------------------------------------------------------------------
# 3. virgil.grid_fit.{Gaussian, LogUniform, LinearFluxGrid, optimized_likelihood_grid}
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.grid_fit.Gaussian",
    "virgil.grid_fit.LogUniform",
    "virgil.grid_fit.LinearFluxGrid",
    roots=["mathematics"],
    headline="max_prior_bf_err",
)
def test_grid_fit_gaussian_loguniform_and_linear_flux_grid(tmp_path):
    """Gaussian, LogUniform, and LinearFluxGrid match closed-form Luger et al.
    (2017) and adaptive SciPy log-uniform quadrature posteriors and Bayes
    factors, including the exact Occam bound-widening shift."""
    fits_path = tmp_path / "binary_grid.fits"
    truth_cloud = _binary_cloud(0.015, 8.0, -5.0)
    xsim.observe(
        fits_path,
        lambda u, v, wl: xsky.visibility(truth_cloud, u, v, wl),
        UTS3,
        hour_angles_h=[-1.5, 0.0, 1.5],
        wavelengths=np.array([2.1e-6, 2.3e-6]),
        sigma_v2=0.004,
        sigma_cp_deg=0.15,
    )
    oid = OIData(read_oifits(str(fits_path)))
    grid = {
        "dra": np.array([8.0, -8.0, 3.0]),
        "ddec": np.array([-5.0, 5.0]),
        "flux": np.array([0.01]),
    }

    res_plain = vgf.linear_flux_grid(BinaryModelCartesian, oid, grid)
    assert isinstance(res_plain, vgf.LinearFluxGrid)
    assert res_plain.posterior_mean is None
    assert res_plain.posterior_sd is None
    assert res_plain.log_bayes_factor is None
    np.testing.assert_allclose(
        np.asarray(res_plain.snr),
        np.asarray(res_plain.flux) / np.asarray(res_plain.flux_error),
        rtol=1e-13,
    )

    g_prior = vgf.Gaussian(0.002, 0.04)
    res_g = vgf.linear_flux_grid(BinaryModelCartesian, oid, grid, prior=g_prior)
    want_g = ref.linear_flux_gaussian_prior_ref(
        np.asarray(res_g.flux), np.asarray(res_g.flux_error), g_prior.mean, g_prior.sd
    )
    err_g_mean = float(np.max(np.abs(np.asarray(res_g.posterior_mean) - want_g["posterior_mean"])))
    err_g_sd = float(np.max(np.abs(np.asarray(res_g.posterior_sd) - want_g["posterior_sd"])))
    err_g_bf = float(
        np.max(np.abs(np.asarray(res_g.log_bayes_factor) - want_g["log_bayes_factor"]))
    )
    assert err_g_mean < 1e-13 and err_g_sd < 1e-13 and err_g_bf < 1e-11

    lu_prior = vgf.LogUniform(1e-4, 0.2)
    res_lu = vgf.linear_flux_grid(BinaryModelCartesian, oid, grid, prior=lu_prior)
    err_lu_bf = 0.0
    err_lu_mean = 0.0
    err_lu_sd = 0.0
    for i in range(grid["dra"].size):
        for j in range(grid["ddec"].size):
            want_lu = ref.linear_flux_loguniform_prior_ref(
                float(res_lu.flux[i, j]),
                float(res_lu.flux_error[i, j]),
                lu_prior.f_min,
                lu_prior.f_max,
            )
            err_lu_bf = max(err_lu_bf, abs(float(res_lu.log_bayes_factor[i, j]) - want_lu["log_bayes_factor"]))
            err_lu_mean = max(err_lu_mean, abs(float(res_lu.posterior_mean[i, j]) - want_lu["posterior_mean"]))
            err_lu_sd = max(err_lu_sd, abs(float(res_lu.posterior_sd[i, j]) - want_lu["posterior_sd"]))
    assert err_lu_bf < 1e-7
    assert err_lu_mean < 1e-7
    assert err_lu_sd < 1e-7

    # Direct NumPyro dist.Normal and dist.LogUniform coercion matches Gaussian and LogUniform wrappers
    res_dist_g = vgf.linear_flux_grid(
        BinaryModelCartesian, oid, grid, prior=dist.Normal(g_prior.mean, g_prior.sd)
    )
    res_dist_lu = vgf.linear_flux_grid(
        BinaryModelCartesian, oid, grid, prior=dist.LogUniform(lu_prior.f_min, lu_prior.f_max)
    )
    np.testing.assert_allclose(res_dist_g.log_bayes_factor, res_g.log_bayes_factor, atol=1e-14)
    np.testing.assert_allclose(res_dist_g.posterior_sd, res_g.posterior_sd, atol=1e-14)
    np.testing.assert_allclose(res_dist_lu.log_bayes_factor, res_lu.log_bayes_factor, atol=1e-14)
    np.testing.assert_allclose(res_dist_lu.posterior_sd, res_lu.posterior_sd, atol=1e-14)

    # Exact Occam factor at the high-SNR peak (0, 0) when widening LogUniform bounds across empty tails
    lu_wide = vgf.LogUniform(1e-5, 0.8)
    res_wide = vgf.linear_flux_grid(BinaryModelCartesian, oid, grid, prior=lu_wide)
    want_occam = -math.log(math.log(0.8 / 1e-5) / math.log(0.2 / 1e-4))
    got_occam = float(res_wide.log_bayes_factor[0, 0] - res_lu.log_bayes_factor[0, 0])
    assert abs(got_occam - want_occam) < 1e-9

    # Boundary-clamped high-SNR LogUniform cells: bright companion above f_max ((f_hat - f_max)/sf > 50)
    # and anti-symmetric closure-phase mirror trough below 0 (f_hat / sf < -50)
    clamp_fits = tmp_path / "binary_grid_clamp_trough.fits"
    bright_cloud = _binary_cloud(0.10, 8.0, -5.0)
    xsim.observe(
        clamp_fits,
        lambda u, v, wl: xsky.visibility(bright_cloud, u, v, wl),
        UTS3,
        hour_angles_h=[-1.5, -0.5, 0.5, 1.5],
        wavelengths=np.array([2.0e-6, 2.2e-6]),
        sigma_v2=0.06,
        sigma_cp_deg=0.04,
    )
    oid_clamp = OIData(read_oifits(str(clamp_fits)))
    lu_clamp = vgf.LogUniform(1e-4, 0.04)
    grid_clamp = {"dra": np.array([8.0, -8.0]), "ddec": np.array([-5.0, 5.0]), "flux": np.array([0.01])}
    res_clamp = vgf.linear_flux_grid(BinaryModelCartesian, oid_clamp, grid_clamp, prior=lu_clamp)
    assert float((res_clamp.flux[0, 0] - lu_clamp.f_max) / res_clamp.flux_error[0, 0]) > 50.0
    assert float(res_clamp.flux[1, 1] / res_clamp.flux_error[1, 1]) < -50.0
    for idx_c in [(0, 0), (1, 1)]:
        want_c = ref.linear_flux_loguniform_prior_ref(
            float(res_clamp.flux[idx_c]),
            float(res_clamp.flux_error[idx_c]),
            lu_clamp.f_min,
            lu_clamp.f_max,
        )
        err_lu_bf = max(err_lu_bf, abs(float(res_clamp.log_bayes_factor[idx_c]) - want_c["log_bayes_factor"]))
        err_lu_mean = max(err_lu_mean, abs(float(res_clamp.posterior_mean[idx_c]) - want_c["posterior_mean"]))
        err_lu_sd = max(err_lu_sd, abs(float(res_clamp.posterior_sd[idx_c]) - want_c["posterior_sd"]))
    assert err_lu_bf < 1e-7
    assert err_lu_mean < 1e-7
    assert err_lu_sd < 1e-7

    max_bf_err = max(err_g_bf, err_lu_bf, err_lu_sd)
    record("max_prior_bf_err", max_bf_err)
    record("occam_shift_err", abs(got_occam - want_occam))


@pytest.mark.slow
@pytest.mark.validates(
    "virgil.grid_fit.optimized_likelihood_grid",
    roots=["mathematics"],
    headline="max_opt_ll_err",
    tier="B",
)
def test_optimized_likelihood_grid_matches_independent_brent_profile(tmp_path):
    """optimized_likelihood_grid matches independent 1D Brent optimization of
    -1/2 crosscheck.chi2 over companion flux at every sky grid point (V^2-only
    file so likelihood differences are exact -1/2 delta chi^2)."""
    fits_path = tmp_path / "binary_v2_only.fits"
    truth_cloud = _binary_cloud(0.035, 7.5, -4.5)
    xsim.observe(
        fits_path,
        lambda u, v, wl: xsky.visibility(truth_cloud, u, v, wl),
        UTS3,
        hour_angles_h=[-1.5, -0.5, 0.5, 1.5],
        wavelengths=np.array([2.0e-6, 2.2e-6]),
        sigma_v2=0.003,
        closure_phases=False,
    )
    oid = OIData(read_oifits(str(fits_path)))
    loaded = xchi2.load(fits_path)

    dra_ax = np.array([7.5, 4.0, -7.5])
    ddec_ax = np.array([-4.5, 2.0])
    grid = {"dra": dra_ax, "ddec": ddec_ax, "flux": np.array([0.005, 0.02, 0.06])}

    got_ll = np.asarray(vgf.optimized_likelihood_grid(BinaryModelCartesian, oid, grid))
    assert got_ll.shape == (3, 2)

    null_ll = float(
        vgf.likelihood_grid(
            BinaryModelCartesian,
            data=oid,
            grid={"dra": np.array([0.0]), "ddec": np.array([0.0]), "flux": np.array([0.0])},
        )[0, 0, 0]
    )
    null_c2 = float(
        xchi2.chi2(loaded, lambda u, v, wl: xsky.visibility(_binary_cloud(0.0, 0.0, 0.0), u, v, wl))
    )
    ll_const = null_ll + 0.5 * null_c2

    want_ll = np.empty_like(got_ll)
    for i, dra in enumerate(dra_ax):
        for j, ddec in enumerate(ddec_ax):
            def half_chi2(f_val: float, dx: float = float(dra), dy: float = float(ddec)) -> float:
                cloud = _binary_cloud(f_val, dx, dy)
                c2 = xchi2.chi2(
                    loaded, lambda u, v, wl, c=cloud: xsky.visibility(c, u, v, wl)
                )
                return 0.5 * float(c2)

            best_hc2 = min(
                optimize.minimize_scalar(half_chi2, bounds=(0.0, 0.2), method="bounded").fun,
                optimize.minimize_scalar(half_chi2, bounds=(-0.15, 0.05), method="bounded").fun,
            )
            want_ll[i, j] = ll_const - float(best_hc2)

    max_err = float(np.max(np.abs(got_ll - want_ll)))
    assert max_err < 1e-6

    # Must strictly improve on unoptimized grid at fixed flux = 0.005
    fixed_ll = np.asarray(
        vgf.likelihood_grid(
            BinaryModelCartesian,
            data=oid,
            grid={"dra": dra_ax, "ddec": ddec_ax, "flux": np.array([0.005])},
        )
    )[..., 0]
    assert np.all(got_ll >= fixed_ll - 1e-9)
    assert float(got_ll[0, 0] - fixed_ll[0, 0]) > 25.0

    # 4-UT V^2 + correlated closure-phase dataset on noisy companion-free data,
    # exercising chord closure-phase residuals and unconstrained negative-flux noise troughs
    uts4 = np.array(
        [[-9.925, -20.335], [14.887, 30.502], [44.915, 66.183], [103.306, 43.999]]
    )
    fits_cp = tmp_path / "null_v2_cp_4ut.fits"
    xsim.observe(
        fits_cp,
        lambda u, v, wl: xsky.visibility(_binary_cloud(0.0, 0.0, 0.0), u, v, wl),
        uts4,
        hour_angles_h=[-1.0, 0.0, 1.0],
        wavelengths=np.array([2.0e-6, 2.2e-6]),
        sigma_v2=0.015,
        sigma_cp_deg=0.8,
        rng=np.random.default_rng(42),
    )
    oid_cp = OIData(read_oifits(str(fits_cp)))
    loaded_cp = xchi2.load(fits_cp)

    # Negative seed values in grid["flux"] are rejected by check_flux_axes
    with pytest.raises(ValueError):
        vgf.optimized_likelihood_grid(
            BinaryModelCartesian,
            oid_cp,
            {"dra": np.array([5.0]), "ddec": np.array([5.0]), "flux": np.array([-0.01, 0.01])},
        )

    dra_cp = np.array([-6.0, 5.5, 11.0])
    ddec_cp = np.array([-7.0, 6.5])
    grid_cp = {"dra": dra_cp, "ddec": ddec_cp, "flux": np.array([0.002, 0.02])}
    got_cp_ll = np.asarray(vgf.optimized_likelihood_grid(BinaryModelCartesian, oid_cp, grid_cp))
    null_cp_ll = float(
        vgf.likelihood_grid(
            BinaryModelCartesian,
            data=oid_cp,
            grid={"dra": np.array([0.0]), "ddec": np.array([0.0]), "flux": np.array([0.0])},
        )[0, 0, 0]
    )
    null_cp_c2 = float(
        xchi2.chi2(
            loaded_cp,
            lambda u, v, wl: xsky.visibility(_binary_cloud(0.0, 0.0, 0.0), u, v, wl),
            correlated=True,
            sine=True,
        )
    )
    cp_const = null_cp_ll + 0.5 * null_cp_c2

    want_cp_ll = np.empty_like(got_cp_ll)
    best_fluxes = []
    for i, dra in enumerate(dra_cp):
        for j, ddec in enumerate(ddec_cp):
            def half_cp_chi2(f_val: float, dx: float = float(dra), dy: float = float(ddec)) -> float:
                c = _binary_cloud(f_val, dx, dy)
                return 0.5 * float(
                    xchi2.chi2(
                        loaded_cp,
                        lambda u, v, wl, cl=c: xsky.visibility(cl, u, v, wl),
                        correlated=True,
                        sine=True,
                    )
                )

            r_pos = optimize.minimize_scalar(half_cp_chi2, bounds=(0.0, 0.25), method="bounded")
            r_neg = optimize.minimize_scalar(half_cp_chi2, bounds=(-0.25, 0.0), method="bounded")
            best_res = r_neg if r_neg.fun < r_pos.fun else r_pos
            best_fluxes.append(float(best_res.x))
            want_cp_ll[i, j] = cp_const - float(best_res.fun)

    assert min(best_fluxes) < -3e-4 and max(best_fluxes) > 3e-4  # both negative-flux troughs and positive peaks tested
    max_cp_err = float(np.max(np.abs(got_cp_ll - want_cp_ll)))
    assert max_cp_err < 1e-5

    record("max_opt_ll_err", max(max_err, max_cp_err))


@pytest.mark.validates(
    "virgil.grid_fit.Gaussian",
    "virgil.grid_fit.LogUniform",
    "virgil.grid_fit.optimized_likelihood_grid",
    roots=["mathematics"],
    kind="control",
    headline="unoptimized_ll_shortfall",
)
def test_grid_fit_priors_and_unoptimized_flux_controls(tmp_path):
    """Control: holding flux fixed instead of optimizing misses the profile peak
    by > 25 nats, and omitting the Gaussian prior shrinkage width misses the
    Bayes factor Occam penalty."""
    fits_path = tmp_path / "binary_ctrl.fits"
    truth_cloud = _binary_cloud(0.035, 7.5, -4.5)
    xsim.observe(
        fits_path,
        lambda u, v, wl: xsky.visibility(truth_cloud, u, v, wl),
        UTS3,
        hour_angles_h=[-1.0, 1.0],
        wavelengths=np.array([2.1e-6]),
        sigma_v2=0.003,
        closure_phases=False,
    )
    oid = OIData(read_oifits(str(fits_path)))
    grid = {"dra": np.array([7.5]), "ddec": np.array([-4.5]), "flux": np.array([0.002])}
    opt_ll = float(vgf.optimized_likelihood_grid(BinaryModelCartesian, oid, grid)[0, 0])
    fixed_ll = float(vgf.likelihood_grid(BinaryModelCartesian, data=oid, grid=grid)[0, 0, 0])
    shortfall = opt_ll - fixed_ll
    assert shortfall > 25.0

    res_narrow = vgf.linear_flux_grid(BinaryModelCartesian, oid, grid, prior=vgf.Gaussian(0.0, 0.04))
    res_wide = vgf.linear_flux_grid(BinaryModelCartesian, oid, grid, prior=vgf.Gaussian(0.0, 10.0))
    assert abs(float(res_narrow.log_bayes_factor[0, 0] - res_wide.log_bayes_factor[0, 0])) > 4.0
    record("unoptimized_ll_shortfall", shortfall)


# ---------------------------------------------------------------------------
# 4. virgil.crossval.{fold_masks, save_record, load_records, aggregate,
#    aggregate_files, cv_task, cross_validate}
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.crossval.fold_masks",
    "virgil.crossval.save_record",
    "virgil.crossval.load_records",
    "virgil.crossval.aggregate",
    "virgil.crossval.aggregate_files",
    roots=["mathematics"],
    headline="max_agg_err",
)
def test_crossval_fold_masks_io_and_breiman_one_se_aggregation(tmp_path):
    """fold_masks forms an exact partition, save_record / load_records round-trip
    single and list JSON files, and aggregate / aggregate_files reproduce sample
    SD (ddof=1), SE, best, Breiman 1-SE selection, edge flags, and group curves."""
    fold = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    seen_test = np.zeros_like(fold, dtype=int)
    for k in range(4):
        tr, te = vcv.fold_masks(fold, k)
        tr_np, te_np = np.asarray(tr), np.asarray(te)
        assert not np.any(tr_np & te_np)
        assert np.all(tr_np | te_np)
        seen_test += te_np.astype(int)
    assert np.all(seen_test == 1)
    with pytest.raises(ValueError):
        vcv.fold_masks(fold, 99)

    # Synthetic 4-setting x 4-fold table where setting 0.1 is best but 1.0 is within 1 SE
    settings = [0.01, 0.1, 1.0, 10.0]
    fold_matrix = np.array(
        [
            [5.8, 6.2, 6.0, 6.4],   # mean 6.10
            [3.9, 4.3, 4.1, 4.5],   # mean 4.20, sd=0.2582, se=0.1291 -> thresh 4.3291
            [4.0, 4.4, 4.2, 4.6],   # mean 4.30 <= 4.3291 -> one_se picks index 2 (lam=1.0)!
            [7.1, 7.5, 7.3, 7.7],   # mean 7.40
        ]
    )
    recs = []
    for i, lam in enumerate(settings):
        for k in range(4):
            sc = float(fold_matrix[i, k])
            recs.append(
                {
                    "setting": lam,
                    "fold": k,
                    "score": sc,
                    "chi2": 2.0 * sc,
                    "n_test": 3,
                    "n_train": 9,
                    "group_labels": [10, 20],
                    "group_score": [0.3 * sc, 0.7 * sc],
                    "group_chi2": [0.6 * sc, 1.4 * sc],
                    "group_n": [1, 2],
                }
            )

    p_single = tmp_path / "rec0.json"
    p_rest = tmp_path / "rec_rest.json"
    vcv.save_record(recs[0], p_single)
    vcv.save_record(recs[1:], p_rest)
    loaded = vcv.load_records([p_single, p_rest])
    assert len(loaded) == 16

    agg = vcv.aggregate(loaded, larger_is_simpler=True)
    agg_files = vcv.aggregate_files([p_single, p_rest], larger_is_simpler=True)
    want = ref.aggregate_cv_ref(recs, larger_is_simpler=True)

    err_mean = float(np.max(np.abs(np.asarray(agg["mean"]) - want["mean"])))
    err_se = float(np.max(np.abs(np.asarray(agg["se"]) - want["se"])))
    err_chi2 = float(np.max(np.abs(np.asarray(agg["chi2"]) - want["chi2"])))
    assert err_mean < 1e-14 and err_se < 1e-14 and err_chi2 < 1e-14
    assert agg["best"] == want["best"] == 1
    assert agg["one_se"] == want["one_se"] == 2
    assert agg["edge"] is False
    assert agg_files["one_se"] == 2
    assert list(agg["groups"]) == [10, 20]
    assert list(agg["group_best"]) == [1, 1]

    # Flipping larger_is_simpler=False picks the smaller-index setting within 1 SE (index 1)
    agg_small = vcv.aggregate(loaded, larger_is_simpler=False)
    assert agg_small["one_se"] == 1

    # Grid-boundary edge=True checks (lower boundary index 0 and upper boundary index 3)
    recs_lo = [
        {**r, "score": (1.0 + 0.1 * r["fold"]) if r["setting"] == settings[0] else r["score"]}
        for r in recs
    ]
    agg_lo = vcv.aggregate(recs_lo, larger_is_simpler=True)
    assert agg_lo["best"] == 0 and agg_lo["edge"] is True

    recs_hi = [
        {**r, "score": (4.05 + 0.1 * r["fold"]) if r["setting"] == settings[-1] else r["score"]}
        for r in recs
    ]
    agg_hi = vcv.aggregate(recs_hi, larger_is_simpler=True)
    assert agg_hi["one_se"] == len(settings) - 1 and agg_hi["edge"] is True

    # Non-numeric string setting labels and chi2=None (unit_scale=None)
    str_labels = ["tv", "tsv", "mem"]
    recs_str = [
        {"setting": name, "fold": k, "score": float((idx + 1) * 2.0 + 0.1 * k), "chi2": None, "n_test": 3, "n_train": 9}
        for idx, name in enumerate(str_labels)
        for k in range(4)
    ]
    agg_str = vcv.aggregate(recs_str)
    assert list(agg_str["settings"]) == str_labels
    assert agg_str["best"] == 0
    assert agg_str["one_se"] is None and agg_str["edge"] is None
    assert agg_str["chi2"] is None

    # Diverged NaN fold emits UserWarning and ranks last, never winning argmin
    recs_nan = [
        {**r, "score": float("nan")}
        if (r["setting"] == settings[1] and r["fold"] == 0)
        else dict(r)
        for r in recs
    ]
    with pytest.warns(UserWarning, match="NaN scores"):
        agg_nan = vcv.aggregate(recs_nan, larger_is_simpler=True)
    assert agg_nan["best"] == 2  # setting[1] was best=1 before NaN, now excluded from argmin

    # Duplicate or missing records raise ValueError
    with pytest.raises(ValueError):
        vcv.aggregate(recs[:-1])
    with pytest.raises(ValueError):
        vcv.aggregate(recs + [recs[0]])

    record("max_agg_err", max(err_mean, err_se, err_chi2))


@pytest.mark.slow
@pytest.mark.validates(
    "virgil.crossval.cv_task",
    "virgil.crossval.cross_validate",
    roots=["mathematics"],
    headline="max_ridge_cv_score_err",
    tier="B",
)
def test_cv_task_and_cross_validate_match_closed_form_ridge_regression():
    """cv_task and cross_validate reproduce the exact closed-form K-fold ridge
    regression held-out negative log-likelihood, raw chi^2, per-group breakdown,
    and optimal hyperparameter selection to < 1e-5."""
    rng = np.random.default_rng(31415)
    n_pts, p_dim, n_folds = 24, 3, 4
    design = rng.normal(size=(n_pts, p_dim))
    sigma = rng.uniform(0.08, 0.22, size=n_pts)
    theta_true = np.array([1.1, -0.75, 0.45])
    y = design @ theta_true + rng.normal(0.0, sigma)
    fold = np.tile(np.arange(n_folds), n_pts // n_folds)
    groups = np.where(np.arange(n_pts) < n_pts // 2, 0, 1)
    settings = [1e-3, 0.05, 0.5, 2.0, 20.0]

    x_j = jnp.asarray(design, dtype=jnp.float64)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    sig_j = jnp.asarray(sigma, dtype=jnp.float64)

    class RidgeTerm:
        has_log_norm = True

        def __init__(self, mask):
            self.mask = jnp.asarray(mask, dtype=bool)

        def __call__(self, values):
            res = (y_j - x_j @ values["theta"]) / sig_j
            return jnp.where(self.mask, res, 0.0)

        def log_norm(self, values):
            return jnp.sum(jnp.where(self.mask, jnp.log(sig_j), 0.0))

    def fit_kwargs(lam):
        sd = 1.0 / jnp.sqrt(jnp.asarray(lam, dtype=jnp.float64))
        return {
            "model": lambda **kw: kw,
            "priors": {"theta": dist.Normal(0.0, sd).expand([p_dim]).to_event(1)},
            "data": (),
        }

    # Single task check
    rec0 = vcv.cv_task(
        settings[2],
        0,
        fold,
        fit_kwargs=fit_kwargs,
        make_term=RidgeTerm,
        init={"theta": jnp.zeros(p_dim)},
        groups=groups,
        unit_scale=lambda v: v,
    )
    assert rec0["converged"] is True

    # Full cross_validate sweep
    summary = vcv.cross_validate(
        settings,
        n_folds,
        fold,
        fit_kwargs=fit_kwargs,
        make_term=RidgeTerm,
        init={"theta": jnp.zeros(p_dim)},
        groups=groups,
        unit_scale=lambda v: v,
    )
    want_recs = ref.ridge_kfold_cv_ref(
        design, y, sigma, fold, settings, groups=groups, include_log_norm=True
    )
    want_agg = ref.aggregate_cv_ref(want_recs, larger_is_simpler=True)

    score_err = float(np.max(np.abs(np.asarray(summary["score"]) - want_agg["score"])))
    chi2_err = float(np.max(np.abs(np.asarray(summary["chi2"]) - want_agg["chi2"])))
    se_err = float(np.max(np.abs(np.asarray(summary["se"]) - want_agg["se"])))
    assert score_err < 1e-5
    assert chi2_err < 1e-5
    assert se_err < 1e-5
    assert summary["best"] == want_agg["best"]
    assert summary["one_se"] == want_agg["one_se"]

    # Multi-component residuals per mask entry (m = 2 complex visibilities) + unit_scale=None
    y_im = -0.5 * (design @ theta_true) + rng.normal(0.0, sigma)
    y_im_j = jnp.asarray(y_im, dtype=jnp.float64)

    class ComplexRidgeTerm:
        has_log_norm = False

        def __init__(self, mask):
            self.mask = jnp.asarray(mask, dtype=bool)

        def __call__(self, values):
            r_re = jnp.where(self.mask, (y_j - x_j @ values["theta"]) / sig_j, 0.0)
            r_im = jnp.where(self.mask, (y_im_j + 0.5 * (x_j @ values["theta"])) / sig_j, 0.0)
            return jnp.concatenate([r_re, r_im])

    rec_m2 = vcv.cv_task(
        settings[2],
        0,
        fold,
        fit_kwargs=fit_kwargs,
        make_term=ComplexRidgeTerm,
        init={"theta": jnp.zeros(p_dim)},
        groups=groups,
        unit_scale=None,
    )
    tr = fold != 0
    te = fold == 0
    x_tr_c = np.vstack([design[tr] / sigma[tr, None], -0.5 * design[tr] / sigma[tr, None]])
    y_tr_c = np.concatenate([y[tr] / sigma[tr], y_im[tr] / sigma[tr]])
    th_m2 = linalg.solve(x_tr_c.T @ x_tr_c + settings[2] * np.eye(p_dim), x_tr_c.T @ y_tr_c)
    r_te_re = (y[te] - design[te] @ th_m2) / sigma[te]
    r_te_im = (y_im[te] + 0.5 * (design[te] @ th_m2)) / sigma[te]
    want_m2_score = 0.5 * float(np.sum(r_te_re**2 + r_te_im**2))
    assert rec_m2["n_test"] == int(np.sum(te))
    assert rec_m2["n_train"] == int(np.sum(tr))
    assert rec_m2["chi2"] is None
    assert abs(float(rec_m2["score"]) - want_m2_score) < 1e-6

    record("max_ridge_cv_score_err", score_err)
    record("max_ridge_cv_chi2_err", chi2_err)


@pytest.mark.validates(
    "virgil.crossval.cv_task",
    "virgil.crossval.cross_validate",
    roots=["mathematics", "statistics"],
    kind="control",
    headline="leakage_score_drop",
)
def test_crossval_in_sample_leakage_overfits_zero_regularization_control():
    """Control: if make_term ignores the training mask (leaking held-out points
    into training), the apparent CV score monotonically favours lam -> 0 even on
    pure noise where genuine held-out CV prefers heavy shrinkage."""
    rng = np.random.default_rng(2718)
    n_pts, p_dim, n_folds = 16, 8, 4
    design = rng.normal(size=(n_pts, p_dim))
    sigma = np.full(n_pts, 0.5)
    # Pure noise target (true theta == 0): optimal ridge shrinkage is large lam!
    y = rng.normal(0.0, sigma)
    fold = np.tile(np.arange(n_folds), n_pts // n_folds)
    settings = [1e-4, 0.1, 10.0, 1e3]

    x_j, y_j, sig_j = jnp.asarray(design), jnp.asarray(y), jnp.asarray(sigma)

    class HonestTerm:
        def __init__(self, mask):
            self.mask = jnp.asarray(mask, dtype=bool)

        def __call__(self, values):
            return jnp.where(self.mask, (y_j - x_j @ values["theta"]) / sig_j, 0.0)

    class LeakyTerm:
        def __init__(self, mask):
            self.mask = jnp.asarray(mask, dtype=bool)

        def __call__(self, values):
            # Buggy term ignores mask, leaking test fold into the fit!
            return (y_j - x_j @ values["theta"]) / sig_j

    def fit_kwargs(lam):
        sd = 1.0 / jnp.sqrt(jnp.asarray(lam, dtype=jnp.float64))
        return {
            "model": lambda **kw: kw,
            "priors": {"theta": dist.Normal(0.0, sd).expand([p_dim]).to_event(1)},
            "data": (),
        }

    honest = vcv.cross_validate(
        settings, n_folds, fold, fit_kwargs=fit_kwargs, make_term=HonestTerm, init={"theta": jnp.zeros(p_dim)}
    )
    leaky = vcv.cross_validate(
        settings, n_folds, fold, fit_kwargs=fit_kwargs, make_term=LeakyTerm, init={"theta": jnp.zeros(p_dim)}
    )
    assert honest["best"] >= 2  # genuine CV selects strong shrinkage on pure noise
    assert leaky["best"] == 0   # leaked CV selects unregularized lam = 1e-4
    drop = float(honest["score"][0] - honest["score"][-1])
    assert drop > 3.0
    record("leakage_score_drop", drop)


# ---------------------------------------------------------------------------
# 5. virgil.fitting.{FitResult, gauss_newton_mass, GaussNewtonMass} &
#    virgil.likelihood.chain_init_params
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.fitting.FitResult",
    "virgil.fitting.gauss_newton_mass",
    "virgil.fitting.GaussNewtonMass",
    "virgil.likelihood.chain_init_params",
    roots=["mathematics"],
    headline="gn_inv_mass_rel_err",
)
def test_fit_result_gauss_newton_mass_and_chain_init_params(tmp_path):
    """FitResult matches crosscheck.chi2 diagnostics, gauss_newton_mass /
    GaussNewtonMass match central-difference unconstrained J^T J + diag(P)
    (including the 2N Fisher row for vis_scale and correlation condition number),
    and chain_init_params builds single- and multi-chain NUTS initializations."""
    fits_path = tmp_path / "gn_mass_v2.fits"
    truth = _binary_cloud(0.025, 9.0, -6.0)
    xsim.observe(
        fits_path,
        lambda u, v, wl: xsky.visibility(truth, u, v, wl),
        UTS3,
        hour_angles_h=[-1.5, -0.5, 0.5, 1.5],
        wavelengths=np.array([2.0e-6, 2.2e-6, 2.4e-6]),
        sigma_v2=0.0025,
        closure_phases=False,
    )
    oid = OIData(read_oifits(str(fits_path)))
    loaded = xchi2.load(fits_path)
    n_data = int(loaded["v2"].size)

    priors = {
        "dra": dist.Normal(8.5, 5.0),
        "ddec": dist.Normal(-5.5, 5.0),
        "flux": dist.LogNormal(math.log(0.02), 0.5),
    }
    noise = {"vis_scale": dist.LogNormal(0.0, 0.25)}
    tmpl = BinaryModelCartesian(8.5, -5.5, 0.02)

    fres = vfit.fit(tmpl, priors, oid, noise=noise)
    assert isinstance(fres, vfit.FitResult)
    assert fres.info["converged"] is True and fres.info["stop"] is None
    assert fres.info["ndata"] == [n_data]
    assert fres.info["grad_norm"] < 1e-4
    assert fres.info["at_bound"] == ()

    v_opt = fres.values
    s_opt = float(v_opt["noise.vis_scale"])
    fit_cloud = _binary_cloud(float(v_opt["flux"]), float(v_opt["dra"]), float(v_opt["ddec"]))
    our_c2 = float(
        xchi2.chi2(loaded, lambda u, v, wl: xsky.visibility(fit_cloud, u, v, wl))
    ) / (s_opt**2)
    assert float(np.asarray(fres.info["chi2"]).ravel()[0]) == pytest.approx(our_c2, rel=1e-7)

    gnm = vfit.gauss_newton_mass(tmpl, priors, oid, v_opt, noise=noise, flat_coordinates=False)
    assert isinstance(gnm, vfit.GaussNewtonMass)
    assert gnm.flat_coordinates is False
    site_order = next(iter(gnm["inverse_mass_matrix"].keys()))
    got_inv_mass = np.asarray(gnm["inverse_mass_matrix"][site_order])

    # Build unconstrained coordinates z in the exact site_order of GaussNewtonMass
    # Normal -> identity (z = x); LogNormal -> exp bijection (z = ln x, x = exp(z))
    def to_z(name: str, val: float) -> float:
        return math.log(val) if name in ("flux", "noise.vis_scale") else val

    def from_z(name: str, z_val: float) -> float:
        return math.exp(z_val) if name in ("flux", "noise.vis_scale") else z_val

    z_star = np.array([to_z(k, float(v_opt[k])) for k in site_order], dtype=np.float64)
    u_m = loaded["u"]
    v_m = loaded["v"]
    wl_m = loaded["wl"]
    v2_obs = loaded["v2"].ravel()
    dv2_obs = loaded["dv2"].ravel()

    def residual_fn(z_vec: np.ndarray) -> np.ndarray:
        p = {k: from_z(k, float(zv)) for k, zv in zip(site_order, z_vec, strict=True)}
        c = _binary_cloud(p["flux"], p["dra"], p["ddec"])
        v2_pred = (np.abs(xsky.visibility(c, u_m, v_m, wl_m)) ** 2).ravel()
        r_data = (v2_pred - v2_obs) / (p["noise.vis_scale"] * dv2_obs)
        # Fisher row for Gaussian error-scale normalization: sqrt(2 N) ln(s)
        r_scale = np.array([math.sqrt(2.0 * n_data) * math.log(p["noise.vis_scale"])])
        return np.concatenate([r_data, r_scale])

    def unconstr_log_prior(z_vec: np.ndarray) -> float:
        p_raw = dict(zip(site_order, z_vec, strict=True))
        # Include log|dx/dz| = z for LogNormal -> exact Normal in z!
        lp_dra = -0.5 * ((p_raw["dra"] - 8.5) / 5.0) ** 2
        lp_ddec = -0.5 * ((p_raw["ddec"] - (-5.5)) / 5.0) ** 2
        lp_flux = -0.5 * ((p_raw["flux"] - math.log(0.02)) / 0.5) ** 2
        lp_s = -0.5 * ((p_raw["noise.vis_scale"] - 0.0) / 0.25) ** 2
        return float(lp_dra + lp_ddec + lp_flux + lp_s)

    want_gn = ref.gauss_newton_precision_fd_ref(residual_fn, unconstr_log_prior, z_star)
    inv_rel_err = float(
        linalg.norm(got_inv_mass - np.asarray(want_gn["inverse_mass"]))
        / linalg.norm(np.asarray(want_gn["inverse_mass"]))
    )
    cond_rel_err = abs(gnm.condition - float(want_gn["condition"])) / float(want_gn["condition"])
    assert inv_rel_err < 5e-6
    assert cond_rel_err < 5e-6

    # Unpacks cleanly into NUTS and validates chain_init_params shapes + flat_coordinates guard
    posterior = vlike.numpyro_model(tmpl, priors, oid, noise=noise, flat_coordinates=False)
    kernel = NUTS(posterior, **gnm)
    assert kernel is not None

    init_single = vlike.chain_init_params(posterior, [v_opt], mass=gnm)
    assert all(np.shape(arr) == () for arr in init_single.values())
    assert float(init_single["flux"]) == pytest.approx(math.log(float(v_opt["flux"])), rel=1e-12)

    alt_start = {**v_opt, "dra": jnp.asarray(float(v_opt["dra"]) + 0.2)}
    init_three = vlike.chain_init_params(posterior, [v_opt, alt_start, v_opt], mass=gnm)
    assert all(np.shape(arr) == (3,) for arr in init_three.values())
    assert float(init_three["dra"][1] - init_three["dra"][0]) == pytest.approx(0.2, abs=1e-12)

    gnm_flat = vfit.gauss_newton_mass(tmpl, priors, oid, v_opt, noise=noise, flat_coordinates=True)
    with pytest.raises(ValueError):
        vlike.chain_init_params(posterior, [v_opt], mass=gnm_flat)

    # flat_coordinates=True on a LogUniform flux prior: sigmoid(z) in log-flux replaces NumPyro's
    # linear-flux sigmoid(z), matching standard logistic prior curvature + logit(log-fraction) init
    lu_a, lu_b = 1e-4, 0.2
    priors_lu = {
        "dra": dist.Normal(8.5, 5.0),
        "ddec": dist.Normal(-5.5, 5.0),
        "flux": dist.LogUniform(lu_a, lu_b),
    }
    gnm_lu_flat = vfit.gauss_newton_mass(tmpl, priors_lu, oid, v_opt, noise=noise, flat_coordinates=True)
    gnm_lu_raw = vfit.gauss_newton_mass(tmpl, priors_lu, oid, v_opt, noise=noise, flat_coordinates=False)
    site_lu = next(iter(gnm_lu_flat["inverse_mass_matrix"].keys()))
    im_lu_flat = np.asarray(gnm_lu_flat["inverse_mass_matrix"][site_lu], dtype=np.float64)
    im_lu_raw = np.asarray(gnm_lu_raw["inverse_mass_matrix"][site_lu], dtype=np.float64)
    assert linalg.norm(im_lu_flat - im_lu_raw) / linalg.norm(im_lu_flat) > 1e-5
    ln_a, ln_span = math.log(lu_a), math.log(lu_b / lu_a)

    def residual_lu_flat(z_vec: np.ndarray) -> np.ndarray:
        d = dict(zip(site_lu, z_vec, strict=True))
        f_val = math.exp(ln_a + ln_span * float(special.expit(d["flux"])))
        s_val = math.exp(float(d["noise.vis_scale"]))
        c = _binary_cloud(f_val, float(d["dra"]), float(d["ddec"]))
        v2_pred = (np.abs(xsky.visibility(c, u_m, v_m, wl_m)) ** 2).ravel()
        r_data = (v2_pred - v2_obs) / (s_val * dv2_obs)
        r_scale = np.array([math.sqrt(2.0 * n_data) * math.log(s_val)])
        return np.concatenate([r_data, r_scale])

    def log_prior_lu_flat(z_vec: np.ndarray) -> float:
        d = dict(zip(site_lu, z_vec, strict=True))
        zf = float(d["flux"])
        zv = float(d["noise.vis_scale"])
        lp_logistic = -float(np.logaddexp(0.0, -zf) + np.logaddexp(0.0, zf))
        return (
            -0.5 * ((float(d["dra"]) - 8.5) / 5.0) ** 2
            - 0.5 * ((float(d["ddec"]) - (-5.5)) / 5.0) ** 2
            + lp_logistic
            - 0.5 * (zv / 0.25) ** 2
        )

    z_lu_map = {
        "dra": float(v_opt["dra"]),
        "ddec": float(v_opt["ddec"]),
        "flux": float(special.logit((math.log(float(v_opt["flux"])) - ln_a) / ln_span)),
        "noise.vis_scale": math.log(float(v_opt["noise.vis_scale"])),
    }
    gn_lu_ref = ref.gauss_newton_precision_fd_ref(
        residual_lu_flat,
        log_prior_lu_flat,
        np.array([z_lu_map[k] for k in site_lu], dtype=np.float64),
    )
    lu_flat_rel_err = float(
        linalg.norm(im_lu_flat - np.asarray(gn_lu_ref["inverse_mass"]))
        / linalg.norm(np.asarray(gn_lu_ref["inverse_mass"]))
    )
    assert lu_flat_rel_err < 5e-6
    posterior_lu_flat = vlike.numpyro_model(tmpl, priors_lu, oid, noise=noise, flat_coordinates=True)
    init_lu_flat = vlike.chain_init_params(posterior_lu_flat, [v_opt], mass=gnm_lu_flat)
    assert float(init_lu_flat["flux"]) == pytest.approx(z_lu_map["flux"], abs=1e-12)

    # Bound detection on FitResult when Uniform box clips the optimum
    fres_clipped = vfit.fit(
        tmpl,
        {
            "dra": dist.Uniform(7.5, 10.5),
            "ddec": dist.Uniform(-7.5, -4.5),
            "flux": dist.Uniform(1e-4, 0.012),
        },
        oid,
    )
    assert "flux" in fres_clipped.info["at_bound"]

    record("gn_inv_mass_rel_err", inv_rel_err)
    record("gn_cond_rel_err", cond_rel_err)


@pytest.mark.validates(
    "virgil.fitting.gauss_newton_mass",
    roots=["mathematics"],
    kind="control",
    headline="missing_jacobian_factor",
)
def test_gauss_newton_mass_rejects_constrained_space_jacobian_omission(tmp_path):
    """Control: omitting the unconstrained bijection Jacobian dx/dz = f on a
    LogNormal flux parameter at f ~ 0.025 mis-scales the unconstrained flux
    variance by ~1/f^2 > 1000x, and an unidentifiable flat parameter raises
    SingularCurvature."""
    fits_path = tmp_path / "gn_ctrl.fits"
    truth = _binary_cloud(0.025, 9.0, -6.0)
    xsim.observe(
        fits_path,
        lambda u, v, wl: xsky.visibility(truth, u, v, wl),
        UTS3,
        hour_angles_h=[-1.0, 1.0],
        wavelengths=np.array([2.2e-6]),
        sigma_v2=0.003,
        closure_phases=False,
    )
    oid = OIData(read_oifits(str(fits_path)))
    loaded_ctrl = xchi2.load(fits_path)
    priors = {
        "dra": dist.Normal(9.0, 5.0),
        "ddec": dist.Normal(-6.0, 5.0),
        "flux": dist.LogNormal(math.log(0.025), 1.0),
    }
    tmpl = BinaryModelCartesian(9.0, -6.0, 0.025)
    fres = vfit.fit(tmpl, priors, oid)
    gnm = vfit.gauss_newton_mass(tmpl, priors, oid, fres.values)
    site_order = next(iter(gnm["inverse_mass_matrix"].keys()))
    flux_idx = site_order.index("flux")
    var_z_flux = float(gnm["inverse_mass_matrix"][site_order][flux_idx, flux_idx])

    # Compute constrained-space Gauss-Newton covariance wrt (dra, ddec, flux) WITHOUT the exp(z_flux) pullback
    x_star = np.array([float(fres.values[k]) for k in site_order], dtype=np.float64)
    u_c = loaded_ctrl["u"].ravel()
    v_c = loaded_ctrl["v"].ravel()
    wl_c = loaded_ctrl["wl"].ravel()
    v2_c, dv2_c = loaded_ctrl["v2"].ravel(), loaded_ctrl["dv2"].ravel()

    def constrained_residual_fn(x_vec: np.ndarray) -> np.ndarray:
        p = dict(zip(site_order, x_vec, strict=True))
        c = _binary_cloud(float(p["flux"]), float(p["dra"]), float(p["ddec"]))
        return ((np.abs(xsky.visibility(c, u_c, v_c, wl_c)) ** 2).ravel() - v2_c) / dv2_c

    def constrained_log_prior_fn(x_vec: np.ndarray) -> float:
        p = dict(zip(site_order, x_vec, strict=True))
        f_p = float(p["flux"])
        return float(
            -0.5 * ((float(p["dra"]) - 9.0) / 5.0) ** 2
            - 0.5 * ((float(p["ddec"]) - (-6.0)) / 5.0) ** 2
            - math.log(f_p)
            - 0.5 * ((math.log(f_p) - math.log(0.025)) / 1.0) ** 2
        )

    gn_constrained = ref.gauss_newton_precision_fd_ref(
        constrained_residual_fn, constrained_log_prior_fn, x_star
    )
    var_flux_constrained = float(np.asarray(gn_constrained["inverse_mass"])[flux_idx, flux_idx])
    f_val = float(fres.values["flux"])
    measured_ratio = var_z_flux / var_flux_constrained
    pullback_rel_err = abs(var_z_flux - var_flux_constrained / (f_val**2)) / var_z_flux
    assert measured_ratio > 1.0e3
    # Data J^T J pulls back by exact f^2; constrained -ln(f) prior adds +1/f^2 (~7e-4 relative)
    assert pullback_rel_err < 2e-3

    # Unconstrained uninformative direction ignored by model raises SingularCurvature
    with pytest.raises(vfit.SingularCurvature):
        vfit.gauss_newton_mass(
            lambda dra, ghost: BinaryModelCartesian(dra, -6.0, 0.025),
            {"dra": dist.Normal(9.0, 5.0), "ghost": dist.ImproperUniform(dist.constraints.real, (), ())},
            oid,
            {"dra": jnp.array(9.0), "ghost": jnp.array(0.0)},
        )
    record("missing_jacobian_factor", measured_ratio)


# ---------------------------------------------------------------------------
# 6. virgil.svi.{variational, VariationalResult}
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.validates(
    "virgil.svi.variational",
    "virgil.svi.VariationalResult",
    roots=["mathematics", "statistics"],
    tier="B",
    headline="svi_mvn_mean_zscore",
)
def test_svi_laplace_mvn_and_normalizing_flows_against_quadrature(tmp_path):
    """variational and VariationalResult recover Laplace / Gaussian posteriors
    under 'laplace' and 'mvn' guides with Pareto khat < 0.5, and 'bnaf' / 'iaf'
    flow guides match exact 2D Simpson-quadrature moments and quantiles on a
    skewed low-SNR log-flux posterior."""
    fits_path = tmp_path / "svi_high_snr.fits"
    truth = _binary_cloud(0.03, 8.0, -5.0)
    xsim.observe(
        fits_path,
        lambda u, v, wl: xsky.visibility(truth, u, v, wl),
        UTS3,
        hour_angles_h=[-1.5, 0.0, 1.5],
        wavelengths=np.array([2.0e-6, 2.2e-6]),
        sigma_v2=0.0025,
        closure_phases=False,
    )
    oid = OIData(read_oifits(str(fits_path)))
    loaded_3d = xchi2.load(fits_path)
    priors_3d = {
        "dra": dist.Normal(8.0, 4.0),
        "ddec": dist.Normal(-5.0, 4.0),
        "flux": dist.LogNormal(math.log(0.03), 0.5),
    }
    tmpl = BinaryModelCartesian(8.0, -5.0, 0.03)
    fres = vfit.fit(tmpl, priors_3d, oid)
    gnm = vfit.gauss_newton_mass(tmpl, priors_3d, oid, fres.values)
    site_order = next(iter(gnm["inverse_mass_matrix"].keys()))
    gn_cov = np.asarray(gnm["inverse_mass_matrix"][site_order])
    gn_sd = np.sqrt(np.diag(gn_cov))
    z_map = np.array(
        [math.log(float(fres.values[k])) if k == "flux" else float(fres.values[k]) for k in site_order],
        dtype=np.float64,
    )

    # Exact unconstrained log-joint density at MAP from independent xchi2.chi2 + priors
    c_map = _binary_cloud(
        float(fres.values["flux"]), float(fres.values["dra"]), float(fres.values["ddec"])
    )
    dv2_3d = loaded_3d["dv2"].ravel()
    ll_map = (
        -0.5 * float(xchi2.chi2(loaded_3d, lambda u, v, wl: xsky.visibility(c_map, u, v, wl)))
        - 0.5 * dv2_3d.size * math.log(2.0 * math.pi)
        - float(np.sum(np.log(dv2_3d)))
    )
    lp_unconstr_map = (
        float(stats.norm.logpdf(float(fres.values["dra"]), 8.0, 4.0))
        + float(stats.norm.logpdf(float(fres.values["ddec"]), -5.0, 4.0))
        + float(stats.norm.logpdf(math.log(float(fres.values["flux"])), math.log(0.03), 0.5))
    )
    want_log_joint = ll_map + lp_unconstr_map

    vi_lap = vsvi.variational(
        tmpl, priors_3d, oid, start=fres, guide="laplace", num_samples=4000, key=jax.random.PRNGKey(11)
    )
    vi_mvn = vsvi.variational(
        tmpl,
        priors_3d,
        oid,
        start=fres,
        guide="mvn",
        steps=800,
        learning_rate=3e-3,
        num_samples=4000,
        key=jax.random.PRNGKey(12),
    )
    assert isinstance(vi_lap, vsvi.VariationalResult) and vi_lap.converged is True
    assert vi_lap.info["guide"] == "laplace"
    assert vi_lap.info["khat"] is None and vi_lap.info["elbo"] is None
    assert abs(float(vi_lap.info["log_joint"]) - want_log_joint) < 5e-4  # float32 EFF_WAVE + BFGS tolerance

    assert isinstance(vi_mvn, vsvi.VariationalResult) and vi_mvn.converged is True
    assert vi_mvn.info["guide"] == "mvn"
    assert vi_mvn.info["khat"] is not None and float(vi_mvn.info["khat"]) < 0.5
    assert math.isfinite(float(vi_mvn.info["elbo"]))

    # Verify unconstrained posterior means AND full 3x3 covariances for both laplace and warm-start mvn
    max_mvn_z = 0.0
    max_cov_rel_3d = 0.0
    for vi_res in (vi_lap, vi_mvn):
        s_mat = np.column_stack(
            [
                np.log(np.asarray(vi_res.samples[k])) if k == "flux" else np.asarray(vi_res.samples[k])
                for k in site_order
            ]
        )
        z_err = float(np.max(np.abs(np.mean(s_mat, axis=0) - z_map) / gn_sd))
        cov_emp = np.cov(s_mat, rowvar=False)
        cov_rel = float(linalg.norm(cov_emp - gn_cov) / linalg.norm(gn_cov))
        assert z_err < 0.12
        assert cov_rel < 0.08
        max_mvn_z = max(max_mvn_z, z_err)
        max_cov_rel_3d = max(max_cov_rel_3d, cov_rel)

    # Cold-start MVN without start=fres on displaced priors (> 15 posterior SD from MAP)
    priors_cold = {
        "dra": dist.Normal(8.6, 4.0),
        "ddec": dist.Normal(-4.4, 4.0),
        "flux": dist.LogNormal(math.log(0.022), 0.5),
    }
    tmpl_cold = BinaryModelCartesian(8.6, -4.4, 0.022)
    fres_cold = vfit.fit(tmpl_cold, priors_cold, oid)
    gnm_cold = vfit.gauss_newton_mass(tmpl_cold, priors_cold, oid, fres_cold.values)
    gn_cov_cold = np.asarray(gnm_cold["inverse_mass_matrix"][site_order])
    gn_sd_cold = np.sqrt(np.diag(gn_cov_cold))
    z_map_cold = np.array(
        [
            math.log(float(fres_cold.values[k])) if k == "flux" else float(fres_cold.values[k])
            for k in site_order
        ],
        dtype=np.float64,
    )
    vi_mvn_cold = vsvi.variational(
        tmpl_cold,
        priors_cold,
        oid,
        start=None,
        guide="mvn",
        steps=2500,
        learning_rate=8e-3,
        num_samples=4000,
        key=jax.random.PRNGKey(13),
    )
    assert vi_mvn_cold.converged is True and float(vi_mvn_cold.info["khat"]) < 0.5
    s_cold = np.column_stack(
        [
            np.log(np.asarray(vi_mvn_cold.samples[k])) if k == "flux" else np.asarray(vi_mvn_cold.samples[k])
            for k in site_order
        ]
    )
    cold_z_err = float(np.max(np.abs(np.mean(s_cold, axis=0) - z_map_cold) / gn_sd_cold))
    cold_cov_rel = float(linalg.norm(np.cov(s_cold, rowvar=False) - gn_cov_cold) / linalg.norm(gn_cov_cold))
    assert cold_z_err < 0.20
    assert cold_cov_rel < 0.12
    max_mvn_z = max(max_mvn_z, cold_z_err)
    max_cov_rel_3d = max(max_cov_rel_3d, cold_cov_rel)

    # Genuinely skewed ~2.3-sigma faint-companion 2D posterior in (dra, u = ln flux):
    # above the mode, exp(u) hits a steep interferometric chi^2 wall, while below the
    # mode exp(u) -> 0 relaxes into the broad log-normal prior tail (left_w / right_w ~ 1.37).
    fits_low = tmp_path / "svi_skew2d.fits"
    faint = _binary_cloud(0.010, 8.0, -5.0)
    xsim.observe(
        fits_low,
        lambda u, v, wl: xsky.visibility(faint, u, v, wl),
        UTS3,
        hour_angles_h=[-1.0, 0.0, 1.0],
        wavelengths=np.array([2.1e-6, 2.3e-6]),
        sigma_v2=0.025,
        sigma_cp_deg=2.2,
        rng=np.random.default_rng(23),
    )
    oid_low = OIData(read_oifits(str(fits_low)))
    loaded_low = xchi2.load(fits_low)

    mu_logf = math.log(0.006)
    sd_logf = 1.0
    sd_dra = 0.50
    priors_2d = {
        "dra": dist.Normal(8.0, sd_dra),
        "flux": dist.LogNormal(mu_logf, sd_logf),
    }
    tmpl_2d = BinaryModelCartesian(8.0, -5.0, 0.010)
    fres_2d = vfit.fit(tmpl_2d, priors_2d, oid_low)

    # Vectorized exact 2D log-posterior in (dra, u = ln flux) cross-checked against xchi2.chi2
    u_v2 = loaded_low["u"].ravel()[None, None, :]
    v_v2 = loaded_low["v"].ravel()[None, None, :]
    wl_v2 = loaded_low["wl"].ravel()[None, None, :]
    val_v2 = loaded_low["v2"].ravel()[None, None, :]
    err_v2 = loaded_low["dv2"].ravel()[None, None, :]
    u1 = loaded_low["u1"].ravel()[None, None, :]
    v1 = loaded_low["v1"].ravel()[None, None, :]
    u2 = loaded_low["u2"].ravel()[None, None, :]
    v2 = loaded_low["v2_"].ravel()[None, None, :]
    u3 = u1 + u2
    v3 = v1 + v2
    wl_t3 = loaded_low["wl3"].ravel()[None, None, :]
    val_cp = loaded_low["cp"].ravel()[None, None, :]
    err_cp = loaded_low["dcp"].ravel()[None, None, :]
    ddec_rad = -5.0 * xsky.MAS

    def exact_log_post(dra_grid: np.ndarray, logf_grid: np.ndarray) -> np.ndarray:
        dra_rad = (dra_grid * xsky.MAS)[:, :, None]
        fr = np.exp(logf_grid)[:, :, None]
        f1 = 1.0 / (1.0 + fr)
        f2 = fr / (1.0 + fr)
        ph_v2 = -2.0 * np.pi * (u_v2 * dra_rad + v_v2 * ddec_rad) / wl_v2
        vis = f1 + f2 * np.exp(1j * ph_v2)
        chi2_v2 = np.sum(((vis.real**2 + vis.imag**2 - val_v2) / err_v2) ** 2, axis=-1)
        ph1 = -2.0 * np.pi * (u1 * dra_rad + v1 * ddec_rad) / wl_t3
        ph2 = -2.0 * np.pi * (u2 * dra_rad + v2 * ddec_rad) / wl_t3
        ph3 = -2.0 * np.pi * (u3 * dra_rad + v3 * ddec_rad) / wl_t3
        model_cp = (
            np.angle(f1 + f2 * np.exp(1j * ph1))
            + np.angle(f1 + f2 * np.exp(1j * ph2))
            - np.angle(f1 + f2 * np.exp(1j * ph3))
        )
        chi2_cp = np.sum((2.0 * np.sin((model_cp - val_cp) / 2.0) / err_cp) ** 2, axis=-1)
        return (
            -0.5 * (chi2_v2 + chi2_cp)
            - 0.5 * ((dra_grid - 8.0) / sd_dra) ** 2
            - 0.5 * ((logf_grid - mu_logf) / sd_logf) ** 2
        )

    chk_dra = np.array([[7.4, 8.3]])
    chk_lf = np.array([[-5.8, -4.7]])
    chk_vec = exact_log_post(chk_dra, chk_lf)
    for j in range(2):
        dx_j = float(chk_dra[0, j])
        lv_j = float(chk_lf[0, j])
        c2_j = xchi2.chi2(
            loaded_low,
            lambda u, v, wl, cl=_binary_cloud(math.exp(lv_j), dx_j, -5.0): xsky.visibility(cl, u, v, wl),
        )
        lp_ref = -0.5 * c2_j - 0.5 * ((dx_j - 8.0) / sd_dra) ** 2 - 0.5 * ((lv_j - mu_logf) / sd_logf) ** 2
        assert float(chk_vec[0, j]) == pytest.approx(lp_ref, abs=1e-12)

    exact_2d = ref.posterior_2d_grid_moments_ref(
        exact_log_post,
        (8.0 - 5.5 * sd_dra, 8.0 + 5.5 * sd_dra),
        (mu_logf - 5.0 * sd_logf, mu_logf + 3.5 * sd_logf),
        n_grid=221,
    )
    cov_exact_2d = np.asarray(exact_2d["cov"])
    sd_exact = np.sqrt(np.diag(cov_exact_2d))
    qx_ex = np.asarray(exact_2d["quantiles_x"])
    qy_ex = np.asarray(exact_2d["quantiles_y"])
    exact_skew_ratio = float((qy_ex[1] - qy_ex[0]) / (qy_ex[2] - qy_ex[1]))
    assert exact_skew_ratio > 1.30

    # Symmetric "mvn" in unconstrained space cannot represent the skewed ln(flux) tail
    vi_mvn_2d = vsvi.variational(
        tmpl_2d,
        priors_2d,
        oid_low,
        start=fres_2d,
        guide="mvn",
        steps=2000,
        optimizer=optax.adam(8e-3),
        num_samples=4000,
        key=jax.random.PRNGKey(2),
    )
    lf_mvn = np.log(np.asarray(vi_mvn_2d.samples["flux"]))
    qy_mvn = np.quantile(lf_mvn, [0.16, 0.50, 0.84])
    mvn_skew_ratio = float((qy_mvn[1] - qy_mvn[0]) / (qy_mvn[2] - qy_mvn[1]))
    qy_err_mvn = float(np.max(np.abs(qy_mvn - qy_ex)) / sd_exact[1])
    cov_mvn_2d = np.cov(
        np.column_stack([np.asarray(vi_mvn_2d.samples["dra"]), lf_mvn]),
        rowvar=False,
    )
    cov_rel_mvn_2d = float(linalg.norm(cov_mvn_2d - cov_exact_2d) / linalg.norm(cov_exact_2d))
    assert abs(mvn_skew_ratio - 1.0) < 0.10
    assert qy_err_mvn > 0.15

    flow_z_err = 0.0
    flow_cov_rel = 0.0
    for seed_val, flow_name in ((3, "bnaf"), (4, "iaf")):
        vi_flow = vsvi.variational(
            tmpl_2d,
            priors_2d,
            oid_low,
            start=fres_2d,
            guide=flow_name,
            steps=2500,
            optimizer=optax.adam(4e-3),
            num_samples=4000,
            key=jax.random.PRNGKey(seed_val),
        )
        assert vi_flow.converged is True
        assert float(vi_flow.info["khat"]) < 0.65
        dra_draws = np.asarray(vi_flow.samples["dra"])
        logf_draws = np.log(np.asarray(vi_flow.samples["flux"]))
        z_dra = abs(float(np.mean(dra_draws)) - float(exact_2d["mean"][0])) / sd_exact[0]
        z_logf = abs(float(np.mean(logf_draws)) - float(exact_2d["mean"][1])) / sd_exact[1]
        qx_flow = np.quantile(dra_draws, [0.16, 0.50, 0.84])
        qy_flow = np.quantile(logf_draws, [0.16, 0.50, 0.84])
        flow_skew_ratio = float((qy_flow[1] - qy_flow[0]) / (qy_flow[2] - qy_flow[1]))
        qx_err = float(np.max(np.abs(qx_flow - qx_ex)) / sd_exact[0])
        qy_err = float(np.max(np.abs(qy_flow - qy_ex)) / sd_exact[1])
        cov_flow = np.cov(np.column_stack([dra_draws, logf_draws]), rowvar=False)
        cov_rel_2d = float(linalg.norm(cov_flow - cov_exact_2d) / linalg.norm(cov_exact_2d))
        flow_z_err = max(flow_z_err, z_dra, z_logf)
        flow_cov_rel = max(flow_cov_rel, cov_rel_2d)
        assert flow_skew_ratio > 1.20
        assert qy_err < qy_err_mvn and cov_rel_2d < cov_rel_mvn_2d
        assert z_dra < 0.28 and z_logf < 0.15
        assert qx_err < 0.25 and qy_err < 0.15
        assert cov_rel_2d < 0.20

    record("svi_mvn_mean_zscore", max_mvn_z)
    record("svi_mvn_cov_rel_err", max_cov_rel_3d)
    record("svi_flow_mean_zscore", flow_z_err)
    record("svi_flow_cov_rel_err", flow_cov_rel)


@pytest.mark.validates(
    "virgil.svi.variational",
    roots=["mathematics", "statistics"],
    kind="control",
    headline="underdispersed_khat",
)
def test_svi_underdispersed_unconverged_guide_flags_khat_control(tmp_path):
    """Control: an unconverged, severely under-dispersed diagonal start
    (init_scale=1e-3, 2 steps, tiny learning rate) fails convergence and
    exhibits Pareto khat > 0.7."""
    fits_path = tmp_path / "svi_ctrl.fits"
    truth = _binary_cloud(0.02, 8.0, -5.0)
    xsim.observe(
        fits_path,
        lambda u, v, wl: xsky.visibility(truth, u, v, wl),
        UTS3,
        hour_angles_h=[-1.0, 1.0],
        wavelengths=np.array([2.2e-6]),
        sigma_v2=0.01,
        closure_phases=False,
    )
    oid = OIData(read_oifits(str(fits_path)))
    priors = {"dra": dist.Normal(8.0, 5.0), "ddec": dist.Normal(-5.0, 5.0)}
    tmpl = BinaryModelCartesian(8.0, -5.0, 0.02)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        vi_bad = vsvi.variational(
            tmpl,
            priors,
            oid,
            guide="mvn",
            dense_start=False,
            init_scale=1e-3,
            steps=10,
            window=10,
            learning_rate=1e-8,
            num_samples=1000,
            key=jax.random.PRNGKey(99),
        )
    msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert any("Pareto" in m for m in msgs)
    assert any("settled" in m for m in msgs)
    assert vi_bad.converged is False
    khat = float(vi_bad.info["khat"])
    assert khat > 0.7
    record("underdispersed_khat", khat)
