"""Independent mathematical and statistical validation of virgil 0.4.2
sequential Monte Carlo (`virgil.smc`), multimodal importance sampling
(`virgil.mode_search`), and correlated spectral-channel tempering
(`virgil.tempering`), together with reference self-checks for
`crosscheck.smc_tempering_ref`.
"""

from __future__ import annotations

import gc
import math

from astropy.io import fits
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, linalg, stats

from crosscheck import simulate as xsim, smc_tempering_ref as ref
from evidence.plugin import record

vsmc = pytest.importorskip("virgil.smc")
vms = pytest.importorskip("virgil.mode_search")
vtemp = pytest.importorskip("virgil.tempering")
from virgil.oidata import OIData  # noqa: E402
from virgil.oifits import read_oifits  # noqa: E402

pytestmark = pytest.mark.x64


@pytest.fixture(scope="module", autouse=True)
def _clear_jax_caches():
    yield
    jax.clear_caches()
    gc.collect()


# ---------------------------------------------------------------------------
# 0. Reference module self-checks (crosscheck.smc_tempering_ref)
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "crosscheck.smc_tempering_ref",
    roots=["mathematics"],
    kind="reference",
    headline="max_ref_err",
)
def test_smc_tempering_ref_against_quadrature_and_matrix_identities():
    """Verify crosscheck.smc_tempering_ref formulas against 2D SciPy numerical
    quadrature, Woodbury marginal densities, and exact covariance quadratic
    forms."""
    mu0 = np.array([0.4, -0.7])
    cov0 = np.array([[1.3, 0.35], [0.35, 0.9]])
    design = np.array([[1.1, -0.4], [0.2, 1.5], [-0.8, 0.6]])
    cov_y = np.array([[0.25, 0.04, -0.02], [0.04, 0.30, 0.05], [-0.02, 0.05, 0.20]])
    y = np.array([0.9, -0.8, 0.2])

    # beta = 0 gives ln Z = 0 and prior moments; beta = 1 equals exact marginal y density
    r0 = ref.gaussian_conjugate_tempering_ref(mu0, cov0, design, y, cov_y, beta=0.0)
    r1 = ref.gaussian_conjugate_tempering_ref(mu0, cov0, design, y, cov_y, beta=1.0)
    want_marginal_lz = float(
        stats.multivariate_normal.logpdf(y, mean=design @ mu0, cov=cov_y + design @ cov0 @ design.T)
    )
    err_woodbury = abs(float(r1["log_z"]) - want_marginal_lz)
    assert abs(float(r0["log_z"])) < 1e-15
    assert err_woodbury < 1e-13

    # Intermediate beta = 0.4 checked against direct 2D dblquad
    beta = 0.4
    rb = ref.gaussian_conjugate_tempering_ref(mu0, cov0, design, y, cov_y, beta=beta)

    def tempered_integrand(th1: float, th0: float) -> float:
        th = np.array([th0, th1])
        lp = stats.multivariate_normal.logpdf(th, mean=mu0, cov=cov0)
        ll = stats.multivariate_normal.logpdf(y, mean=design @ th, cov=cov_y)
        return math.exp(float(lp + beta * ll - rb["log_z"]))

    ratio_quad, _ = integrate.dblquad(
        tempered_integrand, -6.0, 6.0, -6.0, 6.0, epsabs=1e-11, epsrel=1e-10
    )
    err_beta_quad = abs(ratio_quad - 1.0)
    assert err_beta_quad < 1e-9

    # Bimodal Gaussian mixture checked against direct 2D dblquad
    m1 = np.array([-1.8, -1.0])
    m2 = np.array([1.9, 1.4])
    s1 = np.array([[0.30, 0.08], [0.08, 0.25]])
    s2 = np.array([[0.22, -0.05], [-0.05, 0.35]])
    bm = ref.bimodal_gaussian_mixture_ref(mu0, cov0, [m1, m2], [s1, s2], [0.7, -0.2])

    def bimodal_integrand(th1: float, th0: float) -> float:
        th = np.array([th0, th1])
        p0 = stats.multivariate_normal.pdf(th, mean=mu0, cov=cov0)
        like = math.exp(0.7) * stats.multivariate_normal.pdf(th, mean=m1, cov=s1) + math.exp(
            -0.2
        ) * stats.multivariate_normal.pdf(th, mean=m2, cov=s2)
        return float(p0 * like / math.exp(float(bm["log_z"])))

    bm_ratio, _ = integrate.dblquad(
        bimodal_integrand, -7.0, 7.0, -7.0, 7.0, epsabs=1e-10, epsrel=1e-9
    )
    err_bm_quad = abs(bm_ratio - 1.0)
    assert err_bm_quad < 1e-8

    # Pointwise closed-form check + symmetry-image invariance + 2D normalization for defensive_mixture_log_q_ref
    inv_map = -np.eye(2)
    x_pt = np.array([0.65, -0.40])
    lp_fn = lambda pts: stats.multivariate_normal.logpdf(pts, mean=mu0, cov=cov0)
    got_lq_pt = float(
        ref.defensive_mixture_log_q_ref(
            x_pt, [m1, m2], [-1.0, -1.5], [s1, s2], lp_fn, alpha=0.2, df=5.0, images=[inv_map]
        )
    )
    lap1 = -1.0 + math.log(2.0 * math.pi) + 0.5 * math.log(float(linalg.det(s1)))
    lap2 = -1.5 + math.log(2.0 * math.pi) + 0.5 * math.log(float(linalg.det(s2)))
    w1_lap = math.exp(lap1) / (math.exp(lap1) + math.exp(lap2))
    w2_lap = 1.0 - w1_lap
    want_q_pt = 0.2 * float(stats.multivariate_normal.pdf(x_pt, mean=mu0, cov=cov0)) + 0.8 * (
        0.5
        * w1_lap
        * (
            float(stats.multivariate_t.pdf(x_pt, loc=m1, shape=s1, df=5.0))
            + float(stats.multivariate_t.pdf(x_pt, loc=-m1, shape=s1, df=5.0))
        )
        + 0.5
        * w2_lap
        * (
            float(stats.multivariate_t.pdf(x_pt, loc=m2, shape=s2, df=5.0))
            + float(stats.multivariate_t.pdf(x_pt, loc=-m2, shape=s2, df=5.0))
        )
    )
    err_dm_pt = abs(got_lq_pt - math.log(want_q_pt))
    assert err_dm_pt < 1e-14

    # Closed-form checks for logmeanexp_ref, laplace_log_evidence_ref, distinct_modes_ref, importance_diagnostics_ref
    err_lme = abs(
        ref.logmeanexp_ref(np.array([-1000.0, -1000.0 + math.log(3.0)]))
        - (-1000.0 + math.log(2.0))
    )
    assert err_lme < 5e-13  # float64 ULP at magnitude 1000 is 2.22e-13
    assert math.isinf(ref.logmeanexp_ref(np.array([-math.inf, -math.inf])))
    err_lap = abs(ref.laplace_log_evidence_ref(-1.0, s1) - lap1)
    assert err_lap < 1e-15

    assert ref.distinct_modes_ref(
        [np.array([0.0, 0.0]), np.array([0.8, 0.0]), np.array([1.2, 0.0])],
        [5.0, 4.0, 3.0],
        [np.eye(2), 0.01 * np.eye(2), np.eye(2)],
        mahal=1.0,
        two_sided=True,
    ) == [0, 1, 2]
    assert ref.distinct_modes_ref(
        [np.array([0.0, 0.0]), np.array([0.8, 0.0]), np.array([1.2, 0.0])],
        [5.0, 4.0, 3.0],
        [np.eye(2), 0.01 * np.eye(2), np.eye(2)],
        mahal=1.0,
        two_sided=False,
    ) == [0, 2]

    imp_d = ref.importance_diagnostics_ref(np.array([0.0, math.log(3.0)]))
    err_imp = max(
        abs(imp_d["log_z"] - math.log(2.0)),
        abs(imp_d["ess"] - 1.6),
        abs(imp_d["log_z_error"] - math.sqrt(0.125)),
    )
    assert err_imp < 1e-15

    dm_norm, _ = integrate.dblquad(
        lambda y_c, x_c: math.exp(
            float(
                ref.defensive_mixture_log_q_ref(
                    np.array([x_c, y_c]),
                    [m1, m2],
                    [-1.0, -1.5],
                    [s1, s2],
                    lp_fn,
                    alpha=0.2,
                    df=5.0,
                    images=[inv_map],
                )
            )
        ),
        -18.0,
        18.0,
        -18.0,
        18.0,
        epsabs=1e-8,
        epsrel=1e-8,
    )
    err_dm_norm = abs(dm_norm - 1.0)
    assert err_dm_norm < 5e-5

    # Exact Toeplitz, equicorrelated, finite-chain, lag-correlation, and block-DOF identities
    n_ch, rho, c_corr = 7, 0.68, 0.45
    idx = np.arange(n_ch)
    sigma_ar1 = rho ** np.abs(idx[:, None] - idx[None, :])
    sigma_cm = np.full((n_ch, n_ch), c_corr)
    np.fill_diagonal(sigma_cm, 1.0)
    err_ar1 = abs(ref.ar1_finite_chain_factor(n_ch, rho) - float(np.sum(sigma_ar1) / n_ch))
    err_cm = abs(ref.common_mode_chain_factor(n_ch, c_corr) - float(np.sum(sigma_cm) / n_ch))
    err_ar1_lim = abs(
        ref.ar1_finite_chain_factor(20000, rho) - ref.ar1_asymptotic_factor(rho)
    )
    assert err_ar1 < 1e-14
    assert err_cm < 1e-14
    assert err_ar1_lim < 1e-3

    f_raw_hw, f_fl_hw = ref.finite_chain_variance_ratio_ref(
        [np.array([1.0, 2.0]), np.array([1.0, -1.0])]
    )
    _, f_alt_hw = ref.finite_chain_variance_ratio_ref([np.array([1.0, -1.0])])
    rho1_hw, np1_hw = ref.chain_lag_correlation_ref([np.array([1.0, 2.0, 4.0])], lag=1)
    rho2_hw, np2_hw = ref.chain_lag_correlation_ref([np.array([1.0, 2.0, 4.0])], lag=2)
    err_temp_hw = max(
        abs(f_raw_hw - 9.0 / 7.0),
        abs(f_fl_hw - 9.0 / 7.0),
        abs(f_alt_hw - 1.0),
        abs(rho1_hw - 0.8),
        abs(rho2_hw - 4.0 / 8.5),
    )
    assert err_temp_hw < 1e-15 and np1_hw == 2 and np2_hw == 1
    assert ref.flat_signature_ref(0.4, 0.50, 400) is True
    assert ref.flat_signature_ref(0.7, 0.52, 400) is False
    assert ref.out_of_range_ref(1.0, 0.3) is True and ref.out_of_range_ref(1.4, 1.4) is False
    assert ref.block_dof_ref({"a": 2.5, "b": 0.4}) == {"a": 0.4, "b": 1.0}

    max_err = max(
        err_woodbury,
        err_beta_quad,
        err_bm_quad,
        err_dm_pt,
        err_lme,
        err_lap,
        err_imp,
        err_ar1,
        err_cm,
        err_temp_hw,
    )
    record("max_ref_err", max_err)
    record("dm_tail_trunc_err", err_dm_norm)


# ---------------------------------------------------------------------------
# 1. virgil.smc.{SMCResult, tempered_smc, bridge_smc}
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.validates(
    "virgil.smc.tempered_smc",
    "virgil.smc.SMCResult",
    roots=["mathematics", "statistics"],
    tier="B",
    headline="abs_mean_log_z_err",
)
def test_tempered_smc_conjugate_gaussian_evidence_se_and_flags():
    """tempered_smc and SMCResult recover exact conjugate Gaussian log Z,
    posterior moments, calibrated Lee-Whiteley replicate SE, and stopping /
    undersampling flags."""
    mu0 = np.array([0.3, -0.5, 0.2])
    cov0 = np.array(
        [[1.4, 0.30, -0.15], [0.30, 1.1, 0.20], [-0.15, 0.20, 0.9]]
    )
    design = np.array(
        [
            [1.3, -0.4, 0.2],
            [0.1, 1.5, -0.3],
            [-0.7, 0.5, 1.2],
            [0.8, 0.6, 0.9],
            [-0.4, -0.9, 0.7],
        ]
    )
    cov_y = np.diag([0.09, 0.12, 0.08, 0.10, 0.11])
    y = np.array([1.25, -1.10, 0.55, 0.40, -0.25])

    exact = ref.gaussian_conjugate_tempering_ref(mu0, cov0, design, y, cov_y, beta=1.0)
    chol0 = linalg.cholesky(cov0, lower=True)
    prec0 = linalg.inv(cov0)
    prec_y = linalg.inv(cov_y)
    _, logdet0 = np.linalg.slogdet(cov0)
    _, logdet_y = np.linalg.slogdet(cov_y)
    const0 = -0.5 * 3 * math.log(2.0 * math.pi) - 0.5 * float(logdet0)
    const_y = -0.5 * y.size * math.log(2.0 * math.pi) - 0.5 * float(logdet_y)

    mu0_j = jnp.asarray(mu0)
    chol0_j = jnp.asarray(chol0)
    prec0_j = jnp.asarray(prec0)
    design_j = jnp.asarray(design)
    prec_y_j = jnp.asarray(prec_y)
    y_j = jnp.asarray(y)

    def sample_prior(key, n):
        z = jax.random.normal(key, (n, 3), dtype=jnp.float64)
        return mu0_j[None, :] + z @ chol0_j.T

    def logprior(th):
        d = th - mu0_j
        return const0 - 0.5 * jnp.einsum("i,ij,j->", d, prec0_j, d)

    def loglike(th):
        r = y_j - design_j @ th
        return const_y - 0.5 * jnp.einsum("i,ij,j->", r, prec_y_j, r)

    n_rep = 6
    res = vsmc.tempered_smc(
        logprior,
        loglike,
        sample_prior,
        key=jax.random.PRNGKey(200),
        n_particles=384,
        replicates=n_rep,
        steps=4,
        ess_target=0.25,
    )
    assert isinstance(res, vsmc.SMCResult)
    assert not res.stopped
    assert not res.undersampled
    assert not res.variance_degenerate
    assert len(res.betas) == n_rep and len(res.ess) == n_rep
    for b_arr, e_arr in zip(res.betas, res.ess, strict=True):
        assert float(b_arr[-1]) == 1.0
        assert np.all(np.diff(np.concatenate([[0.0], np.asarray(b_arr)])) > 0.0)
        assert len(e_arr) == len(b_arr)
    assert math.isfinite(res.log_z_variance) and res.log_z_variance > 0.0
    assert math.isfinite(res.log_z_error) and res.log_z_error > 0.0
    assert float(res.log_z_error) == pytest.approx(
        math.sqrt(float(res.log_z_variance)), rel=1e-12
    )

    lz_arr = np.asarray(res.log_z_replicates, dtype=np.float64)
    assert lz_arr.shape == (n_rep,)
    assert float(res.log_z) == pytest.approx(ref.logmeanexp_ref(lz_arr), abs=1e-13)
    mean_lz_err = abs(float(res.log_z) - float(exact["log_z"]))

    pooled = np.asarray(res.particles)
    weights = np.asarray(res.weights)
    assert pooled.shape == (n_rep * 384, 3)
    assert float(np.sum(weights)) == pytest.approx(1.0, abs=1e-12)
    post_mean = np.sum(weights[:, None] * pooled, axis=0)
    centered = pooled - post_mean[None, :]
    post_cov = (centered.T * weights[None, :]) @ centered
    post_sd = np.sqrt(np.diag(np.asarray(exact["cov"])))
    mean_zscore = float(np.max(np.abs(post_mean - np.asarray(exact["mean"])) / post_sd))
    cov_rel_err = float(
        linalg.norm(post_cov - np.asarray(exact["cov"]))
        / linalg.norm(np.asarray(exact["cov"]))
    )

    assert mean_lz_err < max(4.0 * float(res.log_z_error), 0.05)
    assert mean_zscore < 0.15
    assert cov_rel_err < 0.15

    # Single-run (replicates=1) Lee & Whiteley (2018, Theorem 1) genealogy variance identity
    # across 96 independent trials under multinomial resampling (Du & Guyader 2021):
    # (Z_hat)^2 * log_z_variance is an unbiased estimator of Var(Z_hat) when all non-collapsed
    # finite realizations (including finite non-positive draws) are retained.
    c0_1d = -0.5 * math.log(2.0 * math.pi * 1.5)
    cy_1d = -math.log(2.0 * math.pi) - 0.5 * math.log(0.10 * 0.12)
    d_1d = jnp.array([[1.3], [-0.9]], dtype=jnp.float64)
    y_1d = jnp.array([1.1, -0.7], dtype=jnp.float64)
    py_1d = jnp.diag(jnp.array([10.0, 1.0 / 0.12], dtype=jnp.float64))
    ex_1d = ref.gaussian_conjugate_tempering_ref(
        np.array([0.2]),
        np.array([[1.5]]),
        np.array([[1.3], [-0.9]]),
        np.array([1.1, -0.7]),
        np.diag([0.10, 0.12]),
    )
    exact_lz_1d = float(ex_1d["log_z"])
    single_lzs = []
    single_vars = []
    for t_idx in range(96):
        r_single = vsmc.tempered_smc(
            lambda th: c0_1d - 0.5 * ((th[0] - 0.2) ** 2) / 1.5,
            lambda th: cy_1d - 0.5 * ((y_1d - d_1d @ th) @ py_1d @ (y_1d - d_1d @ th)),
            lambda k_rng, n_p: 0.2 + math.sqrt(1.5) * jax.random.normal(k_rng, (n_p, 1), dtype=jnp.float64),
            key=jax.random.PRNGKey(5000 + t_idx),
            n_particles=128,
            replicates=1,
            steps=1,
            leapfrogs=2,
        )
        v_s = float(r_single.log_z_variance)
        e_s = float(r_single.log_z_error)
        deg_s = bool(r_single.variance_degenerate)
        assert deg_s == (not (v_s > 0.0))
        if deg_s:
            assert math.isnan(e_s)
        else:
            assert e_s == pytest.approx(math.sqrt(v_s), rel=1e-12)
        single_lzs.append(float(r_single.log_z))
        single_vars.append(v_s)

    lzs_1d = np.asarray(single_lzs, dtype=np.float64)
    vars_1d = np.asarray(single_vars, dtype=np.float64)
    non_collapsed = np.isfinite(vars_1d)
    assert int(np.sum(non_collapsed)) >= 92
    z_ratios = np.exp(lzs_1d[non_collapsed] - exact_lz_1d)
    var_hats = (z_ratios**2) * vars_1d[non_collapsed]
    est_var = float(np.mean(var_hats))
    se_est = float(np.std(var_hats, ddof=1) / math.sqrt(var_hats.size))
    emp_var = float(np.var(z_ratios, ddof=1))
    rel_se = math.sqrt((se_est / emp_var) ** 2 + 2.0 / (var_hats.size - 1))
    single_var_ratio = est_var / emp_var
    assert abs(float(np.mean(z_ratios)) - 1.0) < 3.5 * math.sqrt(emp_var / var_hats.size)
    assert rel_se < 0.22
    assert abs(single_var_ratio - 1.0) < 3.0 * rel_se

    # Multi-replicate (replicates=4) Lee & Whiteley (2018) pooling identity across 32 trials:
    # exp(2 * (log_z - exact_lz_1d)) * log_z_variance has exact expectation Var(Z_bar / Z) = Var(Z_1 / Z) / R,
    # rejecting factor-of-R pooling errors (> 4 sigma against both 4x and 0.25x mis-scalings).
    n_rep_mc = 4
    pooled_lzs = []
    pooled_vars = []
    for m_idx in range(32):
        r_pool = vsmc.tempered_smc(
            lambda th: c0_1d - 0.5 * ((th[0] - 0.2) ** 2) / 1.5,
            lambda th: cy_1d - 0.5 * ((y_1d - d_1d @ th) @ py_1d @ (y_1d - d_1d @ th)),
            lambda k_rng, n_p: 0.2 + math.sqrt(1.5) * jax.random.normal(k_rng, (n_p, 1), dtype=jnp.float64),
            key=jax.random.PRNGKey(7000 + m_idx),
            n_particles=128,
            replicates=n_rep_mc,
            steps=1,
            leapfrogs=2,
        )
        vp = float(r_pool.log_z_variance)
        ep = float(r_pool.log_z_error)
        assert bool(r_pool.variance_degenerate) == (not (vp > 0.0))
        if not r_pool.variance_degenerate:
            assert ep == pytest.approx(math.sqrt(vp), rel=1e-12)
        pooled_lzs.append(float(r_pool.log_z))
        pooled_vars.append(vp)

    p_lzs = np.asarray(pooled_lzs, dtype=np.float64)
    p_vars = np.asarray(pooled_vars, dtype=np.float64)
    p_ok = np.isfinite(p_vars)
    assert int(np.sum(p_ok)) == 32
    z_bar_ratios = np.exp(p_lzs[p_ok] - exact_lz_1d)
    pooled_var_hats = (z_bar_ratios**2) * p_vars[p_ok]
    est_pooled = float(np.mean(pooled_var_hats))
    se_pooled = float(np.std(pooled_var_hats, ddof=1) / math.sqrt(pooled_var_hats.size))
    emp_pooled = float(np.var(z_bar_ratios, ddof=1))
    target_from_single = emp_var / n_rep_mc
    rel_se_vs_single = math.sqrt((se_pooled / target_from_single) ** 2 + 2.0 / (var_hats.size - 1))
    rel_se_vs_emp = math.sqrt((se_pooled / emp_pooled) ** 2 + 2.0 / (pooled_var_hats.size - 1))
    assert rel_se_vs_single < 0.22
    assert abs(est_pooled / target_from_single - 1.0) < 3.0 * rel_se_vs_single
    assert abs(est_pooled / emp_pooled - 1.0) < 2.5 * rel_se_vs_emp

    # Zero-log-likelihood single-stage multinomial resampling over sequential PRNGKey(0)..PRNGKey(15):
    # exact Var(Z_hat) = 0 and E[log_z_variance] = 0 independent of HMC dynamics, so n_particles=8
    # hits finite non-positive realizations on ~1/2 of draws and n_particles=2 collapses onto a single
    # Eve lineage with exact probability 3/4 per trial without scanned magic seeds.
    saw_nonpos_finite = False
    saw_pos_finite = False
    saw_collapsed_nan = False
    for k_seq in range(16):
        r8 = vsmc.tempered_smc(
            lambda th: -0.5 * math.log(2.0 * math.pi) - 0.5 * (th[0] ** 2),
            lambda th: jnp.zeros((), dtype=th.dtype),
            lambda k_rng, n_p: jax.random.normal(k_rng, (n_p, 1), dtype=jnp.float64),
            key=jax.random.PRNGKey(k_seq),
            n_particles=8,
            replicates=1,
            steps=1,
            leapfrogs=1,
        )
        v8, e8, d8 = float(r8.log_z_variance), float(r8.log_z_error), bool(r8.variance_degenerate)
        assert d8 == (not (v8 > 0.0))
        if math.isfinite(v8) and v8 <= 0.0:
            saw_nonpos_finite = True
            assert d8 is True and math.isnan(e8)
        elif v8 > 0.0:
            saw_pos_finite = True
            assert d8 is False and e8 == pytest.approx(math.sqrt(v8), rel=1e-12)

        r2 = vsmc.tempered_smc(
            lambda th: -0.5 * math.log(2.0 * math.pi) - 0.5 * (th[0] ** 2),
            lambda th: jnp.zeros((), dtype=th.dtype),
            lambda k_rng, n_p: jax.random.normal(k_rng, (n_p, 1), dtype=jnp.float64),
            key=jax.random.PRNGKey(k_seq),
            n_particles=2,
            replicates=1,
            steps=1,
            leapfrogs=1,
        )
        v2_val, e2_val, d2_val = float(r2.log_z_variance), float(r2.log_z_error), bool(r2.variance_degenerate)
        assert d2_val is True and math.isnan(e2_val)
        if math.isnan(v2_val):
            saw_collapsed_nan = True
    assert saw_nonpos_finite and saw_pos_finite and saw_collapsed_nan

    # Dict particles ({"x": continuous, "k": discrete}) + custom Gibbs move leaving target invariant
    def init_dict(k_rng, n_p):
        kx, kk = jax.random.split(k_rng)
        return {
            "x": jax.random.normal(kx, (n_p, 1), dtype=jnp.float64),
            "k": jax.random.bernoulli(kk, 0.5, (n_p,)).astype(jnp.int32),
        }

    def gibbs_k_move(k_rng, particle, logdensity):
        ld0 = logdensity({**particle, "k": jnp.int32(0)})
        ld1 = logdensity({**particle, "k": jnp.int32(1)})
        k_new = jax.random.bernoulli(k_rng, jax.nn.sigmoid(ld1 - ld0)).astype(jnp.int32)
        return {**particle, "k": k_new}, ()

    res_dict = vsmc.tempered_smc(
        lambda p: -0.5 * math.log(2.0 * math.pi) - 0.5 * jnp.sum(p["x"] ** 2) - math.log(2.0),
        lambda p: -0.5 * math.log(2.0 * math.pi * 0.25**2)
        - 0.5 * ((1.5 - (jnp.where(p["k"] == 1, 1.4, -1.2) + p["x"][0])) / 0.25) ** 2,
        init_dict,
        key=jax.random.PRNGKey(277),
        n_particles=512,
        steps=3,
        moves=(gibbs_k_move,),
    )
    sd_marg = math.sqrt(1.0 + 0.25**2)
    z_dict_exact = math.log(
        0.5 * float(stats.norm.pdf(1.5, loc=-1.2, scale=sd_marg))
        + 0.5 * float(stats.norm.pdf(1.5, loc=1.4, scale=sd_marg))
    )
    pk1_exact = 0.5 * float(stats.norm.pdf(1.5, loc=1.4, scale=sd_marg)) / math.exp(z_dict_exact)
    pk1_got = float(
        np.sum(np.asarray(res_dict.weights) * (np.asarray(res_dict.particles["k"]) == 1))
    )
    assert abs(float(res_dict.log_z) - z_dict_exact) < 0.15
    assert abs(pk1_got - pk1_exact) < 0.05

    # Diagnostic flags: ess_target = 0.99 flags undersampled; max_stages = 1 stops early
    res_under = vsmc.tempered_smc(
        logprior,
        loglike,
        sample_prior,
        key=jax.random.PRNGKey(299),
        n_particles=256,
        steps=3,
        ess_target=0.99,
    )
    assert res_under.undersampled is True and res_under.stopped is False

    res_stopped = vsmc.tempered_smc(
        logprior,
        loglike,
        sample_prior,
        key=jax.random.PRNGKey(300),
        n_particles=256,
        steps=2,
        max_stages=1,
    )
    assert res_stopped.stopped is True
    assert res_stopped.particles is None and math.isnan(res_stopped.log_z)

    record("abs_mean_log_z_err", mean_lz_err)
    record("genealogy_var_ratio", single_var_ratio)
    record("posterior_mean_zscore", mean_zscore)
    record("posterior_cov_rel_err", cov_rel_err)


@pytest.mark.validates(
    "virgil.smc.tempered_smc",
    roots=["mathematics", "statistics"],
    headline="bimodal_log_z_err",
)
def test_tempered_smc_bimodal_gaussian_mixture_posterior():
    """tempered_smc captures both modes, relative mode weights, and total log Z
    on a well-separated 2D bimodal Gaussian-mixture target."""
    mu0 = np.zeros(2)
    cov0 = 9.0 * np.eye(2)
    m1 = np.array([-2.4, -1.6])
    m2 = np.array([2.5, 1.9])
    s1 = np.array([[0.18, 0.05], [0.05, 0.14]])
    s2 = np.array([[0.15, -0.04], [-0.04, 0.20]])
    log_scales = [1.1, 0.4]

    exact = ref.bimodal_gaussian_mixture_ref(mu0, cov0, [m1, m2], [s1, s2], log_scales)
    prec1, prec2 = jnp.asarray(linalg.inv(s1)), jnp.asarray(linalg.inv(s2))
    _, ld1 = np.linalg.slogdet(s1)
    _, ld2 = np.linalg.slogdet(s2)
    c1 = log_scales[0] - math.log(2.0 * math.pi) - 0.5 * float(ld1)
    c2 = log_scales[1] - math.log(2.0 * math.pi) - 0.5 * float(ld2)
    m1_j, m2_j = jnp.asarray(m1), jnp.asarray(m2)
    lp_const = -math.log(2.0 * math.pi) - math.log(9.0)

    def sample_prior(key, n):
        return 3.0 * jax.random.normal(key, (n, 2), dtype=jnp.float64)

    def logprior(th):
        return lp_const - 0.5 * jnp.sum(th**2) / 9.0

    def loglike(th):
        d1 = th - m1_j
        d2 = th - m2_j
        ll1 = c1 - 0.5 * jnp.einsum("i,ij,j->", d1, prec1, d1)
        ll2 = c2 - 0.5 * jnp.einsum("i,ij,j->", d2, prec2, d2)
        return jnp.logaddexp(ll1, ll2)

    res = vsmc.tempered_smc(
        logprior,
        loglike,
        sample_prior,
        key=jax.random.PRNGKey(401),
        n_particles=1024,
        steps=5,
    )
    lz_err = abs(res.log_z - float(exact["log_z"]))
    pts = np.asarray(res.particles)
    w_smc = np.asarray(res.weights)
    w2_hat = float(np.sum(w_smc[pts[:, 0] > 0.0]))
    w2_err = abs(w2_hat - float(exact["weights"][1]))
    mean_err = float(linalg.norm(np.sum(w_smc[:, None] * pts, axis=0) - np.asarray(exact["mean"])))

    # Single-mode evidence misses > 0.35 nats; SMC recovers total evidence within 0.08 nats
    assert lz_err < 0.08
    assert w2_err < 0.06
    assert mean_err < 0.25

    record("bimodal_log_z_err", lz_err)
    record("mode2_weight_err", w2_err)
    record("bimodal_mean_err", mean_err)


@pytest.mark.validates(
    "virgil.smc.bridge_smc",
    roots=["mathematics", "statistics"],
    headline="bridge_log_bf_err",
)
def test_bridge_smc_exact_linear_gaussian_bayes_factor():
    """bridge_smc recovers the closed-form log Bayes factor ln(Z_2 / Z_1)
    between two conjugate Gaussian models and satisfies antisymmetry."""
    mu0 = np.array([0.0, 0.0])
    cov0 = 4.0 * np.eye(2)
    design1 = np.array([[1.0, 0.2], [-0.3, 1.1], [0.6, -0.8]])
    design2 = np.array([[1.2, -0.5], [0.4, 0.9], [-0.5, 1.0]])
    cov_y = np.diag([0.20, 0.25, 0.18])
    y = np.array([0.85, -0.60, 0.30])

    r1 = ref.gaussian_conjugate_tempering_ref(mu0, cov0, design1, y, cov_y, beta=1.0)
    r2 = ref.gaussian_conjugate_tempering_ref(mu0, cov0, design2, y, cov_y, beta=1.0)
    want_log_bf = float(r2["log_z"]) - float(r1["log_z"])

    prec0 = jnp.asarray(linalg.inv(cov0))
    prec_y = jnp.asarray(linalg.inv(cov_y))
    _, ld0 = np.linalg.slogdet(cov0)
    _, ldy = np.linalg.slogdet(cov_y)
    c0 = -math.log(2.0 * math.pi) - 0.5 * float(ld0)
    cy = -0.5 * 3 * math.log(2.0 * math.pi) - 0.5 * float(ldy)
    d1_j, d2_j, y_j = jnp.asarray(design1), jnp.asarray(design2), jnp.asarray(y)

    def logprior(th):
        return c0 - 0.5 * (th @ prec0 @ th)

    def loglike_1(th):
        r = y_j - d1_j @ th
        return cy - 0.5 * (r @ prec_y @ r)

    def loglike_2(th):
        r = y_j - d2_j @ th
        return cy - 0.5 * (r @ prec_y @ r)

    rng = np.random.default_rng(512)
    init_1 = jnp.asarray(
        rng.multivariate_normal(np.asarray(r1["mean"]), np.asarray(r1["cov"]), size=1024)
    )
    init_2 = jnp.asarray(
        rng.multivariate_normal(np.asarray(r2["mean"]), np.asarray(r2["cov"]), size=1024)
    )

    fwd = vsmc.bridge_smc(
        logprior, loglike_1, loglike_2, init_1, key=jax.random.PRNGKey(51), steps=4
    )
    rev = vsmc.bridge_smc(
        logprior, loglike_2, loglike_1, init_2, key=jax.random.PRNGKey(52), steps=4
    )
    bf_err = abs(fwd.log_z - want_log_bf)
    cycle_err = abs(fwd.log_z + rev.log_z)

    assert not fwd.stopped and not rev.stopped
    assert bf_err < 0.06
    assert cycle_err < 0.06

    # Importance-weighted input cloud from an overdispersed proposal q(theta) = N(mean_1, 1.5 * cov_1)
    prop_cov = 1.5 * np.asarray(r1["cov"])
    prop_draws = rng.multivariate_normal(np.asarray(r1["mean"]), prop_cov, size=4096)
    log_q = stats.multivariate_normal.logpdf(
        prop_draws, mean=np.asarray(r1["mean"]), cov=prop_cov
    )
    log_p1 = np.asarray(jax.vmap(lambda th: logprior(th) + loglike_1(th))(jnp.asarray(prop_draws)))
    lw_imp = log_p1 - log_q
    lw_imp -= np.max(lw_imp)
    w_imp = np.exp(lw_imp)
    w_imp /= np.sum(w_imp)
    fwd_weighted = vsmc.bridge_smc(
        logprior,
        loglike_1,
        loglike_2,
        jnp.asarray(prop_draws),
        weights=jnp.asarray(w_imp),
        n_particles=1536,
        key=jax.random.PRNGKey(53),
        steps=6,
    )
    weighted_bf_err = abs(float(fwd_weighted.log_z) - want_log_bf)
    assert not fwd_weighted.stopped
    assert weighted_bf_err < 0.08

    record("bridge_log_bf_err", bf_err)
    record("bridge_weighted_bf_err", weighted_bf_err)
    record("bridge_cycle_err", cycle_err)


@pytest.mark.validates(
    "virgil.smc.tempered_smc",
    "virgil.smc.bridge_smc",
    roots=["mathematics", "statistics"],
    kind="control",
    headline="unnormalized_loglike_bias",
)
def test_smc_unnormalized_loglike_and_swapped_bridge_controls():
    """Control: shifting loglike by +2.5 nats (omitting Gaussian normalization)
    biases tempered_smc log_z by +2.5, and swapping bridge_smc endpoints flips
    the sign of log(Z_2 / Z_1)."""
    def sample_prior(key, n):
        return jax.random.normal(key, (n, 2), dtype=jnp.float64)

    def logprior(th):
        return -math.log(2.0 * math.pi) - 0.5 * jnp.sum(th**2)

    def unnorm_loglike(th):
        return 2.5 - math.log(2.0 * math.pi) - 0.5 * jnp.sum((th - 1.0) ** 2)

    exact = ref.gaussian_conjugate_tempering_ref(
        np.zeros(2), np.eye(2), np.eye(2), np.ones(2), np.eye(2), beta=1.0
    )
    res_bad = vsmc.tempered_smc(
        logprior,
        unnorm_loglike,
        sample_prior,
        key=jax.random.PRNGKey(7),
        n_particles=384,
        steps=3,
    )
    bias = abs(res_bad.log_z - float(exact["log_z"]))
    assert bias > 2.3

    rng = np.random.default_rng(77)
    draws_2 = jnp.asarray(rng.multivariate_normal([0.5, 0.5], 0.5 * np.eye(2), size=384))
    ll1 = lambda th: -0.5 * jnp.sum(th**2)
    ll2 = lambda th: -0.5 * jnp.sum((th - 1.0) ** 2) - 2.0
    swapped = vsmc.bridge_smc(
        logprior, ll2, ll1, draws_2, key=jax.random.PRNGKey(8), steps=3
    )
    # True ln(Z2/Z1) = -2.5, whereas swapped computes +2.5
    assert abs(swapped.log_z - (-2.5)) > 4.5
    record("unnormalized_loglike_bias", bias)


# ---------------------------------------------------------------------------
# 2. virgil.mode_search.*
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.mode_search.Mode",
    "virgil.mode_search.logmeanexp",
    "virgil.mode_search.laplace_log_evidence",
    "virgil.mode_search.distinct_modes",
    property="two_sided_dedup",
    roots=["mathematics"],
    headline="max_laplace_err",
)
def test_mode_search_laplace_logmeanexp_and_two_sided_distinct_modes():
    """laplace_log_evidence and logmeanexp match closed forms to machine
    precision, and distinct_modes enforces two-sided Mahalanobis deduplication
    at the documented threshold d^2 = 1.0 so broad covariances never swallow
    sharp needle modes."""
    vals = jnp.array([-850.0, -848.5, -852.1, -849.0], dtype=jnp.float64)
    lme_err = abs(float(vms.logmeanexp(vals)) - ref.logmeanexp_ref(np.asarray(vals)))
    assert lme_err < 1e-13
    assert math.isinf(float(vms.logmeanexp(jnp.array([-jnp.inf, -jnp.inf]))))

    # Broad mode A, sharp needle B inside A's 1-sigma ellipse, true duplicate C, anisotropic ridge D
    mode_a = vms.Mode(
        x=jnp.array([0.0, 0.0]), log_target=10.0, cov=100.0 * jnp.eye(2), n_inliers=5.0
    )
    mode_b = vms.Mode(
        x=jnp.array([3.0, 0.0]), log_target=9.0, cov=0.01 * jnp.eye(2), n_inliers=5.0
    )
    mode_c = vms.Mode(
        x=jnp.array([0.3, -0.2]), log_target=8.5, cov=80.0 * jnp.eye(2), n_inliers=4.0
    )
    mode_d = vms.Mode(
        x=jnp.array([0.0, 2.0]),
        log_target=8.0,
        cov=jnp.diag(jnp.array([25.0, 0.04])),
        n_inliers=5.0,
    )

    lap_errs = [
        abs(
            vms.laplace_log_evidence(m)
            - ref.laplace_log_evidence_ref(m.log_target, np.asarray(m.cov))
        )
        for m in (mode_a, mode_b, mode_c, mode_d)
    ]
    max_lap_err = max(lap_errs)
    assert max_lap_err < 1e-14

    candidates = [mode_c, mode_d, mode_b, mode_a]
    kept = vms.distinct_modes(candidates)
    want_idx = ref.distinct_modes_ref(
        [np.asarray(m.x) for m in candidates],
        [m.log_target for m in candidates],
        [np.asarray(m.cov) for m in candidates],
        mahal=1.0,
        two_sided=True,
    )
    assert len(kept) == len(want_idx) == 3
    for got_m, idx in zip(kept, want_idx, strict=True):
        np.testing.assert_allclose(np.asarray(got_m.x), np.asarray(candidates[idx].x), atol=1e-15)

    # Unit-covariance boundary check around documented squared-Mahalanobis threshold 1.0
    m_origin = vms.Mode(x=jnp.array([0.0, 0.0]), log_target=5.0, cov=jnp.eye(2), n_inliers=4.0)
    m_inside = vms.Mode(
        x=jnp.array([math.sqrt(0.94), 0.0]), log_target=4.0, cov=jnp.eye(2), n_inliers=4.0
    )
    m_outside = vms.Mode(
        x=jnp.array([math.sqrt(1.06), 0.0]), log_target=4.0, cov=jnp.eye(2), n_inliers=4.0
    )
    assert len(vms.distinct_modes([m_origin, m_inside])) == 1
    assert len(vms.distinct_modes([m_origin, m_outside])) == 2

    record("max_laplace_err", max_lap_err)
    record("logmeanexp_err", lme_err)


@pytest.mark.validates(
    "virgil.mode_search.distinct_modes",
    property="two_sided_dedup",
    roots=["mathematics"],
    kind="control",
    headline="swallowed_needle_modes",
)
def test_distinct_modes_one_sided_mahalanobis_control():
    """Control: a one-sided Mahalanobis rule under only the dominant broad mode's
    covariance swallows both sharp needle modes that distinct_modes preserves."""
    mode_a = vms.Mode(
        x=jnp.array([0.0, 0.0]), log_target=10.0, cov=100.0 * jnp.eye(2), n_inliers=5.0
    )
    mode_b = vms.Mode(
        x=jnp.array([3.0, 0.0]), log_target=9.0, cov=0.01 * jnp.eye(2), n_inliers=5.0
    )
    mode_d = vms.Mode(
        x=jnp.array([0.0, 2.0]),
        log_target=8.0,
        cov=jnp.diag(jnp.array([25.0, 0.04])),
        n_inliers=5.0,
    )
    candidates = [mode_b, mode_d, mode_a]
    kept_two_sided = vms.distinct_modes(candidates)
    kept_one_sided = ref.distinct_modes_ref(
        [np.asarray(m.x) for m in candidates],
        [m.log_target for m in candidates],
        [np.asarray(m.cov) for m in candidates],
        mahal=1.0,
        two_sided=False,
    )
    swallowed = len(kept_two_sided) - len(kept_one_sided)
    assert len(kept_two_sided) == 3 and len(kept_one_sided) == 1
    record("swallowed_needle_modes", float(swallowed))


@pytest.mark.validates(
    "virgil.mode_search.DefensiveMixture",
    roots=["mathematics", "statistics"],
    headline="max_log_q_err",
)
def test_defensive_mixture_log_density_symmetry_and_sampling():
    """DefensiveMixture matches SciPy Student-t + prior mixture log-density to
    machine precision, normalizes to 1, honours symmetry images, and falls back
    to the pure prior when modes is empty."""
    cov0 = 4.0 * np.eye(2)
    lp_const = -math.log(2.0 * math.pi) - math.log(4.0)

    def log_prior_j(x):
        return lp_const - 0.5 * jnp.sum(x**2, axis=-1) / 4.0

    def sample_prior_j(key, n):
        return 2.0 * jax.random.normal(key, (n, 2), dtype=jnp.float64)

    m1 = vms.Mode(
        x=jnp.array([1.4, -0.8]),
        log_target=-1.2,
        cov=jnp.array([[0.20, 0.04], [0.04, 0.12]]),
        n_inliers=4.0,
    )
    m2 = vms.Mode(
        x=jnp.array([-0.6, 1.7]),
        log_target=-1.9,
        cov=jnp.array([[0.15, -0.03], [-0.03, 0.25]]),
        n_inliers=4.0,
    )
    inv_img = jnp.diag(jnp.array([-1.0, -1.0]))
    dm = vms.DefensiveMixture(
        [m1, m2], log_prior_j, sample_prior_j, alpha=0.15, df=5.0, images=[inv_img]
    )

    rng = np.random.default_rng(604)
    test_pts = rng.uniform(-3.5, 3.5, size=(40, 2))
    eval_lq = jax.vmap(dm.log_q)
    got_lq = np.asarray(eval_lq(jnp.asarray(test_pts)))
    want_lq = ref.defensive_mixture_log_q_ref(
        test_pts,
        [np.asarray(m1.x), np.asarray(m2.x)],
        [m1.log_target, m2.log_target],
        [np.asarray(m1.cov), np.asarray(m2.cov)],
        lambda pts: stats.multivariate_normal.logpdf(pts, mean=np.zeros(2), cov=cov0),
        alpha=0.15,
        df=5.0,
        images=[-np.eye(2)],
    )
    max_lq_err = float(np.max(np.abs(got_lq - want_lq)))
    sym_err = float(np.max(np.abs(got_lq - np.asarray(eval_lq(jnp.asarray(-test_pts))))))
    assert max_lq_err < 1e-13
    assert sym_err < 1e-14

    # Empty modes falls back to alpha = 1 pure prior
    dm_empty = vms.DefensiveMixture([], log_prior_j, sample_prior_j, dim=2)
    np.testing.assert_allclose(
        np.asarray(jax.vmap(dm_empty.log_q)(jnp.asarray(test_pts))),
        np.asarray(log_prior_j(jnp.asarray(test_pts))),
        atol=1e-14,
    )

    # Empirical sample distribution agrees with numerical marginal CDF (KS test)
    draws = np.asarray(dm.sample(jax.random.PRNGKey(605), 3000))
    xs_grid = np.linspace(-10.0, 10.0, 251)
    ys_grid = np.linspace(-10.0, 10.0, 251)
    xx, yy = np.meshgrid(xs_grid, ys_grid, indexing="ij")
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
    pdf_2d = np.exp(np.asarray(eval_lq(jnp.asarray(grid_pts)))).reshape(xx.shape)
    marg_x = integrate.simpson(pdf_2d, x=ys_grid, axis=1)
    norm_2d = float(integrate.simpson(marg_x, x=xs_grid))
    cdf_x = integrate.cumulative_trapezoid(marg_x / norm_2d, xs_grid, initial=0.0)
    ks_res = stats.kstest(draws[:, 0], lambda v: np.interp(v, xs_grid, cdf_x))
    assert abs(norm_2d - 1.0) < 2e-3
    assert ks_res.pvalue > 1e-3

    record("max_log_q_err", max_lq_err)
    record("symmetry_err", sym_err)
    record("sample_ks_pvalue", float(ks_res.pvalue))


@pytest.mark.validates(
    "virgil.mode_search.importance_evidence",
    "virgil.mode_search.Evidence",
    "virgil.mode_search.resample_move",
    "virgil.mode_search.Move",
    property=("is_estimator", "ess_floor"),
    roots=["mathematics", "statistics"],
    headline="is_log_z_err",
)
def test_importance_evidence_floor_sentinel_and_resample_move():
    """importance_evidence matches exact bimodal log Z, Kish ESS, and delta-method
    SE, fires at_floor when a sharp multi-night mode is missed by the draw set,
    and resample_move rejuvenates duplicates while preserving posterior moments."""
    mu0 = np.zeros(2)
    cov0 = 9.0 * np.eye(2)
    m1 = np.array([-2.0, -1.2])
    m2 = np.array([2.1, 1.5])
    s1 = np.array([[0.22, 0.04], [0.04, 0.18]])
    s2 = np.array([[0.16, -0.03], [-0.03, 0.24]])
    log_scales = [0.9, 0.3]

    exact = ref.bimodal_gaussian_mixture_ref(mu0, cov0, [m1, m2], [s1, s2], log_scales)
    prec1, prec2 = jnp.asarray(linalg.inv(s1)), jnp.asarray(linalg.inv(s2))
    _, ld1 = np.linalg.slogdet(s1)
    _, ld2 = np.linalg.slogdet(s2)
    c1 = log_scales[0] - math.log(2.0 * math.pi) - 0.5 * float(ld1)
    c2 = log_scales[1] - math.log(2.0 * math.pi) - 0.5 * float(ld2)
    m1_j, m2_j = jnp.asarray(m1), jnp.asarray(m2)
    lp_const = -math.log(2.0 * math.pi) - math.log(9.0)

    def log_prior(x):
        return lp_const - 0.5 * jnp.sum(x**2, axis=-1) / 9.0

    def sample_prior(key, n):
        return 3.0 * jax.random.normal(key, (n, 2), dtype=jnp.float64)

    def loglike_fn(x, theta):
        d1 = x - (m1_j + theta)
        d2 = x - (m2_j + theta)
        ll1 = c1 - 0.5 * jnp.einsum("i,ij,j->", d1, prec1, d1)
        ll2 = c2 - 0.5 * jnp.einsum("i,ij,j->", d2, prec2, d2)
        return jnp.logaddexp(ll1, ll2)

    def log_target(x, theta):
        return log_prior(x) + loglike_fn(x, theta)

    theta0 = jnp.zeros(2, dtype=jnp.float64)
    vm1 = vms.Mode(
        x=jnp.asarray(exact["comp_means"][0]),
        log_target=float(log_target(jnp.asarray(exact["comp_means"][0]), theta0)),
        cov=jnp.asarray(exact["comp_covs"][0]),
        n_inliers=5.0,
    )
    vm2 = vms.Mode(
        x=jnp.asarray(exact["comp_means"][1]),
        log_target=float(log_target(jnp.asarray(exact["comp_means"][1]), theta0)),
        cov=jnp.asarray(exact["comp_covs"][1]),
        n_inliers=5.0,
    )
    dm = vms.DefensiveMixture([vm1, vm2], log_prior, sample_prior, alpha=0.1, df=5.0)

    ev = vms.importance_evidence(
        log_target,
        theta0,
        dm,
        jax.random.PRNGKey(701),
        n=3000,
        inliers_fn=lambda x, th: jnp.asarray(5.0, dtype=jnp.float64),
        log_floor=-20.0,
    )
    assert isinstance(ev, vms.Evidence)
    assert ev.at_floor is False
    assert ev.undersampled is False
    assert ev.n_nonfinite == 0
    assert ev.log_q is not None
    np.testing.assert_allclose(
        np.asarray(ev.log_q), np.asarray(jax.vmap(dm.log_q)(ev.x)), atol=1e-13
    )
    diag_ref = ref.importance_diagnostics_ref(np.asarray(ev.log_w))
    is_lz_err = abs(ev.log_z - float(exact["log_z"]))
    ess_formula_err = abs(ev.ess - diag_ref["ess"])
    se_formula_err = abs(ev.log_z_error - diag_ref["log_z_error"])
    assert is_lz_err < max(3.5 * ev.log_z_error, 0.03)
    assert ess_formula_err < 1e-10
    assert se_formula_err < 1e-13

    # undersampled flag when draw count guarantees ESS < ESS_TARGET (50)
    inliers_5 = lambda x, th: jnp.asarray(5.0, dtype=jnp.float64)
    ev_under = vms.importance_evidence(
        log_target,
        theta0,
        dm,
        jax.random.PRNGKey(706),
        n=16,
        inliers_fn=inliers_5,
        log_floor=-20.0,
    )
    assert ev_under.undersampled is True

    # Non-finite target evaluations (NaN and +inf) are counted in n_nonfinite and masked to -inf
    def partly_nonfinite_target(x, theta):
        base = log_target(x, theta)
        return jnp.where(x[0] > 2.4, jnp.nan, jnp.where(x[0] < -2.4, jnp.inf, base))

    ev_nf = vms.importance_evidence(
        partly_nonfinite_target,
        theta0,
        dm,
        jax.random.PRNGKey(707),
        n=512,
        inliers_fn=inliers_5,
        log_floor=-20.0,
    )
    assert ev_nf.n_nonfinite > 0
    assert math.isfinite(float(ev_nf.log_z))

    # Sentinel check: if weighted mean n_inliers < M_MIN while a mode has Laplace ln Z > log_floor + W
    ev_floor = vms.importance_evidence(
        log_target,
        theta0,
        dm,
        jax.random.PRNGKey(702),
        n=256,
        inliers_fn=lambda x, th: jnp.asarray(0.5, dtype=jnp.float64),
        log_floor=-20.0,
    )
    assert ev_floor.at_floor is True

    # resample_move preserves bimodal target moments and rejuvenates duplicates
    mv = vms.resample_move(
        log_target, theta0, ev, jax.random.PRNGKey(703), n_draws=600, n_moves=5
    )
    assert isinstance(mv, vms.Move)
    assert mv.draws.shape == (600, 2)
    assert mv.stuck is False
    assert 0.4 < mv.acceptance <= 1.0
    mv_pts = np.asarray(mv.draws)
    mv_mean_err = float(linalg.norm(np.mean(mv_pts, axis=0) - np.asarray(exact["mean"])))
    mv_w2_err = abs(float(np.mean(mv_pts[:, 0] > 0.0)) - float(exact["weights"][1]))
    assert mv_mean_err < 0.25
    assert mv_w2_err < 0.07

    # Explicit HMC mass-covariance override cov=...
    mv_cov = vms.resample_move(
        log_target,
        theta0,
        ev,
        jax.random.PRNGKey(705),
        n_draws=400,
        n_moves=5,
        cov=0.2 * jnp.eye(2, dtype=jnp.float64),
    )
    assert mv_cov.stuck is False
    assert 0.2 < mv_cov.acceptance <= 1.0
    assert float(linalg.norm(np.mean(np.asarray(mv_cov.draws), axis=0) - np.asarray(exact["mean"]))) < 0.35

    mv_zero = vms.resample_move(
        log_target, theta0, ev, jax.random.PRNGKey(704), n_draws=64, n_moves=0
    )
    assert mv_zero.stuck is True

    record("is_log_z_err", is_lz_err)
    record("ess_formula_err", ess_formula_err)
    record("move_mean_err", mv_mean_err)


# ---------------------------------------------------------------------------
# 3. virgil.tempering.* (WF3)
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.tempering.estimate_factors",
    "virgil.tempering.ChannelFactors",
    "virgil.tempering.block_dof",
    "virgil.tempering.out_of_range",
    "virgil.tempering.flat_signature",
    property=("variance_ratio", "fallback_rules"),
    roots=["mathematics"],
    headline="max_factor_err",
)
def test_tempering_exact_variance_ratio_floor_fallback_and_range_flags():
    """estimate_factors, ChannelFactors, block_dof, out_of_range, and
    flat_signature reproduce exact finite-chain variance ratios, MIN_CHAINS
    pooling fallback, unit-flooring of anticorrelated chains, and lag-2 flags."""
    rng = np.random.default_rng(811)
    # 12 chains of length 6 for 'vis' (>= MIN_CHAINS=10), 6 chains of length 5 for 'phi' (< MIN_CHAINS)
    vis_chains_z = [rng.normal(0.0, 1.0, size=6) + 0.6 * rng.normal() for _ in range(12)]
    phi_chains_z = [rng.normal(0.0, 1.0, size=5) for _ in range(6)]
    z_vis = np.concatenate(vis_chains_z)
    z_phi = np.concatenate(phi_chains_z)
    vis_idx = [np.arange(6 * i, 6 * (i + 1)) for i in range(12)]
    phi_idx = [np.arange(5 * i, 5 * (i + 1)) for i in range(6)]

    cf = vtemp.estimate_factors(
        [{"vis": z_vis, "phi": z_phi}],
        [{"vis": vis_idx, "phi": phi_idx}],
        names=("vis", "phi", "extra[0]"),
    )
    assert isinstance(cf, vtemp.ChannelFactors)

    want_vis_raw, want_vis_f = ref.finite_chain_variance_ratio_ref(vis_chains_z)
    want_pool_raw, want_pool_f = ref.finite_chain_variance_ratio_ref(vis_chains_z + phi_chains_z)
    want_rho1, want_np1 = ref.chain_lag_correlation_ref(vis_chains_z, lag=1)
    want_rho2, want_np2 = ref.chain_lag_correlation_ref(vis_chains_z, lag=2)

    err_vis_f = abs(cf.blocks["vis"] - want_vis_f)
    err_pool_f = abs(cf.f - want_pool_f)
    err_rho1 = abs(cf.rho["vis"] - want_rho1)
    err_rho2 = abs(cf.rho2["vis"] - want_rho2)
    assert err_vis_f < 1e-14
    assert err_pool_f < 1e-14
    assert err_rho1 < 1e-14 and cf.n_pairs["vis"] == want_np1
    assert err_rho2 < 1e-14 and cf.n_pairs2["vis"] == want_np2

    # phi (< MIN_CHAINS) and extra[0] (no chains) fall back to pooled f
    assert set(cf.fallback) == {"phi", "extra[0]"}
    assert cf.blocks["phi"] == pytest.approx(cf.f, abs=1e-15)
    assert cf.blocks["extra[0]"] == pytest.approx(cf.f, abs=1e-15)

    # Multi-dataset input (len == 2): splitting 12 vis chains (6 + 6) and 6 phi chains (3 + 3)
    # across two datasets matches the pooled single-dataset result identically
    cf_multi = vtemp.estimate_factors(
        [
            {"vis": np.concatenate(vis_chains_z[:6]), "phi": np.concatenate(phi_chains_z[:3])},
            {"vis": np.concatenate(vis_chains_z[6:]), "phi": np.concatenate(phi_chains_z[3:])},
        ],
        [
            {
                "vis": [np.arange(6 * i, 6 * (i + 1)) for i in range(6)],
                "phi": [np.arange(5 * i, 5 * (i + 1)) for i in range(3)],
            },
            {
                "vis": [np.arange(6 * i, 6 * (i + 1)) for i in range(6)],
                "phi": [np.arange(5 * i, 5 * (i + 1)) for i in range(3)],
            },
        ],
        names=("vis", "phi", "extra[0]"),
    )
    assert cf_multi.blocks["vis"] == pytest.approx(cf.blocks["vis"], abs=1e-15)
    assert cf_multi.f == pytest.approx(cf.f, abs=1e-15)
    assert cf_multi.f_raw == pytest.approx(cf.f_raw, abs=1e-15)
    assert cf_multi.n_pairs == cf.n_pairs and cf_multi.n_pairs2 == cf.n_pairs2
    assert set(cf_multi.fallback) == {"phi", "extra[0]"}

    # All-blocks-short (< MIN_CHAINS = 10 each): both blocks fall back to pooled f,
    # and single-wavelength / zero-chain input leaves tempering off (f = f_raw = 1.0)
    want_short_raw, want_short_f = ref.finite_chain_variance_ratio_ref(
        vis_chains_z[:4] + phi_chains_z[:3]
    )
    cf_short = vtemp.estimate_factors(
        [{"vis": np.concatenate(vis_chains_z[:4]), "phi": np.concatenate(phi_chains_z[:3])}],
        [
            {
                "vis": [np.arange(6 * i, 6 * (i + 1)) for i in range(4)],
                "phi": [np.arange(5 * i, 5 * (i + 1)) for i in range(3)],
            }
        ],
    )
    assert set(cf_short.fallback) == {"vis", "phi"}
    assert set(cf_short.raw.keys()) == {"pooled"}
    assert cf_short.f == pytest.approx(want_short_f, abs=1e-14)
    assert cf_short.f_raw == pytest.approx(want_short_raw, abs=1e-14)
    assert cf_short.blocks["vis"] == pytest.approx(cf_short.f, abs=1e-15)
    assert cf_short.blocks["phi"] == pytest.approx(cf_short.f, abs=1e-15)

    cf_zero = vtemp.estimate_factors(
        [{"vis": np.zeros(4), "phi": np.zeros(2)}],
        [{"vis": [], "phi": []}],
    )
    assert cf_zero.f == pytest.approx(1.0, abs=1e-15)
    assert cf_zero.f_raw == pytest.approx(1.0, abs=1e-15)
    assert cf_zero.blocks == {"vis": 1.0, "phi": 1.0}
    assert math.isnan(cf_zero.rho["pooled"])

    # block_dof and ChannelFactors.block_dof invert max(f_b, 1) and never sharpen (> 1)
    got_dof = vtemp.block_dof({"vis": cf.blocks["vis"], "anticorr": 0.4})
    want_dof = ref.block_dof_ref({"vis": cf.blocks["vis"], "anticorr": 0.4})
    assert got_dof == pytest.approx(want_dof, abs=1e-15)
    assert cf.block_dof() == pytest.approx(ref.block_dof_ref(cf.blocks), abs=1e-15)

    # Anticorrelated alternating chains: f_raw = 0 < 0.5 -> floored to f = 1.0, out_of_range True
    alt_z = np.tile(np.array([1.0, -1.0, 1.0, -1.0]), 12)
    alt_idx = [np.arange(4 * i, 4 * (i + 1)) for i in range(12)]
    cf_alt = vtemp.estimate_factors([{"vis": alt_z}], [{"vis": alt_idx}])
    assert cf_alt.f_raw == pytest.approx(0.0, abs=1e-15)
    assert cf_alt.f == pytest.approx(1.0, abs=1e-15)
    assert vtemp.out_of_range(cf_alt.f, cf_alt.f_raw) is True
    assert vtemp.out_of_range(cf_alt.f) is False  # floored f=1.0 alone is inside [0.5, 2.0]
    assert vtemp.out_of_range(0.4) is True  # single-argument f_raw=None checks f against [0.5, 2.0]
    assert vtemp.out_of_range(2.2) is True
    assert vtemp.out_of_range(1.4) is False
    assert vtemp.out_of_range(1.0, 0.4) is True
    assert vtemp.out_of_range(2.2, 2.2) is True
    assert vtemp.out_of_range(1.4, 1.4) is False

    # Direct boundary test of flat_signature
    assert vtemp.flat_signature(0.4, 0.50, 400) is ref.flat_signature_ref(0.4, 0.50, 400)
    assert vtemp.flat_signature(0.7, 0.52, 400) is False  # AR(1): 0.52 - 0.49 = 0.03 < 0.15
    assert vtemp.flat_signature(0.4, 0.50, 20) is False  # 3 / sqrt(20) = 0.67 > 0.34

    max_err = max(err_vis_f, err_pool_f, err_rho1, err_rho2)
    record("max_factor_err", max_err)


@pytest.mark.validates(
    "virgil.tempering.estimate_factors",
    "virgil.tempering.flat_signature",
    "virgil.tempering.channel_chains",
    "virgil.tempering.adjacent_channels",
    property="variance_ratio",
    roots=["mathematics", "statistics"],
    headline="ar1_finite_chain_err",
)
def test_tempering_finite_ar1_common_mode_and_oidata_channel_chains(tmp_path):
    """Stationary AR(1) and equicorrelated common-mode ensembles follow the
    exact finite-n laws 1 + 2 sum (1 - k/n) rho^k and 1 + (n - 1) c, and
    channel_chains / adjacent_channels partition multi-wavelength OIData by
    baseline and snapshot without crossing boundaries."""
    rng = np.random.default_rng(912)
    n_len, rho, n_chains = 6, 0.70, 4000
    idx = np.arange(n_len)
    chol_ar1 = linalg.cholesky(rho ** np.abs(idx[:, None] - idx[None, :]), lower=True)
    ar1_mat = rng.normal(size=(n_chains, n_len)) @ chol_ar1.T
    ar1_idx = [np.arange(n_len * i, n_len * (i + 1)) for i in range(n_chains)]

    cf_ar1 = vtemp.estimate_factors([{"vis": ar1_mat.ravel()}], [{"vis": ar1_idx}])
    want_ar1 = ref.ar1_finite_chain_factor(n_len, rho)
    ar1_err = abs(cf_ar1.blocks["vis"] - want_ar1)
    assert ar1_err < 0.12
    assert cf_ar1.flat == ()

    # Common-mode offset model: z_{c,i} = sqrt(c) * b_c + sqrt(1 - c) * eps_{c,i}
    n_cm, c_val, n_cm_chains = 8, 0.55, 3000
    b_offset = rng.normal(size=(n_cm_chains, 1))
    cm_mat = math.sqrt(c_val) * b_offset + math.sqrt(1.0 - c_val) * rng.normal(
        size=(n_cm_chains, n_cm)
    )
    cm_idx = [np.arange(n_cm * i, n_cm * (i + 1)) for i in range(n_cm_chains)]
    cf_cm = vtemp.estimate_factors([{"vis": cm_mat.ravel()}], [{"vis": cm_idx}])
    want_cm = ref.common_mode_chain_factor(n_cm, c_val)
    cm_err = abs(cf_cm.blocks["vis"] - want_cm)
    assert cm_err < 0.20
    assert "vis" in cf_cm.flat

    # Multi-wavelength OIFITS layout check for channel_chains and adjacent_channels
    uts4 = np.array(
        [[-9.925, -20.335], [14.887, 30.502], [44.915, 66.183], [103.306, 43.999]]
    )
    wavels = np.linspace(2.0e-6, 2.4e-6, 5)
    n_bl = 3 * 6  # 3 snapshots * 6 baselines
    n_tri = 3 * 4  # 3 snapshots * 4 triangles
    fits_path = tmp_path / "tempering_5ch.fits"
    xsim.observe(
        fits_path,
        lambda u, v, wl: np.ones(np.broadcast(u, v, wl).shape, dtype=np.complex128),
        uts4,
        hour_angles_h=[-1.0, 0.0, 1.0],
        wavelengths=wavels,
        sigma_v2=0.01,
        sigma_cp_deg=0.5,
    )
    oid = OIData(read_oifits(str(fits_path)))
    chains = vtemp.channel_chains(oid)
    adj = vtemp.adjacent_channels(oid)

    assert len(chains["vis"]) == n_bl and all(len(c) == 5 for c in chains["vis"])
    assert len(chains["phi"]) == n_tri and all(len(c) == 5 for c in chains["phi"])
    a_vis, b_vis = adj["vis"]
    a_phi, b_phi = adj["phi"]
    assert a_vis.size == b_vis.size == n_bl * 4
    assert a_phi.size == b_phi.size == n_tri * 4

    # Visibility chains & adjacent pairs: index into the unflagged OI_VIS2 rows via oid.vis_index
    u_m_data = np.asarray(oid.u)
    v_m_data = np.asarray(oid.v)
    mjd_data = np.asarray(oid.mjd)
    wl_data = np.asarray(oid.wavel)
    sta_data = np.asarray(oid.stations)
    vis_sel = slice(None) if oid.vis_index is None else np.asarray(oid.vis_index)
    u_vis = u_m_data[vis_sel]
    v_vis = v_m_data[vis_sel]
    mjd_vis = mjd_data[vis_sel]
    wl_vis = wl_data[vis_sel]
    sta_vis = sta_data[vis_sel]
    max_bl_jump = float(
        np.max(np.hypot(u_vis[a_vis] - u_vis[b_vis], v_vis[a_vis] - v_vis[b_vis]))
    )
    assert max_bl_jump < 1e-6
    assert np.array_equal(sta_vis[a_vis], sta_vis[b_vis])
    assert float(np.max(np.abs(mjd_vis[a_vis] - mjd_vis[b_vis]))) == 0.0
    assert bool(np.all(wl_vis[a_vis] < wl_vis[b_vis]))
    for c_vis in chains["vis"]:
        assert np.all(mjd_vis[c_vis] == mjd_vis[c_vis[0]])
        assert np.all(sta_vis[c_vis] == sta_vis[c_vis[0]])
        assert bool(np.all(np.diff(wl_vis[c_vis]) > 0.0))

    # Reversed-wavelength OIFITS with flagged OI_VIS2 rows: oid.vis_index is a non-identity
    # integer index array and channel_chains / adjacent_channels sort into ascending wavelength
    fits_rev = tmp_path / "tempering_rev_flagged.fits"
    xsim.observe(
        fits_rev,
        lambda u, v, wl: np.ones(np.broadcast(u, v, wl).shape, dtype=np.complex128),
        uts4,
        hour_angles_h=[-1.0, 1.0],
        wavelengths=wavels[::-1],
        sigma_v2=0.01,
        sigma_cp_deg=0.5,
    )
    with fits.open(fits_rev, mode="update") as hdul:
        hdul["OI_VIS2"].data["FLAG"][0, :] = True
        hdul["OI_VIS2"].data["FLAG"][3, 2] = True
        hdul.flush()
    oid_rev = OIData(read_oifits(str(fits_rev)))
    assert oid_rev.vis_index is not None
    vis_idx_rev = np.asarray(oid_rev.vis_index)
    assert vis_idx_rev.size < np.asarray(oid_rev.u).size
    assert not np.array_equal(vis_idx_rev, np.arange(vis_idx_rev.size))
    chains_rev = vtemp.channel_chains(oid_rev)
    adj_rev = vtemp.adjacent_channels(oid_rev)
    u_rev = np.asarray(oid_rev.u)[vis_idx_rev]
    v_rev = np.asarray(oid_rev.v)[vis_idx_rev]
    mjd_rev = np.asarray(oid_rev.mjd)[vis_idx_rev]
    wl_rev = np.asarray(oid_rev.wavel)[vis_idx_rev]
    sta_rev = np.asarray(oid_rev.stations)[vis_idx_rev]
    ar_vis, br_vis = adj_rev["vis"]
    assert float(np.max(np.hypot(u_rev[ar_vis] - u_rev[br_vis], v_rev[ar_vis] - v_rev[br_vis]))) < 1e-6
    assert np.array_equal(sta_rev[ar_vis], sta_rev[br_vis])
    assert float(np.max(np.abs(mjd_rev[ar_vis] - mjd_rev[br_vis]))) == 0.0
    assert bool(np.all(wl_rev[ar_vis] < wl_rev[br_vis]))
    for c_rev in chains_rev["vis"]:
        assert bool(np.all(np.diff(wl_rev[c_rev]) > 0.0))
    wl_phi_rev = np.asarray(oid_rev.wavel)[np.asarray(oid_rev.i_cps1)]
    ar_phi, br_phi = adj_rev["phi"]
    assert bool(np.all(wl_phi_rev[ar_phi] < wl_phi_rev[br_phi]))

    # Closure-phase chains & adjacent pairs: identical triangle legs (u1, v1, u2, v2, stations), snapshot MJD, and strictly increasing wavelength
    i1 = np.asarray(oid.i_cps1)
    i2 = np.asarray(oid.i_cps2)
    i3 = np.asarray(oid.i_cps3)
    tri_coords = np.column_stack(
        [u_m_data[i1], v_m_data[i1], u_m_data[i2], v_m_data[i2], u_m_data[i3], v_m_data[i3]]
    )
    tri_sta = np.column_stack([sta_data[i1], sta_data[i2], sta_data[i3]])
    tri_mjd = mjd_data[i1]
    tri_wl = wl_data[i1]
    assert float(np.max(np.abs(mjd_data[i1] - mjd_data[i2]))) == 0.0
    assert float(np.max(np.abs(mjd_data[i1] - mjd_data[i3]))) == 0.0
    max_tri_jump = float(np.max(np.linalg.norm(tri_coords[a_phi] - tri_coords[b_phi], axis=1)))
    assert max_tri_jump < 1e-6
    assert np.array_equal(tri_sta[a_phi], tri_sta[b_phi])
    assert float(np.max(np.abs(tri_mjd[a_phi] - tri_mjd[b_phi]))) == 0.0
    assert bool(np.all(tri_wl[a_phi] < tri_wl[b_phi]))
    for c_phi in chains["phi"]:
        assert np.all(tri_mjd[c_phi] == tri_mjd[c_phi[0]])
        assert np.all(tri_sta[c_phi] == tri_sta[c_phi[0]])
        assert bool(np.all(np.diff(tri_wl[c_phi]) > 0.0))

    record("ar1_finite_chain_err", ar1_err)
    record("common_mode_chain_err", cm_err)
    record("max_baseline_jump_m", max_bl_jump)


@pytest.mark.validates(
    "virgil.tempering.estimate_factors",
    property="variance_ratio",
    roots=["mathematics", "statistics"],
    kind="control",
    headline="asymptotic_bartlett_overprediction",
)
def test_tempering_rejects_asymptotic_bartlett_formula_on_short_chains():
    """Control (WF3): on short spectral chains (n = 6, rho = 0.70), the
    infinite-chain Bartlett formula (1 + rho)/(1 - rho) = 5.667 overpredicts
    estimate_factors by > 2.2."""
    rng = np.random.default_rng(999)
    n_len, rho, n_chains = 6, 0.70, 800
    idx = np.arange(n_len)
    chol = linalg.cholesky(rho ** np.abs(idx[:, None] - idx[None, :]), lower=True)
    z_mat = rng.normal(size=(n_chains, n_len)) @ chol.T
    ch_idx = [np.arange(n_len * i, n_len * (i + 1)) for i in range(n_chains)]

    cf = vtemp.estimate_factors([{"vis": z_mat.ravel()}], [{"vis": ch_idx}])
    asymptotic = ref.ar1_asymptotic_factor(rho)
    gap = asymptotic - cf.blocks["vis"]
    assert gap > 2.2
    record("asymptotic_bartlett_overprediction", gap)
