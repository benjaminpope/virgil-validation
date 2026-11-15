"""First-principles statistical and mathematical reference implementations for
sequential Monte Carlo, mode-search defensive importance sampling, spectral
channel tempering, Jeffreys outlier mixtures, linearized flux Bayes factors,
Gauss-Newton unconstrained precision, and K-fold ridge cross-validation.

This module belongs to ``crosscheck`` and shares nothing with ``virgil``: it
uses only the Python standard library, NumPy, and SciPy. Published statistical
and mathematical identities cite their literature sources in each docstring;
helpers that encode ``virgil``'s documented internal diagnostic or deduplication
specifications (``distinct_modes_ref``, ``chain_lag_correlation_ref``,
``flat_signature_ref``, and ``out_of_range_ref``) explicitly state that they
implement ``virgil``'s documented API specification rather than an external
published theorem.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
from scipy import integrate, linalg, special, stats


# ---------------------------------------------------------------------------
# 1. Conjugate Gaussian & bimodal Gaussian-mixture evidence (SMC references)
# ---------------------------------------------------------------------------


def gaussian_conjugate_tempering_ref(
    mu0: np.ndarray,
    cov0: np.ndarray,
    design: np.ndarray,
    y: np.ndarray,
    cov_y: np.ndarray,
    beta: float = 1.0,
) -> dict[str, np.ndarray | float]:
    r"""Exact posterior mean, covariance, and log normalizing constant
    :math:`\ln Z(\beta) = \ln \int p_0(\theta)\,L(\theta)^\beta\,\mathrm{d}\theta`
    for a linear-Gaussian model with prior :math:`\theta \sim \mathcal{N}(\mu_0, \Sigma_0)`
    and normalized likelihood :math:`y \mid \theta \sim \mathcal{N}(A\theta, \Sigma_y)`
    at inverse temperature :math:`\beta \in [0, 1]`.

    Completing the square (Gelman et al. 2013, *Bayesian Data Analysis*, 3rd ed.,
    sec. 3.5; Rasmussen & Williams 2006, *Gaussian Processes for Machine Learning*,
    Appendix A.2) gives posterior precision
    :math:`\Lambda_\beta = \Sigma_0^{-1} + \beta A^\top \Sigma_y^{-1} A`,
    covariance :math:`\Sigma_\beta = \Lambda_\beta^{-1}`, mean
    :math:`\mu_\beta = \Sigma_\beta(\Sigma_0^{-1}\mu_0 + \beta A^\top \Sigma_y^{-1} y)`,
    and exact tempered log marginal likelihood

    .. math::

        \ln Z(\beta) = -\frac{\beta m}{2}\ln(2\pi)
        - \frac{\beta}{2}\ln|\Sigma_y| - \frac{1}{2}\ln|\Sigma_0|
        + \frac{1}{2}\ln|\Sigma_\beta|
        - \frac{1}{2}\!\left(
            \mu_0^\top \Sigma_0^{-1}\mu_0
            + \beta y^\top \Sigma_y^{-1} y
            - \mu_\beta^\top \Lambda_\beta \mu_\beta
          \right).
    """
    mu0 = np.asarray(mu0, dtype=np.float64)
    cov0 = np.asarray(cov0, dtype=np.float64)
    design = np.asarray(design, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    cov_y = np.asarray(cov_y, dtype=np.float64)
    beta = float(beta)

    m = y.shape[0]
    prec0 = linalg.inv(cov0)
    prec_y = linalg.inv(cov_y)

    prec_beta = prec0 + beta * (design.T @ prec_y @ design)
    cov_beta = linalg.inv(prec_beta)
    rhs = prec0 @ mu0 + beta * (design.T @ (prec_y @ y))
    mu_beta = cov_beta @ rhs

    sign0, logdet0 = np.linalg.slogdet(cov0)
    sign_y, logdet_y = np.linalg.slogdet(cov_y)
    sign_b, logdet_b = np.linalg.slogdet(cov_beta)
    if sign0 <= 0 or sign_y <= 0 or sign_b <= 0:
        raise ValueError("Covariances must be symmetric positive definite.")

    quad = (
        float(mu0 @ prec0 @ mu0)
        + beta * float(y @ prec_y @ y)
        - float(mu_beta @ prec_beta @ mu_beta)
    )
    log_z = (
        -0.5 * beta * m * math.log(2.0 * math.pi)
        - 0.5 * beta * float(logdet_y)
        - 0.5 * float(logdet0)
        + 0.5 * float(logdet_b)
        - 0.5 * quad
    )
    return {"mean": mu_beta, "cov": cov_beta, "log_z": float(log_z)}


def bimodal_gaussian_mixture_ref(
    mu0: np.ndarray,
    cov0: np.ndarray,
    mode_means: Sequence[np.ndarray],
    mode_covs: Sequence[np.ndarray],
    log_scales: Sequence[float],
) -> dict[str, np.ndarray | float]:
    r"""Exact log marginal likelihood, mode responsibilities, posterior mean,
    and posterior covariance for a Gaussian prior :math:`\mathcal{N}(\mu_0, \Sigma_0)`
    multiplied by a multimodal Gaussian-sum likelihood

    .. math::

        L(\theta) = \sum_{k=1}^K \exp(s_k)\,\mathcal{N}(\theta; m_k, S_k).

    By the Gaussian product identity (Petersen & Pedersen 2012, *The Matrix
    Cookbook*, sec. 8.1.8), each component integrates in closed form to
    :math:`Z_k = \exp(s_k)\,\mathcal{N}(\mu_0; m_k, \Sigma_0 + S_k)` with
    component posterior covariance :math:`V_k = (\Sigma_0^{-1} + S_k^{-1})^{-1}`
    and mean :math:`\nu_k = V_k(\Sigma_0^{-1}\mu_0 + S_k^{-1} m_k)`.
    """
    mu0 = np.asarray(mu0, dtype=np.float64)
    cov0 = np.asarray(cov0, dtype=np.float64)
    prec0 = linalg.inv(cov0)

    log_zk = []
    comp_means = []
    comp_covs = []
    for mk, sk, ls in zip(mode_means, mode_covs, log_scales, strict=True):
        mk_a = np.asarray(mk, dtype=np.float64)
        sk_a = np.asarray(sk, dtype=np.float64)
        preck = linalg.inv(sk_a)
        vk = linalg.inv(prec0 + preck)
        nuk = vk @ (prec0 @ mu0 + preck @ mk_a)
        lzk = float(ls) + float(
            stats.multivariate_normal.logpdf(mu0, mean=mk_a, cov=cov0 + sk_a)
        )
        log_zk.append(lzk)
        comp_means.append(nuk)
        comp_covs.append(vk)

    log_zk_arr = np.asarray(log_zk, dtype=np.float64)
    log_z = float(special.logsumexp(log_zk_arr))
    weights = np.exp(log_zk_arr - log_z)

    means_arr = np.stack(comp_means, axis=0)
    covs_arr = np.stack(comp_covs, axis=0)
    post_mean = np.sum(weights[:, None] * means_arr, axis=0)
    second_moment = np.zeros_like(cov0)
    for wk, nuk, vk in zip(weights, comp_means, comp_covs, strict=True):
        second_moment += wk * (vk + np.outer(nuk, nuk))
    post_cov = second_moment - np.outer(post_mean, post_mean)
    return {
        "log_z": log_z,
        "weights": weights,
        "comp_means": means_arr,
        "comp_covs": covs_arr,
        "mean": post_mean,
        "cov": post_cov,
    }


# ---------------------------------------------------------------------------
# 2. Mode search, Laplace evidence & defensive Student-t importance sampling
# ---------------------------------------------------------------------------


def logmeanexp_ref(a: np.ndarray) -> float:
    """Numerically stable ``ln(mean(exp(a)))`` using ``scipy.special.logsumexp``."""
    arr = np.asarray(a, dtype=np.float64)
    if np.all(np.isneginf(arr)):
        return -math.inf
    return float(special.logsumexp(arr) - math.log(arr.size))


def laplace_log_evidence_ref(log_target: float, cov: np.ndarray) -> float:
    r"""Laplace approximation to :math:`\ln Z` about a mode with unnormalized
    peak :math:`\ln p^*(x_{\mathrm{MAP}})` and local covariance :math:`\Sigma`
    (Tierney & Kadane 1986, *JASA* 81, 82; MacKay 2003, *Information Theory,
    Inference, and Learning Algorithms*, eq. 27.2):

    .. math::

        \ln \hat{Z}_{\mathrm{Laplace}}
        = \ln p^*(x_{\mathrm{MAP}}) + \frac{d}{2}\ln(2\pi) + \frac{1}{2}\ln|\Sigma|.
    """
    cov_a = np.asarray(cov, dtype=np.float64)
    d = cov_a.shape[0]
    sign, logdet = np.linalg.slogdet(cov_a)
    if sign <= 0:
        raise ValueError("Mode covariance must be positive definite.")
    return float(log_target) + 0.5 * d * math.log(2.0 * math.pi) + 0.5 * float(logdet)


def distinct_modes_ref(
    xs: Sequence[np.ndarray],
    log_targets: Sequence[float],
    covs: Sequence[np.ndarray],
    mahal: float = 1.0,
    two_sided: bool = True,
) -> list[int]:
    r"""Greedy highest-peak-first mode deduplication returning the kept original
    indices, implementing the documented deduplication specification of
    ``virgil.mode_search.distinct_modes`` (a documented ``virgil`` API rule with
    default squared-Mahalanobis threshold ``mahal = 1.0``, rather than an
    external published theorem).

    With ``two_sided=True``, candidate :math:`b` is merged into an already-kept
    mode :math:`a` if and only if the squared Mahalanobis distance is below
    ``mahal`` under *both* :math:`\Sigma_a` and :math:`\Sigma_b`:

    .. math::

        (x_b - x_a)^\top \Sigma_a^{-1}(x_b - x_a) < \tau
        \quad \text{and} \quad
        (x_b - x_a)^\top \Sigma_b^{-1}(x_b - x_a) < \tau,

    preventing a wide background/outlier covariance from swallowing a sharp
    narrow mode sitting inside its footprint. With ``two_sided=False`` (used
    only for negative controls), only :math:`\Sigma_a` is checked.
    """
    order = sorted(range(len(xs)), key=lambda idx: float(log_targets[idx]), reverse=True)
    xs_a = [np.asarray(x, dtype=np.float64) for x in xs]
    precs = [linalg.inv(np.asarray(c, dtype=np.float64)) for c in covs]

    kept: list[int] = []
    for idx_b in order:
        xb = xs_a[idx_b]
        pb = precs[idx_b]
        duplicate = False
        for idx_a in kept:
            diff = xb - xs_a[idx_a]
            d2_a = float(diff @ precs[idx_a] @ diff)
            if two_sided:
                d2_b = float(diff @ pb @ diff)
                if d2_a < mahal and d2_b < mahal:
                    duplicate = True
                    break
            else:
                if d2_a < mahal:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(idx_b)
    return kept


def defensive_mixture_log_q_ref(
    x: np.ndarray,
    mode_xs: Sequence[np.ndarray],
    mode_log_targets: Sequence[float],
    mode_covs: Sequence[np.ndarray],
    log_prior_fn: Callable[[np.ndarray], np.ndarray],
    alpha: float = 0.1,
    df: float = 5.0,
    images: Sequence[np.ndarray] = (),
) -> np.ndarray:
    r"""Exact log proposal density :math:`\ln q(x)` of a Hesterberg (1995,
    *Technometrics* 37, 185) defensive importance mixture combining prior weight
    :math:`\alpha` with Laplace-evidence-weighted multivariate Student-:math:`t_\nu`
    components (Kotz & Nadarajah 2004, eq. 1.1) replicated across linear
    symmetry maps :math:`\mathcal{G} = \{I_d\} \cup \text{images}`:

    .. math::

        q(x) = \alpha\,p_0(x) + (1 - \alpha)\sum_{m=1}^M \frac{w_m}{|\mathcal{G}|}
        \sum_{g \in \mathcal{G}} t_\nu\!\left(x;\, g x_m,\, g \Sigma_m g^\top\right).
    """
    pts = np.atleast_2d(np.asarray(x, dtype=np.float64))
    lp = np.asarray(log_prior_fn(pts), dtype=np.float64).reshape(-1)
    if len(mode_xs) == 0:
        return lp if x.ndim == 2 else lp[0]

    d = pts.shape[1]
    maps = [np.eye(d, dtype=np.float64)] + [
        np.asarray(g, dtype=np.float64) for g in images
    ]
    n_maps = len(maps)

    lap_ev = np.array(
        [
            laplace_log_evidence_ref(lt, c)
            for lt, c in zip(mode_log_targets, mode_covs, strict=True)
        ],
        dtype=np.float64,
    )
    log_wm = lap_ev - special.logsumexp(lap_ev)

    terms = [math.log(alpha) + lp]
    log_comp_pref = math.log(1.0 - alpha) - math.log(n_maps)
    for lwm, xm, cm in zip(log_wm, mode_xs, mode_covs, strict=True):
        xm_a = np.asarray(xm, dtype=np.float64)
        cm_a = np.asarray(cm, dtype=np.float64)
        for g in maps:
            loc = g @ xm_a
            shape = g @ cm_a @ g.T
            lt_pdf = stats.multivariate_t.logpdf(pts, loc=loc, shape=shape, df=df)
            terms.append(log_comp_pref + float(lwm) + np.asarray(lt_pdf, dtype=np.float64).reshape(-1))

    log_q = special.logsumexp(np.stack(terms, axis=0), axis=0)
    return log_q if np.asarray(x).ndim == 2 else float(log_q[0])


def importance_diagnostics_ref(log_w: np.ndarray) -> dict[str, float]:
    r"""Self-normalized Kish (1965, *Survey Sampling*) effective sample size
    :math:`\mathrm{ESS} = (\sum_i \tilde{w}_i^2)^{-1}`, importance evidence
    :math:`\ln\hat{Z} = \operatorname{logmeanexp}(\ln w)`, and delta-method
    standard error :math:`\sqrt{\max(1/\mathrm{ESS} - 1/n, 0)}` (Owen 2013,
    *Monte Carlo Theory, Methods and Examples*, sec. 9.2).
    """
    lw = np.asarray(log_w, dtype=np.float64).reshape(-1)
    n = lw.size
    log_z = logmeanexp_ref(lw)
    log_norm_w = lw - float(special.logsumexp(lw))
    w_norm = np.exp(log_norm_w)
    ess = float(1.0 / np.sum(w_norm**2))
    se = math.sqrt(max(1.0 / ess - 1.0 / n, 0.0))
    return {"log_z": log_z, "ess": ess, "log_z_error": se}


# ---------------------------------------------------------------------------
# 3. Correlated spectral-channel tempering (finite-chain variance ratios)
# ---------------------------------------------------------------------------


def finite_chain_variance_ratio_ref(
    chains_z: Sequence[np.ndarray],
) -> tuple[float, float]:
    r"""Finite-chain sum-of-squares variance ratio :math:`f_{\mathrm{raw}}` and
    floored effective-channel factor :math:`f = \max(f_{\mathrm{raw}}, 1)`:

    .. math::

        f_{\mathrm{raw}} = \frac{\sum_{c=1}^C \left(\sum_{i=1}^{n_c} z_{c,i}\right)^2}
                                {\sum_{c=1}^C \sum_{i=1}^{n_c} z_{c,i}^2},
        \qquad f = \max(f_{\mathrm{raw}}, 1).

    For independent standard-normal residuals :math:`\mathbb{E}[\text{num}] = \mathbb{E}[\text{den}] = N`,
    giving :math:`f_{\mathrm{raw}} \to 1` as :math:`C \to \infty`.
    """
    num = 0.0
    den = 0.0
    for ch in chains_z:
        z = np.asarray(ch, dtype=np.float64).reshape(-1)
        if z.size == 0:
            continue
        s = float(np.sum(z))
        num += s * s
        den += float(np.dot(z, z))
    if den == 0.0:
        return 1.0, 1.0
    f_raw = num / den
    return f_raw, max(f_raw, 1.0)


def ar1_finite_chain_factor(n: int, rho: float) -> float:
    r"""Exact expectation of :math:`\frac{1}{n}\mathbb{E}[(\sum_{i=1}^n z_i)^2]`
    for a unit-variance stationary AR(1) chain of length :math:`n` with lag-:math:`k`
    autocorrelation :math:`\gamma(k) = \rho^k` (Priestley 1981, *Spectral
    Analysis and Time Series*, sec. 5.3.2, eq. 5.3.13):

    .. math::

        f_{\mathrm{AR1}}(n, \rho) = 1 + 2\sum_{k=1}^{n-1}\left(1 - \frac{k}{n}\right)\rho^k.

    Unlike the infinite-chain Bartlett limit :math:`(1 + \rho)/(1 - \rho)`,
    this finite-:math:`n` formula accounts for spectral window boundaries on
    short interferometric wavelength tables.
    """
    if n <= 1:
        return 1.0
    ks = np.arange(1, n, dtype=np.float64)
    return float(1.0 + 2.0 * np.sum((1.0 - ks / float(n)) * (float(rho) ** ks)))


def ar1_asymptotic_factor(rho: float) -> float:
    """Infinite-chain Bartlett limit ``(1 + rho) / (1 - rho)`` as :math:`n \\to \\infty`."""
    return float((1.0 + rho) / (1.0 - rho))


def common_mode_chain_factor(n: int, c: float) -> float:
    r"""Exact expectation of :math:`\frac{1}{n}\mathbb{E}[(\sum_{i=1}^n z_i)^2]`
    under equicorrelated (per-baseline common-mode offset) residuals with
    :math:`\gamma(k) = c` for all :math:`k \ge 1`:

    .. math::

        f_{\mathrm{common}}(n, c)
        = 1 + 2c\sum_{k=1}^{n-1}\left(1 - \frac{k}{n}\right)
        = 1 + (n - 1)c.
    """
    return float(1.0 + (n - 1) * float(c))


def chain_lag_correlation_ref(
    chains_z: Sequence[np.ndarray], lag: int = 1
) -> tuple[float, int]:
    """Zero-mean standardized lag-``k`` correlation across spectral chains,
    implementing ``virgil.tempering``'s documented symmetric endpoint
    mean-square normalization ``0.5 * (a.a + b.b)`` (``docs/api/tempering.md``)."""
    a_parts = []
    b_parts = []
    for ch in chains_z:
        z = np.asarray(ch, dtype=np.float64).reshape(-1)
        if z.size > lag:
            a_parts.append(z[:-lag])
            b_parts.append(z[lag:])
    if not a_parts:
        return float("nan"), 0
    a = np.concatenate(a_parts)
    b = np.concatenate(b_parts)
    denom = 0.5 * float(np.dot(a, a) + np.dot(b, b))
    if denom == 0.0:
        return 0.0, int(a.size)
    return float(np.dot(a, b) / denom), int(a.size)


def flat_signature_ref(
    rho1: float, rho2: float, n_pairs2: int, flat_excess: float = 0.15
) -> bool:
    """Lag-2 excess test distinguishing a flat baseline offset (``rho2 ~ rho1``)
    from geometric AR(1) decay (``rho2 ~ rho1^2``), implementing the documented
    diagnostic rule of ``virgil.tempering.flat_signature``
    (``docs/api/tempering.md``: ``rho2 - rho1**2 > max(0.15, 3 / sqrt(n_pairs2))``).

    Note: ``1 / sqrt(n_pairs2)`` is the white-noise lag-2 standard-error scale;
    under a stationary AR(1) null with nonzero ``rho1``, Bartlett's formula
    (Bartlett 1946, *J. R. Stat. Soc. Suppl.* 8, 27) gives an asymptotic
    variance for ``rho2 - rho1**2`` that depends on ``rho1``.
    """
    if n_pairs2 <= 0 or math.isnan(rho1) or math.isnan(rho2):
        return False
    excess = float(rho2) - float(rho1) ** 2
    return bool(excess > flat_excess and excess > 3.0 / math.sqrt(float(n_pairs2)))


def out_of_range_ref(
    f: float, f_raw: float | None = None, flag_range: tuple[float, float] = (0.5, 2.0)
) -> bool:
    """Flag strong anticorrelation (``f_raw < 0.5``) or strong positive
    correlation (``f > 2.0``), implementing the documented diagnostic bounds of
    ``virgil.tempering.out_of_range`` (``docs/api/tempering.md``)."""
    lo, hi = flag_range
    if float(f) > hi:
        return True
    if f_raw is not None and float(f_raw) < lo:
        return True
    return False


def block_dof_ref(blocks: dict[str, float]) -> dict[str, float]:
    """Per-block likelihood tempering weights ``1 / max(f_b, 1)``."""
    return {k: 1.0 / max(float(v), 1.0) for k, v in blocks.items()}


# ---------------------------------------------------------------------------
# 4. Jeffreys Beta(1/2, 1/2) group outlier mixture (quadrature & exact moments)
# ---------------------------------------------------------------------------


def beta_outlier_mixture_quad_ref(
    loglikes: np.ndarray,
    outlier_loglikes: np.ndarray,
    alpha: float = 0.5,
    beta_param: float = 0.5,
) -> tuple[float, np.ndarray]:
    r"""Continuous numerical quadrature of the hierarchical group outlier mixture

    .. math::

        Z = \int_0^1 \mathrm{Beta}(\varepsilon; \alpha, \beta)
            \prod_{k=1}^K \left[(1 - \varepsilon) L_k + \varepsilon B_k\right]
            \mathrm{d}\varepsilon

    and marginal posterior outlier probabilities :math:`q_j = P(z_j = 1 \mid d)`.
    For :math:`\alpha = \beta = \tfrac{1}{2}` (Jeffreys 1946, *Proc. R. Soc. A*
    186, 453), substituting :math:`\varepsilon = \frac{1 - \cos\theta}{2}` removes
    both endpoint square-root singularities analytically so ``scipy.integrate.quad``
    over :math:`\theta \in [0, \pi]` achieves machine precision.
    """
    ll = np.asarray(loglikes, dtype=np.float64).reshape(-1)
    ol = np.asarray(outlier_loglikes, dtype=np.float64).reshape(-1)
    k_groups = ll.size
    shift = np.maximum(ll, ol)
    base_shift = float(np.sum(shift))
    l_scaled = np.exp(ll - shift)
    b_scaled = np.exp(ol - shift)

    if alpha == 0.5 and beta_param == 0.5:
        def integrand_z(theta: float) -> float:
            eps = 0.5 * (1.0 - math.cos(theta))
            terms = (1.0 - eps) * l_scaled + eps * b_scaled
            return float(np.prod(terms)) / math.pi

        z_scaled, _ = integrate.quad(integrand_z, 0.0, math.pi, epsabs=1e-13, epsrel=1e-13, limit=200)
        probs = np.empty(k_groups, dtype=np.float64)
        for j in range(k_groups):
            mask = np.ones(k_groups, dtype=bool)
            mask[j] = False

            def integrand_qj(theta: float, idx: int = j, m: np.ndarray = mask) -> float:
                eps = 0.5 * (1.0 - math.cos(theta))
                rest = float(np.prod((1.0 - eps) * l_scaled[m] + eps * b_scaled[m])) if k_groups > 1 else 1.0
                return (eps * float(b_scaled[idx]) * rest) / math.pi

            num_j, _ = integrate.quad(integrand_qj, 0.0, math.pi, epsabs=1e-13, epsrel=1e-13, limit=200)
            probs[j] = num_j / z_scaled
        return base_shift + math.log(z_scaled), probs

    def gen_integrand_z(eps: float) -> float:
        prior = float(stats.beta.pdf(eps, alpha, beta_param))
        return prior * float(np.prod((1.0 - eps) * l_scaled + eps * b_scaled))

    z_scaled, _ = integrate.quad(gen_integrand_z, 0.0, 1.0, epsabs=1e-14, epsrel=1e-13, limit=200)
    probs = np.empty(k_groups, dtype=np.float64)
    for j in range(k_groups):
        mask = np.ones(k_groups, dtype=bool)
        mask[j] = False

        def gen_integrand_qj(eps: float, idx: int = j, m: np.ndarray = mask) -> float:
            prior = float(stats.beta.pdf(eps, alpha, beta_param))
            rest = float(np.prod((1.0 - eps) * l_scaled[m] + eps * b_scaled[m])) if k_groups > 1 else 1.0
            return prior * eps * float(b_scaled[idx]) * rest

        num_j, _ = integrate.quad(gen_integrand_qj, 0.0, 1.0, epsabs=1e-14, epsrel=1e-13, limit=200)
        probs[j] = num_j / z_scaled
    return base_shift + math.log(z_scaled), probs


def jeffreys_outlier_mixture_exact_subset_ref(
    loglikes: np.ndarray, outlier_loglikes: np.ndarray
) -> tuple[float, np.ndarray]:
    r"""Quadrature-free exact combinatorial evaluation of the Jeffreys
    :math:`\mathrm{Beta}(1/2, 1/2)` group outlier mixture over all :math:`2^K`
    binary indicator configurations :math:`z \in \{0, 1\}^K`, using the exact
    Beta-function moments (Abramowitz & Stegun 1964, eq. 6.2.1):

    .. math::

        \int_0^1 \frac{\varepsilon^m (1 - \varepsilon)^{K - m}}
                      {\pi\sqrt{\varepsilon(1 - \varepsilon)}}\,\mathrm{d}\varepsilon
        = \frac{\mathrm{B}(m + \tfrac{1}{2},\, K - m + \tfrac{1}{2})}
               {\mathrm{B}(\tfrac{1}{2}, \tfrac{1}{2})}.
    """
    ll = np.asarray(loglikes, dtype=np.float64).reshape(-1)
    ol = np.asarray(outlier_loglikes, dtype=np.float64).reshape(-1)
    k_groups = ll.size
    if k_groups > 20:
        raise ValueError("Exact subset expansion is intended for K <= 20.")

    states = ((np.arange(1 << k_groups)[:, None] >> np.arange(k_groups)[None, :]) & 1).astype(bool)
    m_counts = np.sum(states, axis=1).astype(np.float64)
    log_beta_weight = special.betaln(m_counts + 0.5, k_groups - m_counts + 0.5) - special.betaln(0.5, 0.5)
    state_ll = np.sum(np.where(states, ol[None, :], ll[None, :]), axis=1)
    log_terms = log_beta_weight + state_ll

    log_z = float(special.logsumexp(log_terms))
    probs = np.empty(k_groups, dtype=np.float64)
    for j in range(k_groups):
        log_num_j = float(special.logsumexp(log_terms[states[:, j]]))
        probs[j] = math.exp(log_num_j - log_z)
    return log_z, probs


# ---------------------------------------------------------------------------
# 5. Linearized flux prior Bayes factors (Luger et al. 2017 & LogUniform)
# ---------------------------------------------------------------------------


def linear_flux_gaussian_prior_ref(
    f_hat: np.ndarray,
    sigma_f: np.ndarray,
    prior_mean: float,
    prior_sd: float,
) -> dict[str, np.ndarray]:
    r"""Closed-form posterior mean, posterior SD, and log Bayes factor vs
    :math:`f = 0` for a linearized Gaussian profile likelihood
    :math:`L(f)/L(0) = \exp[(\hat{f}^2 - (f - \hat{f})^2)/(2\sigma_f^2)]`
    under :math:`f \sim \mathcal{N}(\mu_0, \tau_0^2)` (Luger et al. 2017,
    *AJ* 154, 237; Wakefield 2009, *Genetic Epidemiology* 33, 79).
    """
    fh = np.asarray(f_hat, dtype=np.float64)
    sf = np.asarray(sigma_f, dtype=np.float64)
    mu0 = float(prior_mean)
    tau0 = float(prior_sd)

    prec_post = 1.0 / (sf**2) + 1.0 / (tau0**2)
    sd_post = 1.0 / np.sqrt(prec_post)
    mean_post = (sd_post**2) * (fh / (sf**2) + mu0 / (tau0**2))

    log_bf = (
        0.5 * (fh / sf) ** 2
        + 0.5 * np.log((sf**2) / (sf**2 + tau0**2))
        - 0.5 * ((fh - mu0) ** 2) / (sf**2 + tau0**2)
    )
    return {
        "posterior_mean": mean_post,
        "posterior_sd": sd_post,
        "log_bayes_factor": log_bf,
    }


def linear_flux_loguniform_prior_ref(
    f_hat: float,
    sigma_f: float,
    f_min: float,
    f_max: float,
) -> dict[str, float]:
    r"""Adaptive SciPy quadrature for the posterior mean, posterior SD, and log
    Bayes factor vs :math:`f = 0` under the scale-invariant log-uniform prior
    :math:`p(f) = [f \ln(f_{\max}/f_{\min})]^{-1}` on :math:`[f_{\min}, f_{\max}]`.
    """
    fh = float(f_hat)
    sf = float(sigma_f)
    f_min_f = float(f_min)
    f_max_f = float(f_max)
    u_min = math.log(f_min_f)
    u_max = math.log(f_max_f)
    span = u_max - u_min

    # Cancellation-free peak shift and relative log-weight on [u_min, u_max]:
    #   ln(L(f_*) / L(0)) = f_* (2 fh - f_*) / (2 sf^2),
    #   ln(L(f) / L(f_*)) = -(f - f_*)(f + f_* - 2 fh) / (2 sf^2),
    # with boundary-layer support truncation where relative weight drops below exp(-700).
    f_star = min(max(fh, f_min_f), f_max_f)
    u_star = math.log(f_star)
    peak_log_rel = 0.5 * f_star * (2.0 * fh - f_star) / (sf * sf)

    def df_from_star(u: float) -> float:
        return f_star * math.expm1(u - u_star)

    def weight(u: float) -> float:
        df = df_from_star(u)
        log_rel_diff = -0.5 * df * (df + 2.0 * (f_star - fh)) / (sf * sf)
        return math.exp(log_rel_diff) if log_rel_diff > -745.0 else 0.0

    cut = 1400.0 * sf * sf
    if fh <= f_min_f:
        d0 = f_min_f - fh
        df_cut = cut / (d0 + math.hypot(d0, math.sqrt(cut)))
        u_lo = u_min
        u_hi = min(u_max, u_min + math.log1p(df_cut / f_min_f))
        du = math.log1p(min((sf * sf) / max(d0 * f_min_f, sf * f_min_f), df_cut / f_min_f))
        pts = sorted(
            {u_lo, *(u_lo + c * du for c in (0.5, 2.0, 8.0) if u_lo + c * du < u_hi), u_hi}
        )
    elif fh >= f_max_f:
        d0 = fh - f_max_f
        df_cut = cut / (d0 + math.hypot(d0, math.sqrt(cut)))
        u_hi = u_max
        u_lo = max(u_min, u_max + math.log1p(-min(df_cut / f_max_f, 1.0 - 1e-15)))
        du = -math.log1p(
            -min((sf * sf) / max(d0 * f_max_f, sf * f_max_f), df_cut / f_max_f, 0.5)
        )
        pts = sorted(
            {u_lo, *(u_hi - c * du for c in (0.5, 2.0, 8.0) if u_hi - c * du > u_lo), u_hi}
        )
    else:
        u_hat = u_star
        u_lo = max(u_min, math.log(max(f_min_f, fh - 38.0 * sf)))
        u_hi = min(u_max, math.log(min(f_max_f, fh + 38.0 * sf)))
        delta_u = min(6.0 * sf / fh, 0.45 * (u_hi - u_lo))
        pts = sorted(
            {u_lo, max(u_lo, u_hat - delta_u), u_hat, min(u_hi, u_hat + delta_u), u_hi}
        )

    z0, _ = integrate.quad(
        weight, u_lo, u_hi, points=pts[1:-1], epsabs=0.0, epsrel=1e-13, limit=300
    )
    d1, _ = integrate.quad(
        lambda u: df_from_star(u) * weight(u),
        u_lo,
        u_hi,
        points=pts[1:-1],
        epsabs=0.0 if fh <= f_min_f or fh >= f_max_f else 1e-15 * f_star * z0,
        epsrel=1e-13,
        limit=300,
    )
    mean_offset = d1 / z0
    mean = f_star + mean_offset
    v2, _ = integrate.quad(
        lambda u: ((df_from_star(u) - mean_offset) ** 2) * weight(u),
        u_lo,
        u_hi,
        points=pts[1:-1],
        epsabs=0.0,
        epsrel=1e-13,
        limit=300,
    )
    sd = math.sqrt(max(v2 / z0, 0.0))
    log_bf = peak_log_rel + math.log(z0 / span)
    return {
        "posterior_mean": mean,
        "posterior_sd": sd,
        "log_bayes_factor": log_bf,
    }


# ---------------------------------------------------------------------------
# 6. K-fold linear-Gaussian ridge regression & Breiman 1-SE aggregation
# ---------------------------------------------------------------------------


def ridge_kfold_cv_ref(
    design: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    fold: np.ndarray,
    settings: Sequence[float],
    groups: np.ndarray | None = None,
    include_log_norm: bool = False,
) -> list[dict]:
    r"""Exact closed-form K-fold cross-validation records for weighted linear
    ridge regression minimizing

    .. math::

        \mathcal{L}_{\mathrm{train}}(\theta; \lambda)
        = \frac{1}{2}\sum_{i \in T_k}\!\left(\frac{y_i - (X\theta)_i}{\sigma_i}\right)^2
        + \frac{\lambda}{2}\|\theta\|_2^2

    via :math:`\hat{\theta}^{(-k)}(\lambda) = (X_{T_k}^\top W_{T_k} X_{T_k} + \lambda I)^{-1}
    X_{T_k}^\top W_{T_k} y_{T_k}` (Hoerl & Kennard 1970, *Technometrics* 12, 55;
    Hastie, Tibshirani & Friedman 2009, *Elements of Statistical Learning*,
    sec. 7.10).
    """
    x_mat = np.asarray(design, dtype=np.float64)
    y_vec = np.asarray(y, dtype=np.float64)
    sig = np.asarray(sigma, dtype=np.float64)
    fold_arr = np.asarray(fold, dtype=np.int64)
    p_dim = x_mat.shape[1]
    fold_ids = sorted(int(v) for v in np.unique(fold_arr))
    grp_arr = None if groups is None else np.asarray(groups, dtype=np.int64)
    grp_labels = None if grp_arr is None else [int(g) for g in np.unique(grp_arr)]

    records: list[dict] = []
    for lam in settings:
        lam_f = float(lam)
        for k in fold_ids:
            train = fold_arr != k
            test = fold_arr == k
            w_tr = 1.0 / (sig[train] ** 2)
            x_tr = x_mat[train]
            lhs = (x_tr.T * w_tr[None, :]) @ x_tr + lam_f * np.eye(p_dim, dtype=np.float64)
            rhs = x_tr.T @ (w_tr * y_vec[train])
            theta_hat = linalg.solve(lhs, rhs, assume_a="pos")

            res_te = (y_vec[test] - x_mat[test] @ theta_hat) / sig[test]
            chi2_te = float(np.sum(res_te**2))
            log_norm_te = float(np.sum(np.log(sig[test]))) if include_log_norm else 0.0
            score_te = 0.5 * chi2_te + log_norm_te

            rec: dict = {
                "setting": lam_f,
                "fold": k,
                "score": score_te,
                "chi2": chi2_te,
                "n_test": int(np.sum(test)),
                "n_train": int(np.sum(train)),
                "theta_hat": theta_hat,
            }
            if grp_arr is not None and grp_labels is not None:
                g_te = grp_arr[test]
                rec["group_labels"] = grp_labels
                rec["group_score"] = [
                    float(
                        0.5 * np.sum(res_te[g_te == g] ** 2)
                        + (float(np.sum(np.log(sig[test][g_te == g]))) if include_log_norm else 0.0)
                    )
                    for g in grp_labels
                ]
                rec["group_chi2"] = [float(np.sum(res_te[g_te == g] ** 2)) for g in grp_labels]
                rec["group_n"] = [int(np.sum(g_te == g)) for g in grp_labels]
            records.append(rec)
    return records


def aggregate_cv_ref(records: Sequence[dict], larger_is_simpler: bool = True) -> dict:
    """Reference K-fold CV curve aggregator with sample SD (``ddof=1``), standard
    error ``sd / sqrt(K)``, minimum-mean ``best``, and Breiman et al. (1984,
    *Classification and Regression Trees*, sec. 3.4.3) one-standard-error rule."""
    settings = sorted({float(r["setting"]) for r in records})
    folds = sorted({int(r["fold"]) for r in records})
    n_s, n_f = len(settings), len(folds)
    s_idx = {s: i for i, s in enumerate(settings)}
    f_idx = {f: j for j, f in enumerate(folds)}

    fold_scores = np.empty((n_s, n_f), dtype=np.float64)
    chi2_sum = np.zeros(n_s, dtype=np.float64)
    has_chi2 = all(r.get("chi2") is not None for r in records)
    for r in records:
        i = s_idx[float(r["setting"])]
        j = f_idx[int(r["fold"])]
        fold_scores[i, j] = float(r["score"])
        if has_chi2:
            chi2_sum[i] += float(r["chi2"])

    score = np.sum(fold_scores, axis=1)
    mean = np.mean(fold_scores, axis=1)
    sd = np.std(fold_scores, axis=1, ddof=1)
    se = sd / math.sqrt(float(n_f))
    best = int(np.argmin(mean))
    thresh = float(mean[best] + se[best])
    eligible = np.flatnonzero(mean <= thresh)
    one_se = int(eligible[-1] if larger_is_simpler else eligible[0])
    edge = bool(best in (0, n_s - 1) or one_se in (0, n_s - 1))
    return {
        "settings": np.asarray(settings),
        "folds": np.asarray(folds),
        "fold_scores": fold_scores,
        "score": score,
        "mean": mean,
        "sd": sd,
        "se": se,
        "chi2": chi2_sum if has_chi2 else None,
        "best": best,
        "one_se": one_se,
        "edge": edge,
    }


# ---------------------------------------------------------------------------
# 7. Unconstrained Gauss-Newton precision & 2D quadrature posterior moments
# ---------------------------------------------------------------------------


def gauss_newton_precision_fd_ref(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    unconstrained_log_prior_fn: Callable[[np.ndarray], float],
    z_star: np.ndarray,
    eps_j: float = 1e-6,
    eps_p: float = 1e-5,
) -> dict[str, np.ndarray | float]:
    r"""Central-difference reference for the unconstrained Gauss-Newton precision
    :math:`F = J^\top J + \operatorname{diag}(P)`, inverse mass matrix
    :math:`\Sigma = F^{-1}`, and correlation condition number :math:`\kappa(C)`
    (Nocedal & Wright 2006, *Numerical Optimization*, 2nd ed., sec. 10.3).
    """
    z0 = np.asarray(z_star, dtype=np.float64).reshape(-1)
    d = z0.size
    r0 = np.asarray(residual_fn(z0), dtype=np.float64).reshape(-1)
    j_mat = np.empty((r0.size, d), dtype=np.float64)
    p_diag = np.empty(d, dtype=np.float64)
    lp0 = float(unconstrained_log_prior_fn(z0))

    for j in range(d):
        ej = np.zeros(d, dtype=np.float64)
        ej[j] = 1.0
        rp = np.asarray(residual_fn(z0 + eps_j * ej), dtype=np.float64).reshape(-1)
        rm = np.asarray(residual_fn(z0 - eps_j * ej), dtype=np.float64).reshape(-1)
        j_mat[:, j] = (rp - rm) / (2.0 * eps_j)

        lpp = float(unconstrained_log_prior_fn(z0 + eps_p * ej))
        lpm = float(unconstrained_log_prior_fn(z0 - eps_p * ej))
        neg_d2 = -(lpp - 2.0 * lp0 + lpm) / (eps_p**2)
        p_diag[j] = max(neg_d2, 0.0)

    precision = j_mat.T @ j_mat + np.diag(p_diag)
    inv_mass = linalg.inv(precision)
    sd = np.sqrt(np.diag(inv_mass))
    corr = inv_mass / np.outer(sd, sd)
    cond = float(np.linalg.cond(corr))
    return {"precision": precision, "inverse_mass": inv_mass, "condition": cond}


def posterior_2d_grid_moments_ref(
    log_posterior_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    n_grid: int = 201,
) -> dict[str, np.ndarray | float]:
    """Composite Simpson-rule 2D posterior normalization, mean, covariance, and
    marginal quantiles on a dense Cartesian grid for non-Gaussian SVI checks."""
    xs = np.linspace(x_bounds[0], x_bounds[1], n_grid)
    ys = np.linspace(y_bounds[0], y_bounds[1], n_grid)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    lp = np.asarray(log_posterior_fn(xx, yy), dtype=np.float64)
    lp_max = float(np.max(lp))
    unnorm = np.exp(lp - lp_max)

    z_inner = integrate.simpson(integrate.simpson(unnorm, x=ys, axis=1), x=xs, axis=0)
    log_z = lp_max + math.log(float(z_inner))
    pdf = unnorm / float(z_inner)

    mx = float(integrate.simpson(integrate.simpson(xx * pdf, x=ys, axis=1), x=xs, axis=0))
    my = float(integrate.simpson(integrate.simpson(yy * pdf, x=ys, axis=1), x=xs, axis=0))
    vxx = float(integrate.simpson(integrate.simpson(((xx - mx) ** 2) * pdf, x=ys, axis=1), x=xs, axis=0))
    vyy = float(integrate.simpson(integrate.simpson(((yy - my) ** 2) * pdf, x=ys, axis=1), x=xs, axis=0))
    vxy = float(
        integrate.simpson(integrate.simpson((xx - mx) * (yy - my) * pdf, x=ys, axis=1), x=xs, axis=0)
    )

    px = integrate.simpson(pdf, x=ys, axis=1)
    py = integrate.simpson(pdf, x=xs, axis=0)
    cdf_x = integrate.cumulative_trapezoid(px, xs, initial=0.0)
    cdf_x /= cdf_x[-1]
    cdf_y = integrate.cumulative_trapezoid(py, ys, initial=0.0)
    cdf_y /= cdf_y[-1]
    q_levels = np.array([0.16, 0.50, 0.84])
    qx = np.interp(q_levels, cdf_x, xs)
    qy = np.interp(q_levels, cdf_y, ys)

    return {
        "log_z": log_z,
        "mean": np.array([mx, my]),
        "cov": np.array([[vxx, vxy], [vxy, vyy]]),
        "quantiles_x": qx,
        "quantiles_y": qy,
    }
