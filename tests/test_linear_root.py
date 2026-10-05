"""virgil._linear, the shared analytic marginalisation of linear parameters
(Luger, Foreman-Mackey & Hogg 2017), against dense Gaussians.

The documented model: d = m + A w + n, n ~ N(0, D), w ~ N(mu, Lambda), so
d ~ N(m + A mu, D + A Lambda A^T). Every check here builds that covariance
densely with NumPy and evaluates it with SciPy: the log density, the
whitened residual (a square root of the inverse covariance), the
log-determinant, the conditional posterior of w (direct Gaussian
conditioning of the joint of w and d), and the limits the docs name
(degenerate columns, widths going to zero, a prior so broad it is flat).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from evidence.plugin import record

L = pytest.importorskip("virgil._linear")

pytestmark = pytest.mark.x64

NODE = "virgil._linear"  # LinearMarginal, posterior and the three whitening functions


def case(seed, n=30, k=3, full=False, degenerate=False):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, k))
    if degenerate and k >= 2:
        A[:, 1] = A[:, 0]  # two equal columns: A has rank k - 1
    sigma = rng.uniform(0.5, 2.0, n)
    mu = rng.normal(size=k)
    if full:
        G = rng.normal(size=(k, k))
        cov = G @ G.T + 0.3 * np.eye(k)
        sd = None
    else:
        sd = rng.uniform(0.3, 3.0, k)
        cov = np.diag(sd**2)
    resid = rng.normal(size=n) * 2.0
    return A, sigma, mu, sd, cov, resid


def marginal(A, mu, sd, cov, method="cholesky"):
    if sd is None:
        return L.LinearMarginal(A, mu, prior_cov=cov, method=method)
    return L.LinearMarginal(A, mu, prior_sd=sd, method=method)


def dense_logpdf(A, sigma, mu, cov, resid):
    C = np.diag(sigma**2) + A @ cov @ A.T
    return stats.multivariate_normal(A @ mu, C).logpdf(resid), C


CASES = [
    dict(seed=1),
    dict(seed=2, n=8, k=1),
    dict(seed=3, n=40, k=6),
    dict(seed=4, full=True),
    dict(seed=5, degenerate=True),
    dict(seed=6, k=4, full=True, degenerate=True),
]


@pytest.mark.parametrize("method", ["cholesky", "rank_one"])
@pytest.mark.parametrize("kw", CASES, ids=lambda kw: "-".join(f"{k}{v}" for k, v in kw.items()))
@pytest.mark.validates(NODE, roots=["mathematics"])
def test_loglike_and_whitening_match_a_dense_gaussian(kw, method):
    """The log density, the whitened residual's squared norm (the
    Mahalanobis distance) and the log-normalisation, for diagonal and full
    priors, with and without rank-deficient designs, by both methods."""
    A, sigma, mu, sd, cov, resid = case(**kw)
    m = marginal(A, mu, sd, cov, method)
    want, C = dense_logpdf(A, sigma, mu, cov, resid)
    got = float(m.loglike(resid, sigma))
    u, log_norm = (np.asarray(v) for v in m.whiten(resid, sigma))
    r = resid - A @ mu
    record("abs_dloglike", abs(got - want))
    assert abs(got - want) < 1e-10 * max(1.0, abs(want))
    assert u.shape == resid.shape
    assert abs(u @ u - r @ np.linalg.solve(C, r)) < 1e-10 * max(1.0, r @ np.linalg.solve(C, r))
    assert abs(float(log_norm) - 0.5 * np.linalg.slogdet(C)[1]) < 1e-10


@pytest.mark.parametrize("method", ["cholesky", "rank_one"])
@pytest.mark.validates(NODE, roots=["mathematics"])
def test_whitening_is_a_square_root_of_the_inverse_covariance(method):
    """u = M r is linear in r: M^T M must equal (D + A Lambda A^T)^-1, so
    that u is whitened for every r, not just its norm for one."""
    A, sigma, mu, sd, cov, _ = case(7, n=12, k=3, full=True)
    m = marginal(A, mu, sd, cov, method)
    M = np.asarray(jax.jacfwd(lambda r: m.whiten(r, sigma)[0])(jnp.zeros(12)))
    C = np.diag(sigma**2) + A @ cov @ A.T
    np.testing.assert_allclose(M.T @ M, np.linalg.inv(C), atol=1e-12)


@pytest.mark.validates(NODE, roots=["mathematics"])
def test_blocks_with_spanning_columns_match_the_dense_covariance():
    """Blocks of rows with their own columns, plus columns spanning every
    block: the whitened norm and the summed per-row log-normalisation
    against I + U U^T built densely (rows padded out of range are ignored)."""
    rng = np.random.default_rng(8)
    n = 14
    x = rng.normal(size=n)
    rows = np.array([[0, 1, 2, 3, 4, n], [5, 6, 7, 8, n, n], [9, 10, 11, 12, 13, n]])
    local = np.zeros(rows.shape + (2,))
    U = np.zeros((n, 0))
    for b in range(rows.shape[0]):
        real = rows[b] < n
        cols = rng.normal(size=(int(real.sum()), 2))
        local[b, real] = cols
        block = np.zeros((n, 2))
        block[rows[b][real]] = cols
        U = np.hstack([U, block])
    spanning = rng.normal(size=(n, 2)) * 0.7
    U = np.hstack([U, spanning])
    u, extra = (np.asarray(v) for v in L.whiten_blocks(jnp.asarray(x), jnp.asarray(rows), jnp.asarray(local), jnp.asarray(spanning)))
    C = np.eye(n) + U @ U.T
    assert abs(u @ u - x @ np.linalg.solve(C, x)) < 1e-11
    assert abs(extra.sum() - 0.5 * np.linalg.slogdet(C)[1]) < 1e-11


@pytest.mark.parametrize("kw", [dict(seed=9), dict(seed=10, full=True), dict(seed=11, degenerate=True)],
                         ids=["diagonal", "full", "degenerate"])
@pytest.mark.validates(NODE, roots=["mathematics"])
def test_posterior_is_the_gaussian_conditional(kw):
    """w | d from the joint Gaussian of (w, d): mean mu + Lambda A^T C^-1 r,
    covariance Lambda - Lambda A^T C^-1 A Lambda, in w's own units."""
    A, sigma, mu, sd, cov, resid = case(**kw)
    got_mean, got_cov = (np.asarray(v) for v in marginal(A, mu, sd, cov).posterior(resid, sigma))
    C = np.diag(sigma**2) + A @ cov @ A.T
    r = resid - A @ mu
    want_mean = mu + cov @ A.T @ np.linalg.solve(C, r)
    want_cov = cov - cov @ A.T @ np.linalg.solve(C, A @ cov)
    np.testing.assert_allclose(got_mean, want_mean, atol=1e-10)
    np.testing.assert_allclose(got_cov, want_cov, atol=1e-10)


@pytest.mark.validates(NODE, roots=["mathematics"])
def test_narrow_prior_is_the_fixed_parameter_limit():
    """As the prior width goes to zero, w is fixed at mu and the density is
    the plain Gaussian of d - m - A mu."""
    A, sigma, mu, _, _, resid = case(12)
    got = float(marginal(A, mu, np.full(3, 1e-9), None).loglike(resid, sigma))
    want = stats.multivariate_normal(A @ mu, np.diag(sigma**2)).logpdf(resid)
    assert abs(got - want) < 1e-8


@pytest.mark.validates(NODE, roots=["mathematics"])
def test_broad_prior_is_flat_only_up_to_its_width():
    """As Lambda = s^2 I grows, log p(d) + k log s tends to the flat-prior
    likelihood, -1/2 r^T P r - 1/2 log det D - 1/2 log det(A^T D^-1 A)
    - n/2 log 2 pi, with P = D^-1 - D^-1 A (A^T D^-1 A)^-1 A^T D^-1 (the
    Gaussian prior's own normaliser keeps the k/2 log 2 pi a flat prior
    would drop): the width enters as a constant, as the docs say."""
    A, sigma, _, _, _, resid = case(13, n=20, k=3)
    n, k = A.shape
    Di = np.diag(1 / sigma**2)
    F = A.T @ Di @ A
    P = Di - Di @ A @ np.linalg.solve(F, A.T @ Di)
    flat = (-0.5 * resid @ P @ resid - np.sum(np.log(sigma)) - 0.5 * np.linalg.slogdet(F)[1]
            - 0.5 * n * np.log(2 * np.pi))
    gaps = []
    for s in (1e2, 1e4, 1e6):
        m = marginal(A, np.zeros(k), np.full(k, s), None)
        gaps.append(abs(float(m.loglike(resid, sigma)) + k * np.log(s) - flat))
    record("gap_at_1e6", gaps[-1])
    assert gaps[0] > gaps[1] > gaps[2] or gaps[-1] < 1e-9
    assert gaps[-1] < 1e-6


@pytest.mark.validates(NODE, roots=["mathematics"])
def test_rank_one_gradients_are_smooth_at_degenerate_columns_and_zero_width():
    """The documented reason for the rank-one method: gradients with
    respect to the prior widths stay finite and equal finite differences
    of the dense density where two columns are equal and where a width is
    zero."""
    A, sigma, mu, _, _, resid = case(14, k=3, degenerate=True)

    def f(sd):
        return L.LinearMarginal(A, mu, prior_sd=sd, method="rank_one").loglike(resid, sigma)

    def dense_var(var):
        return dense_logpdf(A, sigma, mu, np.diag(var), resid)[0]

    for sd in (np.array([0.7, 1.3, 2.0]), np.array([0.7, 1.3, 1e-12])):
        g = np.asarray(jax.grad(f)(jnp.asarray(sd)))
        assert np.all(np.isfinite(g))
        # d/dsd = 2 sd d/dvar: central differences in the variance, forward
        # from zero where the width is tiny (a central step in sd would
        # straddle zero, where the density is even in sd, leaving roundoff)
        var = np.asarray(sd) ** 2
        fd = np.empty(3)
        for n, e in enumerate(np.eye(3)):
            h = 1e-6 * max(var[n], 1e-6)
            if var[n] > h:
                dv = (dense_var(var + h * e) - dense_var(var - h * e)) / (2 * h)
            else:
                dv = (dense_var(var + h * e) - dense_var(var)) / h
            fd[n] = 2 * sd[n] * dv
        np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-9)


@pytest.mark.validates(NODE, roots=["mathematics"], kind="control")
def test_the_prior_must_be_stated_and_finite():
    """The docs' rules: no flat or data-derived prior. Missing, infinite
    or zero widths, and non-finite means, are refused."""
    A = np.ones((4, 1))
    for kw in (dict(), dict(prior_sd=np.inf), dict(prior_sd=0.0), dict(prior_sd=1.0, prior_cov=np.eye(1))):
        with pytest.raises(ValueError):
            L.LinearMarginal(A, 0.0, **kw)
    with pytest.raises(ValueError):
        L.LinearMarginal(A, np.nan, prior_sd=1.0)
