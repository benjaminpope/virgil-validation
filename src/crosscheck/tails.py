"""The tail of a companion-search statistic under Gaussian noise: a linear
reference statistic, importance sampling, and a peaks-over-threshold fit.

Everything here is ours (no virgil). It serves the detection campaign
(scripts/detection_campaign.py, design/detection_v3.md), which compares the
exceedance probability of a companion-search statistic from two simulators
of the same Gaussian noise, at false-alarm probabilities down to 5 sigma.

The data vector is d = [V² (row-major), closure phases (radians, row-major)]
and the null hypothesis is d ~ N(m0, C), with C block-diagonal: V² errors
independent, and closure phases of one snapshot and channel correlated by
C = D^1/2 (T T^T / 3) D^1/2 (Kammerer et al. 2020, A&A 644, A110, sec. 2.2;
the covariance of baseline-phase noise of equal variance on every baseline,
the construction of crosscheck.simulate.observe). Its precision is the
generalised inverse W = C^+ (crosscheck.chi2.triangle_matrix builds T).

Linear reference statistic. A companion of flux f at position j changes the
data by about f g_j (g_j = d m / d f at f = 0), so the maximum-likelihood
flux at j is f_j = g_j^T W r / g_j^T W g_j for residuals r = d - m0 and, with
f constrained to be non-negative, the improvement in chi-squared is
max(0, s_j)^2 with the score s_j = g_j^T W r / sqrt(g_j^T W g_j). The search
statistic is S = max_j max(0, s_j)^2, the maximum over search positions
(Gauss-Newton; virgil's detection_statistics, a grid over flux refined by
BFGS, equals it to 0.015 for the campaign template).

Importance sampling (Rubinstein & Kroese, *Simulation and the Monte Carlo
Method*, ch. 5; the defensive mixture of Hesterberg 1995, Technometrics 37,
185). The target is P0(S >= t), t far in the tail. The proposal is the
mixture Q = pi_0 N(m0, C) + sum_j pi_j N(m_j, C): the same noise around the
mean of a companion at position j and flux f_j, chosen so that the companion
alone would give a chi-squared improvement t (so about half the draws exceed
t), plus the null itself (the defensive component). The weight is the exact
density ratio

    w(d) = dP0/dQ(d) = 1 / sum_k pi_k exp(-1/2 q_k + b_k . W (d - m0)),

with m_k - m0 = b_k and q_k = b_k^T W b_k, and P0(S >= t) = E_Q[w 1(S >= t)].
Noise is drawn by the simulator under test, at the shifted mean, so that
non-Gaussian noise would show as a disagreement between the weighted
estimate and the plain one; the weights themselves are the Gaussian density
ratio.

Peaks over threshold (Balkema & de Haan 1974; Pickands 1975): exceedances of
a high threshold u are generalised Pareto, and P(S >= t) = P(S >= u) ·
GPD.sf(t - u).
"""

import numpy as np
from scipy import special, stats

from . import chi2


def precision(d):
    """W = C^+ for the data vector [V² row-major, closure phases row-major]
    of a loaded file (``chi2.load``): V² errors independent, closure phases
    of one snapshot (MJD) and channel with C = D^1/2 (T T^T / 3) D^1/2."""
    nv, nc = d["v2"].size, d["cp"].size
    w = np.zeros((nv + nc, nv + nc))
    w[np.arange(nv), np.arange(nv)] = 1.0 / d["dv2"].ravel() ** 2
    flat = np.arange(nc).reshape(d["cp"].shape)
    for mjd in np.unique(d["t3_mjd"]):
        rows = np.flatnonzero(d["t3_mjd"] == mjd)
        tri = chi2.triangle_matrix(d["t3_sta"][rows])
        r = tri @ tri.T / 3.0
        for k in range(d["cp"].shape[1]):
            s = d["dcp"][rows, k]
            c = s[:, None] * r * s[None, :]
            idx = nv + flat[rows, k]
            w[np.ix_(idx, idx)] = np.linalg.pinv(c, rcond=1e-10, hermitian=True)
    return w


def response(clean, positions, h=1e-6):
    """g_j = d m / d flux at 0 for each position (x, y), by a forward
    difference of ``clean(flux, x, y)``, which returns the noiseless data
    vector. Returns (m0, g) with g of shape (n_position, n)."""
    m0 = clean(0.0, 0.0, 0.0)
    g = np.array([(clean(h, x, y) - m0) / h for x, y in positions])
    return m0, g


class Search:
    """The linear reference statistic over a set of search positions."""

    def __init__(self, w, g):
        self.w = w
        norm = np.sqrt(np.einsum("pn,nm,pm->p", g, w, g))
        self.keep = norm > 1e-9 * norm.max()  # a companion on the primary changes nothing
        self.norm = norm[self.keep]
        self.g = g[self.keep]
        self.a = (self.g @ w) / self.norm[:, None]  # score = a r

    def scores(self, r):
        """Scores s_j of residuals r (shape (n,) or (N, n))."""
        return np.atleast_2d(r) @ self.a.T

    def statistic(self, r):
        """S = max_j max(0, s_j)^2, one per row of ``r``."""
        return np.max(np.maximum(self.scores(r), 0.0), axis=1) ** 2


class Mixture:
    """The defensive mixture proposal of the module docstring for target
    threshold ``t``: component 0 is the null, component j >= 1 a companion
    at ``positions[keep][j - 1]`` with the flux at which a noiseless
    companion would give chi-squared improvement ``t`` (at most ``max_flux``).
    ``clean(flux, x, y)`` is the noiseless data vector, as for ``response``;
    the means use it exactly, not the linearisation."""

    def __init__(self, search, positions, clean, t, null_weight=0.1, max_flux=0.1):
        self.t = float(t)
        pos = np.asarray(positions, float)[search.keep]
        flux = np.minimum(np.sqrt(self.t) / search.norm, max_flux)
        m0 = clean(0.0, 0.0, 0.0)
        shift = [np.zeros_like(m0)] + [clean(f, x, y) - m0 for f, (x, y) in zip(flux, pos)]
        self.shift = np.array(shift)  # (1 + P, n): b_k
        self.flux = np.concatenate([[0.0], flux])
        self.pos = np.vstack([[0.0, 0.0], pos])
        self.q = np.einsum("kn,nm,km->k", self.shift, search.w, self.shift)
        self.wb = self.shift @ search.w  # (1 + P, n): W b_k
        p = len(pos)
        self.pi = np.concatenate([[null_weight], np.full(p, (1.0 - null_weight) / p)])
        self.log_pi = np.log(self.pi)

    def component(self, rng):
        """Draw a component index."""
        return int(rng.choice(self.pi.size, p=self.pi))

    def log_weight(self, r):
        """ln w for residuals r = d - m0 (shape (n,) or (N, n))."""
        r = np.atleast_2d(r)
        log_q = special.logsumexp(self.log_pi[None] - 0.5 * self.q[None] + r @ self.wb.T, axis=1)
        return -log_q


def estimate(values, log_weights, t):
    """Importance-sampling estimate of P(value >= t) from draws of the
    proposal: (p, standard error, effective sample size of the exceedances,
    mean weight and its standard error). The mean weight must be 1
    (a check of the weights and of the noise)."""
    values = np.asarray(values, float)
    w = np.exp(np.asarray(log_weights, float))
    n = values.size
    hit = w * (values >= t)
    p = hit.mean()
    se = hit.std(ddof=1) / np.sqrt(n)
    ess = hit.sum() ** 2 / np.sum(hit**2) if np.any(hit) else 0.0
    return float(p), float(se), float(ess), float(w.mean()), float(w.std(ddof=1) / np.sqrt(n))


def gpd_tail(values, t, tail_fraction=0.02, n_boot=200, seed=0):
    """Peaks over threshold: P(value >= t) from the generalised Pareto fit
    to the top ``tail_fraction`` of ``values`` (maximum likelihood, location
    fixed at the threshold), with a parametric-bootstrap 90% interval for
    the fit (the exceedance count is binomial and added in quadrature on
    the log scale). Returns (p, (low, high))."""
    values = np.asarray(values, float)
    n = values.size
    u = np.quantile(values, 1.0 - tail_fraction)
    ex = values[values > u] - u
    k = ex.size
    shape, _, scale = stats.genpareto.fit(ex, floc=0.0)

    def prob(shape_, scale_):
        return k / n * stats.genpareto.sf(t - u, shape_, loc=0.0, scale=scale_)

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        e = stats.genpareto.rvs(shape, loc=0.0, scale=scale, size=k, random_state=rng)
        s, _, c = stats.genpareto.fit(e, floc=0.0)
        boot.append(prob(s, c) * np.exp(rng.normal(0.0, 1.0 / np.sqrt(k))))
    lo, hi = np.quantile(boot, [0.05, 0.95])
    return float(prob(shape, scale)), (float(lo), float(hi))


def rate_ratio_interval(p1, se1, p2, se2, confidence=0.95):
    """Interval on p1 / p2 from independent estimates with standard errors,
    by the delta method on the log scale (a plain Monte Carlo rate k / n has
    se = sqrt(k) / n). Returns (ratio, low, high)."""
    if p1 <= 0 or p2 <= 0:
        return float("nan"), 0.0, float("inf")
    s = np.sqrt((se1 / p1) ** 2 + (se2 / p2) ** 2)
    z = stats.norm.ppf(0.5 + confidence / 2)
    r = p1 / p2
    return float(r), float(r * np.exp(-z * s)), float(r * np.exp(z * s))


def equivalence_pvalue(ratio, log_se, margin):
    """The larger p-value of the two one-sided tests that ln(ratio) lies
    within +-ln(margin) (TOST; Schuirmann 1987): small when the ratio is
    shown to be within the margin."""
    m = np.log(margin)
    x = np.log(ratio)
    return float(max(stats.norm.sf((x + m) / log_se), stats.norm.cdf((x - m) / log_se)))
