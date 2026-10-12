"""Precision of calibration campaigns: equivalence tests and their power.

A campaign that only fails to reject "calibrated" has not shown that it is: with
few draws a 10% error in the error bars passes. These helpers state the smallest
miscalibration a campaign must rule out as an equivalence test (two one-sided
tests, Schuirmann 1987, which is the same as a 1 - 2 alpha confidence interval
lying inside the margin) and compute its power for the registered counts.

Written from the textbook formulas: the exact (Clopper-Pearson 1934) binomial
interval from beta quantiles; the normal approximation for a mean (standard
error s / sqrt(N)) and for a standard deviation (standard error
s sqrt((kappa - 1) / (4 N)), kappa the kurtosis, equal to s / sqrt(2 N) for a
Gaussian). ``alpha`` is the size of each one-sided test; a family of m tests
with familywise level A uses A / m, which is what Holm gives its least
stringent member, so the power computed is a lower bound for Holm.
"""

import numpy as np
from scipy import stats


def cp_interval(k, n, alpha):
    """Exact interval of a binomial proportion, alpha in each tail."""
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha, k + 1, n - k))
    return lo, hi


def cp_upper(k, n, alpha):
    """One-sided exact upper limit of a binomial proportion at confidence 1 - alpha."""
    return 1.0 if k >= n else float(stats.beta.ppf(1 - alpha, k + 1, n - k))


def max_failures_below(n, margin, alpha):
    """Largest k whose one-sided upper limit is at most ``margin`` (-1 if none)."""
    ks = np.arange(n + 1)
    ok = ks[stats.beta.ppf(1 - alpha, ks + 1, np.maximum(n - ks, 1)) <= margin]
    ok = ok[ok < n]
    return int(ok.max()) if ok.size else -1


def power_upper_below(n, margin, true_rate, alpha):
    """P(the one-sided upper limit of a failure rate is at most ``margin``) when
    the true rate is ``true_rate``."""
    k = max_failures_below(n, margin, alpha)
    return float(stats.binom.cdf(k, n, true_rate)) if k >= 0 else 0.0


def power_ci_inside(n, centre, delta, true, alpha):
    """P(the exact 1 - 2 alpha interval of a proportion lies inside centre +- delta)
    when the true proportion is ``true``: the power of the equivalence test."""
    ks = np.arange(n + 1)
    lo = np.where(ks == 0, 0.0, stats.beta.ppf(alpha, np.maximum(ks, 1), n - ks + 1))
    hi = np.where(ks == n, 1.0, stats.beta.ppf(1 - alpha, ks + 1, np.maximum(n - ks, 1)))
    ok = (lo >= centre - delta) & (hi <= centre + delta)
    return float(stats.binom.pmf(ks, n, true)[ok].sum())


def power_mean_equivalence(n, delta, true, alpha, sd=1.0):
    """Power of 'mean within +- delta' (normal approximation, standard error sd / sqrt(n))."""
    se, z = sd / np.sqrt(n), stats.norm.isf(alpha)
    return float(max(0.0, stats.norm.cdf((delta - true) / se - z) - stats.norm.cdf((-delta - true) / se + z)))


def power_sd_equivalence(n, lo, hi, true, alpha, kurtosis=3.0):
    """Power of 'sd within [lo, hi]' for Gaussian-like data (standard error true sqrt((kappa-1)/(4n)))."""
    se, z = true * np.sqrt((kurtosis - 1) / (4 * n)), stats.norm.isf(alpha)
    return float(max(0.0, stats.norm.cdf((hi - true) / se - z) - stats.norm.cdf((lo - true) / se + z)))


def n_for_power(power_fn, target=0.8, start=50, stop=100000, step=10):
    """Smallest n (on a grid of ``step``) with power_fn(n) >= target."""
    for n in range(start, stop, step):
        if power_fn(n) >= target:
            return n
    return None


def mean_ci(mean, sd, n, alpha):
    h = stats.norm.isf(alpha) * sd / np.sqrt(n)
    return float(mean - h), float(mean + h)


def sd_ci(sd, kurtosis, n, alpha):
    h = stats.norm.isf(alpha) * sd * np.sqrt((kurtosis - 1) / (4 * n))
    return float(sd - h), float(sd + h)


def inside(interval, lo, hi):
    return bool(interval[0] >= lo and interval[1] <= hi)
