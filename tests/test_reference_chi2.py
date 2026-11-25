"""crosscheck.chi2, checked before it judges virgil: closure phases made by hand from
independent baseline phases are correlated as (σ²/3) T Tᵀ, so the correlated chi-squared
must follow chi-squared with rank-many degrees of freedom. It uses crosscheck.array for the
baselines and triangles of the data it makes."""

import numpy as np
import pytest
from scipy import stats

from crosscheck import array, chi2
from evidence.plugin import record


def _baseline_closure_data(rng, n_snap, sigma_v2, sigma_cp, scale=None):
    """A loaded-file dict for a point source seen by four telescopes, its
    closure phases made here from independent per-baseline phases (N(0,
    σ²/3) each), so the four triangles share baselines."""
    pairs = array.baselines(4)
    tris = array.triangles(4)
    rows_v, rows_t = n_snap * len(pairs), n_snap * len(tris)
    d = {"u": np.ones((rows_v, 1)), "v": np.ones((rows_v, 1)), "wl": np.full((rows_v, 1), 2e-6),
         "v2": 1 + sigma_v2 * rng.normal(size=(rows_v, 1)), "dv2": np.full((rows_v, 1), sigma_v2)}
    cp, mjd, sta = [], [], []
    for k in range(n_snap):
        phase = dict(zip(pairs, rng.normal(0.0, sigma_cp / np.sqrt(3.0), len(pairs))))
        for a, b, c in tris:
            cp.append(phase[(a, b)] + phase[(b, c)] - phase[(a, c)])
            mjd.append(60000.0 + k)
            sta.append((a + 1, b + 1, c + 1))
    d.update({"u1": np.ones((rows_t, 1)), "v1": np.ones((rows_t, 1)), "u2": np.ones((rows_t, 1)),
              "v2_": np.ones((rows_t, 1)), "wl3": np.full((rows_t, 1), 2e-6),
              "cp": np.array(cp)[:, None], "dcp": np.full((rows_t, 1), sigma_cp),
              "t3_mjd": np.array(mjd), "t3_sta": np.array(sta)})
    return d


@pytest.mark.validates("crosscheck.chi2", roots=["statistics"], kind="reference")
def test_correlated_chi2_follows_chi2_of_the_rank():
    rng = np.random.default_rng(1)
    point = lambda u, v, w: np.ones(np.broadcast(u, v, w).shape, complex)  # noqa: E731
    n_snap, totals = 5, []
    for _ in range(400):
        d = _baseline_closure_data(rng, n_snap, 0.01, 0.05)
        totals.append(chi2.chi2(d, point, correlated=True, chord=False))
    dof = n_snap * len(array.baselines(4)) + n_snap * 3  # V² plus 3 independent closures of 4 telescopes
    p = stats.kstest(totals, stats.chi2(dof).cdf).pvalue
    record("mean_over_dof", float(np.mean(totals) / dof))
    record("ks_p", float(p))
    assert abs(np.mean(totals) / dof - 1) < 4 * np.sqrt(2 / dof) / np.sqrt(len(totals))
    assert p > 1e-3
    # the uncorrelated form on the same data is not chi-squared of 30: a control
    wrong = [chi2.chi2(_baseline_closure_data(rng, n_snap, 0.01, 0.05), point) for _ in range(200)]
    assert abs(np.mean(wrong) / dof - 1) > 0.05
