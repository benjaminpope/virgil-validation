"""crosscheck.simulate, checked before it judges virgil: the noise it draws (V² Gaussian,
closure phases formed from per-baseline phases) has the stated distributions and
correlations, and the file holds the uv and errors it was given. It uses crosscheck.array
for the uv and crosscheck.chi2 for the triangle matrix it compares with."""

import numpy as np
import pytest
from astropy.io import fits
from scipy import stats

from crosscheck import array, chi2, simulate
from evidence.plugin import record

UTS4 = np.array([[-9.925, -20.335, 0.0], [14.887, 30.502, 0.0], [44.915, 66.183, 0.0], [103.306, 43.999, 0.0]])


@pytest.mark.validates("crosscheck.simulate", roots=["statistics", "standards"], kind="reference")
def test_simulated_noise_uv_and_errors(tmp_path):
    vis = lambda u, v, w: np.ones(np.broadcast(u, v, w).shape, complex)  # noqa: E731
    sv2, scp = 0.02, 2.0
    kw = dict(hour_angles_h=[-1.0, 1.0], wavelengths=[1.6e-6, 2.2e-6], dec_deg=-40.0, sigma_v2=sv2, sigma_cp_deg=scp)
    pulls_v2, cps = [], []
    for k in range(150):
        path = tmp_path / f"s{k}.fits"
        simulate.observe(path, vis, UTS4, rng=np.random.default_rng(k), **kw)
        with fits.open(path) as h:
            pulls_v2.append((h["OI_VIS2"].data["VIS2DATA"] - 1.0) / sv2)
            cps.append(h["OI_T3"].data["T3PHI"])
            if k == 0:
                np.testing.assert_array_equal(h["OI_VIS2"].data["VIS2ERR"], sv2)
                np.testing.assert_array_equal(h["OI_T3"].data["T3PHIERR"], scp)
                u, v = array.snapshot_uv(UTS4, -1.0, -40.0, -24.6276)
                np.testing.assert_allclose(h["OI_VIS2"].data["UCOORD"][:6], u, atol=1e-9)
                np.testing.assert_allclose(h["OI_VIS2"].data["VCOORD"][:6], v, atol=1e-9)
    pulls_v2 = np.ravel(pulls_v2)
    assert stats.kstest(pulls_v2, "norm").pvalue > 1e-3
    # closure phases of one snapshot and channel: four triangles from six baseline phases
    x = np.array(cps)[:, :4, 0]  # first snapshot, first channel: (draws, 4 triangles), degrees
    corr = np.corrcoef(x.T)
    T = chi2.triangle_matrix([(1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)])
    want = (T @ T.T) / 3.0
    record("max_abs_dcorr", float(np.max(np.abs(corr - want))))
    assert np.allclose(np.diag(np.cov(x.T)), scp**2, rtol=0.35)
    assert np.max(np.abs(corr - want)) < 4 / np.sqrt(len(x))
    # whitened by our own covariance scp^2 T T^T / 3 (rank 3: four triangles of four
    # telescopes close on three independent phases), each snapshot and channel is chi-squared of 3
    x = np.array(cps).reshape(len(cps), -1, 4, 2).transpose(0, 1, 3, 2).reshape(-1, 4)
    q = np.einsum("ni,ij,nj->n", x, np.linalg.pinv(scp**2 * want), x)
    record("mean_chi2_per_dof", float(q.mean() / 3))
    assert stats.kstest(q, stats.chi2(3).cdf).pvalue > 1e-3
    assert stats.kstest(q, stats.chi2(4).cdf).pvalue < 1e-3  # four independent phases would not fit: a control
