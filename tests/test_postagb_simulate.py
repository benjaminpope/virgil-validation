"""scripts/postagb_noise.py: the simulation hook (WP-02) and the measured white scales and night widths (WP-04).

The hook wraps ``OIData.with_model``; what it must get right is the closure-phase covariance (two triangles that share a baseline share
a baseline phase error) and the error arm it draws from. Expected covariances are written out in numpy below, not imported from virgil."""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
try:
    import postagb_noise as N
    from virgil import OIData
    from virgil.models import PointSource
except ImportError:
    N = None
pytestmark = [
    pytest.mark.skipif(N is None, reason="needs virgil and virgil-vlti (install the postagb extra)"),
    pytest.mark.validates("evidence", roots=["statistics"], kind="guard"),
]

PAIRS = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
TRIANGLES = [((1, 2, 3), (0, 3, 1)), ((1, 2, 4), (0, 4, 2)), ((1, 3, 4), (1, 5, 2)), ((2, 3, 4), (3, 5, 4))]  # (ab, bc, ac) pairs, phi = arg(V_ab V_bc V_ac*)
CFG = dict(key="no_such_target", data_key="no_such_target")


def toy4(n_frames=30, n_ch=2, d_vis=0.05, d_phi=0.05):
    """Four telescopes, ``n_frames`` frames over three nights, six baselines, ``n_ch`` channels, four closure triangles per frame and channel."""
    rows = []
    for f in range(n_frames):
        mjd = 100.1 + (f // 10) + 0.001 * f  # three nights of ten frames (UT before 12 UT: the night that began the day before)
        for p, (a, b) in enumerate(PAIRS):
            for c in range(n_ch):
                rows.append((f, mjd, a, b, 1.6e-6 + 0.1e-6 * c, 20.0 + 9.0 * p + 3.0 * f % 7, -15.0 + 7.0 * p + 2.0 * f % 5))
    r = np.array(rows)
    n = len(rows)
    per = len(PAIRS) * n_ch
    i1, i2, i3 = [], [], []
    for f in range(n_frames):
        for c in range(n_ch):
            for _, (p_ab, p_bc, p_ac) in TRIANGLES:
                i1.append(f * per + n_ch * p_ab + c)
                i2.append(f * per + n_ch * p_bc + c)
                i3.append(f * per + n_ch * p_ac + c)
    m = len(i1)
    d = dict(u=r[:, 5], v=r[:, 6], vis=np.full(n, 0.9), d_vis=np.full(n, d_vis), vis_flag=np.zeros(n, bool), wavel=r[:, 4], mjd=r[:, 1],
             stations=r[:, 2:4].astype(int), frame=r[:, 0].astype(int), phi=np.zeros(m), d_phi=np.full(m, d_phi), phi_flag=np.zeros(m, bool),
             i_cps1=np.array(i1), i_cps2=np.array(i2), i_cps3=np.array(i3))
    return OIData(d)


def truth():
    return PointSource()


def test_white_and_night_truth_chi2_per_independent_observable_is_one():
    """Self-consistency only: virgil draws and virgil whitens, so a mistake common to both would pass. The statistical check of the
    closure covariance is the test below, against an independent numpy construction."""
    d = toy4()
    for noise in ("white", "night"):
        chi = [N.simulate(CFG, truth(), noise=noise, seed=s, data0=d).truth_chi2 for s in range(20)]
        ratio = np.mean([c["total"] / c["n_independent"] for c in chi])
        assert ratio == pytest.approx(1.0, rel=0.03), (noise, ratio)


def test_closure_noise_between_triangles_sharing_a_baseline_is_a_third_of_the_variance():
    d = toy4(n_frames=6, n_ch=1)
    model_phi = np.asarray(d.with_model(truth()).phi)
    draws = np.array([np.asarray(N.simulate(CFG, truth(), noise="white", seed=s, data0=d).data.phi) - model_phi for s in range(2000)])
    # expected covariance, from independent baseline phase errors of equal variance: closure = A @ baseline phases
    n_ch, per = 1, len(PAIRS)
    A = np.zeros((draws.shape[1], 6 * 6 * n_ch))
    row = 0
    for f in range(6):
        for _, (p_ab, p_bc, p_ac) in TRIANGLES:
            A[row, f * per + p_ab], A[row, f * per + p_bc], A[row, f * per + p_ac] = 1.0, 1.0, -1.0
            row += 1
    cov_expected = A @ A.T
    corr_expected = cov_expected / np.sqrt(np.outer(np.diag(cov_expected), np.diag(cov_expected)))
    corr = np.corrcoef(draws, rowvar=False)
    first = slice(0, 4)  # the four triangles of frame 0: every pair shares exactly one baseline
    off = ~np.eye(4, dtype=bool)
    assert np.all(np.abs(np.abs(corr_expected[first, first][off]) - 1 / 3) < 1e-12)
    np.testing.assert_allclose(corr[first, first][off], corr_expected[first, first][off], atol=0.06)
    # the negative control goes through ``simulate``: independent noise per triangle (what the earlier injections drew) would have zero
    # off-diagonal correlation, and the hook's draws are not that
    wrong = np.eye(4)
    assert not np.allclose(corr[first, first][off], wrong[off], atol=0.06)


def test_same_seed_same_data_different_seed_different_data():
    d = toy4(n_frames=4)
    a, b, c = (np.asarray(N.simulate(CFG, truth(), seed=s, data0=d).data.vis) for s in (3, 3, 4))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_mackay_round_trip_of_a_white_simulation():
    d = toy4()
    sim = N.simulate(CFG, truth(), noise="white", scale=1.3, seed=1, data0=d)
    assert sim.meta["scale_drawn"] == 1.3
    assert sim.meta["scale_fitted"] == pytest.approx(1.3, rel=0.05)
    # the data carry the quoted errors; the fit data carry the scaled ones
    np.testing.assert_allclose(np.asarray(sim.fit_data.d_vis), 1.3 * np.asarray(d.d_vis))
    np.testing.assert_allclose(np.asarray(sim.data.d_vis), np.asarray(d.d_vis))


def test_unknown_noise_is_refused():
    with pytest.raises(ValueError):
        N.simulate(CFG, truth(), noise="bogus", data0=toy4(n_frames=2))


# ----------------------------------------------------------------------------
# white stage: scales from the channel scatter about a pilot (WP-04)


def toy_residuals(scale_v2, scale_t3, gain=0.03, seed=0, n_ch=6):
    """Residuals in units of the quoted errors about a smooth pilot, from data = pilot x (night gain) + white noise at ``scale`` x quoted."""
    d = toy4(n_frames=30, n_ch=n_ch, d_vis=0.02, d_phi=0.02)
    rng = np.random.default_rng(seed)
    lab = N.night_labels(d)
    g = rng.normal(0.0, gain, lab["u_cell"].max() + 1)[lab["u_cell"]]
    vi = np.arange(np.size(d.vis))
    res_v2 = (0.9 * (np.exp(2 * g[vi]) - 1.0)) / 0.02 + rng.normal(0, scale_v2, vi.size)
    res_t3 = rng.normal(0, scale_t3, np.size(d.phi))
    return d, res_v2, res_t3


@pytest.mark.parametrize("scale", [0.5, 1.5])
def test_white_scales_recover_the_noise_and_ignore_a_night_gain(scale):
    d, rv, rt = toy_residuals(scale, scale)
    out = N.white_scales(d, rv, rt)
    assert out["v2_white"] == pytest.approx(scale, rel=0.10)
    assert out["t3_white"] == pytest.approx(scale, rel=0.10)
    assert out["skipped_v2"] == out["skipped_t3"] == 0


def test_groups_with_fewer_than_four_channels_are_skipped_and_counted():
    d, rv, rt = toy_residuals(1.0, 1.0, n_ch=3)
    out = N.white_scales(d, rv, rt)
    assert out["n_groups_v2"] == 0 and out["skipped_v2"] > 0 and np.isnan(out["v2_white"])
    assert out["skipped_t3"] > 0


def test_night_width_about_a_pilot_recovers_a_known_gain():
    rng = np.random.default_rng(2)
    d = toy4(n_frames=60, n_ch=3)
    lab = N.night_labels(d)
    for tau in (0.03, 0.10):
        g = rng.normal(0.0, tau, lab["u_cell"].max() + 1)[lab["u_cell"]]
        rel = 2.0 * g + rng.normal(0, 0.01, g.size)  # relative V^2 residual: 2 x the log|V| gain
        w = N.night_widths_about(lab, rel[np.arange(np.size(d.vis))], np.zeros(np.size(d.phi)))
        assert w["night_gain_baseline"] == pytest.approx(tau, rel=0.20)
        assert w["night_t3_offset_deg"] == 0.0


def test_pilot_widths_replace_published_ones_only_beyond_thirty_percent():
    pub = dict(night_gain_baseline=0.03, night_t3_offset_deg=2.0)
    near = dict(night_gain_baseline=0.035, night_t3_offset_deg=2.5)  # +17 %, +25 %: published kept
    far = dict(night_gain_baseline=0.05, night_t3_offset_deg=0.0)  # +67 %, -100 %: pilot taken
    assert N.night_widths_used(pub, near) == pub
    assert N.night_widths_used(pub, far) == far
    assert N.night_widths_used(dict(night_gain_baseline=0.0, night_t3_offset_deg=2.0), far)["night_gain_baseline"] == 0.05  # a zero width is replaced
