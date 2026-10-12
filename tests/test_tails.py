"""The tail of the detection statistic: our linear reference statistic,
importance sampling and the generalised-Pareto fit (crosscheck.tails), and
what they show about the two noise simulators (ledger C3).

The linear statistic S = max_j max(0, s_j)^2 is virgil's delta_chi2 for the
campaign template (to 0.015). With it, the version 2 campaign's excess of
exceedances (288 against 240 at the 1e-2 threshold) is reproduced from its
seeds and shown to be a fluctuation of the sample, and our simulator and
virgil's gaussian_null are shown to draw the same noise: same covariance,
same exceedance rate at 1e-2 and 1e-3, same importance-sampling estimates
down to 5 sigma.
"""

import importlib.util
import pathlib

import numpy as np
import pytest
from scipy import stats

from crosscheck import chi2 as ours, tails
from evidence.plugin import record

ROOT = pathlib.Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.x64


@pytest.fixture(scope="module")
def dc():
    spec = importlib.util.spec_from_file_location("detection_campaign", ROOT / "scripts" / "detection_campaign.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def template(dc, tmp_path_factory):
    path = tmp_path_factory.mktemp("tails") / "template.fits"
    dc.observe(path, 0.0, 0.0, 0.0, None)
    return path


# --- the tools, against closed forms


def orthogonal_search(n=3, dim=6):
    """n unit-norm, orthogonal response vectors in dim dimensions, identity precision: the scores
    s_j are independent standard normal, so P(S >= t) = 1 - (1 - chi2_1.sf(t) / 2)^n exactly."""
    g = np.eye(dim)[:n]
    clean = lambda f, x, y: f * g[int(round(x))] if f else np.zeros(dim)  # noqa: E731
    return tails.Search(np.eye(dim), g), [(float(j), 0.0) for j in range(n)], clean


@pytest.mark.validates("crosscheck.tails", roots=["mathematics"], kind="reference")
def test_importance_sampling_recovers_the_exact_tail_of_a_maximum_of_scores():
    search, positions, clean = orthogonal_search()
    t = 16.0  # P = 3 * 3.2e-5
    exact = 1 - (1 - stats.chi2.sf(t, 1) / 2) ** 3
    mixture = tails.Mixture(search, positions, clean, t, max_flux=np.inf)
    rng = np.random.default_rng(1)
    n = 20000
    r = rng.standard_normal((n, 6))
    comp = rng.choice(mixture.pi.size, p=mixture.pi, size=n)
    r = r + mixture.shift[comp]  # noise about the component's mean
    p, se, ess, mean_w, se_w = tails.estimate(search.statistic(r), mixture.log_weight(r), t)
    record("is_estimate_over_exact", p / exact)
    assert abs(p - exact) < 4 * se and se / p < 0.03
    assert abs(mean_w - 1) < 4 * se_w  # the weights integrate to one
    assert ess > 1000


@pytest.mark.validates("crosscheck.tails", roots=["statistics"], kind="reference")
def test_gpd_fit_recovers_a_chi_squared_tail():
    x = stats.chi2.rvs(1, size=500_000, random_state=np.random.default_rng(4))
    t = stats.chi2.isf(1e-4, 1)
    p, (lo, hi) = tails.gpd_tail(x, t, tail_fraction=0.02, n_boot=60)
    assert lo < 1e-4 < hi or abs(np.log(p / 1e-4)) < 0.25
    record("gpd_over_exact", p / 1e-4)


@pytest.mark.validates("crosscheck.tails", roots=["mathematics"], kind="reference")
def test_equivalence_test_and_ratio_interval():
    # a ratio of 1 with s.e. 0.05 is within +-30%; a ratio of 1.2 is not shown to be
    assert tails.equivalence_pvalue(1.0, 0.05, 1.3) < 1e-6
    assert tails.equivalence_pvalue(1.2, 0.065, 1.3) > 0.05
    assert tails.equivalence_pvalue(1.4, 0.01, 1.3) > 0.999
    r, lo, hi = tails.rate_ratio_interval(2e-3, 1e-4, 1e-3, 5e-5)
    assert r == 2.0 and lo < 2.0 < hi and np.isclose(np.log(hi / r), np.log(r / lo))


# --- the likelihood's covariance is the simulators' noise


@pytest.mark.validates("virgil.oidata.OIData.cp_noise", roots=["literature"])
def test_our_closure_covariance_is_virgils(dc, template):
    """crosscheck.tails.precision (Kammerer et al. 2020: C = D^1/2 (T T^T / 3) D^1/2, generalised inverse)
    is the generalised inverse of the covariance virgil builds from the file: C W C = C and W C W = W."""
    from virgil.oidata import OIData

    data = OIData(str(template))
    d = ours.load(template)
    w = tails.precision(d)
    nv, nphi = data.vis.size, data.phi.size
    sigma = np.asarray(data.d_phi)
    c = np.zeros_like(w)
    c[:nv, :nv] = np.diag(np.asarray(data.d_vis) ** 2)
    c[nv:, nv:] = sigma[:, None] * data.cp_noise.correlation(nphi) * sigma[None, :]
    assert np.allclose(c @ w @ c, c, atol=1e-12 * np.abs(c).max())
    assert np.allclose(w @ c @ w, w, atol=1e-9 * np.abs(w).max())
    assert np.linalg.matrix_rank(w[nv:, nv:], tol=1e-6 * np.abs(w).max()) == data.cp_noise.size


@pytest.mark.validates("virgil.detection.gaussian_null", roots=["statistics"])
def test_both_simulators_draw_the_templates_covariance(dc, template):
    """3000 draws of the residual from each simulator have the closure-phase correlation R of the template
    (±1/3 between triangles sharing a baseline) and unit variance; ours is not independent per triangle."""
    import jax

    from crosscheck import simulate
    from virgil import detection, models
    from virgil.oidata import OIData

    data = OIData(str(template))
    nv, nphi = data.vis.size, data.phi.size
    r_theory = data.cp_noise.correlation(nphi)
    n = 3000
    grabbed = {}

    def capture(path, *, wavelengths, stations_xyz, vis2, t3):
        grabbed["phi"] = np.deg2rad(t3["phi"]).ravel()

    real, simulate.oifits_writer.write = simulate.oifits_writer.write, capture
    try:
        mine = []
        for k in range(n):
            simulate.observe("unused.fits", dc.vis_fn(0.0, 0.0, 0.0), dc.UTS4, rng=np.random.default_rng(900 + k), **dc.OBS)
            mine.append(grabbed["phi"])
    finally:
        simulate.oifits_writer.write = real
    sim = detection.gaussian_null(models.BinaryModelCartesian(0.0, 0.0, 0.0), data)
    theirs = np.array([np.asarray(sim(jax.random.PRNGKey(k)).phi) for k in range(n)])
    sigma = np.asarray(data.d_phi)
    for name, x in (("ours", np.array(mine)), ("virgil", theirs)):
        c = (x / sigma).T @ (x / sigma) / n
        z = (c - r_theory) / np.sqrt((1 + r_theory**2) / n)
        record(f"closure_cov_mean_z2_{name}", float(np.mean(z**2)))
        assert abs(np.mean(z**2) - 1) < 0.15 and np.abs(z).max() < 5
        assert (r_theory == 0).mean() < 1  # there is correlation to find
        assert np.abs(c[(r_theory != 0) & ~np.eye(nphi, dtype=bool)]).min() > 0.2


@pytest.mark.validates("virgil.detection.detection_statistics", roots=["statistics"])
def test_delta_chi2_is_the_linear_search_statistic(dc, template):
    """virgil's delta_chi2 on draws of either simulator equals S = max_j max(0, s_j)^2 within 0.05."""
    import jax

    from virgil import detection, models
    from virgil.oidata import OIData

    data = OIData(str(template))
    sim = detection.gaussian_null(models.BinaryModelCartesian(0.0, 0.0, 0.0), data)
    diffs = []
    for k in range(12):
        draw = sim(jax.random.PRNGKey(300 + k))
        diffs.append(dc.stats_of(draw, dc.GRID)["delta_chi2"] - dc.reference()["search"].statistic(dc.residual(draw))[0])
    record("max_abs_delta_chi2_minus_S", float(np.max(np.abs(diffs))))
    assert np.max(np.abs(diffs)) < 0.05


# --- the tails


@pytest.mark.slow
@pytest.mark.validates("virgil.detection.gaussian_null", roots=["statistics"], tier="B")
def test_simulators_agree_in_the_tail_at_one_percent_and_one_per_mille(dc):
    """Equivalence (within a factor 1.3) of the exceedance rates of S above 13.93 (P = 1e-2) and 19.03 (1e-3),
    2e4 draws of ours against 2e5 of virgil's: the version 3 `noise_tail` tests, at one fifth of their counts
    (so the 1e-3 count is a check, not a proof)."""
    mine = np.array(dc.draw_proxy("ours", 1, 20000))
    theirs = np.array(dc.draw_proxy("virgil", 5, 200000))
    res = dc.plain_ratio(mine, theirs, 13.93, 1.3)
    record("tail_ratio_1e-2", res["ratio"])
    assert res["p"] < 0.05 and 0.8 < res["ratio"] < 1.25
    low = dc.plain_ratio(mine, theirs, 19.03, 1.3)
    record("tail_ratio_1e-3", low["ratio"])
    assert low["ci95"][0] < 1.0 < low["ci95"][1]


@pytest.mark.slow
@pytest.mark.validates("virgil.detection.gaussian_null", roots=["statistics"], tier="B")
def test_importance_sampling_of_both_simulators_agrees_to_five_sigma(dc, tmp_path):
    """The two simulators' exceedance probabilities by importance sampling (2000 draws each, ours through
    a file and read by OIData as in the campaign, virgil's gaussian_null at the component's scene): equal within
    their errors at 1e-3, and at 5 sigma (2.9e-7), where plain Monte Carlo would need 10^9 draws."""
    import jax

    from virgil import detection, models
    from virgil.oidata import OIData

    template = tmp_path / "template.fits"
    dc.observe(template, 0.0, 0.0, 0.0, None)
    sim = detection.gaussian_null(models.BinaryModelCartesian(0.0, 0.0, 0.0), OIData(str(template)))
    thresholds = dc.criteria()["thresholds"]
    for name in ("1e-3", "5sigma"):
        t = thresholds[name]
        mine = [dc.draw_is(40_000 + k, name, tmp_path) for k in range(600)]
        theirs = [dc.draw_virgil_is(50_000 + k, name, sim) for k in range(600)]
        # keep the statistic the campaign uses for the importance estimate: the linear one is cheap
        est_m = tails.estimate([r["proxy"] for r in mine], [r["logw"] for r in mine], t)
        est_v = tails.estimate([r["proxy"] for r in theirs], [r["logw"] for r in theirs], t)
        for est in (est_m, est_v):
            assert abs(est[3] - 1) < 4 * est[4]
        ratio, lo, hi = tails.rate_ratio_interval(est_m[0], est_m[1], est_v[0], est_v[1])
        record(f"is_ratio_{name}", ratio)
        assert lo < 1.0 < hi
        assert 0.5 < est_m[0] / dc.NOMINAL[name] < 2.0


@pytest.mark.slow
@pytest.mark.validates("virgil.detection.gaussian_null", roots=["statistics"], tier="B")
def test_the_version_2_excess_is_a_fluctuation_of_its_sample(dc):
    """C3. The version 2 null seeds, through S, give the recorded 288 exceedances of 13.93 and 25 of 19.03
    (240 and 24 expected); the same number of draws from other seeds give the expectation."""
    v2 = np.array(dc.draw_proxy("ours", 20261006, 24000))
    k, k3 = int(np.sum(v2 >= 13.93)), int(np.sum(v2 >= 19.03))
    record("v2_replay_exceed_1e-2", k)
    assert k == 288 and k3 == 25, (k, k3)
    expected = 24000 * 1e-2
    assert (k - expected) / np.sqrt(expected * 0.99) > 2.5  # a 3-sigma sample
    fresh = [int(np.sum(np.array(dc.draw_proxy("ours", 20261006 + 10_000_000 * b, 24000)) >= 13.93)) for b in (1, 2, 3)]
    record("fresh_blocks_exceed_1e-2", float(np.mean(fresh)))
    assert abs(np.mean(fresh) - expected) < 3 * np.sqrt(expected * 0.99 / 3)  # the mean of three blocks
