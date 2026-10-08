"""Projected-orbit comparison (design/orbit_comparison.md): Thiele-Innes,
statistics A and B, folding, and the reference-samples rule. Numpy only."""

import numpy as np
import pytest
from scipy import stats

from crosscheck import orbits as co
from evidence.plugin import record

BASE = dict(P=800.0, e=0.45, t_p=120.0, a=100.0, inc=62.0, omega=40.0, Omega=110.0)
SIGMA = dict(P=2.0, e=0.01, t_p=1.0, a=0.3, inc=0.3, omega=0.4, Omega=0.4)
T_OBS = 120.0 + np.linspace(0.0, 900.0, 10)
T_MEAN = float(T_OBS.mean())


def draw(seed, n=4000, **shift):
    rng = np.random.default_rng(seed)
    return {k: BASE[k] + shift.get(k, 0.0) + SIGMA[k] * rng.standard_normal(n) for k in BASE}


def mirror(s):
    return {**s, "omega": s["omega"] + 180.0, "Omega": s["Omega"] + 180.0}


def both(ref, ours=None):
    ours = draw(1) if ours is None else ours
    ot, rt = co.predict_track(ours, T_OBS), co.predict_track(ref, T_OBS)
    return (co.per_epoch_d2(ot, rt)[0].max(), co.joint_d2(ot, rt, 0.05)["d2"],
            co.statistic_b(ours, ref, T_MEAN))


@pytest.mark.validates("crosscheck.orbits", roots=["mathematics"], kind="reference")
@pytest.mark.parametrize("e", [0.0, 0.3, 0.8])
def test_thiele_innes_reproduces_sky_position(e):
    """A, F give dDec and B, G give dRA, exactly as sky_position."""
    f = np.linspace(-3, 3, 13)
    a, inc, w, W = 7.0, 55.0, 130.0, 250.0
    A, B, F, G = co.thiele_innes(a, inc, w, W)
    E = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(f / 2), np.sqrt(1 + e) * np.cos(f / 2))
    x, y = np.cos(E) - e, np.sqrt(1 - e**2) * np.sin(E)
    dra, ddec = co.sky_position(f, e, inc, w, W, a)
    assert np.allclose(A * x + F * y, ddec, atol=1e-12)
    assert np.allclose(B * x + G * y, dra, atol=1e-12)


@pytest.mark.validates("crosscheck.orbits", roots=["mathematics"], kind="reference")
@pytest.mark.parametrize("e", [0.45, 0.85, 0.95])
def test_predict_track_matches_position(e):
    s = {k: np.array([v]) for k, v in {**BASE, "e": e}.items()}
    t = BASE["t_p"] + np.linspace(-0.6, 0.6, 41) * BASE["P"]  # spans periastron
    got = co.predict_track(s, t)[0]
    dra, ddec = co.position(t, BASE["P"], BASE["t_p"], e, BASE["inc"],
                            BASE["omega"], BASE["Omega"], BASE["a"])
    assert np.allclose(got[:, 0], dra, atol=1e-9) and np.allclose(got[:, 1], ddec, atol=1e-9)


@pytest.mark.validates("crosscheck.orbits", roots=["mathematics", "statistics"], kind="reference")
def test_mirror_is_invisible_to_a_and_b_but_not_to_raw_elements():
    ours, ref = draw(1), mirror(draw(2))
    dA, dJ, b = both(ref, ours)
    record("mirror_d2_B", b["d2"])
    null = both(draw(2), ours)
    # same posterior up to the mirror: indistinguishable from the unmirrored comparison
    assert dA == pytest.approx(null[0], rel=1e-6) and b["d2"] == pytest.approx(null[2]["d2"], rel=1e-6)
    assert stats.chi2.sf(b["d2"], 7) > 0.01
    pull = (ours["omega"].mean() - ref["omega"].mean()) / ours["omega"].std()
    assert abs(pull) > 100


@pytest.mark.validates("crosscheck.orbits", roots=["mathematics"], kind="reference")
def test_same_posterior_passes():
    dA, dJ, b = both(draw(2))
    ot, rt = co.predict_track(draw(1), T_OBS), co.predict_track(draw(2), T_OBS)
    j = co.joint_d2(ot, rt, 0.05)
    assert stats.chi2.sf(dA, 2) > 1e-3 and j["p"] > 0.01 and b["p"] > 0.01


def _large(ref, separated=False):
    dA, dJ, b = both(ref)
    ot, rt = co.predict_track(draw(1), T_OBS), co.predict_track(ref, T_OBS)
    j = co.joint_d2(ot, rt, 0.05)
    assert j["p"] < 1e-2
    if separated:
        assert co.max_track_separation(draw(1), ref) > 0.3
    record("control_d2_B", b["d2"])
    record("control_max_epoch_d2_A", dA)
    assert stats.chi2.sf(dA, 2) < 1e-3 / len(T_OBS)
    assert b["p"] < 1e-3


@pytest.mark.validates("crosscheck.orbits", roots=["mathematics"], kind="control")
def test_control_omega_plus_180_alone():
    ref = draw(2)
    _large({**ref, "omega": ref["omega"] + 180.0}, separated=True)


@pytest.mark.validates("crosscheck.orbits", roots=["mathematics"], kind="control")
def test_control_reversed_sense_of_motion():
    ref = draw(2)
    _large({**ref, "inc": 180.0 - ref["inc"]}, separated=True)


@pytest.mark.validates("crosscheck.orbits", roots=["mathematics"], kind="control")
@pytest.mark.parametrize("key", ["t_p", "P"])
def test_control_shift_in_tp_or_period(key):
    # 3 sigma of one posterior against C_ours + C_ref = 2 sigma^2 is d^2 ~ 4.5, which
    # chi^2_7 does not reject (p ~ 0.72), so the control uses 8 sigma (d^2 ~ 32).
    _large(draw(2, **{key: 8 * SIGMA[key]}))
    # at 3 sigma the d^2 still rises above the null by about the shift squared
    assert both(draw(2, **{key: 3 * SIGMA[key]}))[2]["d2"] > both(draw(2))[2]["d2"] + 2


@pytest.mark.validates("crosscheck.orbits", roots=["mathematics"], kind="reference")
def test_fold_and_mirror_fraction():
    rng = np.random.default_rng(0)
    W = np.where(rng.random(5000) < 0.3, 110.0 + 180.0, 110.0) + 5 * rng.standard_normal(5000)
    w = 40.0 + 0 * W + np.where(W > 200, 180.0, 0.0)
    assert co.mirror_fraction(W, 110.0) == pytest.approx(0.3, abs=0.03)
    wf, Wf, _ = co.fold_samples(w, W, 110.0)
    assert np.all(np.abs(Wf - 110.0) < 90) and np.allclose(wf, 40.0)


@pytest.mark.validates("crosscheck.orbits", roots=["mathematics"], kind="reference")
def test_eccentricity_to_zero_is_not_applicable_for_b():
    lo = {**draw(1), "e": np.full(4000, 0.01)}
    r = co.statistic_b(lo, lo, T_MEAN)
    assert r["applicable"] is False and np.isnan(r["d2"])


@pytest.mark.validates("crosscheck.orbits", roots=["mathematics"], kind="reference")
def test_marginals_only_reference_is_never_scored():
    marg = {k: (BASE[k], SIGMA[k]) for k in BASE}
    out = co.compare_orbits(draw(1), marg, T_OBS, ref_kind="marginals", sigma=0.05, rng=np.random.default_rng(3))
    assert out["scored"] is False
    assert out["ours_alone"]["scored"] is False and out["independent_draws"]["scored"] is False
    # independent draws inflate C_ref in position space only; both variants are reported
    assert out["independent_draws"]["joint"]["rank"] >= 1
    full = co.compare_orbits(draw(1), draw(2), T_OBS, sigma=0.05)
    assert full["scored"] is True and full["max_separation"] < 0.05


@pytest.mark.validates("crosscheck.orbits", roots=["mathematics"], kind="reference")
def test_tp_anchor_is_shared_when_mean_epoch_is_at_apastron():
    """With t_mean half a period after periastron the per-sample wrap would split t_p."""
    t_mean = BASE["t_p"] + BASE["P"] / 2
    ours = draw(1)
    th = co.projected_elements(ours, t_mean, co.periastron_anchor(ours, t_mean))
    assert th[:, 2].std() < 5 * SIGMA["t_p"]  # one mode, not two P apart
    assert co.statistic_b(ours, draw(2), t_mean)["p"] > 0.01
    b = co.statistic_b(draw(1), draw(2, t_p=8 * SIGMA["t_p"]), t_mean)
    assert b["p"] < 1e-3


@pytest.mark.validates("crosscheck.orbits", roots=["mathematics"], kind="control")
def test_control_residual_fires_for_offset_in_dropped_direction():
    rng = np.random.default_rng(0)
    n, N = 500, 4
    ours = np.zeros((n, N, 2)) + rng.standard_normal((n, 1, 1)) * np.array([1.0, 0.0])  # no spread in y
    ref = ours.copy() + np.array([0.0, 0.5])  # offset in the dropped direction
    j = co.joint_d2(ours, ref, 0.05)
    assert j["residual"] > 1 and j["residual_ok"] is False
