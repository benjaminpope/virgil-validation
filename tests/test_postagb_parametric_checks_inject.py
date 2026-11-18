"""scripts/postagb_parametric_checks.py: injected night-gain variance, the injection verdict and a recovery under N2 (synthetic data)."""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
try:  # needs virgil main (MultipoleDisk, Harmonics); skipped where the CI pin predates it
    import postagb_imaging as P  # noqa: E402,F401
    import postagb_parametric as Q  # noqa: E402
    import postagb_parametric_checks as C  # noqa: E402
    from _postagb_parametric_synth import CFG, FLUXES, REFS, make_data, start  # noqa: E402
except ImportError:
    P = Q = C = None

pytestmark = [pytest.mark.skipif(P is None, reason="virgil without MultipoleDisk (install the postagb extra)"),
              pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")]


@pytest.fixture(autouse=True)
def _synthetic(monkeypatch):
    monkeypatch.setattr(Q, "N_R", 24)
    monkeypatch.setattr(Q, "geometry_refs", lambda cfg: REFS)
    monkeypatch.setattr(Q, "header_fluxes", lambda cfg: FLUXES)


def _truth(model_id="M1", **kw):
    return Q.build_scene(CFG, model_id, start(**kw))


def test_injected_night_terms_have_the_requested_variance():
    """Design test: with negligible white noise the drawn log|V| gains per (night, baseline) have sd 0.03 and the closure offsets per (night, triangle) 2 deg,
    and the noise on a sample not touched by either is s times the quoted error."""
    truth = _truth("M1")
    s = 1.3
    raw = make_data(truth, nights=(0, 1, 2, 3), frames=3, sigma_v2=1e-8, sigma_cp_deg=1e-6, seed=0, noise=False)
    v2, t3 = P.observables(truth, raw)
    lab = Q.night_labels(raw)
    g, o = [], []
    for seed in range(30):
        sim = C.simulate(truth, raw, CFG, s, seed)
        lg = 0.5 * np.log(np.asarray(sim.vis) / np.asarray(v2))
        dt = np.degrees(np.asarray(sim.phi)) - np.asarray(t3)
        dt = (dt + 180.0) % 360.0 - 180.0
        g += [lg[lab["u_cell"] == c].mean() for c in np.unique(lab["u_cell"])]
        o += [dt[lab["t3_cell"] == c].mean() for c in np.unique(lab["t3_cell"])]
    assert np.std(g) == pytest.approx(0.03, rel=0.15) and abs(np.mean(g)) < 0.01
    assert np.std(o) == pytest.approx(2.0, rel=0.15) and abs(np.mean(o)) < 0.5
    # the same cell shares one gain: the spread inside a (night, baseline) cell is the (negligible) white noise
    sim = C.simulate(truth, raw, CFG, s, 0)
    lg = 0.5 * np.log(np.asarray(sim.vis) / np.asarray(v2))
    assert max(np.ptp(lg[lab["u_cell"] == c]) for c in np.unique(lab["u_cell"])) < 1e-4
    # white noise scales with s: nights off, quoted error 0.01
    raw2 = make_data(truth, nights=(0, 1, 2), frames=6, sigma_v2=0.01, sigma_cp_deg=1.0, seed=0, noise=False)
    arm = Q.noise_arm(raw2, CFG, "N0", s=s)
    assert arm is raw2
    inj = raw2.with_error_scale(s).with_model(truth, key=jax_key(3))
    assert np.std(np.asarray(inj.vis) - np.asarray(P.observables(truth, raw2)[0])) == pytest.approx(s * 0.01, rel=0.15)


def jax_key(i):
    import jax

    return jax.random.PRNGKey(i)


def test_simulate_returns_quoted_errors_and_reproduces_the_n2_arm():
    truth = _truth("M1")
    raw = make_data(truth, sigma_v2=0.01, sigma_cp_deg=1.0, seed=2)
    sim = C.simulate(truth, raw, CFG, 1.2, 5)
    assert np.array_equal(np.asarray(sim.d_vis), np.asarray(raw.d_vis)) and np.array_equal(np.asarray(sim.d_phi), np.asarray(raw.d_phi))
    assert not np.array_equal(np.asarray(sim.vis), np.asarray(raw.vis))
    assert np.array_equal(np.asarray(C.simulate(truth, raw, CFG, 1.2, 5).vis), np.asarray(sim.vis))  # fixed seed, same draw
    assert not np.array_equal(np.asarray(C.simulate(truth, raw, CFG, 1.2, 6).vis), np.asarray(sim.vis))
    arm = Q.noise_arm(sim, CFG, "N2", s=1.2)
    assert arm.gains is not None and np.asarray(arm.d_vis) == pytest.approx(1.2 * np.asarray(raw.d_vis))


def test_inject_verdict_branches():
    truth = dict(inc=50.0, pa_mod180=40.0)
    good = lambda sd, sel="M2", inc=51.0, pa=45.0: dict(seed=sd, selected=sel, inc=inc, pa_mod180=pa)  # noqa: E731
    v = C.inject_verdict("M2", truth, [good(0), good(1), good(2, sel="M1")])
    assert v["passed"] and v["n_rung_recovered"] == 2
    assert not C.inject_verdict("M2", truth, [good(0), good(1, sel="M1"), good(2, sel="M1")])["passed"]  # one of three
    assert not C.inject_verdict("M2", truth, [good(0), good(1), good(2, sel="M2o")])["passed"]  # a higher rung accepted
    assert not C.inject_verdict("M2", truth, [good(0), good(1, inc=65.0), good(2)])["passed"]  # inc off by 15
    assert not C.inject_verdict("M2", truth, [good(0), good(1, pa=170.0), good(2)])["passed"]  # PA off by 130 mod 180 = 50
    assert C.inject_verdict("M2", truth, [good(0, pa=175.0), good(1, pa=35.0), good(2)])["passed"] is False  # 175 vs 40 is 135 apart mod 180 -> 45
    assert C.inject_verdict("M2", truth, [good(0, pa=30.0), good(1, pa=55.0), good(2)])["passed"]  # 10 and 15 deg: inside
    face = dict(inc=10.0, pa_mod180=40.0)
    assert C.inject_verdict("M2", face, [good(0, inc=12.0, pa=140.0), good(1, inc=8.0, pa=100.0), good(2, sel="M1")])["passed"]  # PA waived below inc 20
    assert not C.inject_verdict("M2", truth, [dict(seed=0, selected=None, inc=None, pa_mod180=None)] * 3)["passed"]


def test_injection_recovers_a_simulated_ring_under_n2():
    """One injection seed through the same ladder code as the real fit (N2, one structure to keep it inside two minutes): the ring is fitted from
    data with drawn night terms and its geometry recovered; the rung-choice logic is covered by ``Q.select`` and ``inject_verdict`` tests."""
    st = start(f_env=0.5, f_bg=0.1)
    truth = Q.build_scene(CFG, "M1", st | dict(r=4.0, sigma=0.7))
    raw = make_data(truth, nights=(0, 1, 2), frames=3, sigma_v2=0.004, sigma_cp_deg=0.4, seed=1)
    row = C.inject_seed(CFG, raw, truth, 1.0, 0, models=("M1",), smoke=True, time_limit=25.0, log=lambda m: None)
    assert row["selected"] == "M1" and row["accepted"] == [] and row["models"]["M1"]["eligible"]
    assert row["inc"] == pytest.approx(50.0, abs=10.0) and C._circ180(row["pa_mod180"], 40.0) < 15.0
    v = C.inject_verdict("M1", dict(inc=50.0, pa_mod180=40.0), [row], models=("M1",), need=1)
    assert v["passed"]
