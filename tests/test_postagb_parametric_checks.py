"""scripts/postagb_parametric_checks.py: NUTS against the Laplace sds, injected night-gain variance, the injection verdict (synthetic data)."""

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


def test_nuts_under_night_terms_matches_the_laplace_sds():
    """Design test: NUTS (50 + 100) at a ring fit gives posterior sds within 30 % of the Laplace sds. The data carry the N2 night modes, which also
    settles that ``numpyro_model`` and ``gauss_newton_mass`` work with them, and ``_residual_jacobian`` with no GP latent."""
    st = start(f_env=0.5, f_bg=0.1)
    truth = Q.build_scene(CFG, "M1", st)
    raw = make_data(truth, frames=5, sigma_v2=0.004, sigma_cp_deg=0.4, seed=7)
    data = Q.noise_arm(raw, CFG, "N2", s=1.0)
    assert data.gains is not None and data.phase_offsets is not None
    pri = {k: v for k, v in Q.priors_for(CFG, truth, "M1", st).items() if k != "env.flux.index"}  # the index is nearly unconstrained over three channels
    res = Q.fit(truth, pri, data, max_steps=500)
    assert res.info["converged"]
    vals = {k: np.asarray(v, float) for k, v in res.values.items() if k in pri}
    ev = Q.param_evidence(res.model, pri, data)
    assert ev["sign"] > 0 and ev["log_norm"] != 0.0  # night terms are in the likelihood
    run = C.nuts_mode(res.model, pri, data, vals, warmup=50, samples=100, chains=1, seed=1, log=lambda m: None)
    assert run["draws"].shape == (1, 100, ev["k"]) and run["names"] == ev["names"]
    rec = C.summarise(run, ev["sd"])
    lo, hi = rec["sd_ratio_range"]
    assert 0.7 <= lo and hi <= 1.3, rec["params"]
    assert "single_chain" in rec["flags"]  # one chain cannot be called converged by the two-chain rule


def test_summarise_flags_a_stuck_chain_and_passes_a_good_one():
    rng = np.random.default_rng(0)
    good = dict(draws=rng.standard_normal((2, 2000, 3)), names=["a", "b", "c"], theta=np.zeros(3), diverging=[0, 0], seconds=[1.0, 1.0], time_box_hit=False)
    r = C.summarise(good, np.ones(3))
    assert r["flags"] == [] and r["rhat_max"] < 1.01 and r["ess_min"] > 400
    stuck = dict(good, draws=good["draws"] + np.array([0.0, 3.0])[:, None, None])  # the second chain sits elsewhere
    r = C.summarise(stuck, np.ones(3))
    assert "not_converged" in r["flags"] and r["rhat_max"] > 1.01
    r = C.summarise(good, np.full(3, 0.2))
    assert "laplace_sd_mismatch" in r["flags"]
    r = C.summarise(dict(good, draws=good["draws"][:1], time_box_hit=True))
    assert "single_chain" in r["flags"]
