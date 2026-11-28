"""scripts/postagb_parametric.py: a simulated ring is recovered, selected by the pre-registered rule, and the jackknife runs (synthetic data)."""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
try:  # needs virgil main (MultipoleDisk, Harmonics); skipped where the CI pin predates it
    import postagb_imaging as P  # noqa: E402,F401
    import postagb_parametric as Q  # noqa: E402
    from _postagb_parametric_synth import CFG, FLUXES, REFS, make_data, start  # noqa: E402
except ImportError:
    P = Q = None

pytestmark = [pytest.mark.skipif(P is None, reason="needs virgil main (MultipoleDisk) and virgil-vlti with virgil_vlti.with_night_terms (install the postagb extra)"),
              pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")]


@pytest.fixture(autouse=True)
def _synthetic(monkeypatch):
    monkeypatch.setattr(Q, "N_R", 24)
    monkeypatch.setattr(Q, "geometry_refs", lambda cfg: REFS)
    monkeypatch.setattr(Q, "header_fluxes", lambda cfg: FLUXES)


def _truth():
    return Q.build_scene(CFG, "M2", start())


def test_simulated_ring_is_recovered_and_selected_by_the_rule():
    """Design test (d): inc +-3 deg, PA +-5 deg, r0 +-5 %; M2 beats M1 and M0 by the section 5 rule; M2o is not accepted."""
    data = make_data(_truth(), frames=5, sigma_v2=0.005, sigma_cp_deg=0.5, seed=11)
    res = Q.fit_ladder(CFG, data, arms=("N1",), models=("M0", "M1", "M2", "M2o"), smoke=True, do_is=False, s=1.0, time_limit=40.0, log=lambda m: None)
    sel = Q.select(res, "N1")
    assert sel["selected"] == "M2", [(c["pair"], c["dlnZ"], c["dbic"]) for c in sel["chain"]]
    chain = {c["child"]: c for c in sel["chain"]}
    assert chain["M2"]["accepted"] and chain["M2"]["dlnZ"] >= 5 and chain["M2"]["dbic"] <= -10
    assert not chain["M2o"]["accepted"]
    d = sel["best"]["derived"]
    assert d["inc"] == pytest.approx(50.0, abs=3.0)
    assert abs((d["pa_mod180"] - 40.0 + 90.0) % 180.0 - 90.0) < 5.0
    assert d["r0"] == pytest.approx(4.0, rel=0.05)
    assert sel["is_ring"]
    # chi2r divides by N = independent rows + one (near-zero) penalty row per closure phase, so a good fit gives n_independent / N, not 1
    b = sel["best"]
    assert b["raw_chi2"] == pytest.approx(b["n_independent"] / b["n_rows"], abs=0.2) and b["own_chi2"] == pytest.approx(b["raw_chi2"], rel=1e-9)


def test_leave_one_night_out_jackknife():
    truth = _truth()
    subsets = {n: make_data(truth, nights=tuple(m for m in (0, 1, 2) if m != n), frames=4, sigma_v2=0.005, sigma_cp_deg=0.5, seed=20 + n) for n in (0, 1, 2)}
    vals = {"env.inc": 50.0, "env.pa": 40.0, "env.profile.radius": 4.0, "env.profile.sigma": 1.0, "env.harmonics.re": [0.5], "env.harmonics.im": [-0.4],
            "env.flux.ratio": 0.5, "env.flux.index": 0.0, "bg.flux.ratio": 0.1}
    r = Q.jackknife(CFG, subsets, "M2", vals, arm="N1", s=1.0, time_limit=40.0, log=lambda m: None)
    assert r["n_nights"] == 3 and r["n_ok"] == 3
    assert set(r["sd"]) == {"inc", "pa", "r0", "sigma", "c1", "s1", "f_env", "f_bg"}
    assert all(v is not None and v >= 0 for v in r["sd"].values())
    assert r["sd"]["inc"] < 10.0 and r["sd"]["r0"] < 1.0
