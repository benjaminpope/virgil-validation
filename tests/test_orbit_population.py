"""Orbit population gate: criteria hash, seeding, helpers and the aggregator on a toy summary.
CPU-light; virgil is not run."""

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("orbit_population_script", ROOT / "scripts" / "orbit_population.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

REGISTERED = "b0a81b29cf9743602d4be134582d3581f5eb948070bd903ab6e135c068351ec2"


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_criteria_hash_is_the_registered_one():
    assert gate.criteria_hash() == REGISTERED


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_criteria_hash_covers_only_the_block():
    text = gate.gate_text()
    assert gate.criteria_hash(text + "\nA change log line.\n") == REGISTERED
    assert gate.criteria_hash(text.replace("master_seed: 20261028", "master_seed: 1")) != REGISTERED


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_spec_parses_and_sizes():
    s, repaired = gate.spec()
    assert s["master_seed"] == 20261028 and isinstance(repaired, bool)
    assert gate.strata_sizes() == {"A": 200, "B": 240, "N": 100, "D": 120}
    assert gate.locate(0) == ("A", 0) and gate.locate(200) == ("B", 0) and gate.locate(659) == ("D", 119)
    with pytest.raises(SystemExit):
        gate.locate(660)


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_draws_are_reproducible_and_distinct():
    from crosscheck import orbit_population as op

    a1, a2 = op.draw_system("A", 3), op.draw_system("A", 3)
    assert json.dumps(a1) == json.dumps(a2)
    assert json.dumps(op.draw_system("A", 4)) != json.dumps(a1)
    assert json.dumps(op.draw_system("B", 3)) != json.dumps(a1)
    d = op.draw_system("D", 0)  # D 0..59 are drawn as B
    assert d["kind"] == "B"


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_m6_limit_table_is_the_binomial_q99():
    s, _ = gate.spec()
    table = s["must_hold"]["M6_false_orbits"]["limits_by_n"]
    for n in (2, 15, 16, 44, 45, 83, 84, 100, 129):
        assert gate.m6_limit(n, table) == gate.binomial_q99(n, 0.01)


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_credible_set_and_poisson_binomial():
    chosen, mass = gate.credible_set([0.1, 0.6, 0.3], 0.85)
    assert chosen == [1, 2] and mass == pytest.approx(0.9)
    assert gate.poisson_binomial_p(5, [0.5] * 10, "two") == pytest.approx(1.0)
    assert gate.poisson_binomial_p(0, [0.9] * 40, "under") < 1e-6


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_fap_convention():
    f, ci = gate.fap(100.0, np.arange(99.0))
    assert f == pytest.approx(1 / 100) and ci[0] <= 0.0 + 1e-12


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_mixture_rank_is_uniform_for_a_calibrated_posterior():
    rng = np.random.default_rng(0)
    u = []
    for _ in range(400):
        truth = rng.normal()
        u.append(gate.mixture_u([rng.normal(size=199)], [1.0], truth, rng.uniform()))
    from scipy import stats

    assert stats.kstest(u, "uniform").pvalue > 0.01


def _fit(stratum, index, **kw):
    truth = dict(stratum=stratum, index=index, kind=stratum, template="T1", n_nights=6, band=1, span_days=100.0,
                 schedule="s", misspecified=False, misspecification=None, envelope_fraction=None, flux=0.1, c=0.0,
                 n_redraws=0, phase_coverage=0.8)
    bands = [dict(n=1, p=0.9, log_z=0.0, se_lnz=0.1, se_source="ess", flags=[]),
             dict(n=2, p=0.1, log_z=-2.0, se_lnz=0.1, se_source="ess", flags=[])]
    r = dict(part="fit", stratum=stratum, index=index, status="ok", truth=truth, bands=bands, seconds=1.0, peak_rss_mb=1.0,
             verdict=dict(value="orbit", winner_n=1), tempering=None)
    if stratum == "N":
        r["verdict"] = dict(value="no-orbit", winner_n=0)
    if stratum == "A":
        r["posterior"] = dict(sampled=[1], divergences=0, rhat={"1:period": 1.0}, ess={"1:period": 900.0}, covers95={q: True for q in ("period", "ecc", "a_mas")},
                              u={q: (index + 0.5) / 200 for q in gate.RANKED})
    elif stratum == "B":
        r["posterior"] = dict(sampled=[1], covers95={q: True for q in ("period", "ecc", "a_mas")})
    r.update(kw)
    return r


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_aggregate_on_a_toy_summary():
    s, _ = gate.spec()
    sizes = gate.strata_sizes()
    results = [_fit(k, i) for k in "ABN" for i in range(sizes[k])] + [_fit("D", i) for i in range(sizes["D"])]
    for r in results:
        r.update(criteria_hash=REGISTERED)
    results += [dict(_fit("B", i), part="oracle") for i in range(sizes["B"])]
    out = gate.aggregate_results(results, s, s["master_seed"])
    assert out["counts"] == {"A": 200, "B": 240, "N": 100, "D": 120, "oracle": 240}
    assert out["must_hold"]["M6"]["N_all"]["false_orbits"] == 0
    assert out["must_hold"]["M7"]["failed"] == 0
    assert out["must_hold"]["M4"]["misses"] == 0
    assert set(out["pvalues"]) >= {"M1:0.5", "M3a"} or "M3a" in out["pvalues"]
    # one confident wrong answer in B breaks M3b only if it exceeds the limit; false orbits in N break M6
    bad = [dict(r, verdict=dict(value="orbit", winner_n=1)) if r["stratum"] == "N" else r for r in results]
    assert not gate.aggregate_results(bad, s, s["master_seed"])["pass"]
