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

REGISTERED = "31fd09f6a81645fc7c2d34ebd2e3435530342dd514b0b57ab534925cfb96ceaa"


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
    s = gate.spec()
    assert s["master_seed"] == 20261028
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
    s = gate.spec()
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
    """A toy result: 90% of systems have their true period in the band virgil gives 0.9."""
    hit = index % 10 != 0
    truth = dict(stratum=stratum, index=index, kind=stratum, template="T1", n_nights=6, band=1 if hit else 2,
                 period=15.0 if hit else 25.0, span_days=100.0, schedule="s", misspecified=False,
                 misspecification=None, envelope_fraction=None, flux=0.1, c=0.0, n_redraws=0, phase_coverage=0.8)
    bands = [dict(n=1, p_lo=10.0, p_hi=20.0, p=0.9, log_z=0.0, se_lnz=0.1, se_source="ess", flags=[]),
             dict(n=2, p_lo=20.0, p_hi=30.0, p=0.1, log_z=-2.0, se_lnz=0.1, se_source="ess", flags=[])]
    r = dict(part="fit", stratum=stratum, index=index, status="ok", truth=truth, bands=bands, seconds=1.0, peak_rss_mb=1.0,
             verdict=dict(value="orbit", winner_n=1), tempering=None)
    if stratum == "N":
        r["verdict"] = dict(value="no-orbit", winner_n=0)
    if stratum == "A":
        r["posterior"] = dict(sampled=[1], divergences=0, rhat={"1:period": 1.0}, ess={"1:period": 900.0},
                              covers95={q: True for q in ("period", "ecc", "a_mas")},
                              u={q: (index + 0.5) / 200 for q in gate.RANKED})
    elif stratum == "B":
        r["posterior"] = dict(sampled=[1], covers95={q: True for q in ("period", "ecc", "a_mas")})
    r.update(kw)
    return r


def _toy():
    sizes = gate.strata_sizes()
    results = [_fit(k, i) for k in "ABND" for i in range(sizes[k])]
    return results + [dict(_fit("B", i), part="oracle", oracle_init_calls=1) for i in range(sizes["B"])]


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_aggregate_clean_toy_has_no_failures_but_is_incomplete():
    s = gate.spec()
    out = gate.aggregate_results(_toy(), s, s["master_seed"])
    assert out["counts"] == {"A": 200, "B": 240, "N": 100, "D": 120, "oracle": 240}
    assert out["rejected"] == []
    assert out["must_hold"]["M6"]["N_all"]["provisional_pass"] and out["must_hold"]["M4"]["provisional_pass"]
    assert out["must_hold"]["M7"]["pass"] is None  # the fail flags cannot be observed on this virgil commit
    assert out["pass"] is None and out["unavailable"]  # never True while registered tests were not evaluated


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_aggregate_flags_false_orbits_and_missing_oracles():
    s = gate.spec()
    toy = _toy()
    bad = [dict(r, verdict=dict(value="orbit", winner_n=1)) if r["stratum"] == "N" else r for r in toy]
    out = gate.aggregate_results(bad, s, s["master_seed"])
    assert out["must_hold"]["M6"]["N_all"]["provisional_pass"] is False and out["pass"] is False
    assert out["must_hold"]["M4"]["provisional_pass"] and out["rejected"] == []  # only M6 flipped
    few = [r for r in toy if r["part"] == "fit" or r["index"] < 10]
    out = gate.aggregate_results(few, s, s["master_seed"])
    assert out["must_hold"]["M4"]["provisional_pass"] is False and out["must_hold"]["M7"]["pass"] is False
    unhooked = [dict(r, oracle_init_calls=0) if r["part"] == "oracle" else r for r in toy]
    assert gate.aggregate_results(unhooked, s, s["master_seed"])["must_hold"]["M4"]["provisional_pass"] is False


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_true_band_is_read_from_the_reported_window():
    r = _fit("B", 1)
    assert gate.true_band(r)["n"] == 1
    r["truth"]["period"] = 20.5
    assert gate.true_band(r)["n"] == 2


@pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")
def test_elliptical_disk_matches_the_uniform_disk_and_a_numerical_transform():
    from crosscheck import orbit_population as op
    from crosscheck import sky

    rng = np.random.default_rng(1)
    u, v = rng.uniform(-60, 60, 12), rng.uniform(-60, 60, 12)
    wavel = 2.2e-6
    np.testing.assert_allclose(op.vis_elliptical_disk(u, v, wavel, 3.0, 1.0, 40.0),
                               sky.vis_uniform_disk(u, v, wavel, 3.0).real, atol=1e-12)
    # an ellipse (major 3 mas, ratio 0.5, major axis at PA 30 deg, N through E) against a sampled integral
    major, ratio, pa = 3.0, 0.5, 30.0
    pts = rng.uniform(-1, 1, (400000, 2)) * major / 2
    phi = np.radians(pa)  # coordinates along (major, minor) axes -> (East, North)
    keep = (pts[:, 0] / (major / 2)) ** 2 + (pts[:, 1] / (ratio * major / 2)) ** 2 <= 1
    a, b = pts[keep, 0], pts[keep, 1]
    east, north = a * np.sin(phi) + b * np.cos(phi), a * np.cos(phi) - b * np.sin(phi)
    fu, fv = sky._q(u, v, wavel)
    numerical = np.exp(-2j * np.pi * (np.outer(fu, east) + np.outer(fv, north))).mean(axis=1)
    np.testing.assert_allclose(op.vis_elliptical_disk(u, v, wavel, major, ratio, pa), numerical.real, atol=0.02)


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_virgil_judged_metrics_make_every_run_incomplete():
    s = gate.spec()
    out = gate.aggregate_results(_toy(), s, s["master_seed"])
    assert out["judged_by_virgil"] == ["M4", "M6", "M8"]
    assert out["must_hold"]["M4"]["judged_by"] == "virgil" and out["must_hold"]["M4"]["provisional_pass"]
    assert out["pass"] is None and out["label"].startswith("INCOMPLETE")


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_an_empty_subset_has_no_verdict_of_its_own():
    s = gate.spec()
    no_n = [r for r in _toy() if r["stratum"] != "N"]
    m6 = gate.aggregate_results(no_n, s, s["master_seed"])["must_hold"]["M6"]["N_all"]
    assert m6["provisional_pass"] is None  # nothing to test: not True
    assert gate.aggregate_results(no_n, s, s["master_seed"])["pass"] is not True


GATE_COMMIT = "05e50c28f0945dcd265f9a895db71bc22eff4a9b"  # virgil main this copy was taken from
GATE_SHA256 = "902c0e7097a858cdcf02cb4054f7c113cf9f66dc24ad6e771fa82c407d4e9c3e"


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_gate_copy_has_the_content_hash_of_virgil_main():
    import hashlib

    assert hashlib.sha256((ROOT / gate.CRITERIA_FILE).read_bytes()).hexdigest() == GATE_SHA256, (
        "design/orbit_gate_042.md is not the copy taken from virgil main: re-sync it and update GATE_COMMIT/GATE_SHA256")


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_gate_copy_is_byte_identical_to_virgil_at_the_pinned_commit():
    import os
    import subprocess

    virgil = pathlib.Path(os.environ.get("VIRGIL_REPO", pathlib.Path.home() / "code" / "virgil"))
    ref = subprocess.run(["git", "-C", str(virgil), "show", f"{GATE_COMMIT}:design/orbit_gate_042.md"], capture_output=True)
    if ref.returncode:
        pytest.skip("no virgil checkout (set VIRGIL_REPO); the content hash test above runs everywhere")
    assert (ROOT / gate.CRITERIA_FILE).read_bytes() == ref.stdout


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_stratum_a_noise_scales_and_d_nulls_have_discs():
    import math

    from crosscheck import orbit_population as pop

    scales = [pop.draw_system("A", i)["noise"] for i in range(40)]
    assert all(abs(math.log(n["s_v2"])) <= math.log(pop.S_MAX) and abs(math.log(n["s_cp"])) <= math.log(pop.S_MAX) for n in scales)
    assert len({n["s_v2"] for n in scales}) == 40 and any(n["s_v2"] != n["s_cp"] for n in scales)
    assert "s_v2" not in pop.draw_system("B", 0)["noise"]
    for i in (60, 89, 90, 119):  # D 60..119 are the N population 0..59; the first 30 carry a disc primary
        assert (pop.draw_system("D", i)["primary"]["disc_on"] == "single") == (i - 60 < pop.N_DISC_FIRST)


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_criteria_commit_is_the_last_commit_and_refuses_edits(tmp_path):
    import subprocess

    from crosscheck import preregistration as prereg

    def git(*a):
        return subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", *a],
                              check=True, capture_output=True, text=True).stdout.strip()

    git("init", "-q")
    (tmp_path / "gate.md").write_text("a\n")
    git("add", "gate.md")
    git("commit", "-qm", "one")
    first = git("rev-parse", "HEAD")
    assert prereg.criteria_commit("gate.md", root=tmp_path) == first
    (tmp_path / "other.txt").write_text("x\n")
    git("add", "other.txt")
    git("commit", "-qm", "unrelated")
    assert prereg.criteria_commit("gate.md", root=tmp_path) == first  # the last commit that touched the file
    (tmp_path / "gate.md").write_text("b\n")
    with pytest.raises(SystemExit):
        prereg.criteria_commit("gate.md", root=tmp_path)  # edited criteria cannot be run


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_runs_from_different_criteria_commits_are_not_combined():
    from crosscheck import preregistration as prereg

    assert prereg.require_same_commit([{"criteria_commit": "a"}, {"criteria_commit": "a"}]) == "a"
    for bad in ([{"criteria_commit": "a"}, {"criteria_commit": "b"}], [{"criteria_commit": "a"}, {}]):
        with pytest.raises(SystemExit):
            prereg.require_same_commit(bad)


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_the_recorded_criteria_commit_is_the_last_commit_that_touched_the_file(capsys):
    import subprocess

    from crosscheck import preregistration as prereg

    if subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", gate.CRITERIA_FILE],
                      capture_output=True, text=True).stdout.strip():
        pytest.skip("the gate file has uncommitted edits")
    want = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", gate.CRITERIA_FILE],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert prereg.criteria_commit(gate.CRITERIA_FILE) == want
    gate.cmd_criteria(None)  # the record a run writes uses the same value
    assert json.loads(capsys.readouterr().out)["criteria_commit"] == want
