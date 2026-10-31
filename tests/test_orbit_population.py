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

REGISTERED = "b54cc5e71987cc2dbdbd1b44a7fe285be2f69c5ff54281463c914c47d3e1177b"


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
                 misspecification=None, envelope_fraction=None, flux=0.1, c=0.0, n_redraws=0, phase_coverage=0.8,
                 p_cut=1.0, below_p_cut=False)
    bands = [dict(n=1, p_lo=10.0, p_hi=20.0, p=0.9, log_z=0.0, se_lnz=0.1, se_source="ess", flags=[]),
             dict(n=2, p_lo=20.0, p_hi=30.0, p=0.1, log_z=-2.0, se_lnz=0.1, se_source="ess", flags=[])]
    r = dict(part="fit", stratum=stratum, index=index, status="ok", truth=truth, bands=bands, p_range=[0.2, 400.0], seconds=1.0, peak_rss_mb=1.0,
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
    assert out["counts"] == {"A": 200, "B": 240, "N": 100, "D": 120, "oracle": 240, "B_below_p_cut": 0}
    assert out["rejected"] == []
    assert {"M1:0.68", "M1:0.95", "M3a"} <= set(out["pvalues"])  # the M1 levels are the gate's, and each is evaluated
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


GATE_COMMIT = "4f8693197133553fbc6da0b2d9e4693a837eb82f"  # virgil#625 head this copy was taken from; becomes main when #625 merges
GATE_SHA256 = "03a7671dc044ce8aa6533ae699389a181778811cd84abd9f8ddcac9da4c0d98d"


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


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_null_verdict_reads_virgils_delta_log_z_not_a_decoy():
    modes = {"winner": {"n": 2}, "no_orbit": {"verdict": "no-orbit (weak)", "delta_log_z": -3.5, "delta": 99.0, "delta_nats": 98.0}}
    v = gate.null_verdict(modes)
    assert v["delta_nats"] == -3.5 and v["value"] == "no-orbit" and v["winner_n"] == 2
    assert gate.null_verdict({})["delta_nats"] is None


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
@pytest.mark.parametrize("drop,metric", [("A", None), ("B", "M4"), ("B", "M8"), ("N", "M6")])
def test_empty_subsets_never_give_a_pass_per_metric(drop, metric):
    s = gate.spec()
    out = gate.aggregate_results([r for r in _toy() if r["stratum"] != drop], s, s["master_seed"])
    assert out["pass"] is not True
    if metric:
        entry = out["must_hold"][metric]
        flat = [entry] if "provisional_pass" in entry else list(entry.values())
        assert all(e["provisional_pass"] is not True for e in flat if isinstance(e, dict) and "provisional_pass" in e)


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_m3b_with_no_confident_systems_has_no_pass():
    s = gate.spec()
    toy = [dict(r, bands=[dict(b, p=0.5) for b in r["bands"]]) if r["stratum"] == "B" else r for r in _toy()]
    out = gate.aggregate_results(toy, s, s["master_seed"])
    assert all(v["pass"] is None for v in out["must_hold"]["M3b"].values()) and out["pass"] is not True


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_the_criteria_content_check_does_not_skip(monkeypatch, tmp_path):

    monkeypatch.setattr(gate, "ROOT", tmp_path)  # a directory with no git object for the commit and no env hash
    (tmp_path / "design").mkdir()
    (tmp_path / "design" / "orbit_gate_042.md").write_text("g\n")
    monkeypatch.delenv("CRITERIA_FILE_SHA256", raising=False)
    with pytest.raises(SystemExit, match="not aggregating"):
        gate.verify_criteria_content("deadbeef")
    import hashlib

    monkeypatch.setenv("CRITERIA_FILE_SHA256", hashlib.sha256(b"g\n").hexdigest())
    gate.verify_criteria_content("deadbeef")
    monkeypatch.setenv("CRITERIA_FILE_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="not the file at criteria commit"):
        gate.verify_criteria_content("deadbeef")


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_criteria_commit_trusts_the_environment_only_outside_a_repository(monkeypatch, tmp_path):
    import subprocess

    from crosscheck import preregistration as prereg

    monkeypatch.setenv("CRITERIA_COMMIT", "abc")
    (tmp_path / "gate.md").write_text("a\n")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert prereg.criteria_commit("gate.md", root=tmp_path) == "abc"  # a git-archive snapshot
    subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
    with pytest.raises(SystemExit, match="not tracked"):  # in a repository the variable is ignored
        prereg.criteria_commit("gate.md", root=tmp_path)


def _below(r, every=4):
    """The toy B systems whose truth lies below P_cut (decided at the draw)."""
    return dict(r, truth=dict(r["truth"], below_p_cut=True, p_cut=40.0)) if r["stratum"] == "B" and r["index"] % every == 0 else r


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_p_cut_equals_the_engines_period_cut():
    """The independence rule allows this copy of the engine's P_cut only with a cross-check against it (the truth's a
    as ``field``, as the gate specifies)."""
    import os

    from crosscheck import orbit_population as pop

    gate_run = bool(os.environ.get("VV_GATE_RUN"))  # in a gate run a skipped cross-check is a failure
    try:
        aliases = importlib.import_module("virgil.aliases")
    except ImportError:
        aliases = None
    if aliases is None or not hasattr(aliases, "period_cut"):
        msg = "this virgil has no aliases.period_cut (virgil#625)"
        pytest.fail(msg) if gate_run else pytest.skip(msg)
    rng = np.random.default_rng(0)
    for _ in range(200):
        times = rng.uniform(0.0, 800.0, int(rng.integers(2, 13)))
        a, night, res = rng.uniform(1.0, 30.0), rng.uniform(0.0, 0.4), rng.uniform(1.0, 5.0)
        assert pop.p_cut(times, a, night, res) == pytest.approx(aliases.period_cut(times, a, night, resolution=res), rel=1e-12)
    for times in ([5.0, 5.0, 30.0, 90.0], [0.0, 10.0, 10.0]):  # repeated night times: both must merge them
        assert pop.p_cut(times, 3.0, 0.2, 2.0) == pytest.approx(aliases.period_cut(np.array(times), 3.0, 0.2, resolution=2.0), rel=1e-12)


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_b_draws_exclude_and_count_truths_below_p_cut_without_changing_the_stream():
    from crosscheck import orbit_population as pop

    systems = [pop.draw_system("B", i) for i in range(40)]
    assert all(s["below_p_cut"] == (s["orbit"]["period"] < s["p_cut"]) for s in systems)
    assert any(s["below_p_cut"] for s in systems) and not all(s["below_p_cut"] for s in systems)
    # Literals taken from origin/main's draw_system before p_cut existed (period, redraws, a, noise drawn after
    # the orbit): the stream is the registered one, so runs before and after this commit pair draw for draw.
    pinned = {3: (13.113151543301496, 0, 3.9355384005970357, 0.009677465716676437),
              7: (62.30282307210372, 0, 1.9052667611152811, 0.005315406235919474)}
    for i, (period, redraws, a_mas, sigma_v2) in pinned.items():
        s = systems[i]
        assert (s["orbit"]["period"], s["n_redraws"], s["orbit"]["a_mas"], s["noise"]["sigma_v2"]) == (period, redraws, a_mas, sigma_v2)


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_a_b_prime_truth_is_fitted_with_the_wide_window_and_no_other_b_truth_is():
    s = {"kind": "B", "below_p_cut": True, "span_days": 400.0}
    assert gate.settings_for(s) == {"period_range": (1.0, 1200.0)}
    assert gate.settings_for(dict(s, below_p_cut=False)) == {} and gate.settings_for(dict(s, kind="D")) == {}


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_aggregate_excludes_below_p_cut_truths_and_reports_them_as_b_prime():
    s = gate.spec()
    toy = [_below(r) for r in _toy()]
    out = gate.aggregate_results(toy, s, s["master_seed"])
    assert out["counts"]["B"] == 180 and out["counts"]["B_below_p_cut"] == 60 and out["counts"]["oracle"] == 180
    assert out["reported"]["B_prime"]["n"] == 60 and out["reported"]["B_prime"]["indices"][:3] == [0, 4, 8]


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_clean_toy_with_b_prime_passes_m4_and_m7_and_b_prime_failures_are_not_counted():
    s = gate.spec()
    toy = [_below(r) for r in _toy()]
    out = gate.aggregate_results(toy, s, s["master_seed"])
    assert out["must_hold"]["M4"]["provisional_pass"] and out["must_hold"]["M4"]["limit"] == s["must_hold"]["M4_search_misses"]["max_fraction"] * 180
    assert out["must_hold"]["M7"]["missing"] == {} and out["must_hold"]["M7"]["failed"] == 0
    # an excluded system's crashed oracle is not counted (its fit is known to be excluded); a crashed A fit with the
    # same index still counts
    broken = [dict(r, status="crashed") if r["part"] == "oracle" and r["truth"]["index"] == 0 else r for r in toy]
    assert gate.aggregate_results(broken, s, s["master_seed"])["must_hold"]["M7"]["failed"] == 0
    broken_a = [dict(r, status="crashed") if r["truth"]["index"] == 0 and r["truth"]["stratum"] == "A" else r for r in toy]
    assert gate.aggregate_results(broken_a, s, s["master_seed"])["must_hold"]["M7"]["failed"] == 1


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_window_disagreement_is_reported_both_ways_and_a_crashed_fit_is_an_m7_failure():
    s = gate.spec()
    toy = [_below(r) for r in _toy()]
    # a scored truth outside the engine's window (search failure) and an excluded truth inside it
    toy = [dict(r, p_range=[20.0, 400.0]) if r["stratum"] == "B" and r["part"] == "fit" and r["index"] % 4 == 1 else r for r in toy]
    out = gate.aggregate_results(toy, s, s["master_seed"])
    d = out["reported"]["window_disagreement"]
    assert d["scored_below_window"] > 0 and d["scored_above_window"] == 0 and d["excluded_inside_window"] == 60
    crashed = [dict(r, status="crashed", p_range=None) if r["stratum"] == "B" and r["index"] == 5 and r["part"] == "fit" else r for r in toy]
    assert gate.aggregate_results(crashed, s, s["master_seed"])["must_hold"]["M7"]["failed"] == 1


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_a_scored_b_fit_with_status_ok_but_no_recorded_window_fails_m7():
    s = gate.spec()
    clean = gate.aggregate_results(_toy(), s, s["master_seed"])["must_hold"]["M7"]
    assert clean["no_recorded_window"] == 0
    for missing in (None, []):
        toy = [dict(r, p_range=missing) if r["stratum"] == "B" and r["part"] == "fit" and r["index"] == 7 else r for r in _toy()]
        m7 = gate.aggregate_results(toy, s, s["master_seed"])["must_hold"]["M7"]
        assert m7["no_recorded_window"] == 1 and m7["pass"] is False
