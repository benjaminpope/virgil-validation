"""Simulation-based calibration of virgil.likelihood.numpyro_model, from the
summary of the OzSTAR campaign (scripts/sbc_numpyro.py; ozstar_scripts job
sbc_numpyro).

The campaign (tier C) draws truths from Jeffreys priors, simulates data with
our own crosscheck.simulate, samples numpyro_model's posterior with NUTS, and
records each truth's rank among 99 posterior draws. Its criteria are
registered in the script (CRITERIA): rank uniformity by chi-squared with a
Bonferroni family-wise 1%, and 68% and 95% coverage within the 99%
binomial band, over at least 400 replicates. This test reads the committed
summary, checks it was made under the same criteria, and reports it; it
skips until the campaign has run.
"""

import importlib.util
import json
import pathlib

import pytest

from evidence.meta import virgil_source
from evidence.plugin import record

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "trust" / "campaigns" / "sbc_numpyro_model.json"
PREREGISTERED = 1000  # docs/method/index.md, "Monte Carlo campaigns for retrievals"


def script():
    spec = importlib.util.spec_from_file_location("sbc_numpyro", ROOT / "scripts" / "sbc_numpyro.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not SUMMARY.exists(), reason="the SBC campaign has not run (ozstar_scripts job sbc_numpyro)")
@pytest.mark.validates("virgil.likelihood.numpyro_model", roots=["statistics"], tier="C")
def test_sbc_ranks_are_uniform():
    summary = json.loads(SUMMARY.read_text())
    assert summary["criteria_hash"] == script().criteria_hash(), "criteria changed after the campaign ran"
    installed = virgil_source()["commit"]
    if summary.get("virgil_commit") != installed:
        pytest.skip(f"the campaign ran on virgil {str(summary.get('virgil_commit'))[:7]}, not the installed "
                    f"{str(installed)[:7]}: rerun it")
    for name, p in summary["parameters"].items():
        record(f"chi2_p_{name}", p["chi2_p"])
    record("replicates", summary["replicates"])
    # docs/method/index.md registered 1000 posteriors for this campaign before it ran;
    # the script's own minimum (400) came later and does not override it
    if summary["replicates"] < PREREGISTERED:
        pytest.skip(f"campaign incomplete: {summary['replicates']} of the {PREREGISTERED} preregistered replicates")
    assert summary["replicates"] >= summary["criteria"]["min_replicates"]
    failing = [k for k, p in summary["parameters"].items() if not p["pass"]]
    assert not failing, f"ranks not uniform or coverage off for {failing}"


@pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")
def test_sbc_ranks_and_criteria_are_well_formed():
    """The rank of a circular parameter is measured away from the wrap, and
    the registered criteria are what the docstrings say."""
    s = script()
    assert s.circular_rank([350.0, 355.0, 5.0, 10.0], 359.0) == 2
    assert s.circular_rank([170.0, 180.0, 190.0], 200.0) == 3
    assert s.CRITERIA["draws_per_replicate"] == 99 and s.CRITERIA["family_alpha"] == 0.01


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_sbc_aggregate_refuses_repeats_and_mixed_commits(tmp_path):
    """Replicates are counted once each, from one virgil commit, and the
    coverage band uses the exact null probability of the accepted ranks."""
    import argparse
    import subprocess
    import sys

    s = script()
    rep = {"seed": 1, "truth": {}, "ranks": {p: 50 for p in s.PARAMS}, "ess": {p: 1e3 for p in s.PARAMS},
           "rhat": {p: 1.0 for p in s.PARAMS}, "divergences": 0}
    task = {"task": 0, "virgil": "x", "virgil_commit": "a" * 40, "criteria_hash": s.criteria_hash(),
            "replicates": [rep]}
    (tmp_path / "task_0000.json").write_text(json.dumps(task))
    run = lambda *dirs: subprocess.run([sys.executable, str(ROOT / "scripts" / "sbc_numpyro.py"), "aggregate",  # noqa: E731
                                        *map(str, dirs)], capture_output=True, text=True)
    assert run(tmp_path).returncode == 0
    assert "repeat a seed" in run(tmp_path, tmp_path).stderr
    other = tmp_path / "other"
    other.mkdir()
    (other / "task_0001.json").write_text(json.dumps({**task, "virgil_commit": "b" * 40,
                                                      "replicates": [{**rep, "seed": 2}]}))
    assert "different or unknown virgil commits" in run(tmp_path, other).stderr
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "no results found" in run(empty).stderr
    L = s.CRITERIA["draws_per_replicate"]
    accepted = sum(abs(r - L / 2) < 0.95 * (L + 1) / 2 for r in range(L + 1))
    assert accepted == 94  # so the 95% interval's null coverage is 0.94, not 0.95


@pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")
def test_sbc_tasks_write_incrementally_and_resume(tmp_path, monkeypatch):
    """A task writes its file after every replicate and is marked complete
    only at the end; rerun after a stop, it keeps what was done and runs
    only the missing seeds (a stub stands in for NUTS)."""
    import argparse

    s = script()
    monkeypatch.setenv("PIN_COMMIT", "c" * 40)
    calls = []

    class TimeLimit(Exception):
        pass

    stop = {"after": 2}

    def stub(seed, *_):
        if stop["after"] is not None and len(calls) == stop["after"]:
            raise TimeLimit  # the time limit, after two replicates
        calls.append(seed)
        return {"seed": seed, "truth": {}, "ranks": {p: 1 for p in s.PARAMS}, "ess": {}, "rhat": {}, "divergences": 0}

    args = argparse.Namespace(task=3, replicates=4, seed_base=100, warmup=1, samples=1, chains=1, out=str(tmp_path))
    with pytest.raises(TimeLimit):
        s.run(args, replicate_fn=stub)
    saved = json.loads((tmp_path / "task_0003.json").read_text())
    assert [r["seed"] for r in saved["replicates"]] == [112, 113] and saved["complete"] is False
    calls.clear()
    stop["after"] = None
    s.run(args, replicate_fn=stub)
    saved = json.loads((tmp_path / "task_0003.json").read_text())
    assert calls == [114, 115]  # only the missing seeds
    assert [r["seed"] for r in saved["replicates"]] == [112, 113, 114, 115] and saved["complete"] is True
    monkeypatch.setenv("PIN_COMMIT", "d" * 40)
    with pytest.raises(SystemExit, match="another commit"):
        s.run(args, replicate_fn=stub)
