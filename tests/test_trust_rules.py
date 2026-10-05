"""The verdict rules of scripts/trust.py, one synthetic case per rule
(docs/method/index.md): what may verify a part of virgil, and what may not."""

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")


@pytest.fixture(scope="module")
def trust():
    spec = importlib.util.spec_from_file_location("trust", ROOT / "scripts" / "trust.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(test, obj, kind="check", outcome="passed", roots=("mathematics",), **extra):
    return {"record": "test", "test": test, "objects": [obj], "roots": list(roots), "kind": kind,
            "tier": "A", "outcome": outcome, "metrics": {}, **extra}


def evidence(tmp_path, name, records, date="2026-10-06T00:00:00+00:00", commit="c" * 40):
    path = tmp_path / name
    head = {"record": "run", "date": date, "virgil": {"commit": commit}, "runner": "test"}
    path.write_text("\n".join(json.dumps(x) for x in [head, *records]) + "\n")
    return path


def verdicts(trust, tmp_path, records, graph=None, ledger=(), files=None, golden=frozenset()):
    graph = graph or {"nodes": {"virgil.a": {"layer": "models"}}}
    paths = files or [evidence(tmp_path, "latest.jsonl", records)]
    runs, recs = trust.load_evidence(paths)
    model = trust.build(graph, list(ledger), recs, runs, None, golden=set(golden))
    return {k: v["verdict"] for k, v in model["nodes"].items()}, model


@pytest.mark.parametrize("kind", ["guard", "control", "reference", "definition"])
def test_only_checks_verify_a_part(trust, tmp_path, kind):
    got, _ = verdicts(trust, tmp_path, [record("t::x", "virgil.a", kind=kind)])
    assert got["virgil.a"] == "unchecked"
    got, _ = verdicts(trust, tmp_path, [record("t::x", "virgil.a")])
    assert got["virgil.a"] == "verified"


def test_regression_is_weak(trust, tmp_path):
    got, model = verdicts(trust, tmp_path, [record("t::x", "virgil.a", kind="regression")])
    assert got["virgil.a"] == "partly" and model["nodes"]["virgil.a"]["strong"] == []


def test_xfailed_check_does_not_count(trust, tmp_path):
    got, _ = verdicts(trust, tmp_path, [record("t::x", "virgil.a", outcome="xfailed")])
    assert got["virgil.a"] == "unchecked"


def test_virgils_own_tests_are_weak(trust, tmp_path):
    got, model = verdicts(trust, tmp_path, [record("t::x", "virgil.a", source="virgil-ci")])
    assert got["virgil.a"] == "partly" and model["nodes"]["virgil.a"]["weak"] == [trust.VIRGIL_CI]


def test_golden_sources_count_once_and_only_if_registered(trust, tmp_path):
    graph = {"nodes": {"virgil.a": {"layer": "models", "roots": 2}}}
    recs = [record("t::x", "virgil.a", roots=["golden:one"]), record("t::y", "virgil.a", roots=["golden:two"])]
    got, model = verdicts(trust, tmp_path, recs, graph, golden={"one", "two"})
    assert model["nodes"]["virgil.a"]["strong"] == [trust.GOLDEN] and got["virgil.a"] == "partly"
    got, _ = verdicts(trust, tmp_path, recs[:1], golden=set())
    assert got["virgil.a"] == "unchecked"


def test_newest_outcome_wins_but_a_campaign_survives_a_skip(trust, tmp_path):
    old = evidence(tmp_path, "old.jsonl", [record("t::x", "virgil.a")], date="2026-10-01")
    new = evidence(tmp_path, "new.jsonl", [record("t::x", "virgil.a", outcome="skipped")], date="2026-10-06")
    got, _ = verdicts(trust, tmp_path, None, files=[old, new])
    assert got["virgil.a"] == "unchecked"  # the old pass does not outlive the newer skip
    camp = evidence(tmp_path, "campaigns.jsonl", [record("t::x", "virgil.a")], date="2026-10-01")
    got, _ = verdicts(trust, tmp_path, None, files=[camp, new])
    assert got["virgil.a"] == "verified"  # a campaign summary is skipped by ordinary runs by design


@pytest.mark.parametrize("status,verdict", [("open", "bug"), ("to-raise", "bug"), ("raised", "bug"), ("fixed", "verified")])
def test_unfixed_virgil_findings_are_bugs(trust, tmp_path, status, verdict):
    ledger = [{"id": "F1", "ruling": "virgil", "status": status, "objects": ["virgil.a"]}]
    got, _ = verdicts(trust, tmp_path, [record("t::x", "virgil.a")], ledger=ledger)
    assert got["virgil.a"] == verdict


def test_open_external_problem_marks_the_package_partly(trust, tmp_path):
    graph = {"nodes": {"pkg": {"layer": "references"}}}
    ledger = [{"id": "P1", "ruling": "external:pkg", "status": "to-raise", "objects": ["pkg"]}]
    got, _ = verdicts(trust, tmp_path, [record("t::x", "pkg", kind="reference")], graph, ledger)
    assert got["pkg"] == "partly"


def test_references_are_verified_by_reference_checks_but_parts_of_virgil_are_not(trust, tmp_path):
    graph = {"nodes": {"crosscheck.x": {"layer": "references"}, "virgil.a": {"layer": "models"}}}
    recs = [record("t::r", "crosscheck.x", kind="reference"), record("t::s", "virgil.a", kind="reference")]
    got, _ = verdicts(trust, tmp_path, recs, graph)
    assert got == {"crosscheck.x": "verified", "virgil.a": "unchecked"}


def test_pipelines_count_only_checks(trust, tmp_path):
    graph = {"nodes": {"virgil.a": {"layer": "models"}}, "pipelines": {"p": {"title": "p", "steps": ["virgil.a"]}}}
    recs = [record("t::a", "virgil.a"), record("t::p", "pipeline:p", kind="reference")]
    _, model = verdicts(trust, tmp_path, recs, graph)
    assert model["pipelines"][0]["status"] == "planned"
    recs[1]["kind"] = "check"
    _, model = verdicts(trust, tmp_path, recs, graph)
    assert model["pipelines"][0]["status"] == "validated"


def test_failure_outweighs_passes(trust, tmp_path):
    got, _ = verdicts(trust, tmp_path, [record("t::x", "virgil.a"), record("t::y", "virgil.a", outcome="failed")])
    assert got["virgil.a"] == "failing"


def test_a_part_relying_on_an_unverified_part_is_not_verified(trust, tmp_path):
    graph = {"nodes": {"virgil.a": {"layer": "models", "depends": ["virgil.b"]}, "virgil.b": {"layer": "models"}}}
    got, _ = verdicts(trust, tmp_path, [record("t::x", "virgil.a")], graph)
    assert got == {"virgil.a": "relies-unverified", "virgil.b": "unchecked"}


def test_a_check_through_unverified_reference_code_does_not_verify(trust, tmp_path):
    graph = {"nodes": {"virgil.a": {"layer": "models"}, "crosscheck.r": {"layer": "references"}}}
    rec = record("t::x", "virgil.a", via=["crosscheck.r"])
    got, model = verdicts(trust, tmp_path, [rec], graph)
    assert got["virgil.a"] == "relies-unverified"
    assert [c["id"] for c in model["nodes"]["virgil.a"]["because"]] == ["crosscheck.r"]
    got, _ = verdicts(trust, tmp_path, [rec, record("t::r", "crosscheck.r", kind="reference")], graph)
    assert got["virgil.a"] == "verified"
    # a reference relying on an unverified reference is not trusted either
    graph["nodes"]["crosscheck.r"]["depends"] = ["crosscheck.s"]
    graph["nodes"]["crosscheck.s"] = {"layer": "references"}
    got, _ = verdicts(trust, tmp_path, [rec, record("t::r", "crosscheck.r", kind="reference")], graph)
    assert got["virgil.a"] == "relies-unverified"


def test_reference_code_missing_from_the_graph_is_not_trusted(trust, tmp_path):
    got, model = verdicts(trust, tmp_path, [record("t::x", "virgil.a", via=["crosscheck.nowhere"])])
    assert got["virgil.a"] == "relies-unverified"
