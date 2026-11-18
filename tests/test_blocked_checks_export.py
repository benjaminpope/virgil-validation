"""The model says which counted checks wait on an unverified ``via`` reference (display only, no verdict changes)."""

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")


@pytest.fixture(scope="module")
def trust():
    spec = importlib.util.spec_from_file_location("trust", ROOT / "scripts" / "trust.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(test, obj, kind="check", outcome="passed", roots=("mathematics",), **extra):
    return {"record": "test", "test": test, "objects": [obj], "roots": list(roots), "kind": kind,
            "tier": "A", "outcome": outcome, "metrics": {}, **extra}


def build(trust, tmp_path, records, graph):
    path = tmp_path / "latest.jsonl"
    head = {"record": "run", "date": "2026-10-06T00:00:00+00:00", "virgil": {"commit": "c" * 40}, "runner": "test"}
    path.write_text("\n".join(json.dumps(x) for x in [head, *records]) + "\n")
    runs, recs = trust.load_evidence([path])
    return trust.build(graph, [], recs, runs, None, golden=set())["nodes"]


GRAPH = {"nodes": {"virgil.a": {"layer": "models", "roots": 2},
                   "virgil.b": {"layer": "models"},
                   "crosscheck.r": {"layer": "references"}}}


def test_blocked_checks_are_listed_whatever_the_verdict(trust, tmp_path):
    recs = [record("t::via", "virgil.a", via=["crosscheck.r"]),
            record("t::direct", "virgil.a", roots=["standards"]),
            record("t::b", "virgil.b", via=["crosscheck.r"])]
    nodes = build(trust, tmp_path, recs, GRAPH)
    # virgil.a needs two roots; only the direct one is counted, the other waits on the reference
    a = nodes["virgil.a"]
    assert a["verdict"] == "relies-unverified" and a["strong"] == ["standards"]
    assert a["blocked_checks"] == [{"test": "t::via", "via": ["crosscheck.r"]}]
    assert a["blocked_by_references"] == ["crosscheck.r"]
    # virgil.b is ok through the unverified reference: the same export
    assert nodes["virgil.b"]["verdict"] == "relies-unverified"
    assert nodes["virgil.b"]["blocked_by_references"] == ["crosscheck.r"]
    assert nodes["crosscheck.r"]["blocked_checks"] == []


def test_nothing_is_blocked_once_the_reference_is_verified(trust, tmp_path):
    recs = [record("t::via", "virgil.b", via=["crosscheck.r"]), record("t::r", "crosscheck.r", kind="reference")]
    nodes = build(trust, tmp_path, recs, GRAPH)
    assert nodes["virgil.b"]["verdict"] == "verified"
    assert nodes["virgil.b"]["blocked_checks"] == [] and nodes["virgil.b"]["blocked_by_references"] == []


def test_coverage_page_says_waiting_not_checked_against_itself(trust, tmp_path):
    graph = {"nodes": {"virgil.a": {"layer": "models", "roots": 2}, "virgil.c": {"layer": "models", "roots": 2},
                       "crosscheck.r": {"layer": "references"}}}
    recs = [record("t::via", "virgil.a", via=["crosscheck.r"]),                       # all its checks wait
            record("t::own", "virgil.c", kind="regression")]                          # only virgil's own checks
    path = tmp_path / "latest.jsonl"
    head = {"record": "run", "date": "2026-10-06T00:00:00+00:00", "virgil": {"commit": "c" * 40}, "runner": "test"}
    path.write_text("\n".join(json.dumps(x) for x in [head, *recs]) + "\n")
    runs, records = trust.load_evidence([path])
    model = trust.build(graph, [], records, runs, None, golden=set())
    assert model["nodes"]["virgil.a"]["verdict"] == "partly" and model["nodes"]["virgil.a"]["strong"] == []
    page = trust.render_coverage(model, extra=None)
    row = {line.split("|")[1].strip(): line for line in page.splitlines() if line.startswith("| `")}
    assert "waiting on an unverified reference (`crosscheck.r`)" in row["`a`"]
    assert "checked against itself" not in row["`a`"]
    assert "only checked against itself" in row["`c`"]
