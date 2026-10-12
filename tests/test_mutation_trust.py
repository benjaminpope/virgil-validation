"""Mutation results on the Trust page and the sign-off checklist (scripts/mutation_trust.py, scripts/trust.py).

Synthetic reports and a made-up graph: no virgil, no mutant run.
"""

import importlib.util
import json
import pathlib

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mt():
    return _load("mutation_trust")


@pytest.fixture(scope="module")
def trust():
    return _load("trust")


def mutant(id, cls="pa_origin", nodes=("virgil.a",), **extra):
    return {"id": id, "cls": cls, "nodes": list(nodes), "description": f"slip {id}.", **extra}


def row(id, status):
    return {"id": id, "status": status}


def report(full=(), targeted=()):
    return {"mutants": [row(*x) for x in full], "targeted": {i: row(i, s) for i, s in targeted}}


def test_a_mutant_is_killed_if_any_run_killed_it(mt):
    listed = {m["id"]: m for m in [mutant("M1"), mutant("M2"), mutant("M3"), mutant("M4"), mutant("M5", equivalent="same matrix")]}
    rep = report(full=[("M1", "timeout"), ("M2", "survived"), ("M4", "timeout")],
                 targeted=[("M1", "survived"), ("M2", "killed"), ("M3", "survived"), ("M5", "survived")])
    got = {i: mt.mutant_status(i, rep, listed) for i in listed}
    assert got == {"M1": "survived", "M2": "killed", "M3": "survived", "M4": "unresolved", "M5": "equivalent"}
    assert mt.mutant_status("M9", rep, listed) == "not run"


def test_node_summary_counts_and_says_mutants_not_run(mt):
    listed = [mutant("M1", nodes=["virgil.a", "virgil.b"]), mutant("M2", nodes=["virgil.a"]), mutant("M3", nodes=["virgil.c"])]
    got = mt.node_summary(report(full=[("M1", "killed"), ("M2", "survived")]), listed)
    assert got["virgil.a"]["survivors"] == ["M2"] and got["virgil.a"]["killed"] == 1
    assert "M2" in got["virgil.a"]["text"] and "survive" in got["virgil.a"]["text"]
    assert got["virgil.b"]["text"].startswith("all 1 mutants killed")
    assert got["virgil.c"]["text"].startswith("mutants not run")
    assert "virgil.d" not in got


def test_an_equivalent_mutant_counts_for_nothing(mt):
    listed = [mutant("M1", equivalent="the same"), mutant("M2")]
    got = mt.node_summary(report(full=[("M1", "survived"), ("M2", "killed")]), listed)["virgil.a"]
    assert got["survivors"] == [] and got["equivalent"] == 1 and got["text"].startswith("all 1 mutants killed")


def _build(trust, tmp_path, mutation):
    graph = {"nodes": {"virgil.a": {"layer": "models"}, "virgil.b": {"layer": "models", "depends": ["virgil.a"]}}}
    rec = lambda obj: {"record": "test", "test": f"t::{obj}", "objects": [obj], "roots": ["mathematics"], "kind": "check",
                       "tier": "A", "outcome": "passed", "metrics": {}}
    path = tmp_path / "e.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in [
        {"record": "run", "date": "2026-10-06T00:00:00+00:00", "virgil": {"commit": "c" * 40}, "runner": "test"},
        rec("virgil.a"), rec("virgil.b")]) + "\n")
    runs, recs = trust.load_evidence([path])
    return trust.build(graph, [], recs, runs, None, golden=set(), signoffs={}, mutation=mutation), graph


def test_the_page_builds_without_a_report_and_is_unchanged(trust, tmp_path):
    model, _ = _build(trust, tmp_path, None)
    assert {n["verdict"] for n in model["nodes"].values()} == {"verified"}
    assert all(n["mutation"] is None for n in model["nodes"].values()) and model["mutation"] is None
    page = trust.render(model)
    assert "Mutation testing" not in page and "Mutants:" not in page
    assert trust.load_mutation(tmp_path / "absent.json") is None


def test_a_surviving_mutant_stops_a_part_being_verified(trust, mt, tmp_path):
    listed = [mutant("M1", nodes=["virgil.a"]), mutant("M2", nodes=["virgil.a"])]
    data = mt.page_data(report(full=[("M1", "killed"), ("M2", "survived")]), listed)
    model, _ = _build(trust, tmp_path, data)
    a, b = model["nodes"]["virgil.a"], model["nodes"]["virgil.b"]
    assert a["verdict"] == "partly" and a["mutation"]["survivors"] == ["M2"]
    assert b["verdict"] == "relies-unverified"  # it rests on a part that is no longer verified
    assert b["mutation"]["state"] == "none" and "no mutants written" in b["mutation"]["text"]
    page = trust.render(model)
    assert "M2" in page and "1 survive" in page
    assert "mutants survive: M2" in trust.render_coverage(model)


def test_killed_or_unrun_mutants_leave_the_verdict(trust, mt, tmp_path):
    listed = [mutant("M1", nodes=["virgil.a"]), mutant("M2", nodes=["virgil.b"])]
    model, _ = _build(trust, tmp_path, mt.page_data(report(full=[("M1", "killed")]), listed))
    assert {n["verdict"] for n in model["nodes"].values()} == {"verified"}
    assert model["nodes"]["virgil.a"]["mutation"]["state"] == "killed"
    assert model["nodes"]["virgil.b"]["mutation"]["text"].startswith("mutants not run")


def test_signoff_candidates_are_convention_survivors_only(mt):
    listed = [mutant("M1", "pa_origin", ["virgil.a", "virgil.b"]), mutant("M2", "mas_rad", ["virgil.c"]),
              mutant("M3", "uv_swap", ["virgil.d"]), mutant("M4", "pa_sign", ["virgil.e"], equivalent="x"),
              mutant("M5", "pa_origin", ["virgil.f"])]
    rep = report(full=[("M1", "survived"), ("M2", "survived"), ("M3", "killed"), ("M4", "survived")], targeted=[("M5", "survived")])
    got = mt.signoff_candidates(rep, listed, signed={"virgil.f": {"position_angle_origin": {"by": "Ben", "date": "2026-10-12", "commit": "c" * 40}}})
    assert sorted(got) == [("virgil.a", "position_angle_origin"), ("virgil.b", "position_angle_origin")]
    assert got[("virgil.a", "position_angle_origin")]["mutants"] == ["M1"]
    assert "[ ] virgil.a" in mt.render_checklist(got)
    assert "nothing new" in mt.render_checklist({})


def test_pending_entries_are_written_refreshed_and_confirm_nothing(mt, trust, tmp_path):
    path = tmp_path / "signoffs.yml"
    path.write_text("# header\n# more\nvirgil.x:\n  calibration:\n    by: Ben\n    date: 2026-10-06\n    commit: " + "a" * 40 + "\n")
    listed = [mutant("M1", "pa_origin", ["virgil.a"])]
    cands = mt.signoff_candidates(report(full=[("M1", "survived")]), listed)
    assert mt.write_pending(cands, path) == [("virgil.a", "position_angle_origin")]
    data = yaml.safe_load(path.read_text())
    entry = data["virgil.a"]["position_angle_origin"]
    assert entry["status"] == "pending" and entry["signed"] is False and entry["mutants"] == ["M1"]
    assert data["virgil.x"]["calibration"]["by"] == "Ben" and path.read_text().startswith("# header\n# more\n")
    assert "virgil.a" not in trust.load_signoffs(path) and "virgil.x" in trust.load_signoffs(path)
    # a second run refreshes the pending entry, and never overwrites a confirmed one
    data["virgil.x"]["position_angle_origin"] = {"by": "Ben", "date": "2026-10-12", "commit": "b" * 40, "note": "ok"}
    path.write_text(yaml.safe_dump(data))
    cands = {("virgil.x", "position_angle_origin"): {"mutants": ["M9"], "descriptions": ["d"]},
             ("virgil.a", "position_angle_origin"): {"mutants": ["M1", "M2"], "descriptions": ["d"]}}
    assert mt.write_pending(cands, path) == [("virgil.a", "position_angle_origin")]
    after = yaml.safe_load(path.read_text())
    assert after["virgil.x"]["position_angle_origin"]["by"] == "Ben"
    assert after["virgil.a"]["position_angle_origin"]["mutants"] == ["M1", "M2"]


def test_the_committed_mutant_list_is_tagged_and_the_report_renders(mt):
    """Every regression mutant names a fixed ruling-virgil finding of the ledger; M64 is marked equivalent."""
    listed = mt.load_listed()
    ledger = {e["id"]: e for e in yaml.safe_load(open(ROOT / "trust" / "ledger.yml"))}
    tagged = [m for m in listed if m.get("finding")]
    assert len(tagged) >= 15
    for m in tagged:
        assert ledger[m["finding"]]["ruling"] == "virgil" and ledger[m["finding"]]["status"] == "fixed", m["id"]
    assert {m["finding"] for m in tagged} >= {"F2", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "F13", "F14", "F15", "F16", "F18"}
    assert next(m for m in listed if m["id"] == "M64").get("equivalent")
    assert mt.page_data(None, listed) is None
