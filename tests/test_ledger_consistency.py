"""The mismatch ledger is internally consistent: a status outside the
vocabulary cannot silently clear a known bug, every upstream or finding test
is cited by an unfixed entry, and an entry with no test says why."""

import ast
import pathlib

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEDGER = yaml.safe_load(open(ROOT / "trust" / "ledger.yml"))
STATUSES = {"open", "to-raise", "raised", "documented", "fixed"}
UNFIXED = {"open", "to-raise", "raised"}


def _kind_tests():
    """{test id: kind} for every test whose validates marker has kind upstream or finding."""
    found = {}
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        for node in ast.parse(path.read_text()).body:
            if not (isinstance(node, ast.FunctionDef) and node.name.startswith("test_")):
                continue
            for dec in node.decorator_list:
                if not (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
                        and dec.func.attr == "validates"):
                    continue
                for kw in dec.keywords:
                    if (kw.arg == "kind" and isinstance(kw.value, ast.Constant)
                            and kw.value.value in ("upstream", "finding")):
                        found[f"tests/{path.name}::{node.name}"] = kw.value.value
    return found


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_every_status_is_in_the_vocabulary():
    bad = {e["id"]: e.get("status") for e in LEDGER if e.get("status") not in STATUSES}
    assert not bad, bad


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_every_upstream_or_finding_test_is_cited_by_an_unfixed_entry_with_the_matching_ruling():
    problems = []
    for test, kind in _kind_tests().items():
        citing = [e for e in LEDGER if test in e["evidence"]]
        if kind == "finding":
            ok = [e for e in citing if e["ruling"] == "virgil"]
        else:
            ok = [e for e in citing if e["ruling"].startswith("external:")]
        ok = [e for e in ok if e["status"] in UNFIXED]
        if not ok:
            problems.append((test, kind, [(e["id"], e["status"], e["ruling"]) for e in citing]))
    assert not problems, problems


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_every_fixed_entry_cites_a_test():
    assert not [e["id"] for e in LEDGER if e["status"] == "fixed" and not e["evidence"]]


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_an_entry_without_evidence_says_why_it_is_pending():
    bad = [e["id"] for e in LEDGER if not e["evidence"] and not str(e.get("pending", "")).strip()]
    assert not bad, bad
