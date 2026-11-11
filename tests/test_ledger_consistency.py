"""The mismatch ledger is internally consistent: a status outside the
vocabulary cannot silently clear a known bug, every upstream or finding test
is cited by an unfixed entry, and an entry with no test says why."""

import ast
import pathlib
import re

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEDGER = yaml.safe_load((ROOT / "trust" / "ledger.yml").read_text())
STATUSES = {"open", "to-raise", "raised", "documented", "fixed"}
UNFIXED = {"open", "to-raise", "raised"}


KIND_RE = re.compile(r"""kind\s*=\s*["'](upstream|finding)["']""")


def _marker_kind(call):
    """The kind of a validates(...) call if it is upstream or finding, else None."""
    func = call.func
    if not (isinstance(func, ast.Attribute) and func.attr == "validates"):
        return None
    for kw in call.keywords:
        if (kw.arg == "kind" and isinstance(kw.value, ast.Constant)
                and kw.value.value in ("upstream", "finding")):
            return kw.value.value
    return None


def _scan_kind_markers():
    """(found, count): {test id: kind} for every test carrying a validates marker of kind
    upstream or finding (decorators on module-level functions, methods of classes, async
    functions, and module-level pytestmark, which marks every test in the file), and the
    number of such marker calls anywhere in the test files."""
    found, count = {}, 0
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text())
        count += sum(1 for n in ast.walk(tree) if isinstance(n, ast.Call) and _marker_kind(n))
        tests = [
            (n, f"tests/{path.name}::{n.name}") for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_")
        ]
        module_kinds = [
            _marker_kind(c) for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in n.targets)
            for c in ast.walk(n.value) if isinstance(c, ast.Call) and _marker_kind(c)
        ]
        for node, tid in tests:
            for kind in module_kinds:
                found[tid] = kind
            for dec in node.decorator_list:
                for c in ast.walk(dec):
                    if isinstance(c, ast.Call) and _marker_kind(c):
                        found[tid] = _marker_kind(c)
    return found, count


def _kind_tests():
    return _scan_kind_markers()[0]


def _regex_marker_count():
    return sum(len(KIND_RE.findall(p.read_text())) for p in (ROOT / "tests").glob("test_*.py")
               if p.name != pathlib.Path(__file__).name)


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_every_status_is_in_the_vocabulary():
    bad = {e["id"]: e.get("status") for e in LEDGER if e.get("status") not in STATUSES}
    assert not bad, bad


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_every_upstream_or_finding_test_is_cited_by_an_unfixed_entry_with_the_matching_ruling():
    problems = []
    for test, kind in _kind_tests().items():
        citing = [e for e in LEDGER if test in (e.get("evidence") or ())]
        if kind == "finding":
            ok = [e for e in citing if e.get("ruling", "") == "virgil"]
        else:
            ok = [e for e in citing if e.get("ruling", "").startswith("external:")]
        ok = [e for e in ok if e["status"] in UNFIXED]
        if not ok:
            problems.append((test, kind, [(e["id"], e["status"], e.get("ruling")) for e in citing]))
    assert not problems, problems


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_every_fixed_entry_cites_a_test():
    assert not [e["id"] for e in LEDGER if e["status"] == "fixed" and not e.get("evidence")]


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_an_entry_without_evidence_says_why_it_is_pending():
    bad = [e["id"] for e in LEDGER if not e.get("evidence") and not str(e.get("pending", "")).strip()]
    assert not bad, bad


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_the_ast_scan_sees_every_upstream_or_finding_marker():
    """A marker form the scanner cannot see (a new decorator style, say) would let its test
    escape the ledger check silently, so the AST count must equal a plain text count."""
    _, count = _scan_kind_markers()
    assert count == _regex_marker_count()
