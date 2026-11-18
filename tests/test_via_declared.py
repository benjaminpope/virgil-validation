"""A reference loaded by a computed name at run time is invisible to the file scan of ``via`` (src/evidence/via.py)
and to its import recorder (``importlib`` does not pass through ``__import__``), so a test file that does it must
declare ``via=[...]`` on its markers."""

import ast
import pathlib
import re

import pytest

from evidence import via

ROOT = pathlib.Path(__file__).resolve().parents[1]
DYNAMIC_LOADERS = {"import_module", "__import__", "spec_from_file_location"}
REFERENCE_NAME = re.compile(r"\b(?:" + "|".join(via.REFERENCE_PACKAGES) + r")\b")


def _strings(node, bindings):
    """String constants in an expression, and in the module-level constants it names."""
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            out.append(n.value)
        elif isinstance(n, ast.Name) and n.id in bindings:
            out += [c.value for c in ast.walk(bindings[n.id]) if isinstance(c, ast.Constant) and isinstance(c.value, str)]
    return out


def undeclared_dynamic_loads(source):
    """Lines of calls to importlib's loaders (or ``__import__``) that name a reference package, other than by
    a literal module name (which the file scan reads), when some
    ``validates`` marker of the file lacks ``via=``; else empty."""
    tree = ast.parse(source)
    bindings = {
        t.id: stmt.value for stmt in tree.body if isinstance(stmt, ast.Assign) for t in stmt.targets if isinstance(t, ast.Name)
    }
    loads = []
    markers = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
        if name in DYNAMIC_LOADERS:
            if name in via.STRING_LOADERS and node.args and isinstance(node.args[0], ast.Constant):
                continue  # a literal name: the file scan reads it
            args = [*node.args, *(k.value for k in node.keywords)]
            if any(REFERENCE_NAME.search(s) for a in args for s in _strings(a, bindings)):
                loads.append(node.lineno)
        elif name == "validates":
            markers.append(node)
    if loads and any(not any(k.arg == "via" for k in m.keywords) for m in markers):
        return loads
    return []


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_guard_sees_dynamic_loads_of_reference_packages_and_accepts_declared_via():
    loader = "import importlib\nm = importlib.import_module(NAME)\nNAME = 'crosscheck.sky'\n"
    marked = "import pytest\n@pytest.mark.validates('a', roots=['standards']{})\ndef test_x(): pass\n"
    assert undeclared_dynamic_loads(loader + marked.format("")) == [2]
    assert undeclared_dynamic_loads(loader + marked.format(", via=['crosscheck.sky']")) == []
    spec = (
        "import importlib.util\nSRC = 'src/external_bridge'\n"
        "spec = importlib.util.spec_from_file_location('w', SRC + '/alma_worker.py')\n"
    )
    assert undeclared_dynamic_loads(spec + marked.format("")) == [3]
    assert undeclared_dynamic_loads("m = __import__('virgil_bridge.' + x)\n" + marked.format("")) == [1]
    assert undeclared_dynamic_loads("import importlib\nm = importlib.import_module('crosscheck.sky')\n" + marked.format("")) == []
    assert undeclared_dynamic_loads("import importlib\nm = importlib.import_module(f'crosscheck.{x}')\n" + marked.format("")) == [2]
    # other packages, and scripts that are not reference code, are not this guard's business
    other = "import importlib\nm = importlib.import_module('virgil.aliases')\n"
    assert undeclared_dynamic_loads(other + marked.format("")) == []


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_test_files_declare_via_when_they_load_reference_packages_by_name():
    """No test file calls ``importlib.import_module``, ``__import__`` or ``spec_from_file_location`` on
    ``crosscheck``, ``external_bridge`` or ``virgil_bridge`` unless every ``validates`` marker in it says
    ``via=[...]``: the file scan cannot see such a load, and a missing ``via`` would let the check count
    without the reference being verified."""
    bad = {}
    for py in sorted((ROOT / "tests").rglob("*.py")):
        if any(via.is_ignored_repo_part(part) for part in py.relative_to(ROOT).parts):
            continue
        lines = undeclared_dynamic_loads(py.read_text(encoding="utf-8"))
        if lines:
            bad[py.relative_to(ROOT).as_posix()] = lines
    assert not bad, bad
