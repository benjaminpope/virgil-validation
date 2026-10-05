"""Our reference code must not lean on virgil (or its Bessel functions).

Every Python file under src/crosscheck, src/external_bridge and src/evidence,
at any depth, is parsed and searched for imports of virgil, drpangloss or
jaxbessel: plain and nested imports, ``importlib.import_module("virgil...")``,
``__import__("virgil")`` and ``sys.modules["virgil..."]``. src/virgil_bridge
is the only code allowed to import virgil.
"""

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).parents[1] / "src"
INDEPENDENT = [ROOT / "crosscheck", ROOT / "external_bridge", ROOT / "evidence"]
FORBIDDEN = ("virgil", "drpangloss", "jaxbessel")


def _forbidden(name):
    return isinstance(name, str) and name.split(".")[0] in FORBIDDEN


def offences(source):
    """Lines of a module that import a forbidden package, by any route."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import) and any(_forbidden(a.name) for a in node.names):
            found.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and _forbidden(node.module):
            found.append(node.lineno)
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in ("import_module", "__import__") and node.args and isinstance(node.args[0], ast.Constant) \
                    and _forbidden(node.args[0].value):
                found.append(node.lineno)
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) \
                and node.value.attr == "modules" and isinstance(node.slice, ast.Constant) and _forbidden(node.slice.value):
            found.append(node.lineno)
    return found


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_reference_code_does_not_import_virgil():
    bad = {str(p.relative_to(ROOT)): lines for src in INDEPENDENT for p in sorted(src.rglob("*.py"))
           if (lines := offences(p.read_text()))}
    assert not bad, bad


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_the_guard_catches_every_route():
    """A control: each route of import is caught, and harmless names are not."""
    for source in ("import virgil", "from virgil.models import UniformDisk", "import jaxbessel.j1",
                   "def f():\n    import drpangloss", "import importlib\nimportlib.import_module('virgil.models')",
                   "__import__('virgil')", "import sys\nsys.modules['virgil.models']"):
        assert offences(source), source
    assert not offences("import numpy\nfrom scipy import special\nvirgil_like = 1")
