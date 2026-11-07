"""The pull-campaign design hash drops exactly two functions of
src/crosscheck/sky.py (arc_curve, arc_curve_gaussian) and blank lines
(ledger D22). These tests pin that: nothing else is stripped, so any other
edit to sky.py changes what design_hash() hashes."""

import ast
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec_ = importlib.util.spec_from_file_location("pull_campaign", ROOT / "scripts" / "pull_campaign.py")
pc = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(pc)

SKY_PATH = ROOT / "src" / "crosscheck" / "sky.py"
SKY = SKY_PATH.read_text()
ARCS = {"arc_curve", "arc_curve_gaussian"}


def nonblank(src):
    return [ln for ln in src.splitlines() if ln.strip()]


def without_arc_defs(src):
    """The full source with the two arc function definitions removed, found
    by indentation rather than by ast, so it does not copy the walk in
    sky_source_without_arcs: a definition runs from its ``def`` line to the
    next non-blank line at column 0."""
    kept, skipping = [], False
    for ln in src.splitlines(keepends=True):
        if ln.strip() and not ln[0].isspace():
            skipping = any(ln.startswith(f"def {name}(") for name in ARCS)
        if not skipping:
            kept.append(ln)
    return "".join(kept)


def stripped_of(monkeypatch, text):
    real = pathlib.Path.read_text
    monkeypatch.setattr(pathlib.Path, "read_text", lambda self, *a, **k: text if self == SKY_PATH else real(self, *a, **k))
    return pc.sky_source_without_arcs()


@pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")
def test_only_the_arc_functions_and_blank_lines_are_stripped():
    funcs = {n.name for n in ast.parse(SKY).body if isinstance(n, ast.FunctionDef)}
    assert ARCS <= funcs, "the arc functions the hash excludes no longer exist; update ledger D22"
    stripped = pc.sky_source_without_arcs()
    assert nonblank(stripped) == nonblank(without_arc_defs(SKY))
    assert {n.name for n in ast.parse(stripped).body if isinstance(n, ast.FunctionDef)} == funcs - ARCS


@pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")
def test_any_other_edit_to_sky_py_changes_the_hash_input(monkeypatch):
    base = pc.sky_source_without_arcs()
    h0 = pc.design_hash()
    # edits outside the arc functions change the hashed text and the hash itself
    assert stripped_of(monkeypatch, SKY + "\n# extra comment\n") != base
    assert pc.design_hash() != h0
    assert stripped_of(monkeypatch, SKY.replace("import numpy as np", "import numpy as np  # edited", 1)) != base
    assert pc.design_hash() != h0
    # edits inside an arc function, and blank lines, do not
    arc_line = next(ln for ln in SKY.splitlines() if ln.startswith("def arc_curve("))
    assert stripped_of(monkeypatch, SKY.replace(arc_line, arc_line + "\n    _x = 1", 1)) == base
    assert pc.design_hash() == h0
    assert stripped_of(monkeypatch, SKY + "\n\n\n") == base
    assert pc.design_hash() == h0
