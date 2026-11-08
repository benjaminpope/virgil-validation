"""scripts/marker_inventory.py reads and diffs validates markers on synthetic trees."""

import importlib.util
import pathlib
import textwrap

import pytest

_spec = importlib.util.spec_from_file_location(
    "marker_inventory", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "marker_inventory.py")
mi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mi)


def _tree(tmp_path, name, source):
    (tmp_path / name / "tests").mkdir(parents=True)
    (tmp_path / name / "tests" / "test_x.py").write_text(textwrap.dedent(source))
    return tmp_path / name


BASE = """
    import pytest
    @pytest.mark.validates("a", roots=["standards"], kind="check", headline="m")
    def test_one(): ...
    @pytest.mark.validates("b", roots=["standards"])
    def test_two(): ...
    def test_unmarked(): ...
"""
HEAD = """
    import pytest
    @pytest.mark.validates("a", roots=["standards", "data"], kind="check", headline="n")
    def test_one(): ...
    @pytest.mark.validates("c", roots=["standards"], kind="guard")
    def test_three(): ...
"""


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_marker_diff_reports_added_removed_and_changed(tmp_path):
    base = mi.inventory(_tree(tmp_path, "base", BASE))
    head = mi.inventory(_tree(tmp_path, "head", HEAD))
    assert set(base) == {"tests/test_x.py::test_one", "tests/test_x.py::test_two"}
    out = mi.diff(base, head)
    assert "`tests/test_x.py::test_two` | removed" in out
    assert "`tests/test_x.py::test_three` | added | guard/A" in out
    assert "`tests/test_x.py::test_one` | changed" in out and "headline=n" in out


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_marker_diff_of_equal_trees_is_empty(tmp_path):
    inv = mi.inventory(_tree(tmp_path, "t", BASE))
    assert "(no marker changes)" in mi.diff(inv, inv)
