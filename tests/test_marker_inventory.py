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


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_calls_in_the_body_are_not_markers(tmp_path):
    src = """
        import pytest
        @pytest.mark.validates("a", roots=["standards"])
        def test_one():
            schema.validates(data)
        def test_body_only():
            schema.validates(data)
    """
    inv = mi.inventory(_tree(tmp_path, "t", src))
    assert set(inv) == {"tests/test_x.py::test_one"}
    assert len(inv["tests/test_x.py::test_one"]) == 1


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_property_and_via_are_part_of_the_claim(tmp_path):
    a = _tree(tmp_path, "a", 'import pytest\n@pytest.mark.validates("a", roots=["standards"], property="p")\ndef test_one(): ...\n')
    b = _tree(tmp_path, "b", 'import pytest\n@pytest.mark.validates("a", roots=["standards"], property=["q"], via=["m"])\ndef test_one(): ...\n')
    ia, ib = mi.inventory(a), mi.inventory(b)
    claim = ia["tests/test_x.py::test_one"][0]
    assert claim["property"] == ["p"] and claim["via"] == []
    out = mi.diff(ia, ib)
    assert "`tests/test_x.py::test_one` | changed" in out
    assert "property=q" in out and "via=m" in out


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_class_marks_propagate_to_test_methods_of_test_classes_only(tmp_path):
    src = """
        import pytest
        @pytest.mark.validates("c", roots=["standards"], kind="guard")
        class TestK:
            @pytest.mark.validates("m", roots=["data"])
            def test_a(self): ...
            def test_b(self): ...
            def helper(self): ...
        @pytest.mark.validates("n", roots=["standards"])
        class Helper:
            def test_c(self): ...
    """
    inv = mi.inventory(_tree(tmp_path, "t", src))
    assert set(inv) == {"tests/test_x.py::TestK::test_a", "tests/test_x.py::TestK::test_b"}
    assert [c["objects"] for c in inv["tests/test_x.py::TestK::test_a"]] == [["m"], ["c"]]
    assert [c["objects"] for c in inv["tests/test_x.py::TestK::test_b"]] == [["c"]]


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_module_pytestmark_applies_to_every_test_alongside_its_own_marks(tmp_path):
    src = """
        import pytest
        pytestmark = [pytest.mark.validates("g", roots=["standards"]), pytest.mark.slow]
        @pytest.mark.validates("a", roots=["data"])
        def test_marked(): ...
        def test_plain(): ...
        class TestK:
            def test_m(self): ...
    """
    inv = mi.inventory(_tree(tmp_path, "t", src))
    assert set(inv) == {"tests/test_x.py::test_marked", "tests/test_x.py::test_plain", "tests/test_x.py::TestK::test_m"}
    assert not any("pytestmark" in t for t in inv)
    # pytest applies function, class and module marks together, one claim each
    assert [c["objects"] for c in inv["tests/test_x.py::test_marked"]] == [["a"], ["g"]]
    assert [c["objects"] for c in inv["tests/test_x.py::test_plain"]] == [["g"]]
    assert [c["objects"] for c in inv["tests/test_x.py::TestK::test_m"]] == [["g"]]
