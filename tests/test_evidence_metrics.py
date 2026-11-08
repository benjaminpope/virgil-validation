"""evidence.plugin.metrics_of: a metric name recorded twice keeps both values."""
import pytest

from evidence.plugin import metrics_of


@pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")
def test_repeated_metric_names_keep_every_value():
    props = [("err", 1.0), ("other", 5.0), ("err", 2.0), ("err", "inf")]
    assert metrics_of(props) == {"err": [1.0, 2.0, "inf"], "other": 5.0}
    assert dict(props)["err"] == "inf"  # what dict() did: the earlier values were lost


@pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")
def test_unrepeated_metrics_stay_scalars():
    assert metrics_of([("a", 1.0), ("b", 2.0)]) == {"a": 1.0, "b": 2.0}
    assert metrics_of([]) == {}


@pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")
def test_the_trust_headline_reduces_a_repeated_metric_to_its_worst_case():
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    import trust

    rec = {"metrics": {"rel_err": [1e-9, -3e-7, 2e-8], "n": 3.0}}
    assert trust._headline(rec) == ("rel_err", -3e-7)
    assert trust._headline(dict(rec, headline="rel_err")) == ("rel_err", -3e-7)
    assert trust._headline({"metrics": {"rel_err": [1e-9, "inf"]}, "headline": "rel_err"}) == ("rel_err", "inf")
    assert trust._headline({"metrics": {"rel_err": 1e-9}}) == ("rel_err", 1e-9)  # a scalar is unchanged


# --- missing evidence made visible: collection skips, selection, virgil import skips ---

pytest_plugins = ["pytester"]

_SKIPPING_MODULE = '''
import pytest

pytest.importorskip("virgil.nope")

pytestmark = pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")


def test_one():
    """First paragraph of the docstring."""


@pytest.mark.validates("virgil.models.UniformDisk", roots=["mathematics"], tier="B")
def test_two():
    pass
'''


def _records(path):
    import json

    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")
def test_a_module_skipped_at_import_leaves_one_record_per_test(pytester, monkeypatch):
    monkeypatch.delenv("VV_REQUIRE_VIRGIL_IMPORTS", raising=False)
    pytester.makepyfile(test_gone=_SKIPPING_MODULE)
    result = pytester.runpytest_subprocess("--evidence", "ev.jsonl")
    assert result.ret == pytest.ExitCode.NO_TESTS_COLLECTED  # a skip is not an error
    header, *records = _records(pytester.path / "ev.jsonl")
    assert header["record"] == "run"
    by_test = {}
    for r in records:
        assert r["outcome"] == "skipped" and r["phase"] == "collection"
        assert "virgil.nope" in r["reason"]
        by_test.setdefault(r["test"], []).append(r)
    assert sorted(by_test) == ["test_gone.py::test_one", "test_gone.py::test_two"]
    # the module-level marker covers both tests; the decorator adds a second claim to test_two
    assert [c["objects"] for c in by_test["test_gone.py::test_one"]] == [["evidence"]]
    assert [c["tier"] for c in by_test["test_gone.py::test_two"]] == ["A", "B"]
    assert by_test["test_gone.py::test_one"][0]["doc"] == "First paragraph of the docstring."


@pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")
def test_a_missing_virgil_module_is_a_collection_error_when_required(pytester, monkeypatch):
    monkeypatch.setenv("VV_REQUIRE_VIRGIL_IMPORTS", "1")
    pytester.makepyfile(test_gone=_SKIPPING_MODULE)
    result = pytester.runpytest_subprocess("--evidence", "ev.jsonl")
    assert result.ret != 0
    result.stdout.fnmatch_lines(["*VV_REQUIRE_VIRGIL_IMPORTS=1*"])
    assert not _records(pytester.path / "ev.jsonl")[1:]  # an error is not recorded as a skip


@pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")
def test_only_a_missing_virgil_module_is_required():
    from evidence.plugin import _missing_virgil

    assert _missing_virgil("could not import 'virgil.nope': No module named 'virgil.nope'")
    assert _missing_virgil("could not import 'virgil': No module named 'virgil'")
    # a virgil module whose own optional dependency is absent stays a skip
    assert not _missing_virgil("could not import 'virgil.radio': No module named 'casatools'")
    # other modules, including ones whose names begin with virgil
    assert not _missing_virgil("could not import 'virgil_vlti': No module named 'virgil_vlti'")
    assert not _missing_virgil("could not import 'pmoired': No module named 'pmoired'")
    assert not _missing_virgil("virgil without imaging.clean")


@pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")
def test_the_header_records_how_tests_were_selected(pytester):
    pytester.makepyfile(
        test_sel='''
import pytest

@pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")
def test_fast():
    pass

@pytest.mark.slow
@pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")
def test_slow():
    pass
'''
    )
    pytester.makeini("[pytest]\nmarkers =\n    slow: slow\n")
    result = pytester.runpytest_subprocess("-m", "not slow", "--evidence", "ev.jsonl")
    assert result.ret == 0
    invocation = _records(pytester.path / "ev.jsonl")[0]["invocation"]
    assert invocation["markexpr"] == "not slow"
    assert invocation["keyword"] == ""
    assert invocation["deselected"] == 1
    assert "not slow" in invocation["argv"]
