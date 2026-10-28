"""The release gate that all evidence names one virgil commit (scripts/check_evidence_pin.py)."""

import importlib.util
import json
import os
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_evidence_pin", ROOT / "scripts" / "check_evidence_pin.py")
pin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pin)

A, B = "a" * 40, "b" * 40


def _file(path, *runs, runner=None):
    path.write_text("".join(json.dumps({"record": "run", "virgil": r, "runner": runner}) + "\n" for r in runs)
                    + json.dumps({"record": "test", "test": "t"}) + "\n")
    return path


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_one_commit_across_files_passes(tmp_path):
    f1 = _file(tmp_path / "x.jsonl", {"commit": A, "dirty": False})
    f2 = _file(tmp_path / "y.jsonl", {"commit": A, "dirty": False})
    assert pin.problems([f1, f2]) == []
    assert pin.problems([f1, f2], expect=A[:7], release=True) == []


@pytest.mark.validates("evidence", roots=["standards"], kind="control")
def test_two_commits_fail_and_are_named(tmp_path):
    f1, f2 = _file(tmp_path / "x.jsonl", {"commit": A}), _file(tmp_path / "y.jsonl", {"commit": B})
    (msg,) = pin.problems([f1, f2])
    assert "more than one" in msg and A[:7] in msg and B[:7] in msg and "x.jsonl" in msg
    # two runs in one file count too
    assert pin.problems([_file(tmp_path / "z.jsonl", {"commit": A}, {"commit": B})])


@pytest.mark.validates("evidence", roots=["standards"], kind="control")
def test_missing_commit_wrong_expected_and_dirty_fail(tmp_path):
    assert pin.problems([_file(tmp_path / "n.jsonl", {"commit": None})])
    assert pin.problems([_file(tmp_path / "e.jsonl")])
    ok = _file(tmp_path / "a.jsonl", {"commit": A, "dirty": True})
    assert pin.problems([ok]) == []  # dirty is only refused for a release
    assert pin.problems([ok], release=True)
    assert pin.problems([_file(tmp_path / "b.jsonl", {"commit": A})], expect=B[:7])


@pytest.mark.validates("evidence", roots=["standards"], kind="control")
def test_a_release_needs_the_dirty_state_unless_it_is_virgils_own_ci(tmp_path):
    unrecorded = _file(tmp_path / "u.jsonl", {"commit": A})
    assert pin.problems([unrecorded]) == []  # only a release asks for it
    assert any("does not record" in p for p in pin.problems([unrecorded], release=True))
    assert pin.problems([_file(tmp_path / "c.jsonl", {"commit": A, "dirty": False})], release=True) == []
    assert pin.problems([_file(tmp_path / "v.jsonl", {"commit": A}, runner="virgil-ci")], release=True) == []


@pytest.mark.validates("evidence", roots=["standards"], kind="guard", tier="B")
@pytest.mark.skipif(not os.environ.get("VALIDATION_RELEASE"), reason="release gate: set VALIDATION_RELEASE=1 before a virgil release")
def test_committed_evidence_is_pinned_to_one_commit():
    expect = os.environ.get("VALIDATION_RELEASE_COMMIT") or None
    assert pin.problems(sorted((ROOT / "trust" / "evidence").glob("*.jsonl")), expect, release=True) == []


@pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")
def test_xfail_records_xfailed_or_xpassed_never_passed():
    """An expected failure is recorded as xfailed, and an unexpected pass (XPASS)
    is recorded as xpassed so that the Trust page treats it as failing."""
    from types import SimpleNamespace

    from evidence.plugin import _outcome

    assert _outcome(SimpleNamespace(wasxfail="bug", skipped=True, outcome="skipped")) == "xfailed"
    assert _outcome(SimpleNamespace(wasxfail="bug", skipped=False, outcome="passed")) == "xpassed"
    assert _outcome(SimpleNamespace(skipped=False, outcome="failed")) == "failed"


@pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")
def test_collection_enforces_strict_xfail_and_finding_bi_implication(tmp_path):
    """xfail requires strict=True and kind in ('finding', 'upstream'), and vice versa."""
    from types import SimpleNamespace

    from evidence.plugin import pytest_collection_modifyitems

    dummy_file = tmp_path / "test_dummy.py"
    dummy_file.write_text("pass\n")
    cfg = SimpleNamespace(getoption=lambda name: True)

    def make_item(kind, xfail_kwargs=None):
        v_mark = pytest.mark.validates("virgil.models.UniformDisk", roots=["mathematics"], kind=kind).mark
        x_marks = [] if xfail_kwargs is None else [pytest.mark.xfail(**xfail_kwargs).mark]

        def iter_markers(name):
            return [v_mark] if name == "validates" else (x_marks if name == "xfail" else [])

        return SimpleNamespace(nodeid="test_dummy.py::test_x", path=dummy_file, iter_markers=iter_markers)

    # Valid combinations pass collection
    pytest_collection_modifyitems(cfg, [make_item("check")])
    pytest_collection_modifyitems(cfg, [make_item("finding", {"strict": True})])
    pytest_collection_modifyitems(cfg, [make_item("upstream", {"strict": True})])

    # Invalid combinations fail collection
    for bad in (
        make_item("check", {"strict": True}),
        make_item("check", {"strict": False}),
        make_item("finding", {"strict": False}),
        make_item("finding"),
        make_item("upstream"),
    ):
        with pytest.raises(pytest.UsageError, match="invalid xfail / finding markers"):
            pytest_collection_modifyitems(cfg, [bad])
