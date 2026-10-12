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


def _file(path, *runs):
    path.write_text("".join(json.dumps({"record": "run", "virgil": r}) + "\n" for r in runs)
                    + json.dumps({"record": "test", "test": "t"}) + "\n")
    return path


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_one_commit_across_files_passes(tmp_path):
    f1, f2 = _file(tmp_path / "x.jsonl", {"commit": A}), _file(tmp_path / "y.jsonl", {"commit": A})
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


@pytest.mark.validates("evidence", roots=["standards"], kind="guard", tier="B")
@pytest.mark.skipif(not os.environ.get("VALIDATION_RELEASE"), reason="release gate: set VALIDATION_RELEASE=1 before a virgil release")
def test_committed_evidence_is_pinned_to_one_commit():
    expect = os.environ.get("VALIDATION_RELEASE_COMMIT") or None
    assert pin.problems(sorted((ROOT / "trust" / "evidence").glob("*.jsonl")), expect, release=True) == []
