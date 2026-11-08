"""The run header's ``validation_dirty`` flag: clean for a CI checkout, dirty for edited evidence."""

import subprocess

import pytest

from evidence.meta import _dirty, _evidence_path, _relative_to_root

pytestmark = pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")


def _git(path, *args):
    subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)


@pytest.fixture
def checkout(tmp_path):
    (tmp_path / "trust/evidence").mkdir(parents=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "trust/evidence/campaigns.jsonl").write_text("{}\n")
    (tmp_path / "src/a.py").write_text("x = 1\n")
    (tmp_path / ".gitignore").write_text(".venv/\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "init")
    return tmp_path


def test_a_ci_checkout_with_its_own_untracked_material_reads_clean(checkout):
    """CI puts the virgil checkout, downloaded artefacts and the evidence file
    beside the tests; none of it is the code under test."""
    (checkout / ".venv").mkdir()
    (checkout / ".venv/x").write_text("")
    (checkout / "virgil-src/src").mkdir(parents=True)
    (checkout / "virgil-src/src/v.py").write_text("")
    (checkout / "ci/push").mkdir(parents=True)
    (checkout / "ci/push/evidence.jsonl").write_text("")
    (checkout / "virgil-ci").mkdir()
    (checkout / "virgil-ci/virgil-evidence.jsonl").write_text("")
    (checkout / "evidence.jsonl").write_text("")
    assert _dirty(str(checkout)) is False


def test_edited_or_new_files_in_tracked_folders_read_dirty(checkout):
    evidence = "trust/evidence/campaigns.jsonl"
    (checkout / evidence).write_text("{}\nedited\n")
    assert _dirty(str(checkout)) is True
    assert _dirty(str(checkout), ignore=(evidence,)) is False  # the file being written is exempt
    (checkout / "trust/evidence/new.jsonl").write_text("")
    assert _dirty(str(checkout), ignore=(evidence,)) is True
    (checkout / "trust/evidence/new.jsonl").unlink()
    (checkout / "src/a.py").write_text("x = 2\n")
    assert _dirty(str(checkout), ignore=(evidence,)) is True


def test_a_new_untracked_source_file_reads_dirty(checkout):
    (checkout / "src/b.py").write_text("")
    assert _dirty(str(checkout)) is True


def test_the_evidence_path_comes_from_the_command_line():
    assert _evidence_path(["pytest", "--evidence", "e.jsonl"]) == "e.jsonl"
    assert _evidence_path(["pytest", "--evidence=e.jsonl"]) == "e.jsonl"
    assert _evidence_path(["pytest"]) is None
    assert _relative_to_root("/elsewhere/e.jsonl") is None
