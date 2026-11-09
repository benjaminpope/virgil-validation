"""The run header's ``validation_dirty`` flag: clean for a CI checkout, dirty for edited evidence."""

import subprocess

import pytest

from evidence.meta import _dirty, _relative_to_root, jax_info

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


def test_a_sibling_evidence_file_is_dirty_when_edited_but_the_output_is_exempt(checkout):
    (checkout / "trust/evidence/other.jsonl").write_text("{}\n")
    _git(checkout, "add", ".")
    _git(checkout, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "more")
    (checkout / "trust/evidence/other.jsonl").write_text("edited\n")
    assert _dirty(str(checkout), ignore=("trust/evidence/campaigns.jsonl",)) is True
    (checkout / "trust/evidence/other.jsonl").write_text("{}\n")
    (checkout / "trust/evidence/campaigns.jsonl").write_text("being written\n")
    assert _dirty(str(checkout), ignore=("trust/evidence/campaigns.jsonl",)) is False


def test_a_rename_reads_dirty_by_its_new_path(checkout):
    _git(checkout, "mv", "src/a.py", "src/renamed.py")
    assert _dirty(str(checkout)) is True
    assert _dirty(str(checkout), ignore=("src/renamed.py",)) is False  # the new path, not "old -> new"


def test_odd_file_names_are_not_mangled(checkout):
    (checkout / "src/sp ace \u00e9.py").write_text("")
    assert _dirty(str(checkout)) is True
    assert _dirty(str(checkout), ignore=("src/sp ace \u00e9.py",)) is False


def test_not_a_git_checkout_cannot_be_told(tmp_path):
    assert _dirty(str(tmp_path)) is None


def test_the_evidence_path_is_relative_to_the_checkout_after_resolving_symlinks(tmp_path):
    root = tmp_path / "root"
    (root / "trust/evidence").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "link_out").symlink_to(outside)
    (root / "link_in").symlink_to(root / "trust/evidence")
    (tmp_path / "root_sibling").mkdir()
    assert _relative_to_root(str(root / "trust/evidence/e.jsonl"), root=root) == "trust/evidence/e.jsonl"
    assert _relative_to_root(str(root / "link_in/e.jsonl"), root=root) == "trust/evidence/e.jsonl"
    assert _relative_to_root(str(root / "link_out/e.jsonl"), root=root) is None
    assert _relative_to_root(str(tmp_path / "root_sibling/e.jsonl"), root=root) is None  # not a prefix match
    assert _relative_to_root(str(root / "..foo"), root=root) == "..foo"  # a name starting "..", still inside
    assert _relative_to_root(None, root=root) is None


def test_jax_info_records_a_backend_failure_instead_of_hiding_it(monkeypatch):
    jax = pytest.importorskip("jax")

    def boom(*a):
        raise RuntimeError("no backend")

    monkeypatch.setattr(jax, "devices", boom)
    assert jax_info() == {"error": "RuntimeError: no backend"}


