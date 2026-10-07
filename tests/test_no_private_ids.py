"""The private L2 collection's id must not appear anywhere in the tracked tree.

The id is built from two halves so that this file does not contain it. Private
collections are named by a placeholder (``private-<name>``); the real id lives in
``OIDB_PRIVATE_COLLECTIONS`` or ``~/.config/virgil-validation/private_collections.txt``
(``scripts/private_collections.py``).
"""

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_IDS = ["647a" + "22a9", ("647a" + "22a9") + "-5047-4220-ba22-a95047022072"]


def _tracked():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    return [p for p in out.decode().split("\0") if p]


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_private_collection_id_not_in_tracked_tree():
    hits = []
    for name in _tracked():
        path = ROOT / name
        if not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", "replace")
        hits += [name for pid in PRIVATE_IDS if pid in name or pid in text]
    assert not hits, f"private collection id in tracked files: {sorted(set(hits))}"


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_private_collections_resolve_and_skip(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("private_collections", ROOT / "scripts" / "private_collections.py")
    pc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pc)
    monkeypatch.setattr(pc, "FILE", tmp_path / "none.txt")
    monkeypatch.delenv(pc.ENV, raising=False)
    assert pc.private_ids() == {} and pc.real_id("private-workshop") is None
    pc.warn_skip("private-workshop")
    assert "skipped" in capsys.readouterr().err
    monkeypatch.setenv(pc.ENV, "workshop=abc, other=def")
    assert pc.real_id("private-workshop") == "abc" and pc.key_of("def") == "private-other"
    assert pc.key_of("public-id") == "public-id" and pc.real_id("public-id") == "public-id"
