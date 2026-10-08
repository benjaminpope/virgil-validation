"""The private L2 collection's id must not appear anywhere in the tracked tree.

This file holds only SHA-256 hashes of the id (and of its first 8 hex digits), compared
with the hash of every UUID-shaped or 8-hex token in each tracked file name and body. Private
collections are named by a placeholder (``private-<name>``); the real id lives in
``OIDB_PRIVATE_COLLECTIONS`` or ``~/.config/virgil-validation/private_collections.txt``
(``scripts/private_collections.py``).
"""

import hashlib
import importlib.util
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_SHA256 = {
    "a9c8ab1bb62be8362fb5b6bf2128c47d03d4669f2313e665f5b2a65921ccf251",  # the full id
    "1fc430335c576930002d900ac0ae3adcc6ba036eb7062732aee58038a22c46c9",  # its first 8 hex digits
}
TOKEN = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}|\b[0-9a-f]{8}\b")


def _hits(s):
    return any(hashlib.sha256(t.encode()).hexdigest() in PRIVATE_SHA256 for t in TOKEN.findall(s))


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
        if _hits(name) or _hits(text):
            hits.append(name)
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
