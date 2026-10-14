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
# Empty: the one L2 collection (CHARA workshop 647a22a9) was cleared by Ben Pope on 2026-10-13 for reduction and
# display, so its id is now tracked (oidb/manifest.yml). Add the SHA-256 of any future private id (and of its first
# 8 hex digits) here before its first fetch.
PRIVATE_SHA256 = set()
TOKEN = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}|(?<![0-9a-f])[0-9a-f]{8}(?![0-9a-f])")  # not \b: "fit_647a22a9.json" has no word boundary


def _hits(s, hashes=None):
    hashes = PRIVATE_SHA256 if hashes is None else hashes
    return any(hashlib.sha256(t.encode()).hexdigest() in hashes for t in TOKEN.findall(s))


def _tracked():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    return [p for p in out.decode().split("\0") if p]


@pytest.mark.validates("evidence", roots=["standards"], kind="control")
def test_the_scan_catches_a_listed_id():
    """The machinery that the next private collection relies on can fail: with the hash of a made-up id listed, a
    file name or body that holds the id (or its first 8 hex digits) is a hit, and one that does not is not."""
    fake = "0123abcd-4567-89ab-cdef-0123456789ab"
    hashes = {hashlib.sha256(t.encode()).hexdigest() for t in (fake, fake[:8])}
    assert _hits(f"collection {fake} fetched", hashes) and _hits(f"fit_{fake[:8]}.json", hashes)
    assert not _hits("collection fac164e1-d9d0-4500-8164-e1d9d0450099", hashes)
    assert not _hits(f"collection {fake}", set())  # no listed id: nothing is a hit


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_private_collection_id_not_in_tracked_tree():
    if not PRIVATE_SHA256:
        pytest.skip("no private collection is listed")
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
