"""A committed record's ``criteria_commit`` must be in this repository's history (git ancestry only; no hashing).

A squash merge or force-push would orphan a recorded commit, and then nobody could check which criteria a committed
campaign or gate summary was run against. Every tracked JSON that carries a ``criteria_commit`` (``trust/campaigns/``
and any gate output committed elsewhere) is checked with ``git merge-base --is-ancestor``. Skipped on a shallow clone,
where the history needed is not there."""

import json
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")


def git(*args, root=ROOT):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def commits_in(obj):
    """Every ``criteria_commit`` string anywhere in a parsed JSON document."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "criteria_commit" and isinstance(value, str):
                yield value
            else:
                yield from commits_in(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from commits_in(value)


def records_with_commits(root=ROOT):
    """(repo-relative path, commit) for every tracked JSON that records a ``criteria_commit``."""
    out = []
    for rel in git("ls-files", "--", "*.json", root=root).stdout.splitlines():
        path = root / rel
        if not path.is_file() or b"criteria_commit" not in path.read_bytes():
            continue
        try:
            doc = json.loads(path.read_text())
        except ValueError:
            continue
        out += [(rel, c) for c in commits_in(doc)]
    return out


def non_ancestors(root=ROOT):
    """The records whose commit is not an ancestor of HEAD in ``root``."""
    return [(rel, c) for rel, c in records_with_commits(root) if git("merge-base", "--is-ancestor", c, "HEAD", root=root).returncode]


def require_full_history(root=ROOT):
    if git("rev-parse", "--git-dir", root=root).returncode:
        pytest.skip("not a git checkout: ancestry cannot be checked")
    if git("rev-parse", "--is-shallow-repository", root=root).stdout.strip() == "true":
        pytest.skip("shallow clone: the history needed to check ancestry is not here")


def test_every_recorded_criteria_commit_is_an_ancestor_of_head():
    require_full_history()
    bad = non_ancestors()
    assert not bad, "criteria_commit not in this repository's history (squash merge or force-push?): " + ", ".join(f"{r}: {c}" for r, c in bad)


def test_a_non_ancestor_commit_is_caught_in_a_scratch_repo(tmp_path):
    def run(*a):
        return subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", *a], check=True, capture_output=True, text=True)

    run("init", "-q", "-b", "main")
    (tmp_path / "crit.md").write_text("criteria\n")
    run("add", ".")
    run("commit", "-qm", "criteria")
    good = run("rev-parse", "HEAD").stdout.strip()
    run("checkout", "-q", "-b", "side")
    (tmp_path / "crit.md").write_text("changed\n")
    run("commit", "-qam", "orphaned")
    orphan = run("rev-parse", "HEAD").stdout.strip()
    run("checkout", "-q", "main")
    (tmp_path / "ok.json").write_text(json.dumps({"summary": {"criteria_commit": good}}))
    (tmp_path / "bad.json").write_text(json.dumps({"runs": [{"criteria_commit": orphan}]}))
    run("add", ".")
    run("commit", "-qm", "records")
    assert non_ancestors(tmp_path) == [("bad.json", orphan)]
