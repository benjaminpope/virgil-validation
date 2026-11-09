"""A committed record's ``criteria_commit`` must be in this repository's history (git ancestry only; no hashing).

A squash merge or force-push would orphan a recorded commit, and then nobody could check which criteria a committed
campaign or gate summary was run against. Every tracked ``*.json`` and ``*.jsonl`` that carries a ``criteria_commit``
(``trust/campaigns/`` and any gate output committed elsewhere) is checked. The value must be a full 40-character
lower-case sha (not ``HEAD``, a short sha, an empty string or a non-string), must name a commit object, and must be an
ancestor of HEAD (``git merge-base --is-ancestor``). A file that mentions the key but does not parse is a failure, not a
skip. YAML design files are out of scope: they hold design text, not run records. Skipped on a shallow clone, where the
history needed is not there."""

import json
import pathlib
import re
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")
FULL_SHA = re.compile(r"[0-9a-f]{40}")


def git(*args, root=ROOT):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def commits_in(obj):
    """Every ``criteria_commit`` value (of any type) anywhere in a parsed JSON document."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "criteria_commit":
                yield value
            else:
                yield from commits_in(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from commits_in(value)


def records_with_commits(root=ROOT):
    """((path, value) records, [paths that mention the key but do not parse]) over tracked json and jsonl files."""
    records, unparseable = [], []
    for rel in git("ls-files", "--", "*.json", "*.jsonl", root=root).stdout.splitlines():
        path = root / rel
        if not path.is_file() or b"criteria_commit" not in path.read_bytes():
            continue
        text = path.read_text()
        try:
            docs = [json.loads(line) for line in text.splitlines() if line.strip()] if rel.endswith(".jsonl") else [json.loads(text)]
        except ValueError:
            unparseable.append(rel)
            continue
        records += [(rel, value) for doc in docs for value in commits_in(doc)]
    return records, unparseable


def problems(root=ROOT):
    """One message per record or file that breaks the rule."""
    records, unparseable = records_with_commits(root)
    out = [f"{rel}: mentions criteria_commit but is not valid JSON" for rel in unparseable]
    for rel, value in records:
        if not isinstance(value, str) or not FULL_SHA.fullmatch(value):
            out.append(f"{rel}: criteria_commit {value!r} is not a full 40-character lower-case sha")
        elif git("cat-file", "-e", f"{value}^{{commit}}", root=root).returncode:
            out.append(f"{rel}: criteria_commit {value} is an unknown object here (not a commit in this repository)")
        elif git("merge-base", "--is-ancestor", value, "HEAD", root=root).returncode:
            out.append(f"{rel}: criteria_commit {value} is not an ancestor of HEAD")
    return out


def require_full_history(root=ROOT):
    if git("rev-parse", "--git-dir", root=root).returncode:
        pytest.skip("not a git checkout: ancestry cannot be checked")
    if git("rev-parse", "--is-shallow-repository", root=root).stdout.strip() == "true":
        pytest.skip("shallow clone: the history needed to check ancestry is not here")


def test_every_recorded_criteria_commit_is_an_ancestor_of_head():
    require_full_history()
    bad = problems()
    assert not bad, "criteria_commit records that cannot be checked (squash merge or force-push?):\n" + "\n".join(bad)


def test_the_check_reports_every_bad_record_in_a_scratch_repo(tmp_path):
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
    dump = lambda name, doc: (tmp_path / name).write_text(json.dumps(doc))
    dump("ok.json", {"summary": {"criteria_commit": good}})
    (tmp_path / "ok.jsonl").write_text(json.dumps({"criteria_commit": good}) + "\n")
    dump("orphan.json", {"runs": [{"criteria_commit": orphan}]})
    (tmp_path / "orphan.jsonl").write_text(json.dumps({"x": 1}) + "\n" + json.dumps({"criteria_commit": orphan}) + "\n")
    dump("head.json", {"criteria_commit": "HEAD"})
    dump("null.json", {"criteria_commit": None})
    dump("empty.json", {"criteria_commit": ""})
    dump("short.json", {"criteria_commit": good[:7]})
    dump("unknown.json", {"criteria_commit": "0" * 40})
    (tmp_path / "broken.json").write_text('{"criteria_commit": "' + good)
    run("add", ".")
    run("commit", "-qm", "records")
    reported = {m.split(":")[0] for m in problems(tmp_path)}
    assert reported == {"orphan.json", "orphan.jsonl", "head.json", "null.json", "empty.json", "short.json", "unknown.json", "broken.json"}
    messages = "\n".join(problems(tmp_path))
    assert "unknown object" in messages and "not an ancestor" in messages and "not valid JSON" in messages
