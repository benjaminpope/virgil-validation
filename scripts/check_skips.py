#!/usr/bin/env python
"""Fail when a check that claims to validate something was skipped without an allowed reason.

    scripts/check_skips.py EVIDENCE.jsonl --job pr|weekly [--allowlist trust/skip_allowlist.yml]

``EVIDENCE.jsonl`` is what ``pytest --evidence`` writes; every record in it is a test that
carries a ``validates`` marker. A record whose outcome is ``skipped`` is a claim that was
never checked: the Trust page shows it as unchecked, but nothing in CI turns red. This script
does, unless the test is on the allow-list for the job (``pr``: the fast tests on pull
requests and pushes; ``weekly``: the full suite on virgil's main).

An allow-list entry has

    test:     a test id, or a glob in which ``*`` matches any run of characters (brackets are
              literal, so a parametrized id such as ``test_x[True]`` is written as is)
    because:  ``package`` (CI does not install it) or ``data`` (CI does not have it): nothing else
    why:      one line saying which package or file, and how CI would get it
    message:  text the recorded skip reason must contain, so that a test allowed because a file
              is absent cannot go on skipping for another reason unnoticed (required)

A skip that is not covered is a gap: either CI should run the check (fix the environment or the
test), or it belongs on the list with a reason. Entries that no longer match a skip are listed
as stale (they never fail the check). Exit status 1 when any skip is uncovered.
"""

import argparse
import json
import pathlib
import re
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / "trust" / "skip_allowlist.yml"
JOBS = ("pr", "weekly")
BECAUSE = ("package", "data")


def glob_to_regex(pattern):
    """``*`` is the only wildcard; a bracket in a parametrized id is a bracket."""
    return re.compile("".join(".*" if c == "*" else re.escape(c) for c in pattern))


def load_allowlist(path, job):
    """The entries of one job's list, each with its compiled pattern; a malformed entry is an error."""
    data = yaml.safe_load(pathlib.Path(path).read_text()) or {}
    unknown = set(data) - set(JOBS)
    if unknown:
        raise ValueError(f"{path}: unknown section(s) {sorted(unknown)}; the sections are {list(JOBS)}")
    entries = data.get(job) or []
    for entry in entries:
        missing = [k for k in ("test", "because", "why", "message") if not str(entry.get(k) or "").strip()]
        if missing:
            raise ValueError(f"{path}: {job} entry {entry.get('test')!r} lacks {missing}")
        file_part, sep, name_part = str(entry["test"]).partition("::")
        if not sep or not file_part.replace("*", "") or not name_part.replace("*", ""):
            raise ValueError(
                f"{path}: {job} entry {entry['test']!r} is too loose: name a test file and a test (path.py::test_name), "
                "with * only inside either part"
            )
        if entry["because"] not in BECAUSE:
            raise ValueError(
                f"{path}: {job} entry {entry['test']!r}: because is {entry['because']!r}; the only allowed reasons are {list(BECAUSE)}"
            )
        entry["_pattern"] = glob_to_regex(entry["test"])
    return entries


def skipped_records(evidence):
    """One record per skipped test id (a test with several claims is one test)."""
    skipped = {}
    for line in pathlib.Path(evidence).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("record") == "test" and rec.get("outcome") == "skipped":
            skipped.setdefault(rec["test"], rec)
    return skipped


def covers(entry, rec):
    """Whether an allow-list entry covers a skipped record: the id matches and, if the entry says
    what the skip should say, the recorded reason says it."""
    if not entry["_pattern"].fullmatch(rec["test"]):
        return False
    needed = entry.get("message")
    return not needed or needed.lower() in str(rec.get("reason", "")).lower()


def check(evidence, entries):
    """(uncovered skips, stale entries)."""
    skipped = skipped_records(evidence)
    uncovered = [rec for rec in skipped.values() if not any(covers(e, rec) for e in entries)]
    stale = [e for e in entries if not any(covers(e, rec) for rec in skipped.values())]
    return uncovered, stale


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("evidence", help="evidence.jsonl from pytest --evidence")
    ap.add_argument("--job", required=True, choices=JOBS, help="which allow-list applies")
    ap.add_argument("--allowlist", default=str(ALLOWLIST))
    args = ap.parse_args(argv)
    entries = load_allowlist(args.allowlist, args.job)
    uncovered, stale = check(args.evidence, entries)
    total = len(skipped_records(args.evidence))
    print(f"{total} skipped check(s) with a validates marker; {total - len(uncovered)} on the {args.job} allow-list")
    for e in stale:
        print(f"note: stale allow-list entry (no skip matches it any more): {e['test']}")
    if not uncovered:
        return 0
    print(f"\n{len(uncovered)} check(s) skipped without an allowed reason (see {pathlib.Path(args.allowlist).name}):")
    for rec in sorted(uncovered, key=lambda r: r["test"]):
        print(f"  {rec['test']}\n      reason: {rec.get('reason', '(none recorded)')}")
    print(
        "\nA check that never runs verifies nothing. Make CI run it, or, if the package or data it needs is "
        "really not in CI, add it to the list with that reason. Do not skip, xfail or mark it slow to pass."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
