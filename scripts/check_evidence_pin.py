"""Fail unless all evidence records name one virgil commit (the release gate).

Every ``trust/evidence/*.jsonl`` starts a run with a ``"record": "run"`` header
naming the virgil commit it ran against. Evidence is only comparable, and the
Trust page only means "virgil at commit X", if every file names the same
commit. Before a virgil release, regenerate all of it against the release
commit (AGENTS.md, "Evidence records") and run

    .venv/bin/python scripts/check_evidence_pin.py --release [--expect <sha>]

It exits 1, listing each file's commit(s), when the headers disagree, when a
header has no commit, or (with ``--expect``) when the single commit is not the
one given (a prefix of at least 7 characters is enough). ``--release`` also
refuses ``dirty`` runs and runs that do not record it (except virgil's own CI,
``"runner": "virgil-ci"``, a fresh checkout at the sha); without it the check
still requires one commit.
Kept apart from scripts/trust.py, which only reads the evidence.
"""

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
# virgil's own CI evidence (virgil.jsonl) is ingested from a run at a pinned
# sha of a fresh checkout and cannot record ``dirty``; every other file must.
CI_RUNNER = "virgil-ci"


def headers(path):
    """(commit or None, dirty, runner) for every run header of an evidence file;
    ``dirty`` is None when the header does not record it."""
    out = []
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("record") == "run":
            v = rec.get("virgil") or {}
            out.append((v.get("commit"), v.get("dirty"), rec.get("runner")))
    return out


def problems(paths, expect=None, release=False):
    """Human-readable reasons the evidence in ``paths`` is not pinned to one commit."""
    found, bad = {}, []
    for path in map(pathlib.Path, paths):
        runs = headers(path)
        if not runs:
            bad.append(f"{path.name}: no run header")
        for commit, dirty, runner in runs:
            if not commit:
                bad.append(f"{path.name}: a run header names no virgil commit")
                continue
            found.setdefault(commit, set()).add(path.name)
            if release and dirty:
                bad.append(f"{path.name}: run at {commit[:7]} was on a dirty virgil tree")
            elif release and dirty is None and runner != CI_RUNNER:
                bad.append(f"{path.name}: run at {commit[:7]} does not record whether the virgil tree was dirty")
    if len(found) > 1:
        bad.append("evidence names more than one virgil commit: " + "; ".join(
            f"{c[:7]} in {', '.join(sorted(f))}" for c, f in sorted(found.items())))
    if expect and found and not all(c.startswith(expect) for c in found):
        bad.append(f"evidence is at {', '.join(c[:7] for c in found)}, not the expected {expect}")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="*", help="default: trust/evidence/*.jsonl")
    ap.add_argument("--release", action="store_true", help="also refuse dirty runs")
    ap.add_argument("--expect", help="the release commit (sha or prefix, 7+ characters)")
    args = ap.parse_args(argv)
    if args.expect and len(args.expect) < 7:
        ap.error("--expect needs at least 7 characters")
    paths = args.files or sorted((ROOT / "trust" / "evidence").glob("*.jsonl"))
    bad = problems(paths, args.expect, args.release)
    for line in bad:
        print("FAIL", line)
    if not bad:
        print(f"ok: {len(paths)} evidence files, one virgil commit")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
