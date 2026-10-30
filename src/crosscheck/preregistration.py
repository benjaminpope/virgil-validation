"""Gate criteria and the git commit they were run at.

The rule (Ben, 2026-10-30): a gate's criteria file is a committed file; a run is made only on a clean copy and
records the last commit that touched it; an aggregate refuses to combine runs from different criteria commits.
Changed criteria mean a rerun, with no override flag. Post-hoc changes are recorded in the change-log table
inside the criteria file ("changed after seeing a result: yes/no").

    commit = prereg.criteria_commit("design/orbit_gate_042.md")   # raises if the file has uncommitted changes
    record["criteria_commit"] = commit
    prereg.require_same_commit(run_records)                        # the aggregator

A checkout is asked with git. A snapshot made by ``git archive`` (the OzSTAR jobs) has no repository: it is
committed content by construction, and the commit comes from ``CRITERIA_COMMIT``, which the submit script computes
with the same definition (the last commit that touched the file), so the Mac and the cluster record one value.

Everything below ``LEGACY`` is the earlier SHA-256 registry (``design/criteria_registry.json``), kept only
for the gate scripts that still call it. It is removed when they move to ``criteria_commit``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


class CriteriaError(SystemExit):
    """Raised (as an exit) when a gate would run on criteria that are not a committed, single version."""


def criteria_commit(path, root=None) -> str:
    """The last commit that touched ``path`` (repo-relative or absolute). Raises if the file has uncommitted changes."""
    root = pathlib.Path(root or ROOT)
    path = pathlib.Path(path)
    rel = (path if path.is_absolute() else root / path).resolve().relative_to(root.resolve()).as_posix()
    git = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True)
    if git("rev-parse", "--git-dir").returncode:
        commit = os.environ.get("CRITERIA_COMMIT", "")
        if not commit:
            raise CriteriaError(f"{rel}: not a git checkout and CRITERIA_COMMIT is not set")
        return commit
    if git("status", "--porcelain", "--", rel).stdout.strip():
        raise CriteriaError(f"{rel} has uncommitted changes: commit it, so the run records the criteria it used")
    commit = git("log", "-n1", "--format=%H", "--", rel).stdout.strip()
    if not commit:
        raise CriteriaError(f"{rel} is not committed")
    return commit


def require_same_commit(records) -> str:
    """The single ``criteria_commit`` of ``records``; raises if the runs disagree or any lacks one."""
    seen = {r.get("criteria_commit") for r in records}
    if len(seen) != 1 or None in seen:
        raise CriteriaError(f"runs come from different criteria commits (or none): {sorted(map(str, seen))}; rerun")
    return seen.pop()


# LEGACY: the SHA-256 registry, kept only until the follow-up PR that migrates the other gate scripts to
# criteria_commit; that PR deletes everything below and design/criteria_registry.json.
REGISTRY = ROOT / "design" / "criteria_registry.json"
UNREGISTERED_STAMP = "UNREGISTERED: not gate evidence"


@dataclasses.dataclass(frozen=True)
class Registration:
    file: str  # path relative to the repository root, as keyed in the registry
    sha256: str  # of the file now
    pinned: str | None  # the registry's hash, None when the file has no entry
    registered: bool  # sha256 == pinned
    reason: str  # why it is not registered ("" when it is)


def sha256_file(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def load_registry(registry=None) -> dict:
    registry = registry or REGISTRY
    return json.loads(pathlib.Path(registry).read_text())["criteria"]


def check(criteria_file, *, allow_unregistered=False, registry=None, root=None) -> Registration:
    """Hash ``criteria_file`` and compare it with its pin. Exits non-zero on a mismatch (or no pin), unless
    ``allow_unregistered``: then the returned ``Registration`` has ``registered`` False and the caller stamps
    its output."""
    registry, root = registry or REGISTRY, root or ROOT  # looked up now, so tests can point the module at a scratch repository
    path = pathlib.Path(criteria_file)
    path = path if path.is_absolute() else pathlib.Path(root) / path
    key = path.resolve().relative_to(pathlib.Path(root).resolve()).as_posix()
    entries = load_registry(registry)
    pinned = entries.get(key, {}).get("sha256")
    sha = sha256_file(path)
    if pinned is None:
        reason = f"{key} has no pinned hash in {pathlib.Path(registry).name}"
    elif sha != pinned:
        reason = f"{key} has sha256 {sha}, not the pinned {pinned}: the criteria changed after registration"
    else:
        return Registration(key, sha, pinned, True, "")
    if not allow_unregistered:
        raise CriteriaError(f"criteria not pre-registered: {reason} (a development run may pass --allow-unregistered)")
    print(f"WARNING {UNREGISTERED_STAMP}: {reason}", file=sys.stderr)
    return Registration(key, sha, pinned, False, reason)


def for_args(args, criteria_file, **kw) -> Registration:
    """``check`` with ``--allow-unregistered`` read from parsed ``args`` (False when the namespace has none)."""
    return check(criteria_file, allow_unregistered=getattr(args, "allow_unregistered", False), **kw)


def add_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--allow-unregistered", action="store_true",
                        help=f"development only: run on criteria that are not the pinned ones; every output is stamped '{UNREGISTERED_STAMP}'")


def stamp(record: dict, reg: Registration) -> dict:
    """``record`` with the criteria file, its hash and whether it is registered (plus the stamp when not)."""
    out = {**record, "criteria_file": reg.file, "criteria_hash": reg.sha256, "registered": reg.registered}
    if not reg.registered:
        out["stamp"] = UNREGISTERED_STAMP
        if "pass" in out:
            out["pass"] = False  # an unregistered run is never a pass
        if out.get("verdict") == "PASS":
            out["verdict"] = "UNREGISTERED"  # the IM Lup gates report a verdict, not a pass flag
        out.setdefault("unregistered_reason", reg.reason)
    return out


def stamp_lines(lines, reg: Registration) -> list[str]:
    """Text or markdown lines headed by the criteria file and hash; unregistered ones lead with the stamp."""
    head = [f"**{UNREGISTERED_STAMP}** ({reg.reason})", ""] if not reg.registered else []
    return head + [f"Criteria: `{reg.file}`, sha256 `{reg.sha256}`" + ("" if reg.registered else " (not the pinned hash)"), ""] + list(lines)


def runs_registered(records) -> bool:
    """False if any run record says it was made unregistered (older records carry no flag: the hash comparison covers them)."""
    return all(r.get("registered", True) for r in records)


def require_registered_runs(records, reg: Registration):
    """An aggregate that is not itself unregistered must not be built from unregistered runs."""
    if reg.registered and not runs_registered(records):
        raise CriteriaError("some runs were made with --allow-unregistered: they are not gate evidence")


def main(argv=None):
    ap = argparse.ArgumentParser(description="print a criteria file's SHA-256 and whether it matches its pin")
    ap.add_argument("file")
    args = ap.parse_args(argv)
    reg = check(args.file, allow_unregistered=True)
    print(reg.sha256, "registered" if reg.registered else f"UNREGISTERED ({reg.reason})")
    return 0 if reg.registered else 1


if __name__ == "__main__":
    sys.exit(main())
