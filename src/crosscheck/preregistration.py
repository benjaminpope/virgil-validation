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
"""

from __future__ import annotations

import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]


class CriteriaError(SystemExit):
    """Raised (as an exit) when a gate would run on criteria that are not a committed, single version."""


def criteria_commit(path, root=None) -> str:
    """The last commit that touched ``path`` (repo-relative or absolute). Raises if the file has uncommitted changes."""
    root = pathlib.Path(root or ROOT)
    path = pathlib.Path(path)
    rel = (path if path.is_absolute() else root / path).resolve().relative_to(root.resolve()).as_posix()
    git = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True)
    if git("rev-parse", "--git-dir").returncode:  # not inside any repository: a git-archive snapshot
        commit = os.environ.get("CRITERIA_COMMIT", "")
        if not commit:
            raise CriteriaError(f"{rel}: not a git checkout and CRITERIA_COMMIT is not set")
        return commit
    if git("ls-files", "--error-unmatch", "--", rel).returncode:  # in a repository, CRITERIA_COMMIT is never trusted
        raise CriteriaError(f"{rel} is not tracked by git")
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
