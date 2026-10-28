"""Pre-registered gate criteria: a committed file, its pinned SHA-256, and the check every gate script runs.

The rule (virgil 0.4.2 release review, section 3.1): a criteria file is committed and hashed before any
run; the run records the hash; a mismatch fails. ``design/criteria_registry.json`` pins one hash per
criteria file. ``check`` hashes the file now and compares it with the pin; a gate script writes the returned
``Registration`` into its output with ``stamp`` (JSON) or ``stamp_lines`` (text). A gate run with another
hash exits non-zero. ``--allow-unregistered`` (``add_argument``) is for development runs only: it lets the
run go ahead, and every output then carries ``UNREGISTERED_STAMP`` and ``registered: false``.

Changing a pinned file means changing its pin in the same commit, so the diff shows it. A change after a
result was seen is "post-hoc": say so in the registry entry (``changed_after_seeing_a_result``) and the
release notes. Nothing here changes a criterion's value.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "design" / "criteria_registry.json"
UNREGISTERED_STAMP = "UNREGISTERED: not gate evidence"


class CriteriaError(SystemExit):
    """Raised (as an exit) when a gate would run on criteria that are not the pinned ones."""


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
