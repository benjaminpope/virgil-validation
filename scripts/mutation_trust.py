#!/usr/bin/env python
"""Mutation results as trust evidence: per-node status and the sign-off checklist.

Reads the mutation report (``trust/mutation/results.json``, written by
``scripts/mutation.py report``) and the mutant list (``mutants/mutants.yml``).

* ``node_summary``: for every trust-graph node, how many mutants target it and how
  many were killed, survived, unresolved (timed out) or not run. ``scripts/trust.py``
  shows it on the Trust page and lets a node be "verified" only if no mutant of it
  survived. A node with no mutant run is labelled "mutants not run".
* ``signoff_candidates`` / ``write_pending``: the nodes where a mutant survives and
  the slip is in a convention that virgil's own documentation defines (position-angle
  origin and sense, uv axes, the Fourier sign, the closure-phase sign). They go to
  ``trust/signoffs.yml`` as PENDING entries for Ben; a pending entry confirms nothing.

Usage (from the repository root)::

    python scripts/mutation_trust.py checklist             # print the checklist from the report
    python scripts/mutation_trust.py checklist --write     # add pending entries to trust/signoffs.yml
    python scripts/mutation_trust.py checklist --report PATH --write   # e.g. a report from the OzSTAR run

Run it again after every ``mutation.py report``: it re-derives the list, adds entries,
and refreshes (never removes, never signs) the pending ones.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORT = ROOT / "trust" / "mutation" / "results.json"
SIGNOFFS = ROOT / "trust" / "signoffs.yml"

# Classes of slip whose right answer is a convention, defined by virgil's documentation
# (and the usual astronomical one) rather than by mathematics: the property each is
# filed under in trust/signoffs.yml.
CONVENTION_CLASSES = {
    "pa_origin": "position_angle_origin",
    "pa_sign": "position_angle_sense",
    "uv_swap": "uv_axes",
    "uv_sign": "uv_sign",
    "ft_conj_sign": "fourier_sign",
    "closure_sign": "closure_phase_sign",
}
PENDING = "pending"


def _mutation_module():
    spec = importlib.util.spec_from_file_location("mutation", ROOT / "scripts" / "mutation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_report(path=REPORT):
    """The report as a dict, or ``None`` when there is none (the page then builds without it)."""
    path = pathlib.Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def load_listed(path=None):
    """The mutants of mutants/mutants.yml, ``[{id, cls, nodes, finding, equivalent, ...}]``."""
    mutation = _mutation_module()
    return mutation.load_mutants(path or mutation.MUTANTS)[1]


def mutant_status(mutant_id, report, listed_by_id=None):
    """killed | survived | equivalent | unresolved | not run, for one mutant.

    A full-suite result and a targeted one (only the tests of the mutant's nodes and
    their dependents) both count: a kill in either is a kill; a survivor of either (with
    no kill) is a survivor, since the tests written for its node do not notice it; a
    timeout or error with nothing better is unresolved.
    """
    meta = (listed_by_id or {}).get(mutant_id, {})
    if meta.get("equivalent"):
        return "equivalent"
    rows = [r for r in (*([r for r in report.get("mutants", []) if r["id"] == mutant_id]),
                        report.get("targeted", {}).get(mutant_id)) if r]
    states = {r["status"] for r in rows}
    if "killed" in states:
        return "killed"
    if "survived" in states:
        return "survived"
    return "unresolved" if states else "not run"


def node_summary(report, listed):
    """``{node: {listed, killed, survived, unresolved, not_run, equivalent, survivors, text}}``.

    ``text`` is the sentence the Trust page shows. Nodes that no listed mutant targets
    are absent: the page says "no mutants written" for them.
    """
    by_id = {m["id"]: m for m in listed}
    out = {}
    for m in listed:
        status = mutant_status(m["id"], report, by_id)
        for node in m.get("nodes", []):
            s = out.setdefault(node, {"listed": 0, "killed": 0, "survived": 0, "unresolved": 0,
                                      "not_run": 0, "equivalent": 0, "survivors": [], "unresolved_ids": []})
            s["listed"] += 1
            key = {"not run": "not_run"}.get(status, status)
            s[key] += 1
            if status == "survived":
                s["survivors"].append(m["id"])
            if status == "unresolved":
                s["unresolved_ids"].append(m["id"])
    for s in out.values():
        s["text"] = _text(s)
    return out


def _text(s):
    """One sentence: the mutants of a node and what became of them."""
    real = s["listed"] - s["equivalent"]
    eq = f" ({s['equivalent']} equivalent, not counted)" if s["equivalent"] else ""
    if s["survivors"]:
        n = len(s["survivors"])
        return (f"{n} of {real} mutants survive ({', '.join(s['survivors'])}): "
                f"a slip of this kind would not be noticed{eq}.")
    if real == 0:
        return f"only equivalent mutants listed{eq}."
    if s["killed"] == real:
        return f"all {real} mutants killed{eq}."
    run = s["killed"] + s["unresolved"]
    if run == 0:
        return f"mutants not run ({real} listed){eq}."
    rest = []
    if s["unresolved"]:
        rest.append(f"{s['unresolved']} unresolved ({', '.join(s['unresolved_ids'])})")
    if s["not_run"]:
        rest.append(f"{s['not_run']} not run")
    return f"{s['killed']} of {real} mutants killed, {', '.join(rest)}{eq}."


def page_data(report, listed):
    """What the Trust page needs: ``{"nodes": node_summary, "summary": {...}}``, or ``None`` without a report."""
    if report is None:
        return None
    by_id = {m["id"]: m for m in listed}
    counts = collections.Counter(mutant_status(m["id"], report, by_id) for m in listed)
    run = counts["killed"] + counts["survived"] + counts["unresolved"]
    return {
        "nodes": node_summary(report, listed),
        "summary": {"listed": len(listed), "run": run, "killed": counts["killed"], "survived": counts["survived"],
                    "unresolved": counts["unresolved"], "equivalent": counts["equivalent"]},
    }


def signoff_candidates(report, listed, graph=None, signed=None):
    """Nodes where a mutant survives and the convention is virgil's documentation alone.

    ``{(node, property): {"mutants": [ids], "descriptions": [...]}}`` over the
    non-equivalent survivors in a convention class (``CONVENTION_CLASSES``), or on a
    node that declares the property a ``convention`` in the graph. Skips what
    ``signed`` (trust/signoffs.yml, a node -> property -> entry mapping) already
    confirms.
    """
    by_id = {m["id"]: m for m in listed}
    signed = signed or {}
    out = collections.OrderedDict()
    for m in listed:
        prop = CONVENTION_CLASSES.get(m["cls"])
        if not prop or mutant_status(m["id"], report, by_id) != "survived":
            continue
        for node in m["nodes"]:
            entry = (signed.get(node) or {}).get(prop)
            if entry and entry.get("status") != PENDING:
                continue
            slot = out.setdefault((node, prop), {"mutants": [], "descriptions": []})
            slot["mutants"].append(m["id"])
            slot["descriptions"].append(m["description"])
    return out


def pending_entry(prop, slot, status_of=None):
    """The pending entry for one (node, property); ``status_of(id)`` adds what became of each mutant."""
    entry = {
        "status": PENDING,
        "signed": False,
        "mutants": sorted(slot["mutants"]),
        "note": (f"UNSIGNED, for Ben. A mutant of the convention `{prop}` survived, so no test failed if virgil "
                 "took the other reading: " + " ".join(slot["descriptions"])
                 + " Confirm the convention against a paper or by reading the code, then replace this entry by "
                 "`by`, `date` (YYYY-MM-DD) and `commit` (full hash), or `citation`, and declare the property a "
                 "`convention` of the node in trust/graph.yml. Killing the mutant with a test written from the "
                 "same documentation does not confirm the convention."),
    }
    if status_of:
        entry["mutants_now"] = {m: status_of(m) for m in entry["mutants"]}
    return entry


def _split_blocks(text):
    """``(header, [(node, block_text)])``: the comment header, then one text block per top-level key."""
    header, blocks, current = [], [], None
    for line in text.splitlines(keepends=True):
        if line.strip() and not line.startswith(("#", " ", "\t")):
            current = [line.split(":")[0].strip(), line]
            blocks.append(current)
        elif current is None:
            header.append(line)
        else:
            current[1] += line
    return "".join(header), [(k, v) for k, v in blocks]


def write_pending(candidates, path=SIGNOFFS, status_of=None):
    """Add pending entries for ``candidates`` to the signoffs file; return the (node, property) pairs written.

    Confirmed entries are never touched (their text is kept as it is); existing pending entries are refreshed
    (never removed, never signed: that is Ben's); the comment header is kept.
    """
    path = pathlib.Path(path)
    text = path.read_text() if path.exists() else ""
    data = yaml.safe_load(text) or {}
    header, blocks = _split_blocks(text)
    touched, changed = {}, []
    for (node, prop), slot in candidates.items():
        old = (data.get(node) or {}).get(prop)
        if old and old.get("status") != PENDING:
            continue
        touched.setdefault(node, dict(data.get(node) or {}))[prop] = pending_entry(prop, slot, status_of)
        changed.append((node, prop))
    for node, props in data.items():  # refresh the status of pending entries that are no longer candidates
        for prop, entry in props.items():
            if entry.get("status") == PENDING and status_of and (node, prop) not in candidates:
                touched.setdefault(node, dict(props))[prop] = {**entry, "mutants_now": {m: status_of(m) for m in entry["mutants"]}}
    if not touched:
        return changed
    dump = lambda node, props: yaml.safe_dump({node: props}, sort_keys=False, width=100, allow_unicode=True)
    out, seen = [header.rstrip("\n") + "\n" if header.strip() else ""], set()
    for node, block in blocks:
        seen.add(node)
        out.append(dump(node, touched[node]) if node in touched else block)
    out += [dump(node, props) for node, props in touched.items() if node not in seen]
    path.write_text("".join(out))
    return changed


def render_checklist(candidates):
    if not candidates:
        return "No convention-defined mutant survives in the report: nothing new for sign-off.\n"
    lines = ["Sign-off checklist: a mutant survives and the convention rests on virgil's documentation alone", ""]
    for (node, prop), slot in candidates.items():
        lines.append(f"[ ] {node}  /  {prop}   (survivors: {', '.join(slot['mutants'])})")
        lines += [f"      - {d}" for d in slot["descriptions"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("checklist")
    c.add_argument("--report", default=str(REPORT))
    c.add_argument("--signoffs", default=str(SIGNOFFS))
    c.add_argument("--write", action="store_true", help="add the pending entries to the signoffs file")
    args = ap.parse_args(argv)
    report = load_report(args.report)
    if report is None:
        print(f"no report at {args.report} (run scripts/mutation.py report)", file=sys.stderr)
        return 2
    signed = (yaml.safe_load(open(args.signoffs)) or {}) if pathlib.Path(args.signoffs).exists() else {}
    candidates = signoff_candidates(report, load_listed(), signed=signed)
    print(render_checklist(candidates), end="")
    if args.write:
        listed = load_listed()
        by_id = {m["id"]: m for m in listed}
        done = write_pending(candidates, args.signoffs, lambda i: mutant_status(i, report, by_id))
        print(f"{len(done)} pending entr{'y' if len(done) == 1 else 'ies'} written to {args.signoffs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
