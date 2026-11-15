"""Which parts and pipelines change verdict between two Trust models (trust.py --json).

    .venv/bin/python scripts/trust_diff.py before.json after.json

Prints one line per part or pipeline whose verdict, status or roots changed, for
a pull request's description: every drop must be explained.
"""

import json
import sys


def load_doc(path):
    return json.load(open(path))


def load(path):
    return load_doc(path)["nodes"]


def diff_rows(before_doc, after_doc):
    a, b = before_doc["nodes"], after_doc["nodes"]
    rows = []
    for name in sorted(set(a) | set(b)):
        x, y = a.get(name), b.get(name)
        if x is None or y is None:
            rows.append(f"| `{name}` | {'new' if x is None else x['verdict']} | {'removed' if y is None else y['verdict']} | |")
            continue
        if x["verdict"] != y["verdict"] or x["strong"] != y["strong"]:
            why = []
            if set(x["strong"]) - set(y["strong"]):
                why.append("no longer counted: " + ", ".join(sorted(set(x["strong"]) - set(y["strong"]))))
            if set(y["strong"]) - set(x["strong"]):
                why.append("newly counted: " + ", ".join(sorted(set(y["strong"]) - set(x["strong"]))))
            if y.get("because"):
                why.append("relies on " + ", ".join(c["id"] for c in y["because"]))
            if y.get("findings"):
                why.append("finding " + ", ".join(y["findings"]))
            rows.append(f"| `{name}` | {x['verdict']} | {y['verdict']} | {'; '.join(why)} |")
    pa = {p["id"]: p for p in before_doc.get("pipelines", [])}
    pb = {p["id"]: p for p in after_doc.get("pipelines", [])}
    for pid in sorted(set(pa) | set(pb)):
        x, y = pa.get(pid), pb.get(pid)
        if x is None or y is None:
            rows.append(f"| `pipeline:{pid}` | {'new' if x is None else x['status']} | {'removed' if y is None else y['status']} | |")
            continue
        if x["status"] != y["status"]:
            why = []
            if y.get("failing_checks"):
                why.append("failing checks: " + ", ".join(y["failing_checks"]))
            after_nodes = after_doc.get("nodes", {})
            failing_steps = [
                s for s in y.get("steps", [])
                if after_nodes.get(s, {}).get("verdict") == "failing"
            ]
            if failing_steps:
                why.append("failing steps: " + ", ".join(failing_steps))
            step_set = set(y.get("steps", []))
            failing_via = [
                v for v in y.get("unverified", [])
                if v not in step_set and after_nodes.get(v, {}).get("verdict") == "failing"
            ]
            if failing_via:
                why.append("failing via: " + ", ".join(failing_via))
            rem_unverified = [
                u for u in y.get("unverified", [])
                if u not in set(failing_steps) and u not in set(failing_via)
            ]
            if rem_unverified:
                why.append("unverified steps: " + ", ".join(rem_unverified))
            rows.append(f"| `pipeline:{pid}` | {x['status']} | {y['status']} | {'; '.join(why)} |")
    return rows


def main(before, after):
    rows = diff_rows(load_doc(before), load_doc(after))
    print("| Part | Before | After | Why |\n| --- | --- | --- | --- |")
    print("\n".join(rows) if rows else "| (no changes) | | | |")


if __name__ == "__main__":
    main(*sys.argv[1:3])
