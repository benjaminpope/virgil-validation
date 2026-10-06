"""Which parts change verdict between two Trust models (trust.py --json).

    .venv/bin/python scripts/trust_diff.py before.json after.json

Prints one line per part whose verdict or roots changed, for a pull
request's description: every drop must be explained.
"""

import json
import sys


def load(path):
    return json.load(open(path))["nodes"]


def main(before, after):
    a, b = load(before), load(after)
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
    print("| Part | Before | After | Why |\n| --- | --- | --- | --- |")
    print("\n".join(rows) if rows else "| (no changes) | | | |")


if __name__ == "__main__":
    main(*sys.argv[1:3])
