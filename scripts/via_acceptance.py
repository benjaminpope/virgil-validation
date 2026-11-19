"""ONE-OFF (B2 of the trust-graph plan; deleted before the pull request is reviewed).

Compares, for every test that PASSED in a full run made with ``VV_VIA_ACCEPTANCE=1``, the ``via`` recorded by the new
mechanism (file scan + run-time recording) with ``via_old`` recorded by the item-level scanner it replaces:

    python scripts/via_acceptance.py evidence.jsonl [out.md]

A ``via`` that gates a passing check is complete for this purpose if it contains the old one. Reference nodes that
``scripts/trust.py`` verifies only when their ``depends`` are verified count as contained when a node that depends on
them is in the new ``via``. The old scanner's "unresolved" sentinels are scanner failures, not references: they are
listed apart. Exit status 1 when some passed test's new ``via`` lacks an old reference.
"""

import collections
import json
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
SENTINELS = {"external_bridge.<unresolved_worker>", "crosscheck.<unresolved_dynamic_import>"}


def closure(nodes, depends):
    out, todo = set(), list(nodes)
    while todo:
        n = todo.pop()
        if n not in out:
            out.add(n)
            todo.extend(depends.get(n, ()))
    return out


def main(evidence, out_path=None):
    graph = yaml.safe_load((ROOT / "trust" / "graph.yml").read_text(encoding="utf-8"))
    depends = {n: (v or {}).get("depends", []) for n, v in graph["nodes"].items()}
    records = [json.loads(line) for line in pathlib.Path(evidence).read_text().splitlines()[1:]]
    records = [r for r in records if r.get("record") == "test"]
    passed = [r for r in records if r["outcome"] == "passed"]
    no_old = [r for r in records if "via_old" not in r]

    equal, covered, missing, sentinel, wider = 0, [], [], [], collections.Counter()
    for r in passed:
        new, old = set(r["via"]), set(r.get("via_old", ()))
        sent = old & SENTINELS
        old -= SENTINELS
        if sent:
            sentinel.append((r, sorted(sent)))
        if old <= new:
            equal += new == old
            wider.update(sorted(new - old)) if new != old else None
            continue
        lacking = old - new
        if lacking <= closure(new, depends):
            covered.append((r, sorted(lacking)))
        else:
            missing.append((r, sorted(lacking - closure(new, depends))))

    def table(rows, head):
        by = collections.defaultdict(list)
        for r, nodes in rows:
            by[tuple(nodes)].append(r["test"])
        lines = [f"| {head} | tests | files | example |", "|---|---|---|---|"]
        for nodes, tests in sorted(by.items(), key=lambda kv: -len(kv[1])):
            files = sorted({t.split("::")[0] for t in tests})
            lines.append(f"| {', '.join(f'`{n}`' for n in nodes)} | {len(tests)} | {', '.join(files)[:200]} | `{tests[0][:90]}` |")
        return lines

    out = [
        "## `via` acceptance (B2): new mechanism against the item-level scanner, tests that passed",
        "",
        f"| records | passed | new `via` equals old | new wider than old | old not literally in new, but via `depends` | **old not in new** | old had scanner sentinels | records without `via_old` |",
        f"|---|---|---|---|---|---|---|---|",
        f"| {len(records)} | {len(passed)} | {equal} | {sum(1 for r in passed if set(r['via']) > set(r.get('via_old', ())) - SENTINELS)} "
        f"| {len(covered)} | **{len(missing)}** | {len(sentinel)} | {len(no_old)} |",
        "",
    ]
    if missing:
        out += ["### Old `via` NOT contained in the new one (each needs a fix or a reason)", "", *table(missing, "missing from new"), ""]
    if covered:
        out += ["### Contained through `depends` (a node that depends on it is in the new `via`)", "", *table(covered, "old nodes"), ""]
    if sentinel:
        out += ["### Old scanner could not resolve these (sentinel dropped; the new `via` is what ran)", ""]
        out += [f"- `{r['test']}`: old had {s}; new `via` = {r['via']}" for r, s in sentinel[:60]]
        out += [""]
    if wider:
        out += ["### Nodes the new `via` adds (file-level over-approximation, or things the old scanner missed)", "",
                "| node | tests |", "|---|---|", *[f"| `{n}` | {c} |" for n, c in wider.most_common(40)], ""]
    text = "\n".join(out)
    print(text)
    if out_path:
        pathlib.Path(out_path).write_text(text + "\n", encoding="utf-8")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
