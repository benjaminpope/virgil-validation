"""Join the trust graph, the evidence and virgil's history into docs/trust.md.

    python scripts/trust.py --evidence evidence/latest.jsonl \
        [--evidence evidence/virgil.jsonl] [--virgil ~/code/drpangloss] \
        [--out docs/trust.md]

Each node of trust/graph.yml gets a status:

* failing     a check of it failed (or a strict xfail passed unexpectedly)
* open        a ledger entry ruled against virgil is still open
* stale       trusted once, but virgil's source under it changed since the
              commit the evidence ran on
* trusted     passing evidence from enough distinct strong roots, and every
              dependency trusted
* partial     some evidence, but too few roots, a dependency not trusted, or
              only self-consistency
* unchecked   no evidence

A pipeline is validated when its own evidence passes and all its steps are
trusted.
"""

import argparse
import collections
import json
import pathlib
import subprocess

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
WEAK = {"self-consistency"}
BAD = {"failed", "error", "xpassed"}
GOOD = {"passed", "xfailed"}


def load_evidence(paths):
    runs, records = [], []
    for path in paths:
        lines = [json.loads(line) for line in open(path)]
        run, tests = lines[0], lines[1:]
        runs.append(run)
        for t in tests:
            t["_commit"] = (run.get("virgil") or {}).get("commit")
            t["_source"] = t.get("source", "ours")
            records.append(t)
    return runs, records


def changed_since(virgil, commit, paths):
    """Whether any of ``paths`` changed in virgil between ``commit`` and
    origin/main (None if that cannot be told)."""
    if virgil is None or commit is None or not paths:
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(virgil), "diff", "--quiet", commit, "origin/main", "--", *paths],
            capture_output=True,
        )
    except OSError:
        return None
    if out.returncode in (0, 1):
        return out.returncode == 1
    return None


def statuses(graph, ledger, records, virgil):
    nodes = graph["nodes"]
    by_obj = collections.defaultdict(list)
    for r in records:
        for obj in r["objects"]:
            by_obj[obj].append(r)
    open_findings = {
        obj
        for e in ledger
        if e["ruling"] == "virgil" and e["status"] == "open"
        for obj in e["objects"]
    }
    info = {}
    for name, node in nodes.items():
        recs = by_obj.get(name, [])
        failing = [r for r in recs if r["outcome"] in BAD and r["kind"] not in ("finding", "upstream")]
        good = [r for r in recs if r["outcome"] in GOOD and r["kind"] not in ("finding", "upstream")]
        strong = sorted({root for r in good for root in r["roots"] if root not in WEAK})
        weak = sorted({root for r in good for root in r["roots"] if root in WEAK})
        commits = {r["_commit"] for r in good if r["_commit"]}
        stale = None
        if commits:
            flags = [changed_since(virgil, c, node.get("source", [])) for c in commits]
            stale = all(f for f in flags) if all(f is not None for f in flags) else None
        need = node.get("roots", 1)
        if failing:
            status = "failing"
        elif name in open_findings:
            status = "open"
        elif len(strong) >= need:
            status = "stale" if stale else "trusted"
        elif strong or weak:
            status = "partial"
        else:
            status = "unchecked"
        info[name] = {
            "status": status,
            "strong": strong,
            "weak": weak,
            "need": need,
            "tests": len({r["test"] for r in recs}),
            "commits": sorted(commits),
            "failing": [r["test"] for r in failing],
        }
    # trust only flows upwards: a trusted node with an untrusted dependency
    # is partial (iterate to a fixed point)
    changed = True
    while changed:
        changed = False
        for name, node in nodes.items():
            if info[name]["status"] != "trusted":
                continue
            weak_deps = [d for d in node.get("depends", []) if info.get(d, {}).get("status") != "trusted"]
            if weak_deps:
                info[name]["status"] = "partial"
                info[name]["blocked_by"] = weak_deps
                changed = True
    return info, by_obj


def pipeline_statuses(graph, info, by_obj):
    out = {}
    for name, p in graph.get("pipelines", {}).items():
        recs = by_obj.get(f"pipeline:{name}", [])
        bad = [r for r in recs if r["outcome"] in BAD]
        good = [r for r in recs if r["outcome"] in GOOD]
        steps_ok = all(info.get(s, {}).get("status") == "trusted" for s in p["steps"])
        if bad:
            status = "failing"
        elif good and steps_ok:
            status = "validated"
        elif good:
            status = "end-to-end only"
        else:
            status = "planned" if p.get("planned") else "no evidence"
        out[name] = {
            "status": status,
            "title": p["title"],
            "steps": p["steps"],
            "roots": sorted({root for r in good for root in r["roots"]}),
            "tests": len({r["test"] for r in recs}),
            "untrusted_steps": [s for s in p["steps"] if info.get(s, {}).get("status") != "trusted"],
        }
    return out


ICON = {
    "trusted": "✅", "partial": "🟡", "stale": "🕒", "open": "🔶",
    "failing": "❌", "unchecked": "⬜",
    "validated": "✅", "end-to-end only": "🟡", "planned": "⬜", "no evidence": "⬜",
}
CLASS = {
    "trusted": "done", "partial": "part", "stale": "part", "open": "part",
    "failing": "bad", "unchecked": "todo",
}


def render(graph, ledger, info, pipes, runs):
    lines = [
        "# Trust",
        "",
        "<!-- Generated by scripts/trust.py from trust/graph.yml, trust/ledger.yml and the evidence. Do not edit. -->",
        "",
    ]
    for run in runs:
        v = run.get("virgil") or {}
        lines.append(
            f"- Evidence from {run.get('runner')} on {run.get('date')}, virgil "
            f"`{(v.get('commit') or 'unknown')[:10]}`"
            + (f" ([run]({run['run_url']}))" if run.get("run_url") else "")
        )
    counts = collections.Counter(i["status"] for i in info.values())
    lines += [
        "",
        "Statuses: " + ", ".join(f"{ICON[k]} {k} {counts[k]}" for k in ICON if k in counts) + ".",
        "",
        "A node is **trusted** when it passes checks against enough distinct strong roots",
        "(mathematics, standards, statistics, literature, or a trusted package) and every",
        "node it is built on is trusted; **stale** when virgil's source under it changed",
        "after the evidence was measured. See [design](design.md).",
        "",
        "## Graph",
        "",
        "```mermaid",
        "flowchart BT",
        "    classDef done fill:#c8e6c9,stroke:#2e7d32,color:#000",
        "    classDef part fill:#fff3c4,stroke:#b28704,color:#000",
        "    classDef bad fill:#ffcdd2,stroke:#c62828,color:#000",
        "    classDef todo fill:#eeeeee,stroke:#9e9e9e,color:#555,stroke-dasharray: 5 5",
    ]
    ids = {name: f"n{k}" for k, name in enumerate(graph["nodes"])}
    layers = collections.defaultdict(list)
    for name, node in graph["nodes"].items():
        layers[node.get("layer", "other")].append(name)
    for layer in ["references", "data", "models", "likelihood", "inference", "imaging"]:
        if layer not in layers:
            continue
        lines.append(f'    subgraph {layer}["{layer}"]')
        for name in layers[layer]:
            label = name.replace("virgil.", "").replace("models.", "")
            lines.append(f'        {ids[name]}["{label}"]:::{CLASS[info[name]["status"]]}')
        lines.append("    end")
    for name, node in graph["nodes"].items():
        for dep in node.get("depends", []):
            if dep in ids:
                lines.append(f"    {ids[dep]} --> {ids[name]}")
    lines += ["```", "", "## Objects", "",
              "| object | status | strong roots (needed) | weak | tests | evidence on virgil |",
              "| --- | --- | --- | --- | --- | --- |"]
    for name in graph["nodes"]:
        i = info[name]
        note = f" (blocked by {', '.join(f'`{d}`' for d in i['blocked_by'])})" if i.get("blocked_by") else ""
        lines.append(
            f"| `{name}` | {ICON[i['status']]} {i['status']}{note} | {', '.join(i['strong']) or '—'} ({i['need']}) "
            f"| {', '.join(i['weak']) or ''} | {i['tests']} | {', '.join(c[:7] for c in i['commits'])} |"
        )
    lines += ["", "## Pipelines", "",
              "End-to-end chains, validated when their own evidence passes and every step is trusted.",
              "", "| pipeline | status | roots | tests | steps not yet trusted |", "| --- | --- | --- | --- | --- |"]
    for name, p in pipes.items():
        lines.append(
            f"| **{name}**: {p['title']} | {ICON[p['status']]} {p['status']} | {', '.join(p['roots']) or '—'} "
            f"| {p['tests']} | {', '.join(f'`{s}`' for s in p['untrusted_steps']) or '—'} |"
        )
    lines += ["", "## Ledger", "",
              "Every mismatch found, and its ruling: `virgil`, `external:<package>`, `definition` or `crosscheck` (our own code).",
              "", "| id | ruling | status | finding | referee | links |", "| --- | --- | --- | --- | --- | --- |"]
    for e in ledger:
        links = " ".join(f"[{link.rstrip('/').split('/')[-1]}]({link})" for link in e.get("links", []))
        lines.append(f"| {e['id']} | {e['ruling']} | {e['status']} | {e['title']} | {e['referee']} | {links} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", action="append", required=True)
    ap.add_argument("--graph", default=ROOT / "trust" / "graph.yml")
    ap.add_argument("--ledger", default=ROOT / "trust" / "ledger.yml")
    ap.add_argument("--virgil", type=pathlib.Path, default=None)
    ap.add_argument("--out", default=ROOT / "docs" / "trust.md")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    graph = yaml.safe_load(open(args.graph))
    ledger = yaml.safe_load(open(args.ledger))
    runs, records = load_evidence(args.evidence)
    virgil = args.virgil.expanduser() if args.virgil else None
    info, by_obj = statuses(graph, ledger, records, virgil)
    pipes = pipeline_statuses(graph, info, by_obj)
    pathlib.Path(args.out).write_text(render(graph, ledger, info, pipes, runs))
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps({"nodes": info, "pipelines": pipes}, indent=1))
    counts = collections.Counter(i["status"] for i in info.values())
    print(dict(counts), {k: v["status"] for k, v in pipes.items()})


if __name__ == "__main__":
    main()
