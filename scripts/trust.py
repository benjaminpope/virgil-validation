"""Join the trust graph, the evidence and virgil's history into the Trust page.

    python scripts/trust.py --evidence trust/evidence/latest.jsonl \\
        [--evidence trust/evidence/virgil.jsonl] [--virgil ~/code/drpangloss] \\
        [--out docs/trust.md] [--index docs/index.md] [--json docs/assets/trust.json]

Two questions are kept apart:

* the **verdict**, from the evidence at the virgil commit it was measured on:

  - verified      independent checks against enough distinct strong roots
                  pass, and every part it relies on is verified;
  - relies        its own checks pass, but a part it relies on is not
                  verified (the reason names the parts at the bottom of the
                  chain, and any open finding there);
  - bug           a ledger entry ruled against virgil is still open;
  - partly        checked, but against too few independent roots (or only
                  virgil against itself);
  - failing       a check of it fails (or a strict xfail passes);
  - unchecked     no evidence;

* the **freshness**: whether virgil's source under a part has changed since
  the commit its evidence ran on (from ``git diff`` in a virgil checkout).

A pipeline (an end-to-end chain) passes when its own evidence passes; it is
validated when, in addition, every step is verified.
"""

import argparse
import collections
import html
import json
import math
import pathlib
import re
import subprocess

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = "https://github.com/benjaminpope/virgil-validation"
WEAK = {"self-consistency"}
BAD = {"failed", "error", "xpassed"}
GOOD = {"passed", "xfailed"}
LAYER_ORDER = ["imaging", "inference", "orbits", "likelihood", "priors", "models", "data", "references"]  # top first

VERDICTS = {
    "verified": ("Verified", "independent checks agree, and so does everything it relies on"),
    "relies": ("Works, but relies on a known bug", "its own checks pass, but it uses a part with an unfixed bug"),
    "relies-unverified": ("Works, but relies on an unverified part", "its own checks pass, but it uses a part not yet verified"),
    "bug": ("Known bug, fix pending", "the checks found a mistake in virgil that is not fixed yet"),
    "partly": ("Partly checked", "it needs another independent check"),
    "failing": ("Check failing", "a check of it fails"),
    "unchecked": ("Not yet checked", "no independent check yet"),
}
ROOT_NAMES = {
    "mathematics": "mathematics", "standards": "standards", "statistics": "statistics",
    "literature": "published result", "dlux": "dLux", "pmoired": "PMOIRED", "candid": "CANDID",
    "fouriever": "fouriever", "ehtim": "eht-imaging", "mpol": "MPoL", "orbitize": "orbitize!",
    "self-consistency": "virgil itself",
}
# metric names that state a difference or error (raw statistics such as a
# reduced chi-squared need an explicit headline=)
AGREEMENT = re.compile(r"(^|_)(rel|abs|diff|difference|err|error|dv|dv2|dsigma|dloglike|dlogb)(_|$)", re.I)


# ------------------------------------------------------------------ inputs


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


def changed_files(virgil, commit, paths):
    """Files under ``paths`` changed in virgil between ``commit`` and
    origin/main ([] if none, None if that cannot be told)."""
    if virgil is None or commit is None or not paths:
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(virgil), "diff", "--name-only", commit, "origin/main", "--", *paths],
            capture_output=True, text=True,
        )
    except OSError:
        return None
    if out.returncode != 0:
        return None
    return [line for line in out.stdout.splitlines() if line]


# --------------------------------------------------------------- the model


def _headline(record):
    """The number that summarises a check: the marker's ``headline``, else
    the first recorded metric that measures agreement."""
    metrics = record.get("metrics") or {}
    name = record.get("headline")
    if name in metrics:
        return name, metrics[name]
    for key, value in metrics.items():
        if AGREEMENT.search(key) and isinstance(value, (int, float)):
            return key, value
    return None, None


def _check(record):
    name, value = _headline(record)
    path, _, test = record["test"].partition("::")
    line = record.get("line")
    return {
        "test": record["test"],
        "name": test,
        "doc": record.get("doc", ""),
        "roots": record["roots"],
        "kind": record["kind"],
        "tier": record.get("tier", "A"),
        "outcome": record["outcome"],
        "headline": name,
        "value": value,
        "url": f"{REPO}/blob/main/{path}" + (f"#L{line}" if line else "") if record["_source"] == "ours" else None,
        "source": record["_source"],
    }


def build(graph, ledger, records, runs, virgil=None):
    nodes = graph["nodes"]
    by_obj = collections.defaultdict(list)
    for r in records:
        for obj in r["objects"]:
            by_obj[obj].append(r)
    open_findings = collections.defaultdict(list)
    for e in ledger:
        if e["ruling"] == "virgil" and e["status"] == "open":
            for obj in e["objects"]:
                open_findings[obj].append(e["id"])

    own = {}
    for name, node in nodes.items():
        recs = by_obj.get(name, [])
        counted = [r for r in recs if r["kind"] not in ("finding", "upstream")]
        failing = [r for r in counted if r["outcome"] in BAD]
        good = [r for r in counted if r["outcome"] in GOOD]
        strong = sorted({root for r in good for root in r["roots"] if root not in WEAK})
        weak = sorted({root for r in good for root in r["roots"] if root in WEAK})
        need = node.get("roots", 1)
        if failing:
            status = "failing"
        elif name in open_findings:
            status = "bug"
        elif len(strong) >= need:
            status = "ok"
        elif strong or weak:
            status = "partly"
        else:
            status = "unchecked"
        commits = sorted({r["_commit"] for r in good if r["_commit"]})
        changes = [changed_files(virgil, c, node.get("source", [])) for c in commits]
        changed = None if not changes or any(c is None for c in changes) else sorted({f for c in changes for f in c})
        own[name] = {
            "status": status, "strong": strong, "weak": weak, "need": need,
            "checks": [_check(r) for r in recs], "commits": commits, "changed": changed,
            "findings": open_findings.get(name, []),
        }

    # verdicts: a part whose own checks pass is verified only if everything
    # it relies on is; otherwise name the parts at the bottom of the chain
    def problems(name, seen):
        found = set()
        for dep in nodes[name].get("depends", []):
            if dep in seen or dep not in nodes:
                continue
            seen.add(dep)
            if own[dep]["status"] != "ok":
                found.add(dep)
            found |= problems(dep, seen)
        return found

    out_nodes = {}
    for name, node in nodes.items():
        o = own[name]
        verdict = o["status"]
        causes = sorted(problems(name, set()))
        if verdict == "ok":
            if not causes:
                verdict = "verified"
            elif any(own[c]["status"] == "bug" for c in causes):
                verdict = "relies"
            else:
                verdict = "relies-unverified"
        dependents = sorted(n for n, other in nodes.items() if name in other.get("depends", []))
        out_nodes[name] = {
            "id": name,
            "label": re.sub(r"^virgil\.((models|imaging|likelihood|grid_fit|limits|oidata|fitting|inference|spectra|orbits|gains)\.)?", "", name),
            "layer": node.get("layer", "references"),
            "verdict": verdict,
            "verdict_label": VERDICTS[verdict][0],
            "strong": o["strong"], "weak": o["weak"], "need": o["need"],
            "findings": o["findings"],
            "because": [{"id": c, "findings": own[c]["findings"], "status": own[c]["status"]} for c in causes],
            "depends": node.get("depends", []), "dependents": dependents,
            "checks": o["checks"], "commits": o["commits"], "changed": o["changed"],
            "source": node.get("source", []),
        }

    pipelines = []
    for name, p in graph.get("pipelines", {}).items():
        recs = by_obj.get(f"pipeline:{name}", [])
        bad = [r for r in recs if r["outcome"] in BAD]
        good = [r for r in recs if r["outcome"] in GOOD]
        unverified = [s for s in p["steps"] if out_nodes.get(s, {}).get("verdict") != "verified"]
        if bad:
            status = "failing"
        elif good and not unverified:
            status = "validated"
        elif good:
            status = "passes"
        else:
            status = p.get("state", "planned")
        pipelines.append({
            "id": name, "title": p["title"], "data": p.get("data", "simulated"),
            "dataset": p.get("dataset"), "reference": p.get("reference"), "url": p.get("url"),
            "headline": p.get("headline"), "status": status, "steps": p["steps"],
            "unverified": unverified, "checks": [_check(r) for r in recs],
        })

    counts = collections.Counter(n["verdict"] for n in out_nodes.values())
    changed = sum(1 for n in out_nodes.values() if n["changed"])
    unknown = all(n["changed"] is None for n in out_nodes.values() if n["commits"])
    return {
        "runs": [{k: r.get(k) for k in ("date", "runner", "run_url")} | {"virgil": (r.get("virgil") or {}).get("commit")} for r in runs],
        "counts": dict(counts),
        "changed": None if unknown else changed,
        "nodes": out_nodes,
        "pipelines": pipelines,
        "ledger": ledger,
    }


# ------------------------------------------------------------------ render

ICON = {"verified": "✓", "relies": "↧", "relies-unverified": "↧", "bug": "!", "partly": "◐", "failing": "✗", "unchecked": "–"}
ORDER = ["verified", "relies", "relies-unverified", "partly", "bug", "failing", "unchecked"]
COLOURS = {  # Okabe–Ito based, readable on light and dark
    "verified": "#2e9e5b", "relies": "#e0a526", "relies-unverified": "#c9b458", "partly": "#56b4e9",
    "bug": "#d55e00", "failing": "#b0003a", "unchecked": "#9e9e9e",
}


def e(text):
    return html.escape(str(text), quote=True)


def _bar(counts, total):
    x, parts = 0.0, []
    for v in ORDER:
        n = counts.get(v, 0)
        if not n:
            continue
        w = 100.0 * n / total
        parts.append(f'<rect x="{x:.3f}%" y="0" width="{w:.3f}%" height="10" fill="{COLOURS[v]}"><title>{n} {e(VERDICTS[v][0].lower())}</title></rect>')
        x += w
    return f'<svg class="vt-bar" width="100%" height="10" role="img" aria-label="share of parts by verdict">{"".join(parts)}</svg>'


def _legend(counts):
    items = []
    for v in ORDER:
        if counts.get(v):
            items.append(f'<span class="vt-key"><span class="vt-sw vt-{v}" aria-hidden="true">{ICON[v]}</span>{e(VERDICTS[v][0])} ({counts[v]})</span>')
    return '<div class="vt-legend">' + "".join(items) + "</div>"


def _value(v):
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return f"{v:.2g}" if (v != 0 and (abs(v) < 1e-2 or abs(v) >= 1e4)) else f"{v:.3g}"


def _pipeline_card(p, nodes):
    badge = {"validated": "agrees", "passes": "agrees (some steps not yet verified)", "failing": "disagrees",
             "planned": "planned", "running": "running"}.get(p["status"], p["status"])
    ref = e(p["reference"]) if p["reference"] else ""
    if p.get("url") and ref:
        ref = f'<a href="{e(p["url"])}">{ref}</a>'
    head = ""
    good = [c for c in p["checks"] if c["outcome"] in GOOD and c["value"] is not None]
    if good:
        head = f'{e(good[0]["headline"])} = {_value(good[0]["value"])}'
    elif p.get("headline"):
        head = e(p["headline"])
    return (
        f'<div class="vt-card vt-p-{e(p["status"])}"><div class="vt-card-t">{e(p["title"])}</div>'
        f'<div class="vt-muted">{e(p["dataset"] or "")}{" · " + ref if ref else ""}</div>'
        f'<div class="vt-card-h">{head}</div><span class="vt-badge">{e(badge)}</span></div>'
    )


def _chain(p, nodes):
    steps = []
    for s in p["steps"]:
        n = nodes.get(s)
        v = n["verdict"] if n else "unchecked"
        steps.append(f'<button type="button" class="vt-step vt-{v}" data-node="{e(s)}" title="{e(s)}: {e(VERDICTS[v][0])}">'
                     f'<span aria-hidden="true">{ICON[v]}</span> {e(n["label"] if n else s)}</button>')
    good = [c for c in p["checks"] if c["outcome"] in GOOD and c["value"] is not None]
    result = e(p.get("headline") or p["status"])
    if good:
        result += f'<br><span class="vt-muted">{e(good[0]["headline"])} = {_value(good[0]["value"])}</span>'
    return (
        f'<div class="vt-chain"><span class="vt-chain-t">{e(p["title"])}</span>'
        f'<span class="vt-steps">{" › ".join(steps)}</span><span class="vt-chain-r">{result}</span></div>'
    )


def _tooltip(n):
    lines = [f'{n["id"]}: {VERDICTS[n["verdict"]][0]}']
    if n["strong"] or n["weak"]:
        lines.append("checked against " + ", ".join(ROOT_NAMES.get(r, r) for r in n["strong"] + n["weak"]))
    for b in n["because"]:
        lines.append(("relies on " if n["verdict"].startswith("relies") else "also relies on ")
                     + b["id"] + (" (" + ", ".join(b["findings"]) + ")" if b["findings"] else ""))
    return " · ".join(lines)


def layers(nodes):
    """Every layer in the graph, top first: the known order, then any other."""
    present = {n["layer"] for n in nodes.values()}
    return [layer for layer in LAYER_ORDER if layer in present] + sorted(present - set(LAYER_ORDER))


def _map(nodes):
    rows = []
    for layer in layers(nodes):
        chips = [n for n in nodes.values() if n["layer"] == layer]
        if not chips:
            continue
        cells = "".join(
            f'<button type="button" class="vt-chip vt-{n["verdict"]}" data-node="{e(n["id"])}" title="{e(_tooltip(n))}">'
            f'<span aria-hidden="true">{ICON[n["verdict"]]}</span> {e(n["label"])}'
            + ('<span class="vt-fresh" title="virgil has changed this since">●</span>' if n["changed"] else "")
            + "</button>"
            for n in chips
        )
        rows.append(f'<div class="vt-row"><div class="vt-layer">{e(layer)}</div><div class="vt-chips">{cells}</div></div>')
    return '<div class="vt-map">' + "".join(rows) + '</div><div class="vt-detail" id="vt-detail" aria-live="polite">Select a part to see what it was checked against.</div>'


def _matrix(nodes):
    roots = collections.Counter(r for n in nodes.values() for r in n["strong"])
    cols = [r for r in ("mathematics", "standards", "statistics", "literature") if r in roots]
    cols += sorted(r for r in roots if r not in cols and not r.startswith("golden:"))
    cols += sorted(r for r in roots if r.startswith("golden:"))
    head = "".join(f'<th scope="col"><span>{e(ROOT_NAMES.get(c, c.replace("golden:", "reference code: ")))}</span></th>' for c in cols)
    body = []
    for layer in reversed(layers(nodes)):
        for n in (n for n in nodes.values() if n["layer"] == layer and (n["strong"] or n["weak"])):
            cells = "".join(f'<td>{"●" if c in n["strong"] else ""}</td>' for c in cols)
            body.append(f'<tr><th scope="row"><span class="vt-sw vt-{n["verdict"]}" title="{e(VERDICTS[n["verdict"]][0])}">{ICON[n["verdict"]]}</span>{e(n["label"])}</th>{cells}</tr>')
    return f'<div class="vt-matrix-wrap"><table class="vt-matrix"><thead><tr><th></th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _link_label(url):
    """https://github.com/owner/repo/pull/12 -> repo#12."""
    m = re.match(r"https?://github\.com/[^/]+/([^/]+)/(?:pull|issues)/(\d+)", url)
    return f"{m.group(1)}#{m.group(2)}" if m else url


def _ledger(ledger):
    groups = [
        ("Bugs found in virgil", lambda x: x["ruling"] == "virgil"),
        ("Where codes define things differently", lambda x: x["ruling"] == "definition"),
        ("Problems found in other packages", lambda x: x["ruling"].startswith("external")),
        ("Mistakes in our own checks", lambda x: x["ruling"] == "crosscheck"),
    ]
    out = []
    for title, keep in groups:
        items = [x for x in ledger if keep(x)]
        if not items:
            continue
        cards = []
        for x in items:
            links = " ".join(f'<a href="{e(u)}">{e(_link_label(u))}</a>' for u in x.get("links", []))
            cards.append(f'<div class="vt-finding vt-f-{e(x["status"])}" id="{e(x["id"])}"><span class="vt-fid">{e(x["id"])}</span> '
                         f'<span class="vt-badge">{e(x["status"])}</span><div>{e(x["title"])}</div>'
                         f'<div class="vt-muted">{links}</div></div>')
        out.append(f'<h3>{e(title)} ({len(items)})</h3><div class="vt-findings">{"".join(cards)}</div>')
    return "".join(out)


def _precision(nodes):
    """One dot per check with a numeric agreement in (0, 1), on a log axis."""
    pts = []
    for n in nodes.values():
        for c in n["checks"]:
            v = c["value"]
            if c["kind"] in ("check",) and c["outcome"] in GOOD and isinstance(v, (int, float)) and 0 < abs(v) < 1:
                pts.append((math.log10(abs(v)), n["layer"], n["label"], c))
    if not pts:
        return ""
    lo, hi = -16.5, 0
    rows = [layer for layer in layers(nodes) if any(p[1] == layer for p in pts)]
    h = 24 * len(rows) + 30
    out = [f'<svg class="vt-prec" viewBox="0 0 640 {h}" role="img" aria-label="agreement of each check, log scale">']
    for k in range(-16, 1, 2):
        x = 90 + 540 * (k - lo) / (hi - lo)
        out.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{h - 22}" class="vt-grid"/><text x="{x:.1f}" y="{h - 6}" class="vt-tick" text-anchor="middle">{"1" if k == 0 else f"1e{k}"}</text>')
    for i, layer in enumerate(rows):
        y = 14 + 24 * i
        out.append(f'<text x="0" y="{y + 4}" class="vt-tick">{e(layer)}</text>')
        for lv, lay, label, c in pts:
            if lay != layer:
                continue
            x = 90 + 540 * (max(lv, lo) - lo) / (hi - lo)
            out.append(f'<circle cx="{x:.1f}" cy="{y}" r="3.5" class="vt-dot"><title>{e(label)}: {e(c["headline"])} = {_value(c["value"])} ({e(c["name"])})</title></circle>')
    out.append("</svg>")
    return "".join(out)


def _tables(model):
    lines = ["| part | verdict | independent roots (needed) | virgil changed since | checks |", "| --- | --- | --- | --- | --- |"]
    for n in model["nodes"].values():
        changed = "unknown" if n["changed"] is None else (", ".join(f"`{f}`" for f in n["changed"]) or "no")
        roots = ", ".join(ROOT_NAMES.get(r, r) for r in n["strong"]) or "—"
        lines.append(f'| `{n["id"]}` | {VERDICTS[n["verdict"]][0]} | {roots} ({n["need"]}) | {changed} | {len(n["checks"])} |')
    return "\n".join(lines)


def render(model):
    nodes = model["nodes"]
    total = len(nodes)
    counts = model["counts"]
    commits = sorted({r["virgil"][:7] for r in model["runs"] if r.get("virgil")})
    ours = [r for r in model["runs"] if r.get("runner") != "virgil-ci"]
    pin = (ours[0]["virgil"] or "")[:7] if ours and ours[0].get("virgil") else ", ".join(commits)
    fresh = ("" if model["changed"] is None else
             f'<i aria-hidden="true">⏲</i> virgil has changed {model["changed"]} of these parts since; the weekly run refreshes the evidence')
    real = [p for p in model["pipelines"] if p["data"] == "real"]
    sim = [p for p in model["pipelines"] if p["data"] != "real"]
    slim = {k: {f: n[f] for f in ("id", "verdict", "strong", "weak", "findings", "because", "depends", "dependents", "changed")}
                | {"checks": [{f: c[f] for f in ("name", "doc", "outcome", "kind", "headline", "value", "url")} for c in n["checks"]]}
            for k, n in nodes.items()}
    data = json.dumps({"nodes": slim, "verdicts": {k: list(v) for k, v in VERDICTS.items()}, "roots": ROOT_NAMES},
                      separators=(",", ":"), default=str).replace("</", "<\\/")
    parts = [
        "---\nhide:\n  - toc\n---\n<!-- Generated by scripts/trust.py from trust/graph.yml, trust/ledger.yml and the evidence. Do not edit. -->",
        "# Can virgil be trusted?\n",
        "virgil's calculations, checked part by part against mathematics, published standards and "
        "independent packages written by other people, then whole chains end to end. "
        "[How this works](design.md).\n",
        '<div class="vt">',
        f'<div class="vt-head"><div><span class="vt-big">{counts.get("verified", 0)}</span> of {total} parts verified '
        f'<span class="vt-muted">at virgil <code>{e(pin)}</code></span></div><div class="vt-muted">{fresh}</div></div>',
        _bar(counts, total),
        _legend(counts),
        '<h2 id="real-data">On real data: does virgil reproduce published results?</h2>',
        '<div class="vt-cards">' + ("".join(_pipeline_card(p, nodes) for p in real) or '<div class="vt-muted">None yet.</div>') + "</div>",
        '<h2 id="simulated-data">On simulated data: whole chains, from file to answer</h2>',
        '<div class="vt-chains">' + "".join(_chain(p, nodes) for p in sim) + "</div>",
        '<h2 id="parts">Every part, from data up</h2>',
        '<p class="vt-muted">Each chip is one part of virgil. Select it to see how it was checked; '
        "a dot marks parts virgil has changed since their evidence was measured.</p>",
        _map(nodes),
        '<h2 id="roots">Who checks what</h2>',
        '<p class="vt-muted">Independent roots behind each part: trust does not rest on one source.</p>',
        _matrix(nodes),
        '<h2 id="ledger">What the checks found</h2>',
        _ledger(model["ledger"]),
        '<h2 id="precision">How precise</h2>',
        '<p class="vt-muted">Each dot is one check’s agreement with its reference (hover for which). '
        "Most agree to machine precision; the looser ones are noisy simulations or deliberate approximations.</p>",
        _precision(nodes),
        "</div>",
        f'<script type="application/json" id="vt-data">{data}</script>\n',
        "## All parts\n",
        "<details><summary>Table of every part, its verdict and evidence</summary>\n",
        _tables(model) + "\n",
        "</details>\n",
        "Evidence: " + "; ".join(f'{e(r.get("runner"))} on {e(r.get("date"))}, virgil `{e((r.get("virgil") or "unknown")[:10])}`'
                                 + (f' ([run]({r["run_url"]}))' if r.get("run_url") else "") for r in model["runs"]) + ".\n",
    ]
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", action="append", required=True)
    ap.add_argument("--graph", default=ROOT / "trust" / "graph.yml")
    ap.add_argument("--ledger", default=ROOT / "trust" / "ledger.yml")
    ap.add_argument("--virgil", type=pathlib.Path, default=None)
    ap.add_argument("--out", default=ROOT / "docs" / "trust.md")
    ap.add_argument("--index", default=None, help="also write the page here (the site's home page)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    graph = yaml.safe_load(open(args.graph))
    ledger = yaml.safe_load(open(args.ledger))
    runs, records = load_evidence(args.evidence)
    virgil = args.virgil.expanduser() if args.virgil else None
    model = build(graph, ledger, records, runs, virgil)
    page = render(model)
    pathlib.Path(args.out).write_text(page)
    if args.index:
        pathlib.Path(args.index).write_text(page)
    if args.json:
        pathlib.Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.json).write_text(json.dumps(model, indent=1, default=str))
    print(dict(collections.Counter(n["verdict"] for n in model["nodes"].values())),
          {p["id"]: p["status"] for p in model["pipelines"]})


if __name__ == "__main__":
    main()
