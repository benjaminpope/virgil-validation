"""Join the trust graph, the evidence and virgil's history into the Trust page.

    python scripts/trust.py --evidence trust/evidence/latest.jsonl \\
        [--evidence trust/evidence/virgil.jsonl] [--virgil ~/code/drpangloss] \\
        [--out docs/index.md] [--json docs/assets/trust.json]
        [--findings docs/method/findings.md] [--coverage docs/method/coverage.md]

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
GOOD = {"passed", "xfailed"}  # xfailed is good only for finding/upstream records, which never count as agreement
# Kinds that count as agreement: a check of virgil; for our reference nodes,
# also the checks of our own reference code. Guards, controls, regressions
# and definition differences never verify anything.
AGREEING = {"check"}
AGREEING_FOR_REFERENCES = {"check", "reference"}
VIRGIL_CI = "virgil's own tests"  # weak: written inside virgil, by authors who can read its implementation
GOLDEN = "golden"  # every registered golden source together counts as this one root
LAYER_ORDER = ["imaging", "inference", "orbits", "likelihood", "priors", "models", "data", "references"]  # top first

VERDICTS = {
    "verified": ("Verified", "independent checks agree, and so does everything it relies on"),
    "relies": ("Works, but relies on a known bug", "its own checks pass, but it uses a part with an unfixed bug"),
    "relies-unverified": ("Works, but relies on an unverified part", "its own checks pass, but it uses a part not yet verified"),
    "bug": ("Known bug, fix pending", "the checks found a mistake in virgil that is not fixed yet"),
    "partly": ("Partly checked", "it needs another independent check"),
    "convention": ("Convention from virgil's docs", "its checks agree, but a convention they rest on is taken from "
                   "virgil's own documentation and is not yet confirmed by another code, a paper or a sign-off"),
    "failing": ("Check failing", "a check of it fails"),
    "unchecked": ("Not yet checked", "no independent check yet"),
}
ROOT_NAMES = {
    "mathematics": "mathematics", "standards": "standards", "statistics": "statistics",
    "literature": "published result", "render": "independent render", "dlux": "dLux", "pmoired": "PMOIRED", "candid": "CANDID",
    "fouriever": "fouriever", "ehtim": "eht-imaging", "mpol": "MPoL", "orbitize": "orbitize!",
    "self-consistency": "virgil itself",
}
# metric names that state a difference or error (raw statistics such as a
# reduced chi-squared need an explicit headline=)
AGREEMENT = re.compile(r"(^|_)(rel|abs|diff|difference|err|error|dv|dv2|dsigma|dloglike|dlogb)(_|$)", re.I)


# ------------------------------------------------------------------ inputs


def provenance_problem(run, repo=ROOT):
    """Why a run's evidence cannot be published, or None: it ran on
    uncommitted code here or in virgil, or on a commit of this repository
    that is not in the history of the one building the page."""
    if run.get("validation_dirty"):
        return "ran on uncommitted changes to virgil-validation"
    if (run.get("virgil") or {}).get("dirty"):
        return "ran on uncommitted changes to virgil"
    commit = run.get("validation_commit")
    if not commit:
        return "does not say which commit of virgil-validation it ran on"
    try:
        ok = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", commit, "HEAD"],
                            capture_output=True).returncode == 0
    except OSError:
        return None
    return None if ok else f"ran on {commit[:7]}, which is not in this branch's history"


def load_evidence(paths, strict=False):
    """Runs and test records. A test that appears in several runs keeps only
    its newest outcome, so an old pass cannot outlive a newer skip or failure;
    the exception is a campaign summary (files named campaigns*.jsonl), which
    an ordinary run skips because it ran on another virgil commit.

    With ``strict``, a run with a provenance problem is left out entirely."""
    runs, by_test = [], {}
    for path in paths:
        lines = [json.loads(line) for line in open(path)]
        run, tests = lines[0], lines[1:]
        if strict and (why := provenance_problem(run)):
            print(f"left out {path}: {why}")
            continue
        runs.append(run)
        campaign = pathlib.Path(path).name.startswith("campaigns")
        for t in tests:
            t["_commit"] = (run.get("virgil") or {}).get("commit")
            t["_source"] = t.get("source", "ours")
            t["_date"] = run.get("date") or ""
            t["_campaign"] = campaign
            t["_validation_commit"] = run.get("validation_commit")
            by_test.setdefault(t["test"], []).append(t)
    records = []
    for recs in by_test.values():
        recs.sort(key=lambda r: r["_date"])
        newest = recs[-1]
        if newest["outcome"] == "skipped":
            ran = [r for r in recs if r["_campaign"] and r["outcome"] != "skipped"]
            newest = ran[-1] if ran else newest
        records.append(newest)
    return runs, records


def load_golden(path=ROOT / "trust" / "golden.yml"):
    return set(yaml.safe_load(open(path)) or {}) if pathlib.Path(path).exists() else set()


def roots_of(record, golden):
    """(strong, weak) roots a passing record contributes."""
    if record["_source"] != "ours":
        return set(), {VIRGIL_CI}
    if record["kind"] == "regression":
        return set(), {"regression"}
    strong, weak = set(), set()
    for root in record["roots"]:
        if root in WEAK:
            weak.add(root)
        elif root.startswith("golden:"):
            if root.split(":", 1)[1] in golden:
                strong.add(GOLDEN)
        else:
            strong.add(root)
    return strong, weak


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
        # the test as it was when the evidence was measured
        "url": f"{REPO}/blob/{record.get('_validation_commit') or 'main'}/{path}" + (f"#L{line}" if line else "")
        if record["_source"] == "ours" else None,
        "source": record["_source"],
    }


# Reference nodes (our code and other packages) and the page that describes each.
REFERENCE_DOCS = {
    "crosscheck": "method/index.md", "crosscheck.sky": "models/visibilities.md",
    "crosscheck.limb": "models/limb_darkening.md", "crosscheck.elr": "models/rapid_rotators.md",
    "crosscheck.nrm": "data/masking.md", "external_bridge": "method/index.md", "evidence": "method/index.md",
    "external_bridge.pmoired_models": "method/pmoired.md", "pmoired": "method/pmoired.md",
    "orbitize": "method/orbitize.md", "external_bridge.orbitize_bridge": "method/orbitize.md",
    "candid": "method/candid.md", "external_bridge.candid_bridge": "method/candid.md",
    "external_bridge.fouriever_worker": "method/fouriever.md", "fouriever": "method/fouriever.md",
    "ehtim": "imaging/ehtim.md",
    "crosscheck.array": "data/long_baseline.md", "crosscheck.chi2": "method/index.md",
    "crosscheck.oifits_writer": "data/oifits_observables.md", "crosscheck.simulate": "data/long_baseline.md",
    "crosscheck.orbits": "orbits/kepler.md", "crosscheck.disks": "models/flared_disks.md",
}
# Pages that are not about one topic: never the explanation of a part.
NOT_TOPIC_PAGES = {"index.md", "trust.md", "evidence.md", "results.md"}


def slugify(text):
    """The heading anchor Zensical (Python-Markdown's toc) gives a heading."""
    import unicodedata

    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "-", text)


def nav_pages(mkdocs=ROOT / "mkdocs.yml"):
    """The site's pages in navigation order (docs-relative paths)."""
    out = []

    def walk(items):
        for item in items:
            for value in (item.values() if isinstance(item, dict) else [item]):
                if isinstance(value, list):
                    walk(value)
                elif isinstance(value, str) and value.endswith(".md"):
                    out.append(value)

    walk(yaml.load(open(mkdocs), Loader=yaml.BaseLoader)["nav"])
    return out


def page_sections(path, docs=ROOT / "docs"):
    """[(anchor, text)] for each heading of a page, the text running to the next heading."""
    sections, anchor, buf = [], "", []
    for line in (docs / path).read_text().splitlines():
        m = re.match(r"^(#{1,4}) (.+?)\s*$", line)
        if m:
            sections.append((anchor, "\n".join(buf)))
            anchor, buf = ("" if m.group(1) == "#" else slugify(m.group(2))), []
        else:
            buf.append(line)
    sections.append((anchor, "\n".join(buf)))
    return sections


def _mentions(text, dotted, last):
    """Backticked mentions of a part: `models.X`, `X`, `X(...)`, `virgil.models.X`, `obj.X`."""
    n = 0
    for tok in re.findall(r"`([^`]+)`", text):
        tok = tok.strip()
        if tok in (dotted, last, "virgil." + dotted) or tok.endswith("." + last) or tok.startswith(last + "("):
            n += 1
    return n


def resolve_docs(graph, docs=ROOT / "docs", mkdocs=ROOT / "mkdocs.yml"):
    """Each node's page and section: an explicit `doc:` in the graph, the
    reference pages above, or the topic page that mentions the part most
    (its section with the first mention). {node: "path.md#anchor"}."""
    pages = [p for p in nav_pages(mkdocs) if p not in NOT_TOPIC_PAGES and (docs / p).exists()]
    sections = {p: page_sections(p, docs) for p in pages}
    out = {}
    for name, node in graph["nodes"].items():
        if node.get("doc"):
            out[name] = node["doc"]
            continue
        if name in REFERENCE_DOCS:
            out[name] = REFERENCE_DOCS[name]
            continue
        dotted = name.removeprefix("virgil.")
        last = dotted.rsplit(".", 1)[-1]
        best = None
        # the topic sections first; the method pages only for parts no topic covers
        for tier in ([p for p in pages if not p.startswith("method/")], [p for p in pages if p.startswith("method/")]):
            for p in tier:
                counts = [(a, _mentions(t, dotted, last)) for a, t in sections[p]]
                total = sum(c for _, c in counts)
                if total and (best is None or total > best[0]):
                    best = (total, p, next(a for a, c in counts if c))
            if best:
                break
        if best:
            out[name] = best[1] + (f"#{best[2]}" if best[2] else "")
    return out


def doc_url(doc):
    """docs-relative "path.md#anchor" as the site's directory URL."""
    path, _, anchor = doc.partition("#")
    url = path[:-len("index.md")] if path.endswith("index.md") else path[:-3] + "/"
    return url + (f"#{anchor}" if anchor else "")


UNFIXED = {"open", "to-raise", "raised"}
# Roots that can confirm a convention taken from virgil's documentation: code
# or results written by other people, or an image we render from the physical
# geometry (what the convention means on the sky) rather than from virgil's
# formula. Mathematics and standards are our own transcriptions of the
# documented formula, so they cannot.
EXTERNAL = {"pmoired", "candid", "fouriever", "orbitize", "ehtim", "mpol", "dlux", "literature", "render"}


def load_signoffs(path=ROOT / "trust" / "signoffs.yml"):
    """{node: {property: entry}}: conventions confirmed by a cited paper or Ben's sign-off."""
    return (yaml.safe_load(open(path)) or {}) if pathlib.Path(path).exists() else {}


def build(graph, ledger, records, runs, virgil=None, golden=None, signoffs=None):
    nodes = graph["nodes"]
    golden = load_golden() if golden is None else golden
    signoffs = load_signoffs() if signoffs is None else signoffs
    by_obj = collections.defaultdict(list)
    for r in records:
        for obj in r["objects"]:
            by_obj[obj].append(r)
    open_findings = collections.defaultdict(list)  # unfixed bugs in virgil
    external_open = collections.defaultdict(list)  # unfixed problems in another package
    for e in ledger:
        if e["status"] not in UNFIXED:
            continue
        if e["ruling"] == "virgil":
            for obj in e["objects"]:
                open_findings[obj].append(e["id"])
        elif e["ruling"].startswith("external"):
            for obj in e["objects"]:
                external_open[obj].append(e["id"])

    own = {}

    def own_status(name, trusted_references=None):
        """A part's own verdict from its own checks. For parts of virgil,
        a check that went through our reference code (its ``via``) counts
        only if every such reference is in ``trusted_references``; checks
        blocked that way are returned so the part can say what it relies on."""
        node = nodes[name]
        recs = by_obj.get(name, [])
        reference_node = node.get("layer", "references") == "references"
        agreeing = AGREEING_FOR_REFERENCES if reference_node else AGREEING
        counted = [r for r in recs if r["kind"] in agreeing or r["kind"] == "regression"]
        failing = [r for r in recs if r["kind"] not in ("finding", "upstream") and r["outcome"] in BAD]
        good = [r for r in counted if r["outcome"] == "passed"]
        strong, weak, blocked_roots, blocked_by = set(), set(), set(), set()
        prop_roots = collections.defaultdict(set)
        for r in good:
            s_, w_ = roots_of(r, golden)
            untrusted = set()
            if trusted_references is not None and r["kind"] in agreeing:
                untrusted = {v for v in r.get("via", []) if v not in trusted_references}
            if r["kind"] in agreeing and not untrusted:
                strong |= s_
                for prop in r.get("properties", []):
                    prop_roots[prop] |= s_
            elif r["kind"] in agreeing:
                blocked_roots |= s_
                blocked_by |= untrusted
            weak |= w_ | (s_ if r["kind"] not in agreeing else set())
        strong, weak = sorted(strong), sorted(weak - set(strong))
        need = node.get("roots", 1)
        if failing:
            status = "failing"
        elif name in open_findings:
            status = "bug"
        elif len(strong) >= need and not external_open.get(name) and properties_short(name, prop_roots):
            status = "partly"  # agreement overall, but a property needs more independent roots
        elif len(strong) >= need and not external_open.get(name) and unconfirmed_conventions(name, prop_roots):
            status = "convention"
        elif len(strong) >= need and not external_open.get(name):
            status = "ok"
        elif len(set(strong) | blocked_roots) >= need and not external_open.get(name):
            status = "ok-via"  # its checks pass, but through reference code not yet verified
        elif strong or weak or blocked_roots:
            status = "partly"
        else:
            status = "unchecked"
        return status, strong, weak, need, recs, good, sorted(blocked_by), prop_roots

    def properties_short(name, prop_roots):
        """Properties of a part with fewer independent roots than it needs."""
        return sorted(p for p, n in (nodes[name].get("properties") or {}).items() if len(prop_roots.get(p, ())) < n)

    def unconfirmed_conventions(name, prop_roots):
        """Conventions taken from virgil's docs with no external root and no sign-off."""
        signed = signoffs.get(name) or {}
        return sorted(p for p in nodes[name].get("convention") or []
                      if not (prop_roots.get(p, set()) & EXTERNAL) and p not in signed)

    def reference_verdicts():
        """Our reference nodes, decided first: verified only if their own
        checks pass and every reference they rely on is verified too."""
        refs = [n for n, node in nodes.items() if node.get("layer", "references") == "references"]
        status = {n: own_status(n)[0] for n in refs}

        def ok(n, seen):
            if status.get(n) != "ok":
                return False
            return all(d not in status or (d in seen or ok(d, seen | {d})) for d in nodes[n].get("depends", []))

        return {n for n in refs if ok(n, {n})}

    trusted = reference_verdicts()
    for name, node in nodes.items():
        reference_node = node.get("layer", "references") == "references"
        status, strong, weak, need, recs, good, blocked_by, prop_roots = own_status(
            name, None if reference_node else trusted)
        agreeing = AGREEING_FOR_REFERENCES if reference_node else AGREEING
        commits = sorted({r["_commit"] for r in good if r["_commit"]})
        changes = [changed_files(virgil, c, node.get("source", [])) for c in commits]
        changed = None if not changes or any(c is None for c in changes) else sorted({f for c in changes for f in c})
        own[name] = {
            "status": status, "strong": strong, "weak": weak, "need": need,
            "checks": [_check(r) for r in recs], "commits": commits, "changed": changed,
            "findings": open_findings.get(name, []) + external_open.get(name, []),
            "skipped": sum(1 for r in recs if r["outcome"] == "skipped" and r["kind"] in agreeing),
            "via_blocked": blocked_by,
            "properties": {p: {"need": n, "roots": sorted(prop_roots.get(p, ()))}
                           for p, n in (node.get("properties") or {}).items()},
            "unconfirmed": unconfirmed_conventions(name, prop_roots),
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

    def changed_below(name, seen):
        """virgil files changed under this part or anything it relies on."""
        found = set(own[name]["changed"] or [])
        for dep in nodes[name].get("depends", []):
            if dep in nodes and dep not in seen:
                seen.add(dep)
                found |= changed_below(dep, seen)
        return found

    out_nodes = {}
    for name, node in nodes.items():
        o = own[name]
        verdict = o["status"]
        causes = sorted(problems(name, set()) | set(o["via_blocked"]))
        if verdict == "ok-via":
            verdict = "ok"
        if verdict == "ok":
            if not causes:
                verdict = "verified"
            elif any(own.get(c, {}).get("status") == "bug" for c in causes):
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
            "because": [{"id": c, "findings": own.get(c, {}).get("findings", []),
                         "status": own.get(c, {}).get("status", "unchecked")} for c in causes],
            "depends": node.get("depends", []), "dependents": dependents,
            "properties": o["properties"], "unconfirmed": o["unconfirmed"],
            "checks": o["checks"], "commits": o["commits"],
            "changed": o["changed"] if o["changed"] is None else sorted(changed_below(name, set())),
            "skipped": o["skipped"],
            "source": node.get("source", []),
        }

    pipelines = []
    for name, p in graph.get("pipelines", {}).items():
        recs = [r for r in by_obj.get(f"pipeline:{name}", []) if r["kind"] in AGREEING]
        bad = [r for r in recs if r["outcome"] in BAD]
        good = [r for r in recs if r["outcome"] == "passed"]
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

ICON = {"verified": "✓", "relies": "↧", "relies-unverified": "↧", "bug": "!", "partly": "◐", "convention": "◑",
        "failing": "✗", "unchecked": "–"}
ORDER = ["verified", "relies", "relies-unverified", "convention", "partly", "bug", "failing", "unchecked"]
COLOURS = {  # Okabe–Ito based, readable on light and dark
    "verified": "#2e9e5b", "relies": "#e0a526", "relies-unverified": "#c9b458", "partly": "#56b4e9",
    "convention": "#cc79a7",
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


FINDING_GROUPS = [
    ("Bugs found in virgil", lambda x: x["ruling"] == "virgil"),
    ("Where codes define things differently", lambda x: x["ruling"] == "definition"),
    ("Problems found in other packages", lambda x: x["ruling"].startswith("external")),
    ("Mistakes in our own checks", lambda x: x["ruling"] == "crosscheck"),
]


def render_findings(ledger, extra=ROOT / "trust" / "definition_changes.md"):
    """The Findings page: every ledger entry, grouped as on the Trust page."""
    def cell(text):
        return str(text).replace("|", "\\|").replace("\n", " ")

    def short(obj):
        return obj.removeprefix("virgil.")

    out = ["<!-- Generated by scripts/trust.py from trust/ledger.yml. Do not edit. -->",
           "# Findings\n",
           "Every disagreement the checks have found, how it was ruled, and where it "
           "stands. The same list, with the parts each affects, is on the "
           "[Trust page](../index.md#ledger).\n"]
    for title, keep in FINDING_GROUPS:
        items = [x for x in ledger if keep(x)]
        if not items:
            continue
        out += [f"## {title}\n", "| | Part | Finding | Status |", "| --- | --- | --- | --- |"]
        for x in items:
            links = " ".join(f"[{_link_label(u)}]({u})" for u in x.get("links", []))
            parts = ", ".join(f"`{short(o)}`" for o in x.get("objects", []))
            out.append(f'| <span id="{x["id"]}">{x["id"]}</span> | {cell(parts)} | {cell(x["title"])} | '
                       f'{cell(x["status"])}{" " + links if links else ""} |')
        out.append("")
    if extra and pathlib.Path(extra).exists():
        out.append(pathlib.Path(extra).read_text())
    return "\n".join(out) + "\n"


def render_coverage(model, extra=ROOT / "trust" / "not_covered.md"):
    """The Not yet covered page: every part of virgil whose verdict is not
    verified, with the reason, then what is outside the graph."""
    out = ["<!-- Generated by scripts/trust.py from trust/graph.yml and the evidence. Do not edit. -->",
           "# Not yet covered\n",
           "Parts of virgil that are not yet verified, from the same rules as the "
           "[Trust page](../index.md).\n"]
    nodes = [n for n in model["nodes"].values() if n["id"].startswith("virgil.") and n["verdict"] != "verified"]
    docs = model.get("docs", {})
    for verdict in VERDICTS:
        group = sorted((n for n in nodes if n["verdict"] == verdict), key=lambda n: n["id"])
        if not group:
            continue
        out += [f"## {VERDICTS[verdict][0]}\n", "| Part | Why |", "| --- | --- |"]
        for n in group:
            why = []
            if n["findings"]:
                why.append("open finding " + ", ".join(f"[{f}](findings.md#{f})" for f in n["findings"]))
            if n["because"]:
                why.append("relies on " + ", ".join(f"`{b['id'].removeprefix('virgil.')}`" for b in n["because"]))
            if verdict == "unchecked" and not why:
                why.append("no check yet")
            if verdict == "convention":
                why.append("convention " + ", ".join(f"`{p}`" for p in n["unconfirmed"]) + " from virgil's docs only")
            short = [p for p, v in n["properties"].items() if len(v["roots"]) < v["need"]]
            if verdict == "partly" and short:
                why.append("; ".join(f"`{p}`: {len(n['properties'][p]['roots'])} of {n['properties'][p]['need']} "
                                     "independent references" for p in short))
            elif verdict == "partly" and not why:
                why.append("only checked against itself" if not n["strong"]
                           else f"{len(n['strong'])} of {n['need']} independent references")
            name = n["id"].removeprefix("virgil.")
            doc = docs.get(n["id"])
            label = f"[`{name}`](../{doc})" if doc and not doc.startswith("method/coverage") else f"`{name}`"
            out.append(f"| {label} | {'; '.join(why) or VERDICTS[verdict][1]} |")
        out.append("")
    if extra and pathlib.Path(extra).exists():
        out.append(pathlib.Path(extra).read_text())
    return "\n".join(out) + "\n"


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


def _headline_html(verified, total, stale, commits, refs_ok, refs_total):
    """The headline: parts of virgil only, staleness in the count itself."""
    since = f", {stale} of them since changed in virgil" if stale else ""
    where = ", ".join(f"<code>{e(c)}</code>" for c in commits)
    return (f'<div class="vt-head"><div><span class="vt-big">{verified}</span> of {total} parts of virgil verified{since} '
            f'<span class="vt-muted">(evidence from virgil {where})</span></div>'
            f'<div class="vt-muted">Our own reference code and the packages we compare with: '
            f'{refs_ok} of {refs_total} verified.</div></div>')


def render(model, base=""):
    nodes = model["nodes"]
    ours_nodes = [n for n in nodes.values() if n["id"].startswith("virgil.")]
    total = len(ours_nodes)
    counts = collections.Counter(n["verdict"] for n in ours_nodes)
    stale = sum(1 for n in ours_nodes if n["verdict"] == "verified" and n["changed"])
    refs = [n for n in nodes.values() if not n["id"].startswith("virgil.")]
    refs_ok = sum(1 for n in refs if n["verdict"] == "verified")
    commits = sorted({r["virgil"][:7] for r in model["runs"] if r.get("virgil")})
    ours = [r for r in model["runs"] if r.get("runner") != "virgil-ci"]
    pin = (ours[0]["virgil"] or "")[:7] if ours and ours[0].get("virgil") else ", ".join(commits)
    fresh = ("" if model["changed"] is None else
             f'<i aria-hidden="true">⏲</i> virgil has changed {model["changed"]} of these parts since; the weekly run refreshes the evidence')
    real = [p for p in model["pipelines"] if p["data"] == "real"]
    sim = [p for p in model["pipelines"] if p["data"] != "real"]
    slim = {k: {f: n[f] for f in ("id", "verdict", "strong", "weak", "findings", "because", "depends", "dependents", "changed")}
                | {"doc": base + doc_url(model["docs"][k]) if model.get("docs", {}).get(k) else None}
                | {"checks": [{f: c[f] for f in ("name", "doc", "outcome", "kind", "headline", "value", "url")} for c in n["checks"]]}
            for k, n in nodes.items()}
    data = json.dumps({"nodes": slim, "verdicts": {k: list(v) for k, v in VERDICTS.items()}, "roots": ROOT_NAMES},
                      separators=(",", ":"), default=str).replace("</", "<\\/")
    parts = [
        "---\nhide:\n  - toc\n---\n<!-- Generated by scripts/trust.py from trust/graph.yml, trust/ledger.yml and the evidence. Do not edit. -->",
        "# Can virgil be trusted?\n",
        "> *ma però che già mai di questo fondo<br>\n"
        "> non tornò vivo alcun, s’i’ odo il vero,<br>\n"
        "> sanza tema d’infamia ti rispondo.*\n>\n"
        "> — Dante, *Inferno*, Canto XXVII, lines 64–66\n",
        "virgil's calculations, checked part by part against mathematics, published standards and "
        "independent packages written by other people, then whole chains end to end. "
        "[How this works](method/index.md).\n",
        '<div class="vt">',
        _headline_html(counts.get("verified", 0), total, stale, commits, refs_ok, len(refs)),
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
    ap.add_argument("--strict-provenance", action="store_true",
                    help="leave out runs on uncommitted code or on commits outside this branch (the published page)")
    ap.add_argument("--out", default=ROOT / "docs" / "index.md")
    ap.add_argument("--index", default=None, help="also write the page here (the site's home page)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--findings", default=None, help="also write the Findings page here (docs/method/findings.md)")
    ap.add_argument("--coverage", default=None, help="also write the Not yet covered page here (docs/method/coverage.md)")
    args = ap.parse_args()
    graph = yaml.safe_load(open(args.graph))
    ledger = yaml.safe_load(open(args.ledger))
    runs, records = load_evidence(args.evidence, strict=args.strict_provenance)
    virgil = args.virgil.expanduser() if args.virgil else None
    model = build(graph, ledger, records, runs, virgil)
    model["docs"] = resolve_docs(graph)
    out, index = pathlib.Path(args.out), pathlib.Path(args.index) if args.index else None
    docs_root = ROOT / "docs"
    for path in [out] + ([index] if index else []):
        # links in the embedded data are resolved by the browser from the page's own URL
        depth = 0 if path.name == "index.md" and path.parent.resolve() == docs_root.resolve() else 1
        path.write_text(render(model, "../" * depth))
    if args.findings:
        pathlib.Path(args.findings).write_text(render_findings(ledger))
    if args.coverage:
        pathlib.Path(args.coverage).write_text(render_coverage(model))
    if args.json:
        pathlib.Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.json).write_text(json.dumps(model, indent=1, default=str))
    print(dict(collections.Counter(n["verdict"] for n in model["nodes"].values())),
          {p["id"]: p["status"] for p in model["pipelines"]})


if __name__ == "__main__":
    main()
