"""The rendering half of the Trust page: the model that ``scripts/trust.py`` builds, as HTML and Markdown.

``trust.py`` holds the verdict rules and the evidence and documentation lookups they use; this module only
draws what they decided (the map, the matrix, the pipeline cards, the ledger, and the Findings, Not yet covered
and Function tests pages). Every name here is also importable from ``trust``.

The few constants and helpers of ``trust.py`` that rendering reads (``VERDICTS``, ``ROOT_NAMES``, ``ROOT_ALIAS``, ``GOOD``,
``LAYER_ORDER``, ``UNFIXED``, ``slugify``, ``doc_url``) are handed over by ``trust_render.attach``, which ``trust.py``
calls once its own definitions are in place. That keeps one copy of them, whether ``trust.py`` was run as a
script or loaded by file path, and means this module never imports ``trust`` back.

The names are copied once, so a test that replaces one for rendering (``SHOWCASE``, ``VERDICTS``, ...) must patch it
here, as ``trust_render.X``; patching ``trust.X`` reaches only ``trust.py`` and the ``trust._showcase`` wrapper.
"""

import collections
import html
import json
import math
import pathlib
import re
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]

# set by attach() from trust.py
GOOD = LAYER_ORDER = ROOT_ALIAS = ROOT_NAMES = UNFIXED = VERDICTS = doc_url = slugify = None


def attach(namespace):
    """Take the constants and helpers of ``trust.py`` that rendering reads from its namespace."""
    for name in ("GOOD", "LAYER_ORDER", "ROOT_ALIAS", "ROOT_NAMES", "UNFIXED", "VERDICTS", "doc_url", "slugify"):
        globals()[name] = namespace[name]


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
             "planned": "planned", "running": "running",
             "not run": f'not run (all {len(p["checks"])} check{"s" if len(p["checks"]) != 1 else ""} skipped)'}.get(p["status"], p["status"])
    ref = e(p["reference"]) if p["reference"] else ""
    if p.get("url") and ref:
        ref = f'<a href="{e(p["url"])}">{ref}</a>'
    title = e(p["title"])
    if p.get("page"):
        title = f'<a class="vt-card-link" href="{e(p["page"])}">{title}</a>'
    head = ""
    good = [c for c in p["checks"] if c["outcome"] in GOOD and c["value"] is not None]
    if good:
        head = f'{e(good[0]["headline"])} = {_value(good[0]["value"])}'
    elif p.get("headline"):
        head = e(p["headline"])
    return (
        f'<div class="vt-card vt-p-{e(p["status"].replace(" ", "-"))}"><div class="vt-card-t">{title}</div>'
        f'<div class="vt-muted">{e(p["dataset"] or "")}{" · " + ref if ref else ""}</div>'
        f'<div class="vt-card-h">{head}</div><span class="vt-badge">{e(badge)}</span>{_fails_badge(p)}</div>'
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
        f'<span class="vt-steps">{" › ".join(steps)}</span><span class="vt-chain-r">{result}{_fails_badge(p)}</span></div>'
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


def _fails_badge(item):
    """A badge on a part or pipeline whose passing record hides a failure on another virgil commit."""
    tests = item.get("fails_elsewhere")
    if not tests:
        return ""
    return (f'<span class="vt-failelse" title="{len(tests)} check(s) of this fail on another virgil commit '
            f'than the passing record shown: see the notice at the top of the page">⚠ fails on another virgil commit</span>')


def _map(nodes, base=""):
    rows = []
    for layer in layers(nodes):
        chips = [n for n in nodes.values() if n["layer"] == layer]
        if not chips:
            continue
        cells = "".join(
            f'<button type="button" class="vt-chip vt-{n["verdict"]}" data-node="{e(n["id"])}" title="{e(_tooltip(n))}">'
            f'<span aria-hidden="true">{ICON[n["verdict"]]}</span> {e(n["label"])}'
            + ('<span class="vt-fresh" title="virgil has changed this since">●</span>' if n["changed"] else "")
            + _fails_badge(n)
            + "</button>"
            for n in chips
        )
        label = e(layer)
        if layer in LAYER_SECTION:
            label = f'<a href="{base}tests/#{slugify(LAYER_SECTION[layer])}" title="The {e(layer)} tests">{label}</a>'
        rows.append(f'<div class="vt-row"><div class="vt-layer">{label}</div><div class="vt-chips">{cells}</div></div>')
    return '<div class="vt-map">' + "".join(rows) + '</div><div class="vt-detail" id="vt-detail" aria-live="polite">Select a part to see what it was checked against.</div>'


def _skyline(model, base="", roots=ROOT / "trust" / "roots.yml"):
    """Every part as a square by layer and depth, on the foundation of root classes (scripts/trust_skyline.py)."""
    import trust_skyline  # scripts/ is on sys.path once trust.py is loaded

    return trust_skyline.skyline_html(model, yaml.safe_load(open(roots)), base, names=sys.modules[__name__])


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
        unfixed = sum(1 for x in items if x["status"] in UNFIXED)
        state = f"{unfixed} still open" if unfixed else "all resolved"
        out.append(f'<details class="vt-fold"><summary><b>{e(title)} ({len(items)})</b> '
                   f'<span class="vt-muted">· {state}</span></summary><div class="vt-findings">{"".join(cards)}</div></details>')
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


# The function-test section of the site that explains each layer of the map.
LAYER_SECTION = {"data": "Data", "models": "Models", "likelihood": "Likelihood", "priors": "Likelihood",
                 "inference": "Inference", "orbits": "Orbits", "imaging": "Imaging"}


def test_sections(mkdocs=ROOT / "mkdocs.yml"):
    """[(section, [(title, path)])] from the "Function tests" part of the nav."""
    for item in yaml.load(open(mkdocs), Loader=yaml.BaseLoader)["nav"]:
        if isinstance(item, dict) and "Function tests" in item:
            return [(name, [(t, path) for page in pages for t, path in page.items()])
                    for sub in item["Function tests"] if isinstance(sub, dict) for name, pages in sub.items()]
    raise ValueError(f'no "Function tests" section in the nav of {mkdocs}')


def render_tests(model, mkdocs=ROOT / "mkdocs.yml", about=ROOT / "trust" / "test_pages.yml"):
    """The Function tests landing page: each page, what it checks and how many of its parts are verified."""
    about = yaml.safe_load(open(about)) if pathlib.Path(about).exists() else {}
    by_page = collections.defaultdict(list)
    for k, doc in model.get("docs", {}).items():
        if k.startswith("virgil.") and k in model["nodes"]:
            by_page[doc.partition("#")[0]].append(model["nodes"][k])
    lines = ["---\nhide:\n  - toc\n---",
             "<!-- Generated by scripts/trust.py from mkdocs.yml and trust/test_pages.yml. Do not edit. -->",
             "# Function tests\n",
             "virgil, part by part. Each page below checks a group of virgil's functions against mathematics, "
             "published standards or other people's code, and explains how. The bar shows how many of the parts "
             "explained on that page are verified. To find one part, look for its square in the "
             "[skyline on the Overview](../index.md#parts), or search the list under it.\n",
             '<div class="vt">']
    for section, pages in test_sections(mkdocs):
        lines.append(f'<h2 id="{slugify(section)}">{e(section)}</h2>')
        if about.get("sections", {}).get(section):
            lines.append(f'<p class="vt-muted">{e(about["sections"][section])}</p>')
        rows = []
        for title, path in pages:
            nodes = by_page.get(path, [])
            counts = collections.Counter(n["verdict"] for n in nodes)
            ok = counts.get("verified", 0)
            status = (f'{_bar(counts, len(nodes))}<span class="vt-muted">{ok} of {len(nodes)} verified</span>'
                      if nodes else '<span class="vt-muted">no parts of its own</span>')
            href = "../" + doc_url(path)
            rows.append(f'<a class="vt-tp" href="{e(href)}"><span class="vt-tp-t">{e(title)}</span>'
                        f'<span class="vt-tp-d">{e(about.get("pages", {}).get(path, ""))}</span>'
                        f'<span class="vt-tp-s">{status}</span></a>')
        lines.append('<div class="vt-tps">' + "".join(rows) + "</div>")
    lines.append("</div>\n")
    return "\n".join(lines)


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


# Showcase tiles: (page, images, title, text). A tile with several images is a gallery strip. A tile is drawn
# only once its page and at least one of its images exist under docs/, so a result can be wired in before it is
# made (IM Lup: design/alma_imlup_gallery.md).
SHOWCASE = [
    ("imaging/postagb.md", ["imaging/postagb/HD108015_images.png"], "Disc images",
     "Dusty discs around nine post-AGB binaries, reconstructed by virgil from VLTI/PIONIER data "
     "and set beside the published images."),
    ("orbits/index.md", ["assets/orbit_aliases/gl229/sky.png"], "Binary orbits",
     "Orbits of close binaries from VLTI/GRAVITY and PIONIER archives, fitted without starting from the "
     "published orbit and compared with it."),
    ("imaging/imlup.md", ["imaging/imlup/maxent.png", "imaging/imlup/multipole_gp.png", "imaging/imlup/dsharp.png"],
     "IM Lup with ALMA", "The DSHARP IM Lup disc imaged by virgil from the measurement sets, beside the DSHARP image."),
]


def _showcase(base, docs=ROOT / "docs", showcase=None):
    cards = []
    for page, imgs, title, text in (SHOWCASE if showcase is None else showcase):
        imgs = [i for i in imgs if (docs / i).exists()]
        if not imgs or not (docs / page).exists():
            continue
        pics = "".join(f'<img src="{base}{e(i)}" alt="" loading="lazy">' for i in imgs)
        if len(imgs) > 1:
            pics = f'<span class="vt-show-g">{pics}</span>'
        cards.append(f'<a class="vt-show" href="{base}{e(doc_url(page))}">{pics}'
                     f'<span class="vt-show-t">{e(title)} →</span><span class="vt-show-d">{e(text)}</span></a>')
    return f'<div class="vt-shows">{"".join(cards)}</div>'


def _headline_html(verified, total, stale, commits, refs_ok, refs_total):
    """The headline: parts of virgil only, staleness in the count itself."""
    since = f", {stale} of them since changed in virgil" if stale else ""
    where = ", ".join(f"<code>{e(c)}</code>" for c in commits)
    return (f'<div class="vt-head"><div><span class="vt-big">{verified}</span> of {total} parts of virgil verified{since} '
            f'<span class="vt-muted">(evidence from virgil {where})</span></div>'
            f'<div class="vt-muted">Our own reference code and the packages we compare with: '
            f'{refs_ok} of {refs_total} verified.</div></div>')


def _weekly_html(weekly):
    """Which weekly or manual run supplied the slow tests' evidence, or why none did; None (not asked) prints nothing.
    ``weekly`` is what scripts/fetch_ci_evidence.sh wrote to full.source.json."""
    if weekly is None:
        return ""
    if weekly.get("run_id"):
        run = f'run {e(weekly["run_id"])}'
        if weekly.get("url"):
            run = f'<a href="{e(weekly["url"])}">{run}</a>'
        return f'<div class="vt-muted">Weekly evidence from {e(str(weekly.get("date", ""))[:10])}, {run}.</div>'
    why = {"too_old": lambda w: f'the newest weekly run (run {e(w.get("ignored_run"))}) is {e(w.get("age_days"))} days old, '
                                f'past the {e(w.get("max_age_days"))}-day limit',
           "no_evidence": lambda w: f'the newest weekly run (run {e(w.get("ignored_run"))}) produced none',
           "provenance": lambda w: f'the newest weekly run (run {e(w.get("ignored_run"))}) is left out of this build: '
                                   f'it ran on uncommitted code or on a commit outside this branch',
           }.get(weekly.get("reason"), lambda w: "there is no weekly run yet")(weekly)
    return (f'<div class="vt-muted">No weekly evidence: {why}. The tests that run only weekly show as unchecked, '
            f"and virgil's own CI evidence is missing.</div>")


def _headline_by_order_html(tests):
    """A warning when a check shows a number chosen by recording order, not by its test (no ``headline=``)."""
    if not tests:
        return ""
    names = ", ".join(f"<code>{e(t.rpartition('/')[2])}</code>" for t in tests)  # file basename::function
    return (f'<details class="vt-muted vt-stale"><summary>Headline chosen by order: {len(tests)} test function(s) '
            f'record several agreement metrics and name none as their headline, so the number shown is the first one recorded.'
            f'</summary>{names}</details>')


def _fails_elsewhere_html(fails):
    """A banner listing tests that fail on another virgil commit than the kept (passing) record, with run links."""
    if not fails:
        return ""
    items = "".join(
        f'<li><code>{e(f["test"])}</code>: {e(f["outcome"])} on virgil <code>{e(f["commit"][:7])}</code>'
        + (f' (<a href="{e(f["run_url"])}">run</a>)' if f.get("run_url") else "") + "</li>"
        for f in fails)
    affected = sorted({o for f in fails for o in f["objects"]})
    commits = ", ".join(f"<code>{e(c[:7])}</code>" for c in sorted({f["commit"] for f in fails}))
    return (f'<div class="vt-stale vt-failelse-banner"><strong>{len(fails)} test(s) fail on a different virgil commit '
            f'({commits}) than the passing result this page shows.</strong> A newer run on another commit passed them, '
            f'so the page keeps that pass; the failure is shown here and the verdicts are unchanged. '
            f'Parts and pipelines affected: {", ".join(f"<code>{e(o)}</code>" for o in affected)}.'
            f'<details><summary>The tests</summary><ul>{items}</ul></details></div>')


def render(model, base=""):
    nodes = model["nodes"]
    ours_nodes = [n for n in nodes.values() if n["id"].startswith("virgil.")]
    total = len(ours_nodes)
    counts = collections.Counter(n["verdict"] for n in ours_nodes)
    stale = sum(1 for n in ours_nodes if n["verdict"] == "verified" and n["changed"])
    refs = [n for n in nodes.values() if not n["id"].startswith("virgil.")]
    refs_ok = sum(1 for n in refs if n["verdict"] == "verified")
    commits = sorted({r["virgil"][:7] for r in model["runs"] if r.get("virgil")})
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
        _weekly_html(model.get("weekly")),
        _headline_by_order_html(model.get("headline_by_order", [])),
        _fails_elsewhere_html(model.get("fails_elsewhere", [])),
        _bar(counts, total),
        _legend(counts),
        '<h2 id="showcase">Results on real data</h2>',
        _showcase(base),
        '<h2 id="real-data">Does virgil reproduce published results?</h2>',
        '<div class="vt-cards">' + ("".join(_pipeline_card(p, nodes) for p in real) or '<div class="vt-muted">None yet.</div>') + "</div>",
        '<h2 id="simulated-data">On simulated data: whole chains, from file to answer</h2>',
        '<div class="vt-chains">' + "".join(_chain(p, nodes) for p in sim) + "</div>",
        '<h2 id="parts">Every part, from data up</h2>',
        _skyline(model, base),
        '<details class="vt-fold" id="parts-list"><summary>The same parts as a searchable list, by layer</summary>',
        '<p class="vt-muted">Each chip is one part of virgil: select it to see how it was checked and read the page that '
        'explains it. Each layer name opens its <a href="' + base + 'tests/">function tests</a>. '
        "A dot marks parts virgil has changed since their evidence was measured.</p>",
        _map(nodes, base),
        "</details>",
        '<h2 id="ledger">What the checks found</h2>',
        _ledger(model["ledger"]),
        '<p class="vt-muted">Every finding, with its tests: <a href="' + base + 'method/findings/">Findings</a>.</p>',
        '<h2 id="roots">Who checks what</h2>',
        '<details class="vt-fold"><summary>Independent roots behind each part: trust does not rest on one source.</summary>',
        _matrix(nodes),
        "</details>",
        '<h2 id="precision">How precise</h2>',
        '<details class="vt-fold"><summary>Most checks agree to machine precision; the looser ones are noisy simulations '
        "or deliberate approximations.</summary>",
        '<p class="vt-muted">Each dot is one check’s agreement with its reference (hover for which).</p>',
        _precision(nodes),
        "</details>",
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
