"""Join the trust graph, the evidence and virgil's history into the Trust page.

    python scripts/trust.py --evidence trust/evidence/latest.jsonl \\
        [--evidence trust/evidence/virgil.jsonl] [--virgil ~/code/drpangloss] \\
        [--out docs/index.md] [--json docs/assets/trust.json]
        [--findings docs/method/findings.md] [--coverage docs/method/coverage.md] [--tests docs/tests/index.md]

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
import json
import pathlib
import re
import subprocess
import sys

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
                   "virgil's own documentation and is not yet confirmed by another code or a published result"),
    "failing": ("Check failing", "a check of it fails"),
    "unchecked": ("Not yet checked", "no independent check yet"),
}
ROOT_NAMES = {
    "mathematics": "mathematics", "standards": "standards", "statistics": "statistics",
    "literature": "published result", "dlux": "dLux", "pmoired": "PMOIRED", "candid": "CANDID",
    "fouriever": "fouriever", "frank": "frank / FRAP (one root)", "visr": "vis-r", "frap": "FRAP", "oimodeler": "oimodeler",
    "ehtim": "eht-imaging", "mpol": "MPoL", "casa": "CASA", "orbitize": "orbitize!",
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


def load_evidence(paths, strict=False, virgil=None):
    """Runs and test records. A test that appears in several runs keeps only
    its newest outcome, so an old pass cannot outlive a newer skip or failure.
    A record that carries both ``virgil_commit`` and ``criteria_commit`` (a
    campaign test stamps the commits its summary was measured at,
    ``evidence.plugin.stamp_campaign``) is read at its own commit instead of
    its run header's; with ``strict`` it is left out unless it records
    ``virgil_dirty: false``.

    With ``strict``, a run with a provenance problem is left out entirely.
    ``virgil``: a virgil checkout, used to tell whether a failing commit is newer than the kept one."""
    runs, by_test = [], {}
    for path in paths:
        lines = [json.loads(line) for line in open(path)]
        run, tests = lines[0], lines[1:]
        if strict and (why := provenance_problem(run)):
            print(f"left out {path}: {why}")
            continue
        runs.append(run)
        for t in tests:
            stamped = bool(t.get("virgil_commit") and t.get("criteria_commit"))
            if strict and stamped and t.get("virgil_dirty") is not False:
                print(f"left out {t['test']} of {path}: stamped with virgil {t['virgil_commit'][:7]} but not recorded as a clean tree")
                continue
            t["_commit"] = t["virgil_commit"] if stamped else (run.get("virgil") or {}).get("commit")
            t["_source"] = t.get("source", "ours")
            t["_date"] = run.get("date") or ""
            t["_validation_commit"] = run.get("validation_commit")
            t["_run_url"] = run.get("run_url")
            by_test.setdefault(t["test"], []).append(t)
    records = []
    for recs in by_test.values():
        recs.sort(key=lambda r: r["_date"])
        newest = recs[-1]
        if (other := _fails_on_other_commit(recs, newest, virgil)):
            newest = dict(newest, _fails_elsewhere=other)
        records.append(newest)
    return runs, records


def _is_newer_commit(virgil, kept, other, other_date, kept_date):
    """Is virgil commit ``other`` newer than ``kept``? By git ancestry when a
    virgil checkout can tell (``kept`` is an ancestor of ``other``). Without one
    the run dates decide, which is conservative: the usual case (a weekly run on
    virgil main older than the push run on the pin) is then not flagged, so the
    published page, which has the checkout, is what shows it. The run dates
    alone cannot say which commit is newer, so this never badges a failure at an
    old pin that a later bump fixed."""
    if virgil is not None:
        try:
            return subprocess.run(["git", "-C", str(virgil), "merge-base", "--is-ancestor", kept, other],
                                  capture_output=True).returncode == 0
        except OSError:
            pass
    return other_date > kept_date


def _fails_on_other_commit(recs, kept, virgil=None):
    """The newest failure of a test that the kept record hides: its newest
    record on some *other* virgil commit that is *newer* than the kept record's
    commit failed while the kept record passed (typically a weekly run on virgil
    main, then a push run on the older pin). A failure at an older commit that a
    later one fixed is not flagged. ``None`` if there is none. Display only: it
    never changes a verdict."""
    if kept["outcome"] not in GOOD or not kept["_commit"]:
        return None
    latest = {}
    for r in recs:  # compare dates explicitly: the result must not depend on the order of ``recs``
        if r["_commit"] and r["_commit"] != kept["_commit"] and (
                r["_commit"] not in latest or r["_date"] >= latest[r["_commit"]]["_date"]):
            latest[r["_commit"]] = r
    failed = [r for r in latest.values() if r["outcome"] in BAD
              and _is_newer_commit(virgil, kept["_commit"], r["_commit"], r["_date"], kept["_date"])]
    if not failed:
        return None
    r = max(failed, key=lambda r: r["_date"])
    return {"commit": r["_commit"], "outcome": r["outcome"], "run_url": r.get("_run_url"), "date": r["_date"]}


def load_golden(path=ROOT / "trust" / "golden.yml"):
    return set(yaml.safe_load(open(path)) or {}) if pathlib.Path(path).exists() else set()


# frank and FRAP both apply the Baddour-Chouinard Fourier-Bessel matrix on the J0 zeros: they agree to ~1e-16 for
# the same Rmax and N, so together they are one independent root. Records are counted under the first name.
# Legacy "render" records in older evidence files map to "mathematics".
ROOT_ALIAS = {"frap": "frank", "render": "mathematics"}


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
            strong.add(ROOT_ALIAS.get(root, root))
    return strong, weak


def changed_files(virgil, commit, paths):
    """Files under ``paths`` changed in virgil between ``commit`` and
    origin/main ([] if none, None if that cannot be told)."""
    if virgil is None or commit is None:
        return None
    if not paths:
        try:
            ok = all(
                subprocess.run(
                    ["git", "-C", str(virgil), "rev-parse", "--verify", f"{ref}^{{commit}}"],
                    capture_output=True,
                ).returncode == 0
                for ref in (commit, "origin/main")
            )
        except OSError:
            return None
        return [] if ok else None
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


def _worst(value):
    """A metric recorded more than once is a list (evidence.plugin.metrics_of); the headline is its worst case: a
    non-finite entry (stored as a string) if there is one, else the entry of largest magnitude."""
    if not isinstance(value, list):
        return value
    odd = [v for v in value if not isinstance(v, (int, float))]
    return odd[0] if odd else max(value, key=abs)


def _headline(record):
    """The number that summarises a check: the marker's ``headline``, else
    the first recorded metric that measures agreement. A repeated metric is
    reduced to its worst case."""
    metrics = record.get("metrics") or {}
    name = record.get("headline")
    if name in metrics:
        return name, _worst(metrics[name])
    if name is not None:
        # an explicit headline that was not recorded: show no number rather than one the test did not choose
        return None, None
    if record.get("kind") == "control":
        # a negative control is built to disagree: its "error" metrics are not a result to headline
        return None, None
    for key, value in metrics.items():
        value = _worst(value)
        if AGREEMENT.search(key) and isinstance(value, (int, float)):
            return key, value
    return None, None


def _headline_by_order(records):
    """Tests whose headline is chosen by recording order: a check with no ``headline=`` and more than one
    candidate metric (a distinct number that measures agreement)."""
    out = set()
    for r in records:
        if r.get("kind") != "check" or r.get("headline") is not None:
            continue
        cands = [k for k, v in (r.get("metrics") or {}).items()
                 if AGREEMENT.search(k) and isinstance(_worst(v), (int, float))]
        if len(cands) > 1:
            out.add(r["test"].partition("[")[0])  # one entry per test function, not per parameter set
    return sorted(out)


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
NOT_TOPIC_PAGES = {"index.md", "trust.md", "evidence.md", "results.md", "tests/index.md"}
# Showcase pages (real-data results): the explanation of a part only when no
# function-test page covers it, like the method pages.
SHOWCASE_PAGES = {"imaging/postagb.md", "orbits/index.md", "imaging/imlup.md", "wr_gravity/index.md"}


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
        # the topic sections first; the showcase and method pages only for parts no topic covers
        fallback = [p for p in pages if p in SHOWCASE_PAGES or p.startswith("method/")]
        for tier in ([p for p in pages if p not in fallback], fallback):
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
# or results written by other people. Mathematics and standards are our own
# transcriptions of the documented formula, so they cannot.
EXTERNAL = {"pmoired", "candid", "fouriever", "orbitize", "ehtim", "mpol", "casa", "dlux", "literature",
            "frank", "visr", "frap", "oimodeler"}

def depends_depths(nodes):
    """Length of the longest ``depends`` chain below each node: 0 for a node with no
    ``depends``, else 1 + the largest depth among those it depends on. Targets that
    are not nodes are ignored. Nodes on a ``depends`` cycle share one depth, counted
    from the edges that leave the cycle (a cycle has no bottom, and is reported
    elsewhere as a structural problem), so the result does not depend on the order
    the nodes are listed in."""
    deps = {n: [d for d in node.get("depends", []) if d in nodes and d != n] for n, node in nodes.items()}
    # strongly connected components (Tarjan, iterative)
    index, low, on_stack, stack, comp, order = {}, {}, set(), [], {}, []
    for start in nodes:
        if start in index:
            continue
        work = [(start, iter(deps[start]))]
        index[start] = low[start] = len(index)
        stack.append(start)
        on_stack.add(start)
        while work:
            n, it = work[-1]
            for d in it:
                if d not in index:
                    index[d] = low[d] = len(index)
                    stack.append(d)
                    on_stack.add(d)
                    work.append((d, iter(deps[d])))
                    break
                if d in on_stack:
                    low[n] = min(low[n], index[d])
            else:
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[n])
                if low[n] == index[n]:
                    while True:
                        m = stack.pop()
                        on_stack.discard(m)
                        comp[m] = n
                        if m == n:
                            break
                    order.append(n)
    # ``order`` lists components dependencies first, so each component's depth is final
    # by the time a component that depends on it is reached
    members = collections.defaultdict(list)
    for n, c in comp.items():
        members[c].append(n)
    depth = {}
    for c in order:
        below = [depth[comp[d]] for n in members[c] for d in deps[n] if comp[d] != c]
        depth[c] = 1 + max(below) if below else 0
    return {n: depth[comp[n]] for n in nodes}


def build(graph, ledger, records, runs, virgil=None, golden=None):
    nodes = graph["nodes"]
    golden = load_golden() if golden is None else golden
    by_obj = collections.defaultdict(list)
    for r in records:
        for obj in r["objects"]:
            by_obj[obj].append(r)
    open_findings = collections.defaultdict(list)  # unfixed bugs in virgil or our reference code
    external_open = collections.defaultdict(list)  # unfixed problems in another package (or test-side on non-reference)
    for e in ledger:
        if e["status"] not in UNFIXED:
            continue
        if e["ruling"] == "virgil":
            for obj in e["objects"]:
                open_findings[obj].append(e["id"])
        elif e["ruling"].startswith("external"):
            for obj in e["objects"]:
                external_open[obj].append(e["id"])
        elif e["ruling"] == "crosscheck":
            for obj in e["objects"]:
                if nodes.get(obj, {}).get("layer", "references") == "references":
                    open_findings[obj].append(e["id"])
                else:
                    external_open[obj].append(e["id"])

    own = {}
    ref_set = {n for n, node in nodes.items() if node.get("layer", "references") == "references"}

    def properties_short(name, prop_roots):
        """Properties of a part with fewer independent roots than it needs."""
        return sorted(p for p, n in (nodes[name].get("properties") or {}).items() if len(prop_roots.get(p, ())) < n)

    def unconfirmed_conventions(name, prop_roots):
        """Conventions taken from virgil's docs with no external root."""
        return sorted(p for p in nodes[name].get("convention") or []
                      if not (prop_roots.get(p, set()) & EXTERNAL))

    def _node_is_direct_broken(name):
        return name in nodes and (
            any(r["outcome"] in BAD for r in by_obj.get(name, []))
            or name in open_findings
        )

    def own_status(name, trusted_references, is_broken_ref=None):
        """A part's own verdict from its own checks. A check that went through
        our reference code (its ``via``) counts only if every such reference is
        in ``trusted_references``, for reference nodes as much as for parts of
        virgil (``reference_verdicts`` finds that set); checks
        blocked that way are stratified into usable vs broken ``via`` tiers so
        adding a supplementary check via a broken reference never demotes a
        ``relies-unverified`` part to ``relies``."""
        node = nodes[name]
        recs = by_obj.get(name, [])
        reference_node = node.get("layer", "references") == "references"
        agreeing = AGREEING_FOR_REFERENCES if reference_node else AGREEING
        counted = [r for r in recs if r["kind"] in agreeing or r["kind"] == "regression"]
        # xfailed is not in BAD, so an expected failure is fine; a strict XPASS (pytest records it
        # as failed, with no wasxfail) or a non-strict one (xpassed) means the bug is gone or the
        # xfail is stale, so finding/upstream records count as failing like any other.
        failing = [r for r in recs if r["outcome"] in BAD]
        good = [r for r in counted if r["outcome"] == "passed"]
        strong, weak = set(), set()
        usable_roots, broken_roots = set(), set()
        usable_blocked_by, broken_blocked_by = set(), set()
        blocked_checks = []  # display only: counted checks held back by an unverified via reference
        prop_roots = collections.defaultdict(set)
        prop_roots_usable = collections.defaultdict(set)
        prop_roots_all = collections.defaultdict(set)
        for r in good:
            s_, w_ = roots_of(r, golden)
            untrusted = set()
            if r["kind"] in agreeing:
                untrusted = {v for v in r.get("via", []) if v not in trusted_references}
            if r["kind"] in agreeing and not untrusted:
                strong |= s_
                for prop in r.get("properties", []):
                    prop_roots[prop] |= s_
                    prop_roots_usable[prop] |= s_
                    prop_roots_all[prop] |= s_
            elif r["kind"] in agreeing:
                blocked_checks.append({"test": r["test"], "via": sorted(untrusted)})
                is_broken = is_broken_ref is not None and any(is_broken_ref(v) for v in untrusted)
                if not is_broken:
                    usable_roots |= s_
                    usable_blocked_by |= untrusted
                    for prop in r.get("properties", []):
                        prop_roots_usable[prop] |= s_
                        prop_roots_all[prop] |= s_
                else:
                    broken_roots |= s_
                    broken_blocked_by |= untrusted
                    for prop in r.get("properties", []):
                        prop_roots_all[prop] |= s_
            weak |= w_ | (s_ if r["kind"] not in agreeing else set())
        strong, weak = sorted(strong), sorted(weak - set(strong))
        blocked_roots = usable_roots | broken_roots
        usable_all_roots = set(strong) | usable_roots
        all_roots = set(strong) | blocked_roots
        need = node.get("roots", 1)
        via_out = []
        shown = prop_roots  # the property roots of the tier that decided the status
        if failing:
            status = "failing"
        elif name in open_findings:
            status = "bug"
        elif (
            len(strong) >= need
            and not external_open.get(name)
            and not properties_short(name, prop_roots)
            and not unconfirmed_conventions(name, prop_roots)
        ):
            status = "ok"
        elif (
            len(usable_all_roots) >= need
            and not external_open.get(name)
            and not properties_short(name, prop_roots_usable)
            and not unconfirmed_conventions(name, prop_roots_usable)
        ):
            status = "ok-via"  # its checks pass through usable (non-broken) reference code not yet verified
            via_out = sorted(usable_blocked_by)
            shown = prop_roots_usable
        elif (
            len(all_roots) >= need
            and not external_open.get(name)
            and not properties_short(name, prop_roots_all)
            and not unconfirmed_conventions(name, prop_roots_all)
        ):
            status = "ok-via"  # its checks pass, relying on reference code with a known bug or failing check
            via_out = sorted(usable_blocked_by | broken_blocked_by)
            shown = prop_roots_all
        elif len(all_roots) >= need and not external_open.get(name) and properties_short(name, prop_roots_all):
            status = "partly"  # agreement overall, but a property needs more independent roots
            shown = prop_roots_all
        elif len(all_roots) >= need and not external_open.get(name) and unconfirmed_conventions(name, prop_roots_all):
            status = "convention"
            shown = prop_roots_all
        elif strong or weak or blocked_roots:
            status = "partly"
            shown = prop_roots_all
        else:
            status = "unchecked"
        return status, strong, weak, need, recs, good, via_out, shown, blocked_checks

    def reference_verdicts():
        """Our reference nodes that are verified: the least fixed point of
        "its own checks pass, counting only checks whose ``via`` references are
        already in the set, and every ``depends`` is in the set". Starting from
        the empty set and growing it, so a reference checked only through a
        failing, unchecked or untrusted reference stays out, and so does a cycle
        of references that vouch only for each other. The step is monotone (a
        larger set only lets more checks count), so the loop ends within
        ``len(refs) + 1`` passes."""
        refs = [n for n in nodes if n in ref_set]
        trusted_refs = set()
        for _ in range(len(refs) + 1):
            grown = {
                n for n in refs
                if all(d in trusted_refs for d in nodes[n].get("depends", []))
                and own_status(n, trusted_refs)[0] == "ok"
            }
            if grown == trusted_refs:
                break
            trusted_refs = grown
        else:  # pragma: no cover - a monotone step cannot get here
            raise RuntimeError("trusted references did not converge")
        return trusted_refs

    diff_cache = {}

    def cached_changed_files(commit, paths):
        key = (commit, paths)
        if key not in diff_cache:
            diff_cache[key] = changed_files(virgil, commit, list(paths))
        return diff_cache[key]

    trusted = reference_verdicts()

    def broken_targets():
        """Nodes that cannot back a check: the least fixed point of "has an open
        finding or a failing check, or relies on (``depends``) a broken node, or
        its own verdict is ``ok-via`` through a broken reference". Built up from
        the empty set, so the answer does not depend on which node is asked about
        first, and a cycle is broken only if one of its members is broken for a
        reason of its own. Monotone: a larger set can only move a check from the
        usable to the broken tier, and both tiers are ``ok-via``, so a node's
        verdict never leaves ``ok-via`` as the set grows."""
        broken = set()
        # a node with no check that goes through untrusted references has the
        # same status for every ``broken``; the others are re-evaluated each pass
        status = {n: own_status(n, trusted, broken.__contains__) for n in nodes}
        for _ in range(len(nodes) + 1):
            grown = set()
            for n in nodes:
                if status[n][0] == "ok-via":
                    status[n] = own_status(n, trusted, broken.__contains__)
                if (
                    _node_is_direct_broken(n)
                    or any(d in broken for d in nodes[n].get("depends", []))
                    or (status[n][0] == "ok-via" and any(u in broken for u in status[n][6]))
                ):
                    grown.add(n)
            if grown == broken:
                return status  # each node's own status at the converged broken set
            broken = grown
        raise RuntimeError("broken targets did not converge")  # pragma: no cover

    own_status_cache = broken_targets()

    for name in [n for n in nodes if n in ref_set] + [n for n in nodes if n not in ref_set]:
        node = nodes[name]
        reference_node = name in ref_set
        status, strong, weak, need, recs, good, blocked_by, prop_roots, blocked_checks = own_status_cache[name]
        agreeing = AGREEING_FOR_REFERENCES if reference_node else AGREEING
        commits = sorted({r["_commit"] for r in good if r["_commit"]})
        src_paths = tuple(sorted(set(node.get("source", []))))
        changes = [cached_changed_files(c, src_paths) for c in commits]
        changed = None if not changes or any(c is None for c in changes) else sorted({f for c in changes for f in c})
        own[name] = {
            "status": status, "strong": strong, "weak": weak, "need": need,
            "checks": [_check(r) for r in recs], "commits": commits, "changed": changed,
            "findings": open_findings.get(name, []) + external_open.get(name, []),
            "skipped": sum(1 for r in recs if r["outcome"] == "skipped" and r["kind"] in agreeing),
            "via_blocked": blocked_by,
            "blocked_checks": blocked_checks,
            "properties": {p: {"need": n, "roots": sorted(prop_roots.get(p, ()))}
                           for p, n in (node.get("properties") or {}).items()},
            "unconfirmed": unconfirmed_conventions(name, prop_roots),
        }

    # verdicts: a part whose own checks pass is verified only if everything
    # it relies on is (with no missing targets, cycles, or reference->subject edges)
    raw_cache = {}

    def _raw_problems(root):
        if root not in raw_cache:
            raw_cache[root] = _raw_problems_uncached(root)
        return raw_cache[root]

    def _raw_problems_uncached(root):
        found = set()
        structural = set()
        in_stack = {root}
        done = set()

        def visit(curr):
            curr_is_ref = nodes[curr].get("layer", "references") == "references"
            for dep in nodes[curr].get("depends", []):
                if dep not in nodes:
                    found.add(dep)
                    structural.add(dep)
                    continue
                if curr_is_ref and nodes[dep].get("layer", "references") != "references":
                    found.add(dep)
                    structural.add(dep)
                if dep in in_stack:
                    found.add(dep)
                    structural.add(dep)
                    continue
                if own[dep]["status"] != "ok":
                    found.add(dep)
                if dep not in done:
                    in_stack.add(dep)
                    visit(dep)
                    in_stack.remove(dep)
                    done.add(dep)

        visit(root)
        return found, structural

    def _new_walk(*active):
        """State of one walk for the causes of one part: the nodes being walked (an ancestor
        reached again contributes nothing, so cycles end), the result already found for each
        node walked (so a node reachable along many routes is walked once, not once per route),
        and the ancestors cut in the node being walked (``hit``)."""
        return {"active": set(active), "done": {True: {}, False: {}}, "hit": set()}

    def _expand_node(v, walk, named):
        """Causes below one reference ``v``. ``named``: the walk is down a part's own chain of
        checks, where a cycle of references that vouch only for each other is itself the cause and
        is named by the node at which the cycle closes; below a dependency a cycle names nothing
        (the dependency is the cause)."""
        memo = walk["done"][named]
        if v in memo:
            return memo[v]
        if v in walk["active"]:
            if named:
                walk["hit"].add(v)
            return set()
        walk["active"].add(v)
        outer_hit, walk["hit"] = walk["hit"], set()
        sub = problems(v, walk)
        out = set(sub)
        if own[v]["status"] == "ok-via":
            out |= _expand_via_blocked(own[v]["via_blocked"], walk, named)
            if named and not out and not walk["hit"] - {v}:
                out = {v}  # the cycle closes here: it names itself (a node cut at an ancestor leaves it to that)
        elif nodes[v].get("layer", "references") != "references" or own[v]["status"] != "ok" or not sub:
            out.add(v)
        inner = walk["hit"] - {v}
        walk["active"].discard(v)
        walk["hit"] = outer_hit | inner
        if not (named and inner):  # a result that stood on an ancestor still being walked is not final
            memo[v] = out
        return out

    def _expand_via_blocked(blocked, walk=None, named=True):
        walk = walk if walk is not None else _new_walk()
        out = set()
        for v in blocked:
            if v not in nodes:
                out.add(v)
            else:
                out |= _expand_node(v, walk, named)
        return out

    def problems(root, walk=None):
        walk = walk if walk is not None else _new_walk(root)
        raw, structural = _raw_problems(root)
        out = set(structural)
        for d in raw:
            if d in nodes and own[d]["status"] == "ok-via":
                # if the walk below it finds nothing (its references only vouch for each other
                # in a cycle) the dependency itself is the cause
                out |= _expand_via_blocked(own[d]["via_blocked"], walk, False) or {d}
            else:
                out.add(d)
        return out

    deps_cache = {}
    ts_cache = {}

    def reachable_deps(name):
        if name not in deps_cache:
            seen = {name}
            stack = [d for d in nodes[name].get("depends", []) if d in nodes and d != name]
            out = []
            while stack:
                d = stack.pop()
                if d in seen:
                    continue
                seen.add(d)
                out.append(d)
                for nxt in nodes[d].get("depends", []):
                    if nxt in nodes and nxt not in seen:
                        stack.append(nxt)
            deps_cache[name] = tuple(out)
        return deps_cache[name]

    def transitive_sources(name):
        """Union of virgil ``source`` paths on ``name`` and every part it relies on."""
        if name not in ts_cache:
            paths = set(nodes[name].get("source", []))
            for dep in reachable_deps(name):
                paths.update(nodes[dep].get("source", ()))
            ts_cache[name] = tuple(sorted(paths))
        return ts_cache[name]

    def changed_below(name):
        """virgil files changed under this part or anything it relies on, evaluated
        against both this part's own evidence commits and each dependency's own
        evidence commits (``None`` when unknown or unreachable)."""
        commits = own[name]["commits"]
        if not commits:
            return None
        pairs = [(c, transitive_sources(name)) for c in commits]
        for dep in reachable_deps(name):
            dep_commits = own[dep]["commits"]
            if dep_commits:
                dep_paths = transitive_sources(dep)
                pairs.extend((c, dep_paths) for c in dep_commits)
        diffs = [cached_changed_files(c, p) for c, p in pairs]
        if any(d is None for d in diffs):
            return None
        return sorted({f for d in diffs for f in d})

    depth = depends_depths(nodes)
    out_nodes = {}
    for name, node in nodes.items():
        o = own[name]
        verdict = o["status"]
        walk = _new_walk(name)
        via_causes = _expand_via_blocked(o["via_blocked"], walk) if o["status"] == "ok-via" else set()
        causes = sorted((problems(name, walk) | via_causes) - {name})  # never its own cause (a via + depends cycle)
        if o["status"] == "ok-via" and not causes:
            # its checks stand on references that are not verified, yet the walk found nothing
            # wrong below them (they only vouch for each other in a cycle): they are the cause
            causes = sorted(o["via_blocked"])
        if verdict == "ok-via":
            verdict = "ok"
        if verdict == "ok":
            if not causes:
                verdict = "verified"
            elif any(own.get(c, {}).get("status") in {"bug", "failing"} for c in causes):
                verdict = "relies"
            else:
                verdict = "relies-unverified"
        if node.get("layer", "references") == "references" and name not in trusted and verdict == "verified":
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
            "because": [
                {
                    "id": c,
                    "findings": own.get(c, {}).get("findings", []),
                    "status": "ok"
                    if own.get(c, {}).get("status") == "ok-via"
                    else own.get(c, {}).get("status", "unchecked"),
                }
                for c in causes
            ],
            "depends": node.get("depends", []), "dependents": dependents, "depth": depth[name],
            "properties": o["properties"], "unconfirmed": o["unconfirmed"],
            "checks": o["checks"], "commits": o["commits"],
            "changed": changed_below(name),
            "skipped": o["skipped"],
            # display only: counted checks held back by a reference that is not verified, and which
            # references (the same set for every status, unlike ``via_blocked`` above of an ok-via part)
            "blocked_checks": o["blocked_checks"],
            "blocked_by_references": sorted({v for c in o["blocked_checks"] for v in c["via"]}),
            "source": node.get("source", []),
        }

    def ref_broken(v):
        return any(
            own.get(c, {}).get("status") in {"bug", "failing"}
            for c in _expand_via_blocked((v,), named=False)
        )

    pipelines = []
    for name, p in graph.get("pipelines", {}).items():
        all_recs = by_obj.get(f"pipeline:{name}", [])
        recs = [r for r in all_recs if r["kind"] in AGREEING]
        bad = [r for r in all_recs if r["outcome"] in BAD]
        good = [r for r in recs if r["outcome"] == "passed"]
        strong_good = [r for r in good if roots_of(r, golden)[0]]
        usable_good = [r for r in strong_good if not any(ref_broken(v) for v in r.get("via", []))]
        trusted_good = [r for r in usable_good if not (set(r.get("via", [])) - trusted)]
        via_failing = bool(strong_good and not usable_good)
        if via_failing:
            untrusted_via = sorted({v for r in strong_good for v in r.get("via", []) if v not in trusted})
        elif not trusted_good:
            untrusted_via = sorted({v for r in usable_good for v in r.get("via", []) if v not in trusted})
        else:
            untrusted_via = []
        unverified_steps = [s for s in p["steps"] if out_nodes.get(s, {}).get("verdict") != "verified"]
        unverified = unverified_steps + [v for v in untrusted_via if v not in unverified_steps]
        step_failing = any(
            out_nodes.get(s, {}).get("verdict") == "failing"
            or any(c.get("status") == "failing" for c in out_nodes.get(s, {}).get("because", []))
            for s in p["steps"]
        )
        if bad or step_failing or via_failing:
            status = "failing"
        elif trusted_good and not unverified:
            status = "validated"
        elif usable_good:
            status = "passes"
        elif recs and all(r["outcome"] == "skipped" for r in recs):
            status = "not run"  # its checks exist but none ran, so the hand-written state would mislead
        else:
            status = p.get("state", "planned")
        pipelines.append({
            "id": name, "title": p["title"], "data": p.get("data", "simulated"),
            "dataset": p.get("dataset"), "reference": p.get("reference"), "url": p.get("url"),
            "headline": p.get("headline"), "page": p.get("page"), "status": status, "steps": p["steps"],
            "depth": 1 + max((depth.get(s, 0) for s in p["steps"]), default=0),
            "unverified": unverified, "failing_checks": [r["test"] for r in bad],
            "checks": [_check(r) for r in recs],
        })

    fails_elsewhere = _mark_fails_elsewhere(records, out_nodes, pipelines)

    counts = collections.Counter(n["verdict"] for n in out_nodes.values())
    changed = sum(1 for n in out_nodes.values() if n["changed"])
    unknown = all(n["changed"] is None for n in out_nodes.values() if n["commits"])
    return {
        "runs": [{k: r.get(k) for k in ("date", "runner", "run_url")} | {"virgil": (r.get("virgil") or {}).get("commit")} for r in runs],
        "counts": dict(counts),
        "changed": None if unknown else changed,
        "headline_by_order": _headline_by_order(records),
        "nodes": out_nodes,
        "pipelines": pipelines,
        "ledger": ledger,
        # tests that fail on another virgil commit than the one whose passing record the page keeps (display only)
        "fails_elsewhere": fails_elsewhere,
    }


def _mark_fails_elsewhere(records, out_nodes, pipelines):
    """The tests that fail on another virgil commit than the kept record, as a
    list for the banner. Each affected node and pipeline gets a
    ``fails_elsewhere`` list of those tests for its badge. Verdicts are untouched."""
    found = [r for r in records if r.get("_fails_elsewhere")]
    for item in [*out_nodes.values(), *pipelines]:
        item["fails_elsewhere"] = []
    for r in found:
        for obj in r["objects"]:
            item = out_nodes.get(obj) or next((p for p in pipelines if f"pipeline:{p['id']}" == obj), None)
            if item is not None and r["test"] not in item["fails_elsewhere"]:
                item["fails_elsewhere"].append(r["test"])
    return sorted(({"test": r["test"], "objects": sorted(r["objects"]), **r["_fails_elsewhere"]} for r in found),
                  key=lambda f: f["test"])


# ------------------------------------------------------------------ render
# The page is drawn by scripts/trust_render.py. Its names stay importable from here (tests, other scripts), and it
# is imported here, after the constants above, because it reads some of them (see trust_render.attach).

_here = str(pathlib.Path(__file__).resolve().parent)
if _here not in sys.path:  # a file-path load (importlib.util.spec_from_file_location) has not put scripts/ there
    sys.path.insert(0, _here)
import trust_render  # noqa: E402

trust_render.attach(globals())
from trust_render import (  # noqa: E402,F401
    ICON, ORDER, COLOURS, _bar, _legend, _value, _pipeline_card, _chain, _tooltip, layers, _fails_badge, _map,
    _matrix, _link_label, _ledger, FINDING_GROUPS, render_findings, render_coverage, LAYER_SECTION, test_sections,
    render_tests, _precision, _tables, SHOWCASE, _headline_html, _headline_by_order_html, _fails_elsewhere_html,
    render, render_roots,
)
e = trust_render.e  # not in the import above: build() uses `e` as a loop variable, which ruff flags against an import


def _showcase(base, docs=ROOT / "docs"):
    """The showcase tiles; ``SHOWCASE`` is looked up here so a replacement of ``trust.SHOWCASE`` is honoured."""
    return trust_render._showcase(base, docs, SHOWCASE)


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
    ap.add_argument("--roots", default=None, help="also write the Roots page here (docs/method/roots.md)")
    ap.add_argument("--tests", default=None, help="also write the Function tests landing page here (docs/tests/index.md)")
    args = ap.parse_args()
    graph = yaml.safe_load(open(args.graph))
    ledger = yaml.safe_load(open(args.ledger))
    virgil = args.virgil.expanduser() if args.virgil else None
    runs, records = load_evidence(args.evidence, strict=args.strict_provenance, virgil=virgil)
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
    if args.roots:
        pathlib.Path(args.roots).write_text(render_roots())
    if args.tests:
        pathlib.Path(args.tests).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.tests).write_text(render_tests(model))
    if args.json:
        pathlib.Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.json).write_text(json.dumps(model, indent=1, default=str))
    print(dict(collections.Counter(n["verdict"] for n in model["nodes"].values())),
          {p["id"]: p["status"] for p in model["pipelines"]})


if __name__ == "__main__":
    main()
