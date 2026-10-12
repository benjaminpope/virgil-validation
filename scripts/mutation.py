#!/usr/bin/env python
"""Mutation testing of virgil-validation's tier A suite.

Each mutant in ``mutants/mutants.yml`` is a precise textual patch of one or more
virgil source files, pinned to a virgil commit. For every mutant the harness

1. exports ``src/virgil`` at the pinned commit (``git archive``: the virgil
   checkout is never touched) into a scratch directory and applies the patch
   (it fails loudly if a snippet does not occur exactly ``count`` times);
2. puts that copy first on ``PYTHONPATH`` and runs tier A (``-m "not slow"``,
   ``-x``, ``-q``, ``-p no:cacheprovider``) with a timeout, tests that validate
   the mutant's trust-graph nodes first (``scripts/mutation_order.py``), and
   deselecting the tests that already fail without any mutation;
3. records killed (a test failed), survived (the suite passed), timeout, or
   error (the mutant broke collection: a harness bug to fix).

Every survivor is a candidate gap in the validation: a slip in virgil that no
test notices. ``report`` writes ``results.json`` and ``report.md``.

Usage (from the repository root)::

    python scripts/mutation.py check                    # every mutant applies cleanly?
    python scripts/mutation.py run --only M01,M11       # a few
    python scripts/mutation.py run --shard 0/8          # one slice (e.g. a Slurm array task)
    python scripts/mutation.py report                   # merge results into results.json + report.md

``VIRGIL_REPO`` is a virgil git checkout (default ``../virgil``).
"""

from __future__ import annotations

import argparse
import ast
import collections
import datetime
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MUTANTS = ROOT / "mutants" / "mutants.yml"
GRAPH = ROOT / "trust" / "graph.yml"
OUT = ROOT / "trust" / "mutation"
DEFAULT_TIMEOUT = 600  # seconds per mutant; tier A takes ~3-5 min on an idle laptop


class MutationError(Exception):
    """A mutant does not apply to the pinned source."""


# ------------------------------------------------------------------ the mutant list


def load_mutants(path=MUTANTS):
    """``(pin, mutants)``, each mutant with an ``edits`` list (file, original, replacement, count)."""
    spec = yaml.safe_load(Path(path).read_text())
    mutants = []
    for m in spec["mutants"]:
        edits = m.get("edits") or [
            {k: m[k] for k in ("file", "original", "replacement") if k in m} | {"count": m.get("count", 1)}
        ]
        for e in edits:
            e.setdefault("count", 1)
        mutants.append({**{k: v for k, v in m.items() if k not in ("file", "original", "replacement", "count")}, "edits": edits})
    return str(spec["pin"]), mutants


def load_graph():
    return yaml.safe_load(GRAPH.read_text())["nodes"]


def virgil_repo():
    return Path(os.environ.get("VIRGIL_REPO") or ROOT.parent / "virgil")


def git_show(repo, commit, file):
    out = subprocess.run(
        ["git", "-C", str(repo), "show", f"{commit}:{file}"], capture_output=True, text=True
    )
    if out.returncode:
        raise MutationError(f"cannot read {file} at {commit} in {repo}: {out.stderr.strip()}")
    return out.stdout


def have_commit(repo, commit):
    if not Path(repo, ".git").exists() and not Path(repo, "HEAD").exists():
        return False
    return subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"], capture_output=True
    ).returncode == 0


def apply_mutant(mutant, read):
    """The patched text of each file a mutant touches, ``{file: text}``.

    ``read(file)`` gives the pinned source. Raises ``MutationError`` unless every
    ``original`` occurs exactly ``count`` times, is changed, and the result parses.
    """
    texts = {}
    for i, e in enumerate(mutant["edits"]):
        file, old, new, count = e["file"], e["original"], e["replacement"], e["count"]
        text = texts[file] if file in texts else read(file)
        found = text.count(old)
        if found != count:
            raise MutationError(
                f"{mutant['id']} edit {i}: {file} has {found} occurrence(s) of the snippet, expected {count}:\n{old}"
            )
        if old == new:
            raise MutationError(f"{mutant['id']} edit {i}: replacement equals the original")
        texts[file] = text.replace(old, new)
    for file, text in texts.items():
        if file.endswith(".py"):
            try:
                ast.parse(text)
            except SyntaxError as err:
                raise MutationError(f"{mutant['id']}: {file} no longer parses: {err}") from err
    return texts


def node_ranks(mutant, graph):
    """``{node: rank}``: 0 for the mutant's own nodes, 1 for nodes that (transitively) depend on them."""
    ranks = {n: 0 for n in mutant["nodes"]}
    changed = True
    while changed:
        changed = False
        for name, node in graph.items():
            if name not in ranks and any(d in ranks for d in node.get("depends", []) or []):
                ranks[name] = 1
                changed = True
    return ranks


def validate_list(mutants, graph):
    """Structural problems of the list (unknown nodes, duplicate ids), as strings."""
    problems, seen = [], set()
    for m in mutants:
        if m["id"] in seen:
            problems.append(f"duplicate id {m['id']}")
        seen.add(m["id"])
        for key in ("cls", "description", "nodes"):
            if not m.get(key):
                problems.append(f"{m['id']}: no {key}")
        problems += [f"{m['id']}: unknown trust-graph node {n}" for n in m.get("nodes", []) if n not in graph]
    return problems


# ------------------------------------------------------------------ running


def export_tree(repo, commit, dest):
    """``src/virgil`` of ``commit`` into ``dest/src/virgil``."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    git = subprocess.Popen(["git", "-C", str(repo), "archive", commit, "src/virgil"], stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(dest)], stdin=git.stdout, check=True)
    if git.wait():
        raise MutationError(f"git archive {commit} failed in {repo}")
    return dest / "src"


def _env(src, rank=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(src), str(ROOT / "src"), str(ROOT / "scripts")])
    env["MUTATION_RANK"] = json.dumps(rank or {})
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    return env


def check_import_path(python, src):
    """Fail loudly unless ``import virgil`` resolves inside ``src`` (and not to an editable install)."""
    out = subprocess.run(
        [python, "-c", "import virgil; print(virgil.__file__)"],
        capture_output=True, text=True, env=_env(src), cwd=ROOT,
    )
    where = out.stdout.strip()
    if out.returncode or not Path(where).resolve().is_relative_to(Path(src).resolve()):
        raise MutationError(f"import virgil resolves to {where!r} (wanted inside {src}):\n{out.stderr[-500:]}")


def run_pytest(python, src, rank, timeout, deselect=(), first_fail=True, extra=()):
    """Run tier A against the virgil copy in ``src``; ``(returncode | 'timeout', output, seconds)``."""
    cmd = [python, "-m", "pytest", "-m", "not slow", "-q", "-p", "no:cacheprovider", "-p", "mutation_order", *extra]
    if first_fail:
        cmd.append("-x")
    for node in deselect:
        cmd += ["--deselect", node]
    cmd.append("tests")
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, cwd=ROOT, env=_env(src, rank), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
        rc = "timeout"
    return rc, out, time.time() - t0


_FAIL = re.compile(r"^(FAILED|ERROR) (\S+?)(?: - (.*))?$", re.M)
_SUMMARY = re.compile(r"=+ (.*?) in [\d.]+s", re.M)


def parse_failures(output):
    """``[(kind, nodeid, message)]`` of the short test summary."""
    return [(k, n, (m or "").strip()) for k, n, m in _FAIL.findall(output)]


def classify(rc, output, baseline_failures=()):
    """``(status, killed_by, message)`` for one mutant run."""
    if rc == "timeout":
        return "timeout", [], ""
    failures = [f for f in parse_failures(output) if f[1] not in baseline_failures]
    collection_error = "error collecting" in output.lower() or "errors during collection" in output
    if rc == 0:
        return "survived", [], ""
    if rc == 1 and failures and not collection_error:
        return "killed", [n for _, n, _ in failures], failures[0][2]
    return "error", [], (output[-1500:])


def run_baseline(python, scratch, repo, pin, timeout):
    """Tier A on the unmutated pinned virgil, no ``-x``: the failures that every mutant inherits."""
    src = export_tree(repo, pin, Path(scratch) / "baseline")
    check_import_path(python, src)
    rc, out, seconds = run_pytest(python, src, {}, timeout, first_fail=False)
    if rc == "timeout":
        raise MutationError(f"the baseline run timed out after {timeout}s")
    failures = sorted({n for _, n, _ in parse_failures(out)})
    summary = _SUMMARY.findall(out)
    return {
        "pin": pin, "returncode": rc, "seconds": round(seconds, 1), "summary": summary[-1] if summary else "",
        "failing_tests": failures, "when": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def run_mutant(mutant, python, scratch, repo, pin, timeout, graph, baseline_failures=()):
    texts = apply_mutant(mutant, lambda f: git_show(repo, pin, f))
    root = Path(scratch) / mutant["id"]
    if root.exists():
        shutil.rmtree(root)
    src = export_tree(repo, pin, root)
    for file, text in texts.items():
        target = root / file
        if not target.exists():
            raise MutationError(f"{mutant['id']}: {file} is not in the exported tree")
        target.write_text(text)
    check_import_path(python, src)
    rc, out, seconds = run_pytest(python, src, node_ranks(mutant, graph), timeout, deselect=baseline_failures)
    status, killed_by, message = classify(rc, out, set(baseline_failures))
    summary = _SUMMARY.findall(out)
    shutil.rmtree(root, ignore_errors=True)
    return {
        "id": mutant["id"], "cls": mutant["cls"], "nodes": mutant["nodes"], "description": mutant["description"],
        "files": sorted({e["file"] for e in mutant["edits"]}),
        "status": status, "killed_by": killed_by, "message": message, "returncode": rc,
        "seconds": round(seconds, 1), "pytest": summary[-1] if summary else "",
    }


# ------------------------------------------------------------------ report


def rate(rows):
    killed = sum(r["status"] == "killed" for r in rows)
    denominator = sum(r["status"] in ("killed", "survived", "timeout") for r in rows)
    return killed, denominator


def build_report(results, baseline, pin, n_listed):
    """The JSON summary: overall, per class and per trust-graph node."""
    killed, denominator = rate(results)
    by_cls, by_node = collections.defaultdict(list), collections.defaultdict(list)
    for r in results:
        by_cls[r["cls"]].append(r)
        for n in r["nodes"]:
            by_node[n].append(r)

    def table(groups):
        out = {}
        for name, rows in sorted(groups.items()):
            k, d = rate(rows)
            out[name] = {"mutants": len(rows), "killed": k, "survived": sum(r["status"] == "survived" for r in rows),
                         "other": sum(r["status"] in ("timeout", "error") for r in rows),
                         "kill_rate": (k / d) if d else None}
        return out

    return {
        "pin": pin, "listed": n_listed, "run": len(results), "killed": killed,
        "design_sha256": hashlib.sha256((ROOT / "design" / "mutation.yml").read_bytes()).hexdigest(),
        "mutants_sha256": hashlib.sha256(MUTANTS.read_bytes()).hexdigest(),
        "survived": sum(r["status"] == "survived" for r in results),
        "timeout": sum(r["status"] == "timeout" for r in results),
        "error": sum(r["status"] == "error" for r in results),
        "kill_rate": (killed / denominator) if denominator else None,
        "baseline": baseline, "by_class": table(by_cls), "by_node": table(by_node),
        "mutants": sorted(results, key=lambda r: r["id"]),
    }


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def render_markdown(report):
    r = report
    base = r.get("baseline") or {}
    lines = [
        "# Mutation testing of the tier A suite", "",
        f"virgil pinned at `{r['pin'][:10]}`. {r['run']} of {r['listed']} listed mutants run: "
        f"**{r['killed']} killed, {r['survived']} survived**, {r['timeout']} timed out, {r['error']} errored. "
        f"Kill rate (killed / (killed + survived + timed out)) **{_pct(r['kill_rate'])}**.", "",
        f"Design `design/mutation.yml` sha256 `{r['design_sha256'][:12]}`, mutant list sha256 `{r['mutants_sha256'][:12]}`.", "",
        f"Baseline (no mutation): {base.get('summary', 'not recorded')}; "
        f"{len(base.get('failing_tests', []))} tests already failing there, deselected from every mutant run.", "",
        "## By class of slip", "", "| class | mutants | killed | survived | kill rate |", "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, t in r["by_class"].items():
        lines.append(f"| {name} | {t['mutants']} | {t['killed']} | {t['survived']} | {_pct(t['kill_rate'])} |")
    lines += ["", "## By trust-graph node", "",
              "A mutant counts under each node it targets.", "",
              "| node | mutants | killed | survived | kill rate |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, t in sorted(r["by_node"].items(), key=lambda kv: (kv[1]["kill_rate"] is None, kv[1]["kill_rate"] or 0, kv[0])):
        lines.append(f"| `{name}` | {t['mutants']} | {t['killed']} | {t['survived']} | {_pct(t['kill_rate'])} |")
    survivors = [m for m in r["mutants"] if m["status"] == "survived"]
    lines += ["", f"## Survivors ({len(survivors)}): candidate gaps", "",
              "Each is a slip no tier A test notices (or an equivalent mutant: check before filing a gap).", ""]
    for m in survivors:
        lines += [f"- **{m['id']}** ({m['cls']}; {', '.join(m['files'])}): {m['description']} "
                  f"Nodes: {', '.join(f'`{n}`' for n in m['nodes'])}."]
    others = [m for m in r["mutants"] if m["status"] in ("timeout", "error")]
    if others:
        lines += ["", "## Timed out or errored", ""]
        lines += [f"- **{m['id']}** {m['status']}: {m['description']}" for m in others]
    killed_rows = [m for m in r["mutants"] if m["status"] == "killed"]
    lines += ["", "## Kills", "", "| mutant | class | first failing test | s |", "| --- | --- | --- | ---: |"]
    for m in killed_rows:
        lines.append(f"| {m['id']} | {m['cls']} | `{m['killed_by'][0] if m['killed_by'] else ''}` | {m['seconds']:.0f} |")
    return "\n".join(lines) + "\n"


def cmd_report(args):
    pin, mutants = load_mutants()
    out = Path(args.out)
    results = [json.loads(p.read_text()) for p in sorted((out / "results").glob("M*.json"))]
    baseline_path = out / "baseline.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    report = build_report(results, baseline, pin, len(mutants))
    (out / "results.json").write_text(json.dumps(report, indent=1) + "\n")
    (out / "report.md").write_text(render_markdown(report))
    print(f"{report['run']}/{report['listed']} run: {report['killed']} killed, {report['survived']} survived, "
          f"{report['timeout']} timeout, {report['error']} error; kill rate {_pct(report['kill_rate'])}")
    return 0


# ------------------------------------------------------------------ commands


def cmd_check(args):
    pin, mutants = load_mutants()
    repo = virgil_repo()
    if not have_commit(repo, pin):
        print(f"commit {pin} not found in {repo} (set VIRGIL_REPO)", file=sys.stderr)
        return 2
    problems = validate_list(mutants, load_graph())
    for m in mutants:
        try:
            apply_mutant(m, lambda f: git_show(repo, pin, f))
        except MutationError as err:
            problems.append(str(err))
    for p in problems:
        print("PROBLEM:", p, file=sys.stderr)
    print(f"{len(mutants)} mutants against virgil {pin[:10]}: {'all apply cleanly' if not problems else f'{len(problems)} problem(s)'}")
    return 1 if problems else 0


def _select(mutants, args):
    if args.only:
        wanted = [s for s in args.only.split(",") if s]
        unknown = set(wanted) - {m["id"] for m in mutants}
        if unknown:
            raise SystemExit(f"unknown mutant ids {sorted(unknown)}")
        mutants = [m for m in mutants if m["id"] in wanted]
    if args.shard:
        index, total = (int(x) for x in args.shard.split("/"))
        mutants = mutants[index::total]
    return mutants


def cmd_run(args):
    pin, mutants = load_mutants()
    repo = virgil_repo()
    if not have_commit(repo, pin):
        raise SystemExit(f"commit {pin} not found in {repo} (set VIRGIL_REPO)")
    graph = load_graph()
    problems = validate_list(mutants, graph)
    if problems:
        raise SystemExit("\n".join(problems))
    out = Path(args.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    scratch = Path(args.scratch or tempfile.mkdtemp(prefix="virgil-mutation-"))
    python = args.python
    baseline_path = Path(args.baseline) if args.baseline else out / "baseline.json"
    if baseline_path.exists() and not args.rebaseline:
        baseline = json.loads(baseline_path.read_text())
        if baseline["pin"] != pin:
            raise SystemExit(f"{baseline_path} is for {baseline['pin']}, not the pinned {pin}; use --rebaseline")
    else:
        print("baseline: tier A on the unmutated pinned virgil ...", flush=True)
        baseline = run_baseline(python, scratch, repo, pin, args.baseline_timeout)
        baseline_path.write_text(json.dumps(baseline, indent=1) + "\n")
        print(f"  {baseline['summary']} in {baseline['seconds']}s; {len(baseline['failing_tests'])} failing", flush=True)
    for m in _select(mutants, args):
        path = out / "results" / f"{m['id']}.json"
        if path.exists() and not args.rerun:
            print(f"{m['id']}: already done ({json.loads(path.read_text())['status']})", flush=True)
            continue
        result = run_mutant(m, python, scratch, repo, pin, args.timeout, graph, baseline["failing_tests"])
        path.write_text(json.dumps(result, indent=1) + "\n")
        first = result["killed_by"][0] if result["killed_by"] else ""
        print(f"{m['id']} {m['cls']}: {result['status']} in {result['seconds']:.0f}s {first}", flush=True)
        if result["status"] == "error":
            print(result["message"], file=sys.stderr)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="apply every mutant to the pinned source in memory")
    r = sub.add_parser("run", help="run tier A against each mutant")
    r.add_argument("--only", help="comma-separated mutant ids")
    r.add_argument("--shard", help="i/n: every n-th mutant from i (for a Slurm array)")
    r.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="seconds per mutant")
    r.add_argument("--baseline-timeout", type=int, default=1800)
    r.add_argument("--python", default=sys.executable)
    r.add_argument("--out", default=str(OUT))
    r.add_argument("--scratch", help="scratch directory (default: a new temporary one)")
    r.add_argument("--baseline", help="reuse this baseline.json")
    r.add_argument("--rebaseline", action="store_true")
    r.add_argument("--rerun", action="store_true", help="redo mutants that already have a result")
    rep = sub.add_parser("report", help="merge per-mutant results into results.json and report.md")
    rep.add_argument("--out", default=str(OUT))
    args = p.parse_args(argv)
    return {"check": cmd_check, "run": cmd_run, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
