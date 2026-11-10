"""The ``mandate`` command."""

from __future__ import annotations

import argparse
import csv
import io
import os
import pathlib
import sys
import urllib.request

from . import audit, scaffold, work
from .model import UNITS, Campaign, MandateError, matches
from .orders import expand, find

PROTOCOL = """\
How a group of agents works a campaign
======================================

Lead agent (one session, holds no science context):
  1. `mandate status`           read what is waiting for a person, what failed, what is ready.
  2. `mandate orders --state ready --worker agent`   the agent orders that can start now.
  3. For each, start a fresh worker and give it only the output of `mandate brief ORDER`.
     Workers do not share context with you or with each other.
  4. `mandate orders --state fail` (and `error`): give each to a fresh worker, same way. Its job is
     to diagnose from the run folder and then decide (from the menu), propose an exclusion, or
     file a request.
  5. Code stages: `mandate run --stage STAGE` locally, or `mandate run --stage STAGE --slurm` to
     write a job array for a person to submit. Never submit to a cluster yourself.
  6. Commit ledger/ and docket.jsonl, run `mandate verify`, write STATUS.md with `mandate status --write`.
  7. Stop when nothing is ready. Orders that are waiting, blocked or parked need upstream work or
     a person; say which, and do not work around them.

Worker agent (one order):
  Everything you may do is in your brief. Set MANDATE_ACTOR to a name for yourself.

People:
  Edit mandate.yaml and the registry (by commit), answer the docket with
  `mandate resolve ID approve|deny --why ...`, and read STATUS.md.
"""


def pairs(items: list[str]) -> dict:
    out = {}
    for item in items:
        if "=" not in item:
            raise MandateError(f"expected COLUMN=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def table_lines(text: str) -> list[str]:
    """The header and data lines of a CSV. MAST exports open with '#' comment lines, and their
    header is either a plain line or the widest '#' line ('#dataproduct_type,...')."""
    lines = [x for x in text.splitlines() if x.strip()]
    start = next((i for i, x in enumerate(lines) if not x.startswith("#")), len(lines))
    comments = [x.lstrip("#") for x in lines[:start] if not x.startswith("#@")]
    if not comments:
        return lines
    header = max(comments, key=lambda x: x.count(","))
    body = lines[start:]
    if body and body[0].split(",")[0] == header.split(",")[0]:
        return body
    return [header, *body]


def ingest(root: pathlib.Path, source: str, id_column: str | None) -> str:
    """Snapshot a table (a CSV file, or a sheet's CSV export URL) as the registry and say what changed."""
    if "://" in source:
        with urllib.request.urlopen(source) as f:  # the user names the sheet
            text = f.read().decode("utf-8-sig")
    else:
        text = pathlib.Path(source).read_text(encoding="utf-8-sig")
    rows = list(csv.DictReader(table_lines(text)))
    plan_id = Campaign(root).unit_id if (root / "mandate.yaml").exists() else "id"
    key = id_column or plan_id
    if not rows or key not in rows[0]:
        raise MandateError(f"the table has no {key!r} column (pass --id)")
    target = root / UNITS
    old = {}
    if target.exists():
        with open(target, newline="") as f:
            old = {r.get(key): r for r in csv.DictReader(f)}
    new = {r[key]: r for r in rows}
    if len(new) != len(rows):
        raise MandateError(f"{key!r} is not unique in the table")
    target.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    target.write_text(buffer.getvalue())
    changed = sum(1 for k in new if k in old and old[k] != new[k])
    return (f"{len(new)} units: {len(set(new) - set(old))} added, {len(set(old) - set(new))} removed, "
            f"{changed} changed. Commit registry/ to make this the mandate.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mandate", description="Fan agents and jobs out over datasets, under a plan.")
    p.add_argument("-C", dest="root", default=os.environ.get("MANDATE_ROOT", "."), help="campaign directory")
    sub = p.add_subparsers(dest="cmd")
    a = sub.add_parser("init", help="start a campaign directory")
    a.add_argument("directory")
    a.add_argument("--name")
    sub.add_parser("guide", help="print instructions for an agent helping someone write a plan")
    sub.add_parser("columns", help="list the registry's columns and their values")
    sub.add_parser("explain", help="read the plan back in plain words")
    a = sub.add_parser("why", help="why an order is waiting, blocked or failed, traced to its cause")
    a.add_argument("order")
    a = sub.add_parser("unit", help="everything about one dataset: row, links, notes, orders")
    a.add_argument("selector", help="a unit id, or COLUMN=VALUE")
    a = sub.add_parser("withdraw", help="take back a proposal or request you made")
    a.add_argument("id")
    a.add_argument("--why", default="withdrawn by its author")
    a.add_argument("--by")
    a = sub.add_parser("review", help="answer everything that is waiting for a person")
    a.add_argument("--by", help="who is answering (default: MANDATE_ACTOR or USER)")
    a.add_argument("--list", action="store_true", help="only list what is waiting, with the commands to answer")
    sub.add_parser("check", help="validate the plan, registry and docket")
    sub.add_parser("protocol", help="print how agents and people work a campaign")
    a = sub.add_parser("ingest", help="snapshot a CSV (file or URL) as the registry")
    a.add_argument("source")
    a.add_argument("--id", help="the column that names a unit (default: the plan's unit_id)")
    a = sub.add_parser("status", help="print the status page")
    a.add_argument("--write", action="store_true", help="also write STATUS.md")
    a = sub.add_parser("orders", help="list work orders")
    a.add_argument("--stage")
    a.add_argument("--state", action="append")
    a.add_argument("--worker", choices=["code", "agent"])
    a = sub.add_parser("brief", help="print the self-contained brief for one order")
    a.add_argument("order")
    a = sub.add_parser("where", help="print one order's run folder")
    a.add_argument("order")
    a = sub.add_parser("run", help="run code-stage orders, or write a Slurm array for them")
    a.add_argument("orders", nargs="*")
    a.add_argument("--stage", help="every ready order of this stage")
    a.add_argument("--fresh", action="store_true", help="rerun from an empty folder and compare with the last attempt")
    a.add_argument("--slurm", action="store_true", help="write a job-array script; do not run or submit")
    a = sub.add_parser("submit", help="hand back an agent-stage result")
    a.add_argument("order")
    a = sub.add_parser("decide", help="choose a listed option for one order")
    a.add_argument("stage")
    a.add_argument("key")
    a.add_argument("set", nargs="+", metavar="NAME=OPTION")
    a = sub.add_parser("exclude", help="drop units from some or all stages")
    a.add_argument("where", nargs="+", metavar="COLUMN=VALUE")
    a.add_argument("--stage", action="append")
    a = sub.add_parser("annotate", help="set registry columns on the units a selector matches")
    a.add_argument("where", nargs="+", metavar="COLUMN=VALUE")
    a.add_argument("--set", nargs="+", required=True, metavar="COLUMN=VALUE")
    a = sub.add_parser("request", help="ask a person for something the plan does not allow; parks the order")
    a.add_argument("stage")
    a.add_argument("key", nargs="?")
    a = sub.add_parser("resolve", help="a person answers a proposal or request")
    a.add_argument("id")
    a.add_argument("verdict", choices=["approve", "deny"])
    for name in ("decide", "exclude", "annotate", "request", "resolve"):
        sub.choices[name].add_argument("--why", required=True)
        sub.choices[name].add_argument("--by", help="who is acting (default: MANDATE_ACTOR or USER)")
    a = sub.add_parser("verify", help="audit the ledger, the docket and the run folders")
    a.add_argument("--base", help="also classify every change since this git ref")
    a.add_argument("--strict", action="store_true", help="fail on items for review as well as violations")
    a = sub.add_parser("collect", help="copy order and attempt records from run folders into the ledger")
    a.add_argument("source", nargs="?")
    args = p.parse_args(argv)
    try:
        return dispatch(args)
    except MandateError as e:
        print(f"mandate: refused: {e}", file=sys.stderr)
        return 2


def find_root(start: str) -> pathlib.Path:
    """The campaign directory: the one named, or the nearest parent holding mandate.yaml."""
    here = pathlib.Path(start).resolve()
    for folder in (here, *here.parents):
        if (folder / "mandate.yaml").exists():
            return folder
    raise MandateError(f"no mandate.yaml in {here} or above it. Start one with `mandate init DIR`, "
                       "or point at one with `mandate -C DIR`.")


def dispatch(args) -> int:
    root = pathlib.Path(args.root)
    if args.cmd == "init":
        print(scaffold.init(pathlib.Path(args.directory), args.name))
        return 0
    if args.cmd == "guide":
        print(scaffold.GUIDE)
        return 0
    if args.cmd == "protocol":
        print(PROTOCOL)
        return 0
    if args.cmd == "ingest":
        print(ingest(root, args.source, args.id))
        return 0
    c = Campaign(find_root(args.root))
    if args.cmd is None:
        print(scaffold.home(c))
        return 0
    if args.cmd == "columns":
        print(scaffold.columns(c))
        return 0
    if args.cmd == "explain":
        print(scaffold.explain(c))
        return 0
    if args.cmd == "why":
        print(scaffold.why(c, args.order))
        return 0
    if args.cmd == "unit":
        print(scaffold.unit(c, args.selector))
        return 0
    if args.cmd == "withdraw":
        entry = c.append("resolve", why=args.why, by=args.by, of=args.id, verdict="deny")
        print(f"{entry['id']}\twithdrew {args.id}")
        return 0
    if args.cmd == "review":
        scaffold.review(c, by=args.by, listing=args.list)
        return 0
    if args.cmd == "check":
        problems = c.problems()
        for line in problems:
            print("problem:", line)
        if not problems:
            orders = expand(c)
            print(f"ok: {len(c.units)} units, " + ", ".join(f"{s} {len(o)}" for s, o in orders.items()))
            for stage, st in c.stages.items():
                if st.get("worker", "code") == "code" and not st.get("requires") and st.get("result") == "exit":
                    print(f"note: {stage} has no result file: it passes whenever its command exits cleanly")
                elif st.get("worker", "code") == "code" and not st.get("requires"):
                    print(f"note: {stage} names no required checks: it passes whenever its result says so")
            loose = len([u for u in c.units if not any(matches(u, s.get("over")) for s in c.stages.values())])
            if loose:
                print(f"note: {loose} units are selected by no stage (`mandate explain` lists them)")
            print("Read it back with `mandate explain`.")
        return 1 if problems else 0
    if args.cmd == "status":
        page = audit.status_page(c)
        print(page)
        if args.write:
            (c.root / "STATUS.md").write_text(page)
        return 0
    if args.cmd == "orders":
        shown = [o for stage, os_ in expand(c).items() for o in os_
                 if (not args.stage or stage == args.stage) and (not args.state or o.state in args.state)
                 and (not args.worker or c.stages[stage].get("worker", "code") == args.worker)]
        width = max([len(o.ref) for o in shown] + [1])
        for o in shown:
            print(f"{o.ref:<{width}}  {o.state:<8} {o.reason}".rstrip())
        if not shown:
            print("no orders match")
        return 0
    if args.cmd == "brief":
        o = find(c, args.order)
        if o.id and o.state in work.RUNNABLE:
            work.issue(c, o)
        print(work.brief(c, o))
        return 0
    if args.cmd == "where":
        o = find(c, args.order)
        work.require_issuable(o)
        print(work.out_dir(c, o))
        return 0
    if args.cmd == "run":
        orders = expand(c)
        chosen = [find(c, ref, orders) for ref in args.orders]
        if args.stage:
            chosen += [o for o in orders.get(args.stage, []) if o.state == "ready"]
        if not chosen:
            print("nothing to run")
            return 0
        if args.slurm:
            for stage in dict.fromkeys(o.stage for o in chosen):
                script = work.slurm_script(c, stage, [o for o in chosen if o.stage == stage])
                print(f"sbatch {script}")
            return 0
        tally: dict[str, int] = {}
        width = max(len(o.ref) for o in chosen)
        for i, o in enumerate(chosen, 1):
            attempt = work.run(c, o, fresh=args.fresh)
            tally[attempt["status"]] = tally.get(attempt["status"], 0) + 1
            print(f"[{i}/{len(chosen)}] {o.ref:<{width}}  {attempt['status']:<5} {attempt.get('problem', '')}".rstrip())
        print(", ".join(f"{n} {s}" for s, n in tally.items()) + ". `mandate` shows what to do next.")
        return 0 if set(tally) <= {"pass", "warn"} else 1
    if args.cmd == "submit":
        o = find(c, args.order)
        attempt = work.submit(c, o)
        print(f"{o.ref}  {attempt['status']}  {attempt.get('problem', '')}".rstrip())
        return 0
    if args.cmd in ("decide", "exclude", "annotate", "request", "resolve"):
        fields = {
            "decide": lambda: {"stage": args.stage, "key": args.key, "set": pairs(args.set)},
            "exclude": lambda: {"where": pairs(args.where), "stages": args.stage},
            "annotate": lambda: {"where": pairs(args.where), "set": pairs(args.set)},
            "request": lambda: {"stage": args.stage, "key": args.key},
            "resolve": lambda: {"of": args.id, "verdict": args.verdict},
        }[args.cmd]()
        if args.cmd in ("decide", "request") and args.key:
            find(c, f"{args.stage}/{args.key}")  # refuse a key that names no current order
        entry = c.append(args.cmd, why=args.why, by=args.by, **fields)
        if args.cmd in ("exclude", "annotate") and not audit.touched(c, entry["where"]):
            print("warning: the selector matches no unit", file=sys.stderr)
        state = c.state(entry) if args.cmd != "resolve" else args.verdict
        print(f"{entry['id']}  {args.cmd}  {state}")
        return 0
    if args.cmd == "verify":
        report = audit.verify(c, args.base)
        for kind in ("violations", "review", "info"):
            for line in report[kind]:
                print(f"{kind[:-1] if kind == 'violations' else kind}: {line}")
        bad = report["violations"] or (args.strict and report["review"])
        print("verify: FAILED" if bad else "verify: ok")
        return 1 if bad else 0
    if args.cmd == "collect":
        print(f"{work.collect(c, args.source or c.runs_root)} records copied into the ledger")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
