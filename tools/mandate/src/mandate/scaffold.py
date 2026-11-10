"""The parts a person meets first: starting a campaign, reading the plan back in plain words,
the home screen, and answering what agents have asked.

``GUIDE`` is for a chat agent helping someone write a plan: what to ask, what not to invent.
"""

from __future__ import annotations

import collections
import pathlib

from .audit import STATES, describe
from .model import PLAN, Campaign, MandateError, actor, git, option_names
from .orders import Order, expand, unassigned

TEMPLATE = """\
# The plan for this campaign. People edit it and commit it; a commit is the approval.
# `mandate guide` prints instructions for an agent helping you write it.
# `mandate explain` reads it back in plain words. `mandate check` validates it.
campaign: {name}
unit_id: id                # the registry column that names one unit of data
humans: [{human}]          # who may approve proposals and answer requests
# sealed_columns: []       # published answers: only stages marked `sealed: true` see them
# ignore_columns: []       # bookkeeping columns: a change is not new work

stages: {{}}
#  fit:
#    doc: One sentence on what this step does.
#    over: {{role: reference}}      # which rows
#    by: [id]                      # one order per group of rows; [] is one order for all
#    run: python steps/fit.py --order {{order}}
#    requires: [converged]         # checks that must be present and passing
#    decisions:
#      n_modes: {{default: 6, options: [4, 6, 8], authority: agent}}
"""

GUIDE = """\
Helping someone write a mandate plan
====================================

You are in a chat with a scientist. Your job is to turn their description of an analysis into
`mandate.yaml`, a plan that many agents and cluster jobs will then follow exactly. They approve
it by committing it. You never commit it.

Work in this order
------------------
1. If there is no campaign yet: `mandate init DIR`, then `mandate ingest FILE_OR_URL` for their
   table of datasets.
2. Run `mandate columns`. It lists the registry's columns and their values. Use only these names.
3. Ask the questions below. Ask them a few at a time, in their words, and skip any they have
   already answered. Do not guess an answer you were not given.
4. Write `mandate.yaml`. Run `mandate check` until it is clean.
5. Run `mandate explain` and show them the output. That text is what they are approving. Ask
   what is wrong with it, fix the plan, and repeat.
6. Tell them to read the diff and commit. Stop there.

What to ask
-----------
- Units. What is one row of the table: an exposure, a night, a target? Which column names it?
- Kinds of data. Which column tells calibrators from science targets, and one science case from
  another? If no column does, how should it be filled: by code when the table is built, by an
  agent stage that reads proposals or papers (with a person approving), or by hand?
- Steps. What are the steps, in order? For each: is it done once per row, per target, per
  night, per filter, or once for everything? What command runs it? What does it read from
  earlier steps?
- Combined products (a basis, a calibration, a catalogue). What goes into each, what is the
  least it needs to be worth building, and what must be kept out?
- "Nearest" inputs. Does a step use the calibrators closest in time or position? How many?
- Pairings. Is any dataset tied to particular others by the observing plan (a science target
  and its PSF reference, two rolls, a calibrator and its science exposure)? Those go in
  `registry/links.csv` as `from,to,kind,source`; ask where the pairing is recorded.
- Success. For each step, which named checks must pass? If they have none, say so plainly:
  a stage with no required checks passes whenever its command exits cleanly.
- Choices. What legitimately varies between datasets (a model family, a number of modes, a
  subtraction method)? List the allowed values. May an agent pick, or must a person approve?
- Published answers. Which datasets have a published result, and in which column or note does
  it live? Those become sealed, with a scoring stage that is the only one to see them.
- Code. Which repositories should be pinned to a commit?
- People. Who may approve?

Rules for you
-------------
- Never invent a column, a command, a check name or an option. If you need one that does not
  exist, ask. A command you were not given is a question, not a placeholder.
- Keep it small: the fewest stages that describe the work. One stage per kind of step, not per
  dataset. Differences between datasets are `over` selectors, `rules` and `decisions`.
- Leave `authority` out (it then needs a person) unless they said an agent may choose.
- A per-dataset special case does not go in the plan. It goes in the docket later, with a reason.
- Human notes and paper extracts go in `notes/` as Markdown, each with `applies_to` and, for
  anything that reveals a published result, `visibility: sealed`.

The plan
--------
    campaign: NAME
    unit_id: COLUMN
    humans: [NAME, ...]
    sealed_columns: [COLUMN, ...]
    ignore_columns: [COLUMN, ...]
    versions: {NAME: {git: PATH, pin: COMMIT}}        # or {cmd: SHELL, pin: TEXT}
    stages:
      STAGE:
        doc: one sentence
        over: {COLUMN: VALUE}                         # or {COLUMN: [VALUES]}; all must match
        by: [COLUMN, ...]                             # default [unit_id]; [] = one order for all
        needs:
          UPSTREAM: {match: [COLUMN], min: N, nearest: {by: COLUMN, k: N, within: X}}
          NAME: {stage: UPSTREAM, link: KIND}         # follow registry/links.csv (from,to,kind,source)
        run: COMMAND                                  # may use {order} {out} {root} {key.COLUMN}
                                                      # {unit.COLUMN} {decision.NAME} {input.STAGE}
        result: result.json                           # JSON: {"status": pass|warn|fail, "checks": ...};
                                                      # or `exit`: a clean exit is a pass
        status_key: status                            # `worst_check` for a virgil pipeline's summary.json
        requires: [CHECK, ...]
        decisions:
          NAME: {default: V, options: [V, ...], authority: agent|human,   # options may be {NAME: "flag text"}
                 rules: [{when: {COLUMN: VALUE}, use: V}], doc: when to change it}
        sealed: true                                  # may see sealed columns, notes and stages
        slurm: ["--gres=gpu:1", "--time=02:00:00"]
      AGENT_STAGE:
        worker: agent
        prompt: prompts/FILE.md                       # what the worker agent is told to do
        expects: {KEY: [ALLOWED, ...]}                # the keys of its result.json
        promote: {KEY: COLUMN}                        # its answer fills this registry column
        review: human                                 # or none

A step's command reads `{order}`: a JSON file with its units, its decisions and the run folder
of every input. It writes its result file into `{out}`.
"""


def init(directory: pathlib.Path, name: str | None = None) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / PLAN).exists():
        raise MandateError(f"{directory / PLAN} already exists")
    human = git(directory, "config", "user.name") or actor()
    (directory / PLAN).write_text(TEMPLATE.format(name=name or directory.resolve().name, human=human.split()[0].lower()))
    for folder in ("registry", "notes", "prompts"):
        (directory / folder).mkdir(exist_ok=True)
        (directory / folder / ".gitkeep").touch()
    (directory / ".gitignore").write_text("runs/\n")
    (directory / ".gitattributes").write_text("docket.jsonl merge=union\n")
    return "\n".join([
        f"Started a campaign in {directory}.", "",
        "Next:",
        "  1. mandate ingest FILE_OR_URL     your table of datasets becomes registry/units.csv",
        "  2. Ask an agent: \"run `mandate guide` and help me write the plan\"",
        "  3. mandate explain                read the plan back in plain words",
        "  4. git commit                     that is the approval; then `mandate` shows what to do",
    ])


def columns(c: Campaign) -> str:
    """The registry's columns and their commonest values: what a plan may select and group on."""
    if not c.units:
        return "The registry is empty. Run `mandate ingest FILE_OR_URL` first."
    lines = [f"{len(c.units)} units. Unit id column: {c.unit_id}.", ""]
    names = list(dict.fromkeys(k for u in c.units for k in u))
    width = max(len(n) for n in names)
    for name in names:
        values = collections.Counter(str(u.get(name, "") or "") for u in c.units)
        empty = values.pop("", 0)
        if name in c.sealed_columns:
            shown = "(sealed)"
        elif len(values) > 8:
            shown = f"{len(values)} different values, e.g. " + ", ".join(v for v, _ in values.most_common(3))
        else:
            shown = ", ".join(f"{v} ({n})" for v, n in values.most_common())
        lines.append(f"  {name:<{width}}  {shown}" + (f"; empty in {empty}" if empty else ""))
    return "\n".join(lines)


def _where(over: dict | None) -> str:
    if not over:
        return "every row"
    parts = []
    for col, want in over.items():
        wants = want if isinstance(want, list) else [want]
        parts.append(f"{col} is " + " or ".join(str(w) or "empty" for w in wants))
    return "rows where " + " and ".join(parts)


def explain(c: Campaign, orders: dict[str, list[Order]] | None = None) -> str:
    """The plan in plain words. This is what a person approves when they commit it."""
    orders = orders or expand(c)
    lines = [f"{c.plan.get('campaign', c.root.name)}: {len(c.stages)} stages over {len(c.units)} units."]
    lines.append("People who approve: " + (", ".join(sorted(c.humans)) or "NOBODY (add `humans:`)") + ".")
    if c.sealed_columns:
        lines.append("Sealed answers, seen only by sealed stages: " + ", ".join(sorted(c.sealed_columns)) + ".")
    for name, spec in (c.plan.get("versions") or {}).items():
        lines.append(f"Code: {name} " + (f"pinned to {spec['pin']}" if spec.get("pin") else "recorded, not pinned") + ".")
    feeds = collections.defaultdict(list)
    for stage, st in c.stages.items():
        for label, need in (st.get("needs") or {}).items():
            feeds[(need or {}).get("stage", label)].append(stage)
        for col in (st.get("promote") or {}).values():
            feeds[stage] += [f"{s} (through column {col})" for s, other in c.stages.items()
                             if col in (other.get("over") or {}) and s != stage]
    lines += ["", "Order of work:"]
    for stage in c.order_of_stages():
        lines.append(f"  {stage}" + (" -> " + ", ".join(feeds[stage]) if feeds[stage] else ""))
    for stage in c.order_of_stages():
        st = c.stages[stage]
        by = st.get("by", [c.unit_id])
        kind = st.get("worker", "code") + (", sealed" if st.get("sealed") else "")
        lines += ["", f"{stage}  [{kind}]  {len(orders.get(stage, []))} order(s) now"]
        if st.get("doc"):
            lines.append(f"  {st['doc'].strip()}")
        group = "Once for everything" if not by else "Once per " + " and ".join(by)
        lines.append(f"  {group}, over {_where(st.get('over'))}.")
        for up, need in (st.get("needs") or {}).items():
            need = need or {}
            which = "every passing result" if not need.get("nearest") else (
                f"the {need['nearest'].get('k', 1)} passing results nearest in {need['nearest']['by']}")
            same = f" with the same {' and '.join(need['match'])}" if need.get("match") else ""
            if need.get("link"):
                same += f" that the registry links to it as {need['link']}"
            up = need.get("stage", up)
            least = int(need.get("min", 1))
            floor = "; none is fine" if least == 0 else (f"; needs at least {least}" if least > 1 else "")
            lines.append(f"  Reads {which} of {up}{same}{floor}.")
        if st.get("worker") == "agent":
            keys = "; ".join(f"{k} as {' or '.join(map(str, v))}" if v else k for k, v in (st.get("expects") or {}).items())
            lines.append(f"  An agent follows {st.get('prompt')} and answers {keys or 'freely'}.")
            for key, col in (st.get("promote") or {}).items():
                gate = "at once" if st.get("review", "human") == "none" else "after a person approves"
                lines.append(f"  Its {key} fills registry column {col} {gate}.")
        else:
            lines.append(f"  Runs: {(st.get('run') or '').strip()}")
        if st.get("requires"):
            lines.append("  Passes only if these checks pass: " + ", ".join(st["requires"]) + ".")
        elif st.get("worker", "code") == "code":
            lines.append("  No required checks: it passes whenever its result says so.")
        for dname, d in (st.get("decisions") or {}).items():
            options = ", ".join(option_names(d))
            who = "an agent may change it for one order, with a reason" if d.get("authority") == "agent" else (
                "a change needs a person")
            lines.append(f"  Choice {dname}: {d.get('default')} by default (allowed: {options}); {who}.")
            for rule in d.get("rules") or []:
                lines.append(f"    {rule['use']} for {_where(rule.get('when'))}.")
        if st.get("sealed"):
            lines.append("  May see the sealed columns and notes.")
    loose = unassigned(c)
    if loose:
        lines += ["", f"{len(loose)} units are selected by no stage: "
                  + ", ".join(u.get(c.unit_id, "?") for u in loose[:6]) + (" ..." if len(loose) > 6 else "")]
    return "\n".join(lines)


def home(c: Campaign) -> str:
    """What `mandate` alone prints: where the campaign stands and what to do next."""
    orders = expand(c)
    lines = [f"{c.plan.get('campaign', c.root.name)}  ·  {len(c.units)} units  ·  {len(c.stages)} stages"]
    try:
        c.plan_commit()
    except MandateError:
        lines.append("! The plan or registry has uncommitted changes. Nothing runs until they are committed.")
    problems = c.problems()
    if problems:
        lines.append(f"! {len(problems)} problems with the plan: run `mandate check`.")
    pending = c.pending()
    lines += ["", f"Waiting for a person ({len(pending)})"]
    for e in pending[:8]:
        text = describe(e).replace("`", "")
        lines.append("  " + (text if len(text) <= 110 else text[:107] + "..."))
    if len(pending) > 8:
        lines.append(f"  ... and {len(pending) - 8} more")
    if not pending:
        lines.append("  Nothing.")
    width = max([len(s) for s in orders] + [5])
    lines += ["", f"{'Stage':<{width}}  " + " ".join(f"{s:>7}" for s in STATES)]
    for stage, os_ in orders.items():
        counts = [sum(o.state == s for o in os_) for s in STATES]
        lines.append(f"{stage:<{width}}  " + " ".join(f"{n or '·':>7}" for n in counts))
    todo = []
    if pending:
        todo.append((f"{len(pending)} waiting for a person", "mandate review"))
    bad = [o for os_ in orders.values() for o in os_ if o.state in ("fail", "error")]
    if bad:
        todo.append((f"{len(bad)} failed: give each brief to a worker agent", "mandate orders --state fail --state error"))
    for stage, os_ in orders.items():
        n = sum(o.state == "ready" for o in os_)
        if n and c.stages[stage].get("worker", "code") == "agent":
            todo.append((f"{n} briefs ready for agents", f"mandate orders --stage {stage} --state ready"))
        elif n:
            todo.append((f"{n} ready to run", f"mandate run --stage {stage}    (add --slurm for the cluster)"))
    lines += ["", "Next"]
    lines += [f"  {what:<46} {cmd}" for what, cmd in todo] or ["  Nothing to do: every order is finished or waiting on something above."]
    return "\n".join(lines)


def card(c: Campaign, e: dict) -> list[str]:
    """One docket entry as a person reads it: what, why, who, and what saying yes would change."""
    head = describe(e).replace("`", "").partition(". ")[0]
    what = head.split(" ", 2)[2] if head.count(" ") >= 2 else head
    return [f"{e['kind']:<9} {what}    ({e['id']})", f"      why   {e['why']}", f"      by    {e['by']}, {e['at'][:10]}",
            f"      then  {effect(c, e)}"]


def effect(c: Campaign, entry: dict) -> str:
    """What approving one entry would change, found by expanding the plan with it in force."""
    if entry["kind"] == "request":
        return f"{entry['stage']}" + (f"/{entry['key']}" if entry.get("key") else "") + " is no longer parked"
    now = {(o.stage, o.key_str): o.id for os_ in expand(c).values() for o in os_}
    trial = Campaign(c.root)
    trial.docket.append({"kind": "resolve", "of": entry["id"], "verdict": "approve"})
    trial._index()
    trial.units = trial._read_units()
    then = {(o.stage, o.key_str): o.id for os_ in expand(trial).values() for o in os_}
    new = [f"{s}/{k}" for (s, k) in then if (s, k) not in now]
    gone = [f"{s}/{k}" for (s, k) in now if (s, k) not in then]
    changed = [f"{s}/{k}" for (s, k), i in then.items() if (s, k) in now and now[(s, k)] != i]
    parts = []
    for label, items in (("new", new), ("removed", gone), ("to redo", changed)):
        if items:
            parts.append(f"{label}: " + ", ".join(items[:4]) + (f" and {len(items) - 4} more" if len(items) > 4 else ""))
    return "; ".join(parts) or "no order changes"


def review(c: Campaign, by: str | None = None, ask=input, say=print, listing: bool = False) -> int:
    """Walk a person through everything waiting for them. Returns how many they answered."""
    who = by or actor()
    pending = c.pending()
    if not pending:
        say("Nothing is waiting for a person.")
        return 0
    if listing:
        for e in pending:
            for line in card(c, e):
                say(line)
            say(f"      mandate resolve {e['id']} approve --why \"...\"   |   mandate resolve {e['id']} deny --why \"...\"")
        return 0
    if c.humans and who not in c.humans:
        raise MandateError(f"{who!r} is not one of the plan's humans ({', '.join(sorted(c.humans))}); pass --by")
    answered, sweep = 0, None
    for i, e in enumerate(pending, 1):
        if sweep == e["kind"]:
            c.append("resolve", why="approved in review (all of this kind)", by=who, of=e["id"], verdict="approve")
            answered += 1
            continue
        lines = card(c, e)
        say(f"\n[{i}/{len(pending)}] " + lines[0])
        for line in lines[1:]:
            say(line)
        request = e["kind"] == "request"
        choice = ask("      (c)lose as answered, (s)kip, (q)uit > " if request
                     else "      (a)pprove, (d)eny, (A)pprove all of this kind, (s)kip, (q)uit > ").strip()
        if choice == "q":
            break
        if choice == "A" and not request:
            sweep = e["kind"]
            choice = "a"
        if choice not in ("a", "d", "c"):
            continue
        verdict = "deny" if choice == "d" else "approve"
        why = ask("      reason" + (" (required) > " if verdict == "deny" or request else " (Enter for 'approved') > ")).strip()
        if not why and (verdict == "deny" or request):
            say("      skipped: that needs a reason")
            continue
        c.append("resolve", why=why or "approved", by=who, of=e["id"], verdict=verdict)
        answered += 1
    say(f"\n{answered} answered, {len(c.pending())} still waiting. Commit docket.jsonl to keep them.")
    return answered



def why(c: Campaign, ref: str) -> str:
    """Why an order is in the state it is in, traced back to the orders that cause it."""
    from .orders import find

    orders = expand(c)
    by_key = {(o.stage, o.key_str): o for os_ in orders.values() for o in os_}
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()

    def walk(o: Order, depth: int) -> None:
        pad = "  " * depth
        if (o.stage, o.key_str) in seen or depth > 6:
            return
        seen.add((o.stage, o.key_str))
        last = o.attempts[-1] if o.attempts else {}
        detail = o.reason or last.get("problem") or ""
        lines.append(f"{pad}{o.ref}: {o.state}" + (f" ({detail})" if detail else ""))
        if o.state in ("fail", "error"):
            lines.append(f"{pad}  -> give a worker `mandate brief {o.stage}/{o.key_str}`, or rerun with `mandate run {o.stage}/{o.key_str}`")
        elif o.state == "ready":
            verb = "brief" if c.stages[o.stage].get("worker") == "agent" else "run"
            lines.append(f"{pad}  -> mandate {verb} {o.stage}/{o.key_str}")
        elif o.state == "parked":
            for e in c.pending():
                if e["kind"] == "request" and e.get("stage") == o.stage and e.get("key") in (o.key_str, None):
                    lines.append(f"{pad}  -> waiting for a person: {e['id']} \"{e['why']}\" (mandate review)")
        for name, dropped in o.dropped.items():
            for d in dropped:
                if d["why"] == "excluded":
                    lines.append(f"{pad}  {d.get('stage', name)}/{d['key']}: excluded by the docket")
                elif d["why"] != "not nearest" and (d.get("stage", name), d["key"]) in by_key and o.state == "blocked":
                    walk(by_key[(d.get("stage", name), d["key"])], depth + 1)
        for ref_ in o.waiting:
            stage, _, rest = ref_.partition("/")
            up = next((p for p in orders.get(stage, []) if rest in (p.id, p.key_str)), None)
            if up:
                walk(up, depth + 1)

    walk(find(c, ref, orders), 0)
    return "\n".join(lines)


def unit(c: Campaign, selector: str) -> str:
    """Everything about one dataset: its row, its links, the notes on it, and its orders at every stage."""
    from .model import matches, read_notes

    where = dict([selector.split("=", 1)]) if "=" in selector else {c.unit_id: selector}
    rows = [u for u in c.units if matches(u, where)]
    if not rows:
        raise MandateError(f"no unit matches {selector!r}")
    orders = expand(c)
    lines = []
    for u in rows[:5]:
        uid = u.get(c.unit_id, "?")
        lines += [uid, "  " + ", ".join(f"{k}={v}" for k, v in c.open_row(u).items() if v and k != c.unit_id)]
        for r in c.links:
            if uid in (r.get("from"), r.get("to")):
                lines.append(f"  link   {r.get('from')} --{r.get('kind')}--> {r.get('to')}"
                             + (f"   ({r['source']})" if r.get("source") else ""))
        for n in read_notes(c, [u], "", sealed=False):
            if not n["meta"].get("stages"):
                lines.append(f"  note   notes/{n['file']}")
        for os_ in orders.values():
            for o in os_:
                if any(x.get(c.unit_id) == uid for x in o.units):
                    lines.append(f"  order  {o.ref}: {o.state}" + (f" ({o.reason})" if o.reason else ""))
        for e in c.docket:
            if e["kind"] in ("exclude", "annotate") and matches(u, e.get("where") or {}):
                lines.append("  docket " + describe(e).replace("`", "") + f" [{c.state(e)}]")
        lines.append("")
    if len(rows) > 5:
        lines.append(f"... and {len(rows) - 5} more units match")
    return "\n".join(lines).rstrip()
