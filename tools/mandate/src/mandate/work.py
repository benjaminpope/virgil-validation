"""Doing one order: the brief an agent starts from, the run, the hand-back.

The numbers always come from the stage's command. ``run`` executes it and records what happened;
an agent never writes a result for a code stage, and for an agent stage it writes one JSON file
that ``submit`` checks against the keys the plan expects.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import pathlib
import re
import shutil
import socket
import subprocess

from .model import Campaign, MandateError, actor, now, option_names, option_text, read_notes
from .orders import Order

NOT_HASHED = ("order.json", "attempt-*.json", "run.log", "*.log")
RUNNABLE = ("ready", "pass", "warn", "fail", "error")

RULES = """\
You hold exactly one work order. It is your whole mandate.

- Write only inside the run folder named below.
- Never edit mandate.yaml, the registry, the docket, the ledger, a check or a threshold, or
  another order's files. Never read anything under a path marked sealed.
- Change how this order is analysed only with `mandate decide`, and only to an option listed
  below. `mandate exclude` proposes dropping data; a person approves it.
- If the order cannot be done as written (a missing option, bad data, a method the plan does
  not have, a check you believe is wrong), run `mandate request` saying what you found and
  stop. That is a successful outcome. Working around it is not.
- Every number you report must be in a file in the run folder, written by the stage's command.
- Do not rerun to get a better answer. Every attempt is recorded and counted.
"""


def out_dir(c: Campaign, o: Order) -> pathlib.Path:
    return c.runs_root / o.stage / o.id


def ledger_dir(c: Campaign, o: Order) -> pathlib.Path:
    return c.root / "ledger" / o.stage / o.id


def require_issuable(o: Order) -> None:
    if o.id is None or o.state not in RUNNABLE:
        raise MandateError(f"{o.ref} is {o.state}" + (f" ({o.reason})" if o.reason else ""))


def order_record(c: Campaign, o: Order, plan_commit: str) -> dict:
    """What the worker is given: units, decisions, inputs and where to write. Nothing sealed
    unless the stage is."""
    sealed = c.level(o.stage)
    inputs = {
        up: [{**m, "path": str(c.runs_root / m["stage"] / m["order"])} for m in members]
        for up, members in o.inputs.items()
    }
    record = {
        "id": o.id,
        "stage": o.stage,
        "key": o.key,
        "units": [c.open_row(u, sealed) for u in o.units],
        "decisions": o.decisions,
        "inputs": inputs,
        "dropped": o.dropped,
        "out": str(out_dir(c, o)),
        "plan_commit": plan_commit,
        "campaign": c.plan.get("campaign"),
        "knowledge": sealed,
    }
    if sealed == "sealed" and c.sealed_root() is not None:
        if not c.sealed_root().exists():
            raise MandateError(f"{o.ref} is a sealed stage and {c.sealed_root()} is not on this machine")
        record["sealed_root"] = str(c.sealed_root())
    return record


def for_ledger(c: Campaign, record: dict) -> dict:
    """The order as the ledger keeps it: unit ids only, and no paths of this machine."""
    small = {k: v for k, v in record.items() if k not in ("out", "sealed_root")}
    small["units"] = [u.get(c.unit_id) if isinstance(u, dict) else u for u in record["units"]]
    small["inputs"] = {up: [{k: v for k, v in m.items() if k != "path"} for m in ms]
                       for up, ms in record["inputs"].items()}
    return small


def issue(c: Campaign, o: Order) -> dict:
    """Write the order into the ledger and the run folder (once). Refuses an uncommitted plan."""
    require_issuable(o)
    record = order_record(c, o, c.plan_commit())
    ledger = ledger_dir(c, o) / "order.json"
    if not ledger.exists():
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text(json.dumps({**for_ledger(c, record), "issued": now()}, indent=1, sort_keys=True) + "\n")
    out = out_dir(c, o)
    out.mkdir(parents=True, exist_ok=True)
    (out / "order.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return record


def fill(template: str, c: Campaign, o: Order, record: dict) -> str:
    """``{out} {order} {root} {key.COL} {unit.COL} {decision.NAME} {input.STAGE}`` and nothing else."""

    def lookup(m: re.Match) -> str:
        name = m.group(1)
        head, _, rest = name.partition(".")
        if name == "out":
            return record["out"]
        if name == "order":
            return str(pathlib.Path(record["out"]) / "order.json")
        if name == "root":
            return str(c.root)
        if head == "key" and rest in o.key:
            return str(o.key[rest])
        if head == "unit" and o.value(rest) is not None and rest not in _hidden(c, o):
            return str(o.value(rest))
        if head == "decision" and rest in o.decisions:
            return option_text(c.stages[o.stage]["decisions"][rest], o.decisions[rest]["value"])
        if head == "input" and len(record["inputs"].get(rest, [])) == 1:
            return record["inputs"][rest][0]["path"]
        raise MandateError(f"stage {o.stage}: cannot fill {{{name}}} for {o.ref}")

    return re.sub(r"\{([A-Za-z_][A-Za-z0-9_.]*)\}", lookup, template)


def _hidden(c: Campaign, o: Order) -> set:
    return set(o.units[0]) - set(c.open_row(o.units[0], c.level(o.stage))) if o.units else set()


def digest(folder: pathlib.Path, include: list[str] | None = None) -> str:
    """One hash over the run folder's files (names and contents), logs and our own records aside."""
    h = hashlib.sha256()
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        rel = path.relative_to(folder).as_posix()
        if include:
            if not any(fnmatch.fnmatch(rel, g) for g in include):
                continue
        elif any(fnmatch.fnmatch(path.name, g) for g in NOT_HASHED):
            continue
        h.update(rel.encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()[:16]


def read_result(c: Campaign, o: Order) -> tuple[str, dict, str]:
    """(status, checks, problem). The stage's own result file decides; ``requires`` names checks
    that must be present and passing, so a step cannot pass by leaving a check out."""
    st = c.stages[o.stage]
    if st.get("result") == "exit":  # no result file: the command's clean exit is the result
        return "pass", {}, ""
    name = fill(st.get("result", "result.json"), c, o, {"out": str(out_dir(c, o)), "inputs": {}})
    path = out_dir(c, o) / name
    if not path.exists():
        return "error", {}, f"no {name}"
    try:
        result = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return "error", {}, f"{path.name} is not JSON: {e}"
    status = result.get(st.get("status_key", "status"), "pass" if st.get("worker") == "agent" else None)
    raw = result.get("checks") or {}
    checks = {x["name"]: x.get("status") for x in raw} if isinstance(raw, list) else dict(raw)
    if status not in ("pass", "warn", "fail"):
        return "error", checks, f"{path.name} gives status {status!r}, not pass, warn or fail"
    missing = [n for n in st.get("requires") or [] if checks.get(n) not in ("pass", "warn")]
    if missing:
        return "fail", checks, "required checks absent or failing: " + ", ".join(missing)
    return status, checks, ""


def record_attempt(c: Campaign, o: Order, **fields) -> dict:
    out, ledger = out_dir(c, o), ledger_dir(c, o)
    n = 1 + max(len(list(ledger.glob("attempt-*.json"))), len(list(out.glob("attempt-*.json"))))
    st = c.stages[o.stage]
    attempt = {
        "order": o.id,
        "stage": o.stage,
        "n": n,
        "by": actor(),
        "host": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID"),
        "finished": now(),
        "campaign": c.campaign_state(),
        "digest": digest(out, st.get("digest")),
        "knowledge": c.level(o.stage),
        **fields,
    }
    text = json.dumps(attempt, indent=1, sort_keys=True) + "\n"
    for folder in (out, ledger):
        with open(folder / f"attempt-{n:03d}.json", "x") as f:  # never overwrite an attempt
            f.write(text)
    o.attempts.append(attempt)
    o.state = attempt["status"]
    return attempt


def run(c: Campaign, o: Order, fresh: bool = False) -> dict:
    """Run a code stage's command for one order and record the attempt."""
    st = c.stages[o.stage]
    if st.get("worker", "code") != "code":
        raise MandateError(f"{o.ref} is an agent stage: use `mandate brief` then `mandate submit`")
    require_issuable(o)
    versions = c.versions(st.get("uses"))
    out = out_dir(c, o)
    before = o.digest
    if fresh and out.exists():
        n = len(list(out.parent.glob(f"{o.id}.prev-*"))) + 1
        shutil.move(str(out), str(out.parent / f"{o.id}.prev-{n}"))
    record = issue(c, o)
    command = fill(st["run"], c, o, record)
    started = now()
    env = {**os.environ, "MANDATE_ORDER": str(out / "order.json"), "MANDATE_OUT": str(out)}
    with open(out / "run.log", "a") as log:
        log.write(f"\n# {started} {command}\n")
        log.flush()
        done = subprocess.run(command, shell=True, cwd=c.root, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    if done.returncode != 0:
        status, checks, problem = "error", {}, f"exit code {done.returncode} (see run.log)"
    else:
        status, checks, problem = read_result(c, o)
    extra = {}
    if fresh and before:
        extra["reproduced"] = digest(out, st.get("digest")) == before
    return record_attempt(
        c, o, status=status, checks=checks, problem=problem, command=command, started=started,
        exit_code=done.returncode, versions=versions, **extra,
    )


def submit(c: Campaign, o: Order) -> dict:
    """Accept an agent stage's result file: check its keys, record the attempt, propose annotations."""
    st = c.stages[o.stage]
    if st.get("worker", "code") != "agent":
        raise MandateError(f"{o.ref} is a code stage: its result comes from `mandate run`")
    require_issuable(o)
    issue(c, o)
    path = out_dir(c, o) / st.get("result", "result.json")
    if not path.exists():
        raise MandateError(f"write {path} first")
    result = json.loads(path.read_text())
    for key, allowed in (st.get("expects") or {}).items():
        if key not in result:
            raise MandateError(f"{path.name} lacks {key!r}")
        if allowed and str(result[key]) not in [str(a) for a in allowed]:
            raise MandateError(f"{path.name}: {key}={result[key]!r} is not one of {allowed}")
    status, checks, problem = read_result(c, o)
    attempt = record_attempt(c, o, status=status, checks=checks, problem=problem)
    promote = st.get("promote") or {}
    if promote and status in ("pass", "warn"):
        c.append(
            "annotate", why=str(result.get("why") or f"result of {o.ref}"), where=dict(o.key),
            set={col: result[key] for key, col in promote.items()}, review=st.get("review", "human"), source=o.ref,
        )
    return attempt


def brief(c: Campaign, o: Order) -> str:
    """Everything a worker needs to do this order and nothing it should not see.

    A fresh agent given only this text can do the work, so no context is lost between sessions.
    """
    st = c.stages[o.stage]
    sealed = c.level(o.stage)
    out = out_dir(c, o) if o.id else None
    lines = [f"# Work order {o.ref}", "", f"Campaign: {c.plan.get('campaign', c.root.name)}. State: {o.state}"
             + (f" ({o.reason})." if o.reason else "."), "", "## Your mandate", "", RULES]
    if sealed == "informed":
        note = ("This is an INFORMED stage: it may use what a publication says the object is (the notes marked "
                + "informed), so nothing it produces is a blind result. It may not use published values.")
        lines += [note, ""]
    lines += [f"Run folder: `{out}`" if out else "Run folder: none yet (the order is not issuable).", ""]
    lines += ["## The task", "", (st.get("doc") or "").strip() or "(the plan gives no description)", ""]
    if st.get("worker") == "agent":
        lines += [(c.root / st["prompt"]).read_text().strip(), ""]
        expects = st.get("expects") or {}
        keys = "; ".join(f"`{k}`" + (f" (one of {', '.join(map(str, v))})" if v else "") for k, v in expects.items())
        name = st.get("result", "result.json")
        lines += [f"Hand back: write `{name}` in the run folder with {keys or 'your result'} and a `why`, "
                  + f"then run `mandate submit {o.ref}`.", ""]
    else:
        lines += [f"Command (run it with `mandate run {o.ref}`, never by hand):", "", f"    {st['run'].strip()}", ""]
    if st.get("playbook"):
        lines += ["## Playbook", "", (c.root / st["playbook"]).read_text().strip(), ""]
    lines += [f"## Units ({len(o.units)})", ""]
    show = [c.open_row(u, sealed) for u in o.units[:20]]
    cols = [k for k in (show[0] if show else {}) if any(u.get(k) for u in show)]
    lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(u.get(k, "")) for k in cols) + " |" for u in show]
    if len(o.units) > 20:
        lines.append(f"... and {len(o.units) - 20} more (all in the run folder's order.json)")
    lines += ["", "## Inputs", ""]
    for up, members in o.inputs.items():
        lines.append(f"- {up}: " + (", ".join(f"{m['key']} ({m['status']})" for m in members) or "none"))
        for d in o.dropped.get(up, []):
            if d["why"] != "not nearest":
                lines.append(f"  - dropped {d['key']}: {d['why']}")
    for u in o.units[:20]:
        for r in c.links:
            if u.get(c.unit_id) in (r.get("from"), r.get("to")):
                lines.append(f"- link: {r.get('from')} --{r.get('kind')}--> {r.get('to')}"
                             + (f" ({r['source']})" if r.get("source") else ""))
    if not o.inputs:
        lines.append("None: this stage reads the registry and the archive only.")
    lines += ["", "## Decisions", ""]
    for name, d in o.decisions.items():
        menu = st["decisions"][name]
        options = ", ".join(option_names(menu))
        who = "you may change it" if menu.get("authority", "human") == "agent" else "a person approves a change"
        lines.append(f"- `{name}` = `{d['value']}` ({d['source']}). Options: {options}; {who}."
                     + (f" {menu['doc']}" if menu.get("doc") else ""))
    if not o.decisions:
        lines.append("None: this stage has no choices to make.")
    notes = read_notes(c, o.units, o.stage, sealed)
    lines += ["", "## Notes", ""]
    for n in notes:
        src = f" (source: {n['meta']['source']})" if n["meta"].get("source") else ""
        lines += [f"### {n['file']}{src}", "", n["body"], ""]
    if not notes:
        lines.append("None apply.")
    lines += ["", "## History", ""]
    for a in o.attempts:
        bad = [k for k, v in (a.get("checks") or {}).items() if v == "fail"]
        lines.append(f"- attempt {a['n']} by {a['by']}: {a['status']}"
                     + (f"; {a['problem']}" if a.get("problem") else "") + (f"; failing: {', '.join(bad)}" if bad else ""))
    for e in c.docket:
        if e.get("stage") == o.stage and e.get("key") in (o.key_str, None) and e["kind"] != "resolve":
            lines.append(f"- docket {e['id']} ({e['kind']}, {c.state(e)}) by {e['by']}: {e['why']}")
    if lines[-1] == "":
        lines.append("No attempts yet.")
    lines += ["", "## What you can do", ""]
    if o.decisions:
        lines.append(f"- `mandate decide {o.stage} {o.key_str} NAME=OPTION --why ...`")
    lines += [f"- `mandate exclude COLUMN=VALUE --stage {o.stage} --why ...`",
              f"- `mandate request {o.stage} {o.key_str} --why ...`", ""]
    return "\n".join(lines)


def collect(c: Campaign, source: str | os.PathLike) -> int:
    """Copy order and attempt records from run folders (or an rsync of just those files) into the ledger."""
    n = 0
    for path in sorted(pathlib.Path(source).glob("*/*/*.json")):
        if path.name != "order.json" and not fnmatch.fnmatch(path.name, "attempt-*.json"):
            continue
        stage, order_id = path.parent.parent.name, path.parent.name
        if stage not in c.stages or ".prev-" in order_id:
            continue
        target = c.root / "ledger" / stage / order_id / path.name
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        record = json.loads(path.read_text())
        if path.name == "order.json":
            record = {**for_ledger(c, record), "issued": now()}
        target.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        n += 1
    return n


def slurm_script(c: Campaign, stage: str, orders: list[Order]) -> pathlib.Path:
    """Write a job-array script for these orders. It is not submitted: a person does that."""
    for o in orders:
        issue(c, o)
    folder = c.runs_root / "_slurm"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = now().replace(":", "").replace("-", "")
    listing = folder / f"{stage}-{stamp}.orders"
    listing.write_text("".join(f"{o.ref}\n" for o in orders))
    header = [f"#SBATCH --job-name=mandate-{stage}", f"#SBATCH --array=1-{len(orders)}",
              f"#SBATCH --output={folder}/{stage}-%A_%a.out"]
    header += [f"#SBATCH {line}" for line in c.stages[stage].get("slurm") or []]
    body = [f'mandate -C "{c.root}" run "$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "{listing}")"']
    script = folder / f"{stage}-{stamp}.sh"
    script.write_text("\n".join(["#!/bin/bash", *header, "", *(c.plan.get("slurm_setup") or []), *body]) + "\n")
    return script
