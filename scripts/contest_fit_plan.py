"""The fit plan seeded by Ben's contest advice (contests/fit_plan.yml).

    python scripts/contest_fit_plan.py            # check, then list arms
    python scripts/contest_fit_plan.py --commands # the runnable arms' commands
    python scripts/contest_fit_plan.py --needs    # features the other arms wait for
    python scripts/contest_fit_plan.py --commands --oz > commands.txt
                                                  # the OzSTAR advice job's lines
    python scripts/contest_fit_plan.py --commands --oz --round 2 > commands.txt
                                                  # round 2's lines (labels _r2)

Every row of contests/fit_advice.csv (scripts/contest_advice.py) must have
an entry in the plan, matched on year and name, with its advice quoted
verbatim; `check` lists what is missing or stale. Each arm either runs with
existing code (`run`) or names what it `needs`.
"""

import argparse
import csv
import json
import pathlib
import shlex
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "contests" / "fit_plan.yml"
ADVICE_PATH = ROOT / "contests" / "fit_advice.csv"
RUN_KINDS = ("cli", "settings", "script")


def load_plan(path=PLAN_PATH):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_advice_rows(path=ADVICE_PATH):
    """The advice CSV's rows (dicts), blank advice dropped."""
    with open(path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["advice"].strip()]


def entries(plan):
    """{(year, name): entry}."""
    return {(str(e["year"]), e["name"]): e for e in plan["datasets"]}


def check(plan, rows):
    """Problems as strings; [] when the plan covers the advice exactly."""
    problems = []
    by_key = entries(plan)
    if len(by_key) != len(plan["datasets"]):
        problems.append("duplicate (year, name) entries in the plan")
    keys = {(r["year"], r["name"]) for r in rows}
    for r in rows:
        e = by_key.get((r["year"], r["name"]))
        if e is None:
            problems.append(f"no plan entry for advice row {r['year']} {r['name']!r}")
        elif e["advice"] != r["advice"]:
            problems.append(f"{r['year']} {r['name']!r}: plan advice differs from the CSV")
    for key, e in by_key.items():
        if key not in keys:
            problems.append(f"plan entry {key} has no advice row")
        if e.get("group") and e["name"] not in plan["groups"].get(e["group"], {}).get("names", []):
            problems.append(f"{key}: not listed in group {e['group']!r}")
        if not e.get("arms"):
            problems.append(f"{key}: no arms")
        for arm in e.get("arms", []):
            run, needs = arm.get("run"), arm.get("needs")
            if (run is None) == (needs is None):
                problems.append(f"{key} arm {arm.get('id')!r}: give exactly one of run and needs")
            elif run is not None and (len(run) != 1 or next(iter(run)) not in RUN_KINDS):
                problems.append(f"{key} arm {arm.get('id')!r}: run must be one of {RUN_KINDS}")
    for g, spec in plan["groups"].items():
        for name in spec["names"]:
            if not any(e.get("group") == g and e["name"] == name for e in plan["datasets"]):
                problems.append(f"group {g!r} lists {name!r}, which has no entry in it")
    return problems


def task_indices():
    """{label: index} from contest_images.py's TASKS (imports JAX)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("contest_images", ROOT / "scripts" / "contest_images.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {t["label"]: i for i, t in enumerate(module.TASKS)}


def commands(plan, indices, data="~/data/imaging_contests", out="contest_images/advice", smoke=False):
    """(entry, arm, shell command) for every runnable arm, one per (task, arm);
    ``smoke`` gives contest_images.py runs its seconds-long settings."""
    seen, result = set(), []
    for e in plan["datasets"]:
        for arm in e["arms"]:
            run = arm.get("run")
            if run is None or (e["task"], arm["id"]) in seen:
                continue
            seen.add((e["task"], arm["id"]))
            if "script" in run:
                cmd = run["script"].replace("OUT", f"{out}/{e['task']}_{arm['id']}")
            else:
                extra = (run["cli"] if "cli" in run else
                         ["--prior", "gp", "--settings", json.dumps(run["settings"]), "--label", f"{e['task']}_{arm['id']}"])
                cmd = shlex.join(["python", "scripts/contest_images.py", "--task", str(indices[e["task"]]),
                                  *extra, "--data", data, "--out", out, *(["--smoke"] if smoke else [])])
            result.append((e, arm, cmd))
    return result


def oz_lines(plan, indices):
    """``<label>\\t<script> <args>`` for the OzSTAR advice job (ozstar_scripts
    contest_imaging/advice.sbatch, one array task per line): contest_images.py
    arms only (script arms have jobs of their own), with "$VV_DATA" and "$OUT"
    left for the job to expand."""
    lines = []
    for e, arm, _ in commands(plan, indices):
        run, label = arm["run"], f"{e['task']}_{arm['id']}"
        if "script" in run:
            continue
        extra = run["cli"] if "cli" in run else [
            "--prior", "gp", "--settings", json.dumps(run["settings"]), "--label", label]
        args = shlex.join(["scripts/contest_images.py", "--task", str(indices[e["task"]]), *extra])
        lines.append(f'{label}\t{args} --data "$VV_DATA" --out "$OUT"')
    return lines


def round2_arms(plan):
    """(task, arm) for every round-2 arm, in plan order."""
    return [(task, arm) for task, arms in (plan.get("round2") or {}).items() for arm in arms]


def check_round2(plan, indices=None):
    """Problems with the round2 section: each arm has an id and exactly one
    of cli / settings / needs (cli may accompany settings), ids are unique
    per task, and (given ``indices``) every task exists."""
    problems, seen = [], set()
    for task, arm in round2_arms(plan):
        key = (task, arm.get("id"))
        if key in seen:
            problems.append(f"round2 {task}: duplicate arm {arm.get('id')!r}")
        seen.add(key)
        kinds = [k for k in ("settings", "needs") if k in arm] or (["cli"] if "cli" in arm else [])
        if len(kinds) != 1 or ("needs" in arm and "cli" in arm):
            problems.append(f"round2 {task} arm {arm.get('id')!r}: give cli, settings (+cli) or needs")
        if indices is not None and task not in indices:
            problems.append(f"round2 {task}: no such contest_images.py task")
    return problems


def oz_lines_round2(plan, indices):
    """Round 2's ``<label>\t<script> <args>`` lines: every runnable arm,
    labelled <task>_<id>_r2 and marked --informed (the advice chose the
    method after the truths were seen)."""
    lines = []
    for task, arm in round2_arms(plan):
        if "needs" in arm:
            continue
        label = f"{task}_{arm['id']}_r2"
        extra = [str(a) for a in arm.get("cli", [])]
        if "settings" in arm:
            extra = ["--prior", "gp", "--settings", json.dumps(arm["settings"]), *extra]
        args = shlex.join(["scripts/contest_images.py", "--task", str(indices[task]), *extra,
                           "--informed", "--label", label])
        lines.append(f'{label}\t{args} --data "$VV_DATA" --out "$OUT"')
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--commands", action="store_true")
    parser.add_argument("--needs", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="add --smoke to contest_images.py commands")
    parser.add_argument("--oz", action="store_true", help="with --commands: the OzSTAR advice job's commands.txt lines")
    parser.add_argument("--round", type=int, default=1, choices=(1, 2), help="with --commands --oz: which round's arms")
    args = parser.parse_args()
    plan = load_plan()
    problems = check(plan, load_advice_rows()) + check_round2(plan)
    for p in problems:
        print("PROBLEM:", p, file=sys.stderr)
    if args.commands and args.oz:
        indices = task_indices()
        if args.round == 2:
            problems += check_round2(plan, indices)
            for p in problems:
                print("PROBLEM:", p, file=sys.stderr)
        print("\n".join(oz_lines_round2(plan, indices) if args.round == 2 else oz_lines(plan, indices)))
    elif args.commands:
        for e, arm, cmd in commands(plan, task_indices(), smoke=args.smoke):
            print(f"# {e['year']} {e['name']}: {arm['id']}\n{cmd}")
    elif args.needs:
        features = {}
        for e in plan["datasets"]:
            for arm in e["arms"]:
                needs = arm.get("needs")
                for n in [needs] if isinstance(needs, str) else needs or []:
                    features.setdefault(n, set()).add(e["task"] or e["name"])
        for n, tasks in sorted(features.items(), key=lambda kv: -len(kv[1])):
            print(f"{len(tasks):2d}  {n}: {', '.join(sorted(tasks))}")
    else:
        for e in plan["datasets"]:
            arms = ", ".join(a["id"] + ("" if "run" in a else "*") for a in e["arms"])
            print(f"{e['year']}  {e['name']:32s} {arms}")
        print("(* needs a new feature: --needs)")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
