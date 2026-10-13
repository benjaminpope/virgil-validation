"""The fit plan seeded by Ben's contest advice (contests/fit_plan.yml).

    python scripts/contest_fit_plan.py            # check, then list arms
    python scripts/contest_fit_plan.py --commands # the runnable arms' commands
    python scripts/contest_fit_plan.py --needs    # features the other arms wait for
    python scripts/contest_fit_plan.py --commands --oz > commands.txt
                                                  # the OzSTAR advice job's lines
    python scripts/contest_fit_plan.py --commands --oz --round 2 > commands.txt
                                                  # round 2's lines (labels _r2), from the plan's briefs; notes on stderr

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


# Round 2: a `brief:` per dataset entry, expanded generically (the keys are
# documented in contests/fit_plan.yml). Nothing here names a dataset.
BRIEF_KEYS = ("scene", "prior", "systematics", "bands", "field", "sweep", "prune", "needs", "departures")
SCENE_PARTS = ("point", "ud", "ld_ellipse", "gauss_halo", "image")
BAND_MODES = ("grey", "per_band", "per_channel_grey_default", "joint_sparco")
SYSTEMATICS = ("none", "block_scales", "gains+block_scales", "gains+closure_offsets")
CLEAN_MS_SCALES = "0,1,2"  # CLEAN component widths in beam minor axes: point, one beam, two (contest_images' own example)
SWEEP_GROUPS = 16  # virgil.ensemble groups: four families drawn uniformly, so ~4 L-curves (12 weights each) per family
# prior name -> (contest_images flags, path): gp and gp2 have their own fits; mem is the regularised L-curve
PRIORS = {"gp": (["--prior", "gp"], "gp"), "gp2": (["--prior", "gp2"], "gp2"), "mem": (["--prior", "mem"], "mem"),
          "tsv": (["--regulariser", "tsv"], "mem"), "tv": (["--regulariser", "tv"], "mem"),
          "starlet": (["--regulariser", "starlet"], "mem"), "logsum": (["--regulariser", "logsum"], "mem"),
          "clean_ms": (["--prior", "clean", "--clean-scales", CLEAN_MS_SCALES], "clean"),
          "mem_prune": (["--prior", "mem", "--init", "clean", "--prune"], "mem")}
BRIEF_PRIORS = tuple(k for k in PRIORS if k != "mem_prune")


def sweep_off(brief):
    """The reason a brief switches its sweep off (None: sweep on). YAML 1.1 reads an unquoted `off:` key as False."""
    sweep = brief.get("sweep")
    return None if sweep is None else sweep.get("off") or sweep.get(False) if isinstance(sweep, dict) else ""


def brief_entries(plan):
    """The dataset entries that carry a brief, in plan order."""
    return [e for e in plan["datasets"] if e.get("brief")]


def scene_alternatives(brief):
    """The brief's scene as a list of part lists (a flat list is one)."""
    scene = brief.get("scene") or []
    return scene if scene and isinstance(scene[0], list) else [scene]


def band_runs(plan, entry):
    """[(mode, tasks)] for the brief's bands: tasks is the list one line fits
    (the entry's task; the group's tasks, in the group's names order, for
    per_band, one run per task, and joint_sparco, one joint run)."""
    modes = entry["brief"].get("bands", "grey")
    group = entry.get("group")
    names = plan["groups"][group]["names"] if group else []
    by_name = {e["name"]: e["task"] for e in plan["datasets"]}
    band_tasks = [by_name[n] for n in names] or [entry["task"]]
    joint_tasks = band_tasks if group and plan["groups"][group].get("joint") else [entry["task"]]
    runs = []
    for mode in modes if isinstance(modes, list) else [modes]:
        if mode == "per_band":
            runs += [(mode, [t]) for t in band_tasks]
        elif mode == "joint_sparco":
            runs.append((mode, joint_tasks))
        else:
            runs.append((mode, [entry["task"]]))
    return runs


def check_briefs(plan, indices=None):
    """Problems with the briefs: only the documented keys and values, a task
    and scene wherever lines are wanted, band tasks that exist (given ``indices``)."""
    problems = []
    for e in brief_entries(plan):
        b, key = e["brief"], f"{e['year']} {e['name']}"
        bad = sorted(set(b) - set(BRIEF_KEYS))
        if bad:
            problems.append(f"{key}: unknown brief keys {bad}")
        if not b.get("prior"):
            if set(b) - {"needs", "departures"}:
                problems.append(f"{key}: a brief with scene or bands needs a prior list")
            continue
        if not e.get("task"):
            problems.append(f"{key}: a brief with priors needs a task")
            continue
        if not b.get("scene"):
            problems.append(f"{key}: no scene")
        for p in b["prior"]:
            if p not in BRIEF_PRIORS:
                problems.append(f"{key}: prior {p!r} not in {BRIEF_PRIORS}")
        for alt in scene_alternatives(b):
            for part in alt:
                if part.split(":")[0] not in SCENE_PARTS + ("points", "hole"):
                    problems.append(f"{key}: scene part {part!r}")
        if b.get("systematics", "none") not in SYSTEMATICS:
            problems.append(f"{key}: systematics {b.get('systematics')!r}")
        modes = b.get("bands", "grey")
        for mode in modes if isinstance(modes, list) else [modes]:
            if mode not in BAND_MODES:
                problems.append(f"{key}: bands {mode!r}")
        field = b.get("field", "data")
        if field != "data" and not (isinstance(field, dict) and {"mas", "source"} <= set(field)):
            problems.append(f"{key}: field is `data` or {{mas, source}}")
        if b.get("sweep") is not None and not sweep_off(b):
            problems.append(f"{key}: sweep is omitted or {{off: reason}}")
        if indices is not None:
            for _, tasks in band_runs(plan, e):
                problems += [f"{key}: no such contest_images.py task {t!r}" for t in tasks if t not in indices]
    return problems


def _scene_flags(parts, path, sparco, notes, where):
    """(--scene text, K points, hole flags) for one scene alternative on one
    prior's ``path`` ("gp", "gp2", "mem", "clean"), with a note for each
    reduction; None if the scene cannot be fitted at all."""
    parts = list(parts)
    named = [p for p in parts if p.split(":")[0] not in ("points", "hole")]
    k = sum(int(p.split(":")[1]) for p in parts if p.startswith("points:"))
    holes = [p for p in parts if p.split(":")[0] == "hole"]
    if "ud" in named and ("point" in named or "ld_ellipse" in named):
        notes.append(("dropped", f"{where}: ud cannot be combined with point or ld_ellipse (scene {parts})"))
        return None
    if "image" not in named:
        notes.append(("dropped", f"{where}: scene {parts} has no image, and contest_images needs one"))
        notes.append(("needs", f"{where}: parametric-only scene (no image) fitted and reported like an image arm"))
        return None
    if "gauss_halo" in named and path in ("gp2", "clean"):
        named.remove("gauss_halo")
        notes.append(("dropped", f"{where}: gauss_halo needs --prior gp or mem; removed from the scene"))
    if k and path in ("gp2", "clean"):
        notes.append(("dropped", f"{where}: --points is not taken by the {path} path; removed"))
        k = 0
    if sparco and not any(p in named for p in ("point", "ud", "ld_ellipse")):
        named.insert(0, "point")
        notes.append(("note", f"{where}: --sparco needs a star; point added to the scene"))
    hole = []
    if holes:
        mas = holes[0].partition(":")[2]
        if mas:
            hole = ["--hole", mas]
        elif not any(p in named for p in ("point", "ud", "ld_ellipse")):
            notes.append(("dropped", f"{where}: bare hole needs a star (its default is half the beam's minor axis); removed"))
    return ",".join(named), k, hole


def expand_brief(plan, entry, indices):
    """(lines, notes) for one entry's brief: ``lines`` are (label, argv) with
    argv the contest_images.py arguments after the script name; ``notes`` are
    (kind, text), kind in dropped / needs / departures / note."""
    b, notes = entry["brief"], []
    key = f"{entry['year']} {entry['name']}"
    for kind in ("needs", "departures"):
        notes += [(kind, f"{key}: {text}") for text in b.get(kind) or []]
    if not b.get("prior"):
        return [], notes
    field = b.get("field", "data")
    field_flags = [] if field == "data" else ["--field", f"{field['mas']:g}"]
    systematics = b.get("systematics", "none")
    alts = scene_alternatives(b)
    priors = list(b["prior"]) + (["mem_prune"] if b.get("prune") else [])
    lines, seen = [], set()
    runs = band_runs(plan, entry)
    for mode, tasks in runs:
        if mode == "per_channel_grey_default":
            notes.append(("needs", f"{key}: per-channel grey default images are not built; grey lines emitted"))
        sparco = mode == "joint_sparco"
        task = tasks[0]
        where_task = ["--task", str(indices[task])] + (["--joint", ",".join(tasks)] if len(tasks) > 1 else [])
        for ai, alt in enumerate(alts):
            for prior in priors:
                if sparco and prior == "mem_prune":
                    continue
                flags, path = PRIORS[prior]
                where = f"{key} [{task} {prior}{' sparco' if sparco else ''}]"
                sc = _scene_flags(alt, path, sparco, notes, where)
                if sc is None:
                    continue
                scene_text, k, hole = sc
                use_sparco = sparco
                if sparco and path in ("gp2", "clean"):
                    use_sparco = False
                    notes.append(("dropped", f"{where}: --sparco needs --prior gp or mem; dropped"))
                sys_here = systematics
                if systematics != "none" and path != "gp":
                    sys_here = "none"
                    notes.append(("dropped", f"{where}: --systematics {systematics} applies to --prior gp only; dropped"))
                argv = [*where_task, "--scene", scene_text, *flags, *(["--points", str(k)] if k else []), *hole,
                        *(["--systematics", sys_here] if sys_here != "none" else []), *field_flags,
                        *(["--sparco"] if use_sparco else [])]
                arm = "_".join([prior, *(["-".join(p.replace(":", "") for p in alt)] if len(alts) > 1 else []),
                                *(["sparco"] if sparco else [])])
                if tuple(argv) in seen:
                    notes.append(("dropped", f"{where}: duplicates an earlier line after reductions"))
                    continue
                seen.add(tuple(argv))
                lines.append((f"{task}_{arm}_r2", argv))
        if sweep_off(b) is None:
            first = alts[0]
            star_part = ["point"] if "point" in first else []
            if any(p in first for p in ("ud", "ld_ellipse")):
                notes.append(("departures", f"{key} [{task}]: sweep fits virgil's point-star scene; the {[p for p in first if p in ('ud', 'ld_ellipse')][0]} star is dropped"))
            if sparco:
                notes.append(("departures", f"{key} [{task}]: sweep takes no --sparco; the joint fit's bands run without it"))
            if field != "data":
                notes.append(("departures", f"{key} [{task}]: sweep grows its own field and takes no --field"))
            argv = [*where_task, "--scene", ",".join([*star_part, "image"]), "--sweep", str(SWEEP_GROUPS)]
            if tuple(argv) not in seen:
                seen.add(tuple(argv))
                lines.append((f"{task}_sweep_r2", argv))
    if sweep_off(b):
        notes.append(("note", f"{key}: sweep off: {sweep_off(b)}"))
    return lines, notes


def round2_lines(plan, indices):
    """(lines, notes) for every brief: lines are (label, argv)."""
    lines, notes = [], []
    for e in brief_entries(plan):
        ls, ns = expand_brief(plan, e, indices)
        lines += ls
        notes += ns
    return lines, notes


def oz_lines_round2(plan, indices, notes_out=None):
    """Round 2's ``<label>\t<script> <args>`` lines: every brief expanded,
    labelled <task>_<arm>_r2 and marked --informed (the advice chose the
    method after the truths were seen). ``notes_out``, a list, receives the notes."""
    lines, notes = round2_lines(plan, indices)
    if notes_out is not None:
        notes_out += notes
    out = []
    for label, argv in lines:
        args = shlex.join(["scripts/contest_images.py", *argv, "--informed", "--label", label])
        out.append(f'{label}\t{args} --data "$VV_DATA" --out "$OUT"')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--commands", action="store_true")
    parser.add_argument("--needs", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="add --smoke to contest_images.py commands")
    parser.add_argument("--oz", action="store_true", help="with --commands: the OzSTAR advice job's commands.txt lines")
    parser.add_argument("--round", type=int, default=1, choices=(1, 2), help="with --commands --oz: which round's arms")
    args = parser.parse_args()
    plan = load_plan()
    problems = check(plan, load_advice_rows()) + check_briefs(plan)
    for p in problems:
        print("PROBLEM:", p, file=sys.stderr)
    if args.commands and args.oz:
        indices = task_indices()
        if args.round == 2:
            problems += check_briefs(plan, indices)
            for p in problems:
                print("PROBLEM:", p, file=sys.stderr)
            notes = []
            lines = oz_lines_round2(plan, indices, notes)
            for kind, text in dict.fromkeys(notes):
                print(f"{kind.upper()}: {text}", file=sys.stderr)
        else:
            lines = oz_lines(plan, indices)
        print("\n".join(lines))
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
