"""Gather every local fit of each contest dataset and pick the best.

Used by scripts/contest_advice.py (the dashboard). NumPy only; no JAX.

    candidates, skipped = scan(results_dir, tasks, jobs)   # cheap: scalars and logs only
    chosen, rejected = select(candidates[task], n=4)       # loads images in rank order

A *candidate* is one fitted image with its provenance:
    task, method, commit (the results directory, named for the virgil commit),
    job (Slurm job id, "" if unknown), raw (raw χ²/N on the quoted errors),
    scale (fitted error scale, None where the run reports none), kept (members
    surviving a sweep's cuts, None for a single fit), informed (method chosen
    after seeing the truth), path (the npz), kind (how to read its image).

Ranking is by raw χ²/N, lowest first, never by an error-rescaled value.
Sanity cuts, applied in `select`: image finite, no more than a sliver of
negative flux, positive total flux (normalized to unit sum on use), and a
sweep must rest on at least ``MIN_KEPT`` surviving members. Candidates with no
recorded raw χ²/N (the early L-curve runs without a log line) are skipped and
counted, not guessed.

Scanned: every results directory except benchmark phantoms (``bench*``),
logs, smoke runs, ensemble products (means, not fits) and diagnostics.
"""

import json
import pathlib
import re

import numpy as np

MIN_KEPT = 3
MAX_NEG_FRACTION = 0.01  # of the summed absolute flux
SKIP_DIRS = re.compile(r"bench|logs|smoke|ensemble", re.I)
SKIP_STEMS = re.compile(r"diag|ensemble|smoke|_paper_field")
NUM = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"


def find_task(stem, tasks):
    """The longest task label that ``stem`` equals or extends with '_', else None."""
    best = None
    for t in tasks:
        if (stem == t or stem.startswith(t + "_")) and (best is None or len(t) > len(best)):
            best = t
    return best


def method_of(task, stem, commit):
    rest = stem[len(task):].strip("_")
    member = re.fullmatch(r"m(\d+)", rest)
    if member:
        return f"round-1 campaign member m{member.group(1)}"
    return rest[: -len("_r2")] if rest.endswith("_r2") else rest or "round-1 single run"


def parse_log(path):
    """(raw χ²/N, error scale, informed) read from a run's text log."""
    if not path.exists():
        return None, None, False
    text = path.read_text(errors="replace")
    raw = re.search(r"raw chi2/N=" + NUM, text)
    scale = re.search(r"error_scale=" + NUM, text)
    return (float(raw.group(1)) if raw else None, float(scale.group(1)) if scale else None,
            "provenance=informed" in text)


def meta_of(path, task, commit, job):
    """Scalar metadata of one npz (no image read), or None if unusable."""
    npz = np.load(path, allow_pickle=True)
    keys = set(npz.files)
    raw, scale, informed = parse_log(path.with_suffix(".txt"))
    kept = None
    if "ref_image" in keys:
        kind = "campaign"
        raw = raw if raw is not None else float(npz["best_chi2_red"])
        scale = float(npz["error_scale"])
    elif "mean" in keys and "kept" in keys:
        kind = "sweep"
        if raw is None and "raw_chi2_red" in keys:
            raw = float(np.mean(npz["raw_chi2_red"]))
        k = npz["kept"]
        kept = len(json.loads(str(k))) if k.dtype.kind in "US" or k.dtype == object else len(k)
    elif "image_iso" in keys:
        kind = "gp"
    elif "images" in keys:
        kind = "lcurve"
    elif "image" in keys:
        kind = "image"
    else:
        return None
    if "informed" in keys:
        informed = informed or bool(npz["informed"])
    stem = path.stem
    return {"task": task, "stem": stem, "method": method_of(task, stem, commit), "commit": commit, "job": job,
            "raw": raw, "scale": scale, "kept": kept, "informed": informed, "path": path, "kind": kind}


def scan(results, tasks, jobs=None):
    """({task: [candidate metadata]}, {reason: count}) over ``results``.
    ``jobs`` maps a results sub-path ('8562a39/campaign') or a stem to a job id."""
    results, jobs, out, skipped = pathlib.Path(results), jobs or {}, {}, {}

    def skip(why):
        skipped[why] = skipped.get(why, 0) + 1

    for path in sorted(results.rglob("*.npz")):
        rel = path.relative_to(results)
        if any(SKIP_DIRS.search(part) for part in rel.parts[:-1]) or SKIP_STEMS.search(path.stem):
            continue
        task = find_task(path.stem, tasks)
        if task is None:
            continue
        commit, sub = rel.parts[0], "/".join(rel.parts[:-1])
        job = jobs.get(path.stem) or jobs.get(sub) or ""
        try:
            m = meta_of(path, task, commit, job)
        except Exception:  # an unreadable npz is a skipped candidate, not a crash
            m = None
        if m is None:
            skip("unreadable or no image")
        elif m["raw"] is None or not np.isfinite(m["raw"]):
            skip("no recorded raw χ²/N")
        else:
            out.setdefault(task, []).append(m)
    return out, skipped


def load_image(m):
    """(2-D float image, pixel scale in mas) of a candidate; image as stored
    (column 0 East, row 0 North, centre at (n − 1)/2)."""
    d = np.load(m["path"], allow_pickle=True)
    kind = m["kind"]
    if kind == "campaign":
        im, fov = d["ref_image"], float(d["ref_fov"])
        im = np.asarray(im, float)
        return im, fov / im.shape[0]
    pix = float(d["pixel_scale_mas"])
    if kind == "sweep":
        im = d["mean"]
    elif kind == "gp":
        aniso = "image_aniso" in d.files and "log_z_aniso" in d.files and np.nanmax(d["log_z_aniso"]) > np.nanmax(d["log_z_iso"])
        im = d["image_aniso"] if aniso else d["image_iso"]
    elif kind == "lcurve":
        index = json.loads(str(d["index"])) if "index" in d.files else {}
        i = index.get("discrepancy")
        i = index.get("corner", -1) if i is None else i
        im = d["images"][i]
    else:
        im = d["image"]
    im = np.asarray(im, float)
    while im.ndim > 2:
        im = im[0] if im.shape[0] <= im.shape[-1] else im.mean(0)
    return im, pix


def sane(im):
    """None if ``im`` passes the sanity cuts, else the reason it fails."""
    if im.ndim != 2 or not np.isfinite(im).all():
        return "not finite"
    total = np.abs(im).sum()
    if im.sum() <= 0 or total == 0:
        return "no positive flux"
    if -im[im < 0].sum() > MAX_NEG_FRACTION * total:
        return "negative flux"
    return None


def select(cands, n=4):
    """(best + up to n−1 runners-up with images attached, rejections). Walks
    ``cands`` from lowest raw χ²/N; one entry per distinct image path."""
    chosen, rejected = [], []
    for m in sorted(cands, key=lambda c: c["raw"]):
        if m["kept"] is not None and m["kept"] < MIN_KEPT:
            rejected.append((m, f"sweep rests on {m['kept']} kept members (< {MIN_KEPT})"))
            continue
        try:
            im, pix = load_image(m)
        except Exception as e:  # noqa: BLE001
            rejected.append((m, f"unreadable image ({type(e).__name__})"))
            continue
        why = sane(im)
        if why:
            rejected.append((m, why))
            continue
        chosen.append({**m, "image": im / im.sum(), "pix": pix})
        if len(chosen) == n:
            break
    return chosen, rejected


# --- the recorded inputs ("prompt") of a dataset's fits --------------------------


def task_entries(source):
    """[(label, is_fstring, comment text, code text)] for the TASKS list in the
    source of scripts/contest_images.py: the comment lines directly above each
    entry, and the entry's own lines, both verbatim."""
    block = source.split("\nTASKS = [", 1)[1].split("\n]\n", 1)[0]
    entries, comments, current = [], [], None
    for line in block.splitlines():
        stripped = line.strip()
        start = re.match(r'(?:\*\[)?dict\(label=(f?)"([^"]+)"', stripped)
        if stripped.startswith("#"):
            if current is not None:
                current = None
            comments.append(stripped.lstrip("#").strip())
        elif start:
            current = [start.group(2), bool(start.group(1)), "\n".join(comments), line.rstrip()]
            entries.append(current)
            comments = []
        elif current is not None and stripped:
            current[3] += "\n" + line.rstrip()
    return [tuple(e) for e in entries]


def task_text(entries, task):
    """The TASKS comment and code for ``task`` (an f-string label matches by its prefix), or ''."""
    for label, is_f, comment, code in entries:
        if label == task or (is_f and task.startswith(label.split("{")[0])):
            return (comment + "\n\n" if comment else "") + code
    return ""


def commands_for(commands_text, task):
    """The round-2 command lines (tab-separated 'label<TAB>command') of ``task``'s fits, verbatim."""
    return "\n".join(line.split("\t", 1)[1] if "\t" in line else line
                     for line in commands_text.splitlines() if line.startswith(task + "_"))


def prompt_text(year, name, task, presub, sources, task_src, advice, plan, commands):
    """The box under a section header: every recorded input for the dataset,
    each part labelled with where it came from, or 'no recorded prompt'."""
    parts = []
    if presub:
        parts.append(f"[contests/manifest.yml, contests.{year}.presubmission (the whole {year} contest)]\n{presub}")
    if sources:
        parts.append(f"[~/data/imaging_contests/{year}/pre_submission/SOURCES.txt: the files contestants had]\n{sources.strip()}")
    if task_src:
        parts.append(f"[scripts/contest_images.py, TASKS entry for {task}: the settings the fitting script is given]\n{task_src}")
    if advice:
        parts.append(f"[contests/fit_advice.csv, row {year} / {name}: Ben's advice, verbatim]\n{advice.strip()}")
    if plan:
        parts.append(f"[contests/fit_plan.yml, round-2 entry for {task}: the brief that was expanded into commands]\n{plan.strip()}")
    if commands:
        parts.append(f"[ozstar_scripts advice/commands.txt: the round-2 command lines run for {task}]\n{commands}")
    if not parts:
        return "no recorded prompt"
    head = ("The fits are run by scripts, not from a chat prompt; below is every recorded input about this dataset, "
            "as it stands in the files named in brackets.")
    return head + "\n\n" + "\n\n".join(parts)
