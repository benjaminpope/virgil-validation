"""A local dashboard for writing fitting advice per contest dataset.

    PYTHONPATH=src python scripts/contest_advice.py            # opens the page
    python scripts/contest_advice.py --port 8765 --no-open

One card per row of contests/gallery.yml (imaged or not): our best-fitting
image over every local fit (scripts/contest_best.py: lowest raw χ²/N on the
quoted errors after sanity cuts) with up to three runners-up, each labelled
with method, job, commit, raw χ²/N and error scale (an "informed" badge where
the method was chosen after seeing the truth); truth-based scores where a
truth exists; a text box for the advice; a box under each year's header with
the recorded inputs of the fits; and a collapsible contestant brief. The
truth and published panels (prev/next, arrow keys) appear only with
``--private``.
Edits autosave (debounced) to contests/fit_advice.csv, and are reloaded on
start. The server binds 127.0.0.1 only and serves images only from under the
data directory (~/data/imaging_contests): the copyrighted paper crops are
never copied into the repository. Needs numpy/matplotlib/yaml (as
contest_gallery.py); no JAX.

contests/fit_advice.csv (UTF-8, written with the csv module; text only)
    year    contest year, e.g. 2004
    label   campaign label (`<label>_m<k>.npz`), empty for a dataset not imaged
    name    the dataset's name in contests/gallery.yml (with year, identifies it)
    advice  free text; may contain newlines and commas
    updated UTC ISO timestamp of the last edit
Only rows with non-empty advice are written. A reader should key on
(year, name) and treat the text as instructions for that dataset's next fit.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import pathlib
import re
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "contests" / "fit_advice.csv"
FIELDS = ["year", "label", "name", "advice", "updated"]
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]*")
BRIEF = ["presubmission", "targets", "instrument", "notes"]
LOCK = threading.Lock()


# --- CSV -------------------------------------------------------------------


def load_advice(path=CSV_PATH):
    """{(year, label, name): {advice, updated}} from the CSV; {} if absent."""
    path = pathlib.Path(path)
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {(r["year"], r["label"], r["name"]): {"advice": r["advice"], "updated": r["updated"]}
                for r in csv.DictReader(f)}


def save_advice(store, path=CSV_PATH):
    """Write ``store`` atomically, skipping empty advice."""
    path = pathlib.Path(path)
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS, lineterminator="\n")
        w.writeheader()
        for (year, label, name), v in sorted(store.items()):
            if v["advice"].strip():
                w.writerow({"year": year, "label": label, "name": name, **v})
    tmp.replace(path)


# --- safe file serving -----------------------------------------------------


def safe_path(roots, rel):
    """The file at ``rel`` under one of ``roots`` ([(prefix, dir)]), or None.
    ``rel`` is '<prefix>/<name>[/<name>]' with plain names only; the resolved
    path must stay inside its directory (no '..', absolute paths, symlink
    escapes)."""
    parts = rel.split("/")
    if len(parts) < 2 or not all(SAFE_NAME.fullmatch(p) and ".." not in p for p in parts):
        return None
    base = dict(roots).get(parts[0])
    if base is None:
        return None
    base = pathlib.Path(base).resolve()
    target = base.joinpath(*parts[1:]).resolve()
    if not target.is_relative_to(base) or not target.is_file() or target.suffix != ".png":
        return None
    return target


# --- dataset rows ----------------------------------------------------------

N_SHOWN = 4  # the best image plus up to three runners-up
ROUND2_JOBS = {"commands": "logs/advice_commands_20261013_161259.txt", "jobs": ["18577328", "18577336"]}
CAMPAIGN_JOBS = {"8562a39/campaign": "18073823", "3018a9f/campaign": "18077616"}
SCRATCH_BENCH = ("/private/tmp/claude-501/-Users-benpope-code-virgil/0bf3a5ea-04c0-4809-9af7-6027ff0d6fce/"
                 "scratchpad/score_composite/score_composite.md")


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def round2_jobs(data, done_dir):
    """{label: Slurm job id} of the completed round-2 runs, from the commands
    file and the jobs' .done lists (local files only); {} if they are absent."""
    commands = data / "results" / ROUND2_JOBS["commands"]
    if not commands.exists():
        return {}
    labels = [line.split("\t")[0] for line in commands.read_text().splitlines() if "\t" in line]
    out = {}
    for job in ROUND2_JOBS["jobs"]:
        done = pathlib.Path(done_dir) / f"{job}.done"
        if not done.exists():
            continue
        for line in done.read_text().splitlines():
            if "|" in line and line.split("|")[1].strip() == "COMPLETED":
                i = int(line.split("|")[0].split("_")[1])
                if i < len(labels):
                    out[labels[i]] = job
    return out


def bench_best(text):
    """[{family, arm, n, raw, ncc, lawson, nfail}]: per family of a
    score_composite.md, the arm with the lowest median Lawson σ/peak."""
    out, family = [], None
    for line in text.splitlines():
        head = re.match(r"## (\w+)\s+\(noise scatter", line)
        if head:
            family, best = head.group(1), None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if family and line.startswith("|") and len(cells) >= 6 and re.fullmatch(r"\d+", cells[1]):
            row = {"family": family, "arm": cells[0], "n": int(cells[1]), "raw": float(cells[2]), "ncc": float(cells[3]),
                   "lawson": float(cells[4]), "nfail": cells[5]}
            same = [r for r in out if r["family"] == family]
            if not same:
                out.append(row)
            elif row["lawson"] < same[0]["lawson"]:
                out[out.index(same[0])] = row
    return out


def render_set(chosen, cg, half, cache, task):
    """Write one PNG per chosen candidate: recentred on its centroid, on a
    common field, unit flux, one asinh stretch for the dataset (to the best
    image's peak; brighter pixels clip). Returns the file names."""
    centred = [cg.centre_on_centroid(c["image"])[0] for c in chosen]
    fovs = [c["pix"] * c["image"].shape[0] for c in chosen]
    peak = float(np.clip(cg.on_grid(centred[0], fovs[0], half), 0, None).max()) or 1.0
    names = []
    for i, c in enumerate(chosen):
        key = hashlib.sha1(f"{c['path']}|{c['path'].stat().st_mtime_ns}|{half}|{peak}".encode()).hexdigest()[:8]
        name = f"best_{task}_{i}_{key}.png"
        if not (cache / name).exists():
            cg.save_png(cg.render_panel(centred[i], fovs[i], half, "best" if i == 0 else f"runner-up {i}", peak=peak), cache / name)
        names.append(name)
    return names


def score_2004(cg, c):
    s = cg._load_script("score_2004")
    f = s.FITTED
    ref = s.truth_image(f["ratio"], f["u_ld"], f["spot_contrast"])
    box = np.zeros_like(ref, bool)
    rows, cols = np.nonzero(ref > 0)
    box[rows.min() - 10 : rows.max() + 11, cols.min() - 10 : cols.max() + 11] = True
    fov = c["pix"] * c["image"].shape[0]
    moved, _ = s.aligned(s.resample(cg.centre_on_centroid(c["image"])[0], fov, s.NPIX + 2 * s.MAX_SHIFT), ref)
    return f"σ/peak {s.sigma_over_peak(moved, ref, box)[0]:.3f} vs the rebuilt truth (published BSMEM 0.116, WISARD 0.163; lower is better)"


def score_2018(cg, c):
    im, _ = cg.centre_on_centroid(c["image"])
    n, pix = im.shape[0], c["pix"]
    yy, xx = np.indices(im.shape)
    east, north = -(xx - (n - 1) / 2) * pix, -(yy - (n - 1) / 2) * pix  # column 0 is East, row 0 North
    (px, py), _ = cg.marker_positions_2018()
    flux = float(im[np.hypot(east - px, north - py) < 0.6].sum() / im.sum())
    r = np.hypot(east, north)
    w = im * ((r > 1.5) & (r < 12))
    sxx, syy, sxy = (w * east**2).sum(), (w * north**2).sum(), (w * east * north).sum()
    pa = float(np.degrees(0.5 * np.arctan2(2 * sxy, syy - sxx)) % 180)
    return f"planet flux fraction {flux:.3f} within 0.6 mas (truth ~0.02); disk PA {pa:.0f}° (truth 44.2°)"


TRUTH_SCORES = {"score_2004": score_2004, "markers_2018": score_2018}


def build_rows(data, gallery, figures, manifest, cg, advice, cb=None, plan=None, private=False, jobs=None,
               results=None, commands="", task_src=""):
    """The page's data: one dict per gallery row. ``cg`` is contest_gallery,
    ``cb`` contest_best, ``plan`` fit_plan.yml's content."""
    cb = cb or load_module("contest_best")
    results = pathlib.Path(results or data / "results")
    cache = data / "advice_cache"
    cache.mkdir(parents=True, exist_ok=True)
    plan_by = {(d["year"], d["name"]): d for d in (plan or {}).get("datasets", [])}
    entries = cb.task_entries(task_src) if task_src else []
    task_of = {(y, r["name"]): r.get("label") or (plan_by.get((y, r["name"])) or {}).get("task")
               for y, b in gallery["years"].items() for r in b["rows"]}
    tasks = sorted({t for t in task_of.values() if t} | {d.get("task") for d in plan_by.values() if d.get("task")})
    cands, skipped = cb.scan(results, tasks, jobs)
    rows, skipped_total = [], skipped
    for year, block in gallery["years"].items():
        paper = figures["papers"].get(year) or {}
        brief = {k: (manifest["contests"].get(year) or {}).get(k) for k in BRIEF}
        pre = data / str(year) / "pre_submission" / "SOURCES.txt"
        for row in block["rows"]:
            task = task_of[(year, row["name"])]
            panels = []
            if private:
                for p in cg.row_panels(paper, row):
                    if (data / "figures" / str(paper.get("sha256", "")) / f"{p['id']}.png").exists():
                        panels.append({"url": f"/crop/{paper['sha256']}/{p['id']}.png", "caption": cg.panel_ref(p),
                                       "role": p["role"], "orientation": p.get("orientation") or ""})
            shown, rejected = (cb.select(cands[task], N_SHOWN) if task in cands else ([], []))
            images = []
            if shown:
                half = float(row["half"]) if row.get("half") else cg.nice(1.1 * max(
                    cg.enclosing_radius(cg.centre_on_centroid(shown[0]["image"])[0], shown[0]["pix"] * shown[0]["image"].shape[0]), 1e-3))
                names = render_set(shown, cg, half, cache, task)
                score = TRUTH_SCORES.get(row["truth"])
                for c, name in zip(shown, names):
                    images.append({"url": f"/cache/{name}", "method": c["method"], "job": c["job"], "commit": c["commit"],
                                   "raw": c["raw"], "scale": c["scale"], "kept": c["kept"], "informed": c["informed"],
                                   "score": score(cg, c) if score else "", "file": f"{c['path'].parent.name}/{c['path'].name}"})
            key = (year, row.get("label") or "", row["name"])
            adv = advice.get(key, {}).get("advice", "")
            pe = plan_by.get((year, row["name"]), {})
            plan_text = ""
            if pe:
                plan_text = (yaml.safe_dump({"advice": pe.get("advice"), "brief": pe.get("brief"), "questions": pe.get("questions")},
                                            sort_keys=False, allow_unicode=True, width=100))
            prompt = cb.prompt_text(
                year, row["name"], task or "(no task yet)", (manifest["contests"].get(year) or {}).get("presubmission"),
                pre.read_text() if pre.exists() else "", cb.task_text(entries, task) if task else "", adv if pe or adv else "",
                plan_text, cb.commands_for(commands, task) if task else "")
            rows.append({"year": year, "title": block["title"], "label": row.get("label") or "", "name": row["name"],
                         "target": row["target"], "task": task or "", "panels": panels, "images": images,
                         "rejected": [f"{m['stem']} ({m['commit']}): {why}" for m, why in rejected],
                         "prompt": prompt, "brief": brief, "advice": adv,
                         "info": "" if images else "No fit with a recorded raw χ²/N yet."})
    return rows, skipped_total


# --- server ----------------------------------------------------------------


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="text/plain; charset=utf-8"):
            body = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _local(self):  # guard against DNS rebinding
            return self.headers.get("Host", "").split(":")[0] in ("127.0.0.1", "localhost")

        def do_GET(self):
            if not self._local():
                return self._send(403, "forbidden")
            path = self.path.split("?")[0]
            if path == "/":
                return self._send(200, PAGE, "text/html; charset=utf-8")
            if path == "/api/rows":
                meta = {k: state[k] for k in ("bench", "skipped", "pending", "private")}
                return self._send(200, json.dumps({"rows": state["rows"], **meta}), "application/json")
            target = safe_path(state["roots"], path.lstrip("/"))
            if target is None:
                return self._send(404, "not found")
            return self._send(200, target.read_bytes(), "image/png")

        def do_POST(self):
            if not self._local() or self.path != "/api/advice":
                return self._send(403, "forbidden")
            try:
                n = int(self.headers.get("Content-Length", 0))
                msg = json.loads(self.rfile.read(min(n, 1_000_000)))
                key = (str(msg["year"]), str(msg["label"]), str(msg["name"]))
                text = str(msg["advice"])
            except (ValueError, KeyError, TypeError):
                return self._send(400, "bad request")
            if key not in state["keys"]:
                return self._send(404, "unknown dataset")
            with LOCK:
                state["store"][key] = {"advice": text, "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                save_advice(state["store"], state["csv"])
            return self._send(200, state["store"][key]["updated"])

    return Handler


def make_state(data, csv_path=CSV_PATH, private=False, bench_md=None, done_dir=None):
    cg, cb = load_module("contest_gallery"), load_module("contest_best")
    load = lambda n: yaml.safe_load((ROOT / "contests" / n).read_text())
    store = load_advice(csv_path)
    done_dir = pathlib.Path(done_dir or "~/code/ozstar_scripts-advice/results/contest_imaging").expanduser()
    jobs = {**CAMPAIGN_JOBS, **round2_jobs(data, done_dir)}
    cmd_file = pathlib.Path("~/code/ozstar_scripts-advice/scripts/contest_imaging/advice/commands.txt").expanduser()
    rows, skipped = build_rows(
        data, load("gallery.yml"), load("figures.yml"), load("manifest.yml"), cg, store, cb, load("fit_plan.yml"), private,
        jobs, commands=cmd_file.read_text() if cmd_file.exists() else "",
        task_src=(ROOT / "scripts" / "contest_images.py").read_text())
    md = pathlib.Path(bench_md or SCRATCH_BENCH)
    bench = bench_best(md.read_text()) if md.exists() else []
    roots = [("img", data / "gallery_private" / "docs" / "assets" / "img"), ("cache", data / "advice_cache")]
    if private:
        roots.append(("crop", data / "figures"))
    return {"rows": rows, "store": store, "csv": csv_path, "roots": roots, "bench": bench, "skipped": skipped,
            "pending": PENDING, "private": private,
            "keys": {(r["year"], r["label"], r["name"]) for r in rows}}


PENDING = ["2004 parametric ladder, job 18588496: submitted, no .done file and no results downloaded (the local "
           "39f9399/parametric_smoke run is a smoke test and is not used)",
           "seed reruns, ozstar_scripts-advice/logs/advice_r2_rerun3_20261015_1020.log: empty, no job ids; nothing to gather"]


def main():
    parser = argparse.ArgumentParser(description="Dashboard for fitting advice per contest dataset.")
    parser.add_argument("--data", default="~/data/imaging_contests")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--private", action="store_true",
                        help="also show the truth and published panels (copyrighted crops, local viewing only)")
    parser.add_argument("--bench-md", default=None, help="score_composite.md for the benchmark table")
    args = parser.parse_args()
    state = make_state(pathlib.Path(args.data).expanduser().resolve(), private=args.private, bench_md=args.bench_md)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"{len(state['rows'])} datasets; advice saves to {state['csv']}\nserving {url} (Ctrl-C to stop)")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


PAGE = """<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fit advice</title>
<style>
:root { color-scheme: dark; --bg:#14161a; --card:#1e2127; --line:#30343c; --fg:#e4e6ea; --dim:#9aa0aa; --acc:#e0a04a; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.45 system-ui,sans-serif; }
header { position:sticky; top:0; z-index:2; background:var(--bg); border-bottom:1px solid var(--line); padding:.6rem 1rem; display:flex; gap:1rem; align-items:baseline; }
header h1 { font-size:1.1rem; margin:0; } #status { color:var(--dim); font-size:.85rem; }
main { max-width:1200px; margin:0 auto; padding:0 1rem 4rem; }
h2 { margin:2rem 0 .5rem; font-size:1.05rem; color:var(--acc); }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:.8rem 1rem; margin:.8rem 0; }
.card h3 { margin:0 0 .1rem; font-size:1rem; } .target { color:var(--dim); font-size:.85rem; margin:0 0 .6rem; }
.cols { display:grid; grid-template-columns:1fr 1fr 1.2fr; gap:.8rem; }
@media (max-width:900px) { .cols { grid-template-columns:1fr; } }
.box { background:#0d0e11; border:1px solid var(--line); border-radius:8px; min-height:14rem; display:flex; flex-direction:column; align-items:center; justify-content:center; padding:.4rem; }
.box img { max-width:100%; max-height:20rem; image-rendering:auto; }
.ph { color:var(--dim); } .cap { font-size:.78rem; color:var(--dim); text-align:center; margin-top:.3rem; }
.nav { display:flex; gap:.5rem; align-items:center; margin-top:.3rem; font-size:.8rem; color:var(--dim); }
button { background:var(--line); color:var(--fg); border:0; border-radius:5px; padding:.15rem .6rem; cursor:pointer; }
.info { font-size:.8rem; color:var(--dim); margin:.4rem 0 0; }
textarea { width:100%; height:100%; min-height:16rem; box-sizing:border-box; background:#0d0e11; color:var(--fg); border:1px solid var(--line); border-radius:8px; padding:.5rem; font:inherit; resize:vertical; }
textarea:focus { outline:1px solid var(--acc); }
details { margin-top:.6rem; font-size:.85rem; color:var(--dim); } summary { cursor:pointer; }
details dt { color:var(--fg); margin-top:.3rem; } details dd { margin:0; white-space:pre-wrap; }
.saved { color:#6fbf73; }
pre.prompt { background:#0d0e11; border:1px solid var(--line); border-radius:8px; padding:.5rem .7rem; margin:.2rem 0 .7rem; max-height:11rem; overflow:auto; resize:vertical; white-space:pre-wrap; font:12px/1.4 ui-monospace,monospace; color:var(--dim); }
.cands { display:grid; grid-template-columns:1fr 1fr; gap:.5rem; }
.cand { background:#0d0e11; border:1px solid var(--line); border-radius:8px; padding:.3rem; font-size:.74rem; color:var(--dim); }
.cand.best { border-color:var(--acc); } .cand img { width:100%; display:block; }
.cand b { color:var(--fg); } .badge { background:#5a3d12; color:#f2c27a; border-radius:4px; padding:0 .3rem; margin-left:.3rem; }
.score { color:#7fc8e8; } table.bench { border-collapse:collapse; font-size:.85rem; } table.bench td, table.bench th { border:1px solid var(--line); padding:.2rem .6rem; text-align:right; } table.bench td:first-child, table.bench th:first-child, table.bench td:nth-child(2) { text-align:left; }
</style>
<header><h1>Fit advice</h1><span id="status">loading…</span><span style="color:var(--dim);font-size:.8rem">← → cycle the panel under the pointer</span></header>
<main id="main"></main>
<script>
const $ = (t, c, x) => { const e = document.createElement(t); if (c) e.className = c; if (x != null) e.textContent = x; return e; };
const status = document.getElementById("status");
let current = null;
function brief(b) {
  const d = $("details"); d.append($("summary", null, "Contestant brief"));
  const dl = $("dl");
  for (const k of ["presubmission", "targets", "instrument", "notes"]) if (b[k]) {
    dl.append($("dt", null, k)); dl.append($("dd", null, typeof b[k] === "string" ? b[k] : Object.entries(b[k]).map(([a, v]) => a + ": " + v).join("\\n")));
  }
  d.append(dl); return d;
}
const f3 = x => (x == null ? "–" : x >= 1000 ? x.toExponential(2) : Number(x.toPrecision(3)).toString());
function cand(c, i) {
  const d = $("div", "cand" + (i ? "" : " best")); const im = $("img"); im.src = c.url; im.loading = "lazy"; d.append(im);
  const h = $("div"); h.append($("b", null, (i ? "runner-up " + i : "best") + ": raw χ²/N " + f3(c.raw)));
  if (c.informed) h.append($("span", "badge", "informed"));
  d.append(h, $("div", null, "error scale " + (c.scale == null ? "none reported" : c.scale.toFixed(2)) + (c.kept != null ? "; sweep, " + c.kept + " members kept" : "")));
  d.append($("div", null, c.method + " · job " + (c.job || "?") + " · virgil " + c.commit));
  if (c.score) d.append($("div", "score", c.score));
  d.title = c.file; return d;
}
function card(r) {
  const c = $("section", "card");
  c.append($("h3", null, r.year + " " + r.name + (r.images.length ? "" : " (no fit yet)")), $("p", "target", r.target));
  const pre = $("pre", "prompt", r.prompt); c.append(pre);
  const cols = $("div", "cols"), left = $("div"), right = $("div"), ta = $("textarea");
  cols.style.gridTemplateColumns = r.panels.length ? "1fr 1.6fr 1fr" : "1.6fr 1fr";
  const box = $("div", "box"), cap = $("div", "cap"), nav = $("div", "nav");
  let i = 0;
  const show = () => {
    box.replaceChildren();
    if (!r.panels.length) return;
    const p = r.panels[i], im = $("img"); im.src = p.url; box.append(im);
    cap.textContent = p.caption + (p.orientation ? " — " + p.orientation : ""); pos.textContent = (i + 1) + " / " + r.panels.length;
  };
  const step = d => { i = (i + d + r.panels.length) % r.panels.length; show(); };
  const prev = $("button", null, "‹ prev"), next = $("button", null, "next ›"), pos = $("span");
  prev.onclick = () => step(-1); next.onclick = () => step(1); nav.append(prev, pos, next);
  nav.hidden = r.panels.length < 2;
  c.step = step; c.addEventListener("mouseenter", () => current = c);
  left.append(box, cap, nav);
  if (r.images.length) { const g = $("div", "cands"); r.images.forEach((x, k) => g.append(cand(x, k))); right.append(g); }
  else right.append($("div", "box", r.info));
  if (r.rejected.length) { const d = $("details"); d.append($("summary", null, r.rejected.length + " candidates rejected"), $("pre", "prompt", r.rejected.join("\\n"))); right.append(d); }
  ta.value = r.advice; ta.placeholder = "Advice for the fit…";
  let t; const mark = $("span", "info");
  ta.oninput = () => { clearTimeout(t); status.textContent = "unsaved…"; mark.textContent = ""; t = setTimeout(async () => {
    const res = await fetch("/api/advice", { method: "POST", body: JSON.stringify({ year: r.year, label: r.label, name: r.name, advice: ta.value }) });
    status.textContent = res.ok ? "saved " + new Date().toLocaleTimeString() : "SAVE FAILED"; status.className = res.ok ? "saved" : ""; mark.textContent = res.ok ? "saved" : "save failed";
  }, 600); };
  const mid = $("div"); mid.append(ta, mark);
  if (r.panels.length) cols.append(left);
  cols.append(right, mid); c.append(cols, brief(r.brief)); show(); return c;
}
document.addEventListener("keydown", e => {
  if (!current || e.target.tagName === "TEXTAREA") return;
  if (e.key === "ArrowLeft") current.step(-1); else if (e.key === "ArrowRight") current.step(1);
});
fetch("/api/rows").then(r => r.json()).then(d => {
  const m = document.getElementById("main"); let y = null;
  m.append($("p", "info", "Best = lowest raw χ²/N on the quoted errors across every local fit (round 1 campaign, round 2 advice runs), after sanity cuts (finite, no negative flux, a sweep needs at least 3 surviving members). Images are recentred on their centroids and share one stretch per dataset (to the best image's peak). " + (d.private ? "Truth and published panels shown (--private)." : "Truth and published panels hidden (start with --private to see them).")));
  for (const r of d.rows) { if (r.year !== y) { y = r.year; m.append($("h2", null, r.year + ": " + r.title)); } m.append(card(r)); }
  m.append($("h2", null, "Benchmark (synthetic phantoms, not contest data): best arm per family"));
  if (d.bench.length) {
    const t = $("table", "bench"); t.append($("tr")); t.firstChild.append(...["family", "arm", "n", "median raw χ²/N", "median NCC", "median Lawson σ/peak", "n fail"].map(h => $("th", null, h)));
    for (const b of d.bench) { const tr = $("tr"); tr.append(...[b.family, b.arm, b.n, b.raw, b.ncc, b.lawson, b.nfail].map(v => $("td", null, v))); t.append(tr); }
    m.append(t, $("p", "info", "From score_composite.md (job 18591640): lowest median Lawson score in each family."));
  } else m.append($("p", "info", "score_composite.md not found (pass --bench-md)."));
  m.append($("h2", null, "Pending and skipped"));
  const ul = $("ul"); d.pending.forEach(p => ul.append($("li", null, "pending: " + p)));
  Object.entries(d.skipped).forEach(([k, n]) => ul.append($("li", null, n + " result files skipped: " + k)));
  m.append(ul);
  status.textContent = d.rows.length + " datasets";
});
</script>
"""

if __name__ == "__main__":
    main()
