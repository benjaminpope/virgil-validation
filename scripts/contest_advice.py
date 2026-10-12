"""A local dashboard for writing fitting advice per contest dataset.

    PYTHONPATH=src python scripts/contest_advice.py            # opens the page
    python scripts/contest_advice.py --port 8765 --no-open

One card per row of contests/gallery.yml (imaged or not): the truth or
published panels (prev/next, arrow keys) on the left, our best-fit virgil
image on the right with its raw χ²/N and error scale, a text box for the
advice, and a collapsible contestant brief from contests/manifest.yml.
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
import importlib.util
import json
import pathlib
import re
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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


def build_rows(data, gallery, figures, manifest, cg, advice):
    """The page's data: one dict per gallery row. ``cg`` is contest_gallery."""
    results = [data / "results" / r for r in gallery["campaign"]["results"]]
    img_dir, cache = data / "gallery_private" / "docs" / "assets" / "img", data / "advice_cache"
    rows = []
    for year, block in gallery["years"].items():
        paper = figures["papers"].get(year) or {}
        brief = {k: (manifest["contests"].get(year) or {}).get(k) for k in BRIEF}
        for row in block["rows"]:
            label = row.get("label") or ""
            panels = []
            for p in cg.row_panels(paper, row):
                if (data / "figures" / str(paper.get("sha256", "")) / f"{p['id']}.png").exists():
                    panels.append({"url": f"/crop/{paper['sha256']}/{p['id']}.png", "caption": cg.panel_ref(p),
                                   "role": p["role"], "orientation": p.get("orientation") or ""})
            ours, info, group = None, "Not imaged yet.", (cg.members(results, label) if label else [])
            if group:
                best, kept = cg.best_member(group)
                info = cg.chi2_line(best, kept, len(group))
                if (img_dir / f"{label}_virgil.png").exists():
                    ours = f"/img/{label}_virgil.png"
                else:
                    cache.mkdir(parents=True, exist_ok=True)
                    truth = None
                    if row["truth"] == "score_2004":
                        ref, ref_fov, _ = cg.truth_2004_data2(best["image"], best["fov"])
                        truth = (ref, ref_fov)
                    half = cg.field(row, best, truth)
                    cg.save_png(cg.render_panel(best["image"], best["fov"], half, "virgil"), cache / f"{label}_virgil.png")
                    ours = f"/cache/{label}_virgil.png"
            key = (year, label, row["name"])
            rows.append({"year": year, "title": block["title"], "label": label, "name": row["name"],
                         "target": row["target"], "panels": panels, "ours": ours, "info": info, "brief": brief,
                         "advice": advice.get(key, {}).get("advice", "")})
    return rows


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
                return self._send(200, json.dumps(state["rows"]), "application/json")
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


def make_state(data, csv_path=CSV_PATH):
    spec = importlib.util.spec_from_file_location("contest_gallery", ROOT / "scripts" / "contest_gallery.py")
    cg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cg)
    load = lambda n: yaml.safe_load((ROOT / "contests" / n).read_text())
    store = load_advice(csv_path)
    rows = build_rows(data, load("gallery.yml"), load("figures.yml"), load("manifest.yml"), cg, store)
    roots = [("crop", data / "figures"), ("img", data / "gallery_private" / "docs" / "assets" / "img"),
             ("cache", data / "advice_cache")]
    return {"rows": rows, "store": store, "csv": csv_path, "roots": roots,
            "keys": {(r["year"], r["label"], r["name"]) for r in rows}}


def main():
    parser = argparse.ArgumentParser(description="Dashboard for fitting advice per contest dataset.")
    parser.add_argument("--data", default="~/data/imaging_contests")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    state = make_state(pathlib.Path(args.data).expanduser().resolve())
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
function card(r) {
  const c = $("section", "card");
  c.append($("h3", null, r.name + (r.label ? "" : " (not imaged)")), $("p", "target", r.target));
  const cols = $("div", "cols"), left = $("div"), right = $("div"), ta = $("textarea");
  const box = $("div", "box"), cap = $("div", "cap"), nav = $("div", "nav");
  let i = 0;
  const show = () => {
    box.replaceChildren();
    if (!r.panels.length) { box.append($("div", "ph", "No truth or published panel")); cap.textContent = ""; nav.hidden = true; return; }
    const p = r.panels[i], im = $("img"); im.src = p.url; box.append(im);
    cap.textContent = p.caption + (p.orientation ? " — " + p.orientation : ""); pos.textContent = (i + 1) + " / " + r.panels.length;
  };
  const step = d => { i = (i + d + r.panels.length) % r.panels.length; show(); };
  const prev = $("button", null, "‹ prev"), next = $("button", null, "next ›"), pos = $("span");
  prev.onclick = () => step(-1); next.onclick = () => step(1); nav.append(prev, pos, next);
  nav.hidden = r.panels.length < 2;
  c.step = step; c.addEventListener("mouseenter", () => current = c);
  left.append(box, cap, nav);
  const obox = $("div", "box");
  if (r.ours) { const im = $("img"); im.src = r.ours; obox.append(im); } else obox.append($("div", "ph", "Not imaged yet"));
  right.append(obox, $("p", "info", r.info));
  ta.value = r.advice; ta.placeholder = "Advice for the fit…";
  let t; const mark = $("span", "info");
  ta.oninput = () => { clearTimeout(t); status.textContent = "unsaved…"; mark.textContent = ""; t = setTimeout(async () => {
    const res = await fetch("/api/advice", { method: "POST", body: JSON.stringify({ year: r.year, label: r.label, name: r.name, advice: ta.value }) });
    status.textContent = res.ok ? "saved " + new Date().toLocaleTimeString() : "SAVE FAILED"; status.className = res.ok ? "saved" : ""; mark.textContent = res.ok ? "saved" : "save failed";
  }, 600); };
  const mid = $("div"); mid.append(ta, mark);
  cols.append(left, right, mid); c.append(cols, brief(r.brief)); show(); return c;
}
document.addEventListener("keydown", e => {
  if (!current || e.target.tagName === "TEXTAREA") return;
  if (e.key === "ArrowLeft") current.step(-1); else if (e.key === "ArrowRight") current.step(1);
});
fetch("/api/rows").then(r => r.json()).then(rows => {
  const m = document.getElementById("main"); let y = null;
  for (const r of rows) { if (r.year !== y) { y = r.year; m.append($("h2", null, r.year + ": " + r.title)); } m.append(card(r)); }
  status.textContent = rows.length + " datasets";
});
</script>
"""

if __name__ == "__main__":
    main()
