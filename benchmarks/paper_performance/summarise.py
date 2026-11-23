"""Rows -> CSV, LaTeX tabular, three-panel figure; ``--check`` validates the schema and the coverage (WP-R).

    BENCH_PYTHON summarise.py [--raw results/raw] [--out results]     writes results.csv, performance_table.tex,
                                                                      performance.pdf / .png, summary.json
    BENCH_PYTHON summarise.py --check [--raw ...]                     schema + every case has rows for both codes

Reports raw numbers: per group (comparison, case, task, config, threads, virgil method) the median and min-max of the warm
times pooled over processes, the median cold time, peak footprint, flags and status. Ratios are other / virgil with the range
implied by the min-max of both (``common.ratio_with_spread``). The counts of groups in which each code is faster are
computed from the pairs that have a time for both codes (warm and cold separately). Nothing is selected or dropped
because of its value; failures, timeouts and gate failures are printed as such.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

OTHER = {"binary": "pmoired", "orbit": "octofitter", "imaging": "ehtim"}
OTHER_LABEL = {"binary": "PMOIRED", "orbit": "Octofitter", "imaging": "eht-imaging"}
C_V, C_O = "#0072B2", "#E69F00"      # colour-blind-safe pair; one colour per code in all panels
LIMITS = [
    "one CPU (run r1: Apple-silicon laptop; run r2: one 64-core EPYC 7543 node); 1 thread primary, 2 threads at most; no code was run on all cores",
    "no GPU row exists (CPU only)",
    "eht-imaging with the direct transform (NFFT not installed), the same transform in virgil",
    "Octofitter interferometry on simulated point-source data only",
    "O-A-real is period-informed (prior from the published period) and is not scored",
    "PMOIRED and Octofitter parallel modes not run beyond 2 workers",
]


# ----------------------------------------------------------------------------------------------------------------
# grouping
# ----------------------------------------------------------------------------------------------------------------

def load(raw):
    rows = []
    for p in sorted(Path(raw).glob("*.json")):
        r = json.loads(p.read_text())
        if "comparison" in r and "code" in r:       # skips compare_*.json and other non-row files
            r["_file"] = p.name
            rows.append(r)
    return rows


def ident(r):
    s = r["settings"]
    unit = s.get("case_id") or r["case"]
    task = s.get("task") or r["config"]
    if task.startswith("orbit-"):
        task = r["config"]
    variant = s.get("method") if (r["comparison"] == "imaging" and r["code"] == "virgil") else None
    return (r["comparison"], unit, task, r["config"], r["threads"]), variant


def med(v):
    v = [x for x in v if x is not None]
    return st.median(v) if v else None


def run_rank(r):
    """r1 (the first matrix) < r2 < ...; rows of a group's latest run replace the earlier ones (earlier rows stay in raw/)."""
    try:
        return int(str(r["settings"].get("run", "r1")).lstrip("r"))
    except ValueError:
        return 1


def fresh(r):
    """Like-for-like cold: import + load + init + first call (the whole of a new session before warm work)."""
    if r.get("t_cold") is None:
        return None
    return (r.get("t_import") or 0.0) + (r.get("t_load") or 0.0) + (r.get("t_init") or 0.0) + r["t_cold"]


def group(rows):
    g, pop = defaultdict(list), defaultdict(list)
    for r in rows:
        key, variant = ident(r)
        (pop if r["settings"].get("jax_cache") == "populated" else g)[(key, r["code"], variant)].append(r)
    out = {}
    for (key, code, variant), rs in g.items():
        latest = max(run_rank(r) for r in rs)
        rs = [r for r in rs if run_rank(r) == latest]
        ok = [r for r in rs if r["status"] == "ok"]
        warm = [x for r in ok for x in r["t_warm"]]
        flags = sorted({f for r in rs for f in r["flags"]})
        statuses = Counter(r["status"] for r in rs)
        status = "ok" if ok else statuses.most_common(1)[0][0]
        first = (ok or rs)[0]
        init = [r["t_init"] for r in ok if r["t_init"] is not None]
        cold = [r["t_cold"] for r in ok if r["t_cold"] is not None]
        tot = [(r["t_init"] or 0) + r["t_cold"] for r in ok if r["t_cold"] is not None]
        fr = [fresh(r) for r in ok if fresh(r) is not None]
        pr_ = [r for r in pop.get((key, code, variant), []) if r["status"] == "ok" and run_rank(r) == latest]
        pw = [x for r in pr_ for x in r["t_warm"]]
        out[(key, code, variant)] = {
            "key": key, "code": code, "variant": variant, "status": status, "n_proc": len(rs), "n_ok": len(ok),
            "run": f"r{latest}",
            "t_cold": med(cold), "t_init": med(init), "t_total_cold": med(tot),
            "t_fresh": med(fr), "fresh_range": (min(fr), max(fr)) if fr else (None, None),
            "cache": {"t_fresh": med([fresh(r) for r in pr_]), "t_cold": med([r["t_cold"] for r in pr_]),
                      "t_warm": med(pw), "n": len(pr_)} if pr_ else None,
            "warm": {"median": med(warm), "min": min(warm) if warm else None, "max": max(warm) if warm else None},
            "cold_range": (min(cold), max(cold)) if cold else (None, None),
            "t_import": med([r["t_import"] for r in ok]), "t_process": med([r["t_process"] for r in ok]),
            "peak_mb": med([r["peak_footprint_mb"] for r in ok]), "baseline_mb": med([r["baseline_mb"] for r in ok]),
            "flags": flags, "n_data": first["n_data"], "n_param": first["n_param"], "size": first["size"],
            "reason": (first["settings"].get("reason") or first["accuracy"].get("reason") or "") if status != "ok" else "",
            "rows": ok or rs,
        }
    return out


def pairs(groups):
    """(virgil group, other group) with the same key; the default-method imaging row pairs with the same ehtim row."""
    by_key = defaultdict(dict)
    for (key, code, variant), g in groups.items():
        by_key[key].setdefault("v" if code == "virgil" else "o", []).append(g)
    res = []
    for key, d in by_key.items():
        for v in d.get("v", []):
            for o in d.get("o", []):
                res.append((key, v, o))
    return res


def ratios(v, o, basis):
    if basis == "warm":
        return common.ratio_with_spread(o["warm"], v["warm"])
    if basis == "fresh":
        mk = lambda g: {"median": g["t_fresh"], "min": g["fresh_range"][0], "max": g["fresh_range"][1]}  # noqa: E731
    else:
        mk = lambda g: {"median": g["t_cold"], "min": g["cold_range"][0], "max": g["cold_range"][1]}  # noqa: E731
    return common.ratio_with_spread(mk(o), mk(v))


# ----------------------------------------------------------------------------------------------------------------
# imaging time to precision (compare.time_to_precision, loaded by path: its sibling modules share names with other dirs)
# ----------------------------------------------------------------------------------------------------------------

def _imaging_compare():
    spec_dir = HERE / "imaging"
    sys.path.insert(0, str(spec_dir))
    try:
        spec = importlib.util.spec_from_file_location("imaging_compare", spec_dir / "compare.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(spec_dir))
    return mod


def imaging_ttp(groups):
    """Per imaging case: time to each precision rung for virgil (L-BFGS-B), virgil (default method) and eht-imaging."""
    mod = _imaging_compare()
    out = {}
    for key in {k for (k, c, v) in groups if k[0] == "imaging" and k[2] == "ladder"}:
        routes = {}
        for (k, code, variant), g in groups.items():
            if k != key or g["status"] != "ok" or not g["rows"]:
                continue
            if code == "virgil" and variant in (None, "lbfgsb"):
                routes["virgil"] = g["rows"]
            elif code == "virgil":
                routes["virgil_lm"] = g["rows"]
            elif code == "ehtim":
                routes["ehtim"] = g["rows"]
        if "ehtim" in routes and "virgil" in routes:
            try:
                out[key] = mod.time_to_precision(routes)
            except Exception as e:  # recorded, not hidden
                out[key] = {"error": repr(e)}
    return out


def ttp_pairs(ttp):
    """[(case key, route, warm ratio, cold ratio, gap)] other / virgil at the headline rung, for both virgil routes."""
    res = []
    for key, t in sorted(ttp.items()):
        gap = t.get("headline_gap")
        if gap is None:
            continue
        o = t["routes"]["ehtim"]["reached"][str(gap)]
        for route in ("virgil", "virgil_lm"):
            if route not in t["routes"]:
                continue
            v = t["routes"][route]["reached"][str(gap)]
            if v is None:
                continue
            res.append((key, route, o["time"] / v["time"], (o["cold_time"] / v["cold_time"]) if o["cold_time"] and v["cold_time"] else None, gap))
    return res


# ----------------------------------------------------------------------------------------------------------------
# CSV
# ----------------------------------------------------------------------------------------------------------------

CSV_COLS = ["comparison", "case", "task", "config", "threads", "code", "variant", "status", "n_proc", "n_ok", "n_data",
            "n_param", "t_cold_median", "t_cold_min", "t_cold_max", "t_warm_median", "t_warm_min", "t_warm_max",
            "t_init_median", "t_total_cold_median", "t_fresh_median", "t_fresh_min", "t_fresh_max", "t_import_median",
            "t_cold_cache_populated", "t_fresh_cache_populated", "t_warm_cache_populated", "run", "counted", "ratio_fresh",
            "ratio_fresh_lo", "ratio_fresh_hi", "t_process_median", "peak_footprint_mb",
            "baseline_mb", "ratio_warm", "ratio_warm_lo", "ratio_warm_hi", "ratio_cold", "ratio_cold_lo",
            "ratio_cold_hi", "flags", "reason"]


def csv_rows(groups):
    pr = {}
    for key, v, o in pairs(groups):
        pr[(key, v["variant"])] = (ratios(v, o, "warm"), ratios(v, o, "cold"), ratios(v, o, "fresh"))
    rows = []
    for (key, code, variant), g in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] != "virgil", kv[0][2] or "")):
        rw, rc, rf = pr.get((key, variant), ({}, {}, {})) if code == "virgil" else ({}, {}, {})
        ca = g["cache"] or {}
        rows.append({
            "comparison": key[0], "case": key[1], "task": key[2], "config": key[3], "threads": key[4], "code": code,
            "variant": variant or "", "status": g["status"], "n_proc": g["n_proc"], "n_ok": g["n_ok"],
            "n_data": g["n_data"], "n_param": g["n_param"], "t_cold_median": g["t_cold"],
            "t_cold_min": g["cold_range"][0], "t_cold_max": g["cold_range"][1], "t_warm_median": g["warm"]["median"],
            "t_warm_min": g["warm"]["min"], "t_warm_max": g["warm"]["max"], "t_init_median": g["t_init"],
            "t_total_cold_median": g["t_total_cold"], "t_fresh_median": g["t_fresh"], "t_fresh_min": g["fresh_range"][0],
            "t_fresh_max": g["fresh_range"][1], "t_import_median": g["t_import"],
            "t_cold_cache_populated": ca.get("t_cold"), "t_fresh_cache_populated": ca.get("t_fresh"),
            "t_warm_cache_populated": ca.get("t_warm"), "run": g["run"], "counted": counted(key),
            "ratio_fresh": rf.get("ratio"), "ratio_fresh_lo": rf.get("lo"), "ratio_fresh_hi": rf.get("hi"),
            "t_process_median": g["t_process"], "peak_footprint_mb": g["peak_mb"], "baseline_mb": g["baseline_mb"],
            "ratio_warm": rw.get("ratio"), "ratio_warm_lo": rw.get("lo"), "ratio_warm_hi": rw.get("hi"),
            "ratio_cold": rc.get("ratio"), "ratio_cold_lo": rc.get("lo"), "ratio_cold_hi": rc.get("hi"),
            "flags": ";".join(g["flags"]), "reason": g["reason"].replace("\n", " ")[:200],
        })
    return rows


# Imaging rows that compare unequal work are diagnostics, not counted: the ladder is compared as time to precision
# (``imaging_ttp``), stop_default pairs different stopping rules, and native pairs virgil's single self-sized grid with
# eht-imaging on a fixed grid (only the nearest grid, 48 px against 41 px, is counted, with the truth error beside it).
# Everything else compares the same work in both codes.
COUNTED_NATIVE = ("I-sim-48", "I-sim-ring")


def counted(key):
    comp, case, task = key[0], key[1], key[2]
    if comp != "imaging":
        return True
    if task == "per_eval":
        return True            # the same objective and gradient at the same image: equal work by construction
    if task == "native":
        return case in COUNTED_NATIVE
    return False               # ladder (scored as time to precision), stop_default, stop_tight


def faster_counts(groups, ttp=None):
    """Groups in which each code is faster, by basis: warm, cold_first_call (run 1's ``t_cold``) and fresh (import + load +
    init + first call, the headline cold). Imaging ladder groups enter through ``ttp_pairs`` at the headline rung."""
    c = {b: Counter() for b in ("warm", "cold_first_call", "fresh")}

    def add(basis, r):
        if r is None:
            c[basis]["no_pair_time"] += 1
        elif r > 1:
            c[basis]["virgil_faster"] += 1
        elif r < 1:
            c[basis]["other_faster"] += 1
        else:
            c[basis]["equal"] += 1

    for key, v, o in pairs(groups):
        if not counted(key):
            continue
        add("warm", ratios(v, o, "warm")["ratio"])
        add("cold_first_call", ratios(v, o, "cold")["ratio"])
        add("fresh", ratios(v, o, "fresh")["ratio"])
    for key, route, rw, rc, gap in ttp_pairs(ttp or {}):
        add("warm", rw)
        add("fresh", rc)
    return {k: dict(v) for k, v in c.items()}


# ----------------------------------------------------------------------------------------------------------------
# LaTeX tabular
# ----------------------------------------------------------------------------------------------------------------

# (label, comparison, case, task, config)
HEADLINE = [
    ("B-sim-6 T2", "binary", "B-sim-6", "T2", "matched"),
    ("B-sim-600 T2", "binary", "B-sim-600", "T2", "matched"),
    ("B-sim-6 T3", "binary", "B-sim-6", "T3", "matched"),
    ("B-real-pionier T2", "binary", "B-real-pionier", "T2", "matched"),
    ("O-A-sim matched", "orbit", "O-A-sim", "matched", "matched"),
    ("O-A-real matched$^\\dagger$", "orbit", "O-A-real", "matched", "matched"),
    ("O-B-sim matched", "orbit", "O-B-sim", "matched", "matched"),
    ("I-sim-24 time to precision", "imaging", "I-sim-24", "ladder", "matched"),
    ("I-sim-48 time to precision", "imaging", "I-sim-48", "ladder", "matched"),
    ("I-sim-96 time to precision", "imaging", "I-sim-96", "ladder", "matched"),
    ("I-sim-ring native", "imaging", "I-sim-ring", "native", "native"),
]


def tex_esc(s):
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def f3(x):
    if x is None:
        return "---"
    if x == 0:
        return "0"
    if 100 <= abs(x) < 1e5:
        return f"{x:.0f}"
    return f"{x:.3g}" if 0.01 <= abs(x) < 100 else f"{x:.2e}".replace("e-0", "e-").replace("e+0", "e")


def cell_time(g, field):
    if g is None:
        return "not run"
    if g["status"] != "ok":
        return tex_esc(g["status"].replace("_", " "))
    return f3(g["t_fresh"] if field == "cold" else g["warm"]["median"])


def accuracy_cell(key, v, o):
    """max |deviation from truth/reference| in sigma where the rows carry it (binary), neutral ncc (imaging)."""
    def dev(g):
        if g is None or not g["rows"]:
            return None
        d = g["rows"][0]["accuracy"].get("dev_from_reference_sigma")
        return max(abs(x) for x in d.values()) if d else None
    if key[0] == "binary":
        a, b = dev(v), dev(o)
        return f"{f3(a)} / {f3(b)}" if (a is not None or b is not None) else "---"
    return "---"


def tex_table(groups, ttp):
    idx = {}
    for (key, code, variant), g in groups.items():
        if variant in (None, "lbfgsb", "native"):
            idx[(key[0], key[1], key[2], key[3], key[4], code)] = g
    hdr = [
        "% performance_table.tex: tabular only (no caption). All rows 1 thread, one CPU (r2: 64-core EPYC node), no GPU; times in seconds,",
        "% footprints in MB (cold = import + load + init + first call of a fresh process, compilation included; warm =",
        "% repeated call; imaging ladder rows give the interpolated time to the stated precision, cold and warm). Limits:",
        *[f"%   - {x}" for x in LIMITS],
        "% $^\\dagger$ period-informed prior; not scored.",
        "\\begin{tabular}{lrrrrrrrrrrr}",
        "\\hline",
        "Task & $N_{\\rm data}$ & $N_{\\rm par}$ & virgil cold & virgil warm & other cold & other warm & "
        "ratio warm & ratio cold & peak virgil & peak other & accuracy \\\\",
        "\\hline",
    ]
    lines = []
    for label, comp, case, task, config in HEADLINE:
        v = idx.get((comp, case, task, config, 1, "virgil"))
        o = idx.get((comp, case, task, config, 1, OTHER[comp]))
        base = v or o
        nd = base["n_data"] if base else None
        npar = base["n_param"] if base else None
        key = (comp, case, task, config, 1)
        if comp == "imaging" and task == "ladder":
            t = ttp.get(key, {})
            gap = t.get("headline_gap")
            tv, to = t.get("routes", {}).get("virgil"), t.get("routes", {}).get("ehtim")
            def tcell(x, g, field):
                if g is not None and g["status"] != "ok":
                    return tex_esc(g["status"].replace("_", " "))
                if not x or gap is None:
                    return "not reached"
                hit = x["reached"][str(gap)]
                return f3(hit[field]) if hit else "not reached"
            pc = lambda f: (f3(to["reached"][str(gap)][f] / tv["reached"][str(gap)][f])  # noqa: E731
                            if tv and to and gap is not None and tv["reached"][str(gap)] and to["reached"][str(gap)] else "---")
            lab = f"{label} (J within {gap * 100:g}\\% of best)" if gap is not None else label
            row = [lab, nd, npar, tcell(tv, v, "cold_time"), tcell(tv, v, "time"), tcell(to, o, "cold_time"),
                   tcell(to, o, "time"), pc("time"), pc("cold_time")]
        else:
            rw = common.ratio_with_spread(o["warm"], v["warm"]) if (v and o and v["status"] == "ok" and o["status"] == "ok") else {}
            rc = ratios(v, o, "fresh") if (v and o and v["status"] == "ok" and o["status"] == "ok") else {}
            row = [label, nd, npar, cell_time(v, "cold"), cell_time(v, "warm"), cell_time(o, "cold"),
                   cell_time(o, "warm"), f3(rw.get("ratio")), f3(rc.get("ratio"))]
        row += [f3(v["peak_mb"]) if v and v["status"] == "ok" else "---",
                f3(o["peak_mb"]) if o and o["status"] == "ok" else "---", accuracy_cell(key, v, o)]
        lines.append(" & ".join(str(x) if x is not None else "---" for x in row) + " \\\\")
    return "\n".join(hdr + lines + ["\\hline", "\\end{tabular}", ""])


# ----------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------

def figure(groups, ttp, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)

    def plot(ax, x, g, color, marker, dx, ymed=None):
        if g is None or g["status"] != "ok":
            return
        w = g["warm"]
        if w["median"] is not None:
            ax.errorbar([x + dx], [w["median"]], yerr=[[w["median"] - w["min"]], [w["max"] - w["median"]]], fmt=marker,
                        color=color, mfc=color, capsize=2, ms=6)
        if g["t_fresh"] is not None:
            lo, hi = g["fresh_range"]
            ax.errorbar([x + dx], [g["t_fresh"]], yerr=[[g["t_fresh"] - lo], [hi - g["t_fresh"]]], fmt=marker,
                        color=color, mfc="none", capsize=2, ms=6)

    get = lambda comp, case, task, config, code: next(  # noqa: E731
        (g for (k, c, var), g in groups.items() if k == (comp, case, task, config, 1) and c == code
         and var in (None, "lbfgsb")), None)
    # binary: T2 matched vs channels
    ax = axes[0]
    for case, n in (("B-sim-6", 6), ("B-sim-60", 60), ("B-sim-600", 600)):
        plot(ax, n, get("binary", case, "T2", "matched", "virgil"), C_V, "o", -0.03 * n)
        plot(ax, n, get("binary", case, "T2", "matched", "pmoired"), C_O, "s", 0.03 * n)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("channels"); ax.set_ylabel("wall time [s]")
    ax.set_xticks([6, 60, 600], ["6", "60", "600"]); ax.minorticks_off(); ax.set_xlim(3, 1200)
    ax.set_title("binary fit (T2)")
    # orbit: cases, cold = total t_init + sampling not shown: t_cold / t_warm of the sampling call
    ax = axes[1]
    names = ["O-A-sim", "O-A-real", "O-B-sim"]
    for i, case in enumerate(names):
        plot(ax, i, get("orbit", case, "matched", "matched", "virgil"), C_V, "o", -0.08)
        plot(ax, i, get("orbit", case, "matched", "matched", "octofitter"), C_O, "s", 0.08)
    ax.set_xticks(range(len(names)), names, fontsize=8); ax.set_yscale("log"); ax.set_title("orbit fit (matched)")
    # imaging: time to precision (tightest rung both codes reach) vs pixels; warm filled, cold (fresh process) open
    ax = axes[2]
    for case, n in (("I-sim-24", 24), ("I-sim-48", 48), ("I-sim-96", 96)):
        t = ttp.get(("imaging", case, "ladder", "matched", 1), {})
        gap = t.get("headline_gap")
        if gap is None:
            continue
        for route, color, marker, dx in (("virgil", C_V, "o", -0.03 * n), ("ehtim", C_O, "s", 0.03 * n)):
            hit = t["routes"][route]["reached"][str(gap)]
            if hit:
                ax.plot([n + dx], [hit["time"]], marker, color=color, mfc=color, ms=6)
                ax.plot([n + dx], [hit["cold_time"]], marker, color=color, mfc="none", ms=6)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("pixels per side"); ax.set_xticks([24, 48, 96], ["24", "48", "96"])
    ax.minorticks_off(); ax.set_xlim(15, 150); ax.set_title("imaging (time to equal precision)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", color=C_V, ls="", label="virgil"),
         Line2D([], [], marker="s", color=C_O, ls="", label="other code"),
         Line2D([], [], marker="o", color="k", mfc="k", ls="", label="warm (median, min-max)"),
         Line2D([], [], marker="o", color="k", mfc="none", ls="", label="cold: import + load + init + first call")]
    fig.legend(handles=h, fontsize=7, loc="outside lower center", ncol=4, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(Path(out_dir) / f"performance.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------
# check
# ----------------------------------------------------------------------------------------------------------------

def expected_cases():
    import yaml
    b = yaml.safe_load((HERE / "binary" / "cases.yml").read_text())["data"]
    o = yaml.safe_load((HERE / "orbit" / "cases.yml").read_text())["cases"]
    i = yaml.safe_load((HERE / "imaging" / "cases.yml").read_text())["cases"]
    return {"binary": [k for k in b if k.startswith("B-")], "orbit": list(o), "imaging": list(i)}


def check(raw, out_dir, strict_outputs):
    problems = []
    errs = common.check_rows(raw)
    for f, e in errs.items():
        problems += [f"{f}: {x}" for x in e]
    rows = load(raw)
    for comp, cases in expected_cases().items():
        for c in cases:
            for code in ("virgil", OTHER[comp]):
                if not any(r["comparison"] == comp and r["code"] == code and ((r["settings"].get("case_id") or r["case"]) == c)
                           for r in rows):
                    problems.append(f"no row: {comp} {c} {code}")
    if strict_outputs:
        for n in ("results.csv", "performance_table.tex", "performance.pdf", "performance.png"):
            if not (Path(out_dir) / n).exists():
                problems.append(f"missing output {n}")
    return rows, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(HERE / "results" / "raw"))
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-outputs-check", action="store_true", help="--check: do not require the output files")
    a = ap.parse_args()
    if a.check:
        rows, problems = check(a.raw, a.out, not a.no_outputs_check)
        print(f"{len(rows)} rows; schema and coverage problems: {len(problems)}")
        for p in problems:
            print("  ", p)
        sys.exit(1 if problems else 0)
    rows = load(a.raw)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    groups = group(rows)
    ttp = imaging_ttp(groups)
    with open(Path(a.out) / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, CSV_COLS)
        w.writeheader()
        w.writerows(csv_rows(groups))
    (Path(a.out) / "performance_table.tex").write_text(tex_table(groups, ttp))
    figure(groups, ttp, a.out)
    summary = {"n_rows": len(rows), "n_groups": len(groups), "faster_counts": faster_counts(groups, ttp),
               "time_to_precision": {"|".join(map(str, k)): v for k, v in ttp.items()},
               "flags": dict(Counter(f for r in rows for f in r["flags"])),
               "status": dict(Counter(r["status"] for r in rows)), "limits": LIMITS}
    (Path(a.out) / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({k: summary[k] for k in ("n_rows", "n_groups", "faster_counts", "status", "flags")}, indent=1))


if __name__ == "__main__":
    main()
