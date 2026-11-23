"""Driver, neutral scoring, gate report and time-to-target for the imaging comparison. BENCH_PYTHON; imports neither
virgil nor eht-imaging (the scoring is NumPy: ``neutral.py`` and ``crosscheck.image_metrics``).

    python compare.py gate-report [--cases I-sim-24 ...]        # needs the two codes' gate outputs (run_*.py gate)
    python compare.py time-to-precision --virgil ROW.json --ehtim ROW.json [--virgil-lm ROW.json]
    python run_virgil.py gate --case I-sim-24 | run --case I-sim-24 --task ladder --config matched [--size tiny]
    python run_ehtim.py   (same arguments)

``drive`` launches one fresh process per row through ``common.timed_process`` (``--wrapper pyslot`` for measured runs),
scores the worker's images **outside** the measured process (so scoring costs neither time nor peak memory), and writes
the row with ``common.write_row``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import common  # noqa: E402
import make_data  # noqa: E402
import neutral  # noqa: E402

BETA = 7.5e5
RAW = BENCH / "results" / "raw"
GATE_DIR = make_data.DATA_DIR / "gate"
GATE_RESULTS = HERE / "gate_results.json"


def cases_yml():
    import yaml

    return yaml.safe_load((HERE / "cases.yml").read_text())


# ----------------------------------------------------------------------------------------------------------------
# scoring (neutral)
# ----------------------------------------------------------------------------------------------------------------


def _case_data(case, npix=None, pixel=None):
    dataset, n0 = make_data.CASES[case]
    d = make_data.load(dataset)
    npix = npix or n0
    pixel = pixel or make_data.pixel_mas(npix)
    return dataset, d, make_data.neutral_data(d, npix, pixel), npix, pixel


def score_image(case, image, grid=None):
    """Neutral scores of one image: J and its terms (``neutral.image_objective``, eht-imaging's edge convention, the
    OIFITS Fourier sign), the raw flux, and the accuracy against the analytic truth rendered on the image's grid."""
    image = np.asarray(image, float)
    npix, pixel = image.shape[0], (grid or {}).get("grid_pixel_mas") or make_data.FIELD_MAS / image.shape[0]
    dataset, d, data, _, _ = _case_data(case, npix, pixel)
    t = neutral.image_terms(image, data, sign=-1)
    unit = image / image.sum()
    tv = neutral.tv2_cost(unit)
    out = {"J": t["half_chi2_amp"] + t["half_chi2_cp"] + BETA * tv, "half_chi2_amp": t["half_chi2_amp"],
           "half_chi2_cp": t["half_chi2_cp"], "tv2": tv, "flux": t["flux"],
           "n_amp": int(d["amp"].size), "n_cp": int(d["cphase_deg"].size)}
    out.update(truth_metrics(dataset, d, unit, pixel))
    return out


def truth_metrics(dataset, d, unit, pixel):
    """ncc after integer-pixel registration and rms(relative) after convolution to the nominal beam, both from
    ``crosscheck.image_metrics`` (plan section 6, accuracy)."""
    from crosscheck import image_metrics as im

    npix = unit.shape[0]
    truth = make_data.truth_image(make_data.DATASETS[dataset]["scene"], npix, pixel)
    best = (-2.0, 0, 0)
    for r in range(-3, 4):
        for c in range(-3, 4):
            v = im.ncc(im.shift(unit, r, c), truth)
            if v > best[0]:
                best = (v, r, c)
    b_max = float(np.max(np.hypot(d["u_amp"], d["v_amp"])))
    beam = float(np.mean(d["wavel_amp"])) / (2 * b_max) / make_data.MAS  # mas, lambda / (2 B_max)
    half = int(np.ceil(2 * beam / pixel))
    k = im.gaussian_kernel(2 * half + 1, pixel, beam, beam, 0.0)
    return {"ncc_registered": best[0], "ncc_shift_rc": [best[1], best[2]], "ncc_unregistered": im.ncc(unit, truth),
            "rms_rel_beam": im.rms(unit, truth, kernel=k, relative=True), "beam_mas": beam}


# ----------------------------------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------------------------------


def _worker_cmd(code, args):
    if code == "virgil":
        return [os.environ.get("BENCH_PYTHON", sys.executable), str(HERE / "run_virgil.py"), "--worker", json.dumps(args)]
    ehtim = os.environ.get("EHTIM_PYTHON")
    if not ehtim:
        raise SystemExit("EHTIM_PYTHON is not set (source ~/.cache/virgil-bench/activate.sh)")
    return [ehtim, str(HERE / "ehtim_bench_worker.py"), "--worker", json.dumps(args)]


def _task_args(code, case, task, size, cfg, method=None):
    """The worker's argument dict, from ``cases.yml`` (nothing is chosen here)."""
    lad, stop = cfg["ladder"], cfg["stopping_rows"]
    reps = lad["reps_by_size"]["tiny" if size == "tiny" else "full"]
    a = {"case": case, "task": task, "reps": reps, "size": size}
    if task == "ladder":
        a["budgets"] = lad["budgets_tiny"] if size == "tiny" else lad["budgets"]
        if code == "virgil":
            a.update(gtol=lad["virgil_gtol"], method=method or "lbfgsb")
            if (method or "lbfgsb") == "lbfgsb":
                a["memory"] = 50
        else:
            a["stop"] = lad["ehtim_stop"]
    elif task in ("stop_default", "stop_tight"):
        s = stop["default" if task == "stop_default" else "tight"][code if code == "virgil" else "ehtim"]
        a.update(s)
        if code == "virgil":
            a["gtol"] = s["gtol"]
            a["memory"] = 50
    elif task == "native":
        a["native"] = cfg["native"]["virgil" if code == "virgil" else "ehtim"]
    elif task == "per_eval":
        a["calls"] = 5 if size == "tiny" else cfg["per_evaluation"]["calls"]
    return a


CONFIG_OF_TASK = {"ladder": "matched", "stop_default": "matched", "stop_tight": "tight", "native": "native",
                  "per_eval": "matched"}


def drive(code, case, task, size="full", threads=1, wrapper=None, method=None, timeout=1800, out_dir=None):
    """One fresh process, one row. Returns the written path."""
    cfg = cases_yml()
    config = CONFIG_OF_TASK[task]
    args = _task_args(code, case, task, size, cfg, method)
    bad = make_data.check_manifest()
    if bad:
        raise SystemExit(f"data manifest mismatch: {bad} (run make_data.py)")
    tmp = tempfile.mkdtemp(prefix="imaging-", dir=str(make_data.DATA_DIR))
    args["images_file"] = str(Path(tmp) / "images.npz")
    res = common.timed_process(_worker_cmd(code, args), threads=threads, timeout=timeout, wrapper=wrapper,
                                cache_key=f"imaging-{case}-{task}-{method or 'lbfgsb'}-{code}-t{threads}")
    identity = dict(case=case, comparison="imaging", code="virgil" if code == "virgil" else "ehtim", config=config,
                    threads=threads, size=size)
    row = common.row_from_process(res, **identity)
    row["settings"]["task"] = task
    if res["status"] != "ok":
        errs = [l.strip() for l in res["stderr_tail"].splitlines() if "Error" in l or "Exception" in l or "exceeded" in l]
        row["settings"]["reason"] = (" | ".join(errs) or row["settings"].get("reason", ""))[-300:]
    row["settings"]["method"] = args.get("method", "native" if task == "native" else None)
    inner = res.get("inner") or {}
    if res["status"] == "ok" and inner.get("images_file") and Path(inner["images_file"]).exists():
        z = np.load(inner["images_file"])
        scores = []
        grid = {k: inner["accuracy"][k] for k in ("grid_pixel_mas",) if k in inner.get("accuracy", {})}
        for i in range(len(z.files)):
            scores.append(score_image(case, z[f"img_{i}"], grid or None))
        if task == "ladder":
            for entry, sc in zip(row["accuracy"]["ladder"], scores):
                entry["score"] = sc
            row["accuracy"]["ladder_J"] = [s["J"] for s in scores]
        else:
            row["accuracy"]["score"] = scores[0]
    gate = json.loads(GATE_RESULTS.read_text())["cases"].get(case) if GATE_RESULTS.exists() else None
    if gate is not None and not gate["passed"] and config != "native" and row["status"] == "ok":
        # plan section 1: a failed gate is a finding; the measurements stay in the row, labelled
        row["status"] = "gate_failed"
        row["settings"]["reason"] = (f"gate against the neutral scorer failed: worst relative difference "
                                     f"{gate['worst_rel_diff']:.3g} (tolerance {cases_yml()['gate']['tolerance_rel']:.0e}); "
                                     "see gate_results.json")
    name = f"imaging__{case}__{task}{'-' + method if method and method != 'lbfgsb' else ''}__{row['code']}__{config}__t{threads}"
    if size == "tiny":
        name += "__tiny"
    return common.write_row(row, out_dir, name=name)


def gate_run(code, case, out_dir=GATE_DIR):
    """Run one code's gate worker (objective at the start and at the truth) and store its output."""
    out_dir.mkdir(parents=True, exist_ok=True)
    res = common.timed_process(_worker_cmd(code, {"case": case, "task": "gate"}), threads=1, timeout=900)
    if res["status"] != "ok" and res.get("inner") is None:
        raise SystemExit(f"gate worker failed: {res['stderr_tail']}")
    gate = res["inner"]["gate"]
    p = out_dir / f"{code}__{case}.json"
    p.write_text(json.dumps(gate, indent=1))
    return gate


def gate_report(cases):
    """Compare both codes' objectives with the neutral scorer at the start and at the truth; write ``gate_results.json``."""
    tol = cases_yml()["gate"]["tolerance_rel"]
    report = {"tolerance_rel": tol, "cases": {}}
    for case in cases:
        dataset, d, _, npix, pixel = _case_data(case)
        scene = make_data.DATASETS[dataset]["scene"]
        points = {"start": make_data.start_image(npix, pixel), "truth": make_data.truth_image(scene, npix, pixel)}
        v = json.loads((GATE_DIR / f"virgil__{case}.json").read_text())
        e = json.loads((GATE_DIR / f"ehtim__{case}.json").read_text())
        data = make_data.neutral_data(d, npix, pixel)
        entry = {"n_amp": int(d["amp"].size), "n_cp": int(d["cphase_deg"].size),
                 "ehtim_counts": [e["n_amp"], e["n_cphase"]], "points": {}}
        for name, img in points.items():
            j_eh = neutral.image_objective(img, data, BETA, sign=-1, edges="ehtim")
            j_vg = neutral.image_objective(img, data, BETA, sign=-1, edges="virgil")
            ve, vv = e["points"][name]["J"], v["points"][name]["loss"]
            entry["points"][name] = {
                "neutral_J_ehtim_edges": j_eh, "ehtim_J": ve, "rel_diff_ehtim": abs(ve / j_eh - 1.0),
                "neutral_J_virgil_edges": j_vg, "virgil_loss": vv, "rel_diff_virgil": abs(vv / j_vg - 1.0),
            }
        worst = max(max(p["rel_diff_ehtim"], p["rel_diff_virgil"]) for p in entry["points"].values())
        entry["worst_rel_diff"] = worst
        entry["counts_match"] = entry["ehtim_counts"] == [entry["n_amp"], entry["n_cp"]]
        entry["passed"] = bool(worst < tol and entry["counts_match"])
        report["cases"][case] = entry
    GATE_RESULTS.write_text(json.dumps(report, indent=1) + "\n")
    return report


# ----------------------------------------------------------------------------------------------------------------
# time to precision (cases.yml: ladder.time_to_precision)
# ----------------------------------------------------------------------------------------------------------------

RUNGS = (0.1, 0.03, 0.01, 0.003, 0.001)     # relative gap to J_ref (cases.yml)
HEADLINE_ROUTES = ("virgil", "ehtim")       # the matched optimiser family; the default-method virgil route is reported beside


def _median(v):
    import statistics

    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def curve(rows):
    """Per budget: median warm wall time (pooled over the processes) and the neutral scores, from ladder rows of one route."""
    rows = [r for r in rows if r.get("status") == "ok" and r["accuracy"].get("ladder")]
    pts = []
    for i, e in enumerate(rows[0]["accuracy"]["ladder"]):
        ts = [x for r in rows for x in r["accuracy"]["ladder"][i]["t_warm"]]
        sc = e["score"]
        pts.append({"budget": e["budget"], "t": _median(ts), "J": sc["J"], "rms": sc.get("rms_rel_beam"),
                    "ncc": sc.get("ncc_registered")})
    return sorted(pts, key=lambda p: p["budget"])


def cold_overhead(rows):
    """Fixed cost of a fresh process before the first budget's warm time: import + load + (cold - warm) of the first
    budget (the compile / set-up part of the first call). Median over the processes."""
    out = []
    for r in rows:
        if r.get("status") != "ok" or r.get("t_cold") is None:
            continue
        w = _median(r["t_warm"]) or 0.0
        out.append((r.get("t_import") or 0.0) + (r.get("t_load") or 0.0) + (r.get("t_init") or 0.0) + max(r["t_cold"] - w, 0.0))
    return _median(out)


def _first_reach(points, target):
    """(interpolated time, first budget at or past the target, its point) or None. Linear in J between budgets."""
    prev = None
    for p in points:
        if p["J"] <= target:
            if prev is None:
                return p["t"], p
            f = (prev["J"] - target) / (prev["J"] - p["J"]) if prev["J"] != p["J"] else 1.0
            return prev["t"] + f * (p["t"] - prev["t"]), p
        prev = p
    return None


def time_to_precision(routes, rungs=RUNGS):
    """Time for each route to first reach each precision rung, J_ref (1 + gap).

    ``routes``: name -> list of ladder rows (one per process); names "virgil", "ehtim" and optionally "virgil_lm".
    J_ref is the lowest J any route reaches at any budget. ``headline_gap`` is the tightest rung reached by both
    ``HEADLINE_ROUTES``. Per route and rung: warm time, cold time (fixed overhead + warm time) and the neutral
    truth diagnostics (rms_rel_beam, ncc) of the first budget at or past the rung."""
    curves = {k: curve(r) for k, r in routes.items() if r}
    j_ref = min(p["J"] for c in curves.values() for p in c)
    out = {"J_ref": j_ref, "rungs": {str(g): j_ref * (1 + g) for g in rungs}, "routes": {}}
    for k, c in curves.items():
        over = cold_overhead(routes[k])
        best = min(p["J"] for p in c)
        res = {"best_J": best, "best_gap": best / j_ref - 1.0, "cold_overhead": over, "reached": {}}
        for g in rungs:
            hit = _first_reach(c, j_ref * (1 + g))
            res["reached"][str(g)] = None if hit is None else {
                "time": hit[0], "cold_time": None if over is None else over + hit[0], "budget": hit[1]["budget"],
                "rms_rel_beam": hit[1]["rms"], "ncc": hit[1]["ncc"]}
        out["routes"][k] = res
    both = [g for g in rungs if all(h in out["routes"] and out["routes"][h]["reached"][str(g)] for h in HEADLINE_ROUTES)]
    out["headline_gap"] = min(both) if both else None
    return out


# ----------------------------------------------------------------------------------------------------------------
# command line shared by run_virgil.py / run_ehtim.py
# ----------------------------------------------------------------------------------------------------------------


def main(code, argv):
    p = argparse.ArgumentParser(prog=f"run_{code}.py")
    p.add_argument("command", choices=["gate", "run"])
    p.add_argument("--case", required=True)
    p.add_argument("--config", default=None, help="accepted for symmetry; the config follows from the task")
    p.add_argument("--task", default="ladder", choices=list(CONFIG_OF_TASK))
    p.add_argument("--size", default="full", choices=["tiny", "full"])
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--method", default=None, help="virgil only: lbfgsb (default) or default (fit's own default method)")
    p.add_argument("--wrapper", default=None, choices=[None, "pyslot"])
    p.add_argument("--out-dir", default=None)
    a = p.parse_args(argv)
    if a.command == "gate":
        print(json.dumps(gate_run(code, a.case), indent=1))
    else:
        method = None if a.method in (None, "lbfgsb") else a.method
        print(drive(code, a.case, a.task, a.size, a.threads, a.wrapper, method, out_dir=a.out_dir))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate-report")
    g.add_argument("--cases", nargs="*", default=list(make_data.CASES))
    t = sub.add_parser("time-to-precision")
    t.add_argument("--virgil", required=True)
    t.add_argument("--ehtim", required=True)
    t.add_argument("--virgil-lm", default=None)
    a = ap.parse_args()
    if a.cmd == "gate-report":
        rep = gate_report(a.cases)
        for case, e in rep["cases"].items():
            print(case, "PASS" if e["passed"] else "FAIL", f"worst rel diff {e['worst_rel_diff']:.3e}", e["points"])
    else:
        rows = {"virgil": [json.loads(Path(a.virgil).read_text())], "ehtim": [json.loads(Path(a.ehtim).read_text())]}
        if a.virgil_lm:
            rows["virgil_lm"] = [json.loads(Path(a.virgil_lm).read_text())]
        print(json.dumps(time_to_precision(rows), indent=1))
