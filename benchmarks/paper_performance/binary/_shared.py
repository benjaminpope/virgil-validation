"""Pieces shared by ``run_virgil.py`` and ``run_pmoired.py`` (and ``compare.py``).

Imports neither virgil nor PMOIRED: NumPy, PyYAML, astropy (to read OIFITS for the neutral scorer), ``common`` and
``neutral`` of the harness, and ``make_data``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402
import make_data as md  # noqa: E402
import neutral  # noqa: E402

CASES = yaml.safe_load((HERE / "cases.yml").read_text())
SIM_CASES = ["B-sim-6", "B-sim-60", "B-sim-600", "B-sim-ud"]
ALL_CASES = [*SIM_CASES, "B-real-pionier"]
PULL_CASES = {"B-pulls-tri": ("triangle", "tri"), "B-pulls-base": ("baseline", "base")}
TASKS = ("gate", "T1", "T2", "T3", "pulls", "import")
CONFIGS = ("matched", "tight", "native", "matched-coarse")
T3_GRID = {"matched": 40, "matched-coarse": 20}
T3_NATIVE = {"virgil_dra_ddec": 81, "virgil_flux": 30, "pmoired_step": 2.0}
PMOIRED_STEP = {40: 1.0256, 20: 2.1}  # exactly 40 / 20 points in PMOIRED's int((max - min) / step + 1) rule
T2_TOL = {"matched": {"gtol": 1e-4, "ftol": 1e-5}, "native": {"gtol": 1e-4, "ftol": 1e-5},
          "tight": {"gtol": 1e-8, "ftol": 1e-10}, "matched-coarse": {"gtol": 1e-4, "ftol": 1e-5}}
BOX = 20.0
FLUX_START_GRID = 0.05
SIZE = {"tiny": {"warm": 2, "t1_calls": 20, "pulls": md.N_PULLS_TINY, "grid": 8},
        "full": {"warm": 5, "t1_calls": 100, "pulls": md.N_PULLS, "grid": None}}


def grid_points(config, size):
    """Points per axis of the T3 start grid (tiny size: 8 for matched, 4 for matched-coarse)."""
    n = T3_GRID.get(config, T3_GRID["matched"])
    if size == "tiny":
        return 8 if n == 40 else 4
    return n


def pmoired_step(n):
    """Grid step reproducing exactly ``n`` points on [-20, 20] in PMOIRED's rule, checked."""
    step = 2 * BOX / (n - 1) * 0.9999  # just below the exact step: int((max - min) / step + 1) == n
    assert int((2 * BOX) / step + 1) == n
    return step


def case_files(case):
    """Data files (list of Path) for a case; verifies the manifest (simulated) or the JSON sha256 (real)."""
    if case == "B-real-pionier":
        paths, _, _ = md.real_files()
        bad = md.verify_real()
        if bad:
            raise SystemExit(f"real files missing or changed: {bad}")
        return paths
    name = f"{md.SIM_FILE[case]}.fits"
    bad = md.verify([name])
    if bad:
        raise SystemExit(f"data missing or changed ({bad}): run make_data.py first")
    return [md.DATA / name]


def pull_files(case, size):
    noise, _ = PULL_CASES[case]
    names = [md.pull_name(noise, k) for k in range(SIZE[size]["pulls"])]
    bad = md.verify(set(names))
    if bad:
        raise SystemExit(f"pull data missing or changed ({bad[:3]}...): run make_data.py first")
    return [md.DATA / n for n in names]


def reference(case):
    """Truth (sim) or published reference (real): dict dra, ddec, flux[, ud]."""
    if case == "B-real-pionier":
        _, _, ep = md.real_files()
        return {"dra": ep["dra_mas"], "ddec": ep["ddec_mas"], "flux": ep["flux_ratio_pct"] / 100.0}
    base = "B-sim-6" if case in PULL_CASES else case
    ref = dict(md.TRUTH)
    if md.CASES[base][1]:
        ref["ud"] = md.CASES[base][1]
    return ref


def start_values(case):
    """Reference offset by (+0.3, -0.3) mas and x1.3 in flux (and in ud)."""
    s = dict(reference(case))
    s["dra"] += 0.3
    s["ddec"] -= 0.3
    s["flux"] *= 1.3
    if "ud" in s:
        s["ud"] *= 1.3
    return s


def free_names(case):
    return ["dra", "ddec", "flux"] + (["ud"] if "ud" in reference(case) else [])


# ---------------------------------------------------------------------------------------------------------------
# neutral scoring (NumPy + astropy only)
# ---------------------------------------------------------------------------------------------------------------


def read_neutral(files):
    """Unflagged V2 and closure-phase samples of OIFITS files, as flat arrays (angles in radians)."""
    from astropy.io import fits

    keys = ("u", "v", "wl", "v2", "dv2", "u1", "v1", "u2", "v2_", "wl3", "cp", "dcp")
    out = {k: [] for k in keys}
    for f in files:
        with fits.open(f) as h:
            wl = np.asarray(h["OI_WAVELENGTH"].data["EFF_WAVE"], float)
            n = wl.size
            d = h["OI_VIS2"].data
            ok = ~np.asarray(d["FLAG"], bool) & np.isfinite(d["VIS2DATA"]) & (d["VIS2ERR"] > 0)
            for key, col in (("u", "UCOORD"), ("v", "VCOORD")):
                out[key].append(np.broadcast_to(np.asarray(d[col], float)[:, None], (len(d), n))[ok])
            out["wl"].append(np.broadcast_to(wl[None, :], (len(d), n))[ok])
            out["v2"].append(np.asarray(d["VIS2DATA"], float)[ok])
            out["dv2"].append(np.asarray(d["VIS2ERR"], float)[ok])
            t = h["OI_T3"].data
            ok = ~np.asarray(t["FLAG"], bool) & np.isfinite(t["T3PHI"]) & (t["T3PHIERR"] > 0)
            for key, col in (("u1", "U1COORD"), ("v1", "V1COORD"), ("u2", "U2COORD"), ("v2_", "V2COORD")):
                out[key].append(np.broadcast_to(np.asarray(t[col], float)[:, None], (len(t), n))[ok])
            out["wl3"].append(np.broadcast_to(wl[None, :], (len(t), n))[ok])
            out["cp"].append(np.deg2rad(np.asarray(t["T3PHI"], float))[ok])
            out["dcp"].append(np.deg2rad(np.asarray(t["T3PHIERR"], float))[ok])
    return {k: np.concatenate(v) for k, v in out.items()}


def neutral_chi2(nd, params):
    """``neutral.chi2_v2_cp`` for a binary (dra, ddec, flux[, ud]) on the arrays of ``read_neutral``."""
    args = (params["dra"], params["ddec"], params["flux"], params.get("ud", 0.0))
    v = neutral.binary_vis(nd["u"], nd["v"], nd["wl"], *args)
    v1 = neutral.binary_vis(nd["u1"], nd["v1"], nd["wl3"], *args)
    v2 = neutral.binary_vis(nd["u2"], nd["v2_"], nd["wl3"], *args)
    v3 = neutral.binary_vis(-nd["u1"] - nd["u2"], -nd["v1"] - nd["v2_"], nd["wl3"], *args)
    cp = neutral.closure_phase(v1, v2, v3)
    return neutral.chi2_v2_cp(nd["v2"], nd["dv2"], np.abs(v) ** 2, nd["cp"], nd["dcp"], cp)


def accuracy_block(case, files, best, sigma):
    """Accuracy record shared by both codes: best fit, sigma, deviation from the reference, neutral chi2."""
    nd = read_neutral(files)
    ref = reference(case)
    c = neutral_chi2(nd, best)
    c_ref = neutral_chi2(nd, ref)
    names = free_names(case)
    ndat = c["n_v2"] + c["n_cp"]
    acc = {
        "best": {k: float(best[k]) for k in names}, "sigma": {k: float(sigma[k]) for k in names},
        "reference": {k: float(ref[k]) for k in names},
        "dev_from_reference_sigma": {k: float((best[k] - ref[k]) / sigma[k]) for k in names},
        "chi2_neutral": c["chi2"], "chi2_neutral_v2": c["chi2_v2"], "chi2_neutral_cp": c["chi2_cp"],
        "chi2_neutral_red": c["chi2"] / (ndat - len(names)), "chi2_neutral_at_reference": c_ref["chi2"],
        "n_v2": c["n_v2"], "n_cp": c["n_cp"],
    }
    if case == "B-real-pionier":  # companion/primary swap: (x, y, f) -> (-x, -y, 1/f)
        acc["nearer_reference_than_swapped"] = bool(
            np.hypot(best["dra"] - ref["dra"], best["ddec"] - ref["ddec"])
            <= np.hypot(-best["dra"] - ref["dra"], -best["ddec"] - ref["ddec"]))
        acc["swapped"] = not acc["nearer_reference_than_swapped"]
    return acc, ndat


def pull_summary(best_list, sigma_list, truth, names):
    b, s = np.asarray(best_list), np.asarray(sigma_list)
    t = np.array([truth[k] for k in names])
    pull = (b - t) / s
    n = len(b)
    return {
        "n": n, "names": names, "pull_mean": pull.mean(0).tolist(),
        "pull_sd": (pull.std(0, ddof=1) if n > 1 else np.full(len(names), np.nan)).tolist(),
        "sigma_mean": s.mean(0).tolist(), "bias": (b.mean(0) - t).tolist(), "pull_sd_stderr": float(1 / np.sqrt(2 * max(n - 1, 1))),
    }


# ---------------------------------------------------------------------------------------------------------------
# command line, outer process, worker prologue
# ---------------------------------------------------------------------------------------------------------------


def parser(code):
    ap = argparse.ArgumentParser(description=f"binary benchmark, {code} side (outer: spawns a fresh worker process)")
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--case", default="B-sim-6")
    ap.add_argument("--config", choices=CONFIGS, default="matched")
    ap.add_argument("--size", choices=["tiny", "full"], default="full")
    ap.add_argument("--threads", type=int, choices=[1, 2], default=1)
    ap.add_argument("--out", default=None, help="directory for the row (default: results/raw of the harness)")
    ap.add_argument("--wrapper", choices=["pyslot"], default=None, help="run_all.sh passes pyslot; smoke runs do not")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--device", default="cpu", help="virgil side: jax platform; only cpu is run on the laptop")
    ap.add_argument("--worker", action="store_true", help="internal: run the task in this process")
    return ap


def task_case_ok(task, case):
    if task in ("T1", "T2"):
        return case in ALL_CASES
    if task == "T3":
        return case in ("B-sim-6", "B-sim-60", "B-real-pionier")
    if task == "pulls":
        return case in PULL_CASES
    return True


def outer(args, code, script):
    """Launch the worker in a fresh process through ``common.timed_process`` and write the row."""
    if not task_case_ok(args.task, args.case):
        sys.exit(f"task {args.task} is not defined for case {args.case} (cases.yml)")
    cmd = [sys.executable, str(script), "--worker", "--task", args.task, "--case", args.case, "--config",
           args.config, "--size", args.size, "--threads", str(args.threads), "--device", args.device]
    res = common.timed_process(cmd, threads=args.threads, timeout=args.timeout, wrapper=args.wrapper)
    row = common.row_from_process(
        res, case=f"{args.case}-{args.task}", comparison="binary", code=code, config=args.config,
        threads=args.threads, size=args.size)
    row["settings"].update({"task": args.task, "case_id": args.case})
    if row["status"] != "ok":
        print(res["stderr_tail"][-1500:], file=sys.stderr)
    path = common.write_row(row, args.out)
    t = row["t_cold"]
    print(f"{path.name}: status={row['status']} t_cold={t if t is None else round(t, 3)} "
          f"t_warm={[round(x, 4) for x in row['t_warm']]} footprint={row['peak_footprint_mb']}")
    return row


def warm_n(size, t_cold):
    """Warm repeats inside one process: 5 (3 above 60 s, 0 above 10 min with flag single_run); 2 at tiny size."""
    n = common.repeat_plan(t_cold)["warm"] if t_cold <= 600 else 0
    return min(n, 2) if size == "tiny" else n


def plan_flags(sw, task):
    return common.repeat_plan(sw.times["t_cold"])["flags"] if task in ("T2", "T3") and "t_cold" in sw.times else []


def calib(sw):
    sys.path.insert(0, str(common.HERE))
    from calib import calibrate

    with sw("t_calib"):
        calibrate()


def provenance_extra():
    return common.provenance(data_manifest=common.manifest_hash(md.MANIFEST))
