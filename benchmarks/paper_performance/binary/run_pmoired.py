"""PMOIRED side of the binary-fitting benchmark. Never imports virgil.

Same command line as ``run_virgil.py`` (see its docstring and ``cases.yml``). PMOIRED conventions follow
``docs/method/pmoired.md`` and are mirrored from ``external_bridge/pmoired_models.py`` (not imported, not edited):
``OI.setupFit({'obs': ['V2', 'T3PHI']}, auto=False)`` (auto=True gives NaNs, P1), point sources as ``{'ud': 0}``,
x = East, y = North, ``f`` the total flux, and the ``sqrt(chi2_red)`` scaling of the uncertainties undone.
"""

from __future__ import annotations

import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _shared as sh  # noqa: E402

common, md, neutral = sh.common, sh.md, sh.neutral

MAXFEV = 10000    # OI.doFit default
GRID_MAXFEV = 5000   # OI.gridFit defaults (stated explicitly)
EPSFCN = 1e-6


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def model_dict(case, x):
    d = {"A,ud": float(x.get("ud", 0.0)), "A,f": 1.0, "B,ud": 0.0, "B,x": float(x["dra"]), "B,y": float(x["ddec"]),
         "B,f": float(x["flux"])}
    return d


def fixed_params(case):
    return ["A,f", "B,ud"] + ([] if "ud" in sh.reference(case) else ["A,ud"])


FREE_KEY = {"dra": "B,x", "ddec": "B,y", "flux": "B,f", "ud": "A,ud"}


def load(pm, files):
    """Read the files and set up the V2 + closure-phase fit (no bandwidth smearing, auto=False)."""
    with quiet():
        oi = pm.OI([str(f) for f in files] if len(files) > 1 else str(files[0]), verbose=False)
        oi.setupFit({"obs": ["V2", "T3PHI"]}, auto=False)
    return oi


def read_best(b, case):
    scale = np.sqrt(b["chi2"]) if b.get("normalized uncertainties", False) else 1.0
    names = sh.free_names(case)
    best = {k: float(b["best"][FREE_KEY[k]]) for k in names}
    sigma = {k: float(b["uncer"][FREE_KEY[k]]) / float(scale) for k in names}
    return best, sigma


def fit_pm(oi, case, start, ftol):
    t0 = time.perf_counter()
    with quiet():
        oi.doFit(model_dict(case, start), doNotFit=fixed_params(case), verbose=0, maxfev=MAXFEV, ftol=ftol,
                 epsfcn=EPSFCN)
    dt = time.perf_counter() - t0
    best, sigma = read_best(oi.bestfit, case)
    return best, sigma, {"nfev": int(oi.bestfit["info"]["nfev"]), "chi2_red_own": float(oi.bestfit["chi2"]),
                         "ndof": int(oi.bestfit["ndof"])}, dt


# ---------------------------------------------------------------------------------------------------------------


def task_gate(pm, om, sw):
    out = {}
    for case in ("B-sim-6", "B-sim-ud"):
        ref = sh.reference(case)
        clean = md.DATA / f"{md.SIM_FILE[case]}_clean.fits"
        bad = md.verify([clean.name])
        if bad:
            raise SystemExit(f"{bad} missing or changed")
        oi = load(pm, [clean])
        data = oi.data[0]
        with quiet():
            model = om.VmodelOI(data, model_dict(case, ref))
        dv2 = max(np.max(np.abs(model["OI_VIS2"][k]["V2"] - data["OI_VIS2"][k]["V2"])) for k in data["OI_VIS2"])
        dcp = max(np.max(np.abs((model["OI_T3"][k]["T3PHI"] - data["OI_T3"][k]["T3PHI"] + 180) % 360 - 180))
                  for k in data["OI_T3"])
        out[case] = {"v2_max_abs_diff": float(dv2), "cp_max_abs_diff_deg": float(dcp)}
        noisy = sh.case_files(case)
        oin = load(pm, noisy)
        with quiet():
            r = np.asarray(om.residualsOI(oin.data, model_dict(case, ref)))
        c_neu = sh.neutral_chi2(sh.read_neutral(noisy), ref)["chi2"]
        out[case].update({"chi2_pmoired_at_reference": float(np.sum(r**2)), "chi2_neutral_at_reference": c_neu,
                          "n_rows_pmoired": int(r.size), "chi2_rel_diff": float((np.sum(r**2) - c_neu) / c_neu)})
    return out


def task_t1(pm, om, sw, case, size, **_):
    files = sh.case_files(case)
    with sw("t_load"):
        oi = load(pm, files)
    ref = sh.reference(case)
    p = model_dict(case, ref)

    def call():
        with quiet():
            return [om.VmodelOI(d, p) for d in oi.data]

    with sw("t_cold"):
        call()
    n_calls = sh.SIZE[size]["t1_calls"]
    for _ in range(sh.SIZE[size]["warm"]):
        with sw("t_warm", repeat=True):
            for _ in range(n_calls):
                call()
    sw.lists["t_warm"] = [t / n_calls for t in sw.lists["t_warm"]]
    nd = sum(d["OI_VIS2"][k]["V2"].size for d in oi.data for k in d["OI_VIS2"]) + \
        sum(d["OI_T3"][k]["T3PHI"].size for d in oi.data for k in d["OI_T3"])
    return {"n_data": int(nd), "n_param": len(sh.free_names(case)),
            "iterations": {"calls_per_warm_batch": n_calls, "data_elements_per_call": len(oi.data)},
            "accuracy": {}, "settings": {"warm_is": "seconds per call (one VmodelOI per file)"}}


def task_t2(pm, om, sw, case, config, size, **_):
    files = sh.case_files(case)
    with sw("t_load"):
        oi = load(pm, files)
    start = sh.start_values(case)
    ftol = sh.T2_TOL[config]["ftol"]
    with sw("t_cold"):
        best, sigma, info, dt = fit_pm(oi, case, start, ftol)
    iters = {**info, "t_fit_cold": dt}
    warm = []
    for _ in range(sh.warm_n(size, sw.times["t_cold"])):
        with sw("t_warm", repeat=True):
            fit_pm(oi, case, start, ftol)
    iters["t_fit_warm"] = list(sw.lists["t_warm"])
    acc, ndat = sh.accuracy_block(case, files, best, sigma)
    return {"n_data": ndat, "n_param": len(sh.free_names(case)), "iterations": iters, "accuracy": acc,
            "settings": {"ftol": ftol, "epsfcn": EPSFCN, "maxfev": MAXFEV, "start": start, "auto": False,
                         "autoPrior": True}}


def task_t3(pm, om, sw, case, config, size, threads, **_):
    files = sh.case_files(case)
    with sw("t_load"):
        oi = load(pm, files)
    ref = sh.reference(case)
    start = {"dra": 0.0, "ddec": 0.0, "flux": sh.FLUX_START_GRID}
    if config == "native":
        step = sh.T3_NATIVE["pmoired_step"] if size == "full" else 8.0
        n = int(2 * sh.BOX / step + 1)
    else:
        n = sh.grid_points(config, size)
        step = sh.pmoired_step(n)
    expl = {"grid": {"B,x": (-sh.BOX, sh.BOX, step), "B,y": (-sh.BOX, sh.BOX, step)}}
    multi = 2 if threads == 2 else False
    om.MAX_THREADS = pm.MAX_THREADS = threads
    fit_only = ["B,x", "B,y", "B,f"]

    def search():
        with quiet():
            oi.gridFit(expl, model=model_dict(case, start), fitOnly=fit_only, multi=multi, maxfev=GRID_MAXFEV,
                       ftol=1e-5, epsfcn=EPSFCN, verbose=0)
        return read_best(oi.bestfit, case)

    with sw("t_cold"):
        best, sigma = search()
    for _ in range(sh.warm_n(size, sw.times["t_cold"])):
        with sw("t_warm", repeat=True):
            search()
    acc, ndat = sh.accuracy_block(case, files, best, sigma)
    acc["within_3_sigma_of_reference"] = bool(all(abs(best[k] - ref[k]) < 3 * sigma[k] for k in sh.free_names(case)))
    return {"n_data": ndat, "n_param": len(sh.free_names(case)),
            "iterations": {"n_starts": n * n, "grid_shape": [n, n], "nfev_best": int(oi.bestfit["info"]["nfev"])},
            "accuracy": acc,
            "settings": {"config": config, "step_mas": step, "multi": multi, "maxfev": GRID_MAXFEV, "ftol": 1e-5,
                         "epsfcn": EPSFCN, "box_mas": sh.BOX, "MAX_THREADS": threads}}


def task_pulls(pm, om, sw, case, config, size, **_):
    files = sh.pull_files(case, size)
    ref = sh.reference(case)
    start = sh.start_values(case)
    ftol = sh.T2_TOL["matched" if config != "tight" else "tight"]["ftol"]
    names = sh.free_names(case)
    best, sig, bad = [], [], 0
    with sw("t_cold"):  # the whole loop: read file, fit
        for f in files:
            try:
                oi = load(pm, [f])
                b, s, _, _ = fit_pm(oi, case, start, ftol)
            except Exception:  # noqa: BLE001 - a failed seed is counted, not hidden
                bad += 1
                continue
            best.append([b[k] for k in names])
            sig.append([s[k] for k in names])
    acc = sh.pull_summary(best, sig, ref, names)
    acc["n_failed"] = bad
    acc["noise_model"] = sh.PULL_CASES[case][0]
    return {"n_data": 300, "n_param": len(names), "iterations": {"seeds": len(files)}, "accuracy": acc,
            "settings": {"ftol": ftol, "epsfcn": EPSFCN, "maxfev": MAXFEV, "start": start}}


def task_import(pm, om, sw, **_):
    with sw("t_cold"):
        pass
    return {"n_data": 0, "n_param": 0, "iterations": {}, "accuracy": {}, "settings": {"import_only": True},
            "baseline_mb": common.rusage_peak_mb()}


TASK = {"T1": task_t1, "T2": task_t2, "T3": task_t3, "pulls": task_pulls, "import": task_import}


def worker(args):
    sw = common.Stopwatch()
    sh.calib(sw)
    with sw("t_import"):
        with quiet():
            import pmoired as pm
            import pmoired.oimodels as om
    if args.task == "gate":
        out = task_gate(pm, om, sw)
        common.emit_inner(sw, n_data=0, n_param=3, iterations={}, accuracy=out, settings={"gate": True},
                          provenance=sh.provenance_extra(), t_cold=0.0, t_warm=[])
        return
    payload = TASK[args.task](pm, om, sw, case=args.case, config=args.config, size=args.size, threads=args.threads)
    prov = sh.provenance_extra()
    prov["pmoired_version"] = getattr(pm, "__version__", None)
    flags = sh.plan_flags(sw, args.task) + ["mem_children_not_summed"] if args.task == "T3" and args.threads == 2 else sh.plan_flags(sw, args.task)
    payload["settings"].update({"code": "pmoired", "case": args.case, "config": args.config, "size": args.size})
    common.emit_inner(sw, provenance=prov, flags=flags, **payload)


def main():
    args = sh.parser("pmoired").parse_args()
    if args.worker:
        worker(args)
    else:
        sh.outer(args, "pmoired", Path(__file__).resolve())


if __name__ == "__main__":
    main()
