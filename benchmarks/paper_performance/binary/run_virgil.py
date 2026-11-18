"""virgil side of the binary-fitting benchmark. Never imports PMOIRED.

    $BENCH_PYTHON binary/run_virgil.py --task T2 --case B-sim-6 --config matched --size tiny --out DIR

The default (outer) mode launches a fresh worker process under ``/usr/bin/time -l`` (``common.timed_process``) and writes
one row. ``--worker`` runs the task in the current process and prints one ``BENCH_INNER`` line. Tasks: ``gate``,
``T1`` forward model, ``T2`` local fit + Laplace covariance, ``T3`` blind grid search then the T2 fit, ``pulls`` (the
200-seed loop, timed once as a whole), ``import`` (import-only baseline). Settings are those of ``cases.yml``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _shared as sh  # noqa: E402

common, md, neutral = sh.common, sh.md, sh.neutral

PATHS = {"point": {"dra": "dra", "ddec": "ddec", "flux": "flux"},
         "ud": {"dra": "comp.dra", "ddec": "comp.ddec", "flux": "comp.flux", "ud": "star.diam"}}


def _imports(device):
    """Import virgil (float64, CPU). Returns a namespace of what the worker needs."""
    import jax

    jax.config.update("jax_enable_x64", True)
    if device != "cpu":
        jax.config.update("jax_platform_name", device)
    from types import SimpleNamespace

    import numpyro.distributions as dist
    import virgil
    from virgil.fitting import fit
    from virgil.grid_fit import best_grid_point, likelihood_grid
    from virgil.inference import laplace_cov
    from virgil.likelihood import whitened_residuals
    from virgil.models import BinaryModelCartesian, PointSource, System, UniformDisk
    from virgil.oidata import OIData

    return SimpleNamespace(
        jax=jax, dist=dist, virgil=virgil, fit=fit, best_grid_point=best_grid_point, likelihood_grid=likelihood_grid,
        laplace_cov=laplace_cov, whitened_residuals=whitened_residuals, Binary=BinaryModelCartesian,
        PointSource=PointSource, System=System, UniformDisk=UniformDisk, OIData=OIData)


def load(v, files):
    return v.OIData(str(files[0])) if len(files) == 1 else v.OIData([str(f) for f in files])


class Problem:
    """Model template, priors and parameter paths for one case (cases.yml: model.virgil_*)."""

    def __init__(self, v, case, values):
        self.v, self.case = v, case
        self.ud = "ud" in sh.reference(case)
        self.kind = "ud" if self.ud else "point"
        self.names = sh.free_names(case)
        self.paths = [PATHS[self.kind][k] for k in self.names]
        self.priors = {PATHS[self.kind]["dra"]: v.dist.Uniform(-60.0, 60.0),
                       PATHS[self.kind]["ddec"]: v.dist.Uniform(-60.0, 60.0),
                       PATHS[self.kind]["flux"]: v.dist.Uniform(0.0, 1.0)}
        if self.ud:
            self.priors[PATHS["ud"]["ud"]] = v.dist.Uniform(0.0, 5.0)
        self.template = self.make(values)

    def make(self, x):
        v = self.v
        if self.ud:
            return v.System(star=v.UniformDisk(x["ud"]), comp=v.PointSource(x["flux"], x["dra"], x["ddec"]))
        return v.Binary(x["dra"], x["ddec"], x["flux"])

    def from_values(self, values):
        return {k: float(np.asarray(values[p])) for k, p in zip(self.names, self.paths)}

    def start_template(self, start):
        t = self.template
        for k, p in zip(self.names, self.paths):
            t = t.set(p, np.asarray(start[k], float))
        return t


def fit_cov(v, prob, data, start, gtol):
    """T2: ``fit`` (LM) then ``laplace_cov``. Returns best, sigma (dicts), info, and the two times."""
    t0 = time.perf_counter()
    res = v.fit(prob.start_template(start), prob.priors, data, method="lm", gtol=gtol)
    common.block(res.values)
    t1 = time.perf_counter()
    flat = np.array([float(np.asarray(res.values[p])) for p in prob.paths])
    cov = np.asarray(v.laplace_cov(flat, prob.paths, res.model, data))
    t2 = time.perf_counter()
    best = prob.from_values(res.values)
    sigma = dict(zip(prob.names, np.sqrt(np.diag(cov))))
    info = {"steps": int(res.info["steps"]), "converged": bool(res.info["converged"]),
            "chi2_own": float(np.sum(res.info["chi2"])) if np.ndim(res.info["chi2"]) else float(res.info["chi2"]),
            "stop": res.info.get("stop")}
    return best, sigma, info, t1 - t0, t2 - t1


def own_chi2(v, prob, data, values):
    r = np.asarray(v.whitened_residuals(prob.make(values), data))
    return float(np.sum(r**2)), int(r.size)


def likelihood_mismatch(v, prob, data, files, case):
    """Relative difference between virgil's chi2 and the neutral independent-Gaussian chi2 at the reference."""
    ref = sh.reference(case)
    c_own, n_rows = own_chi2(v, prob, data, ref)
    c_neu = sh.neutral_chi2(sh.read_neutral(files), ref)["chi2"]
    return {"chi2_virgil_at_reference": c_own, "chi2_neutral_at_reference": c_neu, "n_rows_virgil": n_rows,
            "likelihood_mismatch_rel": (c_own - c_neu) / c_neu}


# ---------------------------------------------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------------------------------------------


def task_gate(v, sw):
    out = {}
    for case in ("B-sim-6", "B-sim-ud"):
        ref = sh.reference(case)
        clean = md.DATA / f"{md.SIM_FILE[case]}_clean.fits"
        noisy = sh.case_files(case)
        bad = md.verify([clean.name])
        if bad:
            raise SystemExit(f"{bad} missing or changed")
        data = load(v, [clean])
        prob = Problem(v, case, ref)
        model = prob.make(ref)
        vec = np.asarray(data.standardize_model(model.model(data.u, data.v, data.wavel)))
        dat = np.asarray(data.flatten_data()[0])
        nv, npi = int(np.asarray(data.vis).size), int(np.asarray(data.phi).size)
        dv2 = float(np.max(np.abs(vec[:nv] - dat[:nv])))
        dcp = np.rad2deg(np.abs(neutral.wrap(vec[nv:nv + npi] - dat[nv:nv + npi])))
        out[case] = {"v2_max_abs_diff": dv2, "cp_max_abs_diff_deg": float(np.max(dcp)), "n_v2": nv, "n_cp": npi}
        data_n = load(v, noisy)
        out[case].update(likelihood_mismatch(v, prob, data_n, noisy, case))
    return out


def task_t1(v, sw, case, size, **_):
    files = sh.case_files(case)
    with sw("t_load"):
        data = load(v, files)
    ref = sh.reference(case)
    prob = Problem(v, case, ref)
    jax = v.jax
    if prob.ud:
        def fwd(ud, dra, ddec, flux):
            m = v.System(star=v.UniformDisk(ud), comp=v.PointSource(flux, dra, ddec))
            return data.standardize_model(m.model(data.u, data.v, data.wavel))
        args = (ref["ud"], ref["dra"], ref["ddec"], ref["flux"])
    else:
        def fwd(dra, ddec, flux):
            return data.standardize_model(v.Binary(dra, ddec, flux).model(data.u, data.v, data.wavel))
        args = (ref["dra"], ref["ddec"], ref["flux"])
    jf = jax.jit(fwd)
    args = tuple(np.float64(a) for a in args)
    with sw("t_cold"):
        common.block(jf(*args))
    n_calls = sh.SIZE[size]["t1_calls"]
    for _ in range(sh.SIZE[size]["warm"]):
        with sw("t_warm", repeat=True):
            for _ in range(n_calls):
                common.block(jf(*args))
    sw.lists["t_warm"] = [t / n_calls for t in sw.lists["t_warm"]]
    return {"n_data": int(np.asarray(data.vis).size + np.asarray(data.phi).size), "n_param": len(prob.names),
            "iterations": {"calls_per_warm_batch": n_calls},
            "accuracy": {"model_vector_size": int(jf(*args).size)}, "settings": {"warm_is": "seconds per call"}}


def _t2_row(v, sw, case, config, size, files, data, prob, repeat_start=None, extra_it=None):
    """Cold then warm repeats of T2 (or of T3's second stage if ``repeat_start``); returns the payload."""
    start = repeat_start or sh.start_values(case)
    tol = sh.T2_TOL[config]["gtol"]
    with sw("t_cold"):
        best, sigma, info, t_fit, t_lap = fit_cov(v, prob, data, start, tol)
    iters = {**info, "t_fit_cold": t_fit, "t_laplace_cold": t_lap, **(extra_it or {})}
    fits, laps = [], []
    for _ in range(sh.warm_n(size, sw.times["t_cold"])):
        with sw("t_warm", repeat=True):
            _, _, _, tf, tl = fit_cov(v, prob, data, start, tol)
        fits.append(tf)
        laps.append(tl)
    iters.update({"t_fit_warm": fits, "t_laplace_warm": laps})
    acc, ndat = sh.accuracy_block(case, files, best, sigma)
    acc.update(likelihood_mismatch(v, prob, data, files, case))
    return {"n_data": ndat, "n_param": len(prob.names), "iterations": iters, "accuracy": acc,
            "settings": {"method": "lm", "gtol": tol, "start": start}}


def task_t2(v, sw, case, config, size, **_):
    files = sh.case_files(case)
    with sw("t_load"):
        data = load(v, files)
    prob = Problem(v, case, sh.reference(case))
    return _t2_row(v, sw, case, config, size, files, data, prob)


def _grid(v, data, config, size, case):
    """Return (grid dict, callable running the search and returning the best point)."""
    ref = sh.reference(case)
    if config == "native":
        n, nf = sh.T3_NATIVE["virgil_dra_ddec"], sh.T3_NATIVE["virgil_flux"]
        if size == "tiny":
            n, nf = 9, 4
        grid = {"dra": np.linspace(-sh.BOX, sh.BOX, n), "ddec": np.linspace(-sh.BOX, sh.BOX, n),
                "flux": 10.0 ** np.linspace(-2.0, 0.0, nf)}
    else:
        n = sh.grid_points(config, size)
        grid = {"dra": np.linspace(-sh.BOX, sh.BOX, n), "ddec": np.linspace(-sh.BOX, sh.BOX, n),
                "flux": np.array([sh.FLUX_START_GRID])}
    return grid


def task_t3(v, sw, case, config, size, **_):
    files = sh.case_files(case)
    with sw("t_load"):
        data = load(v, files)
    prob = Problem(v, case, sh.reference(case))
    grid = _grid(v, data, config, size, case)
    n_starts = int(np.prod([len(g) for g in grid.values()]))
    tol = sh.T2_TOL["matched" if config != "tight" else "tight"]["gtol"]
    times = {}

    def search():
        t0 = time.perf_counter()
        ll = common.block(v.likelihood_grid(v.Binary, data=data, grid=grid))
        t1 = time.perf_counter()
        pt = v.best_grid_point(ll, grid)
        best, sigma, info, tf, tl = fit_cov(v, prob, data, pt, tol)
        times.update({"t_grid": t1 - t0, "t_best_and_fit": time.perf_counter() - t1, "t_fit": tf, "t_laplace": tl})
        return pt, best, sigma, info

    with sw("t_cold"):
        pt, best, sigma, info = search()
    cold = dict(times)
    for _ in range(sh.warm_n(size, sw.times["t_cold"])):
        with sw("t_warm", repeat=True):
            search()
    acc, ndat = sh.accuracy_block(case, files, best, sigma)
    ref = sh.reference(case)
    acc["best_grid_point"] = {k: float(pt[k]) for k in pt}
    acc["within_3_sigma_of_reference"] = bool(all(abs(best[k] - ref[k]) < 3 * sigma[k] for k in prob.names))
    acc.update(likelihood_mismatch(v, prob, data, files, case))
    return {"n_data": ndat, "n_param": len(prob.names),
            "iterations": {**info, "n_starts": n_starts, "grid_shape": [len(g) for g in grid.values()],
                           **{f"{k}_cold": x for k, x in cold.items()}},
            "accuracy": acc, "settings": {"config": config, "gtol": tol, "box_mas": sh.BOX,
                                          "grid_axes": list(grid), "flux_axis": [float(grid["flux"][0]),
                                                                                    float(grid["flux"][-1])]}}


def task_pulls(v, sw, case, config, size, **_):
    files = sh.pull_files(case, size)
    ref = sh.reference(case)
    prob = Problem(v, case, ref)
    start = sh.start_values(case)
    tol = sh.T2_TOL["matched" if config != "tight" else "tight"]["gtol"]
    best, sig, bad = [], [], 0
    first_data = None
    with sw("t_cold"):  # the whole loop: read file, fit, covariance
        for f in files:
            data = load(v, [f])
            first_data = first_data or data
            try:
                b, s, _, _, _ = fit_cov(v, prob, data, start, tol)
            except Exception:  # noqa: BLE001 - a failed seed is counted, not hidden
                bad += 1
                continue
            best.append([b[k] for k in prob.names])
            sig.append([s[k] for k in prob.names])
    acc = sh.pull_summary(best, sig, ref, prob.names)
    acc["n_failed"] = bad
    acc["noise_model"] = sh.PULL_CASES[case][0]
    acc.update(likelihood_mismatch(v, prob, first_data, files[:1], case))
    nd = int(np.asarray(first_data.vis).size + np.asarray(first_data.phi).size)
    return {"n_data": nd, "n_param": len(prob.names), "iterations": {"seeds": len(files)}, "accuracy": acc,
            "settings": {"method": "lm", "gtol": tol, "start": start}}


def task_import(v, sw, **_):
    with sw("t_cold"):
        pass
    return {"n_data": 0, "n_param": 0, "iterations": {}, "accuracy": {}, "settings": {"import_only": True},
            "baseline_mb": common.rusage_peak_mb()}


TASK = {"T1": task_t1, "T2": task_t2, "T3": task_t3, "pulls": task_pulls, "import": task_import}


def worker(args):
    sw = common.Stopwatch()
    sh.calib(sw)
    with sw("t_import"):
        v = _imports(args.device)
    if args.task == "gate":
        out = task_gate(v, sw)
        common.emit_inner(sw, n_data=0, n_param=3, iterations={}, accuracy=out, settings={"gate": True},
                          provenance=sh.provenance_extra(), t_cold=0.0, t_warm=[])
        return
    payload = TASK[args.task](v, sw, case=args.case, config=args.config, size=args.size)
    prov = sh.provenance_extra()
    prov["virgil_version"] = v.virgil.__version__
    flags = sh.plan_flags(sw, args.task)
    mm = payload["accuracy"].get("likelihood_mismatch_rel")
    if mm is not None and abs(mm) > 1e-6:
        flags.append("likelihood_mismatch")
    payload["settings"].update({"code": "virgil", "device": args.device, "case": args.case, "config": args.config,
                                "size": args.size})
    common.emit_inner(sw, provenance=prov, flags=flags, **payload)


def main():
    args = sh.parser("virgil").parse_args()
    if args.worker:
        worker(args)
    else:
        sh.outer(args, "virgil", Path(__file__).resolve())


if __name__ == "__main__":
    main()
