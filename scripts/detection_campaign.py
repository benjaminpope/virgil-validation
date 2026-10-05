"""False-alarm and contrast calibration of virgil.detection (Stage 8).

virgil's detection statistics (detection_statistics: delta_chi2,
log_bayes_factor, max_snr) and its Monte Carlo (injection_recovery,
DetectionMC) are checked against simulations that are ours, not virgil's:
crosscheck.simulate.observe writes a four-UT VLTI file (correlated closure
phases), with or without a companion, and virgil searches it.

Parts, each an OzSTAR array (``run --part P --task T --draws N``):

* ``null``: companion-free files from our simulator. Records the grid
  statistics, and delta_chi2 at one fixed position.
* ``inject``: a companion at a random position angle, at separations SEPS
  and fluxes FLUXES. Records the grid statistics, delta_chi2 at the true
  position, and lambda, our own noiseless chi-squared of the injected data
  under the null model (the noncentrality).
* ``virgil``: virgil's own null simulations (injection_recovery with
  gaussian_null) of the same template, saved as a DetectionMC.

``aggregate`` applies the criteria registered in CRITERIA (hash recorded):

1. Chernoff (1954), as detection_statistics documents: at a fixed position
   the null delta_chi2 is 0 with probability 1/2 and chi2_1 otherwise.
2. virgil's null simulator gives the distributions of ours (two-sample KS
   on delta_chi2, log_bayes_factor and max_snr).
3. DetectionMC's false-alarm probabilities equal the documented (k + 1)/(n + 1)
   from our counts, their intervals SciPy's exact binomial interval, and
   its thresholds our empirical quantiles.
4. Injected companions: delta_chi2 at the true position is noncentral
   chi2_1(lambda), tested by the probability integral transform (uniform
   under the claim) where lambda >= 9 (the flux constraint then rarely binds).
"""

import argparse
import hashlib
import json
import pathlib
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

UTS4 = np.array([[-9.925, -20.335], [14.887, 30.502], [44.915, 66.183], [103.306, 43.999]])
OBS = dict(hour_angles_h=np.linspace(-3, 3, 5), wavelengths=np.linspace(1.5e-6, 2.4e-6, 4), dec_deg=-50.0,
           sigma_v2=0.01, sigma_cp_deg=0.5)
GRID = {"dra": np.linspace(-20.0, 20.0, 17), "ddec": np.linspace(-20.0, 20.0, 17),
        "flux": np.geomspace(1e-4, 0.1, 40)}
FIXED = (6.0, -4.0)
SEPS = np.array([5.0, 10.0, 15.0])
FLUXES = np.array([0.004, 0.008, 0.016])

CRITERIA = {
    "alpha": 1e-3,  # every p-value and band below, per test
    "chernoff_zero_band": 0.999,  # binomial band on the fraction of zeros (p = 1/2)
    "ks_statistics": ["delta_chi2", "log_bayes_factor", "max_snr"],
    "fap_values": [1e-2, 3e-3, 1e-3],  # bookkeeping: DetectionMC against our counts
    "noncentral_min_lambda": 9.0,
    "min_null": 5000,
    "min_injected": 5000,
}


def criteria_hash():
    return hashlib.sha256(json.dumps(CRITERIA, sort_keys=True).encode()).hexdigest()[:16]


def vis_fn(f, x, y):
    from crosscheck import sky

    def vis(u, v, w):
        return (sky.vis_point(u, v, w) + f * sky.vis_point(u, v, w, x, y)) / (1 + f)

    return vis


def observe(path, f, x, y, rng):
    from crosscheck import simulate

    simulate.observe(path, vis_fn(f, x, y), UTS4, rng=rng, **OBS)


def virgil_commit():
    import os

    if os.environ.get("PIN_COMMIT"):
        return os.environ["PIN_COMMIT"]
    from evidence.meta import virgil_source

    return virgil_source()["commit"]


def stats_of(data, grid):
    import warnings

    from virgil import detection, models

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # flux-axis diagnostics on noise draws
        s = detection.detection_statistics(data, models.BinaryModelCartesian, grid)
    return {k: float(np.asarray(v)) for k, v in s.items()}


def one_position(x, y):
    return {"dra": np.array([x]), "ddec": np.array([y]), "flux": GRID["flux"]}


def draw(part, seed, tmp):
    from crosscheck import chi2 as ours
    from virgil.oidata import OIData

    rng = np.random.default_rng(seed)
    path = pathlib.Path(tmp) / f"{seed}.fits"
    out = {"seed": int(seed)}
    if part == "null":
        observe(path, 0.0, 0.0, 0.0, rng)
        data = OIData(str(path))
        out["grid"] = stats_of(data, GRID)
        out["fixed"] = stats_of(data, one_position(*FIXED))["delta_chi2"]
    else:
        sep = float(rng.choice(SEPS))
        f = float(rng.choice(FLUXES))
        pa = float(rng.uniform(0.0, 360.0))
        x, y = sep * np.sin(np.deg2rad(pa)), sep * np.cos(np.deg2rad(pa))
        clean = pathlib.Path(tmp) / f"{seed}_clean.fits"
        observe(clean, f, x, y, None)  # noiseless: lambda is chi2 of the null model to these data
        lam = ours.chi2(ours.load(clean), vis_fn(0.0, 0.0, 0.0), correlated=True, sine=True)
        observe(path, f, x, y, rng)
        data = OIData(str(path))
        out.update(sep=sep, flux=f, pa=pa, dra=x, ddec=y, lam=float(lam))
        out["grid"] = stats_of(data, GRID)
        out["true"] = stats_of(data, one_position(x, y))["delta_chi2"]
    return out


def run(args):
    import jax

    jax.config.update("jax_enable_x64", True)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = virgil_commit()
    if not commit:
        raise SystemExit("cannot tell which virgil commit is running: set PIN_COMMIT")
    seed0 = args.seed_base + {"null": 0, "inject": 10_000_000, "virgil": 20_000_000}[args.part] + args.task * args.draws
    head = {"part": args.part, "task": args.task, "draws": args.draws, "seed0": seed0, "virgil_commit": commit,
            "criteria_hash": criteria_hash()}
    start = time.time()
    if args.part == "virgil":
        from virgil import detection, models
        from virgil.oidata import OIData

        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "template.fits"
            observe(path, 0.0, 0.0, 0.0, None)
            template = OIData(str(path))
        mc = detection.injection_recovery(template, models.BinaryModelCartesian(0.0, 0.0, 0.0),
                                          models.BinaryModelCartesian, GRID, seed0, n_null=args.draws,
                                          progress=False)
        mc.save(out_dir / f"virgil_{args.task:04d}.npz")
        (out_dir / f"virgil_{args.task:04d}.json").write_text(json.dumps({**head, "elapsed_s": time.time() - start}))
        print(f"virgil task={args.task} draws={args.draws} elapsed={time.time() - start:.0f}s")
        return
    path = out_dir / f"{args.part}_{args.task:04d}.json"
    rows = []
    if path.exists():  # resume after a time limit
        old = json.loads(path.read_text())
        if {k: old.get(k) for k in head} != head:
            raise SystemExit(f"{path} was made differently: move it before rerunning")
        rows = old["rows"]
    done = {r["seed"] for r in rows}
    with tempfile.TemporaryDirectory() as tmp:
        for k in range(args.draws):
            seed = seed0 + k
            if seed in done:
                continue
            t0 = time.time()
            rows.append(draw(args.part, seed, tmp))
            tmp_path = path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps({**head, "complete": len(rows) == args.draws, "rows": rows}))
            tmp_path.rename(path)
            print(f"{args.part} seed={seed} elapsed={time.time() - t0:.2f}s", flush=True)
    print(f"wrote {path} elapsed={time.time() - start:.0f}s")


def aggregate(args):
    from scipy import stats

    from virgil.detection import DetectionMC

    files = [f for d in args.dirs for f in sorted(pathlib.Path(d).glob("*_*.json"))]
    if not files:
        raise SystemExit(f"no results found in {', '.join(map(str, args.dirs))}")
    parts = {"null": [], "inject": [], "virgil": []}
    commits, hashes = set(), set()
    for f in files:
        r = json.loads(f.read_text())
        commits.add(r["virgil_commit"])
        hashes.add(r["criteria_hash"])
        if r["part"] == "virgil":
            parts["virgil"].append(DetectionMC.load(f.with_suffix(".npz")))
        else:
            parts[r["part"]] += r["rows"]
    if hashes != {criteria_hash()}:
        raise SystemExit(f"criteria changed since the runs: {hashes} vs {criteria_hash()}")
    if len(commits) != 1:
        raise SystemExit(f"the parts ran on different virgil commits: {commits}")
    for p in ("null", "inject"):
        seeds = [r["seed"] for r in parts[p]]
        if len(set(seeds)) != len(seeds):
            raise SystemExit(f"{p}: repeated seeds")
    a = CRITERIA["alpha"]
    out = {"virgil_commit": commits.pop(), "criteria": CRITERIA, "criteria_hash": criteria_hash(), "checks": {}}
    null, inj = parts["null"], parts["inject"]
    checks = out["checks"]

    # 1. Chernoff at a fixed position
    fixed = np.array([r["fixed"] for r in null])
    zero = fixed <= 1e-9
    lo, hi = stats.binom.interval(CRITERIA["chernoff_zero_band"], fixed.size, 0.5)
    ks = stats.kstest(fixed[~zero], stats.chi2(1).cdf).pvalue if (~zero).any() else 0.0
    checks["chernoff"] = {"n": int(fixed.size), "zero_fraction": float(zero.mean()), "band": [lo / fixed.size, hi / fixed.size],
                          "ks_p_positive_vs_chi2_1": float(ks),
                          "pass": bool(lo <= zero.sum() <= hi and ks > a)}

    # 2. virgil's simulator against ours
    mc = DetectionMC.concatenate(parts["virgil"]) if parts["virgil"] else None
    sim = {}
    for s in CRITERIA["ks_statistics"]:
        mine = np.array([r["grid"][s] for r in null])
        theirs = np.asarray(mc.null[s]) if mc is not None else np.array([])
        p = float(stats.ks_2samp(mine, theirs).pvalue) if theirs.size else 0.0
        sim[s] = {"p": p, "ours_mean": float(mine.mean()), "virgil_mean": float(theirs.mean()) if theirs.size else None}
    checks["simulators"] = {"n_ours": len(null), "n_virgil": int(mc.n_null) if mc is not None else 0, "stats": sim,
                            "pass": all(v["p"] > a / len(sim) for v in sim.values())}

    # 3. DetectionMC bookkeeping against our counts, on virgil's null draws
    book = {}
    if mc is not None:
        for s in CRITERIA["ks_statistics"]:
            v = np.sort(np.asarray(mc.null[s]))
            n = v.size
            for fap in CRITERIA["fap_values"]:
                value = float(np.quantile(v, 1 - fap))
                k = int(np.sum(v >= value))
                # the documented Monte Carlo p-value (k + 1)/(n + 1), and the exact
                # binomial interval for P(null >= value) from SciPy's binomtest
                ci = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
                got, lo, hi = (float(np.asarray(x)) for x in mc.false_alarm_probability(s, value))
                thr, _ = mc.threshold(s, fap, n_boot=0)
                equal = (abs(got - (k + 1) / (n + 1)) < 1e-12 and abs(lo - ci.low) < 1e-9
                         and abs(hi - ci.high) < 1e-9 and abs(thr - value) < 1e-12)
                book[f"{s}@{fap}"] = {"k": k, "n": n, "fap": got, "ci": [lo, hi], "threshold": float(thr),
                                      "equal": bool(equal)}
    checks["bookkeeping"] = {"cases": book, "pass": bool(book) and all(c["equal"] for c in book.values())}

    # 4. noncentral chi2_1 at the true position
    lam = np.array([r["lam"] for r in inj])
    true = np.array([r["true"] for r in inj])
    keep = lam >= CRITERIA["noncentral_min_lambda"]
    u = stats.ncx2.cdf(true[keep], 1, lam[keep])
    p = float(stats.kstest(u, "uniform").pvalue) if keep.any() else 0.0
    checks["noncentral"] = {"n": int(keep.sum()), "lambda_range": [float(lam[keep].min()), float(lam[keep].max())] if keep.any() else None,
                            "ks_p_pit": p, "pass": p > a}

    # reported, not criteria: completeness of the grid search at the 0.1 % FAP threshold
    if mc is not None and inj:
        thr = float(np.quantile(np.asarray(mc.null["delta_chi2"]), 1 - 1e-3))
        comp = {}
        for sep in SEPS:
            for f in FLUXES:
                sel = [r for r in inj if r["sep"] == sep and r["flux"] == f]
                if sel:
                    comp[f"{sep:g}mas_{f:g}"] = float(np.mean([r["grid"]["delta_chi2"] > thr for r in sel]))
        out["completeness_at_fap_1e-3"] = {"threshold_delta_chi2": thr, "by_sep_flux": comp}
    enough = len(null) >= CRITERIA["min_null"] and len(inj) >= CRITERIA["min_injected"]
    out["counts"] = {"null": len(null), "inject": len(inj), "virgil": int(mc.n_null) if mc is not None else 0}
    out["pass"] = bool(enough and all(c["pass"] for c in checks.values()))
    text = json.dumps(out, indent=1)
    if args.summary:
        pathlib.Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.summary).write_text(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--part", choices=["null", "inject", "virgil"], required=True)
    r.add_argument("--task", type=int, required=True)
    r.add_argument("--draws", type=int, default=100)
    r.add_argument("--seed-base", type=int, default=20261006)
    r.add_argument("--out", required=True)
    g = sub.add_parser("aggregate")
    g.add_argument("dirs", nargs="+")
    g.add_argument("--summary")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else aggregate(args)


if __name__ == "__main__":
    main()
