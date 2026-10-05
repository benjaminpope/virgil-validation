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

``aggregate`` applies the criteria registered in design/detection_v2.yml
(its SHA-256 is recorded in the summary; rationale in design/detection_v2.md):
Chernoff's mixture at a fixed position (zero fraction, positive part, mean);
our simulator against virgil's gaussian_null (k-sample Anderson-Darling and
Fisher's exact test of the exceedances above virgil's 1e-2 threshold);
DetectionMC's bookkeeping against our counts; and delta_chi2 at an injected
companion's true position against max(0, sqrt(lambda) + Z)^2 (randomized
PIT and standardized mean), lambda our own noiseless chi-squared. One Holm
family over every p-value. Version 1 (brighter companions) is superseded.
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
        "flux": np.geomspace(1e-5, 0.1, 40)}
FIXED = (6.0, -4.0)
SEPS = np.array([5.0, 10.0, 15.0])
FLUXES = np.geomspace(3e-4, 3e-3, 4)  # lambda ~ 1-80: near the detection threshold

CRITERIA_FILE = ROOT / "design" / "detection_v2.yml"
VIRGIL_CHUNK = 1000  # virgil's draws per saved chunk, so a stopped task resumes


def criteria():
    import yaml

    return yaml.safe_load(CRITERIA_FILE.read_text())


def criteria_hash():
    return hashlib.sha256(CRITERIA_FILE.read_bytes()).hexdigest()


def design_hash():
    """The experiment, apart from the criteria: the array, grid, injections
    and draw code here, and the sources of the simulator and reference
    chi-squared they use. Rows are combined only when it matches."""
    import inspect

    config = {"UTS4": UTS4.tolist(), "OBS": {k: np.asarray(v).tolist() for k, v in OBS.items()},
              "GRID": {k: v.tolist() for k, v in GRID.items()}, "FIXED": FIXED, "SEPS": SEPS.tolist(),
              "FLUXES": FLUXES.tolist(), "VIRGIL_CHUNK": VIRGIL_CHUNK}
    h = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for fn in (vis_fn, observe, stats_of, one_position, draw):
        h.update(inspect.getsource(fn).encode())
    for name in ("simulate.py", "sky.py", "array.py", "chi2.py"):
        h.update((ROOT / "src" / "crosscheck" / name).read_bytes())
    return h.hexdigest()[:16]


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
            "criteria_hash": criteria_hash(), "design_hash": design_hash()}
    start = time.time()
    if args.part == "virgil":
        from virgil import detection, models
        from virgil.oidata import OIData

        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "template.fits"
            observe(path, 0.0, 0.0, 0.0, None)
            template = OIData(str(path))
        meta = out_dir / f"virgil_{args.task:04d}.json"
        chunks = []
        if meta.exists():  # resume: keep the chunks already saved
            old = json.loads(meta.read_text())
            if {k: old.get(k) for k in head} != head:
                raise SystemExit(f"{meta} was made differently: move it before rerunning")
            chunks = old["chunks"]
        n_chunks = -(-args.draws // VIRGIL_CHUNK)
        for c in range(n_chunks):
            name = f"virgil_{args.task:04d}_{c:03d}.npz"
            if name in chunks:
                continue
            n = min(VIRGIL_CHUNK, args.draws - c * VIRGIL_CHUNK)
            # chunk c's key is seed0 + c * VIRGIL_CHUNK: distinct from every other chunk and task
            mc = detection.injection_recovery(template, models.BinaryModelCartesian(0.0, 0.0, 0.0),
                                              models.BinaryModelCartesian, GRID, seed0 + c * VIRGIL_CHUNK,
                                              n_null=n, progress=False)
            mc.save(out_dir / name)
            chunks.append(name)
            tmp_meta = meta.with_suffix(".tmp")
            tmp_meta.write_text(json.dumps({**head, "complete": len(chunks) == n_chunks, "chunks": chunks,
                                            "elapsed_s": time.time() - start}))
            tmp_meta.rename(meta)
            print(f"virgil task={args.task} chunk={c} draws={n} elapsed={time.time() - start:.0f}s", flush=True)
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
    if len({f.resolve() for f in files}) != len(files):
        raise SystemExit("a results directory is given twice")
    if not files:
        raise SystemExit(f"no results found in {', '.join(map(str, args.dirs))}")
    parts = {"null": [], "inject": [], "virgil": []}
    commits, hashes, designs, keys = set(), set(), set(), []
    for f in files:
        r = json.loads(f.read_text())
        commits.add(r["virgil_commit"])
        hashes.add(r["criteria_hash"])
        designs.add(r.get("design_hash"))
        if r["part"] == "virgil":
            for c in r["chunks"]:
                keys.append(r["seed0"] + int(c.rsplit("_", 1)[1].split(".")[0]) * VIRGIL_CHUNK)
                parts["virgil"].append(DetectionMC.load(f.parent / c))
        else:
            parts[r["part"]] += r["rows"]
    if hashes != {criteria_hash()}:
        raise SystemExit(f"criteria changed since the runs: {hashes} vs {criteria_hash()}")
    if designs != {design_hash()}:
        raise SystemExit(f"the experiment changed since the runs (design {designs} vs {design_hash()})")
    if len(set(keys)) != len(keys):
        raise SystemExit("virgil: repeated chunk keys")
    if len(commits) != 1:
        raise SystemExit(f"the parts ran on different virgil commits: {commits}")
    for p in ("null", "inject"):
        seeds = [r["seed"] for r in parts[p]]
        if len(set(seeds)) != len(seeds):
            raise SystemExit(f"{p}: repeated seeds")
    spec = criteria()
    out = {"version": 2, "virgil_commit": commits.pop(), "criteria_hash": criteria_hash(),
           "design_hash": design_hash(), "pvalues": {}, "must_hold": {}, "reported": {}}
    null, inj = parts["null"], parts["inject"]
    pv = out["pvalues"]
    mc = DetectionMC.concatenate(parts["virgil"]) if parts["virgil"] else None
    statistics = ["delta_chi2", "log_bayes_factor", "max_snr"]

    # null at a fixed position: (1/2) delta_0 + (1/2) chi2_1
    fixed = np.array([r["fixed"] for r in null])
    zero = fixed <= 1e-9
    pv["chernoff_zero_fraction"] = float(stats.binomtest(int(zero.sum()), fixed.size, 0.5).pvalue)
    pv["chernoff_positive"] = float(stats.kstest(fixed[~zero], stats.chi2(1).cdf).pvalue) if (~zero).any() else 0.0
    z = (fixed.mean() - 0.5) / np.sqrt(1.25 / fixed.size)
    pv["chernoff_mean"] = float(2 * stats.norm.sf(abs(z)))

    # our simulator against virgil's, in the bulk and in the tail
    ratio = {}
    for s_ in statistics:
        mine = np.array([r["grid"][s_] for r in null])
        theirs = np.asarray(mc.null[s_]) if mc is not None else np.array([])
        if not theirs.size:
            pv[f"simulators_ad_{s_}"] = pv[f"simulators_tail_{s_}"] = 0.0
            continue
        pv[f"simulators_ad_{s_}"] = float(stats.anderson_ksamp([mine, theirs], method=stats.PermutationMethod(
            n_resamples=999, random_state=0)).pvalue)
        thr = np.quantile(theirs, 1 - 1e-2)
        table = [[int(np.sum(mine >= thr)), int(np.sum(mine < thr))], [int(np.sum(theirs >= thr)), int(np.sum(theirs < thr))]]
        pv[f"simulators_tail_{s_}"] = float(stats.fisher_exact(table).pvalue)
        for fap in (1e-2, 1e-3):  # reported: the precision validated, a 95% interval on the rate ratio
            t = np.quantile(theirs, 1 - fap)
            k1, k2 = int(np.sum(mine >= t)), int(np.sum(theirs >= t))
            lo, hi = stats.binomtest(k1, k1 + k2).proportion_ci(0.95, method="exact") if k1 + k2 else (0.0, 1.0)
            scale = theirs.size / mine.size  # odds k1/k2 -> rate ratio
            ratio[f"{s_}@{fap:g}"] = {"ours": k1, "virgil": k2,
                                      "rate_ratio_ci": [lo / (1 - lo) * scale if lo < 1 else float("inf"),
                                                        hi / (1 - hi) * scale if hi < 1 else float("inf")]}
    out["reported"]["exceedance_ratio_ci"] = ratio

    # DetectionMC's bookkeeping against our counts, on virgil's own draws
    book = {}
    if mc is not None:
        for s_ in statistics:
            v = np.sort(np.asarray(mc.null[s_]))
            n = v.size
            for fap in (1e-2, 3e-3, 1e-3):
                value = float(np.quantile(v, 1 - fap))
                k = int(np.sum(v >= value))
                ci = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
                got, lo, hi = (float(np.asarray(x)) for x in mc.false_alarm_probability(s_, value))
                thr_, _ = mc.threshold(s_, fap, n_boot=0)
                book[f"{s_}@{fap}"] = bool(abs(got - (k + 1) / (n + 1)) < 1e-12 and abs(lo - ci.low) < 1e-9
                                           and abs(hi - ci.high) < 1e-9 and abs(thr_ - value) < 1e-12)
    out["must_hold"]["bookkeeping"] = {"cases": book, "pass": bool(book) and all(book.values())}

    # injected companions at the true position: max(0, sqrt(lambda) + Z)^2
    lam = np.array([r["lam"] for r in inj])
    true = np.array([r["true"] for r in inj])
    root = np.sqrt(lam)
    atom = stats.norm.cdf(-root)
    rng = np.random.default_rng(0)
    pit = np.where(true <= 1e-9, rng.uniform(0, 1, true.size) * atom, stats.norm.cdf(np.sqrt(np.maximum(true, 0)) - root))
    pv["boundary_pit"] = float(stats.kstest(pit, "uniform").pvalue) if inj else 0.0
    # moments of X = max(0, m + Z)^2: E = (m^2 + 1) Phi(m) + m phi(m), E[X^2] from the truncated normal
    m = root
    e1 = (m**2 + 1) * stats.norm.cdf(m) + m * stats.norm.pdf(m)
    e2 = (m**4 + 6 * m**2 + 3) * stats.norm.cdf(m) + (m**3 + 5 * m) * stats.norm.pdf(m)
    zb = np.sum(true - e1) / np.sqrt(np.sum(e2 - e1**2)) if inj else np.inf
    pv["boundary_mean"] = float(2 * stats.norm.sf(abs(zb)))
    out["reported"]["lambda_range"] = [float(lam.min()), float(lam.max())] if inj else None

    # Holm over the whole family
    names = sorted(pv, key=pv.get)
    reject, k = {}, len(names)
    for rank, name in enumerate(names):
        reject[name] = pv[name] <= spec["family_alpha"] / (k - rank)
        if not reject[name]:
            for rest in names[rank + 1:]:
                reject[rest] = False
            break
    out["rejected"] = sorted(n for n, r in reject.items() if r)

    if mc is not None and inj:  # reported: completeness at virgil's 1e-3 threshold
        thr = float(np.quantile(np.asarray(mc.null["delta_chi2"]), 1 - 1e-3))
        comp = {}
        for sep in SEPS:
            for f in FLUXES:
                sel = [r for r in inj if r["sep"] == sep and np.isclose(r["flux"], f)]
                if sel:
                    comp[f"{sep:g}mas_{f:.2g}"] = float(np.mean([r["grid"]["delta_chi2"] > thr for r in sel]))
        out["reported"]["completeness"] = {"threshold_delta_chi2": thr, "by_sep_flux": comp}
    n_virgil = int(mc.n_null) if mc is not None else 0
    c = spec["counts"]
    out["counts"] = {"null": len(null), "inject": len(inj), "virgil": n_virgil}
    enough = len(null) >= c["min_null"] and len(inj) >= c["min_injected"] and n_virgil >= c["min_virgil"]
    out["pass"] = bool(enough and not out["rejected"] and out["must_hold"]["bookkeeping"]["pass"])
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
