"""Pull campaign: are virgil's fits unbiased and its Laplace uncertainties
right? (Criteria and design registered in design/pull_campaign.yml, whose
SHA-256 is recorded in every task output and in the summary.)

For each cell (a scene and a fitter), N >= 1000 noisy realisations from our
simulator (crosscheck.simulate.observe) are fitted by virgil, and each
parameter's pull (fit - truth) / sigma is recorded. The mean and the sd of the
pulls are tested against 0 and 1, under one Holm family at 1%. The pull tests
of a few hundred draws in tests/ are regression smoke tests; this is the
calibration claim.

    python scripts/pull_campaign.py run --cell C --task T [--draws N] --out DIR
    python scripts/pull_campaign.py aggregate DIR [DIR ...] --summary FILE

``run`` writes DIR/pull_<cell>_<T>.json after every draw (a task stopped by
its time limit resumes). ``aggregate`` applies the registered criteria.
"""

import argparse
import hashlib
import inspect
import json
import pathlib
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CRITERIA_FILE = ROOT / "design" / "pull_campaign.yml"
UTS4 = np.array([[-9.925, -20.335], [14.887, 30.502], [44.915, 66.183], [103.306, 43.999]])
UTS3 = np.array([[-9.925, -20.335], [14.887, 30.502], [103.306, 43.999]])
HA4, HA3 = np.linspace(-3, 3, 5), np.linspace(-3, 3, 7)
WL = np.linspace(1.5e-6, 2.4e-6, 6)
GRID_TRUTH = (6.0, -4.0, 0.03)  # dra, ddec, flux
DISK_DIAM = 1.8
SCENES = ("binary", "resolved_star_companion", "star_envelope", "star_rim")


def spec():
    import yaml

    return yaml.safe_load(CRITERIA_FILE.read_text())


def criteria_hash():
    return hashlib.sha256(CRITERIA_FILE.read_bytes()).hexdigest()


def design_hash():
    """The experiment the registered file describes only in words: the
    arrays, grids, truths and the code that draws and fits, and our
    simulator's and scenes' sources."""
    config = {"UTS4": UTS4.tolist(), "UTS3": UTS3.tolist(), "HA4": HA4.tolist(), "HA3": HA3.tolist(),
              "WL": WL.tolist(), "GRID_TRUTH": GRID_TRUTH, "DISK_DIAM": DISK_DIAM, "SCENES": list(SCENES)}
    h = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for fn in (draw, param_names):
        h.update(inspect.getsource(fn).encode())
    for name in ("simulate.py", "sky.py", "array.py"):
        h.update((ROOT / "src" / "crosscheck" / name).read_bytes())
    h.update((ROOT / "src" / "virgil_bridge" / "__init__.py").read_bytes())
    return h.hexdigest()[:16]


def virgil_commit():
    import os

    if os.environ.get("PIN_COMMIT"):
        return os.environ["PIN_COMMIT"]
    from evidence.meta import virgil_source

    return virgil_source()["commit"]


def scene_of(name):
    import virgil_bridge as vb

    return getattr(vb, "binary" if name == "uniform_disk" else name)()


def param_names(cell):
    """Parameter labels of a cell, in the order of its pulls."""
    c = spec()["cells"][cell]
    if c["fitter"] == "grid_laplace":
        return ["flux"]
    if c["scene"] == "uniform_disk":
        return ["diam"]
    scene = scene_of(c["scene"])
    names = []
    for p, v in scene.truth.items():
        n = np.size(v)
        names += [p] if n == 1 else [f"{p}[{i}]" for i in range(n)]
    return names


def draw(cell, seed, tmp):
    """One noisy realisation of a cell, fitted: its pulls (NaN where the fit
    or its uncertainty failed)."""
    import numpyro.distributions as dist
    import virgil.models as vm
    import virgil_bridge as vb
    from crosscheck import simulate, sky
    from virgil.fitting import fit
    from virgil.grid_fit import laplace_flux_uncertainty_grid, optimized_flux_grid
    from virgil.inference import laplace_cov
    from virgil.oidata import OIData

    c = spec()["cells"][cell]
    rng = np.random.default_rng(seed)
    path = pathlib.Path(tmp) / f"{seed}.fits"
    if c["fitter"] == "grid_laplace":
        dra, ddec, flux = GRID_TRUTH

        def vis(u, v, w):
            return (sky.vis_point(u, v, w) + flux * sky.vis_point(u, v, w, dra, ddec)) / (1 + flux)

        simulate.observe(path, vis, UTS3, hour_angles_h=HA3, wavelengths=WL, dec_deg=-50.0, sigma_v2=0.01,
                         sigma_cp_deg=0.5, rng=rng)
        data = OIData(str(path))
        samples = {"dra": [dra], "ddec": [ddec], "flux": [0.0, flux, 0.06]}
        best = np.asarray(optimized_flux_grid(data, vm.BinaryModelCartesian, samples))[0, 0]
        sig = np.asarray(laplace_flux_uncertainty_grid(data, vm.BinaryModelCartesian, samples, flux=[[best]]))[0, 0]
        return [float((best - flux) / sig)]
    if c["scene"] == "uniform_disk":
        simulate.observe(path, lambda u, v, w: sky.vis_uniform_disk(u, v, w, DISK_DIAM), UTS4, hour_angles_h=HA4,
                         wavelengths=WL, dec_deg=-50.0, sigma_v2=0.02, sigma_cp_deg=1.0, rng=rng,
                         closure_phases=False)
        data = OIData(str(path))
        result = fit(vm.UniformDisk(1.5), {"diam": dist.Uniform(0.1, 10.0)}, data)
        d = float(result.values["diam"])
        cov = np.asarray(laplace_cov(np.array([d]), ["diam"], data, result.model))
        return [float((d - DISK_DIAM) / np.sqrt(cov[0, 0]))]
    scene = scene_of(c["scene"])
    simulate.observe(path, scene.vis, UTS4, hour_angles_h=HA4, wavelengths=WL, dec_deg=-50.0, sigma_v2=0.02,
                     sigma_cp_deg=1.0, rng=rng, phase_noise="baseline")
    data = vb.load(path)
    result, cov = vb.fit_scene(scene, data, start=scene.truth)  # as the regression pull test: from the truth
    pulls = (vb.flat_values(scene, result.values) - vb.flat_truth(scene)) / np.sqrt(np.diag(cov))
    return [float(p) for p in pulls]


def run(args, draw_fn=None):
    import jax

    jax.config.update("jax_enable_x64", True)
    s = spec()
    if args.cell not in s["cells"]:
        raise SystemExit(f"unknown cell {args.cell}: {sorted(s['cells'])}")
    draws = args.draws or s["counts"]["draws_per_task"]
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = virgil_commit()
    if not commit:
        raise SystemExit("cannot tell which virgil commit is running: set PIN_COMMIT")
    cell_index = list(s["cells"]).index(args.cell)
    seed0 = args.seed_base + cell_index * 1_000_000 + args.task * draws
    head = {"cell": args.cell, "task": args.task, "draws": draws, "seed0": seed0, "virgil_commit": commit,
            "criteria_hash": criteria_hash(), "design_hash": design_hash(), "params": param_names(args.cell)}
    path = out_dir / f"pull_{args.cell}_{args.task:04d}.json"
    rows, start = [], time.time()
    if path.exists():  # resume after a time limit
        old = json.loads(path.read_text())
        if {k: old.get(k) for k in head} != head:
            raise SystemExit(f"{path} was made differently: move it before rerunning")
        rows = old["rows"]
    done = {r["seed"] for r in rows}
    draw_fn = draw_fn or draw
    with tempfile.TemporaryDirectory() as tmp:
        for k in range(draws):
            seed = seed0 + k
            if seed in done:
                continue
            t0 = time.time()
            try:
                pulls = draw_fn(args.cell, seed, tmp)
            except Exception as e:  # a failed fit is a counted failure, not a crash
                print(f"seed={seed} failed: {type(e).__name__}: {e}", flush=True)
                pulls = [float("nan")] * len(head["params"])
            rows.append({"seed": seed, "pulls": [p if np.isfinite(p) else None for p in pulls]})
            tmp_path = path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps({**head, "complete": len(rows) == draws, "rows": rows}))
            tmp_path.rename(path)
            print(f"{args.cell} seed={seed} elapsed={time.time() - t0:.2f}s", flush=True)
    print(f"wrote {path} elapsed={time.time() - start:.0f}s")


def holm(pvalues, alpha):
    names = sorted(pvalues, key=pvalues.get)
    m, rejected, stop = len(names), {}, False
    for i, name in enumerate(names):
        stop = stop or pvalues[name] > alpha / (m - i)
        rejected[name] = not stop
    return rejected


def analyse(cells, s):
    """The registered analysis of {cell: (param names, rows of pulls)}."""
    from scipy import stats

    out = {"cells": {}, "pvalues": {}}
    pv = out["pvalues"]
    band = s["bands_sigma"]
    for cell, (names, rows) in cells.items():
        pulls = np.array([[np.nan if p is None else p for p in r["pulls"]] for r in rows], float).reshape(len(rows), len(names))
        finite = np.all(np.isfinite(pulls), axis=1)
        good = pulls[finite]
        n = good.shape[0]
        entry = {"draws": len(rows), "finite": int(n), "nonfinite_fraction": 1 - n / max(len(rows), 1),
                 "enough": bool(n >= s["counts"]["min_draws_per_cell"]), "params": {}}
        entry["failures_ok"] = entry["nonfinite_fraction"] <= s["fit_failures"]["max_nonfinite_fraction"]
        for j, name in enumerate(names):
            if n < 3:
                entry["params"][name] = {}
                continue
            x = good[:, j]
            mean, sd = float(x.mean()), float(x.std(ddof=1))
            zm, zs = mean * np.sqrt(n), (sd - 1) * np.sqrt(2 * n)
            pm, ps = float(2 * stats.norm.sf(abs(zm))), float(2 * stats.norm.sf(abs(zs)))
            pv[f"{cell}:{name}:mean"], pv[f"{cell}:{name}:sd"] = pm, ps
            entry["params"][name] = {
                "mean": mean, "sd": sd, "kurtosis": float(stats.kurtosis(x, fisher=False)),
                "mean_p": pm, "sd_p": ps, "mean_band": band / np.sqrt(n), "sd_band": band / np.sqrt(2 * n),
                "within_bands": bool(abs(mean) < band / np.sqrt(n) and abs(sd - 1) < band / np.sqrt(2 * n))}
        out["cells"][cell] = entry
    rejected = holm(pv, s["family_alpha"]) if pv else {}
    out["rejected"] = sorted(k for k, v in rejected.items() if v)
    out["tests"] = len(pv)
    if pv:  # reported: what the family can detect at its least stringent threshold
        z = stats.norm.isf(s["family_alpha"] / len(pv) / 2)
        n_min = min(e["finite"] for e in out["cells"].values())
        out["detectable_at_90pct_power"] = {
            "z_threshold": float(z), "n": int(n_min),
            "bias_in_mean_sigma": float((z + stats.norm.ppf(0.9)) / np.sqrt(max(n_min, 1))),
            "sd_miss_fraction": float((z + stats.norm.ppf(0.9)) / np.sqrt(2 * max(n_min, 1)))}
    out["pass"] = bool(pv and not out["rejected"] and set(out["cells"]) == set(s["cells"])
                       and all(e["enough"] and e["failures_ok"] for e in out["cells"].values()))
    return out


def aggregate(args):
    s = spec()
    files = [f for d in args.dirs for f in sorted(pathlib.Path(d).glob("pull_*_*.json"))]
    if len({f.resolve() for f in files}) != len(files):
        raise SystemExit("a results directory is given twice")
    if not files:
        raise SystemExit(f"no results found in {', '.join(map(str, args.dirs))}")
    cells, hashes, designs, commits, seeds, partial = {}, set(), set(), set(), [], 0
    for f in files:
        r = json.loads(f.read_text())
        hashes.add(r["criteria_hash"])
        designs.add(r.get("design_hash"))
        commits.add(r["virgil_commit"])
        partial += not r.get("complete", True)
        names, rows = cells.setdefault(r["cell"], (r["params"], []))
        if names != r["params"]:
            raise SystemExit(f"{r['cell']}: tasks disagree on the parameters")
        rows += r["rows"]
        seeds += [x["seed"] for x in r["rows"]]
    if hashes != {criteria_hash()}:
        raise SystemExit(f"criteria changed since the runs: {hashes} vs {criteria_hash()}")
    if designs != {design_hash()}:
        raise SystemExit(f"the experiment changed since the runs (design {designs} vs {design_hash()})")
    if len(commits) != 1:
        raise SystemExit(f"the tasks ran on different virgil commits: {commits}")
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"{len(seeds) - len(set(seeds))} draws repeat a seed: each must be counted once")
    print(f"{len(files)} task files ({partial} incomplete), {len(seeds)} draws", file=sys.stderr)
    summary = {"version": 1, "virgil_commit": commits.pop(), "criteria_hash": criteria_hash(),
               "design_hash": design_hash(), "tasks": len(files), "incomplete_tasks": partial, "criteria": s,
               **analyse(cells, s)}
    text = json.dumps(summary, indent=1)
    if args.summary:
        pathlib.Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.summary).write_text(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cell", required=True)
    r.add_argument("--task", type=int, required=True)
    r.add_argument("--draws", type=int, default=0, help="per task (default: the registered number)")
    r.add_argument("--seed-base", type=int, default=20261008)
    r.add_argument("--out", required=True)
    g = sub.add_parser("aggregate")
    g.add_argument("dirs", nargs="+")
    g.add_argument("--summary")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else aggregate(args)


if __name__ == "__main__":
    main()
