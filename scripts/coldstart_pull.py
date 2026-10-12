"""Cold-start campaign: does virgil's documented path find the mode?

Every other campaign starts its fits (pull_campaign.py) or chains (SBC v1) at
the truth, so none tests finding the mode, which is what fails on real data.
Here the pipeline is what a user runs, with no truth in it: virgil's
likelihood_grid over a registered grid picks a start, virgil.fitting.fit
refines it under registered priors, laplace_cov gives its uncertainties.
Only the scoring uses the truth: a second fit of the same data started at the
truth (the reference fit) and the pulls. A draw FAILS when the cold fit ends
in a mode worse, by more than a registered chi-squared, than the reference
fit's. Criteria and design: design/coldstart_pull.yml (and .md), whose SHA-256
is recorded in every task output and in the summary; the aggregator refuses
outputs made under another hash.

    python scripts/coldstart_pull.py smoke --out DIR [--draws N]
    python scripts/coldstart_pull.py run --cell C --task T [--draws N] --out DIR
    python scripts/coldstart_pull.py aggregate DIR [DIR ...] --summary FILE

``run`` writes DIR/coldstart_<cell>_<T>.json after every draw (a task stopped
by its time limit resumes). ``smoke`` runs a few draws of every cell; its
outputs are marked and refused by ``aggregate``.
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

CRITERIA_FILE = ROOT / "design" / "coldstart_pull.yml"
SNR_FLOOR = 0.0


def spec():
    import yaml

    return yaml.safe_load(CRITERIA_FILE.read_text())


def criteria_hash():
    return hashlib.sha256(CRITERIA_FILE.read_bytes()).hexdigest()


def design_hash():
    """What the registered file states only in words: the code that draws,
    starts, fits and scores, and our simulator's and scenes' sources."""
    h = hashlib.sha256()
    for fn in (draw_truth, observe, cold_start, scored_fit, draw, grid_axes, stage1_axes, _axis, flat_of, snr_of):
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


def priors_key(scene_name):
    return {"binary": "priors_binary", "resolved_star_companion": "priors_resolved",
            "star_envelope": "priors_envelope"}[scene_name]


def param_names(cell):
    """Parameter labels of a cell, in the order of its pulls."""
    return list(spec()["fit"][priors_key(spec()["cells"][cell]["scene"])])


def draw_truth(cell_spec, rng):
    """One truth, drawn in the order the registered ``draw`` mapping lists."""
    out = {}
    for name, how in cell_spec["draw"].items():
        (kind, (lo, hi)), = how.items()
        out[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))) if kind == "log" else rng.uniform(lo, hi))
    return out


def scene_of(cell_spec, t):
    import virgil_bridge as vb

    name = cell_spec["scene"]
    if name == "binary":
        return vb.binary(sep=t["sep_mas"], pa=t["pa_deg"], flux=t["flux"])
    if name == "resolved_star_companion":
        pa = np.deg2rad(t["pa_deg"])
        return vb.resolved_star_companion(diam=t["diam_mas"], dra=t["sep_mas"] * np.sin(pa),
                                          ddec=t["sep_mas"] * np.cos(pa), flux=t["flux"])
    return vb.star_envelope(fwhm=t["fwhm_mas"], ratio=t["ratio"], pa=t["pa_deg"], flux=t["flux"])


def observe(path, cell_spec, scene, noise_scale, rng):
    """Our simulator's data for the scene, with the registered array and noise."""
    from crosscheck import simulate

    s = spec()
    d = s["data"]
    wl = np.linspace(d["wavelengths_um"]["lo"], d["wavelengths_um"]["hi"], d["wavelengths_um"]["n"]) * 1e-6
    stations = np.array(d["stations_" + cell_spec["stations"]])
    ha = np.array(d["hour_angles_h_" + str(cell_spec["hour_angles"])])
    simulate.observe(path, scene.vis, stations, hour_angles_h=ha, wavelengths=wl, dec_deg=d["dec_deg"],
                     sigma_v2=d["sigma_v2"] * noise_scale, sigma_cp_deg=d["sigma_cp_deg"] * noise_scale, rng=rng,
                     phase_noise="baseline")


def _axis(a):
    if isinstance(a, list):
        return np.array(a, float)
    return (np.geomspace if a["spacing"] == "log" else np.linspace)(a["lo"], a["hi"], a["n"])


def grid_axes(cell_spec):
    """The registered grid of a cell, {path: values}, none depending on a draw.
    For the resolved star it is the companion grid of stage 2 (the diameter is
    fixed from stage 1: ``stage1_axes``)."""
    g = cell_spec["grid"]
    name = cell_spec["scene"]
    if name in ("binary", "resolved_star_companion"):
        step = spec()["fit"]["offset_step_mas"]
        h = g["half_width_mas"]
        off = np.arange(-h, h + 0.5 * step, step)
        pre = "" if name == "binary" else "comp."
        return {pre + "dra": off, pre + "ddec": off, pre + "flux": _axis(g["flux"])}
    return {"env.fwhm": _axis(g["fwhm_mas"]), "env.ratio": _axis(g["ratio"]), "env.pa": _axis(g["pa_deg"]),
            "env.flux": _axis(g["flux"])}


def stage1_axes(cell_spec):
    """The diameter scan of the star alone that precedes the companion grid
    of a resolved star (None for the other scenes)."""
    if cell_spec["scene"] != "resolved_star_companion":
        return None
    return {"star.diam": _axis(cell_spec["grid"]["stage1_diam_mas"])}


def generic_template(scene_name):
    """The model the cold start varies: neutral values, none from the draw."""
    import virgil.models as vm

    if scene_name == "binary":
        return vm.BinaryModelCartesian
    if scene_name == "resolved_star_companion":
        return vm.System(star=vm.UniformDisk(1.0), comp=vm.PointSource(0.01, 0.0, 0.0))
    return vm.System(star=vm.PointSource(), env=vm.EllipticalGaussian(4.0, 0.5, 0.0, 0.3))


def priors_of(scene_name):
    import numpyro.distributions as dist

    return {k: dist.Uniform(float(lo), float(hi)) for k, (lo, hi) in spec()["fit"][priors_key(scene_name)].items()}


def cold_start(cell_spec, data):
    """The user's path to a start: likelihood_grid (keyword arguments), then
    the best grid point; for a resolved star, first a scan of the star's
    diameter alone (as virgil's pipeline does), then the companion grid at
    that diameter. Returns {path: value} and the grid's maximum."""
    import virgil.models as vm
    from virgil.grid_fit import best_grid_point, likelihood_grid

    model = generic_template(cell_spec["scene"])
    start = {}
    first = stage1_axes(cell_spec)
    if first is not None:
        alone = vm.System(star=vm.UniformDisk(1.0), comp=vm.PointSource(0.0, 0.0, 0.0))  # flux 0: the star alone
        ll = np.asarray(likelihood_grid(model=alone, data=data, grid=first))
        start = best_grid_point(ll, first)
        model = model.set("star.diam", start["star.diam"])
    grid = grid_axes(cell_spec)
    ll = np.asarray(likelihood_grid(model=model, data=data, grid=grid))
    return start | best_grid_point(ll, grid), float(np.nanmax(ll))


def flat_of(values, paths):
    return np.array([float(np.asarray(values[p])) for p in paths])


def scored_fit(template, priors, data, paths):
    """Fit from ``template``; returns chi2, the fitted flat values, their
    Laplace sigmas (NaN where the covariance failed), and whether the fit
    reported convergence. Raises if the fit raised."""
    from virgil.fitting import fit
    from virgil.inference import laplace_cov

    result = fit(template, priors, data)
    chi2 = float(np.sum(result.info["chi2"]))
    values = flat_of(result.values, paths)
    try:
        cov = np.asarray(laplace_cov(values=values, params=list(paths), model=result.model, data=data))
        sig = np.sqrt(np.diag(cov))
    except Exception:  # noqa: BLE001 - a failed covariance is a counted NaN
        sig = np.full(len(paths), np.nan)
    return {"chi2": chi2, "values": values, "sigma": sig, "converged": bool(result.info.get("converged")),
            "loss": float(result.info["loss"])}


def snr_of(cell_spec, scene, data, ref_values, paths):
    """sqrt(2 (ll at the reference fit - ll with the companion/envelope off)),
    both from likelihood_grid; the star's diameter (if any) is the reference
    fit's, not re-fitted."""
    from virgil.grid_fit import likelihood_grid

    name = cell_spec["scene"]
    at = dict(zip(paths, ref_values))
    flux_path = [p for p in paths if p.endswith("flux")][0]
    null = {p: [0.0 if p == flux_path else float(v)] for p, v in at.items()}
    if name == "binary":
        null["dra"], null["ddec"] = [0.0], [0.0]
    model = generic_template(name)
    ref = {p: [float(v)] for p, v in at.items()}
    gain = float(np.asarray(likelihood_grid(model=model, data=data, grid=ref)).ravel()[0]
                 - np.asarray(likelihood_grid(model=model, data=data, grid=null)).ravel()[0])
    return float(np.sqrt(max(2.0 * gain, 0.0)))


def draw(cell, seed, tmp):
    """One draw of a cell: the truth, the data, the cold and reference fits,
    and the scoring. Returns the row to record."""
    from virgil_bridge import load

    s = spec()
    cell_spec = s["cells"][cell]
    rng = np.random.default_rng(seed)
    t = draw_truth(cell_spec, rng)
    scene = scene_of(cell_spec, t)
    path = pathlib.Path(tmp) / f"{seed}.fits"
    observe(path, cell_spec, scene, t["noise_scale"], rng)
    data = load(path)
    paths = param_names(cell)
    truth = flat_of(scene.truth, paths)
    priors = priors_of(cell_spec["scene"])
    row = {"seed": seed, "truth": {k: v for k, v in t.items()}, "sep": t.get("sep_mas"), "contrast": t["flux"],
           "cold_failed": False, "ref_failed": False}

    try:
        ref = scored_fit(scene.template, priors, data, paths)
        if not np.isfinite(ref["chi2"]):
            raise ValueError("non-finite reference chi2")
    except Exception as e:  # noqa: BLE001
        print(f"seed={seed} reference fit failed: {type(e).__name__}: {e}", flush=True)
        return {**row, "ref_failed": True}
    try:
        start, ll_max = cold_start(cell_spec, data)
        template = generic_template(cell_spec["scene"])
        if cell_spec["scene"] == "binary":
            template = template(**{k: float(v) for k, v in start.items()})
        else:
            for p, v in start.items():
                template = template.set(p, float(v))
        cold = scored_fit(template, priors, data, paths)
        if not np.isfinite(cold["chi2"]):
            raise ValueError("non-finite cold chi2")
    except Exception as e:  # noqa: BLE001
        print(f"seed={seed} cold fit failed: {type(e).__name__}: {e}", flush=True)
        return {**row, "cold_failed": True, "ref_chi2": ref["chi2"]}

    with np.errstate(invalid="ignore", divide="ignore"):
        pull_truth = (cold["values"] - truth) / cold["sigma"]
        pull_ref = (cold["values"] - ref["values"]) / ref["sigma"]
    row |= {"snr": snr_of(cell_spec, scene, data, ref["values"], paths), "start": {k: float(v) for k, v in start.items()},
            "ref_chi2": ref["chi2"], "cold_chi2": cold["chi2"], "ref_converged": ref["converged"],
            "cold_converged": cold["converged"], "cold_values": cold["values"].tolist(), "ref_values": ref["values"].tolist(),
            "pulls": [None if not np.isfinite(p) else float(p) for p in pull_truth],
            "ref_distance_sigma": [None if not np.isfinite(p) else float(p) for p in pull_ref]}
    return row


def classify(row, s, delta=None):
    """FAILURE / BETTER / SAME MODE / ALIAS of a row (registered definitions)."""
    delta = s["failure_delta_chi2"] if delta is None else delta
    if row.get("ref_failed"):
        return "reference_failed"
    if row.get("cold_failed"):
        return "failure"
    gap = row["cold_chi2"] - row["ref_chi2"]
    if gap > delta:
        return "failure"
    if -gap > delta:
        return "better"
    far = [abs(p) for p in row["ref_distance_sigma"] if p is not None]
    return "alias" if far and max(far) > s["alias_sigma"] else "same_mode"


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
            "criteria_hash": criteria_hash(), "design_hash": design_hash(), "params": param_names(args.cell),
            "smoke": bool(getattr(args, "smoke", False))}
    path = out_dir / f"coldstart_{args.cell}_{args.task:04d}.json"
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
                row = draw_fn(args.cell, seed, tmp)
            except Exception as e:  # noqa: BLE001 - a draw that cannot be made is a counted reference failure
                print(f"seed={seed} failed: {type(e).__name__}: {e}", flush=True)
                row = {"seed": seed, "ref_failed": True, "cold_failed": False}
            rows.append({**row, "seed": seed})
            tmp_path = path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps({**head, "complete": len(rows) == draws, "rows": rows}))
            tmp_path.rename(path)
            print(f"{args.cell} seed={seed} {classify(row, s)} elapsed={time.time() - t0:.1f}s", flush=True)
    print(f"wrote {path} elapsed={time.time() - start:.0f}s")


def smoke(args):
    s = spec()
    for cell in s["cells"]:
        run(argparse.Namespace(cell=cell, task=0, draws=args.draws, seed_base=args.seed_base, out=args.out, smoke=True))


def holm(pvalues, alpha):
    names = sorted(pvalues, key=pvalues.get)
    m, rejected, stop = len(names), {}, False
    for i, name in enumerate(names):
        stop = stop or pvalues[name] > alpha / (m - i)
        rejected[name] = not stop
    return rejected


def stratify(rows, s):
    """Failure rate by S/N bin, and by tertile of separation and of contrast."""
    out = {}
    edges = s["stratification"]["snr_edges"] + [np.inf]
    fail = np.array([classify(r, s) == "failure" for r in rows])
    snr = np.array([r.get("snr", np.nan) if r.get("snr") is not None else np.nan for r in rows], float)
    out["snr"] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (snr >= lo) & (snr < hi)
        out["snr"].append({"lo": lo, "hi": None if hi == np.inf else hi, "n": int(m.sum()), "failures": int(fail[m].sum())})
    m = ~np.isfinite(snr)
    out["snr_missing"] = {"n": int(m.sum()), "failures": int(fail[m].sum())}  # cold fit raised: S/N not computed
    for key in ("sep", "contrast"):
        x = np.array([np.nan if r.get(key) is None else r[key] for r in rows], float)
        if not np.all(np.isfinite(x)) or len(x) < 9:
            continue
        cuts = np.quantile(x, [1 / 3, 2 / 3])
        bins = np.digitize(x, cuts)
        out[key] = [{"lo": float(x[bins == b].min()) if (bins == b).any() else None,
                     "hi": float(x[bins == b].max()) if (bins == b).any() else None, "n": int((bins == b).sum()),
                     "failures": int(fail[bins == b].sum())} for b in range(3)]
    return out


def analyse(cells, s):
    """The registered analysis of {cell: (param names, rows)}."""
    from scipy import stats

    out = {"cells": {}, "primary_pvalues": {}}
    pv = out["primary_pvalues"]
    pull_p = {}
    for cell, (names, rows) in cells.items():
        bound = s["cells"][cell]["failure_bound"]
        kinds = [classify(r, s) for r in rows]
        valid = [r for r, k in zip(rows, kinds) if k != "reference_failed"]
        vk = [k for k in kinds if k != "reference_failed"]
        n = len(valid)
        k = int(sum(x == "failure" for x in vk))
        e = {"draws": len(rows), "reference_failed": len(rows) - n, "n": n, "failures": k, "bound": bound,
             "rate": k / n if n else None,
             "enough": bool(n >= s["counts"]["min_draws_per_cell"]),
             "reference_ok": bool(len(rows) - n <= s["reference_failures"]["max_fraction"] * max(len(rows), 1)),
             "cold_failed_draws": int(sum(bool(r.get("cold_failed")) for r in valid))}
        for kind in ("better", "alias", "same_mode"):
            e[kind] = int(sum(x == kind for x in vk))
        e["gross_errors"] = int(sum(1 for r in valid if r.get("pulls") and any(p is not None and abs(p) > s["gross_pull_sigma"] for p in r["pulls"])))
        e["rate_at_delta"] = {str(d): int(sum(classify(r, s, d) == "failure" for r in valid)) for d in (0.1, 10.0)}
        if n:
            bt = stats.binomtest(k, n, bound, alternative="greater")
            pv[cell] = float(bt.pvalue)
            e["pvalue"] = float(bt.pvalue)
            e["upper99"] = float(stats.binomtest(k, n, bound).proportion_ci(confidence_level=0.98, method="exact").high)
        e["stratified"] = stratify(valid, s)
        good = np.array([r["pulls"] for r, x in zip(valid, vk) if x == "same_mode" and r.get("pulls") and None not in r["pulls"]], float)
        e["pull_draws"] = int(len(good))
        e["pulls"] = {}
        if len(good) >= s["secondary"]["min_draws"]:
            for j, name in enumerate(names):
                x = good[:, j]
                m, sd = float(x.mean()), float(x.std(ddof=1))
                pm = float(2 * stats.norm.sf(abs(m * np.sqrt(len(x)))))
                ps = float(2 * stats.norm.sf(abs((sd - 1) * np.sqrt(2 * len(x)))))
                pull_p[f"{cell}:{name}:mean"], pull_p[f"{cell}:{name}:sd"] = pm, ps
                e["pulls"][name] = {"mean": m, "sd": sd, "kurtosis": float(stats.kurtosis(x, fisher=False)),
                                    "mean_p": pm, "sd_p": ps}
        out["cells"][cell] = e
    rejected = holm(pv, s["family_alpha"]) if pv else {}
    out["rejected"] = sorted(k for k, v in rejected.items() if v)
    pull_rej = holm(pull_p, s["secondary"]["family_alpha"]) if pull_p else {}
    out["pull_tests"] = len(pull_p)
    out["pull_rejected"] = sorted(k for k, v in pull_rej.items() if v)  # reported, not pass/fail
    out["pass"] = bool(pv and not out["rejected"] and set(out["cells"]) == set(s["cells"])
                       and all(c["enough"] and c["reference_ok"] for c in out["cells"].values()))
    return out


def aggregate(args):
    s = spec()
    files = [f for d in args.dirs for f in sorted(pathlib.Path(d).glob("coldstart_*_*.json"))]
    if len({f.resolve() for f in files}) != len(files):
        raise SystemExit("a results directory is given twice")
    if not files:
        raise SystemExit(f"no results found in {', '.join(map(str, args.dirs))}")
    cells, hashes, designs, commits, seeds, partial = {}, set(), set(), set(), [], 0
    for f in files:
        r = json.loads(f.read_text())
        if r.get("smoke"):
            raise SystemExit(f"{f} is a smoke run: it cannot be aggregated")
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
    r.add_argument("--seed-base", type=int, default=20261012)
    r.add_argument("--out", required=True)
    m = sub.add_parser("smoke")
    m.add_argument("--draws", type=int, default=3)
    m.add_argument("--seed-base", type=int, default=7_000_000_000)  # outside the registered seeds
    m.add_argument("--out", required=True)
    g = sub.add_parser("aggregate")
    g.add_argument("dirs", nargs="+")
    g.add_argument("--summary")
    args = ap.parse_args()
    {"run": run, "smoke": smoke, "aggregate": aggregate}[args.cmd](args)


if __name__ == "__main__":
    main()
