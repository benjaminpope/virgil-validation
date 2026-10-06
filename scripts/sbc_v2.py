"""Simulation-based calibration of virgil.likelihood.numpyro_model, version 2.

Version 1 (scripts/sbc_numpyro.py) started every chain at the truth, in one
bright regime. Version 2 (criteria and design registered in design/sbc_v2.yml,
whose SHA-256 is recorded in every task output and in the summary) tests the
pipeline a user runs:

* each replicate's chains start from virgil's own grid search and MAP fit on
  that replicate's data (virgil.grid_fit.likelihood_grid, then
  virgil.fitting.fit), jittered per chain;
* the NUTS mass matrix is dense on (pa_vec, sep, flux) and the AngleVector
  ring is narrower (ring_width does not change the prior on the angle);
* two regimes: ``bright`` (flux log-uniform 0.02 to 0.1, as v1) and ``low``
  (1e-3 to 2e-2, so posteriors reach the flux lower bound);
* the test quantities are the five parameters and the joint log-likelihood
  of the data: the rank of log p(y | truth) among log p(y | draw) is uniform
  when the posterior is right.

The generative model is ours, as in v1: truths from NumPy under the stated
priors, data from crosscheck.simulate.observe.

    python scripts/sbc_v2.py run --regime R --task T --out DIR [--replicates N]
    python scripts/sbc_v2.py aggregate DIR [DIR ...] --summary FILE

``run`` writes DIR/task_<regime>_<T>.json after every replicate (a task
stopped by its time limit resumes). ``aggregate`` applies the registered
criteria: Holm at 1% over every rank test of every regime and quantity, and a
sampler-health criterion per regime.
"""

import argparse
import hashlib
import importlib.util
import inspect
import json
import pathlib
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CRITERIA_FILE = ROOT / "design" / "sbc_v2.yml"
SEED_OFFSET = {"bright": 0, "low": 10_000_000}


def _v1():
    spec = importlib.util.spec_from_file_location("sbc_numpyro", ROOT / "scripts" / "sbc_numpyro.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v1 = _v1()


def spec():
    import yaml

    return yaml.safe_load(CRITERIA_FILE.read_text())


def criteria_hash():
    return hashlib.sha256(CRITERIA_FILE.read_bytes()).hexdigest()


def design_hash():
    """What the registered file does not state: v1's array, noise and
    simulator constants, the sources of the code that draws, starts and
    samples, and our simulator's. Replicates combine only when it matches."""
    config = {"UTS3": v1.UTS3.tolist(), "WAVELENGTHS": v1.WAVELENGTHS.tolist(),
              "HOUR_ANGLES": v1.HOUR_ANGLES.tolist(), "PARAMS": list(v1.PARAMS)}
    h = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for fn in (v1.observe, draw_truth, start_from_fit, jittered_inits, replicate, v1.circular_rank):
        h.update(inspect.getsource(fn).encode())
    for name in ("simulate.py", "sky.py", "array.py"):
        h.update((ROOT / "src" / "crosscheck" / name).read_bytes())
    return h.hexdigest()[:16]


def virgil_commit():
    return v1.virgil_commit()


def draw_truth(rng, regime):
    s = spec()
    g, r = s["generative"], s["regimes"][regime]
    bounds = {"diam": (g["diam"]["lo"], g["diam"]["hi"]), "flux": (r["flux_lo"], r["flux_hi"]),
              "sep": (g["sep"]["lo"], g["sep"]["hi"]), "vis_scale": (g["vis_scale"]["lo"], g["vis_scale"]["hi"])}
    t = {k: float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) for k, (lo, hi) in bounds.items()}
    t["pa"] = float(rng.uniform(0.0, 360.0))
    return t


def observe(path, truth, rng):
    """v1's data (its noise levels are the registered ones: a test checks)."""
    v1.observe(path, truth, rng)


def _axis(a):
    f = np.geomspace if a["spacing"] == "log" else np.linspace
    return f(a["lo"], a["hi"], a["n"])


def start_from_fit(data, regime):
    """virgil's own start: the best point of a likelihood grid over (diam,
    dra, ddec, flux), refined by the MAP fit with the sampler's priors.
    Returns the fitted values (constrained, plus pa in degrees) and whether
    the fit reported convergence."""
    import numpyro.distributions as dist
    from virgil import models as vm
    from virgil.angles import AngleVector
    from virgil.fitting import fit
    from virgil.grid_fit import best_grid_point, likelihood_grid

    s = spec()
    g, rr = s["sampler"]["grid"], s["regimes"][regime]
    off = np.linspace(-g["offset_mas"]["half_width"], g["offset_mas"]["half_width"], g["offset_mas"]["n"])
    samples = {"star.diam": _axis(g["diam"]), "comp.dra": off, "comp.ddec": off,
               "comp.flux": np.geomspace(rr["flux_lo"], rr["flux_hi"], g["flux"]["n"])}
    template = vm.System(star=vm.UniformDisk(1.0), comp=vm.PointSource(0.01, 0.0, 0.0))
    best = best_grid_point(likelihood_grid(data, template, samples), samples)
    sep = float(np.hypot(best["comp.dra"], best["comp.ddec"]))
    pa = float(np.rad2deg(np.arctan2(best["comp.dra"], best["comp.ddec"])) % 360.0)
    gen = s["generative"]
    priors = {"diam": dist.LogUniform(gen["diam"]["lo"], gen["diam"]["hi"]),
              "flux": dist.LogUniform(rr["flux_lo"], rr["flux_hi"]),
              "sep": dist.LogUniform(gen["sep"]["lo"], gen["sep"]["hi"]),
              "pa": AngleVector(ring_width=s["sampler"]["ring_width"])}
    noise = {"vis_scale": dist.LogUniform(gen["vis_scale"]["lo"], gen["vis_scale"]["hi"])}
    init = {"diam": best["star.diam"], "flux": best["comp.flux"], "sep": max(sep, gen["sep"]["lo"]), "pa": pa,
            "noise.vis_scale": 1.0}
    result = fit(scene, priors, data, noise=noise, init=init)
    v = result.values
    values = {"diam": float(v["diam"]), "flux": float(v["flux"]), "sep": float(v["sep"]), "pa": float(v["pa"]),
              "vis_scale": float(v["noise.vis_scale"])}
    return values, bool(result.info.get("converged")), {"grid": {k: float(x) for k, x in best.items()}}


def scene(diam, flux, sep, pa):
    return v1.scene(diam, flux, sep, pa)


def jittered_inits(start, regime, rng, chains):
    """One constrained starting point per chain, the fit's jittered, kept
    inside the prior's support."""
    s = spec()
    j, g, rr = s["sampler"]["jitter"], s["generative"], s["regimes"][regime]
    bounds = {"diam": (g["diam"]["lo"], g["diam"]["hi"]), "flux": (rr["flux_lo"], rr["flux_hi"]),
              "sep": (g["sep"]["lo"], g["sep"]["hi"]), "vis_scale": (g["vis_scale"]["lo"], g["vis_scale"]["hi"])}
    out = []
    for _ in range(chains):
        c = {}
        for k, (lo, hi) in bounds.items():
            x = start[k] * np.exp(j[k] * rng.standard_normal())
            c[k] = float(np.clip(x, lo * 1.001, hi / 1.001))
        pa = np.deg2rad(start["pa"] + j["pa_deg"] * rng.standard_normal())
        c["pa_vec"] = np.array([np.cos(pa), np.sin(pa)])
        out.append(c)
    return out


def replicate(seed, regime, override=None):
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro import handlers
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
    from numpyro.infer import MCMC, NUTS
    from numpyro.infer.util import unconstrain_fn

    from virgil.angles import AngleVector
    from virgil.likelihood import numpyro_model
    from virgil.oidata import OIData

    s = spec()
    smp, g, rr = dict(s["sampler"]), s["generative"], s["regimes"][regime]
    if override:  # a smoke run only: its outputs are refused by aggregate
        smp.update(zip(("warmup", "samples", "chains"), override))
    rng = np.random.default_rng(seed)
    truth = draw_truth(rng, regime)
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "sbc.fits"
        observe(path, truth, rng)
        data = OIData(str(path))
    priors = {"diam": dist.LogUniform(g["diam"]["lo"], g["diam"]["hi"]),
              "flux": dist.LogUniform(rr["flux_lo"], rr["flux_hi"]),
              "sep": dist.LogUniform(g["sep"]["lo"], g["sep"]["hi"]),
              "pa": AngleVector(ring_width=smp["ring_width"])}
    noise = {"vis_scale": dist.LogUniform(g["vis_scale"]["lo"], g["vis_scale"]["hi"])}
    model = numpyro_model(scene, priors, data, noise=noise)

    start, converged, start_info = start_from_fit(data, regime)
    inits = jittered_inits(start, regime, rng, smp["chains"])
    names = {"diam": "diam", "flux": "flux", "sep": "sep", "pa_vec": "pa_vec", "vis_scale": "noise.vis_scale"}
    site_values = [{names[k]: v for k, v in c.items()} for c in inits]
    unconstrained = [unconstrain_fn(model, (), {}, c) for c in site_values]
    init_params = {k: jnp.stack([u[k] for u in unconstrained]) for k in unconstrained[0]}
    kernel = NUTS(model, dense_mass=[tuple(smp["dense_mass"])])
    mcmc = MCMC(kernel, num_warmup=smp["warmup"], num_samples=smp["samples"], num_chains=smp["chains"],
                chain_method="sequential", progress_bar=False)
    mcmc.run(jax.random.key(seed), init_params=init_params)
    grouped = mcmc.get_samples(group_by_chain=True)
    divergences = int(np.sum(np.asarray(mcmc.get_extra_fields()["diverging"])))

    def loglike_of(values):
        trace = handlers.trace(handlers.substitute(model, data=values)).get_trace()
        return trace["loglike"]["fn"].log_factor

    L = s["draws_per_replicate"]
    flat = {k: np.asarray(v).reshape((-1,) + np.asarray(v).shape[2:]) for k, v in grouped.items()}
    idx = np.linspace(0, flat["diam"].size - 1, L).round().astype(int)
    thinned = {k: jnp.asarray(flat[k][idx]) for k in ("diam", "flux", "sep", "pa_vec", "noise.vis_scale")}
    ll_draws = np.asarray(jax.jit(jax.vmap(loglike_of))(thinned))
    p = np.deg2rad(truth["pa"])
    at_truth = {"diam": truth["diam"], "flux": truth["flux"], "sep": truth["sep"],
                "pa_vec": jnp.array([np.cos(p), np.sin(p)]), "noise.vis_scale": truth["vis_scale"]}
    ll_truth = float(loglike_of(at_truth))

    sites = {"diam": "diam", "flux": "flux", "sep": "sep", "pa": "pa", "vis_scale": "noise.vis_scale"}
    out = {"seed": int(seed), "regime": regime, "truth": truth, "ranks": {}, "ess": {}, "rhat": {},
           "divergences": divergences, "fit_converged": converged, "start": {**start, **start_info},
           "loglike_truth": ll_truth}
    for name, site in sites.items():
        chain_draws = np.asarray(grouped[site])
        draws = chain_draws.reshape(-1)[idx]
        if name == "pa":
            out["ranks"][name] = v1.circular_rank(draws, truth["pa"])
            ang = np.deg2rad(chain_draws)
            vec = np.stack([np.cos(ang), np.sin(ang)])
            out["ess"][name] = float(min(effective_sample_size(c) for c in vec))
            out["rhat"][name] = float(max(split_gelman_rubin(c) for c in vec))
        else:
            out["ranks"][name] = int(np.sum(draws < truth[name]))
            out["ess"][name] = float(effective_sample_size(chain_draws))
            out["rhat"][name] = float(split_gelman_rubin(chain_draws))
    out["ranks"]["loglike"] = int(np.sum(ll_draws < ll_truth))
    return out


def run(args, replicate_fn=None):
    import jax

    jax.config.update("jax_enable_x64", True)
    import virgil

    s = spec()
    if args.regime not in s["regimes"]:
        raise SystemExit(f"unknown regime {args.regime}: {sorted(s['regimes'])}")
    reps_per_task = args.replicates or s["regimes"][args.regime]["replicates_per_task"]
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = virgil_commit()
    if not commit:
        raise SystemExit("cannot tell which virgil commit is running: set PIN_COMMIT")
    path = out_dir / f"task_{args.regime}_{args.task:04d}.json"
    head = {"regime": args.regime, "task": args.task, "virgil_commit": commit, "criteria_hash": criteria_hash(),
            "design_hash": design_hash(), "replicates_planned": reps_per_task,
            "override": [args.warmup, args.samples, args.chains] if args.warmup else None}
    results, start = [], time.time()
    if path.exists():  # resume a task stopped by its time limit
        old = json.loads(path.read_text())
        if {k: old.get(k) for k in head} != head:
            raise SystemExit(f"{path} was made with another commit, criteria, design or size: move it before rerunning")
        results = list(old["replicates"])
    done = {r["seed"] for r in results}
    replicate_fn = replicate_fn or replicate
    seeds = [args.seed_base + SEED_OFFSET[args.regime] + args.task * reps_per_task + k for k in range(reps_per_task)]

    def write():
        record = {**head, "virgil": getattr(virgil, "__version__", "?"), "complete": len(results) == len(seeds),
                  "replicates": results, "elapsed_s": time.time() - start}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=1))
        tmp.rename(path)

    if done:
        print(f"resuming {path}: {len(done)} of {len(seeds)} replicates already done", flush=True)
    for seed in seeds:
        if seed in done:
            continue
        t0 = time.time()
        results.append(replicate_fn(seed, args.regime, head["override"]))
        write()
        print(f"replicate regime={args.regime} seed={seed} elapsed={time.time() - t0:.1f}s "
              f"ranks={results[-1]['ranks']}", flush=True)
    write()
    print(f"wrote {path} elapsed={time.time() - start:.0f}s")


def unhealthy(r, health):
    return (r["divergences"] > health["max_divergent_draws"] or max(r["rhat"].values()) > health["max_rhat"]
            or min(r["ess"].values()) < health["min_bulk_ess"])


def analyse(by_regime, s):
    """The registered analysis of {regime: replicates}: Holm over every rank
    test, the sampler-health criterion per regime, descriptive extras."""
    L, bins, tests = s["draws_per_replicate"], s["rank_bins"], s["rank_tests"]
    health = s["sampler_health"]
    out = {"regimes": {}, "pvalues": {}}
    pv = out["pvalues"]
    for regime, reps in by_regime.items():
        entry = {"n": len(reps)}
        for q in s["quantities"]:
            ranks = np.array([r["ranks"][q] for r in reps], dtype=int)
            p = v1.rank_pvalues(ranks, L, bins, tests, s["ecdf_simulations"], s["ecdf_seed"]) if ranks.size else {}
            pv.update({f"{regime}:{q}:{t}": v for t, v in p.items()})
        bad = [r for r in reps if unhealthy(r, health)]
        div = [r for r in reps if r["divergences"] > 0]
        n = max(len(reps), 1)
        entry["health"] = {
            "unhealthy": len(bad), "unhealthy_fraction": len(bad) / n,
            "with_divergences": len(div), "divergent_fraction": len(div) / n,
            "high_rhat": sum(max(r["rhat"].values()) > health["max_rhat"] for r in reps),
            "low_ess": sum(min(r["ess"].values()) < health["min_bulk_ess"] for r in reps),
            "pass": bool(len(bad) / n <= health["max_unhealthy_fraction"]
                         and len(div) / n <= health["max_divergent_fraction"])}
        entry["enough"] = len(reps) >= s["counts"]["min_replicates_per_regime"]
        good = [r for r in reps if not unhealthy(r, health)]
        entry["exclude_unhealthy"] = {
            "n": len(good),
            "p": {q: v1.rank_pvalues(np.array([r["ranks"][q] for r in good], dtype=int), L, bins,
                                     ["chi2_bins", "mean_rank"]) for q in s["quantities"]} if good else {}}
        entry["fit_converged_fraction"] = float(np.mean([bool(r.get("fit_converged")) for r in reps])) if reps else 0.0
        out["regimes"][regime] = entry
    rejected = v1.holm(pv, s["family_alpha"]) if pv else {}
    out["rejected"] = sorted(k for k, v in rejected.items() if v)
    out["tests"] = len(pv)
    out["pass"] = bool(pv and not out["rejected"]
                       and all(e["enough"] and e["health"]["pass"] for e in out["regimes"].values())
                       and set(out["regimes"]) == set(s["regimes"]))
    return out


def aggregate(args):
    s = spec()
    files = [f for d in args.dirs for f in sorted(pathlib.Path(d).glob("task_*_*.json"))]
    if not files:
        raise SystemExit(f"no results found (no task_<regime>_<task>.json) in {', '.join(map(str, args.dirs))}")
    by_regime = {}
    hashes, designs, commits, versions, seeds, partial = set(), set(), set(), set(), [], 0
    for f in files:
        r = json.loads(f.read_text())
        if r.get("override"):
            raise SystemExit(f"{f} is a smoke run (sampler overridden): remove it")
        partial += not r.get("complete", True)
        hashes.add(r["criteria_hash"])
        designs.add(r.get("design_hash"))
        commits.add(r.get("virgil_commit"))
        versions.add(r.get("virgil"))
        by_regime.setdefault(r["regime"], []).extend(r["replicates"])
        seeds += [x["seed"] for x in r["replicates"]]
    if hashes != {criteria_hash()}:
        raise SystemExit(f"criteria changed since the runs: {hashes} vs {criteria_hash()}")
    if designs != {design_hash()}:
        raise SystemExit(f"the experiment changed since the runs (design {designs} vs {design_hash()})")
    if len(commits) != 1 or None in commits:
        raise SystemExit(f"the tasks ran on different or unknown virgil commits: {commits}")
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"{len(seeds) - len(set(seeds))} replicates repeat a seed: each must be counted once")
    print(f"{len(files)} task files ({partial} incomplete), {len(seeds)} replicates", file=sys.stderr)
    out = analyse(by_regime, s)
    summary = {"version": 2, "tasks": len(files), "incomplete_tasks": partial, "virgil": sorted(map(str, versions)),
               "virgil_commit": commits.pop(), "criteria_hash": criteria_hash(), "design_hash": design_hash(),
               "criteria": s, **out}
    text = json.dumps(summary, indent=1)
    if args.summary:
        pathlib.Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.summary).write_text(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--regime", required=True)
    r.add_argument("--task", type=int, required=True)
    r.add_argument("--replicates", type=int, default=0, help="per task (default: the registered number)")
    r.add_argument("--seed-base", type=int, default=20261007)
    r.add_argument("--warmup", type=int, default=0, help="smoke runs only (with --samples, --chains)")
    r.add_argument("--samples", type=int, default=0)
    r.add_argument("--chains", type=int, default=0)
    r.add_argument("--out", required=True)
    a = sub.add_parser("aggregate")
    a.add_argument("dirs", nargs="+")
    a.add_argument("--summary")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else aggregate(args)


if __name__ == "__main__":
    main()
