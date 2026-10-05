"""Simulation-based calibration (Talts et al. 2018, arXiv:1804.06788) of
virgil.likelihood.numpyro_model.

If the posterior a sampler draws from is the true posterior of the
generative model, the rank of a parameter's true value among L posterior
draws, for truths drawn from the prior and data drawn given the truth, is
uniform on 0..L. Every part of numpyro_model enters: its priors (the sites,
the AngleVector site of an angle), the likelihood it adds with
numpyro.factor, a model function of derived parameters and a sampled noise
term (vis_scale).

The generative model is ours, not virgil's: truths from NumPy under the
stated priors, data from crosscheck.simulate.observe (three VLTI UTs, one
closure triangle per snapshot, so closure phases are independent), the V²
noise drawn with sigma * vis_scale and the file stating sigma.

Priors follow the Jeffreys rule for each group: log-uniform for the
diameter, flux ratio, separation and vis_scale (scales), uniform for the
position angle (an angle, sampled as an AngleVector).

Each chain starts at the truth. This tests the density numpyro_model
defines and NUTS's sampling of it within the mode holding the truth; it
does not test finding that mode, which grid searches and fit do. The
companions are strong (flux >= 0.02 at sigma_V2 = 0.005), so the posterior
is concentrated in one mode.

    python scripts/sbc_numpyro.py run --task T --replicates R --seed-base S --out DIR
    python scripts/sbc_numpyro.py aggregate DIR [DIR ...] --summary FILE

``run`` writes DIR/task_<T>.json: per replicate, the truth, the rank of
each parameter among L thinned draws, the bulk ESS, R-hat and divergences.
``aggregate`` applies the criteria registered below (CRITERIA, whose hash is
recorded in the summary) to every replicate found.
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

PARAMS = ("diam", "flux", "sep", "pa", "vis_scale")
BOUNDS = {"diam": (0.5, 3.0), "flux": (0.02, 0.1), "sep": (5.0, 25.0), "vis_scale": (0.7, 1.5)}
SIGMA_V2, SIGMA_CP_DEG = 0.005, 0.3
UTS3 = np.array([[-9.925, -20.335, 0.0], [14.887, 30.502, 0.0], [103.306, 43.999, 0.0]])
WAVELENGTHS = np.linspace(1.5e-6, 2.4e-6, 6)
HOUR_ANGLES = np.linspace(-3.0, 3.0, 7)

# Registered before the campaign is run; the summary records their hash.
CRITERIA = {
    "draws_per_replicate": 99,  # L: ranks take values 0..99
    "rank_bins": 20,
    # rank uniformity per parameter: chi-squared over the bins, Bonferroni
    # over the parameters at a family-wise 1%
    "family_alpha": 0.01,
    # central 68% and 95% posterior intervals cover the truth within the
    # 99% binomial band
    "coverage_levels": [0.68, 0.95],
    "coverage_band": 0.99,
    # sampler health (reported; replicates failing it are counted, not dropped)
    "min_bulk_ess": 400,
    "max_rhat": 1.01,
    "min_replicates": 400,
}


def virgil_commit():
    """The virgil commit the campaign runs: the OzSTAR job's pinned snapshot
    (PIN_COMMIT, from its submit.sh), else the installed package's record."""
    import os

    if os.environ.get("PIN_COMMIT"):
        return os.environ["PIN_COMMIT"]
    from evidence.meta import virgil_source

    return virgil_source()["commit"]


def criteria_hash():
    return hashlib.sha256(json.dumps(CRITERIA, sort_keys=True).encode()).hexdigest()[:16]


def draw_truth(rng):
    t = {k: float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) for k, (lo, hi) in BOUNDS.items()}
    t["pa"] = float(rng.uniform(0.0, 360.0))
    return t


def observe(path, truth, rng):
    """Our own data: V² noise sigma * vis_scale, the file stating sigma."""
    from astropy.io import fits

    from crosscheck import simulate, sky

    dra = truth["sep"] * np.sin(np.deg2rad(truth["pa"]))
    ddec = truth["sep"] * np.cos(np.deg2rad(truth["pa"]))

    def vis(u, v, w):
        return (sky.vis_uniform_disk(u, v, w, truth["diam"]) + truth["flux"] * sky.vis_point(u, v, w, dra, ddec)) \
            / (1 + truth["flux"])

    simulate.observe(path, vis, UTS3, hour_angles_h=HOUR_ANGLES, wavelengths=WAVELENGTHS, dec_deg=-50.0,
                     sigma_v2=SIGMA_V2 * truth["vis_scale"], sigma_cp_deg=SIGMA_CP_DEG, rng=rng)
    with fits.open(path, mode="update") as h:
        h["OI_VIS2"].data["VIS2ERR"][:] = SIGMA_V2


def scene(diam, flux, sep, pa):
    import jax.numpy as jnp
    from virgil import models as vm

    p = jnp.deg2rad(pa)
    return vm.System(star=vm.UniformDisk(diam), comp=vm.PointSource(flux, sep * jnp.sin(p), sep * jnp.cos(p)))


def circular_rank(draws, truth):
    """Rank of an angle among draws, measured from the antipode of the
    draws' circular mean (no wrap within a concentrated posterior)."""
    mean = np.rad2deg(np.angle(np.mean(np.exp(1j * np.deg2rad(draws)))))
    centre = lambda x: (np.asarray(x) - mean + 180.0) % 360.0  # noqa: E731
    return int(np.sum(centre(draws) < centre(truth)))


def replicate(seed, warmup, samples, chains):
    import jax
    import numpyro
    import numpyro.distributions as dist
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
    from numpyro.infer import MCMC, NUTS, init_to_value

    from virgil.angles import AngleVector
    from virgil.likelihood import numpyro_model
    from virgil.oidata import OIData

    rng = np.random.default_rng(seed)
    truth = draw_truth(rng)
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "sbc.fits"
        observe(path, truth, rng)
        data = OIData(str(path))
    priors = {k: dist.LogUniform(*BOUNDS[k]) for k in ("diam", "flux", "sep")}
    priors["pa"] = AngleVector()
    noise = {"vis_scale": dist.LogUniform(*BOUNDS["vis_scale"])}
    model = numpyro_model(scene, priors, data, noise=noise)
    p = np.deg2rad(truth["pa"])
    init = {"diam": truth["diam"], "flux": truth["flux"], "sep": truth["sep"],
            "pa_vec": np.array([np.cos(p), np.sin(p)]), "noise.vis_scale": truth["vis_scale"]}
    mcmc = MCMC(NUTS(model, init_strategy=init_to_value(values=init)), num_warmup=warmup, num_samples=samples,
                num_chains=chains, chain_method="sequential", progress_bar=False)
    mcmc.run(jax.random.key(seed))
    grouped = mcmc.get_samples(group_by_chain=True)
    divergences = int(np.sum(np.asarray(mcmc.get_extra_fields()["diverging"])))
    sites = {"diam": "diam", "flux": "flux", "sep": "sep", "pa": "pa", "vis_scale": "noise.vis_scale"}
    L = CRITERIA["draws_per_replicate"]
    out = {"seed": int(seed), "truth": truth, "ranks": {}, "ess": {}, "rhat": {}, "divergences": divergences}
    for name, site in sites.items():
        chain_draws = np.asarray(grouped[site])
        flat = chain_draws.reshape(-1)
        # L draws spread evenly through the pooled chains (thinning)
        draws = flat[np.linspace(0, flat.size - 1, L).round().astype(int)]
        if name == "pa":
            out["ranks"][name] = circular_rank(draws, truth["pa"])
            ang = np.deg2rad(chain_draws)
            chain_draws = np.stack([np.cos(ang), np.sin(ang)])  # diagnostics on the vector
            out["ess"][name] = float(min(effective_sample_size(c) for c in chain_draws))
            out["rhat"][name] = float(max(split_gelman_rubin(c) for c in chain_draws))
        else:
            out["ranks"][name] = int(np.sum(draws < truth[name]))
            out["ess"][name] = float(effective_sample_size(chain_draws))
            out["rhat"][name] = float(split_gelman_rubin(chain_draws))
    return out


def run(args, replicate_fn=None):
    import jax

    jax.config.update("jax_enable_x64", True)
    import virgil

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = virgil_commit()
    if not commit:
        raise SystemExit("cannot tell which virgil commit is running: set PIN_COMMIT")
    path = out_dir / f"task_{args.task:04d}.json"
    settings = {"warmup": args.warmup, "samples": args.samples, "chains": args.chains}
    results, start = [], time.time()
    if path.exists():
        # resume a task stopped by its time limit: keep its replicates when
        # they were made the same way, and run only the seeds still missing
        old = json.loads(path.read_text())
        same = (old.get("virgil_commit") == commit and old.get("criteria_hash") == criteria_hash()
                and old.get("settings") == settings and old.get("replicates_planned", args.replicates) == args.replicates)
        if not same:
            raise SystemExit(f"{path} was made with another commit, criteria or settings: move it before rerunning")
        results = list(old["replicates"])
    done = {r["seed"] for r in results}
    replicate_fn = replicate_fn or replicate

    def write(complete):
        # after every replicate, so a task stopped by its time limit keeps
        # what it finished; "complete" marks a task that ran every replicate
        record = {"task": args.task, "virgil": getattr(virgil, "__version__", "?"), "virgil_commit": commit,
                  "criteria_hash": criteria_hash(), "replicates_planned": args.replicates, "complete": complete,
                  "settings": settings,
                  "replicates": results, "elapsed_s": time.time() - start}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=1))
        tmp.rename(path)

    seeds = [args.seed_base + args.task * args.replicates + k for k in range(args.replicates)]
    if done:
        print(f"resuming {path}: {len(done)} of {len(seeds)} replicates already done", flush=True)
    for seed in seeds:
        if seed in done:
            continue
        t0 = time.time()
        results.append(replicate_fn(seed, args.warmup, args.samples, args.chains))
        write(complete=len(results) == len(seeds))
        print(f"replicate seed={seed} elapsed={time.time() - t0:.1f}s ranks={results[-1]['ranks']}", flush=True)
    if len(results) == len(seeds):
        write(complete=True)
    print(f"wrote {path} elapsed={time.time() - start:.0f}s")


def binomial_band(n, p, level):
    from scipy import stats

    lo, hi = stats.binom.interval(level, n, p)
    return lo / n, hi / n


def aggregate(args):
    from scipy import stats

    reps, hashes, versions, commits = [], set(), set(), set()
    files = [f for d in args.dirs for f in sorted(pathlib.Path(d).glob("task_*.json"))]
    if not files:
        raise SystemExit(f"no results found (no task_*.json) in {', '.join(map(str, args.dirs))}: "
                         "the tasks may still be running, or they stopped before a replicate finished")
    partial = 0
    for f in files:
        r = json.loads(f.read_text())
        partial += not r.get("complete", True)
        hashes.add(r["criteria_hash"])
        versions.add(r["virgil"])
        commits.add(r.get("virgil_commit"))
        reps += r["replicates"]
    if hashes != {criteria_hash()}:
        raise SystemExit(f"criteria changed since the runs: {hashes} vs {criteria_hash()}")
    if len(commits) != 1 or None in commits:
        raise SystemExit(f"the tasks ran on different or unknown virgil commits: {commits}")
    seeds = [r["seed"] for r in reps]
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"{len(seeds) - len(set(seeds))} replicates repeat a seed: each must be counted once")
    n, L, bins = len(reps), CRITERIA["draws_per_replicate"], CRITERIA["rank_bins"]
    alpha = CRITERIA["family_alpha"] / len(PARAMS)
    print(f"{len(files)} task files ({partial} incomplete), {n} replicates", file=sys.stderr)
    summary = {"replicates": n, "tasks": len(files), "incomplete_tasks": partial, "virgil": sorted(versions), "virgil_commit": commits.pop(), "criteria": CRITERIA,
               "criteria_hash": criteria_hash(),
               "parameters": {}}
    ok = n >= CRITERIA["min_replicates"]
    for name in PARAMS:
        ranks = np.array([r["ranks"][name] for r in reps])
        counts = np.histogram(ranks, bins=bins, range=(-0.5, L + 0.5))[0]
        p_value = float(stats.chisquare(counts).pvalue)
        cov = {}
        for level in CRITERIA["coverage_levels"]:
            # truth inside the central interval: rank within its middle fraction
            half = level * (L + 1) / 2
            accepted = np.abs(np.arange(L + 1) - L / 2) < half
            p_null = float(np.mean(accepted))  # exact: ranks are uniform on 0..L under the null
            inside = np.mean(accepted[ranks])
            lo, hi = binomial_band(n, p_null, CRITERIA["coverage_band"])
            cov[str(level)] = {"coverage": float(inside), "expected": p_null, "band": [lo, hi],
                               "pass": bool(lo <= inside <= hi)}
        passed = p_value > alpha and all(c["pass"] for c in cov.values())
        ok &= passed
        summary["parameters"][name] = {"chi2_p": p_value, "alpha": alpha, "coverage": cov, "pass": passed,
                                       "counts": counts.tolist()}
    summary["sampler"] = {
        "low_ess": int(sum(min(r["ess"].values()) < CRITERIA["min_bulk_ess"] for r in reps)),
        "high_rhat": int(sum(max(r["rhat"].values()) > CRITERIA["max_rhat"] for r in reps)),
        "with_divergences": int(sum(r["divergences"] > 0 for r in reps)),
    }
    summary["pass"] = bool(ok)
    text = json.dumps(summary, indent=1)
    if args.summary:
        pathlib.Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.summary).write_text(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--task", type=int, required=True)
    r.add_argument("--replicates", type=int, default=10)
    r.add_argument("--seed-base", type=int, default=20261005)
    r.add_argument("--warmup", type=int, default=500)
    r.add_argument("--samples", type=int, default=1000)
    r.add_argument("--chains", type=int, default=4)
    r.add_argument("--out", required=True)
    a = sub.add_parser("aggregate")
    a.add_argument("dirs", nargs="+")
    a.add_argument("--summary")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else aggregate(args)


if __name__ == "__main__":
    main()
