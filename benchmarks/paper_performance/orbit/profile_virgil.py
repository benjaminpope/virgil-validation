"""Profiling arm, virgil side: where does a warm orbit fit spend its time? (PR 275, "can Octofitter's faster warm run be improved?")

For one case (O-A-sim, O-A-sim-pulls seed 0, O-B-sim) this builds the same model, start and sampler as ``run_virgil.py`` and
reports, as one JSON line (``PROFILE_VIRGIL {...}``):

* ``per_call_us``: the median time of one compiled log-density call, one value-and-gradient call and the gradient of the chains
  vmapped (``--chains``), after compilation, ``block_until_ready`` inside the timer (one NUTS leapfrog step costs one gradient);
* ``warm``: the wall time of the second NUTS call, the total leapfrog steps it took and the time per step, which is the number to
  compare with ``per_call_us.grad`` (a ratio far above 1 means the time is outside the gradient: the sampler loop, the chains
  running in lock-step to the slowest tree, host dispatch);
* with ``JAX_PROFILE_DIR`` set, an XProf trace (``xplane.pb``) of the warm NUTS call and of a loop of gradient calls
  (``jaxprof.py``; read with ``jax_trace_summary.py``).

    ~/.claude/bin/pyquick $BENCH_PYTHON profile_virgil.py --case O-A-sim --warmup 50 --draws 50 --calls 50
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))

import numpy as onp  # noqa: E402

import common  # noqa: E402
import run_virgil as rv  # noqa: E402


def timed(fn, n):
    """Median and min seconds of n calls of fn() (which must block)."""
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="O-A-sim", choices=list(rv.KINDS))
    ap.add_argument("--config", default="matched", choices=["matched", "native"])
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--calls", type=int, default=500, help="timed calls per microbenchmark")
    ap.add_argument("--out", default=None, help="write the JSON here as well")
    a = ap.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)
    import virgil  # noqa: F401
    import virgil.epochs, virgil.likelihood, virgil.orbits  # noqa: E401,F401
    from numpyro.infer.util import initialize_model

    import jaxprof

    kind, info = rv.KINDS[a.case], rv.case_info(a.case)
    phase = "M0" if a.config == "matched" else "theta"
    t_ref, plx = info["t_ref"], info["plx"]
    if kind == "A":
        path = rv.DATA / ("astrometry_O-A-sim-pulls_seed00.csv" if a.case == "O-A-sim-pulls" else f"astrometry_{a.case}.csv")
        ds = rv.read_astrometry(path)
        pos = rv.positions_from(ds, t_ref)
        model, priors, _ = rv.level_a_model(pos, plx, t_ref, info["p_prior"], phase)
        vals, _ = rv.level_a_values(priors, pos, ds["epoch"], plx, phase, t_ref, a.chains)
    else:
        epochs = rv.read_epochs(independent_cp=True)
        model, priors, _ = rv.level_b_model(epochs, plx, t_ref, info["p_prior"], phase)
        vals, _ = rv.level_b_values(priors, epochs, plx, t_ref, phase, a.chains)
    init = rv.init_params(model, priors, vals, 0)

    # one potential (negative log density) function and its gradient, at each chain's start, in unconstrained space
    mi = initialize_model(jax.random.PRNGKey(0), model, dynamic_args=False)      # for the potential function only
    zs = [jax.tree_util.tree_map(lambda x, k=k: x[k], init) for k in range(a.chains)]      # chain_init_params is already unconstrained
    pot = mi.potential_fn
    z0 = zs[0]
    zstack = jax.tree_util.tree_map(lambda *xs: jax.numpy.stack(xs), *zs)
    f = jax.jit(pot)
    g = jax.jit(jax.value_and_grad(pot))
    gv = jax.jit(jax.vmap(jax.value_and_grad(pot)))
    for fn, arg in ((f, z0), (g, z0), (gv, zstack)):       # compile outside the timers and the trace
        jax.block_until_ready(fn(arg))
    per_call = {}
    for name, fn, arg in (("logdensity", f, z0), ("grad", g, z0), (f"grad_vmap_{a.chains}", gv, zstack)):
        med, mn = timed(lambda fn=fn, arg=arg: jax.block_until_ready(fn(arg)), a.calls)
        per_call[name] = {"median_us": med * 1e6, "min_us": mn * 1e6}
    with jaxprof.maybe_profile("virgil_grad_" + a.case):
        for i in range(20):
            with jaxprof.step(i, "grad"):
                jax.block_until_ready(g(z0))

    # the sampler: cold (compile) call, then the warm call timed and traced; steps from the extra field
    t0 = time.perf_counter()
    mcmc = rv.run_mcmc(model, init, a.warmup, a.draws, a.chains, 1)
    t_cold = time.perf_counter() - t0
    with jaxprof.maybe_profile("virgil_nuts_" + a.case, host_level=1):
        t0 = time.perf_counter()
        rv.run_mcmc(model, init, a.warmup, a.draws, a.chains, 2, mcmc=mcmc)
        t_warm = time.perf_counter() - t0
    steps = onp.asarray(mcmc.get_extra_fields(group_by_chain=True)["num_steps"])      # (chains, warmup-free draws) of the warm call
    total, longest = int(steps.sum()), int(steps.max(axis=0).sum())                   # all chains vs the slowest chain per draw
    out = {
        "case": a.case, "config": a.config, "device": common.DEVICE, "jax_devices": [str(d) for d in jax.devices()],
        "versions": {"virgil": virgil.__version__, "jax": jax.__version__},
        "n_param": len(priors), "chains": a.chains, "warmup": a.warmup, "draws": a.draws, "calls": a.calls,
        "per_call_us": per_call,
        "warm": {"t_s": t_warm, "t_cold_s": t_cold, "steps_all_chains": total, "steps_slowest_chain_per_draw": longest,
                 "mean_steps": total / steps.size,
                 "us_per_step_all": t_warm * 1e6 / max(total, 1),
                 "us_per_step_lockstep": t_warm * 1e6 / max(longest * 1.0, 1),
                 "note": "the warm time covers warmup + draws; steps are counted for the draws only, so per-step figures are upper bounds"},
    }
    line = json.dumps(out, default=float)
    print("PROFILE_VIRGIL", line)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
