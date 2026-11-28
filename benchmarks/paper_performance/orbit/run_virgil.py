"""virgil side of the orbit comparison (plan section 5): NUTS on relative astrometry (Level A) or on OIFITS (Level B).

One process = one (case, config, thread setting, repeat): it times ``t_calib``, ``t_import``, ``t_load``, ``t_init``,
``t_cold`` (the first full sampling call, compilation included) and ``t_warm`` (the second call, a new seed, the same
compiled ``MCMC`` object), writes the draws to ``--out``/samples.npz and prints one ``BENCH_INNER`` line.

virgil only; this file never imports Octofitter. Conventions are those of virgil's documentation
(``docs/conventions.md#orbits``, ``docs/orbit_methods.md``): ``Omega`` the node's position angle where the secondary
recedes, ``omega`` the secondary's argument of periastron, times in days from ``t_ref``.

    ~/.claude/bin/pyquick $BENCH_PYTHON run_virgil.py --case O-A-sim --config matched --draws 200 --warmup 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "src"))

import numpy as onp  # noqa: E402

import common  # noqa: E402
import make_data as md  # noqa: E402

DATA = md.DATA
T_REF_REAL = 59330.0
REAL = dict(P_mean=10.658451, P_sd=0.001, plx=16.176)
SIM_PLX = md.TRUTH["plx"]
SIM_P_RANGE = (150.0, 250.0)
M_TOT_RANGE = (0.5, 5.0)
E_MAX = 0.9
# virgil's documented setting (virgil/pipeline/binary.py, docs/pipeline.md): the chains run in one vectorised program, so the
# sampling loop is compiled once. "sequential" made NumPyro recompile the loop for every chain (run 1 review).
CHAIN_METHOD = "vectorized"
KINDS = {"O-A-sim": "A", "O-A-sim-pulls": "A", "O-A-real": "A", "O-B-sim": "B"}


def case_info(case):
    """Reference epoch, parallax and period prior (name, args) of a case. Chain starts are derived from the priors
    (``trial_periods``, ``start_values``), never from per-case numbers."""
    if case == "O-A-real":
        lo, hi = REAL["P_mean"] - 5 * REAL["P_sd"], REAL["P_mean"] + 5 * REAL["P_sd"]
        return dict(t_ref=T_REF_REAL, plx=REAL["plx"], p_prior=("tnormal", REAL["P_mean"], REAL["P_sd"], lo, hi))
    return dict(t_ref=md.T_REF, plx=SIM_PLX, p_prior=("uniform",) + SIM_P_RANGE)


def read_astrometry(path):
    a = onp.genfromtxt(path, delimiter=",", names=True)
    return {k: onp.atleast_1d(a[k]) for k in a.dtype.names}


def positions_from(a, t_ref):
    from virgil.orbits import PositionData

    cov = onp.zeros((a["epoch"].size, 2, 2))
    cov[:, 0, 0] = a["sigma_ra"] ** 2
    cov[:, 1, 1] = a["sigma_dec"] ** 2
    cov[:, 0, 1] = cov[:, 1, 0] = a["cor"] * a["sigma_ra"] * a["sigma_dec"]
    return PositionData(a["epoch"], a["ra"], a["dec"], cov, t_ref=t_ref)


def make_orbit_fn(plx, t_ref, phase):
    """values -> KeplerOrbit. ``phase``: "M0" (mean anomaly at t_ref, degrees; matched) or "theta" (position angle at
    t_ref, degrees; native, Thompson et al. 2023 form as ``KeplerOrbit.from_position_angle``)."""
    import jax.numpy as np

    from virgil.orbits import KeplerOrbit

    def a_mas(v):
        return plx * (v["M_tot"] * v["P"] ** 2 / md.KEPLER_K) ** (1.0 / 3.0)

    def orbit_m0(v):
        dt_peri = -np.deg2rad(v["M0"]) * v["P"] / (2.0 * np.pi)
        return KeplerOrbit(v["P"], dt_peri, v["e"], v["inc"], v["omega"], v["Omega"], a_mas(v), t_ref)

    def orbit_theta(v):
        return KeplerOrbit.from_position_angle(v["P"], v["theta"], v["e"], v["inc"], v["omega"], v["Omega"],
                                               a_mas(v), t_ref)

    return orbit_m0 if phase == "M0" else orbit_theta


def make_priors(p_prior, phase):
    import numpyro.distributions as dist

    from virgil.angles import AngleVector
    from virgil.priors import IsotropicInclination

    if p_prior[0] == "uniform":
        pp = dist.Uniform(p_prior[1], p_prior[2])
    else:
        pp = dist.TruncatedNormal(p_prior[1], p_prior[2], low=p_prior[3], high=p_prior[4])
    pri = {"P": pp, "M_tot": dist.Uniform(*M_TOT_RANGE), "e": dist.Uniform(0.0, E_MAX),
           "inc": IsotropicInclination(), "omega": AngleVector(), "Omega": AngleVector()}
    pri["M0" if phase == "M0" else "theta"] = AngleVector()
    return pri


# ---- chain starts -------------------------------------------------------------------------------------------------
# One rule for every case, both levels and both configs (CHANGES.md 2026-11-23, README "Chain starts"):
#   1. the period search covers the central [q, 1 - q] quantile interval of the period prior;
#   2. the trial periods are the midpoints (in frequency) of virgil's phase-coherent ``period_grid`` cells over that
#      interval, so no trial is an end point; an interval with fewer cells than chains (a period prior narrower than one
#      phase-coherence step, which the astrometry cannot resolve: O-A-real, whose narrow prior comes from an external
#      period) uses equal-prior-mass strata instead, P_j = icdf((j + 1/2) / chains), clipped into the interval, so the
#      chains start spread as the prior is (about +-0.3 and +-1.2 sigma for 4 chains), not at the tails;
#   3. every scalar start value is clipped into the [q, 1 - q] interval of its own prior, so it is strictly inside the
#      support and at most |logit(q)| from the centre in the coordinates NUTS samples in (a guard: recorded when used);
#   4. no random numbers: the starts are a function of the data and the priors alone.
# q equals virgil's own inset for NUTS starts (virgil/pipeline/_common.py, _START_INSET = 1e-3 of the prior's width,
# the same thing as a quantile, in the flat coordinate (cos i for inclination), for the uniform and isotropic priors here).
START_Q = 1e-3
INIT_RULE = {
    "A": "starting_orbits (Thiele-Innes grid search, n_best = chains) over trial_periods",
    "B": "epoch_positions -> starting_orbits (n_best = 200) over trial_periods -> rank_orbits -> chain_starts; flux = median "
         "of the per-epoch fits",
}
INIT_COMMON = ("trial_periods = frequency midpoints of period_grid cells over the period prior's "
               f"central [q, 1-q] quantile interval (a prior narrower than one phase-coherence cell: equal-prior-mass strata icdf((j+1/2)/chains)); every scalar start clipped into its own prior's [q, 1-q] interval; q = {START_Q}; deterministic, "
               "no random numbers")


def prior_interval(prior, q=START_Q):
    """The central interval ``[icdf(q), icdf(1 - q)]`` of a scalar prior: where a chain may start."""
    return float(prior.icdf(q)), float(prior.icdf(1.0 - q))


def trial_periods(times, p_prior, n_min=1, q=START_Q):
    """Trial periods (days, increasing) for ``starting_orbits``: the midpoints, in frequency, of the cells of virgil's
    ``period_grid`` over the period prior's central ``[q, 1 - q]`` interval, with at least ``n_min`` cells.

    ``period_grid`` returns cell edges, end points included (its two end points alone when the interval is narrower than
    one phase-coherence step). The midpoints are spaced as the edges are, lie strictly inside the interval, and every
    period of the interval is within half a step of one of them. With fewer than ``n_min`` cells the astrometry cannot
    separate the periods of the interval at all (the narrow prior comes from outside the data), so the trials are the
    strata ``icdf((j + 1/2) / n_min)`` of the prior, one per chain, clipped into the interval: dispersion in P follows the
    prior, not the grid. The runners pass the number of chains as ``n_min``.
    """
    from virgil.orbits import period_grid

    lo, hi = prior_interval(p_prior, q)
    f = 1.0 / period_grid(times, lo, hi)
    if f.size - 1 < n_min:      # narrower than the data can resolve: equal-prior-mass strata, derived from the prior alone
        strata = [float(p_prior.icdf((j + 0.5) / n_min)) for j in range(n_min)]
        return onp.sort(onp.clip(strata, lo, hi))
    return onp.sort(1.0 / (0.5 * (f[:-1] + f[1:])))


def start_values(orbit, priors, plx, phase, t_ref, extra=None, q=START_Q):
    """Site values (including the angle vectors) at a ``starting_orbits`` orbit, and the names of the sites that had to
    be clipped. Every scalar value is clipped into ``prior_interval`` of its own prior; ``M_tot`` follows from the
    orbit's semi-major axis and the (clipped) period. ``extra``: values of sites that are not orbital elements."""
    clipped = []

    def inside(name, x):
        lo, hi = prior_interval(priors[name], q)
        y = float(onp.clip(x, lo, hi))
        if y != float(x):
            clipped.append(name)
        return y

    vals = {"P": inside("P", float(orbit.period))}
    P = vals["P"]
    vals["M_tot"] = inside("M_tot", md.KEPLER_K * (float(orbit.a_mas) / plx) ** 3 / P**2)
    vals["e"] = inside("e", float(orbit.ecc))
    vals["inc"] = inside("inc", float(orbit.inc))
    for k, x in (extra or {}).items():
        vals[k] = inside(k, x)
    angles = {"omega": float(orbit.omega) % 360.0, "Omega": float(orbit.Omega) % 360.0}
    if phase == "M0":
        angles["M0"] = float(onp.rad2deg(-float(orbit.dt_peri) * 2 * onp.pi / P)) % 360.0
    else:
        dra, ddec, _ = orbit.relative(t_ref)
        angles["theta"] = float(onp.rad2deg(onp.arctan2(float(dra), float(ddec)))) % 360.0
    for k, th in angles.items():
        vals[k] = th
        vals[k + "_vec"] = onp.array([onp.cos(onp.deg2rad(th)), onp.sin(onp.deg2rad(th))])
    return vals, clipped


def sampled_sites(priors):
    """``{sampled site: prior}``: an ``AngleVector`` prior is sampled at ``<name>_vec``, the others at their name."""
    from virgil.angles import AngleVector

    return {(k + "_vec" if isinstance(d, AngleVector) else k): d for k, d in priors.items()}


def check_starts(priors, starts):
    """Raise ``ValueError`` naming the site if a start lacks a sampled site (it would start at random), or has a value
    on or outside its prior's support (there the unconstrained value is infinite, and NumPyro only says "Cannot find
    valid initial parameters")."""
    import numpy as np_
    from numpyro.distributions import biject_to

    for k, vals in enumerate(starts):
        for site, prior in sampled_sites(priors).items():
            if site not in vals:
                raise ValueError(f"start {k}: no value for {site}")
            x = np_.asarray(vals[site], float)
            u = np_.asarray(biject_to(prior.support).inv(x))
            ok = bool(np_.all(np_.asarray(prior.support(x)))) and bool(np_.all(np_.isfinite(u)))
            if x.ndim and not float(np_.linalg.norm(x)) > 0.0:  # an angle vector of zero length has no angle
                ok = False
            if not ok:
                raise ValueError(f"start {k}: {site} = {x} is on or outside the support of its prior ({prior.support}); "
                                 f"unconstrained value {u}")


def start_report(kind, periods, starts, clipped, p_prior, scores=None):
    """What the starts were, for the row's ``settings``: the rule, the realised trial periods and the clips used."""
    lo, hi = prior_interval(p_prior)
    count = {k: sum(k in c for c in clipped) for k in sorted({k for c in clipped for k in c})}
    out = {"q": START_Q, "period_window_d": [lo, hi], "n_trial_periods": int(len(periods)),
           "trial_period_range_d": [float(periods[0]), float(periods[-1])], "start_P_d": [float(v["P"]) for v in starts],
           "chains_clipped_by_site": count}
    if scores is not None:
        out["start_chi2"] = [float(s) for s in scores]
    out["text"] = (f"{INIT_RULE[kind]}; {INIT_COMMON}; here {len(periods)} trial period(s) in "
                   f"[{periods[0]:.6f}, {periods[-1]:.6f}] d, clipped sites (chains): {count or 'none'}")
    return out


def level_a_values(priors, positions, times, plx, phase, t_ref, chains):
    """Level A starts: the ``chains`` best orbits of ``starting_orbits`` over ``trial_periods``. Returns one dict of
    site values per chain and the ``start_report``."""
    from virgil.orbits import starting_orbits

    periods = trial_periods(times, priors["P"], n_min=chains)
    found = starting_orbits(positions, periods, n_best=chains)
    if len(found) < chains:
        raise ValueError(f"starting_orbits returned {len(found)} orbits for {chains} chains")
    vals, clipped = zip(*(start_values(o, priors, plx, phase, t_ref) for o, _ in found))
    return list(vals), start_report("A", periods, vals, clipped, priors["P"], [s for _, s in found])


def init_params(model, priors, vals, seed):
    """numpyro's ``init_params`` (unconstrained, one row per chain) at ``vals``, after ``check_starts``. ``seed`` keys the
    sites without a value; ``check_starts`` leaves none, so the result does not depend on it."""
    import jax

    from virgil.likelihood import chain_init_params

    check_starts(priors, vals)
    return chain_init_params(model, vals, key=jax.random.PRNGKey(seed))


def level_a_model(positions, plx, t_ref, p_prior, phase):
    from virgil.likelihood import numpyro_model
    from virgil.orbits import position_angle_prior

    orbit_fn = make_orbit_fn(plx, t_ref, phase)
    terms = [positions.term(orbit_fn)]
    if phase == "theta":
        terms.append(position_angle_prior(orbit_fn))
    priors = make_priors(p_prior, phase)
    # no OIData: the position term carries all the data (docs/common_arguments.md#likelihoods); the template is unused
    return numpyro_model(lambda **v: None, priors, (), likelihoods=terms), priors, orbit_fn


def level_a_model_arg(plx, t_ref, p_prior, phase):
    """Level A model with the data as an ARGUMENT: ``model(positions)``. One ``MCMC(jit_model_args=True)`` over it is compiled once
    and reused for every dataset of equal shape (O-A-sim-pulls: 20 seeds), instead of recompiling per seed because the data were
    closed-over constants (docs/orbit-compile-time review, fix 2). Same density as ``level_a_model``. Returns ``(model, priors, orbit_fn)``."""
    from virgil.likelihood import numpyro_model
    from virgil.orbits import position_angle_prior

    orbit_fn = make_orbit_fn(plx, t_ref, phase)
    priors = make_priors(p_prior, phase)

    def model(positions):
        terms = [positions.term(orbit_fn)]
        if phase == "theta":
            terms.append(position_angle_prior(orbit_fn))
        numpyro_model(lambda **v: None, priors, (), likelihoods=terms)()

    return model, priors, orbit_fn


# ---- Level B ------------------------------------------------------------------------------------------------------


def read_epochs(independent_cp=True):
    """The 8 OIFITS files as an ``Epochs`` (times from ``epochs_O-B-sim.csv``). ``independent_cp``: the closure-phase
    covariance of ``ClosureNoise`` is dropped (``cp_noise=None``), i.e. independent errors per triangle."""
    import equinox as eqx

    from virgil.epochs import Epochs
    from virgil.oidata import OIData

    import csv

    with open(DATA / "epochs_O-B-sim.csv") as f:
        meta = list(csv.DictReader(f))
    eps, times = {}, {}
    for r in meta:
        d = OIData(str(DATA / "oifits_O-B-sim" / r["file"]))
        if independent_cp:
            d = eqx.tree_at(lambda x: x.cp_noise, d, None, is_leaf=lambda x: x is None)
        name = r["file"].replace(".fits", "")
        eps[name] = d
        times[name] = float(r["epoch"])
    return Epochs(eps, times=times)


def level_b_model(epochs, plx, t_ref, p_prior, phase):
    import numpyro.distributions as dist

    from virgil.likelihood import numpyro_model
    from virgil.models import Attached, PointSource, System
    from virgil.orbits import position_angle_prior

    orbit_fn = make_orbit_fn(plx, t_ref, phase)
    priors = make_priors(p_prior, phase)
    priors["flux"] = dist.Uniform(0.01, 1.0)

    def scene(**v):
        return System(primary=PointSource(), comp=Attached(PointSource(v["flux"]), orbit_fn(v)))

    terms = [position_angle_prior(orbit_fn)] if phase == "theta" else []
    return numpyro_model(epochs.model_fn(scene), priors, epochs.data, likelihoods=terms), priors, orbit_fn


def level_b_values(priors, epochs, plx, t_ref, phase, chains):
    """virgil's documented start for visibility-orbit fits (docs/orbit_fitting.md, "The engine underneath"):
    ``epoch_positions`` (a static binary per epoch on a grid) -> ``starting_orbits`` (Thiele-Innes grid over
    ``trial_periods``) -> ``rank_orbits`` on all the data -> ``chain_starts`` (distinct modes, one per chain). Returns one
    dict of site values per chain and the ``start_report``; the values follow the same rule as Level A (``start_values``)."""
    import numpy as np_

    from virgil.epochs import chain_starts, epoch_positions, rank_orbits
    from virgil.models import Attached, PointSource, System
    from virgil.orbits import starting_orbits

    axis = np_.arange(-30.0, 30.0 + 1e-9, 0.5)
    found = epoch_positions(epochs, {"dra": axis, "ddec": axis, "flux": np_.geomspace(0.05, 1.0, 13)})
    seeds = found.positions(t_ref=t_ref, min_gap=5.0)
    periods = trial_periods(epochs.times, priors["P"], n_min=chains)
    cands = starting_orbits(seeds, periods, n_best=200)
    flux = float(np_.median(found.flux))
    ranked = rank_orbits(lambda orbit: System(primary=PointSource(), comp=Attached(PointSource(flux), orbit)),
                         epochs, [o for o, _ in cands])
    picked = chain_starts(ranked, chains)
    if len(picked.orbits) < chains:      # as Level A: fewer distinct modes than chains would fail later in numpyro with a shape error
        raise ValueError(f"chain_starts returned {len(picked.orbits)} orbits for {chains} chains")
    vals, clipped = zip(*(start_values(o, priors, plx, phase, t_ref, extra={"flux": flux}) for o in picked.orbits))
    return list(vals), start_report("B", periods, vals, clipped, priors["P"])


# ---- sampling and outputs -----------------------------------------------------------------------------------------


def derive(samples, plx):
    """Derived quantities from flat sample arrays: a_mas and the Thiele-Innes constants (A, B, F, G) via the neutral
    crosscheck function (the convention ``dDec = A x + F y``, ``dRA = B x + G y``)."""
    from crosscheck.orbits import thiele_innes

    a = plx * (samples["M_tot"] * samples["P"] ** 2 / md.KEPLER_K) ** (1.0 / 3.0)
    A, B, F, G = thiele_innes(a, samples["inc"], samples["omega"], samples["Omega"])
    return {"a_mas": a, "A": A, "B": B, "F": F, "G": G}


def run_mcmc(model, init, n_warm, n_draw, chains, seed, target_accept=0.8, max_tree_depth=10, mcmc=None, args=()):
    """NUTS over ``model(*args)``. With ``args`` the MCMC is built with ``jit_model_args=True`` so that one compiled sampler serves
    every dataset passed to it (pass the same ``mcmc`` back in)."""
    import jax
    from numpyro.infer import MCMC, NUTS

    if mcmc is None:
        kernel = NUTS(model, target_accept_prob=target_accept, max_tree_depth=max_tree_depth, dense_mass=True)
        mcmc = MCMC(kernel, num_warmup=n_warm, num_samples=n_draw, num_chains=chains, chain_method=CHAIN_METHOD,
                    progress_bar=False, jit_model_args=bool(args))
    mcmc.run(jax.random.PRNGKey(seed), *args, init_params=init, extra_fields=("diverging", "num_steps"))
    common.block(mcmc.get_samples(group_by_chain=True))
    return mcmc


def collect(mcmc, plx, chains):
    s = mcmc.get_samples(group_by_chain=True)
    out = {k: onp.asarray(v) for k, v in s.items() if not k.endswith("_vec")}
    flat = {k: v.reshape(-1) for k, v in out.items()}
    der = derive(flat, plx)
    for k, v in der.items():
        out[k] = v.reshape(out["P"].shape)
    ex = mcmc.get_extra_fields(group_by_chain=True)
    out["diverging"] = onp.asarray(ex["diverging"])
    out["num_steps"] = onp.asarray(ex["num_steps"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=list(KINDS))
    ap.add_argument("--config", default="matched", choices=["matched", "native"])
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pull-seeds", type=int, default=md.N_PULL_SEEDS)
    ap.add_argument("--max-tree-depth", type=int, default=10)
    ap.add_argument("--no-warm", action="store_true", help="skip the second (warm) sampling call")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=None, help="directory for samples.npz and the row")
    ap.add_argument("--size", default="full", choices=["tiny", "full"])
    ap.add_argument("--write-row", action="store_true", help="write a validated row to results/raw/ (smoke runs)")
    a = ap.parse_args()

    sw = common.Stopwatch()
    sys.path.insert(0, str(HERE.parent))
    from calib import calibrate

    with sw("t_calib"):
        calibrate()
    with sw("t_import"):
        import jax

        jax.config.update("jax_enable_x64", True)
        import virgil  # noqa: F401
        import virgil.epochs, virgil.likelihood, virgil.orbits, virgil.priors  # noqa: E401,F401
        import numpyro  # noqa: F401

        common.enable_jax_cache()  # after the virgil import: uses virgil.enable_compilation_cache

    kind, info = KINDS[a.case], case_info(a.case)
    phase = "M0" if a.config == "matched" else "theta"
    t_ref, plx = info["t_ref"], info["plx"]
    ver = {"virgil": virgil.__version__, "numpyro": numpyro.__version__, "jax": jax.__version__}
    verify = common.verify_manifest(HERE / "manifest.json", DATA)
    if verify:
        raise SystemExit(f"data differ from manifest.json: {verify}")

    datasets = []
    with sw("t_load"):
        if kind == "A":
            if a.case == "O-A-sim-pulls":
                paths = [DATA / f"astrometry_O-A-sim-pulls_seed{k:02d}.csv" for k in range(a.pull_seeds)]
            else:
                paths = [DATA / f"astrometry_{a.case}.csv"]
            datasets = [(p, read_astrometry(p)) for p in paths]
            n_data = 2 * datasets[0][1]["epoch"].size
        else:
            epochs = read_epochs(independent_cp=True)
            n_data = int(sum(d.n_independent for d in epochs.data))

    if kind == "A":     # data as a model argument: one compiled sampler for every dataset (fix 2 of the compile-time review)
        model_a, priors, orbit_fn = level_a_model_arg(plx, t_ref, info["p_prior"], phase)
    else:
        model_b, priors, orbit_fn = level_b_model(epochs, plx, t_ref, info["p_prior"], phase)

    results, reports = [], []
    shared = None       # the MCMC object reused across datasets (Level A)
    t_init_total, t_cold_total, t_warm_list, flags = 0.0, 0.0, [], []
    for ds in datasets or [None]:
        t0 = time.perf_counter()
        if kind == "A":
            pos = positions_from(ds[1], t_ref)
            model, args = (lambda pos=pos: model_a(pos)), (pos,)       # a zero-argument view of the model for the start search
            vals, report = level_a_values(priors, pos, ds[1]["epoch"], plx, phase, t_ref, a.chains)
        else:
            model, args = model_b, ()
            vals, report = level_b_values(priors, epochs, plx, t_ref, phase, a.chains)
        init = init_params(model, priors, vals, a.seed)
        reports.append(report)
        t_init_total += time.perf_counter() - t0
        t0 = time.perf_counter()
        mcmc = run_mcmc(model_a if kind == "A" else model, init, a.warmup, a.draws, a.chains, a.seed, max_tree_depth=a.max_tree_depth,
                        mcmc=shared, args=args)
        shared = mcmc if kind == "A" else None
        t_cold_total += time.perf_counter() - t0
        res = collect(mcmc, plx, a.chains)
        results.append((ds, res))
        if not a.no_warm and ds is (datasets[0] if datasets else None):
            t0 = time.perf_counter()
            run_mcmc(model_a if kind == "A" else model, init, a.warmup, a.draws, a.chains, a.seed + 1, mcmc=mcmc, args=args)
            t_warm_list.append(time.perf_counter() - t0)
    if kind == "B":
        # gate 3 (gates_result.json): independent-CP likelihood differs from neutral by 2.4e-2 (default 0.20); flagged, not gated
        flags.append("likelihood_mismatch")
    sw.times["t_init"] = t_init_total
    sw.times["t_cold"] = t_cold_total
    for t in t_warm_list:
        sw.lists.setdefault("t_warm", []).append(t)
    if t_warm_list and t_warm_list[0] > 0.9 * t_cold_total / max(len(datasets), 1):
        flags.append("warm_recompiled")

    out = Path(a.out) if a.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        first = results[0][1]
        onp.savez(out / "samples.npz", **first)
        if len(results) > 1:
            onp.savez(out / "pulls_samples.npz", **{f"seed{k:02d}_{key}": v for k, (_, r) in enumerate(results)
                                                     for key, v in r.items() if key in ("P", "e", "a_mas", "A", "B", "F", "G")})
    first = results[0][1]
    acc = {"likelihood_mismatch_rel": 0.02383627411791079 if kind == "B" else None, "n_divergent": int(first["diverging"].sum()), "mean_steps_per_iteration": float(first["num_steps"].mean())}
    settings = {"config": a.config, "phase_parameter": phase, "warmup": a.warmup, "draws": a.draws, "chains": a.chains,
                "chain_method": CHAIN_METHOD, "target_accept": 0.8, "max_tree_depth": a.max_tree_depth,
                "mass_matrix": "dense (matches Octofitter: AdvancedHMC DenseEuclideanMetric, sampling.jl)", "case": a.case, "versions": ver,
                "init": reports[0]["text"] + (f" (text and init_detail: dataset 0 of {len(reports)})" if len(reports) > 1 else ""),
                "init_detail": {k: v for k, v in reports[0].items() if k != "text"},
                "init_start_P_range_all_datasets_d": [min(min(r["start_P_d"]) for r in reports), max(max(r["start_P_d"]) for r in reports)],
                "n_datasets": len(reports), "t_cold_note": "t_init and t_cold are sums over n_datasets fits (t_cold_per_fit = t_cold / n_datasets); t_warm is ONE fit (dataset 0)",
                "sampler": "one MCMC(jit_model_args=True) reused for every dataset (Level A: data are a model argument)" if kind == "A" else "MCMC per run (Level B: data in the model)",
                "init_chains_clipped_all_datasets": sum(sum(r["chains_clipped_by_site"].values()) for r in reports),
                "closure_phase_noise": "independent (cp_noise=None; von Mises residual)" if kind == "B" else None}
    common.emit_inner(sw, n_data=n_data, n_param=len(priors) + (0), iterations={
        "sampling_calls": 1 + len(t_warm_list), "total_leapfrog_steps_first_call": int(first["num_steps"].sum()),
        "n_datasets": len(reports), "t_init_per_fit": t_init_total / len(reports), "t_cold_per_fit": t_cold_total / len(reports)},
        accuracy=acc, settings=settings, flags=flags,
        provenance=common.provenance(data_manifest=common.manifest_hash(HERE / "manifest.json")))
    if out:
        (out / "run.json").write_text(json.dumps({"settings": settings, "accuracy": acc}, indent=1, default=float))


if __name__ == "__main__":
    main()
