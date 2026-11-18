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
KINDS = {"O-A-sim": "A", "O-A-sim-pulls": "A", "O-A-real": "A", "O-B-sim": "B"}


def case_info(case):
    """Reference epoch, parallax and period prior (name, args) of a case."""
    if case == "O-A-real":
        lo, hi = REAL["P_mean"] - 5 * REAL["P_sd"], REAL["P_mean"] + 5 * REAL["P_sd"]
        return dict(t_ref=T_REF_REAL, plx=REAL["plx"], p_prior=("tnormal", REAL["P_mean"], REAL["P_sd"], lo, hi),
                    p_range=(lo, hi))
    return dict(t_ref=md.T_REF, plx=SIM_PLX, p_prior=("uniform",) + SIM_P_RANGE, p_range=SIM_P_RANGE)


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


def start_values(orbit, plx, phase, t_ref):
    """Site values (including the angle vectors) at a ``starting_orbits`` orbit, M_tot clipped into its prior."""
    P = float(orbit.period)
    M_tot = float(onp.clip(md.KEPLER_K * (float(orbit.a_mas) / plx) ** 3 / P**2, M_TOT_RANGE[0] * 1.001,
                           M_TOT_RANGE[1] * 0.999))
    vals = {"P": P, "M_tot": M_tot, "e": float(onp.clip(float(orbit.ecc), 1e-3, E_MAX * 0.999)),
            "inc": float(onp.clip(float(orbit.inc), 1.0, 179.0))}
    angles = {"omega": float(orbit.omega) % 360.0, "Omega": float(orbit.Omega) % 360.0}
    if phase == "M0":
        angles["M0"] = float(onp.rad2deg(-float(orbit.dt_peri) * 2 * onp.pi / P)) % 360.0
    else:
        dra, ddec, _ = orbit.relative(t_ref)
        angles["theta"] = float(onp.rad2deg(onp.arctan2(float(dra), float(ddec)))) % 360.0
    for k, th in angles.items():
        vals[k] = th
        vals[k + "_vec"] = onp.array([onp.cos(onp.deg2rad(th)), onp.sin(onp.deg2rad(th))])
    return vals


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


def level_b_start(model, epochs, plx, t_ref, info, phase, chains):
    """virgil's documented start for visibility-orbit fits (docs/orbit_fitting.md, "The engine underneath"):
    ``epoch_positions`` (a static binary per epoch on a grid) -> ``starting_orbits`` (Thiele-Innes grid over the prior's
    period window) -> ``rank_orbits`` on all the data -> ``chain_starts`` (distinct modes, one per chain)."""
    import numpy as np_

    from virgil.epochs import chain_starts, epoch_positions, rank_orbits
    from virgil.likelihood import chain_init_params
    from virgil.models import Attached, PointSource, System
    from virgil.orbits import period_grid, starting_orbits

    axis = np_.arange(-30.0, 30.0 + 1e-9, 0.5)
    found = epoch_positions(epochs, {"dra": axis, "ddec": axis, "flux": np_.geomspace(0.05, 1.0, 13)})
    seeds = found.positions(t_ref=t_ref, min_gap=5.0)
    periods = period_grid(epochs.times, *info["p_range"])
    cands = starting_orbits(seeds, periods, n_best=200)
    flux = float(np_.median(found.flux))
    ranked = rank_orbits(lambda orbit: System(primary=PointSource(), comp=Attached(PointSource(flux), orbit)),
                         epochs, [o for o, _ in cands])
    picked = chain_starts(ranked, chains)
    vals = []
    for orbit in picked.orbits:
        v = start_values(orbit, plx, phase, t_ref)
        v["flux"] = float(np_.clip(flux, 0.0101, 0.999))
        vals.append(v)
    return chain_init_params(model, vals)


# ---- sampling and outputs -----------------------------------------------------------------------------------------


def derive(samples, plx):
    """Derived quantities from flat sample arrays: a_mas and the Thiele-Innes constants (A, B, F, G) via the neutral
    crosscheck function (the convention ``dDec = A x + F y``, ``dRA = B x + G y``)."""
    from crosscheck.orbits import thiele_innes

    a = plx * (samples["M_tot"] * samples["P"] ** 2 / md.KEPLER_K) ** (1.0 / 3.0)
    A, B, F, G = thiele_innes(a, samples["inc"], samples["omega"], samples["Omega"])
    return {"a_mas": a, "A": A, "B": B, "F": F, "G": G}


def run_mcmc(model, init, n_warm, n_draw, chains, seed, target_accept=0.8, max_tree_depth=10, mcmc=None):
    import jax
    from numpyro.infer import MCMC, NUTS

    if mcmc is None:
        kernel = NUTS(model, target_accept_prob=target_accept, max_tree_depth=max_tree_depth, dense_mass=True)
        mcmc = MCMC(kernel, num_warmup=n_warm, num_samples=n_draw, num_chains=chains, chain_method="sequential",
                    progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), init_params=init, extra_fields=("diverging", "num_steps"))
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

    def build(ds):
        if kind == "A":
            pos = positions_from(ds[1], t_ref)
            return level_a_model(pos, plx, t_ref, info["p_prior"], phase)
        return level_b_model(epochs, plx, t_ref, info["p_prior"], phase)

    from virgil.likelihood import chain_init_params
    from virgil.orbits import period_grid, starting_orbits

    results = []
    t_init_total, t_cold_total, t_warm_list, flags = 0.0, 0.0, [], []
    for ds in datasets or [None]:
        model, priors, orbit_fn = build(ds)
        t0 = time.perf_counter()
        if kind == "A":
            pos = positions_from(ds[1], t_ref)
            periods = period_grid(ds[1]["epoch"], *info["p_range"])
            starts = starting_orbits(pos, periods, n_best=a.chains)
            vals = [start_values(o, plx, phase, t_ref) for o, _ in starts]
            while len(vals) < a.chains:
                vals.append(vals[-1])
            init = chain_init_params(model, vals[: a.chains])
        else:
            init = level_b_start(model, epochs, plx, t_ref, info, phase, a.chains)
        t_init_total += time.perf_counter() - t0
        t0 = time.perf_counter()
        mcmc = run_mcmc(model, init, a.warmup, a.draws, a.chains, a.seed, max_tree_depth=a.max_tree_depth)
        t_cold_total += time.perf_counter() - t0
        res = collect(mcmc, plx, a.chains)
        results.append((ds, res))
        if not a.no_warm and ds is (datasets[0] if datasets else None):
            t0 = time.perf_counter()
            run_mcmc(model, init, a.warmup, a.draws, a.chains, a.seed + 1, mcmc=mcmc)
            t_warm_list.append(time.perf_counter() - t0)
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
    acc = {"n_divergent": int(first["diverging"].sum()), "mean_steps_per_iteration": float(first["num_steps"].mean())}
    settings = {"config": a.config, "phase_parameter": phase, "warmup": a.warmup, "draws": a.draws, "chains": a.chains,
                "chain_method": "sequential", "target_accept": 0.8, "max_tree_depth": a.max_tree_depth,
                "mass_matrix": "dense (matches Octofitter: AdvancedHMC DenseEuclideanMetric, sampling.jl)", "case": a.case, "versions": ver,
                "init": "starting_orbits (period_grid over the prior window, n_best = chains)" if kind == "A" else
                        "numpyro default (init_to_uniform)",
                "closure_phase_noise": "independent (cp_noise=None; von Mises residual)" if kind == "B" else None}
    common.emit_inner(sw, n_data=n_data, n_param=len(priors) + (0), iterations={
        "sampling_calls": 1 + len(t_warm_list), "total_leapfrog_steps_first_call": int(first["num_steps"].sum())},
        accuracy=acc, settings=settings, flags=flags,
        provenance=common.provenance(data_manifest=common.manifest_hash(HERE / "manifest.json")))
    if out:
        (out / "run.json").write_text(json.dumps({"settings": settings, "accuracy": acc}, indent=1, default=float))


if __name__ == "__main__":
    main()
