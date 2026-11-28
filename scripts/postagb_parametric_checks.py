"""Robustness checks of the post-AGB parametric disc fits (stages ``sample`` and ``inject``; design section 7, priority P2 and P3).

``sample``: NUTS on the posterior of the N2-selected model of a target (one attempt, 2 chains in sequence, 500 + 1000 by default, a time box), started
at the fit with the Gauss-Newton mass matrix of virgil. Kept as converged if split r-hat < 1.01 and the ESS of every parameter exceeds 400; otherwise
the draws are stored with the flag ``not_converged``. If the selected fit has a ``multimodal`` competitor, that mode is sampled too and reported apart.
The posterior sds are set beside the Laplace sds of the evidence, and a corner plot is drawn.

``inject``: simulates data from the selected model on the real (u, v) with N2 noise (errors scaled by the MacKay factor, night gains and closure
offsets drawn from their widths), reruns the ladder M0-M2o under N2 with the same code as the real fit, and applies the pass rule: the selected rung is
recovered in at least 2 of the seeds, inc within 10 deg and PA within 15 deg (waived below inc 20 deg), no higher rung accepted.

    python scripts/postagb_parametric_checks.py sample OUT --target AI_Sco [--model M2] [--warmup 500 --samples 1000 --chains 2 --time-box 1800]
    python scripts/postagb_parametric_checks.py inject OUT --target IRAS15469-5311 [--seeds 3] [--models M0,M1,M2,M2o] [--model M2] [--smoke]

``OUT`` is the directory given to ``postagb_parametric.py`` (the stages read its ``select`` and ``ladder`` files and write in ``OUT/<target>/``,
``sample.DONE`` or ``inject.DONE`` last). ``--model`` replaces the selected model by the named one (smoke runs, where there is no ``select`` file).
Everything that is not a check (scene, priors, noise arms, evidence, selection) is imported from ``postagb_parametric``.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import postagb_parametric as Q  # noqa: E402  (sets the XLA/OMP thread defaults and float64 before JAX starts)
import jax  # noqa: E402
import numpy as np  # noqa: E402
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin  # noqa: E402
from numpyro.infer import MCMC, NUTS, init_to_value  # noqa: E402
from virgil.fitting import gauss_newton_mass  # noqa: E402
from virgil.likelihood import numpyro_model  # noqa: E402

P = Q.P
RHAT_MAX, ESS_MIN = 1.01, 400.0  # a sample is kept when every parameter has split r-hat < 1.01 and ESS > 400
SD_RATIO = (0.7, 1.3)  # posterior sd / Laplace sd outside this range is flagged
INJECT_SEEDS = 3
INC_TOL, PA_TOL, PA_WAIVED_BELOW = 10.0, 15.0, 20.0
INJECT_MODELS = ("M0", "M1", "M2", "M2o")


# ----------------------------------------------------------------------------
# NUTS


def mode_setup(cfg, model_id, values):
    """``(model, priors, start)`` of ``model_id`` at ``values`` (path -> number or list, as in the ladder json): the scene of ``Q.build_scene`` with the
    fitted values set, and the priors of ``Q.priors_for`` with the 180 deg PA window centred on the fitted PA."""
    st = Q.start_from_values(values, model_id, cfg, name="optimum")
    scene = Q.build_scene(cfg, model_id, st)
    pri = Q.priors_for(cfg, scene, model_id, st)
    return Q._set_values(scene, {k: v for k, v in values.items() if k in pri}), pri, st


def diagnostics(draws):
    """Split r-hat and bulk ESS of every column of ``draws`` (chains, n, k). With one chain the r-hat compares its two halves."""
    d = np.asarray(draws, float)
    return np.asarray(split_gelman_rubin(d), float), np.asarray(effective_sample_size(d), float)


def nuts_mode(model, priors, data, values, warmup=500, samples=1000, chains=2, seed=0, time_box=1800.0, log=print):
    """NUTS at one mode, ``chains`` chains in sequence (all started at ``values``) with the Gauss-Newton mass matrix, ``target_accept_prob=0.9``.

    A chain that would not fit in ``time_box`` seconds (judged from the first) is not started and ``time_box_hit`` is set. Returns a dict with
    ``draws`` (chains, n, k) in the flat order of ``names``, ``theta`` (the start), ``diverging`` per chain, ``seconds`` per chain."""
    paths = tuple(priors)
    vals = {p: np.asarray(values[p], float) for p in paths}
    theta, shapes = Q._flat(model, paths)
    names = Q._names(paths, shapes)
    nm = numpyro_model(model, priors, data)
    t0 = time.time()
    gn = gauss_newton_mass(model, priors, data, vals)
    log(f"  mass matrix in {time.time() - t0:.1f} s (condition {getattr(gn, 'condition', None)})")
    out, secs, div, hit = [], [], [], False
    for c in range(chains):
        if c and time.time() - t0 + secs[0] > time_box:
            hit = True
            log("  time box reached: chain not started")
            break
        t1 = time.time()
        mcmc = MCMC(NUTS(nm, init_strategy=init_to_value(values=vals), target_accept_prob=0.9, **gn), num_warmup=warmup, num_samples=samples,
                    num_chains=1, progress_bar=False)
        mcmc.run(jax.random.PRNGKey(seed + c), extra_fields=("diverging",))
        smp = mcmc.get_samples()
        out.append(np.concatenate([np.asarray(smp[p], float).reshape(samples, -1) for p in paths], axis=1))
        div.append(int(np.sum(mcmc.get_extra_fields()["diverging"])))
        secs.append(time.time() - t1)
        log(f"  chain {c}: {secs[-1]:.1f} s, {div[-1]} divergences")
    return dict(draws=np.stack(out), names=names, paths=paths, shapes=shapes, theta=theta, diverging=div, seconds=secs, time_box_hit=hit)


def summarise(run, laplace_sd=None):
    """The record of one NUTS run: per parameter mean, sd, split r-hat, ESS and (where ``laplace_sd`` is given) sd / Laplace sd; the worst values;
    and the flags ``not_converged`` (any r-hat >= 1.01 or ESS <= 400, a single chain, or the time box) and ``laplace_sd_mismatch``."""
    d = run["draws"]
    rhat, ess = diagnostics(d)
    flat = d.reshape(-1, d.shape[-1])
    sd = flat.std(axis=0, ddof=1)
    rows = {}
    for i, n in enumerate(run["names"]):
        rows[n] = dict(mean=float(flat[:, i].mean()), sd=float(sd[i]), rhat=float(rhat[i]), ess=float(ess[i]))
        if laplace_sd is not None and np.isfinite(laplace_sd[i]) and laplace_sd[i] > 0:
            rows[n] |= dict(laplace_sd=float(laplace_sd[i]), sd_ratio=float(sd[i] / laplace_sd[i]))
    flags = []
    if d.shape[0] < 2 or run.get("time_box_hit"):
        flags.append("single_chain")
    if not (np.all(np.isfinite(rhat)) and rhat.max() < RHAT_MAX and ess.min() > ESS_MIN):
        flags.append("not_converged")
    ratios = [r["sd_ratio"] for r in rows.values() if "sd_ratio" in r]
    if ratios and (min(ratios) < SD_RATIO[0] or max(ratios) > SD_RATIO[1]):
        flags.append("laplace_sd_mismatch")
    return dict(params=rows, rhat_max=float(np.nanmax(rhat)), ess_min=float(np.nanmin(ess)), n_chains=int(d.shape[0]), n_draws=int(d.shape[1]),
                diverging=run["diverging"], seconds=run["seconds"], flags=flags,
                sd_ratio_range=[float(min(ratios)), float(max(ratios))] if ratios else None)


def corner_plot(draws, names, theta, path, max_points=4000):
    """A corner plot of ``draws`` (chains, n, k) with the start (the fit) marked: histograms on the diagonal, 2-D histograms below it."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    x = np.asarray(draws, float).reshape(-1, len(names))[:: max(1, draws.shape[0] * draws.shape[1] // max_points)]
    k = len(names)
    fig, ax = plt.subplots(k, k, figsize=(1.25 * k + 1, 1.25 * k + 1), squeeze=False)
    for i in range(k):
        for j in range(k):
            a = ax[i, j]
            if j > i:
                a.axis("off")
                continue
            if i == j:
                a.hist(x[:, i], bins=30, color="0.4", histtype="stepfilled")
                a.axvline(theta[i], color="C3", lw=0.8)
                a.set_yticks([])
            else:
                a.hist2d(x[:, j], x[:, i], bins=24, cmap="Greys")
                a.plot(theta[j], theta[i], "+", color="C3", ms=6)
            a.tick_params(labelsize=5)
            if i == k - 1:
                a.set_xlabel(names[j].replace("env.", "").replace("flux.", "f_"), fontsize=5)
            else:
                a.set_xticklabels([])
            if j == 0 and i > 0:
                a.set_ylabel(names[i].replace("env.", "").replace("flux.", "f_"), fontsize=5)
            elif j:
                a.set_yticklabels([])
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=90)
    plt.close(fig)


# ----------------------------------------------------------------------------
# injection-recovery


def simulate(model, data_raw, cfg, s, seed):
    """``data_raw`` with its V^2 and closure phases replaced by those of ``model`` under the N2 noise of ``Q.noise_arm``: independent Gaussian noise of
    sd ``s`` x the quoted error, one log|V| gain per (night, baseline) of width 0.03 and one closure offset per (night, triangle) of 2 deg, drawn by
    virgil's ``OIData.with_model``. The returned data carry the quoted errors, so ``Q.noise_arm(., cfg, 'N2', s)`` reproduces the arm they were drawn from."""
    from postagb_noise import simulate as sim_hook  # noqa: PLC0415  (WP-02: the one simulation hook)

    return sim_hook(cfg, model, noise="nightw", scale=s, seed=seed, data0=data_raw).data


def _circ180(a, b):
    return abs((a - b + 90.0) % 180.0 - 90.0)


def inject_verdict(truth_model, truth_derived, seeds, models=INJECT_MODELS, need=2):
    """The pass rule of design section 7 for ``seeds`` (rows with ``selected``, ``inc``, ``pa_mod180``, ``accepted`` = the rungs accepted in the chain).

    A seed *recovers the rung* when it selects the truth's model; its *geometry is right* when inc is within 10 deg and PA (mod 180) within 15 deg of the
    truth (PA waived when the truth's inc is below 20 deg); *a higher rung is accepted* when the selected rung lies above the truth's. The check passes
    when at least ``need`` seeds recover the rung, every recovering seed has the right geometry and no seed accepts a higher rung."""
    rank = {m: i for i, m in enumerate(models)}
    rows = []
    for r in seeds:
        sel = r.get("selected")
        rung = sel == truth_model
        higher = sel is not None and rank.get(sel, -1) > rank[truth_model]
        d_inc = abs(r["inc"] - truth_derived["inc"]) if r.get("inc") is not None else None
        d_pa = _circ180(r["pa_mod180"], truth_derived["pa_mod180"]) if r.get("pa_mod180") is not None else None
        geom = d_inc is not None and d_inc <= INC_TOL and (truth_derived["inc"] < PA_WAIVED_BELOW or (d_pa is not None and d_pa <= PA_TOL))
        rows.append(dict(seed=r.get("seed"), selected=sel, rung_recovered=bool(rung), higher_rung=bool(higher), d_inc=d_inc, d_pa=d_pa, geometry_ok=bool(geom)))
    n_rung = sum(x["rung_recovered"] for x in rows)
    passed = bool(n_rung >= need and all(x["geometry_ok"] for x in rows if x["rung_recovered"]) and not any(x["higher_rung"] for x in rows))
    return dict(truth_model=truth_model, n_seeds=len(rows), n_rung_recovered=int(n_rung), need=need, seeds=rows, passed=passed)


def inject_seed(cfg, data_raw, truth, s, seed, models=INJECT_MODELS, out=None, smoke=False, time_limit=60.0, log=print):
    """One injection: simulate from ``truth``, run the ladder ``models`` under N2 and select. Returns the compact row ``inject_verdict`` reads, with the
    chain and the evidence and chi^2 of every model."""
    sim = simulate(truth, data_raw, cfg, s, seed)
    lad = Q.fit_ladder(cfg, sim, arms=("N2",), models=models, smoke=smoke, do_is=not smoke, out=out, s=s, time_limit=time_limit, log=log)
    sel = Q.select(lad, "N2", models=models)
    best = sel.get("best") or {}
    d = best.get("derived") or {}
    return dict(seed=int(seed), selected=sel.get("selected"), inc=d.get("inc"), pa_mod180=d.get("pa_mod180"), r0=d.get("r0"), sigma=d.get("sigma"), is_ring=sel.get("is_ring"),
                flags=sel.get("flags"), accepted=[c["child"] for c in sel["chain"] if c["accepted"]],
                chain=[{k: c.get(k) for k in ("pair", "d_lnZ", "d_bic", "accepted", "verdict")} for c in sel["chain"]],
                models={m: dict(lnZ_use=(r["best"] or {}).get("lnZ_use"), bic=(r["best"] or {}).get("bic"), own_chi2=(r["best"] or {}).get("own_chi2"),
                                eligible=bool(r["best"]), seconds=r["seconds"]) for (m, _), r in lad.items()})


# ----------------------------------------------------------------------------
# stages


def _selected(out, cfg, model_override):
    """``(model_id, values)`` of the N2 selection: the ``select`` file's choice (or ``model_override``) and the best N2 ladder fit of that model."""
    key = cfg["key"]
    if model_override:
        model_id = model_override
    else:
        f = out / f"{key}_param_selected.json"
        if not f.exists():
            raise SystemExit(f"no {f.name}: run the select stage first (or name a model with --model)")
        model_id = json.load(open(f))["selected"].get(Q.SELECTION_ARM)
        if model_id is None:
            raise SystemExit("no model selected under N2")
    lad = Q._read_ladders(out, key, (Q.SELECTION_ARM,), (model_id,), None).get((model_id, Q.SELECTION_ARM))
    if not lad or not lad.get("best"):
        raise SystemExit(f"no eligible {model_id} fit under {Q.SELECTION_ARM} in {out}")
    return model_id, lad


def sample_stage(args, out):
    cfg = Q._prepare(args)
    model_id, lad = _selected(out, cfg, args.model)
    s, _, _ = Q.stored_error_scale(cfg)
    data = Q.noise_arm(P.load(cfg)[0], cfg, Q.SELECTION_ARM, s=s)
    best = lad["best"]
    modes = [("mode0", best["values"])]
    if "multimodal" in best["flags"]:
        second = [m for m in lad["modes"][1:] if m["eligible"] and not Q.same_mode(best["values"], m["values"]) and best["lnZ_lap"] - m["lnZ_lap"] < 5.0]
        if second:
            modes.append(("mode1", second[0]["values"]))
    rec, arrays, t0 = dict(target=cfg["key"], model=model_id, arm=Q.SELECTION_ARM, commits=Q.commits(), error_scale=s, settings=vars(args) | dict(out=str(out)),
                           modes={}), {}, time.time()
    for name, values in modes:
        log = lambda m, name=name: print(f"{cfg['key']} {model_id} {name}: {m}", flush=True)  # noqa: E731
        model, pri, _ = mode_setup(cfg, model_id, values)
        try:
            ev = Q.param_evidence(model, pri, data)
            lap = ev["sd"]
        except Exception as e:  # noqa: BLE001
            lap = None
            log(f"Laplace failed: {type(e).__name__}: {e}")
        box = args.time_box / len(modes)
        try:
            run = nuts_mode(model, pri, data, {p: values[p] for p in pri}, args.warmup, args.samples, args.chains, args.seed, box, log)
        except Exception as e:  # noqa: BLE001
            rec["modes"][name] = dict(flags=["failed"], error=f"{type(e).__name__}: {str(e)[:300]}")
            log(f"NUTS failed: {type(e).__name__}: {e}")
            continue
        rec["modes"][name] = summarise(run, lap) | dict(start=values)
        arrays[f"{name}_draws"], arrays[f"{name}_theta"] = run["draws"], run["theta"]
        arrays["names"] = np.array(run["names"])
        if not args.no_corner:
            corner_plot(run["draws"], run["names"], run["theta"], out / f"{cfg['key']}_param_corner{'' if name == 'mode0' else '_' + name}.png")
        log(f"flags {rec['modes'][name]['flags']}, r-hat max {rec['modes'][name]['rhat_max']:.3f}, ESS min {rec['modes'][name]['ess_min']:.0f}")
    rec["seconds"] = time.time() - t0
    if arrays:
        np.savez_compressed(out / f"{cfg['key']}_param_posterior.npz", **arrays)
    with open(out / f"{cfg['key']}_param_posterior.json", "w") as f:
        json.dump(Q._jsonable(rec), f, indent=1)
    Q._done(out, "sample")
    print(f"sample done in {rec['seconds']:.0f} s", flush=True)


def inject_stage(args, out):
    cfg = Q._prepare(args)
    model_id, lad = _selected(out, cfg, args.model)
    models = tuple(args.models.split(","))
    if model_id not in models:
        raise SystemExit(f"the truth model {model_id} is not in --models {models}")
    s, _, _ = Q.stored_error_scale(cfg)
    data_raw, _ = P.load(cfg)
    best = lad["best"]
    truth, _, _ = mode_setup(cfg, model_id, best["values"])
    sub = out / "inject"
    rows, t0 = [], time.time()
    for k in range(args.seeds):
        t1 = time.time()
        row = inject_seed(cfg, data_raw, truth, s, k, models, out=sub / f"seed{k}", smoke=args.smoke, time_limit=args.time_limit,
                          log=lambda m, k=k: print(f"{cfg['key']} inject seed {k}: {m}", flush=True))
        rows.append(row | dict(seconds=time.time() - t1))
        print(f"seed {k}: selected {row['selected']}, inc {row['inc']}, PA {row['pa_mod180']}, {rows[-1]['seconds']:.0f} s", flush=True)
    verdict = inject_verdict(model_id, best["derived"], rows, models)
    rec = dict(target=cfg["key"], arm=Q.SELECTION_ARM, commits=Q.commits(), error_scale=s, models=list(models), truth=dict(model=model_id, values=best["values"], derived=best["derived"]),
               smoke=bool(args.smoke), rows=rows, verdict=verdict, seconds=time.time() - t0)
    with open(out / f"{cfg['key']}_param_inject.json", "w") as f:
        json.dump(Q._jsonable(rec), f, indent=1)
    Q._done(out, "inject")
    print(f"inject {'PASS' if verdict['passed'] else 'FAIL'}: rung recovered in {verdict['n_rung_recovered']} of {verdict['n_seeds']} seeds, {rec['seconds']:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["sample", "inject"])
    ap.add_argument("out", type=Path)
    ap.add_argument("--target", required=True, choices=sorted(Q.MACKAY_S))
    ap.add_argument("--model", default=None, help="use this model instead of the one selected under N2")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-box", type=float, default=1800.0, help="sample: seconds for the target (all modes)")
    ap.add_argument("--no-corner", action="store_true")
    ap.add_argument("--seeds", type=int, default=INJECT_SEEDS)
    ap.add_argument("--models", default=",".join(INJECT_MODELS))
    ap.add_argument("--smoke", action="store_true", help="inject: four starts per fit and no importance sampling")
    ap.add_argument("--time-limit", type=float, default=60.0, help="inject: seconds per fit")
    args = ap.parse_args()
    out = args.out.expanduser() / args.target
    out.mkdir(parents=True, exist_ok=True)
    dict(sample=sample_stage, inject=inject_stage)[args.stage](args, out)


if __name__ == "__main__":
    main()
