"""S2 of the DSHARP convergence plan: the optimiser head-to-head on IM Lup (GPU; ``reports/dsharp-convergence-plan.md`` §4, gate G2).

    alma_optim_s2.py plan      --out DIR                       the runs of tier A and B, one per line
    alma_optim_s2.py tune      --data DIR --out DIR            five off-target trials per method on a 256^2 synthetic with IM Lup's sampling -> tuned.json
    alma_optim_s2.py reference --data DIR --out DIR            two method families at 3x the budget, polished; written only if they agree (|dJ| < 1e-6, image < 1e-5 of peak)
    alma_optim_s2.py run       --data DIR --out DIR --method M --weight-mult {1,30} [--tier A|B] [--start flat|gaussian]
    alma_optim_s2.py score     --data DIR --out DIR --run NAME  the external evaluator on one run's snapshots (its time is nobody's)
    alma_optim_s2.py summary   --out DIR                       the table and gate G2's choice

Objective, identical for all methods: ``J(f) = 1/2 chi2(f) + MaxEntropy(w)`` on ``LinearImage`` sqrt pixels with the explicit Toeplitz chi^2, flat default image, 1028^2 at 5 mas,
``w0`` = 3400 and 30 w0 (``--w0``), weights rescaled by the model-free scale (``--shat DIR``, s = 1 after). Tier A holds the nuisances at S1's values (``--nuisance-json``: site name -> values,
default: nominal); tier B fits gains and offsets (gradient-only methods alternate with a dense Newton step on that block). The start is flat (F / N) or the 1 arcsec Gaussian with floor.
Every run has its own stop disabled, a firm ``--budget`` (default 1200 s) and writes a snapshot every ``--snap-every`` seconds (the checkpoints: a killed job keeps its last point); ``score``
judges them afterwards (``alma_certify``). ``meta.json`` carries both commits (``provenance``). A run that hits its step cap is failed. This script does not touch ``dsharp_fit.py``.
"""

import argparse
import dataclasses
import json
import pathlib
import statistics
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PIXEL_MAS, NPIX, W0 = 5.0, 1028, 3400.0
WEIGHT_MULTS = (1, 30)
BUDGET_S, SNAP_S = 1200.0, 30.0
METHODS = ("lbfgsb", "lbfgs", "lbfgsb_newton", "newton_cg", "skilling_bryan", "mirror", "mirror_accel", "fista", "adam", "softmax_lbfgs")
CONTROL = "softmax_lbfgs"  # w0 only
IN_VIRGIL = ("lbfgsb", "lbfgs")  # the tie rule's 'already in virgil'; then the order below is 'less new code'
LESS_CODE = ("lbfgsb", "lbfgs", "softmax_lbfgs", "adam", "lbfgsb_newton", "newton_cg", "mirror", "mirror_accel", "fista", "skilling_bryan")
TUNE_BUDGET_S = 60.0
TUNE_GRID = {  # five trials per method for its free settings, fixed before any IM Lup run
    "lbfgsb": [dict(maxcor=m) for m in (10, 30, 60, 100, 200)],
    "lbfgs": [dict(memory=m) for m in (5, 10, 20, 50, 100)],
    "lbfgsb_newton": [dict(phase1=p, cg_tol=c) for p, c in ((0.25, 1e-3), (0.5, 1e-3), (0.5, 1e-2), (0.75, 1e-3), (0.1, 1e-2))],
    "newton_cg": [dict(cg_tol=c) for c in (1e-1, 3e-2, 1e-2, 1e-3, 1e-4)],
    "skilling_bryan": [dict(ndir=n, radius=r) for n, r in ((3, 0.1), (4, 0.1), (5, 0.1), (6, 0.1), (5, 0.5))],
    "mirror": [dict(step=s) for s in (0.25, 0.5, 1.0, 1.5, 1.9)],
    "mirror_accel": [dict(step=s) for s in (0.25, 0.5, 1.0, 1.5, 1.9)],
    "fista": [dict(step=s) for s in (0.25, 0.5, 1.0, 1.5, 1.9)],
    "adam": [dict(lr=l) for l in (0.5, 0.2, 0.05, 0.02, 0.005)],
    "softmax_lbfgs": [dict(maxcor=m) for m in (10, 30, 60, 100, 200)],
}


def say(*a):
    print(*a, flush=True)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def runs(w0=W0):
    """The planned runs: ``(name, method, weight multiplier, tier, start)``. Tier A: every method at both weights (the control at w0 only); tier B names the three fastest of tier A, chosen from its scores."""
    out = []
    for mult in WEIGHT_MULTS:
        for m in METHODS:
            if m == CONTROL and mult != 1:
                continue
            out.append((f"A_{m}_w{mult}_flat", m, mult, "A", "flat"))
    return out


# ---------------------------------------------------------------- the problem


def load(args, *, npix=None, pixel_mas=None):
    """``(objective, flux, data)`` for the run's data and tier."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import virgil_bridge.alma as va
    import virgil_bridge.alma_certify as ac
    import virgil_bridge.alma_lambda as vl

    data = va.load_dir(args.data)
    if args.shat:
        shat = va.read_shat(args.shat, int(np.asarray(data.ddid).max()) + 1)
        data = dataclasses.replace(data, weight=np.asarray(data.weight, float) / shat[np.asarray(data.ddid)] ** 2)  # s = 1 after the rescale
    ref, _ = vl.reference_obs(data)
    flux = vl.flux_scale(data)
    npix, px = npix or NPIX, pixel_mas or PIXEL_MAS
    nui = json.loads(pathlib.Path(args.nuisance_json).read_text()) if args.nuisance_json else {}
    joint = args.tier == "B"
    prob = ac.build_s2_problem(data, npix=npix, pixel_mas=px, flux=flux, ref_obs=ref, gains=joint or any("gain" in k for k in nui), offsets=joint or any("offset" in k for k in nui),
                               max_offset_mas=args.max_offset_mas)
    free = tuple(prob.nuisance.defaults()) if joint else ()
    obj = ac.VirgilObjective(prob, args.w0 * args.weight_mult, free=free, fixed={k: np.asarray(v) for k, v in nui.items()})
    return obj, flux, data


def beam_of(npix, pixel_mas, beam_mas):
    from scipy.ndimage import gaussian_filter

    sigma = beam_mas / 2.3548 / pixel_mas
    return lambda f: gaussian_filter(np.asarray(f, float).reshape(npix, npix), sigma, mode="constant")


def start_of(args, obj):
    import virgil_bridge.alma_certify as ac

    if args.start == "gaussian":
        return ac.gaussian_floor_start(obj.prob.npix, obj.prob.pixel_mas, obj.flux)
    return obj.flat()


def tuned(args, method):
    if not args.tuned or not pathlib.Path(args.tuned).exists():
        return {}
    return json.loads(pathlib.Path(args.tuned).read_text()).get(method, {})


# ---------------------------------------------------------------- stages


def cmd_plan(args):
    for r in runs(args.w0):
        say(*r)
    say("tier B: the three fastest of tier A (summary), at both weights, --tier B")


def synthetic_truth(npix, pixel_mas, flux):
    """A bright ring with a gap-free core and a faint halo (Jy/arcsec^2): IM Lup's dynamic range, for tuning only."""
    c = (np.arange(npix) - (npix - 1) / 2.0) * pixel_mas
    r = np.hypot(c[:, None], c[None, :])
    img = 1.0 * np.exp(-0.5 * ((r - 700.0) / 120.0) ** 2) + 0.4 * np.exp(-0.5 * (r / 250.0) ** 2) + 1e-4 * np.exp(-0.5 * (r / 1500.0) ** 2)
    return img * flux / (img.sum() * (pixel_mas / 1000.0) ** 2)


def cmd_tune(args):
    import jax

    jax.config.update("jax_enable_x64", True)
    import virgil_bridge.alma as va
    import virgil_bridge.alma_certify as ac
    import virgil_bridge.alma_lambda as vl
    import virgil_bridge.alma_optim as ao
    import virgil_bridge.alma_synth as syn

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = va.load_dir(args.data)
    npix, px = 256, NPIX * PIXEL_MAS / 256
    flux = vl.flux_scale(data)
    truth = synthetic_truth(npix, px, flux)
    vis = syn.noisy(syn.model_visibilities(data, truth, px), np.asarray(data.weight, float), np.random.default_rng(0))
    sdata = dataclasses.replace(data, vis=vis)
    ref, _ = vl.reference_obs(sdata)
    prob = ac.build_s2_problem(sdata, npix=npix, pixel_mas=px, flux=flux, ref_obs=ref)
    obj = ac.VirgilObjective(prob, args.w0)
    ev = ac.Evaluator(obj)
    best = {}
    for m in METHODS:
        rows = []
        for i, opts in enumerate(TUNE_GRID[m]):
            d = out / "tune" / f"{m}_{i}"
            res = ao.run(m, obj, obj.flat(), budget_s=args.tune_budget, out=d, snap_every=2.0, **opts)
            recs = ac.score_snapshots(ev, str(d))
            ser = ac.certify_series(recs)
            rows.append(dict(opts=opts, t=ser["t"] if ser["certified"] else None, decrement=recs[-1]["decrement"] if recs else None, reason=res.reason))
            say(f"tune {m} {opts}: certified {ser['certified']} t {ser['t']}, decrement {rows[-1]['decrement']}")
        key = lambda r: (r["t"] is None, r["t"] if r["t"] is not None else r["decrement"] if r["decrement"] is not None else np.inf)  # noqa: E731
        best[m] = min(rows, key=key)["opts"]
        (out / f"tune_{m}.json").write_text(json.dumps(jsonable(rows), indent=1))
    (out / "tuned.json").write_text(json.dumps(jsonable(best), indent=1))
    say("tuned:", best)


def _run_dir(args, name):
    d = pathlib.Path(args.out) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _peak_gb():
    import jax

    stats = jax.local_devices()[0].memory_stats() or {}
    return stats.get("peak_bytes_in_use", float("nan")) / 1e9


def cmd_run(args):
    import virgil_bridge.alma_certify as ac
    import virgil_bridge.alma_optim as ao

    name = args.name or f"{args.tier}_{args.method}_w{args.weight_mult}_{args.start}"
    if args.method == CONTROL and args.weight_mult != 1:
        raise SystemExit("the softmax control runs at w0 only")
    d = _run_dir(args, name)
    obj, flux, data = load(args)
    opts = {**tuned(args, args.method), **({} if args.opts is None else json.loads(args.opts))}
    meta = dict(name=name, method=args.method, options=opts, w=obj.weight, weight_mult=args.weight_mult, tier=args.tier, start=args.start, flux=flux, npix=obj.prob.npix,
                pixel_mas=obj.prob.pixel_mas, n_points=len(data.u), budget_s=args.budget, snap_every=args.snap_every, provenance=ac.provenance(), args=vars(args))
    (d / "meta.json").write_text(json.dumps(jsonable(meta), indent=1))
    f0 = start_of(args, obj)
    say(f"{name}: {len(data.u):,} points, w {obj.weight:g}, flux {flux:.4f} Jy, budget {args.budget:g} s")
    t0 = time.time()
    res = ao.run(args.method, obj, f0, budget_s=args.budget, out=d, snap_every=args.snap_every, max_steps=args.max_steps, **opts)
    np.savez(d / "final.npz", f=res.f, theta=res.theta)
    result = dict(meta, reason=res.reason, capped=res.capped, failed=ac.run_failed(res), wall_s=res.wall, job_wall_s=time.time() - t0, grads=res.grads, hvps=res.hvps, geq=res.geq,
                  steps=res.steps, n_snapshots=len(res.snapshots), peak_gpu_gb=_peak_gb(), trace_every_100=res.trace[::100])
    (d / "result.json").write_text(json.dumps(jsonable(result), indent=1))
    say(f"{name}: {res.reason} after {res.wall:.0f} s, {res.geq} gradient-equivalents, {res.steps} steps; wrote {d}")


def reference_stem(tier, mult):
    return f"reference_{tier}_w{mult}"


def cmd_reference(args):
    """The reference of each weight (``reference_<tier>_w<mult>.npz``, with the weight and, for tier B, the nuisances): two families at 3x the budget, polished, kept only if they agree.
    Tier A (default) holds the nuisances at S1's; tier B (``--tier B``) fits them jointly, so tier-B runs are scored against a tier-B optimum, never tier A's."""
    import virgil_bridge.alma_certify as ac
    import virgil_bridge.alma_optim as ao

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    args.start = "flat"
    failed = []
    for mult in ((args.weight_mult,) if args.weight_mult else WEIGHT_MULTS):
        args.weight_mult = mult
        obj, flux, _ = load(args)
        beam = beam_of(obj.prob.npix, obj.prob.pixel_mas, args.beam_mas)
        ev = ac.Evaluator(obj, beam=beam)
        polished = []
        for m in args.methods.split(","):  # two families: one quasi-Newton, one Newton
            res = ao.run(m, obj, obj.flat(), budget_s=3 * args.budget, out=out / f"{reference_stem(args.tier, mult)}_{m}", snap_every=args.snap_every, **tuned(args, m))
            obj.set_theta(res.theta)
            f, k = ao.polish(obj, res.f)
            polished.append((m, f, np.array(res.theta, float)))
            say(f"reference {args.tier} w{mult} {m}: {res.reason} {res.wall:.0f} s, polished in {k} Newton rounds")
        (ma, fa, ta), (mb, fb, tb) = polished
        obj.set_theta(ta)
        ev.set_reference(fb, tb)
        agree, dJ, dist = ac.reference_agreement(ev, fa, fb)
        rec = dict(methods=[ma, mb], tier=args.tier, weight_mult=mult, w=obj.weight, agree=agree, dJ=dJ, image_distance=dist, status="ok" if agree else "failed", provenance=ac.provenance())
        if agree:
            obj.set_theta(ta)
            ev.set_reference(fa, ta)
            rec["reference_record"] = ev.evaluate(fa)
            np.savez(out / f"{reference_stem(args.tier, mult)}.npz", f=fa, theta=ta, w=obj.weight)
        else:
            failed.append(mult)
        (out / f"{reference_stem(args.tier, mult)}.json").write_text(json.dumps(jsonable(rec), indent=1))
        say(f"reference {args.tier} w{mult}: agree {agree} (dJ {dJ:.3g}, image {dist:.3g})")
    if failed:
        raise SystemExit(f"the reference methods disagree at weight multiplier(s) {failed}: the stage has failed and nothing is selected (plan section 4.1)")


def cmd_score(args):
    import virgil_bridge.alma_certify as ac

    d = pathlib.Path(args.out) / args.run
    meta = json.loads((d / "meta.json").read_text())
    args.tier, args.weight_mult = meta["tier"], meta["weight_mult"]
    obj, _, _ = load(args)
    ev = ac.Evaluator(obj, beam=beam_of(obj.prob.npix, obj.prob.pixel_mas, args.beam_mas))
    ref = pathlib.Path(args.out) / f"{reference_stem(meta['tier'], meta['weight_mult'])}.npz"
    if ref.exists():  # the reference of this tier and weight only: a reference of another weight is another objective
        z = np.load(ref)
        if abs(float(z["w"]) / obj.weight - 1.0) > 1e-9:
            raise SystemExit(f"{ref.name} was made at w = {float(z['w']):g}, this run has w = {obj.weight:g}: refusing to score against it")
        ev.set_reference(z["f"], z["theta"] if z["theta"].size else None)
    recs = ac.score_snapshots(ev, str(d), log=say)
    series = ac.certify_series(recs)
    result = json.loads((d / "result.json").read_text()) if (d / "result.json").exists() else {}
    failed = bool(result.get("failed", False))  # a step cap is a failed run
    row = dict(run=args.run, certified=bool(series["certified"] and not failed), failed=failed, series=series, final=recs[-1] if recs else None, n_scored=len(recs),
               image_vs_reference=recs[-1].get("image_distance") if recs else None)
    (d / "scores.json").write_text(json.dumps(jsonable(dict(row, records=recs)), indent=1))
    say(f"{args.run}: certified {row['certified']} (t {series['t']}, {series['geq']} grad-eq)")


def matches_reference(image_vs_reference, final):
    """The answer matches the reference: image within 1e-3 of peak and ``|J - J*|`` below the decrement threshold. A run with no reference does not match (so it cannot win G2)."""
    if image_vs_reference is None or not final or final.get("J_minus_Jstar") is None:
        return False
    return bool(image_vs_reference < 1e-3 and abs(final["J_minus_Jstar"]) < final.get("decrement_threshold", 0.0))


def g2_select(rows, tol=0.2):
    """Gate G2 on the tier-B rows (dicts with ``method``, ``weight_mult``, ``certified``, ``t``, ``matches_reference``): the winner has the lowest median time to certification over the weights
    among methods certified at both with an answer matching the reference; within ``tol`` of the best prefer a method already in virgil, then the one with less new code. ``None`` if nobody."""
    by = {}
    for r in rows:
        by.setdefault(r["method"], []).append(r)
    cand = {}
    for m, rs in by.items():
        best = {}  # one eligible row per weight: the earliest certified one that matches the reference (a slower duplicate or an uncertified row cannot enter the median)
        for r in rs:
            if r["certified"] and r.get("matches_reference", False) and r["t"] is not None and (r["weight_mult"] not in best or r["t"] < best[r["weight_mult"]]):
                best[r["weight_mult"]] = r["t"]
        if set(best) >= set(WEIGHT_MULTS):
            cand[m] = statistics.median(best.values())
    if not cand:
        return None
    best = min(cand.values())
    close = [m for m, t in cand.items() if t <= best * (1 + tol)]
    close.sort(key=lambda m: (m not in IN_VIRGIL, LESS_CODE.index(m) if m in LESS_CODE else len(LESS_CODE), cand[m]))
    ranked = sorted(cand, key=cand.get)
    runner = next((m for m in ranked if m != close[0]), None)
    return dict(winner=close[0], runner_up=runner, median_t=cand)


def cmd_summary(args):
    out = pathlib.Path(args.out)
    lines, tier_b = ["| run | certified | t to cert (s) | grad-eq | final J-J* | image vs ref | steps | reason | peak GB |", "|---|---|---|---|---|---|---|---|---|"], []
    for d in sorted(p for p in out.iterdir() if p.is_dir() and (p / "scores.json").exists()):
        s, r = json.loads((d / "scores.json").read_text()), json.loads((d / "result.json").read_text())
        fin = s["final"] or {}
        ok_ref = matches_reference(s["image_vs_reference"], fin)
        lines.append(f"| {d.name} | {s['certified']} | {s['series']['t']} | {s['series']['geq']} | {fin.get('J_minus_Jstar')} | {s['image_vs_reference']} | {r['steps']} | {r['reason']} | {r['peak_gpu_gb']:.1f} |")
        if r["tier"] == "B":
            tier_b.append(dict(method=r["method"], weight_mult=r["weight_mult"], certified=s["certified"], t=s["series"]["t"], matches_reference=bool(ok_ref)))
    text = "\n".join(lines) + "\n\nG2: " + json.dumps(jsonable(g2_select(tier_b)))
    (out / "summary.md").write_text(text + "\n")
    say(text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("plan", "tune", "reference", "run", "score", "summary"))
    p.add_argument("--data")
    p.add_argument("--out", required=True)
    p.add_argument("--shat", default=None, help="directory of ddid_NN.json: weights rescaled by the model-free s^2, s = 1 after")
    p.add_argument("--nuisance-json", default=None, help="S1's nuisance values (site -> values), held fixed in tier A, the start of tier B")
    p.add_argument("--max-offset-mas", type=float, default=20.0)
    p.add_argument("--w0", type=float, default=W0)
    p.add_argument("--method", choices=METHODS)
    p.add_argument("--methods", default="lbfgsb_newton,newton_cg", help="the reference's two families (reference stage)")
    p.add_argument("--weight-mult", type=int, choices=WEIGHT_MULTS, default=None, help="run/score: the weight multiplier (default 1); reference: both weights unless given")
    p.add_argument("--tier", choices=("A", "B"), default="A")
    p.add_argument("--start", choices=("flat", "gaussian"), default="flat")
    p.add_argument("--budget", type=float, default=BUDGET_S)
    p.add_argument("--tune-budget", type=float, default=TUNE_BUDGET_S)
    p.add_argument("--snap-every", type=float, default=SNAP_S)
    p.add_argument("--max-steps", type=int, default=None, help="step cap: a run that reaches it is failed")
    p.add_argument("--tuned", default=None, help="tuned.json of the tune stage")
    p.add_argument("--opts", default=None, help="JSON overriding the method's options")
    p.add_argument("--beam-mas", type=float, default=35.0, help="the beam of the image comparisons (Gaussian FWHM)")
    p.add_argument("--name", default=None)
    p.add_argument("--run", default=None, help="score: the run directory name")
    args = p.parse_args(argv)
    if args.weight_mult is None and args.stage != "reference":
        args.weight_mult = 1
    if args.stage in ("tune", "reference", "run", "score") and not args.data:
        p.error(f"{args.stage} needs --data")
    if args.stage == "run" and not args.method:
        p.error("run needs --method")
    if args.stage == "score" and not args.run:
        p.error("score needs --run")
    {"plan": cmd_plan, "tune": cmd_tune, "reference": cmd_reference, "run": cmd_run, "score": cmd_score, "summary": cmd_summary}[args.stage](args)


if __name__ == "__main__":
    main()
