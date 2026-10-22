"""IM Lup: the joint MaxEnt fit with virgil.radio, its cross-validation and the comparison with MPoL.

    python scripts/alma_imlup_maxent.py lambda0 --n N [--flux F]
    python scripts/alma_imlup_maxent.py folds --data DIR --out FILE.npz
    python scripts/alma_imlup_maxent.py fit --config {R,F} --lam X --data DIR --out DIR [--sb-only] [--init DIR]
                                            [--shat DIR] [--folds FILE.npz --fold J] [--max-steps N] [--time-limit S]
    python scripts/alma_imlup_maxent.py report --data DIR --out FITDIR [--shat DIR] [--chunk-size N]
    python scripts/alma_imlup_maxent.py cv-task --lam X --fold J --init DIR --folds FILE.npz --data DIR --out DIR
    python scripts/alma_imlup_maxent.py cv-aggregate DIR [--out DIR]
    python scripts/alma_imlup_maxent.py compare --virgil DIR --mpol DIR --data DIR [--f-dir DIR] [--dsharp FITS] [--no-grad] --out DIR
    python scripts/alma_imlup_maxent.py mpol-init --mpol DIR --out DIR [--data DIR]

``--data DIR`` holds ``ddid_NN.npz`` (the vv format of ``alma_imlup_read``; no default, so nothing runs on
data by accident). Every path is an argument. Float64 throughout; the fits use the NUFFT at eps 1e-6 and
every reported -ln L, chi^2 and score eps 1e-12 (C7). The design, the decisions C1-C14 and the output
files are in ``design/alma_imlup_p3.md``; the helpers are in ``virgil_bridge.alma``.
"""

import argparse
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import virgil_bridge.alma as va  # noqa: E402
from crosscheck import alma_maxent as cx  # noqa: E402

RA_DEG, DEC_DEG = (15 + 56 / 60 + 9.20 / 3600) * 15, -(37 + 56 / 60 + 6.42 / 3600)  # the MS's phase centre (docs/alma_conventions.md)
LB_FIRST = 18  # ddids 18-25 are the long-baseline EBs (obs 5, 6)


def say(*a):
    print(*a, flush=True)


def dump(path, obj):
    """JSON, written to a temporary file and renamed, so that a crash never leaves a half-written file."""
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    path = pathlib.Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=default))
    tmp.replace(path)


def versions():
    import importlib.metadata as m

    def sha():
        try:
            return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        except OSError:
            return None

    return dict(python=sys.version.split()[0], jax=jax.__version__, devices=[str(d) for d in jax.devices()], x64=bool(jax.config.jax_enable_x64),
                virgil=m.version("virgil-astro"), numpy=np.__version__, vv=sha())


def load(args):
    ddids = getattr(args, "ddids", None)
    data = va.load_dir(args.data, ddids)
    say(f"data: {len(data.u):,} points, {len(np.unique(np.asarray(data.ddid)))} ddids from {args.data}")
    return data


def lam_of(args, n):
    """``--lam`` or ``--lam-r-equiv`` (C11: lam_R's F-equivalent at ``--flux``)."""
    if args.lam is not None:
        return args.lam
    if args.config == "R":
        return va.lam_r(n)  # MPoL's lambda_M = 1e-9, C4
    return va.lam_f_equivalent(va.lam_r(n) if args.lam_r_equiv is None else args.lam_r_equiv, flux=args.flux)


def stage_summary(name, info, traj, wall, why):
    return dict(name=name, steps=info["steps"], rounds=info.get("rounds"), stop=info["stop"], stopped_because=why, fit_converged=info["converged"],
                wall_s=wall, convergence=va.convergence(traj, info), loss_final=info["loss"], trajectory=traj,
                precondition=info.get("precondition"), precondition_rounds=info.get("precondition_rounds", []), restarts=info.get("restarts", []))


def run_stages(args, data, prob_for, init_dir, save=None):
    """C9: stage 1 on SB, stage 2 on all (warm-started); with ``--init`` a single stage. Each stage runs until C8
    holds or its budget (``--max-steps``, ``--time-limit``) runs out (``va.run_fit_c8``). ``save(prob, values,
    stages)`` is called after every round and stage, so a later failure never loses a fit; ``stages`` then ends with
    the stage in progress (``in_progress`` true)."""
    import jax.numpy as jnp

    shat = va.read_shat(args.shat, int(np.asarray(data.ddid).max()) + 1) if args.shat else None
    stages = []
    mpol = getattr(args, "workflow", "p3") == "mpol"
    if mpol and args.sb_only:
        raise SystemExit("--workflow mpol is one all-data stage from MPoL's own start: no --sb-only")
    if mpol and not init_dir:  # with --init (resume or continue) the stage is the same all-data stage from the saved values
        plan = [("all", None)]
        values = None
    elif init_dir:
        z = np.load(pathlib.Path(init_dir) / "values.npz")
        values = {k: jnp.asarray(z[k]) for k in z.files}
        plan = [("all", None)]
    else:
        plan = [("sb", data.select(obs=va.SB_OBS))] + ([] if args.sb_only else [("all", None)])
        values = None
    prob = None
    for name, use in plan:
        prob = prob_for(use)
        if values is not None:
            values = va.drop_unused_sites(values, prob)
        if values is None:
            values = va.mpol_start_values(prob) if mpol else va.start_values(prob, flux=0.25, fwhm_mas=1000.0, shat=shat)
        say(f"stage {name}: max_steps {args.max_steps}, time limit {args.time_limit}, gtol {args.gtol}")
        t = time.time()

        def on_round(vals, info, traj, rnd, name=name, t=t, prob=prob):
            say(f"stage {name}: round {rnd} done, {info['steps']} steps, stop {info['stop']}, loss {info['loss']:.3f}")
            if save:
                part = stage_summary(name, info, traj, time.time() - t, "in_progress")
                save(prob, {k: np.asarray(v) for k, v in vals.items()}, stages + [dict(part, in_progress=True)])

        out, info, traj, why = va.run_fit_c8(prob, values, max_steps=args.max_steps, time_limit=args.time_limit, log=say, gtol=args.gtol, on_round=on_round, precondition=args.precondition)
        stages.append(stage_summary(name, info, traj, time.time() - t, why))
        say(f"stage {name}: stopped because {why} after {info['steps']} steps in {info['rounds']} rounds; C8 converged: {stages[-1]['convergence']['converged']}")
        values = {k: jnp.asarray(v) for k, v in out.items()}
        if save:
            save(prob, {k: np.asarray(v) for k, v in out.items()}, stages)
    return prob, {k: np.asarray(v) for k, v in values.items()}, stages, shat


def acceptance(sigma, shat):
    """C13: SB ddids |s/ŝ - 1| < 3 %; LB ddids s <= 1.01 ŝ. A finding when it fails, not a knob."""
    r = np.asarray(sigma) / np.asarray(shat)
    sb, lb = r[:LB_FIRST], r[LB_FIRST:]
    return dict(ratio=r, sb_max_abs_dev=float(np.max(np.abs(sb - 1))), lb_max=float(lb.max()) if len(lb) else None,
                sb_ok=bool(np.all(np.abs(sb - 1) < 0.03)), lb_ok=bool(np.all(lb <= 1.01)))


def chi2_table(sc, label):
    """Per-ddid chi^2/N lines for the reports (chi2 sums two squares per point)."""
    n = np.asarray(sc["by_ddid"]["n"])
    ok = n > 0
    red = lambda k: np.where(ok, np.asarray(sc["by_ddid"][k]) / (2 * np.maximum(n, 1)), np.nan)  # noqa: E731
    return dict(label=label, n=n, chi2_raw_over_n=red("chi2_raw"), chi2_fit_over_n=red("chi2_fit"))


# ------------------------------------------------------------------ fit


def write_state(out, args, prob, values, stages, lam, n):
    """What a fit must never lose: ``values.npz``, the image (``.npy`` and ``.fits``) and a provisional
    ``result.json`` (the arguments, the stages with their loss histories, ``complete`` false), written after every round
    and stage; ``write_report`` replaces the result with the full report."""
    np.savez(out / "values.npz", **values)
    image = va.image_of(values, prob.npix, prob.pixel_mas, va.beam_log(prob.npix, prob.pixel_mas) if args.config == "F" else None)
    np.save(out / "image.npy", image.astype(np.float32))
    cx.write_fits(out / "image.fits", image, prob.pixel_mas, RA_DEG, DEC_DEG, OBJECT="IM Lup", ORIGIN=f"virgil {args.config}")
    last = stages[-1]
    dump(out / "result.json", dict(
        command="fit", complete=False, config=args.config, workflow=getattr(args, "workflow", None) or "p3", args={k: v for k, v in vars(args).items() if k != "fn"}, lam=lam, lam_R_at_N=va.lam_r(n),
        npix=prob.npix, pixel_mas=prob.pixel_mas, versions=versions(), stages=stages, converged=last["convergence"]["converged"],
        stop=last["stop"], stopped_because=last["stopped_because"], flux_jy=float(values["cal.flux"])))


def res_workflow(args):
    return getattr(args, "workflow", None) or "p3"


def write_report(args, out, data, prob, values, stages, shat, lam, mask, wall):
    """The full report of a fit at ``values`` (-ln L and chi^2 at eps 1e-12, C13, R's objective, the residual dirty images);
    the model is evaluated ``--chunk-size`` points at a time. Needs no optimiser: ``report`` runs it on a saved fit."""
    n = len(data.u)
    final_data = data.select(obs=va.SB_OBS) if stages[-1]["name"] == "sb" else data  # the last stage fitted SB only
    final_mask = mask if final_data is data else None
    ck = args.chunk_size
    sc_raw = va.score(final_data, prob.scene, values, prob.nuis, mask=final_mask, chunk_size=ck)
    sigma = np.asarray(values.get("cal.sigma", np.ones(26)))
    sc_s1 = va.score(final_data, prob.scene, {**values, "cal.sigma": np.ones_like(sigma)}, prob.nuis, mask=final_mask, chunk_size=ck)  # s = 1 first (C10)
    say(f"workflow {res_workflow(args)}: chi2/(2N) raw (s = 1) {sc_raw['chi2_raw'] / (2 * final_data.u.shape[0]):.4f}; -ln L (eps 1e-12) {sc_raw['nll']:.6f}; chi2 raw {sc_raw['chi2_raw']:.1f} fitted {sc_raw['chi2_fit']:.1f}")
    res = dict(
        command="fit", complete=not any(s.get("in_progress") for s in stages), config=args.config, workflow=getattr(args, "workflow", None) or "p3", args={k: v for k, v in vars(args).items() if k != "fn"}, lam=lam, lam_R_at_N=va.lam_r(n), n=int(final_data.u.shape[0]), npix=prob.npix, pixel_mas=prob.pixel_mas,
        versions=versions(), wall_s=wall, stages=stages, converged=stages[-1]["convergence"]["converged"], stop=stages[-1]["stop"], stopped_because=stages[-1]["stopped_because"],
        flux_jy=float(values["cal.flux"]), gain=values.get("cal.gain"), offset_mas=values.get("cal.offset"), sigma=sigma,
        alpha=va.alpha_report(values if "cal.alpha" in prob.init else {})[0], alpha_label=va.alpha_report(values if "cal.alpha" in prob.init else {})[1],
        offset_bound=va.offset_bound_report(values.get("cal.offset"), prob.offset_bound_mas, prob.offset_bound_requested),
        nll=sc_raw["nll"], chi2_raw=sc_raw["chi2_raw"], chi2_fit=sc_raw["chi2_fit"],
        chi2_s1=chi2_table(sc_s1, "s = 1 (quoted weights), fitted g, offsets, image"), chi2_fitted_s=chi2_table(sc_raw, "fitted s"),
    )
    ob = res["offset_bound"]
    say(f"offset bound ±{ob['bound_mas']:.6g} mas ({'--offset-bound-mas' if ob['requested_mas'] is not None else 'virgil default'}); largest |offset|/bound {ob['max_abs_over_bound']:.3g}")
    if ob["warning"]:
        say(ob["warning"])
    if shat is not None:
        res["shat"] = shat
        res["acceptance_C13"] = acceptance(sigma, shat)
    if args.config == "R":
        import jax.numpy as jnp

        s = float(prob.terms[1].entropy({k: jnp.asarray(v) for k, v in values.items()}))
        res.update(entropy_SM=s, objective_R=sc_raw["nll"] + lam * s)
        image = va.image_of(values, prob.npix, prob.pixel_mas)
        np.save(out / "image_tapered.npy", va.tapered_image(image, prob.pixel_mas).astype(np.float32))
    if args.flux_lam0:
        res["lam0_at_fitted_flux"] = va.lam_f_equivalent(va.lam_r(n), flux=float(values["cal.flux"]), pixel_mas=prob.pixel_mas)
    dump(out / "result.json", res)  # before the dirty images: the numbers are not lost if they fail
    if not args.no_dirty:
        for e, img in va.residual_dirty_images(final_data, prob.scene, values, prob.nuis, chunk_size=ck).items():
            np.save(out / f"resid_dirty_obs{e}.npy", img.astype(np.float32))
    say(f"wrote {out}")


def _train_mask(args, data):
    if not args.folds:
        return None
    fold, _, _ = va.load_folds(args.folds, data)
    mask = fold != args.fold
    say(f"training mask: fold != {args.fold}: {int(mask.sum()):,} of {len(data.u):,} points")
    return mask


def _prob_for(args, data, lam, mask):
    def prob_for(use):
        m = None if mask is None else (mask if use is None else mask[np.isin(np.asarray(data.obs), va.SB_OBS)])
        return va.build_problem(data, args.config, lam=lam, npix=args.npix, pixel_mas=args.pixel_mas, mask=m, use=use, chunk_size=args.chunk_size, alpha_free=getattr(args, "alpha_free", None), offset_bound_mas=getattr(args, "offset_bound_mas", None))

    return prob_for


def resolve_alpha(args):
    """Record the resolved alpha setting in the saved args (the CLI default is None, which would hide a changed default)."""
    if getattr(args, "alpha_free", None) is None:
        args.alpha_free = bool(va.FULL["alpha_free"])


def cmd_fit(args):
    resolve_alpha(args)
    if args.workflow == "mpol" and args.config != "R":
        raise SystemExit("--workflow mpol is defined for config R only (MPoL has no beam, no free alpha)")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load(args)
    n = len(data.u)
    lam = lam_of(args, n)
    mask = _train_mask(args, data)
    say(f"config {args.config}, alpha {'free' if getattr(args, 'alpha_free', None) else va.ALPHA_FIXED_LABEL}, lambda {lam:.6g} (lam_R at N = {n:,} is {va.lam_r(n):.6g})")
    t0 = time.time()
    save = lambda prob, values, stages: write_state(out, args, prob, values, stages, lam, n)  # noqa: E731
    prob, values, stages, shat = run_stages(args, data, _prob_for(args, data, lam, mask), args.init, save)
    write_report(args, out, data, prob, values, stages, shat, lam, mask, time.time() - t0)


def cmd_report(args):
    """Recompute the report of a saved fit (``values.npz`` and ``result.json`` in ``--out``) without refitting: for a
    fit whose reporting failed, or that stopped at its budget, or at a different ``--chunk-size``."""
    out = pathlib.Path(args.out)
    saved = json.loads((out / "result.json").read_text())
    for k, v in saved["args"].items():  # the fit's own arguments, except what this command sets
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    z = np.load(out / "values.npz")
    if getattr(args, "alpha_free", None) is None and args.config == "F":  # a fit made before --alpha-free existed: its values say
        args.alpha_free = "cal.alpha" in z.files
    values = {k: np.asarray(z[k]) for k in z.files}
    data = load(args)
    lam = saved["lam"]
    mask = _train_mask(args, data)
    stages = saved["stages"]
    use = data.select(obs=va.SB_OBS) if stages[-1]["name"] == "sb" else None
    prob = _prob_for(args, data, lam, mask)(use)
    values = va.drop_unused_sites(values, prob)
    shat = va.read_shat(args.shat, int(np.asarray(data.ddid).max()) + 1) if args.shat else None
    say(f"report of {out}: config {args.config}, lambda {lam:.6g}, stages {[s['name'] for s in stages]}")
    write_report(args, out, data, prob, values, stages, shat, lam, mask, saved.get("wall_s"))
    if args.grad and (mask is not None or use is not None):
        say("--grad skipped: the diagnostic evaluates the all-data, unmasked objective, not this fit's (fold mask or SB-only last stage)")
    elif args.grad and args.config == "R":
        groups = {"report": va.grad_groups(data, values, lam=lam, pixel_mas=args.pixel_mas or 5.0, chunk_size=args.chunk_size, offset_bound_mas=getattr(args, "offset_bound_mas", None))}
        dump(out / "grad_groups.json", groups)
        (out / "grad_groups.md").write_text("\n".join(grad_table(groups)) + "\n")
        say("\n".join(grad_table(groups)))


# ------------------------------------------------------------------ folds, cv


def cmd_folds(args):
    data = load(args)
    fold, annulus = va.make_folds(data, args.k, args.n_q, args.n_phi, args.seed)
    params = dict(k=args.k, n_q=args.n_q, n_phi=args.n_phi, seed=args.seed)
    meta = va.save_folds(args.out, data, fold, annulus, params)
    dump(pathlib.Path(args.out).with_suffix(".json"), meta)
    say(json.dumps(meta))


def cmd_cv_task(args):
    resolve_alpha(args)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load(args)
    fold, annulus, fmeta = va.load_folds(args.folds, data)
    train, test = fold != args.fold, fold == args.fold
    assert not np.any(train & test) and np.all(train | test)
    n_all = len(data.u)
    lam = lam_of(args, n_all)  # the full-data strength (the grid key)
    lam_train = lam * int(train.sum()) / n_all  # the likelihood has (k-1)/k of the points: keep the data-to-penalty balance of the final fit
    prob = va.build_problem(data, args.config, lam=lam_train, npix=args.npix, pixel_mas=args.pixel_mas, mask=train, chunk_size=args.chunk_size, alpha_free=getattr(args, "alpha_free", None), offset_bound_mas=getattr(args, "offset_bound_mas", None))
    z = np.load(pathlib.Path(args.init) / "values.npz")
    import jax.numpy as jnp

    t = time.time()
    values, info, traj, why = va.run_fit_c8(prob, va.drop_unused_sites({k: jnp.asarray(z[k]) for k in z.files}, prob), max_steps=args.max_steps, time_limit=args.time_limit, log=say, gtol=args.gtol, precondition=args.precondition)
    wall = time.time() - t
    ho = va.score(data, prob.scene, values, prob.nuis, mask=test, annulus=annulus, chunk_size=args.chunk_size)
    ho1 = va.score(data, prob.scene, {**values, "cal.sigma": np.ones_like(values["cal.sigma"])}, prob.nuis, mask=test, annulus=annulus, chunk_size=args.chunk_size)
    tr = va.score(data, prob.scene, values, prob.nuis, mask=train, chunk_size=args.chunk_size)
    res = dict(command="cv-task", config=args.config, lam=lam, lam_train=lam_train, fold=args.fold, n_train=int(train.sum()), n_test=int(test.sum()), folds_sha256=fmeta["sha256"],
               versions=versions(), wall_s=wall, steps=info["steps"], stop=info["stop"], stopped_because=why, convergence=va.convergence(traj, info), trajectory=traj,
               precondition=info.get("precondition"), precondition_rounds=info.get("precondition_rounds", []),
               heldout=dict(nll=ho["nll"], chi2_s1=ho1["chi2_raw"], chi2_fit=ho["chi2_fit"], by_ddid=ho["by_ddid"], by_annulus=ho["by_annulus"],
                            by_ddid_s1=ho1["by_ddid"], by_annulus_s1=ho1["by_annulus"]),
               train=dict(nll=tr["nll"], chi2_fit=tr["chi2_fit"]), sigma=values["cal.sigma"], flux_jy=float(values["cal.flux"]), alpha_label=va.alpha_report(values if "cal.alpha" in prob.init else {})[1])
    dump(out / "cv.json", res)
    say(f"held-out -ln L {ho['nll']:.4f} on {ho['n']:,} points; wrote {out / 'cv.json'}")


def cmd_cv_aggregate(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = pathlib.Path(args.dir)
    recs = [json.loads(p.read_text()) for p in sorted(root.rglob("cv.json"))]
    if not recs:
        raise SystemExit(f"no cv.json under {root}")
    out = pathlib.Path(args.out or root)
    out.mkdir(parents=True, exist_ok=True)
    lams = sorted({r["lam"] for r in recs})
    folds = sorted({r["fold"] for r in recs})
    shas = {r["folds_sha256"] for r in recs}
    S = np.full((len(lams), len(folds)), np.nan)
    A = np.full((len(lams), len(folds), len(recs[0]["heldout"]["by_annulus"]["nll"])), np.nan)
    for r in recs:
        i, j = lams.index(r["lam"]), folds.index(r["fold"])
        S[i, j] = r["heldout"]["nll"]
        A[i, j] = r["heldout"]["by_annulus"]["nll"]
    full = ~np.isnan(S).any(axis=1)  # a lambda counts only with every fold done
    total, mean, sd = np.nansum(S, 1), np.nanmean(S, 1), np.nanstd(S, 1, ddof=1) if len(folds) > 1 else np.zeros(len(lams))
    cand = np.flatnonzero(full)
    best = int(cand[np.argmin(total[cand])])
    # one-SE rule on the paired per-fold differences from the best lambda: the largest (most regularised) lambda
    # whose mean difference is within one standard error of 0
    d = S - S[best]
    se = np.nanstd(d, 1, ddof=1) / np.sqrt(len(folds)) if len(folds) > 1 else np.zeros(len(lams))
    within = [i for i in cand if i == best or np.nanmean(d[i]) <= se[i]]
    one_se = int(max(within, key=lambda i: lams[i]))
    res = dict(lams=lams, folds=folds, n_records=len(recs), folds_sha256=sorted(shas), complete=full.tolist(), heldout_nll=S, sum=total, mean=mean, sd=sd,
               lam_star=lams[best], lam_one_se=lams[one_se], at_edge=bool(best in (0, len(lams) - 1)),
               edge="low" if best == 0 else ("high" if best == len(lams) - 1 else None),
               annulus_sum=np.nansum(A, 1), annulus_best_lam=[lams[int(np.nanargmin(np.nansum(A, 1)[:, a]))] for a in range(A.shape[2])],
               not_converged=[(r["lam"], r["fold"], r["stop"]) for r in recs if not r["convergence"]["converged"]])
    dump(out / "cv_summary.json", res)
    fig, ax = plt.subplots(figsize=(6, 4))
    for j in range(len(folds)):
        ax.plot(lams, S[:, j] - np.nanmin(S[:, j]), color="0.75", lw=0.8)
    ax.errorbar(lams, mean - np.nanmin(mean), sd, color="C0", marker="o", label="mean ± sd over folds")
    ax.axvline(lams[best], color="C3", ls="--", label=f"λ* = {lams[best]:.3g}")
    ax.set_xscale("log")
    ax.set_xlabel("λ")
    ax.set_ylabel("held-out −ln L (minus its minimum)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "cv_curve.png", dpi=130)
    fig, ax = plt.subplots(figsize=(6, 4))
    for a in range(A.shape[2]):
        col = np.nansum(A, 1)[:, a]
        ax.plot(lams, col - col.min(), label=f"q annulus {a}")
    ax.set_xscale("log")
    ax.set_xlabel("λ")
    ax.set_ylabel("held-out −ln L per annulus (minus its minimum)")
    ax.legend(fontsize=5, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "cv_annulus.png", dpi=130)
    md = ["# Cross-validation", "", f"{len(recs)} tasks, folds checksum {sorted(shas)}", "", "| λ | sum of held-out −ln L | mean | sd | complete |", "|---|---|---|---|---|"]
    md += [f"| {l:.6g} | {total[i]:.4f} | {mean[i]:.4f} | {sd[i]:.4f} | {bool(full[i])} |" for i, l in enumerate(lams)]
    md += ["", f"λ* = {lams[best]:.6g}; one-standard-error λ = {lams[one_se]:.6g}; at grid edge: {res['at_edge']} ({res['edge']})."]
    if res["at_edge"]:
        md.append("The minimum is at an edge of the grid: extend it by two half-decades and rerun those tasks (C12).")
    if res["not_converged"]:
        md.append(f"Tasks not converged by C8: {res['not_converged']}")
    (out / "cv.md").write_text("\n".join(md) + "\n")
    say("\n".join(md))


# ------------------------------------------------------------------ compare


def objective_tables(res, vres, obj_m):
    """Markdown for ``compare``: the objective term by term, raw chi^2/N per ddid at both solutions (s = 1), gains and offsets side by
    side with their conventions, and the gradient diagnostic. ``res`` is compare's result (with ``objective``), ``vres`` our fit's
    result.json, ``obj_m`` MPoL's own objective.json (or None)."""
    o = res["objective"]
    a, b = o["ours"], o["mpol_solution_in_virgil"]
    own = (obj_m or {})
    md = ["", "## The objective, term by term (lam_R = %.6g, eps 1e-12)" % o["lam_R"], "",
          "| | -ln L | S_M | lam_R S_M | objective_R |", "|---|---:|---:|---:|---:|",
          f"| ours (our fit's values, virgil) | {a['nll']:.1f} | {a['entropy']:.2f} | {a['penalty']:.1f} | {a['objective']:.1f} |",
          f"| MPoL's solution, evaluated in virgil | {b['nll']:.1f} | {b['entropy']:.2f} | {b['penalty']:.1f} | {b['objective']:.1f} |"]
    if own:
        md.append(f"| MPoL's own numbers (objective.json) | {own.get('neg_log_likelihood', float('nan')):.1f} | {own.get('entropy', float('nan')):.2f} | "
                  f"{o['lam_R'] * own.get('entropy', float('nan')):.1f} | {own.get('objective_R', float('nan')):.1f} |")
        md += ["", f"**virgil minus MPoL, both at MPoL's solution** (zero if our objective reproduces MPoL's): -ln L {b['nll'] - own['neg_log_likelihood']:+.3f}, "
                   f"lam_R S_M {b['penalty'] - o['lam_R'] * own['entropy']:+.3f}, objective {b['objective'] - own['objective_R']:+.3f}."]
    if vres.get("objective_R") is not None:
        md.append(f"Our fit's own result.json: objective_R {vres['objective_R']:.1f}, -ln L {vres['nll']:.1f}.")
    md += [f"**Our objective minus MPoL's solution's, both in virgil: {a['objective'] - b['objective']:+.1f}** (negative: ours is lower, i.e. better)."]
    na, nb = np.asarray(a["by_ddid"]["n"]), np.asarray(b["by_ddid"]["n"])
    ra, rb = (np.asarray(x["by_ddid"]["chi2_raw"]) / (2 * np.maximum(n, 1)) for x, n in ((a, na), (b, nb)))
    rm = np.asarray(own["chi2_raw"], float) / (2 * np.maximum(np.asarray(own["n_ddid"]), 1)) if own.get("chi2_raw") else None
    md += ["", "## Raw chi^2/N per ddid (s = 1, quoted weights; chi^2 / 2N)", "",
           "| ddid | N | ours | MPoL's solution (virgil) | " + ("MPoL's own | " if rm is not None else "") + "ours - MPoL |", "|---:|---:|---:|---:|" + ("---:|" if rm is not None else "") + "---:|"]
    for d in range(len(na)):
        if na[d]:
            md.append(f"| {d} | {na[d]:,} | {ra[d]:.4f} | {rb[d]:.4f} | " + (f"{rm[d]:.4f} | " if rm is not None else "") + f"{ra[d] - rb[d]:+.4f} |")
    md.append(f"| all | {na.sum():,} | {np.sum(a['by_ddid']['chi2_raw']) / (2 * na.sum()):.4f} | {np.sum(b['by_ddid']['chi2_raw']) / (2 * nb.sum()):.4f} | "
              + (f"{np.sum(own['chi2_raw']) / (2 * np.sum(own['n_ddid'])):.4f} | " if rm is not None else "") + f"{(np.sum(a['by_ddid']['chi2_raw']) - np.sum(b['by_ddid']['chi2_raw'])) / (2 * na.sum()):+.4f} |")
    g, off = res["gain"], res["offset_mas"]
    md += ["", "## Gains and offsets, side by side", "",
           "Conventions (docs/alma_conventions.md, Part 3): both multiply the model of observation k by the gain g_k (observation 0 fixed at 1) and rescale its weights by g^2, "
           "so the gains are directly comparable (not reciprocal); virgil's per-ddid scale is s = g s_MPoL. Offsets: MPoL's delta_RA, delta_Dec in arcsec "
           "(phase exp(-2 pi i (dRA u + dDec v))); ours `cal.offset` in mas, the same sign, so mpol_mas = 1000 x arcsec. Obs 0 is fixed at 0 in both.", "",
           "| obs | g ours | g MPoL | dRA ours (mas) | dRA MPoL (mas) | dDec ours (mas) | dDec MPoL (mas) |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for k in range(len(g["mpol"])):
        md.append(f"| {k} | {g['ours'][k]:.4f} | {g['mpol'][k]:.4f} | {off['ours'][k][0]:+.3f} | {off['mpol'][k][0]:+.3f} | {off['ours'][k][1]:+.3f} | {off['mpol'][k][1]:+.3f} |")
    if "grad_groups" in res:
        md += ["", *grad_table(res["grad_groups"])]
    return md


def grad_table(groups):
    """Markdown of ``va.grad_groups`` per ``at`` label: dx/dz and the gradient in physical units (offset per mas, gain per unit gain,
    sigma per ln s, flux per Jy, pixels per unit log-brightness), and the same gradient relative to the pixels'."""
    md = ["## Gradient scaling by parameter group (the loss virgil.fit minimises, eps 1e-6)", "",
          "`g_z`: gradient in the unconstrained coordinates L-BFGS moves in; `dx/dz`: median of the map from z to the parameter x (z is a logit of x/bound for "
          "`cal.offset`); `g_x = g_z / (dx/dz)`: gradient in physical units, `|dL/dx|` per the unit in the last column "
          "(sigma: per ln s, max |s_i g_x,i| over ddids). Raw loss units (-ln L + lam_R S_M; fit's gtol is per data point). A group with small `dx/dz` and large "
          "`g_x`/`g_z` ratio is badly conditioned for a first-order step: z must move 1/(dx/dz) times further than x.", "",
          "| at | site | n | max abs g_z | norm g_z | dx/dz | max abs g_x | unit |", "|---|---|---:|---:|---:|---:|---:|---|"]
    units = {"cal.offset": "per mas", "cal.gain": "per unit gain", "cal.sigma": "per ln s", "cal.flux": "per Jy", "log_brightness": "per unit ln brightness"}
    for label, rows in groups.items():
        if "error" in rows:
            md.append(f"| {label} | error: {rows['error']} | | | | | | |")
            continue
        for site, r in rows.items():
            gx = r["gx_lns_max"] if site == "cal.sigma" else r["gx_max"]
            md.append(f"| {label} | {site} | {r['n']} | {r['gz_max']:.4g} | {r['gz_norm']:.4g} | {r['dxdz_median']:.4g} | {gx:.4g} | {units.get(site, '')} |")
    return md


def cmd_compare(args):
    import jax.numpy as jnp

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vdir, mdir = pathlib.Path(args.virgil), pathlib.Path(args.mpol)
    vres = json.loads((vdir / "result.json").read_text())
    vz = np.load(vdir / "values.npz")
    mz = np.load(mdir / "export.npz")
    mj = json.loads((mdir / "export.json").read_text()) if (mdir / "export.json").exists() else {}
    obj_m = json.loads((mdir / "objective.json").read_text()) if (mdir / "objective.json").exists() else None
    px = vres["pixel_mas"]
    res = dict(command="compare", virgil=str(vdir), mpol=str(mdir), versions=versions())
    amp = np.asarray(mz["amp"], float)
    sig_m = np.asarray(mz["sigma"], float)
    data = va.load_dir(args.data) if args.data else None
    obs_of = np.zeros(len(sig_m), int)
    if data is not None:
        dd, ob = np.asarray(data.ddid), np.asarray(data.obs)
        obs_of = np.array([ob[dd == d][0] for d in range(len(sig_m))])
    else:
        obs_of = np.array([0] * 5 + [1] * 6 + [2] * 2 + [3] * 2 + [4] * 3 + [5] * 4 + [6] * 4)  # IM Lup's obsid_dict
    s_ours = np.asarray(vz["cal.sigma"], float)
    s_map = s_ours / (amp[obs_of] * sig_m)  # s_ours = g s_MPoL (docs/alma_conventions.md, Part 3)
    res["sigma"] = dict(s_ours=s_ours, g_times_s_mpol=amp[obs_of] * sig_m, ratio=s_map, max_abs_dev=float(np.max(np.abs(s_map - 1))))
    res["gain"] = dict(ours=np.concatenate([[1.0], vz["cal.gain"]]), mpol=amp, max_abs_diff=float(np.max(np.abs(np.concatenate([[1.0], vz["cal.gain"]]) - amp))))
    off_m = 1000.0 * np.stack([mz["delta_ra_arcsec"], mz["delta_dec_arcsec"]], -1)
    off_o = np.concatenate([np.zeros((1, 2)), vz["cal.offset"]])
    res["offset_mas"] = dict(ours=off_o, mpol=off_m, max_abs_diff=float(np.max(np.abs(off_o - off_m))))
    dx = float(mz["cell_size_arcsec"])
    flux_m = float(np.sum(mz["base_image"]) * dx**2)
    res["flux_jy"] = dict(ours=float(vz["cal.flux"]), mpol=flux_m, diff=float(vz["cal.flux"]) - flux_m)
    img_o = np.load(vdir / "image.npy").astype(float)
    base_m, tap_m = np.asarray(mz["base_image"], float), np.asarray(mz["image"], float)
    assert img_o.shape == base_m.shape, "R and MPoL's grid must be the same 1028^2 (map D11)"
    tap_o = va.tapered_image(img_o, px)
    res["image"] = dict(untapered=cx.difference_stats(img_o, base_m), tapered_30mas=cx.difference_stats(tap_o, tap_m))
    beam = None
    if args.dsharp:
        _, wcs_d, hdr_d = cx.read_fits(args.dsharp)
        beam = cx.beam_of(hdr_d)
        area = cx.beam_area_arcsec2(beam[0], beam[1])
        co, cm = cx.convolve(img_o, px, *beam) * area, cx.convolve(base_m, px, *beam) * area
        res["image"]["clean_beam"] = dict(beam_mas_deg=beam, **cx.difference_stats(co, cm))
    if data is not None:
        lam = va.lam_r(len(data.u))
        v_ours = {k: jnp.asarray(vz[k]) for k in vz.files}
        v_mpol = va.values_from_mpol_export(mz, obs_of)
        o_ours = va.r_objective(data, v_ours, lam=lam, pixel_mas=px, chunk_size=args.chunk_size)
        o_mpol = va.r_objective(data, v_mpol, lam=lam, pixel_mas=px, chunk_size=args.chunk_size)
        res["objective"] = dict(lam_R=lam, ours=o_ours, mpol_solution_in_virgil=o_mpol, mpol_own=obj_m and {k: obj_m.get(k) for k in ("neg_log_likelihood", "entropy", "objective_R")},
                                ours_le_mpol=bool(o_ours["objective"] <= o_mpol["objective"]),
                                finding=None if o_ours["objective"] <= o_mpol["objective"] else "our optimiser ended above MPoL's solution: a finding about the fit, not tuned away")
        if not args.no_grad:
            res["grad_groups"] = {}
            for label, v in (("ours", v_ours), ("mpol_solution_in_virgil", v_mpol)):
                try:
                    res["grad_groups"][label] = va.grad_groups(data, v, lam=lam, pixel_mas=px, chunk_size=args.chunk_size)
                except Exception as e:  # a diagnostic: never loses the comparison
                    res["grad_groups"][label] = dict(error=f"{type(e).__name__}: {e}")
    if args.f_dir and args.dsharp:
        fz = np.load(pathlib.Path(args.f_dir) / "image.npy").astype(float)
        n = fz.shape[0]
        conv = cx.convolve(fz, px, *beam) * cx.beam_area_arcsec2(beam[0], beam[1])  # Jy/beam
        data_d, wcs_d, _ = cx.read_fits(args.dsharp)
        d_on = cx.resample(data_d, wcs_d, cx.sky_wcs(n, px, RA_DEG, DEC_DEG), (n, n))[::-1]  # back to the repository's layout
        np.save(out / "f_clean_beam.npy", conv.astype(np.float32))
        np.save(out / "dsharp_regridded.npy", d_on.astype(np.float32))
        res["F_vs_dsharp"] = dict(beam_mas_deg=beam, **cx.difference_stats(conv, d_on))
    dump(out / "compare.json", res)
    md = ["# R against MPoL", ""]
    if "objective" in res:
        n_tot = res["objective"]["ours"]["n"]
        raw = lambda o: sum(o["by_ddid"]["chi2_raw"]) / (2 * n_tot)  # noqa: E731
        md += [f"**Raw chi^2/N (s = 1, quoted weights): ours {raw(res['objective']['ours']):.4f}; MPoL's solution evaluated in virgil {raw(res['objective']['mpol_solution_in_virgil']):.4f}** (N = {n_tot:,}, eps 1e-12).", ""]
    md += [f"s_ours / (g s_MPoL): max |ratio − 1| = {res['sigma']['max_abs_dev']:.4g}",
          f"gain: max |diff| = {res['gain']['max_abs_diff']:.4g}; offsets: max |diff| = {res['offset_mas']['max_abs_diff']:.4g} mas; flux {res['flux_jy']['ours']:.5f} vs {flux_m:.5f} Jy", ""]
    for k, v in res["image"].items():
        md.append(f"- image {k}: " + ", ".join(f"{a} {b:.4g}" for a, b in v.items() if isinstance(b, float)))
    if "objective" in res:
        o = res["objective"]
        md += ["", f"objective (λ_R = {o['lam_R']:.6g}): ours {o['ours']['objective']:.6f}, MPoL's solution (virgil) {o['mpol_solution_in_virgil']['objective']:.6f}; ours ≤ MPoL's: {o['ours_le_mpol']}"]
        if o["mpol_own"]:
            md.append(f"MPoL's own objective_full: {o['mpol_own']}")
        md += objective_tables(res, vres, obj_m)
    if "F_vs_dsharp" in res:
        md += ["", "# F against DSHARP (CLEAN beam)", ", ".join(f"{a} {b:.4g}" for a, b in res["F_vs_dsharp"].items() if isinstance(b, float))]
    (out / "compare.md").write_text("\n".join(md) + "\n")
    say("\n".join(md))


def cmd_mpol_init(args):
    """A fit's values.npz (the ``--init`` of ``fit``) from MPoL's export: image, flux, gains, offsets and s = g s_M (``values_from_mpol_export``).
    The ddid -> observation map comes from ``--data`` if given, else IM Lup's ``obsid_dict``. Needs no data otherwise and no GPU."""
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mz = np.load(pathlib.Path(args.mpol) / "export.npz")
    n_ddid = len(mz["sigma"])
    if args.data:
        data = va.load_dir(args.data)
        dd, ob = np.asarray(data.ddid), np.asarray(data.obs)
        obs_of = np.array([ob[dd == d][0] for d in range(n_ddid)])
    else:
        obs_of = np.array([0] * 5 + [1] * 6 + [2] * 2 + [3] * 2 + [4] * 3 + [5] * 4 + [6] * 4)
    values = {k: np.asarray(v) for k, v in va.values_from_mpol_export(mz, obs_of).items()}
    np.savez(out / "values.npz", **values)
    dump(out / "mpol_init.json", dict(command="mpol-init", mpol=str(args.mpol), n_ddid=n_ddid, obs_of_ddid=obs_of, keys=sorted(values),
                                      flux_jy=float(values["cal.flux"]), gain=values["cal.gain"], offset_mas=values["cal.offset"], sigma=values["cal.sigma"]))
    say(f"wrote {out / 'values.npz'}: {sorted(values)}, image {values['log_brightness'].shape}, flux {float(values['cal.flux']):.5f} Jy")


def cmd_lambda0(args):
    r = va.lam_r(args.n)
    say(json.dumps(dict(n=args.n, lam_M=va.MPOL["lam_m"], lam_R=r, flux=args.flux, lam_F_equivalent=va.lam_f_equivalent(r, flux=args.flux))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, fit=True):
        p.add_argument("--data", required=True, help="directory of ddid_NN.npz (no default)")
        p.add_argument("--ddids", type=int, nargs="*", default=None)
        p.add_argument("--out", required=True)
        if fit:
            p.add_argument("--lam", type=float, default=None, help="strength in -ln L units (R: lam_R; F: virgil MaxEntropy weight)")
            p.add_argument("--lam-r-equiv", type=float, default=None, help="instead of --lam: F's equivalent of this lam_R (C11), at --flux")
            p.add_argument("--flux", type=float, default=va.MPOL["f_tot"], help="flux (Jy) of the conversion lam_R -> lam_F")
            p.add_argument("--npix", type=int, default=None)
            p.add_argument("--pixel-mas", type=float, default=5.0)
            p.add_argument("--max-steps", type=int, default=20000, help="step budget per stage (the stage stops earlier when C8 holds)")
            p.add_argument("--time-limit", type=float, default=18000.0, help="seconds per stage (default 5 h)")
            p.add_argument("--gtol", type=float, default=1e-4, help="fit's gradient test; restarts tighten it until C8 holds")
            p.add_argument("--precondition", choices=["none", "diag"], default="none", help="virgil.fit's diagonal Gauss-Newton scaling of L-BFGS (needs a virgil that has it); "
                           "the stage summary records its per-round stats, each restart's steps and wall time")
            p.add_argument("--chunk-size", type=int, default=va.CHUNK, help="points per model evaluation (the fit's term and every score)")
            p.add_argument("--offset-bound-mas", type=float, default=None, help="R, F: bound ±X (mas) of the offsets' uniform prior (default: virgil's own ±1/(2 q_min), ~1e4 mas for IM Lup); "
                           "a flat prior is unchanged inside any bound, but the optimiser's logit coordinate is far better conditioned at, say, 100; `report` warns if an offset ends within 5% of it")
            p.add_argument("--alpha-free", dest="alpha_free", action="store_true", default=None, help="F: fit the spectral index within its bounds (default: fixed 0, as MPoL)")

    p = sub.add_parser("fit")
    common(p)
    p.add_argument("--config", choices=["R", "F"], required=True)
    p.add_argument("--sb-only", action="store_true")
    p.add_argument("--workflow", choices=["p3", "mpol"], default="p3",
                   help="p3 (default): C9's SB stage then all data from a 1\" Gaussian, s at ŝ. mpol (config R only): one all-data stage from MPoL's own start "
                        "(uniform softplus(-3) image, g = 1, offsets 0, s = 1; sgd.py:49-59, images.py BaseCube)")
    p.add_argument("--init", default=None, help="a previous fit's directory (values.npz): one stage on all data")
    p.add_argument("--shat", default=None, help="directory of ddid_NN.json (alma_imlup_read): s_d starts at ŝ_d; C13 is checked")
    p.add_argument("--folds", default=None)
    p.add_argument("--fold", type=int, default=None, help="train on fold != J (needs --folds)")
    p.add_argument("--no-dirty", action="store_true")
    p.add_argument("--flux-lam0", action="store_true", help="also record lam_F's equivalent at the fitted flux")
    p.set_defaults(fn=cmd_fit)

    p = sub.add_parser("report", help="recompute a saved fit's report (result.json, dirty images) without refitting")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True, help="the fit's directory (values.npz, result.json)")
    p.add_argument("--shat", default=None)
    p.add_argument("--no-dirty", action="store_true", default=None)
    p.add_argument("--grad", action="store_true", help="R: also print the per-group gradient scaling (offset-conditioning diagnostic; one value-and-gradient)")
    p.add_argument("--chunk-size", type=int, default=va.CHUNK)
    p.set_defaults(fn=cmd_report, ddids=None, folds=None, fold=None, flux_lam0=None)

    p = sub.add_parser("folds")
    p.add_argument("--data", required=True)
    p.add_argument("--ddids", type=int, nargs="*", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n-q", type=int, default=12)
    p.add_argument("--n-phi", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_folds)

    p = sub.add_parser("cv-task")
    common(p)
    p.add_argument("--config", choices=["R", "F"], default="F")
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--init", required=True)
    p.add_argument("--folds", required=True)
    p.set_defaults(fn=cmd_cv_task, max_steps=1500, time_limit=3000.0)

    p = sub.add_parser("cv-aggregate")
    p.add_argument("dir")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_cv_aggregate)

    p = sub.add_parser("compare")
    p.add_argument("--virgil", required=True)
    p.add_argument("--mpol", required=True)
    p.add_argument("--data", default=None, help="needed for the objective comparison")
    p.add_argument("--f-dir", default=None)
    p.add_argument("--dsharp", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--no-grad", action="store_true", help="skip the gradient-scaling diagnostic (two value-and-gradient evaluations on --data)")
    p.add_argument("--chunk-size", type=int, default=va.CHUNK)
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("mpol-init", help="values.npz from MPoL's export, the --init of `fit --config R` (a warm start at MPoL's solution)")
    p.add_argument("--mpol", required=True, help="MPoL's export directory (export.npz)")
    p.add_argument("--data", default=None, help="optional: the ddid -> observation map is read from it (default IM Lup's obsid_dict)")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_mpol_init)

    p = sub.add_parser("lambda0")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--flux", type=float, default=va.MPOL["f_tot"])
    p.set_defaults(fn=cmd_lambda0)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
