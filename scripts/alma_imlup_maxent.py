"""IM Lup: the joint MaxEnt fit with virgil.radio, its cross-validation and the comparison with MPoL.

    python scripts/alma_imlup_maxent.py lambda0 --n N [--flux F]
    python scripts/alma_imlup_maxent.py folds --data DIR --out FILE.npz
    python scripts/alma_imlup_maxent.py fit --config {R,F} --lam X --data DIR --out DIR [--chi2 {direct,toeplitz}] [--sb-only] [--init DIR] [--workflow {p3,mpol,clean} [--clean-dir DIR]]
                                            [--shat DIR] [--folds FILE.npz --fold J] [--max-steps N] [--time-limit S]
                                            [--method {lbfgs,adam} [--adam-lr LR] [--adam-lr-decay F] [--adam-steps N] [--adam-eps E]]
    python scripts/alma_imlup_maxent.py report --data DIR --out FITDIR [--shat DIR] [--chunk-size N]
    python scripts/alma_imlup_maxent.py cv-task --lam X --fold J --init DIR --folds FILE.npz --data DIR --out DIR
    python scripts/alma_imlup_maxent.py cv-aggregate DIR [--out DIR]
    python scripts/alma_imlup_maxent.py cv-compare --mpol DIR --virgil DIR [--lam-ent 1e-9] --out DIR
    python scripts/alma_imlup_maxent.py compare --virgil DIR --mpol DIR --data DIR [--f-dir DIR] [--dsharp FITS] [--no-grad] [--no-dirty] --out DIR
    python scripts/alma_imlup_maxent.py mpol-init --mpol DIR --out DIR [--data DIR]
    python scripts/alma_imlup_maxent.py speed --fit [LABEL=]DIR [--fit ...] --mpol DIR --out DIR [--log [mpol=|virgil=]FILE ...] [--target LABEL=R ...]

``--data DIR`` holds ``ddid_NN.npz`` (the vv format of ``alma_imlup_read``; no default, so nothing runs on
data by accident). Every path is an argument. Float64 throughout; the fits use the NUFFT at eps 1e-6 and
every reported -ln L, chi^2 and score eps 1e-12 (C7). The design, the decisions C1-C14 and the output
files are in ``design/alma_imlup_p3.md``; the helpers are in ``virgil_bridge.alma``.

``fit --chi2 toeplitz`` (config R) runs every stage's data term through virgil's Toeplitz dirty-beam chi^2 (equal to the direct term to ~1e-13, ~30x faster per
value+gradient on a GPU); the reports and ``compare`` always score with the direct term at eps 1e-12. Fitted offsets need a prior box inside the precompute:
``--offset-bound-mas`` defaults to 20 mas then, and a problem Toeplitz cannot take exactly (baselines past its Nyquist margin) is refused.
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
import virgil_bridge.alma_start as vs  # noqa: E402
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


def stage_dead_pixels(values, prob, config):
    """The dead-pixel fractions (below 1e-6 and 1e-4 of the peak) and the largest live radius of the stage's base image and of the image tapered to
    MPoL's beam (``va.dead_pixel_report`` of each)."""
    image = va.image_of(values, prob.npix, prob.pixel_mas, va.beam_log(prob.npix, prob.pixel_mas) if config == "F" else None)
    return dict(base=va.dead_pixel_report(image, prob.pixel_mas), tapered=va.dead_pixel_report(va.tapered_image(image, prob.pixel_mas), prob.pixel_mas))


def stage_summary(name, info, traj, wall, why):
    """``fit_converged`` is virgil's own ``info['converged']`` for both methods; ``c8_converged`` is vv's C8 verdict (``convergence``: the two-point test for L-BFGS,
    ``va.convergence_window`` for Adam), which the cold-start gate uses."""
    conv = info["adam"]["c8"] if info.get("adam") else va.convergence(traj, info)
    return dict(name=name, method=info.get("method", "lbfgs"), adam=info.get("adam"), post_step_blur_px=(info.get("adam") or {}).get("post_step_blur_px"), steps=info["steps"], rounds=info.get("rounds"), stop=info["stop"], stopped_because=why, fit_converged=info["converged"], c8_converged=conv["converged"],
                wall_s=wall, convergence=conv, loss_final=info["loss"], trajectory=traj,
                precondition=info.get("precondition"), precondition_rounds=info.get("precondition_rounds", []), restarts=info.get("restarts", []))


def run_stages(args, data, prob_for, init_dir, save=None):
    """C9: stage 1 on SB, stage 2 on all (warm-started); with ``--init`` a single stage. Each stage runs until C8
    holds or its budget (``--max-steps``, ``--time-limit``) runs out (``va.run_fit_c8``). ``save(prob, values,
    stages)`` is called after every round and stage, so a later failure never loses a fit; ``stages`` then ends with
    the stage in progress (``in_progress`` true).

    ``--workflow clean`` (no MPoL information): the start is our own CLEAN model (``vs.clean_start``); stage ``selfcal``
    holds that image and fits only the nuisances (flux, gains, offsets, s), stage ``all`` releases everything.
    ``--image-first`` (workflows p3 and mpol): a first stage ``image`` on the first stage's data fits only the image and
    the flux, with gains 1, offsets 0 and s = 1, before the workflow's own stages release the nuisances (the opposite
    of selfcal: the image is made right before the nuisances can wander). ``--prestage-max-steps`` budgets both.
    ``prob_for(use, fixed, image_only)`` builds a stage's problem; ``fixed`` is a held ``log_brightness`` or None."""
    import jax.numpy as jnp

    shat = va.read_shat(args.shat, int(np.asarray(data.ddid).max()) + 1) if args.shat else None
    stages = []
    workflow = getattr(args, "workflow", "p3")
    mpol = workflow == "mpol"
    clean = workflow == "clean" and not init_dir  # with --init (resume or continue) the stage is the same all-data stage from the saved values
    image_first = bool(getattr(args, "image_first", False))
    if (mpol or workflow == "clean") and args.sb_only:
        raise SystemExit(f"--workflow {workflow} is all-data stages from its own start: no --sb-only")
    if image_first and (workflow == "clean" or init_dir):
        raise SystemExit("--image-first is for the p3 and mpol starts (clean has its own image-fixed stage; --init has an image already)")
    held = None  # the log_brightness held in the current stage
    floor_info = None  # info of the Gaussian start's floor (``--start-floor``)
    gfit_info = None  # the fitted Gaussian of ``--start gauss-fit``
    gauss_fit = getattr(args, "start", "gauss") == "gauss-fit"
    if gauss_fit and (workflow in ("mpol", "clean") or init_dir):
        raise SystemExit("--start gauss-fit is the p3 workflow's cold start: --workflow mpol and clean have their own starts, and --init an image already")
    if mpol and not init_dir:
        plan = [("all", None)]
        values = None
    elif init_dir:
        z = np.load(pathlib.Path(init_dir) / "values.npz")
        values = {k: jnp.asarray(z[k]) for k in z.files}
        plan = [("all", None)]
    elif clean:
        values, held, start_info = clean_start_values(args, prob_for(None, None, False), shat)
        plan = [("selfcal", None), ("all", None)]
    else:
        plan = [("sb", data.select(obs=va.SB_OBS))] + ([] if args.sb_only else [("all", None)])
        values = None
    if image_first:
        plan = [("image", plan[0][1])] + plan
    prob, previous = None, None
    for name, use in plan:
        fixed = held if name == "selfcal" else None
        prob = prob_for(use, fixed, name == "image")
        if values is not None:
            values = va.drop_unused_sites(values, prob)
            if previous == "image":  # the nuisances start where a fresh fit starts them
                values = {**prob.init, **values}
                if shat is not None and "cal.sigma" in values:
                    values["cal.sigma"] = jnp.asarray(shat, float)
        if values is None:
            values = va.mpol_start_values(prob) if mpol else va.start_values(prob, flux=0.25, fwhm_mas=1000.0, shat=shat)
            if gauss_fit:  # a data-driven start: the elliptical Gaussian a least-squares fit to the short baselines gives (CASA uvmodelfit's model), not the fixed 1" one
                gfit_info = vs.fit_gaussian(data.u, data.v, data.vis, data.weight, q_max=1e3 * args.gauss_fit_max_klambda)
                say(f"start: elliptical Gaussian fitted to {gfit_info['n_used']:,} of {gfit_info['n_in_cut']:,} visibilities below {gfit_info['q_max']:g} wavelengths: flux {gfit_info['flux_jy']:.5f} Jy, "
                    f"centroid ({gfit_info['east_mas']:.1f}, {gfit_info['north_mas']:.1f}) mas E, N, FWHM {gfit_info['major_mas']:.1f} x {gfit_info['minor_mas']:.1f} mas at PA {gfit_info['pa_deg']:.1f} deg, chi2/point {gfit_info['chi2_per_point']:.3f}")
                values["log_brightness"] = jnp.asarray(vs.gaussian_log_image(prob.npix, prob.pixel_mas, gfit_info["east_mas"], gfit_info["north_mas"], gfit_info["major_mas"], gfit_info["minor_mas"], gfit_info["pa_deg"]))
                values["cal.flux"] = jnp.asarray(gfit_info["flux_jy"])
            if not mpol and getattr(args, "start_floor", 0):  # the Gaussian's outer pixels are 1e-8..1e-15 of the peak and cannot grow back: raise them to MPoL's entropy floor
                log_pb = va.beam_log(prob.npix, prob.pixel_mas) if args.config == "F" else None
                eta, flux0, floor_info = va.floor_start(values, prob.npix, prob.pixel_mas, args.start_floor, log_pb=log_pb)
                values["log_brightness"], values["cal.flux"] = jnp.asarray(eta), jnp.asarray(flux0)
                say(f"start: 1\" Gaussian raised to the floor {args.start_floor:g} Jy/arcsec^2 at {floor_info['n_raised']:,} pixels (+{floor_info['flux_added_jy']:.5f} Jy on {floor_info['flux_before_jy']:.5f} Jy)")
        if fixed is not None:  # the held image is no free site: the optimiser sees the nuisances only
            values.pop("log_brightness")
        limit = args.prestage_max_steps if name in ("selfcal", "image") else args.max_steps
        if getattr(args, "method", "lbfgs") == "adam":
            say(f"stage {name}: Adam lr {args.adam_lr:g}, eps {args.adam_eps:g}, steps {args.adam_steps}, time limit {args.time_limit}")
        else:
            say(f"stage {name}: max_steps {limit}, time limit {args.time_limit}, gtol {args.gtol}")
        t = time.time()

        def with_image(vals, fixed=fixed):
            """``vals`` as numpy, with the held image put back (a saved fit always has its image)."""
            vals = {k: np.asarray(v) for k, v in vals.items()}
            return vals if fixed is None else {**vals, "log_brightness": np.asarray(fixed)}

        def on_round(vals, info, traj, rnd, name=name, t=t, prob=prob, with_image=with_image):
            say(f"stage {name}: round {rnd} done, {info['steps']} steps, stop {info['stop']}, loss {info['loss']:.3f}")
            if save:
                part = dict(stage_summary(name, info, traj, time.time() - t, "in_progress"), chi2=prob.chi2_info)
                save(prob, with_image(vals), stages + [dict(part, in_progress=True)])

        if getattr(args, "method", "lbfgs") == "adam":
            out, info, traj, why = va.run_adam(prob, values, steps=args.adam_steps, lr=args.adam_lr, adam_eps=args.adam_eps, time_limit=args.time_limit, log=say, precondition=args.precondition,
                                                post_step_blur_px=args.post_step_blur_px, lr_decay=args.adam_lr_decay)
            info["method"] = "adam"
        else:
            out, info, traj, why = va.run_fit_c8(prob, values, max_steps=limit, time_limit=args.time_limit, log=say, gtol=args.gtol, on_round=on_round, precondition=args.precondition)
        stages.append(dict(stage_summary(name, info, traj, time.time() - t, why), chi2=prob.chi2_info))
        stages[-1]["dead_pixels"] = dp = stage_dead_pixels(with_image(out), prob, args.config)
        say(f"stage {name}: dead pixels, base {dp['base']['frac_below_1e-6_peak']:.2%} below 1e-6 and {dp['base']['frac_below_1e-4_peak']:.2%} below 1e-4 of the peak, live to {dp['base']['largest_live_radius_arcsec']:.2f} arcsec; "
            f"tapered {dp['tapered']['frac_below_1e-6_peak']:.2%} / {dp['tapered']['frac_below_1e-4_peak']:.2%}, live to {dp['tapered']['largest_live_radius_arcsec']:.2f} arcsec")
        if floor_info is not None or gfit_info is not None:  # the first stage's start (it is the only one that is not warm)
            stages[-1].update({k: v for k, v in (("start_floor", floor_info), ("start_gauss_fit", gfit_info)) if v is not None})
            floor_info = gfit_info = None
        if clean and name == "selfcal":
            stages[-1]["clean_start"] = start_info
            stages[-1]["selfcal_check"] = chk = vs.selfcal_check(out, ref_obs=args.ref_obs)
            say(f"selfcal: gains {chk['gain_min']:.4f}-{chk['gain_max']:.4f} (LB obs 5, 6: {np.round(chk['gain_lb_obs5_6'], 4)}; R_mpol 1.02-1.03, MPoL 0.94-0.98), flux {chk['flux_jy']:.5f} Jy, "
                f"largest offset {chk['offset_max_abs_mas']:.2f} mas" + ("; !!! A GAIN IS OUTSIDE THE REFERENCE RANGE: the held image may be biased !!!" if chk["gain_outside_reference"] else ""))
        say(f"stage {name}: stopped because {why} after {info['steps']} steps in {info['rounds']} rounds; C8 converged: {stages[-1]['convergence']['converged']}")
        values = {k: jnp.asarray(v) for k, v in with_image(out).items()}
        previous = name
        if save:
            save(prob, with_image(out), stages)
    return prob, {k: np.asarray(v) for k, v in values.items()}, stages, shat


def clean_start_values(args, prob, shat):
    """``(values, log_brightness, info)`` of ``--workflow clean``: ``prob``'s defaults with the image and flux of our CLEAN model
    (``vs.clean_start``) and s at ŝ (``--shat``) or 1; ``prob`` is a free-image problem, only its grid and defaults are used."""
    import jax.numpy as jnp

    if not args.clean_dir:
        raise SystemExit("--workflow clean needs --clean-dir (the output directory of alma_imlup_clean.py run: model.fits, clean.json)")
    model, px, beam, cinfo = vs.read_clean(args.clean_dir)
    eta, flux, info = vs.clean_start(model, px, beam, npix=prob.npix, pixel_mas=prob.pixel_mas, smooth_mas=args.clean_smooth_mas, beam_frac=args.clean_beam_frac, floor=args.clean_floor)
    info.update(cinfo, source_pixel_mas=px)
    say(f"CLEAN start from {args.clean_dir}: model {cinfo['flux_jy']:.5f} Jy on {cinfo['shape'][0]}^2 x {px:g} mas, beam {beam}, smoothing {args.clean_smooth_mas:g} mas (beam fraction {args.clean_beam_frac:g}); "
        f"on {prob.npix}^2 x {prob.pixel_mas:g} mas: flux {flux:.5f} Jy (in field {info['flux_in_field_jy']:.5f}, negative clipped {info['negative_flux_clipped_jy']:.2e}, "
        f"floor {args.clean_floor:g} Jy/arcsec^2 at {info['n_at_floor']:,} pixels)")
    values = dict(prob.init)
    values["log_brightness"] = jnp.asarray(eta)
    values["cal.flux"] = jnp.asarray(flux)
    if shat is not None and "cal.sigma" in values:
        values["cal.sigma"] = jnp.asarray(shat, float)
    return values, jnp.asarray(eta), info


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
        stop=last["stop"], stopped_because=last["stopped_because"], flux_jy=float(values["cal.flux"]), ref_obs=getattr(args, "ref_obs", 0),
        at_bound=va.at_bound_report(values, prob.priors), **chi2_record(stages)))


def chi2_record(stages):
    """``chi2_engine`` and the Toeplitz precompute of each stage (time, array size), from the stages' ``chi2`` records."""
    infos = [dict(stage=s["name"], **s["chi2"]) for s in stages if "chi2" in s]
    engine = infos[-1]["engine"] if infos else "direct"
    pre = [i for i in infos if i["engine"] == "toeplitz"]
    return dict(chi2_engine=engine, chi2_stages=infos, toeplitz_precompute_s=sum(i["precompute_s"] for i in pre) if pre else None,
                toeplitz_arrays_gb=max((i["arrays_gb"] for i in pre), default=None))


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
        clean_start=next((s["clean_start"] for s in stages if "clean_start" in s), None),
        start_floor=next((s["start_floor"] for s in stages if "start_floor" in s), None),
        start_gauss_fit=next((s["start_gauss_fit"] for s in stages if "start_gauss_fit" in s), None),
        alpha=va.alpha_report(values if "cal.alpha" in prob.init else {})[0], alpha_label=va.alpha_report(values if "cal.alpha" in prob.init else {})[1],
        at_bound=va.at_bound_report(values, prob.priors), **chi2_record(stages),
        offset_bound=va.offset_bound_report(values.get("cal.offset"), prob.offset_bound_mas, prob.offset_bound_requested, bound_obs=prob.offset_bound_obs, ref_obs=getattr(args, "ref_obs", 0)),
        nll=sc_raw["nll"], chi2_raw=sc_raw["chi2_raw"], chi2_fit=sc_raw["chi2_fit"],
        chi2_s1=chi2_table(sc_s1, "s = 1 (quoted weights), fitted g, offsets, image"), chi2_fitted_s=chi2_table(sc_raw, "fitted s"),
    )
    b3 = va.offset_beam_bounds(data)
    res.update(ref_obs=getattr(args, "ref_obs", 0), ref_obs_info=getattr(args, "ref_obs_info", None),
               offset_bound_3beam=dict(per_obs_mas=b3, bound_in_force_mas=prob.offset_bound_mas, tighter_than_all=bool(np.all(prob.offset_bound_obs < b3)),
                                       note="3 / q_max of each observation: a proxy for 3 nominal beams, not virgil's own definition"))
    ob = res["offset_bound"]
    say(f"offset bound ±{ob['bound_mas']:.6g} mas ({'--offset-bound-mas' if ob['requested_mas'] is not None else 'virgil default'}); largest |offset|/bound {ob['max_abs_over_bound']:.3g}")
    if ob["warning"]:
        say(ob["warning"])
    say(f"chi2 engine {res['chi2_engine']}" + (f": precompute {res['toeplitz_precompute_s']:.1f} s over {len(res['chi2_stages'])} stage(s), arrays {res['toeplitz_arrays_gb']:.2f} GB" if res["toeplitz_precompute_s"] is not None else ""))
    say(res["at_bound"]["warning"] or f"at bound: none of {res['at_bound']['n_checked']} bounded parameters within {res['at_bound']['edge']:.0%} of a prior bound")
    dead = va.dead_pixel_report(va.image_of(values, prob.npix, prob.pixel_mas, va.beam_log(prob.npix, prob.pixel_mas) if args.config == "F" else None), prob.pixel_mas)
    res["dead_pixels"] = dead
    say(f"dead pixels: {dead['frac_below_1e-6_peak']:.2%} below 1e-6 of the peak, {dead['frac_below_1e-3_floor']:.2%} below 1e-3 of the floor; largest live radius {dead['largest_live_radius_arcsec']:.2f} arcsec")
    if dead["warning"]:
        say(dead["warning"])
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


def _prob_for(args, data, lam, mask, chi2=None):
    def prob_for(use, fixed=None, image_only=False):
        m = None if mask is None else (mask if use is None else mask[np.isin(np.asarray(data.obs), va.SB_OBS)])
        return va.build_problem(data, args.config, lam=lam, npix=args.npix, pixel_mas=args.pixel_mas, mask=m, use=use, chunk_size=args.chunk_size, alpha_free=getattr(args, "alpha_free", None), offset_bound_mas=getattr(args, "offset_bound_mas", None),
                              fixed_log_brightness=fixed, image_only=image_only, ref_obs=getattr(args, "ref_obs", 0), chi2=chi2 or getattr(args, "chi2", None) or "direct")

    return prob_for


def resolve_alpha(args):
    """Record the resolved alpha setting in the saved args (the CLI default is None, which would hide a changed default)."""
    if getattr(args, "alpha_free", None) is None:
        args.alpha_free = bool(va.FULL["alpha_free"])


def cmd_fit(args):
    resolve_alpha(args)
    if args.post_step_blur_px and args.method != "adam":
        raise SystemExit("--post-step-blur-px needs --method adam (virgil's post_step hook is not available to L-BFGS)")
    if args.adam_lr_decay is not None and (args.method != "adam" or not 0 < args.adam_lr_decay < 1):
        raise SystemExit("--adam-lr-decay needs --method adam and a factor in (0, 1)")
    if args.post_step_blur_px is not None and args.post_step_blur_px < 0:
        raise SystemExit("--post-step-blur-px must be >= 0")
    if args.method == "adam":
        import virgil

        try:
            va.require_adam(virgil.fit, args.precondition, args.post_step_blur_px, args.adam_lr_decay)
        except RuntimeError as e:
            raise SystemExit(str(e)) from e
    if args.workflow in ("mpol", "clean") and args.config != "R":
        raise SystemExit(f"--workflow {args.workflow} is defined for config R only (MPoL has no beam, no free alpha)")
    if args.chi2 == "toeplitz" and args.config != "R":
        raise SystemExit("--chi2 toeplitz is defined for config R only (one Image, no fitted alpha)")
    if args.chi2 == "toeplitz" and args.offset_bound_mas is None:  # the precompute's box: recorded, so report and --grad replay the fitted prior
        args.offset_bound_mas = va.TOEPLITZ_OFFSET_BOUND_MAS
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load(args)
    n = len(data.u)
    lam = lam_of(args, n)
    mask = _train_mask(args, data)
    args.ref_obs, args.ref_obs_info = va.resolve_ref_obs(data, args.ref_obs)
    say(f"reference observation {args.ref_obs} (--ref-obs {args.ref_obs_info['spec']}; sum of weights of obs {list(va.SB_OBS)}: {np.array2string(args.ref_obs_info['sb_sum_weight'], precision=4)})")
    if args.init:  # a warm start is in its own fit's gauge
        init_json = pathlib.Path(args.init) / "result.json"
        init_ref = int(json.loads(init_json.read_text()).get("ref_obs", 0)) if init_json.exists() else 0
        if init_ref != args.ref_obs:
            raise SystemExit(f"--init {args.init} was fitted with reference observation {init_ref}, not {args.ref_obs}: its gains, flux and offsets are in that gauge")
    bound = va.offset_beam_bounds(data)
    ob = args.offset_bound_mas
    verdict = "virgil's own bound applies" if not ob else f"--offset-bound-mas {ob:g} is tighter than {int((bound > ob).sum())} of {len(bound)}"
    say(f"3 nominal beams (3 / q_max) per observation, mas: {np.array2string(bound, precision=1)}; {verdict}")
    say(f"config {args.config}, alpha {'free' if getattr(args, 'alpha_free', None) else va.ALPHA_FIXED_LABEL}, lambda {lam:.6g} (lam_R at N = {n:,} is {va.lam_r(n):.6g})")
    t0 = time.time()
    save = lambda prob, values, stages: write_state(out, args, prob, values, stages, lam, n)  # noqa: E731
    try:
        prob, values, stages, shat = run_stages(args, data, _prob_for(args, data, lam, mask), args.init, save)
    except va.ToeplitzRefused as e:
        raise SystemExit(str(e)) from e
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
    prob = _prob_for(args, data, lam, mask, chi2="direct")(use)  # the report scores with the direct term; the fit's engine is in its stages' records
    values = va.drop_unused_sites(values, prob)
    shat = va.read_shat(args.shat, int(np.asarray(data.ddid).max()) + 1) if args.shat else None
    say(f"report of {out}: config {args.config}, lambda {lam:.6g}, stages {[s['name'] for s in stages]}")
    write_report(args, out, data, prob, values, stages, shat, lam, mask, saved.get("wall_s"))
    if args.grad and (mask is not None or use is not None):
        say("--grad skipped: the diagnostic evaluates the all-data, unmasked objective, not this fit's (fold mask or SB-only last stage)")
    elif args.grad and args.config == "R":
        groups = {"report": va.grad_groups(data, values, lam=lam, pixel_mas=args.pixel_mas or 5.0, chunk_size=args.chunk_size, offset_bound_mas=getattr(args, "offset_bound_mas", None), ref_obs=getattr(args, "ref_obs", 0))}
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
    init_json = pathlib.Path(args.init) / "result.json"
    init_ref = int(json.loads(init_json.read_text()).get("ref_obs", 0)) if init_json.exists() else 0
    if init_ref != 0:  # cv-task builds the default problem (reference obs 0): another reference's n-1 gains and offsets would land on the wrong observations
        raise SystemExit(f"--init {args.init} was fitted with reference observation {init_ref}: cv-task supports reference observation 0 only")
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


def cmd_cv_compare(args):
    """MPoL's held-out scores (``score.json`` per fold, ``mpol_worker.score_fold``) beside virgil's (``cv.json`` per lambda and fold, with
    ``cv_summary.json``) on the same folds, at lambda* and at the grid lambda nearest MPoL's lam_ent (C11's conversion)."""
    from external_bridge import cv_compare as cc

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mpol = cc.load_mpol(args.mpol)
    virgil, summary = cc.load_virgil(args.virgil)
    if not virgil:
        raise SystemExit(f"no cv.json under {args.virgil}")
    rec = next(iter(virgil.values()))
    n_all = sum(next(r for (lam, f), r in virgil.items() if f == fold)["n_test"] for fold in sorted({f for _, f in virgil}))  # every point is held out once
    lam0 = va.lam_f_equivalent(va.lam_r(n_all, args.lam_ent))
    res = cc.compare(mpol, virgil, summary, lam0, args.lam_ent)
    res["n_all"], res["lam_r"] = int(n_all), va.lam_r(n_all, args.lam_ent)
    res["virgil_versions"] = rec.get("versions")
    dump(out / "mpol_cv_compare.json", res)
    (out / "mpol_cv_compare.md").write_text(cc.markdown(res))
    cc.plot(res, out / "mpol_cv_compare.png")
    say(cc.markdown(res))


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


def fit_notes(vres):
    """Markdown for the gate output: the fit's chi^2 engine and which fitted parameters ended at a prior bound (result.json's ``at_bound``)."""
    ab = vres.get("at_bound")
    pre = vres.get("toeplitz_precompute_s")
    md = [f"chi^2 engine of the fit: **{vres.get('chi2_engine', 'direct')}**" + (f" (precompute {pre:.1f} s, arrays {vres['toeplitz_arrays_gb']:.2f} GB)" if pre is not None else "") + "."]
    if ab is None:
        md.append("At a prior bound: not recorded (a fit made before `at_bound` existed; `report` recomputes it).")
    elif ab["at_bound"]:
        md += [f"**At a prior bound ({len(ab['at_bound'])} of {ab['n_checked']} bounded parameters, within {ab['edge']:.0%} of the range):**", "",
               "| parameter | value | bound |", "|---|---:|---:|"] + [f"| {h['site']}{h['index']} | {h['value']:.6g} | {h['side']} {h['bound']:.6g} |" for h in ab["at_bound"]]
    else:
        md.append(f"At a prior bound: none of {ab['n_checked']} bounded parameters is within {ab['edge']:.0%} of its range of a bound.")
    st = [x for x in vres.get("stages", []) if "dead_pixels" in x]
    if st:
        md += ["", "Dead pixels after each stage (fraction of pixels below a fraction of the peak; base image / tapered to MPoL's beam):", "",
               "| stage | method | steps | final loss | C8 | base <1e-6 | base <1e-4 | base live radius (arcsec) | tapered <1e-6 | tapered <1e-4 | tapered live radius (arcsec) |", "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
        for x in st:
            b, t = x["dead_pixels"]["base"], x["dead_pixels"]["tapered"]
            md.append(f"| {x['name']} | {x.get('method', 'lbfgs')} | {x['steps']} | {x['loss_final']:.2f} | {x['convergence']['converged']} | {b['frac_below_1e-6_peak']:.2%} | {b['frac_below_1e-4_peak']:.2%} | "
                      f"{b['largest_live_radius_arcsec']:.2f} | {t['frac_below_1e-6_peak']:.2%} | {t['frac_below_1e-4_peak']:.2%} | {t['largest_live_radius_arcsec']:.2f} |")
    return md


def cold_start_gate_of(res, vres, image, px, flux, clean_dir, dsharp=None):
    """The cold-start gate (``vs.cold_start_gate``) of a compare: ``res`` holds the objective comparison (empty without ``--data``: (a) and (b) are then not evaluated and
    the verdict is INCOMPLETE), ``vres`` the fit's result.json (its stages' C8 flags), ``image`` the fit's image."""
    o = res.get("objective")
    n = o["ours"]["n"] if o else None
    raw = lambda x: sum(x["by_ddid"]["chi2_raw"]) / (2 * n)  # noqa: E731
    if clean_dir:
        ref, name = float(vs.read_clean(clean_dir)[3]["flux_jy"]), f"CLEAN ({clean_dir})"
    elif dsharp:  # measured from the FITS given: its sum in Jy/beam x pixel area / beam area
        ref, name = vs.clean_image_flux(dsharp)["flux_jy"], f"DSHARP CLEAN image flux ({pathlib.Path(dsharp).name})"
    else:
        ref, name = vs.REFERENCE_FLUX_JY, vs.REFERENCE_FLUX_NAME
    stages = [bool(st.get("c8_converged", st["convergence"]["converged"])) for st in vres.get("stages", [])]
    return vs.cold_start_gate(
        objective_ours=o and o["ours"]["objective"], objective_mpol=o and o["mpol_solution_in_virgil"]["objective"],
        chi2n_ours=o and raw(o["ours"]), chi2n_mpol=o and raw(o["mpol_solution_in_virgil"]), stages_converged=stages,
        dead_fraction=va.dead_pixel_report(image, px)["frac_below_1e-6_peak"], flux_ours=flux, flux_ref=ref, flux_ref_name=name)


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
    ref = int(vres.get("ref_obs", 0))  # our reference observation; MPoL's is obs 0: put ours in its convention (gauge_to_obs0)
    g_o, off_o, flux_o = va.gauge_to_obs0(vz["cal.gain"], vz["cal.offset"], vz["cal.flux"], ref)
    res["ref_obs"] = dict(ours=ref, mpol=0, note="gains, flux and offsets below are renormalised to obs 0 = (gain 1, offset 0), MPoL's convention; "
                          "the offsets' common shift is a translation of the whole sky (not applied to the images); -ln L is gauge invariant and the entropy is evaluated at the obs-0-gauge flux, so the objective below compares like with like")
    res["gain"] = dict(ours=g_o, mpol=amp, max_abs_diff=float(np.max(np.abs(g_o - amp))), ours_in_fit_gauge=np.insert(np.asarray(vz["cal.gain"], float), ref, 1.0))
    off_m = 1000.0 * np.stack([mz["delta_ra_arcsec"], mz["delta_dec_arcsec"]], -1)
    res["offset_mas"] = dict(ours=off_o, mpol=off_m, max_abs_diff=float(np.max(np.abs(off_o - off_m))))
    dx = float(mz["cell_size_arcsec"])
    flux_m = float(np.sum(mz["base_image"]) * dx**2)
    res["flux_jy"] = dict(ours=flux_o, mpol=flux_m, diff=flux_o - flux_m, ours_in_fit_gauge=float(vz["cal.flux"]))
    img_o = np.load(vdir / "image.npy").astype(float) * (flux_o / float(vz["cal.flux"]))  # the image in the same flux gauge
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
        # -ln L is gauge invariant but S_M scales with the flux: compare in MPoL's (obs-0) gauge, with S_M at flux_o = flux g_0 (the offsets' translation gauge needs no change: it never enters S_M)
        o_ours = va.r_objective(data, v_ours, lam=lam, pixel_mas=px, chunk_size=args.chunk_size, ref_obs=ref, entropy_flux=flux_o if ref != 0 else None)
        o_mpol = va.r_objective(data, v_mpol, lam=lam, pixel_mas=px, chunk_size=args.chunk_size)
        res["objective"] = dict(lam_R=lam, gauge="obs 0 (MPoL's): entropy at flux·g_0" if ref != 0 else "obs 0", ours=o_ours, mpol_solution_in_virgil=o_mpol, mpol_own=obj_m and {k: obj_m.get(k) for k in ("neg_log_likelihood", "entropy", "objective_R")},
                                ours_le_mpol=bool(o_ours["objective"] <= o_mpol["objective"]),
                                finding=None if o_ours["objective"] <= o_mpol["objective"] else "our optimiser ended above MPoL's solution: a finding about the fit, not tuned away")
        if not args.no_grad:
            res["grad_groups"] = {}
            for label, v in (("ours", v_ours), ("mpol_solution_in_virgil", v_mpol)):
                try:
                    res["grad_groups"][label] = va.grad_groups(data, v, lam=lam, pixel_mas=px, chunk_size=args.chunk_size, ref_obs=ref if label == "ours" else 0)
                except Exception as e:  # a diagnostic: never loses the comparison
                    res["grad_groups"][label] = dict(error=f"{type(e).__name__}: {e}")
        if not args.no_dirty:  # the residual dirty image of MPoL's solution per observation, beside the ones `fit` saves for ours (docs/imaging/imlup.md)
            prob = va.build_problem(data, "R", lam=lam, npix=v_mpol["log_brightness"].shape[0], pixel_mas=px, chunk_size=args.chunk_size)
            dirty = va.residual_dirty_images(data, prob.scene, v_mpol, prob.nuis, chunk_size=args.chunk_size)
            for e, img in dirty.items():
                np.save(out / f"resid_dirty_mpol_obs{e}.npy", img.astype(np.float32))
            res["resid_dirty_mpol"] = dict(files=[f"resid_dirty_mpol_obs{e}.npy" for e in sorted(dirty)], npix=1024, pixel_mas=3.0, weights="w / s^2 with s = g s_MPoL")
    if args.f_dir and args.dsharp:
        fz = np.load(pathlib.Path(args.f_dir) / "image.npy").astype(float)
        n = fz.shape[0]
        conv = cx.convolve(fz, px, *beam) * cx.beam_area_arcsec2(beam[0], beam[1])  # Jy/beam
        data_d, wcs_d, _ = cx.read_fits(args.dsharp)
        d_on = cx.resample(data_d, wcs_d, cx.sky_wcs(n, px, RA_DEG, DEC_DEG), (n, n))[::-1]  # back to the repository's layout
        np.save(out / "f_clean_beam.npy", conv.astype(np.float32))
        np.save(out / "dsharp_regridded.npy", d_on.astype(np.float32))
        res["F_vs_dsharp"] = dict(beam_mas_deg=beam, **cx.difference_stats(conv, d_on))
    gate = cold_start_gate_of(res, vres, img_o, px, flux_o, args.clean_dir, args.dsharp)
    res["gate"] = gate
    dump(out / "compare.json", res)
    md = ["# R against MPoL", ""] + vs.gate_markdown(gate) + [""] + fit_notes(vres) + [""]
    if ref != 0:
        md += [f"**Our fit's reference observation is {ref}, not 0** (`--ref-obs`): gains, flux and offsets below are renormalised to MPoL's obs-0 convention ({res['ref_obs']['note']}).", ""]
    if "objective" in res:
        n_tot = res["objective"]["ours"]["n"]
        raw = lambda o: sum(o["by_ddid"]["chi2_raw"]) / (2 * n_tot)  # noqa: E731
        md += [f"**Raw chi^2/N (s = 1, quoted weights): ours {raw(res['objective']['ours']):.4f}; MPoL's solution evaluated in virgil {raw(res['objective']['mpol_solution_in_virgil']):.4f}** (N = {n_tot:,}, eps 1e-12).", ""]
    md += [f"s_ours / (g s_MPoL): max |ratio − 1| = {res['sigma']['max_abs_dev']:.4g}",
          f"gain: max |diff| = {res['gain']['max_abs_diff']:.4g}; offsets: max |diff| = {res['offset_mas']['max_abs_diff']:.4g} mas; flux {res['flux_jy']['ours']:.5f} vs {flux_m:.5f} Jy (ours in obs-0 gauge; reference obs {ref})", ""]
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


def cmd_speed(args):
    """Speed of virgil's L-BFGS against MPoL's AdamW run (``crosscheck.alma_speed``): objective against wall time, time to targets, cost per
    full-data pass, fairness notes. Reads small JSON only (each fit's ``result.json``, MPoL's ``stage<N>.time.json``, ``export/objective.json``
    and, if it exists, ``export/objective_chk0.json``) and optional job logs; needs no data and no GPU. Writes speed.json, speed.md, speed.png."""
    from crosscheck import alma_speed as sp

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fits = {}
    for spec in args.fit:
        label, _, d = spec.rpartition("=") if "=" in spec else ("", "", spec)
        d = pathlib.Path(d)
        res = json.loads((d / "result.json").read_text())
        if (d / "anchors.json").exists():  # optional second evaluations of R inside the fit: [{"steps", "objective", "loss"}]
            res["anchors"] = json.loads((d / "anchors.json").read_text())
        fits[label or d.name] = res
    mdir = pathlib.Path(args.mpol)
    run = mdir.parent if (mdir.name == "export" and not list(mdir.glob("stage*.time.json"))) else mdir
    stages = [json.loads(p.read_text()) for p in sorted(run.glob("stage*.time.json"))]
    if not stages:
        raise SystemExit(f"no stage<N>.time.json in {run} (train.sbatch of alma_mpol_ref writes them)")
    odir = run / "export" if (run / "export" / "objective.json").exists() else run
    objective = json.loads((odir / "objective.json").read_text())
    chk0 = json.loads((odir / "objective_chk0.json").read_text()) if (odir / "objective_chk0.json").exists() else None
    mpol = sp.mpol_summary(stages, objective, chk0, startup_s=args.mpol_startup_s)
    logs, owners = {}, {}
    for spec in args.log:  # [mpol=|virgil=]FILE
        owner, _, f = spec.partition("=") if spec.partition("=")[0] in ("mpol", "virgil") else ("", "", spec)
        logs[f] = pathlib.Path(f).read_text()
        if owner:
            owners[f] = owner
    hardware = sp.parse_logs(logs, owners)
    extra = {}
    for t in args.target:
        k, _, v = t.rpartition("=")
        extra[k or f"target {v}"] = float(v)
    res = sp.build(fits, mpol, extra, vg_seconds=tuple(args.vg_seconds), hardware=hardware)
    res["fairness_notes"] = sp.fairness_notes(res)
    res["command"], res["versions"] = "speed", versions()
    dump(out / "speed.json", res)
    (out / "speed.md").write_text(sp.markdown(res, res["fairness_notes"]))
    sp.plot(res, out / "speed.png")
    say((out / "speed.md").read_text())


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
    p.add_argument("--chi2", choices=va.CHI2_ENGINES, default="direct", help="config R: the data term's engine. toeplitz: virgil's dirty-beam chi^2 in every stage (equal to direct to ~1e-13, cost independent of N; "
                   "fitted offsets then need their box inside the precompute: --offset-bound-mas defaults to 20, and baselines past its Nyquist margin are refused). The report and compare always score with the direct term")
    p.add_argument("--method", choices=["lbfgs", "adam"], default="lbfgs", help="lbfgs (default): L-BFGS restarted until C8. adam: virgil.fit's Adam, one run per stage of --adam-steps, stopped early on C8 "
                   "(per-coordinate steps keep unconstrained pixels alive where L-BFGS drives them to logit -inf; needs a virgil with the Adam keywords)")
    p.add_argument("--adam-lr", type=float, default=1e-2, help="adam: learning rate in the unconstrained coordinates")
    p.add_argument("--adam-lr-decay", type=float, default=None, help="adam only: multiply the learning rate by this factor in (0, 1) every 1000 steps (optax exponential decay, per stage); default off. "
                   "Needs a virgil whose fit() takes an optax schedule")
    p.add_argument("--adam-steps", type=int, default=5000, help="adam: step budget per stage (all stages, selfcal and image included)")
    p.add_argument("--post-step-blur-px", type=float, default=None, help="adam only: after every optimiser step blur the normalised image with a Gaussian of this sigma in pixels (dorito's recipe, "
                   "Charles et al. 2026: 0.25) and re-express it as log_brightness; default off. Needs a virgil whose fit() has post_step")
    p.add_argument("--adam-eps", type=float, default=1e-8, help="adam: the epsilon in the denominator of the update")
    p.add_argument("--sb-only", action="store_true")
    p.add_argument("--workflow", choices=["p3", "mpol", "clean"], default="p3",
                   help="p3 (default): C9's SB stage then all data from a 1\" Gaussian, s at ŝ. mpol (config R only): one all-data stage from MPoL's own start "
                        "(uniform softplus(-3) image, g = 1, offsets 0, s = 1; sgd.py:49-59, images.py BaseCube). clean (config R only, no MPoL information): start from "
                        "our own CLEAN model (--clean-dir), stage selfcal fits flux, gains, offsets and s with the image held, stage all releases everything")
    p.add_argument("--clean-dir", default=None, help="clean: the output directory of alma_imlup_clean.py run (model.fits, clean.json)")
    p.add_argument("--clean-smooth-mas", type=float, default=10.0, help="clean: FWHM (mas) of the small circular positivity kernel the CLEAN model is convolved with (default two fit pixels; keeps the long baselines' power)")
    p.add_argument("--clean-beam-frac", type=float, default=0.0, help="clean: instead smooth with this fraction of the restoring beam (0: off, the default; 1 removes the long baselines' power and biases their gains)")
    p.add_argument("--clean-floor", type=float, default=va.MPOL["prior"], help="clean: floor of the start image, Jy/arcsec^2 (default MPoL's entropy prior p = 1e-4)")
    p.add_argument("--start", choices=["gauss", "gauss-fit"], default="gauss",
                   help="p3 workflow's cold start: gauss (default): the fixed 1\" Gaussian of 0.25 Jy; gauss-fit: an elliptical Gaussian (flux, centroid, FWHMs, PA) fitted by least squares to "
                        "the visibilities below --gauss-fit-max-klambda (CASA uvmodelfit's model), recorded in result.json (start_gauss_fit). Both are then raised to --start-floor")
    p.add_argument("--gauss-fit-max-klambda", type=float, default=vs.GAUSS_FIT_MAX_Q / 1e3, help="gauss-fit: the short baselines used, in kilo-wavelengths")
    p.add_argument("--start-floor", type=float, default=va.MPOL["prior"], help="p3 workflow: raise the 1\" Gaussian start to this level (Jy/arcsec^2) so every pixel starts alive (default MPoL's entropy prior 1e-4, as R_clean; "
                   "0: the bare Gaussian, whose outer pixels are 1e-8..1e-15 of the peak and cannot grow back: R ob100, job 18968246_0). Ignored by --workflow mpol and clean (their own starts) and by --init")
    p.add_argument("--prestage-max-steps", type=int, default=3000, help="clean, --image-first: step budget of the selfcal / image stage (stops earlier when C8 holds)")
    p.add_argument("--image-first", action="store_true", help="p3, mpol: a first stage fits only the image and flux (gains 1, offsets 0, s = 1), then the workflow's own stages release the nuisances")
    p.add_argument("--ref-obs", default="0", help="the observation whose gain is 1 and offset 0 (the flux gauge): an index, or 'auto' = the SB observation with the largest sum of weights; "
                   "result.json records it, and compare puts gains, flux and offsets back in MPoL's obs-0 convention")
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

    p = sub.add_parser("cv-compare", help="MPoL's held-out cross-validation scores beside virgil's on the same dartboard folds")
    p.add_argument("--mpol", required=True, help="directory holding fold_J/score.json (mpol_worker score_fold)")
    p.add_argument("--virgil", required=True, help="the cv directory of cv-task / cv-aggregate (cv.json per lambda and fold, cv_summary.json)")
    p.add_argument("--lam-ent", type=float, default=va.MPOL["lam_m"], help="MPoL's entropy weight lam_M (C4/C11: lam_R = 2 N lam_M)")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_cv_compare)

    p = sub.add_parser("compare")
    p.add_argument("--virgil", required=True)
    p.add_argument("--mpol", required=True)
    p.add_argument("--data", default=None, help="needed for the objective comparison")
    p.add_argument("--f-dir", default=None)
    p.add_argument("--dsharp", default=None)
    p.add_argument("--clean-dir", default=None, help="gate (e): the output directory of alma_imlup_clean.py run (clean.json); its flux is the reference; else the flux of --dsharp FITS if given; else the constant 0.2548 Jy (DSHARP CLEAN image, measured)")
    p.add_argument("--out", required=True)
    p.add_argument("--no-grad", action="store_true", help="skip the gradient-scaling diagnostic (two value-and-gradient evaluations on --data)")
    p.add_argument("--no-dirty", action="store_true", help="skip the residual dirty images of MPoL's solution (resid_dirty_mpol_obs*.npy, 4 MB each; one NUFFT pass and one gridding of all points)")
    p.add_argument("--chunk-size", type=int, default=va.CHUNK)
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("mpol-init", help="values.npz from MPoL's export, the --init of `fit --config R` (a warm start at MPoL's solution)")
    p.add_argument("--mpol", required=True, help="MPoL's export directory (export.npz)")
    p.add_argument("--data", default=None, help="optional: the ddid -> observation map is read from it (default IM Lup's obsid_dict)")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_mpol_init)

    p = sub.add_parser("speed", help="virgil's L-BFGS against MPoL's AdamW: objective vs wall time, time to target, cost per pass (small JSON only)")
    p.add_argument("--fit", action="append", required=True, help="a fit directory with result.json (config R), optionally LABEL=DIR; repeat for several fits")
    p.add_argument("--mpol", required=True, help="MPoL's run directory (stage<N>.time.json; export/objective.json and optional export/objective_chk0.json)")
    p.add_argument("--out", required=True)
    p.add_argument("--log", action="append", default=[], help="[mpol=|virgil=]FILE: a job log or digest with `job=… host=…` lines (and the GPU model), for the hardware note; the prefix says whose run it is "
                   "(without it the log counts as MPoL's only if its job is in stage<N>.time.json, else unattributed); repeat")
    p.add_argument("--target", action="append", default=[], help="an extra target LABEL=R (objective R); repeat")
    p.add_argument("--vg-seconds", type=float, nargs=2, default=list(__import__("crosscheck.alma_speed", fromlist=["x"]).BENCH_VG_S), metavar=("LO", "HI"),
                   help="seconds per value+gradient on the A100 bench (job 18862592) used to estimate evaluations per step; virgil's info records none")
    p.add_argument("--mpol-startup-s", type=float, default=0.0, help="MPoL start-up (asdf load, CUDA) per stage to subtract; default 0 (unmeasured, left in)")
    p.set_defaults(fn=cmd_speed)

    p = sub.add_parser("lambda0")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--flux", type=float, default=va.MPOL["f_tot"])
    p.set_defaults(fn=cmd_lambda0)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
