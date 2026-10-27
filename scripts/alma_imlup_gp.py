"""IM Lup: the MultipoleField GP image fit (stages 0-1 of ``design/alma_imlup_gp.md``; the helpers are in ``virgil_bridge.alma_gp``).

    python scripts/alma_imlup_gp.py --data DIR --out DIR --length-r-mas 40 --length-theta-rad 1 [--nuisances-from values.npz] [--mean-from-clean DIR] [--clean-dir DIR]
                                    [--max-steps N] [--time-limit S] [--no-dirty]

The image is ``Image(MultipoleField)`` on the gate grid (1028^2 x 5 mas, flux free): a Matern GP on a polar grid in the disc frame (n_r 160 x n_theta 256, r_max 2.4 arcsec),
geometry fixed at DSHARP's. Its whitened latents have a standard-normal prior, which is the regulariser: no taper and no entropy. The field amplitude sigma is fitted
(``multipole_field_priors``); ``--length-r-mas`` and ``--length-theta-rad`` are fixed. The nuisances are config R's (per-observation gains and offsets, per-ddid error
scales, observation 0 the reference) and the chi^2 is virgil's Toeplitz term.

Stage 1 holds the nuisances at ``--nuisances-from`` (a ``values.npz`` of a config R fit, or its directory; without it, at gains 1, offsets 0 and scales 1) and fits the latents,
sigma and the flux with ``virgil.fit(method="lbfgs")``; stage 2 fits everything. Each stage restarts L-BFGS until C8 holds (loss change < 0.1 over 100 steps) or its budget
runs out; ``--time-limit`` is the budget of both stages together (stage 1 may use half), so the report that follows still fits in the job. ``--mean-from-clean DIR`` sets the
field's mean to the log of the restored CLEAN image of ``alma_imlup_clean.py`` (floored), which also gives the beam; ``--clean-dir`` gives the beam alone.

Writes ``values.npz``, ``image.npy`` and ``image.fits`` (Jy/arcsec^2, rows North first), ``image_tapered.npy`` (30 mas), ``image_clean_beam.npy`` (Jy/beam, with a CLEAN
beam), ``resid_dirty_obs<e>.npy`` and ``result.json``: raw chi^2/N first, then -ln L, the flux and the flux beyond 1.8 arcsec, the nuisances, the hyperparameters, the stages'
steps, times and convergence verdicts, and the dots report of the image against its 30 mas taper. Every score is virgil's direct NUFFT at eps 1e-12 (C7).
"""

import argparse
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import alma_imlup_maxent as mx  # noqa: E402  (say, dump, versions, load: sets float64)
import virgil_bridge.alma as va  # noqa: E402
import virgil_bridge.alma_gp as vg  # noqa: E402
from crosscheck import alma_maxent as cx  # noqa: E402

STAGE1_SHARE = 0.5  # of --time-limit
say = mx.say


def read_nuisances(path, prob, ref_obs):
    """``(held, flux)``: ``cal.gain``, ``cal.offset`` and ``cal.sigma`` (and ``cal.flux``, else None) of a config R fit's ``values.npz`` (a file or its directory), each of the
    shape ``prob`` has; the fit's ``ref_obs`` (its ``result.json``, if beside it) must be ``ref_obs``, the gauge of its gains, flux and offsets."""
    import json

    path = pathlib.Path(path)
    path = path / "values.npz" if path.is_dir() else path
    result = path.with_name("result.json")
    if result.exists():
        theirs = int(json.loads(result.read_text()).get("ref_obs", 0))
        if theirs != ref_obs:
            raise SystemExit(f"--nuisances-from {path} was fitted with reference observation {theirs}, not {ref_obs}: its gains, offsets and flux are in that gauge")
    z = np.load(path)
    held = {}
    for k in vg.HELD_SITES:
        if k not in z:
            raise SystemExit(f"--nuisances-from {path} has no {k}")
        held[k] = np.asarray(z[k], float)
        if held[k].shape != tuple(prob.init[k].shape):
            raise SystemExit(f"--nuisances-from {path}: {k} has shape {held[k].shape}, the data need {tuple(prob.init[k].shape)}")
    return held, (float(z["cal.flux"]) if "cal.flux" in z else None)


def stage_record(name, info, traj, wall, why, eval_s):
    """One stage of the fit: steps, time, seconds per step (``wall_s`` over ``steps``, compile and setup included) and per value-and-gradient evaluation (``eval_s``, timed on the
    compiled term), and the C8 verdict (``va.convergence`` of the loss history; ``fit_converged`` is virgil's own gradient test)."""
    steps = int(info["steps"])
    return dict(name=name, steps=steps, rounds=info.get("rounds"), wall_s=wall, s_per_step=wall / max(steps, 1), eval_s=eval_s, stop=info["stop"], stopped_because=why,
                fit_converged=bool(info["converged"]), c8_converged=bool(va.convergence(traj, info)["converged"]), convergence=va.convergence(traj, info),
                loss_final=float(info["loss"]), trajectory=traj)


def fit(args, data):
    """The two-stage fit of ``args`` on ``data`` (a ``ComplexVisData``); writes the outputs of the module docstring into ``args.out`` and returns ``result.json``'s dict."""
    t_start = time.time()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ref_obs = int(args.ref_obs)
    mean, clean_info, beam = None, None, None
    if args.mean_from_clean or args.clean_dir:
        mean, beam, clean_info = vg.restored_clean(args.mean_from_clean or args.clean_dir, args.npix, args.pixel_mas)
        say(f"restored CLEAN image {clean_info['clean_dir']}: flux in the field {clean_info['flux_in_field_jy']:.5f} Jy, beam {beam[0]:.1f} x {beam[1]:.1f} mas @ {beam[2]:.0f} deg")
        if not args.mean_from_clean:
            mean = None  # the beam only
    field = dict(length_r=args.length_r_mas, length_theta=args.length_theta_rad, npix=args.npix, pixel_mas=args.pixel_mas, n_r=args.n_r, n_theta=args.n_theta,
                 r_max_mas=args.r_max_mas, inc=args.inc, pa=args.pa, mean=mean, mean_floor=args.mean_floor)
    try:
        prob = vg.build_gp_problem(data, ref_obs=ref_obs, offset_bound_mas=args.offset_bound_mas, **field)
    except va.ToeplitzRefused as e:
        raise SystemExit(str(e)) from e
    setup_s = time.time() - t_start
    n = len(data.u)
    ci = prob.chi2_info
    say(f"GP image: length_r {args.length_r_mas:g} mas, length_theta {args.length_theta_rad:g} rad, {args.n_r} x {args.n_theta} = {args.n_r * args.n_theta:,} latents; "
        f"Toeplitz precompute {ci['precompute_s']:.1f} s ({ci['arrays_gb']:.2f} GB) for {n:,} points")
    if args.nuisances_from:
        held, flux0 = read_nuisances(args.nuisances_from, prob, ref_obs)
    else:
        held, flux0 = {k: np.asarray(prob.init[k]) for k in vg.HELD_SITES}, None
    flux0 = va.MPOL["f_tot"] if flux0 is None else flux0
    import jax.numpy as jnp

    full = lambda values: {**{k: np.asarray(v) for k, v in held.items()}, **{k: np.asarray(v) for k, v in values.items()}}  # noqa: E731
    stages = []
    hyper = dict(length_r_mas=args.length_r_mas, length_theta_rad=args.length_theta_rad, n_r=args.n_r, n_theta=args.n_theta, r_max_mas=args.r_max_mas, inc_deg=args.inc, pa_deg=args.pa,
                 nu=vg.GP["nu"], sigma_prior=list(vg.GP["sigma_range"]), mean_from_clean=args.mean_from_clean, mean_floor=args.mean_floor if args.mean_from_clean else None)

    def save(values, stages, complete=False):
        """``values.npz`` and ``result.json``; with ``values=None`` only ``result.json`` (a non-finite stage: ``values.npz`` keeps the last finite round)."""
        if values is not None:
            np.savez(out / "values.npz", **full(values))
        mx.dump(out / "result.json", dict(command="fit", complete=complete, args={k: v for k, v in vars(args).items() if k != "fn"}, hyperparameters=hyper, stages=stages,
                                          flux_jy=None if values is None else float(values["cal.flux"]), versions=mx.versions()))

    budget = args.time_limit
    t_fit = time.time()
    init = {**{k: jnp.asarray(v) for k, v in prob.init.items() if k in ("field.latent", "field.sigma")}, "cal.flux": jnp.asarray(flux0)}
    p1, values = vg.hold_nuisances(prob, held), None
    for name, p, start, share in (("held", p1, init, STAGE1_SHARE), ("all", prob, None, 1.0)):
        if start is None:  # stage 2 starts at stage 1's end, nuisances included
            start = {k: jnp.asarray(v) for k, v in full(values).items()}
        left = None if budget is None else (share * budget if share < 1.0 else budget - (time.time() - t_fit))
        say(f"stage {name}: {len(p.init)} sites, budget {'none' if left is None else f'{left:.0f} s'}, {args.max_steps} steps")
        eval_s = vg.eval_seconds(p, start)
        say(f"  {eval_s * 1e3:.1f} ms per value-and-gradient evaluation (compiled)")
        t0 = time.time()
        values, info, traj, why = va.run_fit_c8(p, start, max_steps=args.max_steps, time_limit=left, log=say, gtol=args.gtol,
                                                on_round=lambda v, i, t, r: save(v, stages + [dict(name=name, in_progress=True, steps=i["steps"], loss=float(i["loss"]))]))
        stages.append(stage_record(name, info, traj, time.time() - t0, why, eval_s))
        say(f"stage {name}: {stages[-1]['steps']} steps in {stages[-1]['wall_s']:.0f} s ({stages[-1]['s_per_step']:.3f} s/step), stopped because {why}; "
            f"C8 {'converged' if stages[-1]['c8_converged'] else 'NOT converged'}")
        if why == "non_finite":  # run_fit_c8 returns the bad round's values and skips on_round for them: never overwrite the last good save with NaNs
            save(None, stages)
            raise SystemExit(f"stage {name} went non-finite; values.npz (if there is one) is the last finite round's")
        save(values, stages)
    fit_s = time.time() - t_fit

    values = full(values)
    image = vg.image_of(prob, values)
    np.save(out / "image.npy", image.astype(np.float32))
    cx.write_fits(out / "image.fits", image, prob.pixel_mas, mx.RA_DEG, mx.DEC_DEG, OBJECT="IM Lup", ORIGIN="virgil MultipoleField GP")
    tapered = va.tapered_image(image, prob.pixel_mas)
    np.save(out / "image_tapered.npy", tapered.astype(np.float32))
    if beam is not None:
        conv = cx.convolve(image, prob.pixel_mas, *beam) * cx.beam_area_arcsec2(beam[0], beam[1])
        np.save(out / "image_clean_beam.npy", conv.astype(np.float32))
    sc = va.score(data, prob.scene, values, prob.nuis, chunk_size=args.chunk_size)
    sigma = np.asarray(values["cal.sigma"], float)
    gain, offset, flux_gauge = va.gauge_to_obs0(values["cal.gain"], values["cal.offset"], values["cal.flux"], ref_obs)
    res = dict(
        chi2_raw_over_n=sc["chi2_raw"] / (2 * sc["n"]), n_points=sc["n"], nll=sc["nll"], chi2_raw=sc["chi2_raw"], chi2_fit=sc["chi2_fit"],
        flux_jy=float(values["cal.flux"]), flux_beyond_1p8_arcsec_jy=vg.flux_beyond(image, prob.pixel_mas),
        nuisances=dict(ref_obs=ref_obs, gain=np.insert(np.asarray(values["cal.gain"], float), ref_obs, 1.0), offset_mas=np.insert(np.asarray(values["cal.offset"], float).reshape(-1, 2), ref_obs, 0.0, axis=0),
                       sigma=sigma, held_from=args.nuisances_from, gain_obs0_gauge=gain, offset_obs0_gauge_mas=offset, flux_obs0_gauge_jy=flux_gauge),
        hyperparameters=dict(hyper, sigma_field=float(values["field.sigma"]), latent_rms=float(np.sqrt(np.mean(np.asarray(values["field.latent"]) ** 2)))),
        timing=dict(setup_s=setup_s, fit_s=fit_s, total_s=None, stages=[{k: s[k] for k in ("name", "steps", "rounds", "wall_s", "s_per_step", "eval_s", "stopped_because", "fit_converged", "c8_converged")} for s in stages]),
        converged=bool(all(s["c8_converged"] for s in stages)), stages=stages, command="fit", complete=True, args={k: v for k, v in vars(args).items() if k != "fn"},
        npix=prob.npix, pixel_mas=prob.pixel_mas, chi2_engine=prob.chi2_info["engine"], toeplitz_info=prob.chi2_info, mean_clean=clean_info, versions=mx.versions(),
        at_bound=va.at_bound_report(values, prob.priors),
        offset_bound=va.offset_bound_report(values["cal.offset"], prob.offset_bound_mas, prob.offset_bound_requested, bound_obs=prob.offset_bound_obs, ref_obs=ref_obs),
        dead_pixels=dict(base=va.dead_pixel_report(image, prob.pixel_mas), tapered=va.dead_pixel_report(tapered, prob.pixel_mas)),
        chi2_by_ddid=mx.chi2_table(sc, "fitted s"), dots=None,
    )
    if hasattr(va, "dots_report"):
        res["dots"] = va.dots_report(image, tapered, prob.pixel_mas)
    else:
        res["dots_note"] = "virgil_bridge.alma.dots_report is not in this checkout (vv#195)"
    say(f"raw chi2/N {res['chi2_raw_over_n']:.4f}; -ln L {res['nll']:.2f}; flux {res['flux_jy']:.5f} Jy ({res['flux_beyond_1p8_arcsec_jy']:.2e} Jy beyond {vg.OUTER_ARCSEC} arcsec); "
        f"field sigma {res['hyperparameters']['sigma_field']:.3f}")
    if res["at_bound"]["warning"]:
        say(res["at_bound"]["warning"])
    res["timing"]["total_s"] = time.time() - t_start
    mx.dump(out / "result.json", res)  # before the dirty images: the numbers are not lost if they fail
    if not args.no_dirty:
        for e, img in va.residual_dirty_images(data, prob.scene, values, prob.nuis, chunk_size=args.chunk_size).items():
            np.save(out / f"resid_dirty_obs{e}.npy", img.astype(np.float32))
    res["timing"]["total_s"] = time.time() - t_start
    mx.dump(out / "result.json", res)
    say(f"wrote {out}")
    return res


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="directory of ddid_NN.npz (no default, so nothing runs on data by accident)")
    p.add_argument("--ddids", type=int, nargs="*", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--length-r-mas", type=float, default=40.0, help="radial correlation length of the field (mas, fixed)")
    p.add_argument("--length-theta-rad", type=float, default=1.0, help="azimuthal correlation length (rad, fixed)")
    p.add_argument("--nuisances-from", default=None, help="a config R fit's values.npz (or its directory): stage 1 holds its gains, offsets and error scales, and starts the flux at its flux")
    p.add_argument("--mean-from-clean", default=None, help="alma_imlup_clean.py output directory: the field's mean is the log of its restored image (floored at --mean-floor of its peak); also the beam")
    p.add_argument("--clean-dir", default=None, help="a CLEAN output directory for its beam only (image_clean_beam.npy)")
    p.add_argument("--mean-floor", type=float, default=vg.GP["mean_floor"], help="floor of the restored CLEAN template, as a fraction of its peak")
    p.add_argument("--npix", type=int, default=vg.GP["npix"])
    p.add_argument("--pixel-mas", type=float, default=vg.GP["pixel_mas"])
    p.add_argument("--n-r", type=int, default=vg.GP["n_r"], help="radial nodes of the polar grid (about 15 mas apart at 160)")
    p.add_argument("--n-theta", type=int, default=vg.GP["n_theta"])
    p.add_argument("--r-max-mas", type=float, default=vg.GP["r_max_mas"])
    p.add_argument("--inc", type=float, default=vg.GP["inc_deg"], help="disc inclination (deg, fixed): DSHARP, Huang et al. 2018")
    p.add_argument("--pa", type=float, default=vg.GP["pa_deg"], help="disc major-axis position angle (deg, fixed)")
    p.add_argument("--ref-obs", type=int, default=0, help="the observation whose gain is 1 and offset 0")
    p.add_argument("--offset-bound-mas", type=float, default=va.TOEPLITZ_OFFSET_BOUND_MAS, help="box of the offsets' uniform prior, inside the Toeplitz precompute")
    p.add_argument("--max-steps", type=int, default=10000, help="step budget per stage (a stage stops earlier when C8 holds)")
    p.add_argument("--time-limit", type=float, default=5400.0, help="seconds for both stages together (stage 1 may use half); the report follows")
    p.add_argument("--gtol", type=float, default=1e-4, help="fit's gradient test; restarts tighten it until C8 holds")
    p.add_argument("--chunk-size", type=int, default=va.CHUNK, help="points per model evaluation of every score")
    p.add_argument("--no-dirty", action="store_true", help="skip the residual dirty maps")
    p.set_defaults(fn=lambda a: fit(a, mx.load(a)))
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
