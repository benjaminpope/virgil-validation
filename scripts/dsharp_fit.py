"""The virgil MaxEnt fit of one DSHARP / HL Tau target (config R of ``virgil_bridge.alma``; GPU).

    dsharp_fit.py --target KEY --data DIR --out DIR [--info info.json] [--shat DIR] [--npix N] [--pixel-mas P]
                  [--taper-mas 30] [--chi2 direct|toeplitz] [--ref-obs auto|N] [--max-steps N] [--time-limit S]

``--data`` holds ``ddid_NN.npz`` (the vv format of ``external_bridge/alma_worker.py read``). The model is the one of the IM Lup validation
(MPoL's objective at our optimiser: a pixel image with a Gaussian taper and MPoL's entropy; per-observation gains and phase-centre offsets, a
per-ddid error scale), fitted in one stage on all data from a 1" Gaussian whose flux is the short-baseline mean. The reference
observation (gain 1, offset 0) is the one with the most weight. The start is the Gaussian raised to MPoL's entropy floor, and MPoL's F_tot is the start flux.
Outputs: ``image.fits`` (Jy/arcsec2, the model's sky: the fitted image convolved with the taper, at the field's phase centre from
``--info``), ``image_base.npy`` (before the taper), ``image_tapered.npy`` (= image.fits), ``values.npz`` (checkpointed each round) and ``result.json`` (flux, gains, offsets and error scales per observation, chi^2 per ddid, convergence).
The comparison with the released ALMA image is ``dsharp_compare.py``. IM Lup's own validation fits stay in ``alma_imlup_maxent.py``.
"""

import argparse
import json
import os
import pathlib
import re
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crosscheck import dsharp as ds  # noqa: E402


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


def heaviest_obs(obs, weight):
    """The observation with the largest total weight: the gauge (gain 1, offset 0) of the fit."""
    ids = np.unique(obs)
    return int(ids[np.argmax([float(weight[obs == o].sum()) for o in ids])])


def provenance(env=None):
    """The commits under test, as AGENTS.md requires of a result: virgil's (the read-only snapshot a job pinned, whose directory is named for the commit,
    else the installed one) and virgil-validation's (the checkout's, else the job's ``PIN_COMMIT``)."""
    from evidence import meta

    env = os.environ if env is None else env
    src = dict(meta.virgil_source())
    pinned = env.get("PIN_VIRGIL_SRC")
    if pinned and re.fullmatch(r"[0-9a-f]{40}", pathlib.Path(pinned).parent.name):
        src = dict(commit=pathlib.Path(pinned).parent.name, url=pinned, dirty=False)  # a snapshot is read-only
    return dict(virgil_commit=src.get("commit"), virgil_dirty=src.get("dirty"), virgil_source=src.get("url"),
                validation_commit=meta.own_commit() or env.get("PIN_COMMIT"), pins=env.get("PIN_DESC"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", required=True, choices=ds.CAMPAIGN)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--info", default=None, help="dsharp_ms_info.py's JSON: the phase centre of the image's WCS (omitted: 0, 0 and a warning)")
    p.add_argument("--shat", default=None, help="directory of ddid_NN.json (alma_imlup_read): s_d starts at the model-free scale")
    p.add_argument("--npix", type=int, default=None)
    p.add_argument("--pixel-mas", type=float, default=None)
    p.add_argument("--taper-mas", type=float, default=30.0)
    p.add_argument("--chi2", choices=("direct", "toeplitz"), default="direct")
    p.add_argument("--ref-obs", default="auto")
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--time-limit", type=float, default=18000.0)
    p.add_argument("--gtol", type=float, default=1e-4)
    p.add_argument("--chunk-size", type=int, default=10**6)
    p.add_argument("--offset-bound-mas", type=float, default=None)
    args = p.parse_args(argv)

    import jax

    jax.config.update("jax_enable_x64", True)
    import virgil_bridge.alma as va
    from crosscheck import alma_maxent as cx

    t = ds.target(args.target)
    npix, px = args.npix or t["npix"], args.pixel_mas or t["pixel_mas"]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = va.load_dir(args.data)
    n = len(data.u)
    obs, weight = np.asarray(data.obs), np.asarray(data.weight)
    ref = heaviest_obs(obs, weight) if args.ref_obs == "auto" else int(args.ref_obs)
    flux0 = ds.auto_flux(np.asarray(data.u), np.asarray(data.v), np.asarray(data.vis), weight)
    lam = va.lam_r(n)
    say(f"{t['name']}: {n:,} points, {len(np.unique(np.asarray(data.ddid)))} ddids, {len(np.unique(obs))} observations; reference observation {ref}; "
        f"start flux {flux0:.4f} Jy; npix {npix} x {px} mas; lam_R {lam:.5g}")
    # MPoL's entropy is (1/F_tot) sum I ln(I/p): with F_tot fixed at IM Lup's 0.253 Jy its weight on the unit-sum image grows as F/0.253, so each target gets
    # its own (the start flux): lam_M then means the same on every disc and band.
    prob = va.build_problem(data, "R", lam=lam, npix=npix, pixel_mas=px, ref_obs=ref, chi2=args.chi2, taper_mas=args.taper_mas,
                            chunk_size=args.chunk_size, offset_bound_mas=args.offset_bound_mas, f_tot=flux0)
    shat = va.read_shat(args.shat, int(np.asarray(data.ddid).max()) + 1) if args.shat else None
    values = va.start_values(prob, flux=flux0, fwhm_mas=1000.0, shat=shat)
    # the Gaussian's outer pixels are 1e-8 to 1e-15 of its peak and the softmax gradient cannot revive them: raise them to MPoL's entropy floor
    import jax.numpy as jnp

    eta, flux_start, floor_info = va.floor_start(values, prob.npix, prob.pixel_mas, va.MPOL["prior"])
    values["log_brightness"], values["cal.flux"] = jnp.asarray(eta), jnp.asarray(flux_start)
    say(f"start: 1\" Gaussian raised to the floor {va.MPOL['prior']:g} Jy/arcsec^2 at {floor_info['n_raised']:,} pixels "
        f"(+{floor_info['flux_added_jy']:.5f} Jy on {floor_info['flux_before_jy']:.5f} Jy)")
    ra = dec = 0.0
    if args.info:
        pc = json.loads(pathlib.Path(args.info).read_text())["phase_centre"]
        ra, dec = pc["ra_deg"], pc["dec_deg"]
    else:
        say("warning: no --info: image.fits is on the sky position (0, 0); the comparison with the ALMA image will not register")
    t0 = time.time()

    def on_round(vals, info, traj, rnd):  # a checkpoint after every round, so a walltime kill keeps the last point
        np.savez(out / "values.npz", **{k: np.asarray(v) for k, v in vals.items()})
        say(f"round {rnd}: {info['steps']} steps, stop {info['stop']}, loss {info['loss']:.3f}; values.npz saved")

    vals, info, traj, why = va.run_fit_c8(prob, values, max_steps=args.max_steps, time_limit=args.time_limit, log=say, gtol=args.gtol, on_round=on_round)
    wall = time.time() - t0
    vals = {k: np.asarray(v) for k, v in vals.items()}
    conv = va.convergence(traj, info)
    sc = va.score(data, prob.scene, vals, prob.nuis, chunk_size=args.chunk_size)
    np.savez(out / "values.npz", **vals)
    base = va.image_of(vals, prob.npix, prob.pixel_mas)  # the hidden image, before the model's Gaussian kernel
    sky = va.tapered_image(base, prob.pixel_mas, args.taper_mas)  # the sky the model's visibilities are of (the kernel is in the forward model)
    np.save(out / "image_base.npy", base.astype(np.float32))
    np.save(out / "image_tapered.npy", sky.astype(np.float32))
    cx.write_fits(out / "image.fits", sky, prob.pixel_mas, ra, dec, OBJECT=t["name"], ORIGIN=f"virgil R, {args.target}", TAPERMAS=float(args.taper_mas))
    res = dict(
        target=args.target, name=t["name"], n=int(n), npix=prob.npix, pixel_mas=prob.pixel_mas, taper_mas=args.taper_mas, chi2_engine=prob.chi2_info["engine"],
        image_fits="the tapered image (the model's sky; image_base.npy is the hidden image before the kernel)", f_tot_jy=flux0, start_floor=floor_info,
        lam_R=lam, ref_obs=ref, start_flux_jy=flux0, flux_jy=float(vals["cal.flux"]), gain=vals.get("cal.gain"), offset_mas=vals.get("cal.offset"),
        sigma=vals.get("cal.sigma"), obs_ids=np.asarray(data.obs_ids), ddid_ids=np.asarray(data.ddid_ids), converged=conv["converged"], convergence=conv,
        stopped_because=why, steps=info.get("steps"),
        nll=sc["nll"], chi2_raw=sc["chi2_raw"], chi2_fit=sc["chi2_fit"], chi2n_raw=sc["chi2_raw"] / (2 * n), by_ddid=sc["by_ddid"],
        dead_pixels=dict(base=va.dead_pixel_report(base, prob.pixel_mas), tapered=va.dead_pixel_report(sky, prob.pixel_mas)),
        at_bound=va.at_bound_report(vals, prob.priors),
        offset_bound=va.offset_bound_report(vals.get("cal.offset"), prob.offset_bound_mas, prob.offset_bound_requested, bound_obs=prob.offset_bound_obs, ref_obs=ref),
        provenance=provenance(), phase_centre_deg=[ra, dec], wall_s=wall, args={k: v for k, v in vars(args).items()},
    )
    (out / "result.json").write_text(json.dumps(jsonable(res), indent=1))
    say(f"flux {res['flux_jy']:.4f} Jy, chi2/2N (s = 1) {res['chi2n_raw']:.4f}, converged {conv['converged']} ({why}), wall {wall:.0f} s; wrote {out}")


if __name__ == "__main__":
    main()
