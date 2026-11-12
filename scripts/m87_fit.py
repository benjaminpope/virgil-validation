"""virgil fits of the EHT M87* fiducial days (design/m87_replication.md, stages 3-5): runs on OzSTAR GPU nodes.

    .venv/bin/python scripts/m87_fit.py --stage 3a --day 2017-04-11 --out DIR          # cheap, local (needs .venv-basis)
    .venv/bin/python scripts/m87_fit.py --stage 3b --day 2017-04-11 --out DIR          # SMC on closure invariants
    .venv/bin/python scripts/m87_fit.py --stage 4  --day 2017-04-11 --out DIR --model mring --nuis 1
    .venv/bin/python scripts/m87_fit.py --stage 5  --day 2017-04-11 --out DIR --imager mem

Stages
------
3a  Closure invariants of fixed models, virgil against BASIS's likelihood (``external_bridge/basis_worker.py``,
    needs ``.venv-basis``): data invariants and sigma alignment, model invariants at random parameter draws,
    log likelihoods. Writes ``summary_3a.json``; the maximum absolute differences are criterion 3a.
3b  The xsringauss-equivalent ``System`` on the invariants with ``virgil.smc``: likelihood (a) diagonal sigma from
    BASIS and (b) the full covariance ``C = J Sigma J^T`` (``--likelihood diag|full|both``).
4   Complex visibilities, per-scan marginalised antenna gains, ``cal.sigma`` fitted, Jeffreys priors: ``--model
    mring|composite`` with ``--nuis 0|1|2`` Gaussians; SMC ``log Z``.
5   CLEAN and RML (``--imager clean|mem|tv``) over fields of view 128/225/450 uas and starts flat/40/100 uas.

Every summary gives the raw chi^2/N on the quoted sigma first, then posterior medians and SDs, ``log Z``, REX
(``crosscheck.ring_extraction.ring_features``) on 200 posterior renders (or on the image), the git commits of
virgil-validation and virgil, and ``criteria_commit`` (the last commit of ``design/m87_replication.md``, which the run
refuses if the plan has uncommitted edits). Data: ``--vis`` (a directory of ``ddid_NN.npz`` from
``scripts/m87_uvfits_to_ms.py``, or ``eht_reading.npz``), default ``$M87_DATA/ms/<day>``; stage 3 also needs the UVFITS
(for BASIS's sigma), default under ``$M87_DATA/zenodo/gendirect``.
"""

import argparse
import glob
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA = pathlib.Path(os.environ.get("M87_DATA", pathlib.Path.home() / "data" / "m87_eht"))
DAYS = {"2017-04-11": "M87_2017_101", "2018-04-21": "M87_2018_111", "2021-04-18": "M87_2021_108"}
PLAN = "design/m87_replication.md"
NOISE = {"paper": {"noise_frac": 0.0, "floor": 0.0}, "example": {"noise_frac": 0.05, "floor": 1e-4}}


def plain(x):
    """JSON-ready: numpy and JAX values to Python."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, (np.ndarray,)) or hasattr(x, "__array__"):
        a = np.asarray(x)
        return a.tolist() if a.ndim else a.item()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def find_uvfits(day):
    hits = sorted(glob.glob(str(DATA / "zenodo" / "gendirect" / "x" / "*" / "uvfits" / f"*{DAYS[day]}*_hops_netcal_StokesI.uvfits")))
    if not hits:
        raise FileNotFoundError(f"no {DAYS[day]} StokesI UVFITS under {DATA}/zenodo/gendirect (scripts/setup_m87.sh data)")
    return hits[0]


def provenance():
    """Commits of virgil-validation and virgil, and the plan's criteria commit (refuses a dirty plan)."""
    import virgil
    from crosscheck.preregistration import criteria_commit
    from virgil_bridge.m87_fit import git_commit

    vdir = pathlib.Path(virgil.__file__).resolve().parent
    return {"vv_commit": git_commit(ROOT), "virgil_commit": git_commit(vdir.parent.parent) or None,
            "virgil_version": getattr(virgil, "__version__", None), "criteria_commit": criteria_commit(PLAN)}


def write(out, name, summary):
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    path.write_text(json.dumps(plain(summary), indent=1))
    print(f"wrote {path}")


def load_data(args):
    from virgil_bridge import m87_fit as mf

    path = args.vis or DATA / "ms" / args.day
    return mf.load_vis(path)


# ---------------------------------------------------------------------------------------------------------------
# Stage 3a
# ---------------------------------------------------------------------------------------------------------------


def stage_3a(args):
    import jax

    from external_bridge import basis_bridge
    from virgil_bridge import m87_fit as mf

    data = load_data(args)
    uvfits = args.uvfits or find_uvfits(args.day)
    rng = np.random.default_rng(args.seed)
    bounds = mf.basis_param_bounds(True)
    draws = [{k: float(rng.uniform(lo, hi)) for k, (lo, hi) in bounds.items()} for _ in range(args.draws)]
    summary = {"stage": "3a", "day": args.day, "uvfits": uvfits, "n_vis": len(data), **provenance(), "settings": {}}
    for label, noise in NOISE.items():
        if args.settings != "both" and label != args.settings:
            continue
        t0 = time.time()
        b = basis_bridge.invariants(uvfits, params=draws, seed=args.seed, **noise)
        fit = mf.ci_fit_from_data(data, full=False)
        pick = mf.match_labels(fit, b["labels"])
        sigma = mf.reorder(b["sigma"], pick)
        fit.sigma = sigma
        data_diff = float(np.max(np.abs(fit.ci - mf.reorder(b["data"], pick))))
        rows = []
        for p, model_b, model_64, ll_b in zip(draws, b["model"], b["model64"], b["loglike"]):
            system = mf.xsringauss_system(p, nuisance_blur=True)
            ci_v = np.asarray(fit.model_ci(system))
            diff64 = float(np.max(np.abs(ci_v - mf.reorder(model_64, pick))))  # criterion 3a: float64 BASIS components
            diff32 = float(np.max(np.abs(ci_v - mf.reorder(model_b, pick))))  # BaseModel.sample_vis (complex64)
            ll_v = float(fit.loglike_diag(system))
            chi2, n = fit.chi2_raw(system)
            rows.append({"max_abs_ci_diff_float64": diff64, "max_abs_ci_diff_basis_float32": diff32,
                         "loglike_basis_float32": ll_b, "loglike_virgil": ll_v, "loglike_diff": ll_v - ll_b,
                         "chi2_over_n_raw": chi2 / n})
        summary["settings"][label] = {
            **noise, "n_triangles": int(fit.tri.n_tri), "n_scans": int(fit.tri.n_scan),
            "n_independent": int(fit.tri.n_independent().sum()), "alignment_basis": b["alignment"],
            "data_ci_max_abs_diff": data_diff, "sigma_range": [float(sigma.min()), float(sigma.max())],
            "draws": rows, "max_abs_ci_diff_float64_over_draws": max(r["max_abs_ci_diff_float64"] for r in rows),
            "criterion_3a_1e-8": bool(max(r["max_abs_ci_diff_float64"] for r in rows) <= 1e-8 and data_diff <= 1e-8),
            "seconds": time.time() - t0, "stations": b["stations"],
        }
        print(label, {k: v for k, v in summary["settings"][label].items() if k not in ("draws", "stations")})
    write(args.out, "summary_3a.json", summary)
    return summary


# ---------------------------------------------------------------------------------------------------------------
# Stage 3b
# ---------------------------------------------------------------------------------------------------------------


def posterior_block(prior, out, weights, build, fov_uas=225.0, n_render=200, npix=128, seed=0):
    """Posterior summary and REX on posterior renders."""
    from virgil_bridge import m87_fit as mf

    block = {"parameters": mf.weighted_summary(out, weights)}
    try:
        imgs = mf.render_posterior(build, out, n=n_render, npix=npix, fov_uas=fov_uas, seed=seed, weights=weights)
        block["rex"] = mf.ring_summary(imgs, fov_uas)
        block["n_renders"] = int(len(imgs))
    except Exception as e:  # noqa: BLE001 - a failed render must not lose the posterior
        block["rex_error"] = f"{type(e).__name__}: {e}"
    return block


def stage_3b(args):
    import jax

    from external_bridge import basis_bridge
    from virgil_bridge import m87_fit as mf

    data = load_data(args)
    uvfits = args.uvfits or find_uvfits(args.day)
    noise = NOISE[args.noise]
    b = basis_bridge.invariants(uvfits, params=[], seed=args.seed, **noise)
    fit = mf.ci_fit_from_data(data, full=args.likelihood in ("full", "both"))
    pick = mf.match_labels(fit, b["labels"])
    fit.sigma = mf.reorder(b["sigma"], pick)
    prior = mf.basis_box_prior(nuisance_blur=args.nuisance_blur)
    summary = {"stage": "3b", "day": args.day, "noise": args.noise, "seed": args.seed, "n_particles": args.particles,
               "nuisance_blur": args.nuisance_blur, **provenance(), "runs": {}}
    build = lambda v: mf.xsringauss_system(v, nuisance_blur=args.nuisance_blur)  # noqa: E731
    for name in (["diag", "full"] if args.likelihood == "both" else [args.likelihood]):
        loglike = (lambda v: fit.loglike_diag(build(v))) if name == "diag" else (lambda v: fit.loglike_full(build(v)))
        t0 = time.time()
        res, out = mf.smc_run(prior, loglike, jax.random.PRNGKey(args.seed), n_particles=args.particles,
                              max_seconds=args.max_seconds)
        run = {"likelihood": name, "stopped": bool(res.stopped), "undersampled": bool(res.undersampled),
               "log_z": float(res.log_z), "log_z_error": float(res.log_z_error), "seconds": time.time() - t0,
               "n_stages": len(res.betas[0]) if res.betas else None}
        if out is not None:
            w = np.asarray(res.weights)
            med = {k: v["median"] for k, v in mf.weighted_summary(out, w).items()}
            med.update(prior.fixed)
            system = build(med)
            chi2, n = fit.chi2_raw(system)
            run["chi2_raw"] = {"chi2": chi2, "n": n, "chi2_over_n": chi2 / n}
            if fit.W is not None:
                c2, k = fit.chi2_full(system)
                run["chi2_full_tangent"] = {"chi2": c2, "k": k, "chi2_over_k": c2 / k}
            run.update(posterior_block(prior, out, w, build))
        summary["runs"][name] = run
        print(name, {k: run[k] for k in ("log_z", "log_z_error", "seconds")}, run.get("chi2_raw"))
    write(args.out, f"summary_3b_{args.noise}_{args.likelihood}_s{args.seed}.json", summary)
    return summary


# ---------------------------------------------------------------------------------------------------------------
# Stage 4
# ---------------------------------------------------------------------------------------------------------------


def stage_4(args):
    import jax

    from virgil_bridge import m87_fit as mf

    data = load_data(args)
    n_ddid = len(data.ddid_ids)
    flux0 = mf.flux_guess(data)
    nuis = mf.gain_nuisance(data)
    prior = mf.native_prior(args.model, args.nuis, data, flux0)
    loglike_values, term = mf.native_loglike(args.model, args.nuis, data, nuis)
    scene = mf.native_scene(args.model, args.nuis)
    summary = {"stage": 4, "day": args.day, "model": args.model, "n_nuisance_gaussians": args.nuis, "seed": args.seed,
               "n_particles": args.particles, "n_vis": len(data), "n_ddid": n_ddid, "flux_guess_jy": flux0,
               **provenance()}
    t0 = time.time()
    res, out = mf.smc_run(prior, loglike_values, jax.random.PRNGKey(args.seed), n_particles=args.particles,
                          max_seconds=args.max_seconds)
    summary.update({"stopped": bool(res.stopped), "undersampled": bool(res.undersampled), "log_z": float(res.log_z),
                    "log_z_error": float(res.log_z_error), "seconds": time.time() - t0})
    if out is not None:
        w = np.asarray(res.weights)
        stats = mf.weighted_summary(out, w)
        med = {k: v["median"] for k, v in stats.items()}
        chi2, n = mf.raw_chi2(term, nuis, med, n_ddid)
        summary["chi2_raw"] = {"chi2": chi2, "n": n, "chi2_over_n": chi2 / n}
        summary["cal_sigma_median"] = [med[f"cal.sigma_{d}"] for d in range(n_ddid)]
        build = lambda v: scene({k: v[k] for k in v})  # noqa: E731
        summary.update(posterior_block(prior, out, w, build))
        if args.model == "composite":
            summary["diameter_from_parameters_uas"] = {
                "mean_of_outer_and_inner": [med["Rp"] * (1 + med["Rn_frac"])],
                "pa_deg_north_through_east": (90.0 - np.degrees(med["phi"])) % 360.0}
        else:
            summary["diameter_from_parameters_uas"] = {"2R": 2 * med["R"]}
    print({k: summary.get(k) for k in ("log_z", "log_z_error", "chi2_raw", "seconds")})
    write(args.out, f"summary_4_{args.model}_n{args.nuis}_s{args.seed}.json", summary)
    return summary


# ---------------------------------------------------------------------------------------------------------------
# Stage 5
# ---------------------------------------------------------------------------------------------------------------


def stage_5(args):
    from virgil_bridge import m87_fit as mf

    data = load_data(args)
    fovs = [args.fov] if args.fov else list(mf.FOVS_UAS)
    starts = [args.start] if args.start else list(mf.STARTS)
    imagers = [args.imager] if args.imager != "all" else ["clean", "mem", "tv"]
    summary = {"stage": 5, "day": args.day, "n_vis": len(data), "npix": args.npix, **provenance(), "runs": []}
    images = {}
    for imager in imagers:
        for fov in fovs:
            for start in starts:
                t0 = time.time()
                label = f"{imager}_fov{int(fov)}_{start}"
                run = {"imager": imager, "fov_uas": fov, "start": start}
                try:
                    if imager == "clean":
                        r = mf.clean_run(data, start, fov, npix=args.npix)
                        from virgil.radio_clean import restore

                        scale_mas = r["pixel_scale_uas"] * 1e-3
                        r["restored"] = np.asarray(restore(r["image"], np.zeros_like(r["image"]), r["beam"], scale_mas))
                        run.update({"beam_mas": r["beam"], "stop": r["stop"], "iterations": r["iterations"]})
                        run["rex_restored"] = mf.image_ring(r["restored"], r["pixel_scale_uas"])
                        images[label + "_restored"] = r["restored"]
                    else:
                        r = mf.rml_run(data, imager, start, fov, npix=args.npix, weight=args.weight)
                        run.update({k: r[k] for k in ("chi2", "n_real", "chi2_per_n", "converged", "steps", "flux")})
                    run["rex"] = mf.image_ring(r["image"], r["pixel_scale_uas"])
                    images[label] = r["image"]
                except Exception as e:  # noqa: BLE001 - keep the other runs
                    run["error"] = f"{type(e).__name__}: {e}"
                run["seconds"] = time.time() - t0
                summary["runs"].append(run)
                print(label, {k: run.get(k) for k in ("chi2_per_n", "error", "seconds")}, (run.get("rex") or {}).get("diameter"))
    by = {}
    for run in summary["runs"]:
        d = (run.get("rex_restored") or run.get("rex") or {}).get("diameter")
        if d is not None:
            by.setdefault(run["imager"], []).append(d)
    summary["diameter_spread"] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "range": [min(v), max(v)], "n": len(v)}
                                  for k, v in by.items()}
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.imager}" + (f"_fov{int(args.fov)}" if args.fov else "") + (f"_{args.start}" if args.start else "")
    np.savez(out / f"images_5_{tag}.npz", **images)
    write(out, f"summary_5_{tag}.json", summary)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["3a", "3b", "4", "5"])
    ap.add_argument("--day", required=True, choices=sorted(DAYS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--vis", help="directory of ddid_NN.npz, or eht_reading.npz (default $M87_DATA/ms/<day>)")
    ap.add_argument("--uvfits", help="the day's UVFITS for BASIS's sigma (stage 3; default under $M87_DATA)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=3, help="3a: random parameter draws")
    ap.add_argument("--settings", choices=["paper", "example", "both"], default="both", help="3a: BASIS noise settings")
    ap.add_argument("--noise", choices=sorted(NOISE), default="paper", help="3b: BASIS noise setting")
    ap.add_argument("--likelihood", choices=["diag", "full", "both"], default="both")
    ap.add_argument("--nuisance-blur", action="store_true", help="3b: per-Gaussian blur parameters (28 free instead of 26)")
    ap.add_argument("--particles", type=int, default=2000)
    ap.add_argument("--max-seconds", type=float, default=float("inf"))
    ap.add_argument("--model", choices=["mring", "composite"], default="mring")
    ap.add_argument("--nuis", type=int, choices=[0, 1, 2], default=0)
    ap.add_argument("--imager", choices=["clean", "mem", "tv", "all"], default="all")
    ap.add_argument("--fov", type=float, choices=[128.0, 225.0, 450.0])
    ap.add_argument("--start", choices=["flat", "gauss40", "gauss100"])
    ap.add_argument("--npix", type=int, default=65)
    ap.add_argument("--weight", type=float, help="5: RML regulariser strength per real datum")
    a = ap.parse_args(argv)
    import jax

    jax.config.update("jax_enable_x64", True)
    {"3a": stage_3a, "3b": stage_3b, "4": stage_4, "5": stage_5}[a.stage](a)


if __name__ == "__main__":
    main()
