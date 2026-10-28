"""IM Lup: synthetic truth for the arms of the better-than-MPoL pass (Track A, ``design/alma_imlup_p3.md``).

    python scripts/alma_imlup_synth.py cells
    python scripts/alma_imlup_synth.py make --data DIR --out DIR --truth {ring_gap,mpol_export,clean_model} [--truth-file export.npz] [--clean-dir DIR] [--seed 0] [--no-noise]
    python scripts/alma_imlup_synth.py make --data DIR --out DIR (--dev-truth-from FITDIR | --dev-truth-image FILE.npy) [--truth-blur-mas 15] [--flux 0.25]   # DEVELOPMENT only: never scored for the gate
    python scripts/alma_imlup_synth.py run --arm {maxent,gp} --data SYNDIR --truth FILE.npy --coverage {full,sb} --out DIR (--clean-dir DIR | --beam BMAJ BMIN PA) -- ARGS...
    python scripts/alma_imlup_synth.py score --truth FILE.npy --fit DIR --out FILE.json (--clean-dir DIR | --beam BMAJ BMIN PA) [--coverage ID]

``cells`` lists the registered truth x coverage cells of ``alma_start.CRITERIA`` (the array the cluster scripts run). ``make --truth ID`` writes the registered truth (``CRITERIA['synth']['truths']``: a parametric ring-and-gap disk, MPoL's exported image, our CLEAN model;
the file truths are refused unless their sha256 is the registered one) blurred to the registered 15 mas at 0.25 Jy, and the synthetic ``ddid_NN.npz`` at the
real uv and weights: model visibilities plus Gaussian noise of variance 1 / w per component, no gains, offsets or error scales. ``run`` fits one arm to them (``alma_imlup_maxent.py fit`` or
``alma_imlup_gp.py``, ARGS passed on after ``--data`` and ``--out``: ``--config R --taper-mas 30 ...``, ``--config F --reg mem:1e-9 ...``) and scores it; ``score`` scores a finished fit. The scores
are ``synth.json``: NRMSE against the truth at 0.5 and 1.0 of the CLEAN beam, ``dots_report`` of the base image, the flux error. Every path is an argument.
"""

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import alma_imlup_maxent as mx  # noqa: E402  (say, dump, load; sets float64)
import virgil_bridge.alma as va  # noqa: E402
import virgil_bridge.alma_start as vs  # noqa: E402
import virgil_bridge.alma_synth as vy  # noqa: E402

say = mx.say


def beam_of(args):
    if args.beam:
        return tuple(float(x) for x in args.beam)
    if args.clean_dir:
        return tuple(vs.read_clean(args.clean_dir)[2])
    raise SystemExit("the CLEAN beam is needed: --clean-dir DIR (alma_imlup_clean.py output) or --beam BMAJ_MAS BMIN_MAS PA_DEG")


def cmd_cells(args):
    for t, c in vs.cells():
        print(t, c)


def cmd_make(args):
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = bool(args.dev_truth_from or args.dev_truth_image)
    if dev == bool(args.truth):
        raise SystemExit("exactly one of --truth ID (registered, scored for the gate) and --dev-truth-from FITDIR | --dev-truth-image FILE.npy (development only)")
    if dev:
        if bool(args.dev_truth_from) == bool(args.dev_truth_image):
            raise SystemExit("exactly one of --dev-truth-from FITDIR and --dev-truth-image FILE.npy")
        base = np.load(pathlib.Path(args.dev_truth_from) / "image.npy" if args.dev_truth_from else args.dev_truth_image).astype(float)
        truth = vy.truth_image(base, args.pixel_mas, args.truth_blur_mas, args.flux)
        truth_id, source_sha, args.truth_from = ("dev_mem" if args.dev_truth_from else "dev_image"), None, args.dev_truth_from or args.dev_truth_image
    else:
        truth, source_sha, spec = vy.registered_truth(args.truth, pixel_mas=args.pixel_mas, file=args.truth_file, clean_dir=args.clean_dir)
        truth_id, args.truth_blur_mas, args.flux, args.truth_from = args.truth, spec["blur_mas"], vs.CRITERIA["synth"]["flux_jy"], args.truth_file or args.clean_dir or "parametric"
    np.save(out / "truth.npy", truth.astype(np.float32))
    files = va.ddid_files(args.data)
    say(f"truth: {truth.shape} x {args.pixel_mas:g} mas, blurred to {args.truth_blur_mas:g} mas, {args.flux:g} Jy; {len(files)} ddids")
    written = vy.make_data(files, out, truth, args.pixel_mas, seed=args.seed, chunk_size=args.chunk_size, noise=not args.no_noise)
    mx.dump(out / "synth_make.json", dict(command="make", truth_id=truth_id, development=dev, truth_source_sha256=source_sha, truth_from=args.truth_from, truth_blur_mas=args.truth_blur_mas, flux_jy=args.flux, pixel_mas=args.pixel_mas, seed=args.seed,
                                          noise=not args.no_noise, n_ddid=len(written), source_data=str(args.data), versions=mx.versions()))
    say(f"wrote {out} ({len(written)} ddids)")


def score_fit(args, fit_dir, out_json):
    fit_dir = pathlib.Path(fit_dir)
    truth = np.load(args.truth).astype(float)
    image = np.load(fit_dir / "image.npy").astype(float)
    if image.shape != truth.shape:
        raise SystemExit(f"{fit_dir}/image.npy is {image.shape}, the truth {truth.shape}")
    res = json_load(fit_dir / "result.json")
    px = float(res["pixel_mas"])
    sc = vy.score_image(image, truth, px, beam_of(args))
    made = pathlib.Path(args.truth).parent / "synth_make.json"
    mk = json_load(made) if made.exists() else {}
    origin = mk.get("truth_from") if mk.get("development") else None
    cov = getattr(args, "coverage", None)
    if cov is not None and cov not in [c["id"] for c in vs.CRITERIA["synth"]["coverages"]]:
        raise SystemExit(f"coverage {cov!r} is not registered in CRITERIA")
    sc.update(truth_id=mk.get("truth_id"), coverage=cov, development=bool(mk.get("development", True)), truth_source_sha256=mk.get("truth_source_sha256"), truth_origin=origin, truth_origin_note="a candidate arm's own image biases the NRMSE contest towards that arm: also score on an independent truth (CLEAN model, analytic ring)" if origin and "image" in str(origin) else None, command="synth", fit=str(fit_dir), truth=str(args.truth), config=res.get("config"), workflow=res.get("workflow"), converged=res.get("converged"),
              chi2n_raw=(res.get("summary") or {}).get("chi2n_raw", res.get("chi2_raw_over_n")), versions=mx.versions())
    mx.dump(out_json, sc)
    say(f"synth: NRMSE {sc['nrmse']} (0.5 / 1.0 CLEAN beam), N_eff ratio {sc['n_eff_ratio']:.3f}, base dead in disk {sc['base_dead_in_disk_1e4']:.2%}, flux error {sc['flux_error']:+.2%}; wrote {out_json}")
    return sc


def json_load(path):
    import json

    return json.loads(pathlib.Path(path).read_text())


def cmd_score(args):
    score_fit(args, args.fit, args.out)


def cmd_run(args):
    rest = [a for a in args.rest if a != "--"]
    cov = {c["id"]: c for c in vs.CRITERIA["synth"]["coverages"]}.get(args.coverage)
    if cov is None:
        raise SystemExit(f"coverage {args.coverage!r} is not registered in CRITERIA ({[c['id'] for c in vs.CRITERIA['synth']['coverages']]})")
    if cov["ddids"] is not None:
        rest += ["--ddids", *[str(d) for d in cov["ddids"]]]
    out = pathlib.Path(args.out)
    if args.arm == "maxent":
        mx.main(["fit", "--data", args.data, "--out", str(out), *rest])
    else:
        import alma_imlup_gp as gp

        gp.main(["--data", args.data, "--out", str(out), *rest])
    score_fit(args, out, out / "synth.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    beam = lambda p: (p.add_argument("--clean-dir", default=None, help="alma_imlup_clean.py output (clean.json): its restoring beam"),  # noqa: E731
                      p.add_argument("--beam", nargs=3, type=float, default=None, metavar=("BMAJ_MAS", "BMIN_MAS", "PA_DEG"), help="the CLEAN beam, when no --clean-dir"))
    p = sub.add_parser("cells")
    p.set_defaults(fn=cmd_cells)
    p = sub.add_parser("make")
    p.add_argument("--data", required=True, help="the real ddid_NN.npz directory: its uv and weights are kept")
    p.add_argument("--out", required=True)
    p.add_argument("--truth", default=None, help="a truth registered in CRITERIA (ring_gap, mpol_export, clean_model)")
    p.add_argument("--truth-file", default=None, help="mpol_export: MPoL's export.npz (its sha256 must be the registered one)")
    p.add_argument("--clean-dir", default=None, help="clean_model: alma_imlup_clean's output (model.fits, registered sha256)")
    p.add_argument("--dev-truth-from", default=None, help="DEVELOPMENT only: a fit directory (image.npy, Jy/arcsec^2), e.g. the classic-MEM arm's; the gate refuses it")
    p.add_argument("--dev-truth-image", default=None, help="DEVELOPMENT only: an image .npy; the gate refuses it")
    p.add_argument("--truth-blur-mas", type=float, default=vy.TRUTH_BLUR_MAS, help="development truths only: registered truths use CRITERIA's blur")
    p.add_argument("--flux", type=float, default=vy.TRUTH_FLUX_JY, help="development truths only")
    p.add_argument("--pixel-mas", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-noise", action="store_true")
    p.add_argument("--chunk-size", type=int, default=va.CHUNK)
    p.set_defaults(fn=cmd_make)
    p = sub.add_parser("run")
    p.add_argument("--arm", choices=["maxent", "gp"], required=True)
    p.add_argument("--data", required=True, help="the synthetic directory of `make`")
    p.add_argument("--truth", required=True)
    p.add_argument("--coverage", required=True, help="a coverage registered in CRITERIA: full, or sb (ddids 0-17, the short-baseline observations)")
    p.add_argument("--out", required=True)
    beam(p)
    p.add_argument("rest", nargs=argparse.REMAINDER, help="after --: the arguments of the arm's fit")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("score")
    p.add_argument("--truth", required=True)
    p.add_argument("--fit", required=True)
    p.add_argument("--coverage", default=None)
    p.add_argument("--out", required=True)
    beam(p)
    p.set_defaults(fn=cmd_score)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
