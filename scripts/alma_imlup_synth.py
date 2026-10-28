"""IM Lup: synthetic truth for the arms of the better-than-MPoL pass (Track A, ``design/alma_imlup_p3.md``).

    python scripts/alma_imlup_synth.py make --data DIR --out DIR [--truth-from FITDIR | --truth-image FILE.npy] [--truth-blur-mas 15] [--flux 0.25] [--seed 0] [--no-noise]
    python scripts/alma_imlup_synth.py run --arm {maxent,gp} --data SYNDIR --truth FILE.npy --out DIR (--clean-dir DIR | --beam BMAJ BMIN PA) -- ARGS...
    python scripts/alma_imlup_synth.py score --truth FILE.npy --fit DIR --out FILE.json (--clean-dir DIR | --beam BMAJ BMIN PA)

``make`` writes ``truth.npy`` (the supplied image, default the classic-MEM arm's ``image.npy`` blurred to 15 mas and normalised to 0.25 Jy, so it has no dots) and the synthetic ``ddid_NN.npz`` at the
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


def cmd_make(args):
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if bool(args.truth_from) == bool(args.truth_image):
        raise SystemExit("exactly one of --truth-from FITDIR and --truth-image FILE.npy")
    base = np.load(pathlib.Path(args.truth_from) / "image.npy" if args.truth_from else args.truth_image).astype(float)
    truth = vy.truth_image(base, args.pixel_mas, args.truth_blur_mas, args.flux)
    np.save(out / "truth.npy", truth.astype(np.float32))
    files = va.ddid_files(args.data)
    say(f"truth: {truth.shape} x {args.pixel_mas:g} mas, blurred to {args.truth_blur_mas:g} mas, {args.flux:g} Jy; {len(files)} ddids")
    written = vy.make_data(files, out, truth, args.pixel_mas, seed=args.seed, chunk_size=args.chunk_size, noise=not args.no_noise)
    mx.dump(out / "synth_make.json", dict(command="make", truth_from=args.truth_from or args.truth_image, truth_blur_mas=args.truth_blur_mas, flux_jy=args.flux, pixel_mas=args.pixel_mas, seed=args.seed,
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
    origin = json_load(made).get("truth_from") if made.exists() else None
    sc.update(truth_origin=origin, truth_origin_note="a candidate arm's own image biases the NRMSE contest towards that arm: also score on an independent truth (CLEAN model, analytic ring)" if origin and "image" in str(origin) else None, command="synth", fit=str(fit_dir), truth=str(args.truth), config=res.get("config"), workflow=res.get("workflow"), converged=res.get("converged"),
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
    p = sub.add_parser("make")
    p.add_argument("--data", required=True, help="the real ddid_NN.npz directory: its uv and weights are kept")
    p.add_argument("--out", required=True)
    p.add_argument("--truth-from", default=None, help="a fit directory (image.npy, Jy/arcsec^2): the classic-MEM arm's, by default recipe")
    p.add_argument("--truth-image", default=None, help="an image .npy instead")
    p.add_argument("--truth-blur-mas", type=float, default=vy.TRUTH_BLUR_MAS)
    p.add_argument("--flux", type=float, default=vy.TRUTH_FLUX_JY)
    p.add_argument("--pixel-mas", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-noise", action="store_true")
    p.add_argument("--chunk-size", type=int, default=va.CHUNK)
    p.set_defaults(fn=cmd_make)
    p = sub.add_parser("run")
    p.add_argument("--arm", choices=["maxent", "gp"], required=True)
    p.add_argument("--data", required=True, help="the synthetic directory of `make`")
    p.add_argument("--truth", required=True)
    p.add_argument("--out", required=True)
    beam(p)
    p.add_argument("rest", nargs=argparse.REMAINDER, help="after --: the arguments of the arm's fit")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("score")
    p.add_argument("--truth", required=True)
    p.add_argument("--fit", required=True)
    p.add_argument("--out", required=True)
    beam(p)
    p.set_defaults(fn=cmd_score)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
