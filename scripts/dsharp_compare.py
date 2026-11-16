"""Compare virgil's image of a DSHARP / HL Tau target with the released ALMA image. Runs locally on downloaded outputs (no GPU, no virgil).

    dsharp_compare.py compare --target KEY --fit DIR --ref ALMA.fits --out DIR
    dsharp_compare.py preview --out FILE.png --b6 FITDIR REF6.fits --b7 FITDIR REF7.fits      # HL Tau, one quick look for each band

``--fit`` is the output directory of ``dsharp_fit.py`` (``image.fits`` in Jy/arcsec2). Our image is convolved with the beam of the released
image (header BMAJ, BMIN, BPA), put on its pixel grid and shown beside it in Jy/beam with one shared arcsinh stretch, with their difference.
``compare`` also writes ``compare.json``: rms of the difference over the released peak, correlation, flux and peak ratios (whole field and the
central 2"), the released image's noise and the fit's flux, convergence and chi^2/N. ``preview`` is the quick look Ben uses to choose a band.
"""

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crosscheck import dsharp as ds  # noqa: E402

CENTRAL_ARCSEC = 2.0


def central_radius_pix(wcs, arcsec=CENTRAL_ARCSEC):
    px = abs(wcs.wcs.cdelt[0]) * 3600.0
    return arcsec / px


def one(fit_dir, ref_path, pbcor=False):
    """``(ours, ref, wcs, beam, stats)`` for one fit directory and released image (``pbcor``: the image is primary-beam corrected, see
    ``ds.our_image_in_beam``). The statistics use only the pixels inside our image's footprint on the released grid."""
    image = pathlib.Path(fit_dir) / "image.fits"
    ours, ref, wcs, beam = ds.our_image_in_beam(image, ref_path, pbcor=pbcor)
    foot = ds.our_footprint(image, ref_path)
    r = central_radius_pix(wcs)
    stats = dict(whole=ds.compare_stats(ours, ref, footprint=foot), central=ds.compare_stats(ours, ref, radius_pix=r, footprint=foot),
                 beam_mas_mas_deg=list(beam), pixel_mas=float(abs(wcs.wcs.cdelt[0]) * 3.6e6), pbcor_applied=bool(pbcor))
    res = pathlib.Path(fit_dir) / "result.json"
    if res.exists():
        j = json.loads(res.read_text())
        stats["fit"] = {k: j.get(k) for k in ("flux_jy", "chi2n_raw", "converged", "stopped_because", "n", "wall_s")}
    return ours, ref, wcs, beam, stats


def _extent(wcs, shape, half_arcsec):
    """The pixel box (x0, x1, y0, y1) of a centred field of half-width ``half_arcsec``."""
    ny, nx = shape
    h = half_arcsec / (abs(wcs.wcs.cdelt[0]) * 3600.0)
    return int(max(nx / 2 - h, 0)), int(min(nx / 2 + h, nx)), int(max(ny / 2 - h, 0)), int(min(ny / 2 + h, ny))


def panels(axs, ours, ref, wcs, title, half_arcsec=CENTRAL_ARCSEC):
    """Draw ours, the released image and their difference (Jy/beam) on three axes, with a shared arcsinh stretch; north up, east left."""
    x0, x1, y0, y1 = _extent(wcs, ref.shape, half_arcsec)
    sl = (slice(y0, y1), slice(x0, x1))
    vmax = float(np.nanmax(ref[sl]))
    soft = 0.02 * vmax
    norm = lambda a: np.arcsinh(np.clip(a, 0, None) / soft)  # noqa: E731
    ext = [(x1 - 0.5 - ref.shape[1] / 2) * abs(wcs.wcs.cdelt[0]) * 3600, (x0 - 0.5 - ref.shape[1] / 2) * abs(wcs.wcs.cdelt[0]) * 3600,
           (y0 - 0.5 - ref.shape[0] / 2) * abs(wcs.wcs.cdelt[1]) * 3600, (y1 - 0.5 - ref.shape[0] / 2) * abs(wcs.wcs.cdelt[1]) * 3600]
    top = float(norm(np.array(vmax)))
    for ax, img, lab in ((axs[0], ours, "virgil"), (axs[1], ref, "ALMA release")):
        ax.imshow(norm(img[sl]), origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=top)
        ax.set_title(f"{title}: {lab}", fontsize=9)
    d = (ours - ref)[sl]
    lim = float(np.nanmax(np.abs(d))) or 1.0
    axs[2].imshow(d, origin="lower", extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axs[2].set_title("virgil - ALMA (Jy/beam)", fontsize=9)
    for ax in axs:
        ax.set_xlabel("RA offset (arcsec)", fontsize=8)
    axs[0].set_ylabel("Dec offset (arcsec)", fontsize=8)


def _fig(rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(rows, 3, figsize=(11, 3.7 * rows), squeeze=False, constrained_layout=True)
    return plt, fig, axs


def cmd_compare(args):
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ours, ref, wcs, beam, stats = one(args.fit, args.ref, ds.target(args.target)["image"]["pbcor"])
    stats["target"] = args.target
    (out / "compare.json").write_text(json.dumps(stats, indent=1))
    plt, fig, axs = _fig(1)
    panels(axs[0], ours, ref, wcs, ds.target(args.target)["name"])
    fig.savefig(out / "compare.png", dpi=130)
    plt.close(fig)
    c = stats["central"]
    print(f"{args.target}: central 2\": rms(diff)/peak {c['rms_over_peak']:.3f}, correlation {c['correlation']:.3f}, flux ratio {c['flux_ratio']:.3f}, peak ratio {c['peak_ratio']:.3f}")


def cmd_preview(args):
    rows = [("hltau_b6", *args.b6), ("hltau_b7", *args.b7)]
    plt, fig, axs = _fig(len(rows))
    summary = {}
    for i, (key, fit, ref) in enumerate(rows):
        ours, rf, wcs, beam, stats = one(fit, ref, ds.target(key)["image"]["pbcor"])
        panels(axs[i], ours, rf, wcs, ds.target(key)["name"])
        summary[key] = stats
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    for k, s in summary.items():
        c = s["central"]
        print(f"{k}: central 2\": rms(diff)/peak {c['rms_over_peak']:.3f}, correlation {c['correlation']:.3f}, flux ratio {c['flux_ratio']:.3f}; fit {s.get('fit')}")
    print(f"wrote {out} and {out.with_suffix('.json')}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("compare")
    q.add_argument("--target", required=True, choices=ds.CAMPAIGN)
    q.add_argument("--fit", required=True)
    q.add_argument("--ref", required=True)
    q.add_argument("--out", required=True)
    q.set_defaults(fn=cmd_compare)
    q = sub.add_parser("preview")
    q.add_argument("--out", required=True)
    q.add_argument("--b6", nargs=2, required=True, metavar=("FITDIR", "REF"))
    q.add_argument("--b7", nargs=2, required=True, metavar=("FITDIR", "REF"))
    q.set_defaults(fn=cmd_preview)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
