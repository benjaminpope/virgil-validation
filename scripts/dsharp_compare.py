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


def one(fit_dir, ref_path, pbcor=False, fov=CENTRAL_ARCSEC):
    """``(ours, ref, wcs, beam, stats)`` for one fit directory and released image (``pbcor``: the image is primary-beam corrected, see
    ``ds.our_image_in_beam``). The statistics use only the pixels inside our image's footprint on the released grid."""
    image = pathlib.Path(fit_dir) / "image.fits"
    ours, ref, wcs, beam = ds.our_image_in_beam(image, ref_path, pbcor=pbcor)
    foot = ds.our_footprint(image, ref_path)
    r = central_radius_pix(wcs, fov)
    stats = dict(whole=ds.compare_stats(ours, ref, footprint=foot), central=ds.compare_stats(ours, ref, radius_pix=r, footprint=foot),
                 beam_mas_mas_deg=list(beam), pixel_mas=float(abs(wcs.wcs.cdelt[0]) * 3.6e6), pbcor_applied=bool(pbcor))
    res = pathlib.Path(fit_dir) / "result.json"
    if res.exists():
        j = json.loads(res.read_text())
        stats["fit"] = {k: j.get(k) for k in ("flux_jy", "chi2n_raw", "converged", "stopped_because", "n", "wall_s")}
    return ours, ref, wcs, beam, stats


def _centre(wcs):
    """The pixel (x, y), 0-based, of the image's reference point (the phase centre), from CRPIX."""
    return float(wcs.wcs.crpix[0]) - 1.0, float(wcs.wcs.crpix[1]) - 1.0


def _extent(wcs, shape, half_arcsec):
    """The pixel box (x0, x1, y0, y1) of a field of half-width ``half_arcsec`` centred on the reference point (CRPIX)."""
    ny, nx = shape
    cx_, cy_ = _centre(wcs)
    hx, hy = half_arcsec / (abs(wcs.wcs.cdelt[0]) * 3600.0), half_arcsec / (abs(wcs.wcs.cdelt[1]) * 3600.0)
    return int(max(round(cx_ - hx), 0)), int(min(round(cx_ + hx), nx)), int(max(round(cy_ - hy), 0)), int(min(round(cy_ + hy), ny))


def sky_extent(wcs, box):
    """``imshow`` extent ``[left, right, bottom, top]`` in arcsec offsets (``origin="lower"``) of the pixel box: the edges of the cut pixels
    (``x0 - 0.5`` and ``x1 - 0.5``) relative to the reference point, times the signed ``CDELT``, so East is left for the usual negative ``CDELT1``."""
    x0, x1, y0, y1 = box
    cx_, cy_ = _centre(wcs)
    dx, dy = float(wcs.wcs.cdelt[0]) * 3600.0, float(wcs.wcs.cdelt[1]) * 3600.0
    return [(x0 - 0.5 - cx_) * dx, (x1 - 0.5 - cx_) * dx, (y0 - 0.5 - cy_) * dy, (y1 - 0.5 - cy_) * dy]


def radial_profile(img, wcs, rmax_arcsec=CENTRAL_ARCSEC, nbins=40):
    """``(r_arcsec, mean)``: the azimuthal mean of ``img`` (NaNs skipped) in rings about the reference point."""
    cx_, cy_ = _centre(wcs)
    yy, xx = np.indices(img.shape)
    r = np.hypot((xx - cx_) * abs(wcs.wcs.cdelt[0]), (yy - cy_) * abs(wcs.wcs.cdelt[1])) * 3600.0
    edges = np.linspace(0.0, rmax_arcsec, nbins + 1)
    ok = np.isfinite(img)
    idx = np.digitize(r, edges) - 1
    mean = np.array([img[ok & (idx == i)].mean() if (ok & (idx == i)).any() else np.nan for i in range(nbins)])
    return 0.5 * (edges[1:] + edges[:-1]), mean


def panels(axs, ours, ref, wcs, title, half_arcsec=CENTRAL_ARCSEC):
    """Draw ours, the released image and their difference (Jy/beam) on three axes with a shared arcsinh stretch (north up, east left), and on a
    fourth their azimuthal profiles about the phase centre."""
    box = _extent(wcs, ref.shape, half_arcsec)
    x0, x1, y0, y1 = box
    sl = (slice(y0, y1), slice(x0, x1))
    vmax = float(np.nanmax(ref[sl]))
    soft = 0.02 * vmax
    norm = lambda a: np.arcsinh(np.clip(a, 0, None) / soft)  # noqa: E731
    ext = sky_extent(wcs, box)
    top = float(norm(np.array(vmax)))
    for ax, img, lab in ((axs[0], ours, "virgil"), (axs[1], ref, "ALMA release")):
        ax.imshow(norm(img[sl]), origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=top)
        ax.set_title(f"{title}: {lab}", fontsize=9)
    d = (ours - ref)[sl]
    lim = float(np.nanmax(np.abs(d))) or 1.0
    axs[2].imshow(d, origin="lower", extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axs[2].set_title("virgil - ALMA (Jy/beam)", fontsize=9)
    for ax in axs[:3]:
        ax.set_xlim(max(ext[0], ext[1]), min(ext[0], ext[1]))  # East left whatever the sign of CDELT1
        ax.set_xlabel("RA offset (arcsec)", fontsize=8)
    axs[0].set_ylabel("Dec offset (arcsec)", fontsize=8)
    for img, lab in ((ours, "virgil"), (ref, "ALMA release")):
        r, m = radial_profile(img, wcs, half_arcsec)
        axs[3].plot(r, m, label=lab)
    axs[3].set_xlabel("radius (arcsec)", fontsize=8)
    axs[3].set_ylabel("azimuthal mean (Jy/beam)", fontsize=8)
    axs[3].set_title("radial profile", fontsize=9)
    axs[3].legend(fontsize=7)


def _fig(rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(rows, 4, figsize=(14.5, 3.7 * rows), squeeze=False, constrained_layout=True)
    return plt, fig, axs


def cmd_compare(args):
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ours, ref, wcs, beam, stats = one(args.fit, args.ref, ds.target(args.target)["image"]["pbcor"], fov=args.fov)
    stats["target"] = args.target
    (out / "compare.json").write_text(json.dumps(stats, indent=1))
    plt, fig, axs = _fig(1)
    panels(axs[0], ours, ref, wcs, ds.target(args.target)["name"], half_arcsec=args.fov)
    fig.savefig(out / "compare.png", dpi=130)
    plt.close(fig)
    c = stats["central"]
    print(f"{args.target}: central {args.fov:g}\": rms(diff)/peak {c['rms_over_peak']:.3f}, correlation {c['correlation']:.3f}, flux ratio {c['flux_ratio']:.3f}, peak ratio {c['peak_ratio']:.3f}")


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
    q.add_argument("--fov", type=float, default=CENTRAL_ARCSEC, help="half-width in arcsec of the plotted field and of the statistics radius")
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
