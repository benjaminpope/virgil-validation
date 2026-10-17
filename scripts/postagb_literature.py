"""Fetch the published figures and images the post-AGB page compares against.

    python scripts/postagb_literature.py [--targets IW_Car ...] [--root DIR]

Figures come from the arXiv HTML render of De Prins et al. (2026,
arXiv:2605.13445v2) and, for IRAS08544, the e-print source of Hillen et al. (2016,
arXiv:1603.03023), reproduced with the authors' permission (see
DATA_ATTRIBUTION.md), and are written to ``docs/imaging/postagb/literature/``
together with ``figures.json`` (file -> figure number and caption stem). If the
Zenodo record has been staged by ``postagb_data.py fetch``, each target's
ORGANIC image FITS is also saved as ``<target>_organic.npz`` (image, pixel
scale in mas, header values) for quantitative comparison.
"""

import argparse
import io
import json
import tarfile
import urllib.request
from pathlib import Path

ARXIV = "https://arxiv.org/html/2605.13445v2/"
OUT = Path(__file__).resolve().parents[1] / "docs/imaging/postagb/literature"
ZENODO_DIR = Path("~/data/postagb_pionier/literature").expanduser()

# target -> {local name: (file on the arXiv render, figure label, what it shows)}
FIGURES = {
    "IW_Car": {
        "IW_Car_disc_organic.png": (
            "IW_Car_disc_organic.png", "Fig. 4",
            "peak-normalised SPARCO+ORGANIC image of IW Car (inner and outer flux arcs)"),
        "IW_Car_disc_organic_no_sparco_secondary.png": (
            "IW_Car_disc_organic_no_sparco_secondary.png", "Fig. 2",
            "peak-normalised ORGANIC image of IW Car without a SPARCO secondary"),
        "IW_Car_sphere_vs_pionier.png": (
            "IW_Car_sphere_vs_pionier.png", "Fig. 6",
            "SPHERE/IRDIS (Andrych et al. 2023) against the PIONIER image"),
        "IW_Car_squeeze.png": (
            "organic_vs_squeeze_images_last_three.png", "Fig. 15 (Appendix D)",
            "ORGANIC (left) against SQUEEZE (right) reconstructions, IW Car among the last three targets"),
    },
}
FIGURES["IRAS08544"] = {}  # figures from the arXiv source tarball, below

# target -> (arXiv ID, {local name: (member of the e-print tarball, figure label, what it shows)}).
# Hillen et al. (2016, A&A 588, L1) has no Zenodo images: the reconstruction is a bitmap, so its
# colours are mapped back to values through the figure's own colour bar (``bitmap_to_image``).
SOURCE_FIGURES = {
    "IRAS08544": ("1603.03023", {
        "IRAS08544_hillen_image.png": (
            "IRAS08544_UD045_015mas-256pixels_531sig-contour_beam.png", "Fig. 4 (upper panel)",
            "single-star-subtracted SPARCO image of IRAS 08544-4431 with 5, 3 and 1 sigma contours"),
    }),
}

# The other seven targets: one ORGANIC image each (Fig. 3 for the first four, Fig. 4 for the last four),
# plus the SQUEEZE comparison panels of Appendix D and the rim-fit example of Fig. 5.
_SHOWS = "peak-normalised SPARCO+ORGANIC image of {}, with the median inner-rim fit (cyan), the SPARCO primary (star) and secondary (cross)"
for _t, _name, _fig in (("AI_Sco", "AI Sco", "Fig. 3"), ("EN_TrA", "EN TrA", "Fig. 3"), ("HD95767", "HD 95767", "Fig. 3"),
                        ("HD108015", "HD 108015", "Fig. 3"), ("HR4049", "HR 4049", "Fig. 4"),
                        ("IRAS15469-5311", "IRAS 15469-5311", "Fig. 4"), ("PS_Gem", "PS Gem", "Fig. 4")):
    FIGURES[_t] = {f"{_t}_disc_organic.png": (f"{_t}_disc_organic.png", _fig, _SHOWS.format(_name))}
_SQ1 = ("organic_vs_squeeze_images_first_four.png", "Fig. 14 (Appendix D)",
        "ORGANIC (left) against SQUEEZE (right) reconstructions of AI Sco, EN TrA, HD 95767 and HD 108015")
_SQ2 = ("organic_vs_squeeze_images_last_three.png", "Fig. 15 (Appendix D)",
        "ORGANIC (left) against SQUEEZE (right) reconstructions of IRAS 15469-5311 and IW Car (inner and outer arcs)")
for _t, (_remote, _label, _what) in (("AI_Sco", _SQ1), ("EN_TrA", _SQ1), ("HD95767", _SQ1), ("HD108015", _SQ1),
                                     ("IRAS15469-5311", _SQ2),):  # Fig. 15 has no panel for HR 4049 or PS Gem
    FIGURES[_t][f"squeeze_{_remote}"] = (_remote, _label, _what)
FIGURES["IRAS15469-5311"]["IRAS15469-5311_inner_rim_fit_organic.png"] = (
    "IRAS15469-5311_inner_rim_fit_organic.png", "Fig. 5",
    "example of the inner-rim ellipse fit for IRAS 15469-5311 (radial rays and 100 MCMC ellipses)")

# Table values and parameter provenance live in the docs page and the imaging script.


def download(name, dest):
    with urllib.request.urlopen(ARXIV + name, timeout=120) as r:
        data = r.read()
    if not data.startswith(b"\x89PNG"):
        raise ValueError(f"{name}: not a PNG")
    dest.write_bytes(data)


def eprint_member(arxiv_id, member):
    """One file of an arXiv e-print tarball, as bytes (read in memory; nothing is extracted to disk)."""
    with urllib.request.urlopen(f"https://arxiv.org/e-print/{arxiv_id}", timeout=300) as r:
        blob = r.read()
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as t:
        return t.extractfile(t.getmember(member)).read()


def bitmap_to_image(png_bytes, axes=(171, 650, 61, 540), cbar_x=672, half_mas=19.2):
    """Recover a peak-normalised image from a published bitmap (colour map plus overlays).

    The axes interior is ``axes`` = (x0, x1, y0, y1) in pixels (here ``half_mas`` = 19.2 mas to each
    side, East left, North up) and the colour bar runs down column ``cbar_x`` from 1 (top) to 0.
    Each pixel takes the value of the nearest colour-bar colour; pixels that match none (the
    blue sigma contours, star markers, beam and compass) are filled from their nearest good
    neighbour. The result is a stretch-limited copy of the published image, not the data: use it
    for morphology, never for photometry. Also returns the drawn beam (major, minor FWHM in mas,
    PA North to East in degrees), measured from the blue ellipse in the upper left corner, the colour map
    as an (n, 3) array from 0 to 1, and the axes interior as RGB (the image as published).
    """
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from scipy import ndimage  # noqa: PLC0415

    im = np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB")).astype(float)
    x0, x1, y0, y1 = axes
    lut = im[y0 + 1:y1 - 1, cbar_x, :]  # one row of the bar's border is dropped at each end
    val = (y1 - (y0 + 1) - np.arange(lut.shape[0]) - 0.5) / (y1 - y0)
    sub = im[y0:y1 - 1, x0:x1 - 1, :]
    dist = np.linalg.norm(sub[:, :, None, :] - lut[None, None, :, :], axis=-1)
    out, bad = val[dist.argmin(-1)], ndimage.binary_dilation(dist.min(-1) > 25, iterations=1)
    scale = 2 * half_mas / (x1 - x0)
    box = sub[:45, :60]
    blue = (box[:, :, 2] > 200) & (box[:, :, 0] < 190) & (box[:, :, 1] < 190)
    ys, xs = np.where(blue)
    ev, vec = np.linalg.eigh(np.cov(np.vstack([xs, ys])))
    major = vec[:, 1]  # (right = West, down = South)
    # a filled ellipse of full axis length L has variance (L/2)^2 / 4 = L^2 / 16 along that axis
    beam = dict(major_mas=float(4 * np.sqrt(ev[1]) * scale), minor_mas=float(4 * np.sqrt(ev[0]) * scale),
                pa_deg=float(np.degrees(np.arctan2(-major[0], -major[1])) % 180.0))
    bad[:45, :60] = True  # beam ellipse
    bad[-70:, -70:] = True  # N/E compass
    out[bad] = np.nan
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    crop = np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB"))[y0:y1 - 1, x0:x1 - 1]
    return out[tuple(idx)], scale, beam, lut[::-1][:: max(1, lut.shape[0] // 256)], crop


def fetch_source_figures(target, out):
    arxiv_id, figs = SOURCE_FIGURES[target]
    manifest = {}
    for local, (member, label, what) in figs.items():
        png = eprint_member(arxiv_id, member)
        (out / local).write_bytes(png)  # the whole published panel, as in the arXiv source
        img, scale, beam, lut, crop = bitmap_to_image(png)
        import numpy as np  # noqa: PLC0415

        np.savez_compressed(out / f"{target}_hillen_image.npz", image=img, pixel_mas=scale, beam=json.dumps(beam), cmap=lut / 255.0,
                 crop=crop,
                 source=f"arXiv:{arxiv_id} {label}")
        manifest[local] = dict(arxiv=arxiv_id, figure=label, source=f"https://arxiv.org/e-print/{arxiv_id}:{member}",
                               shows=what)
        print("wrote", out / local, "beam", beam)
    return manifest


def save_organic_fits(target, out):
    """Save the Zenodo ORGANIC FITS for ``target`` as an npz, if staged. Returns the path or None."""
    from astropy.io import fits  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    cands = sorted(ZENODO_DIR.glob("*/fits/*.fits")) + sorted(ZENODO_DIR.glob("*/*/fits/*.fits"))
    key = target.replace("_", "").lower()
    hit = [f for f in cands if key in f.stem.replace("_", "").replace(" ", "").lower()]
    if not hit:
        return None
    with fits.open(hit[0]) as h:
        hdu = next(x for x in h if x.data is not None)
        header = {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in hdu.header.items()
                  if k and k not in ("COMMENT", "HISTORY")}
        np.savez(out / f"{target}_organic.npz", data=np.asarray(hdu.data, float),
                 header=json.dumps(header), source=hit[0].name)
    return out / f"{target}_organic.npz"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--targets", nargs="+", default=["IW_Car"], choices=list(FIGURES))
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "figures.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for t in args.targets:
        for local, (remote, label, what) in FIGURES[t].items():
            download(remote, args.out / local)
            manifest[local] = dict(arxiv="2605.13445v2", figure=label, source=ARXIV + remote, shows=what)
            print("wrote", args.out / local)
        if t in SOURCE_FIGURES:
            manifest.update(fetch_source_figures(t, args.out))
            continue
        npz = save_organic_fits(t, args.out)
        print("ORGANIC FITS ->", npz or "not staged (run postagb_data.py fetch)")
    manifest_path.write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
