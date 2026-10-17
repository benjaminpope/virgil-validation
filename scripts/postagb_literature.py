"""Fetch the published figures and images the post-AGB page compares against.

    python scripts/postagb_literature.py [--targets IW_Car ...] [--root DIR]

Figures come from the arXiv HTML render of De Prins et al. (2026,
arXiv:2605.13445v2), reproduced with the authors' permission (see
DATA_ATTRIBUTION.md), and are written to ``docs/imaging/postagb/literature/``
together with ``figures.json`` (file -> figure number and caption stem). If the
Zenodo record has been staged by ``postagb_data.py fetch``, each target's
ORGANIC image FITS is also saved as ``<target>_organic.npz`` (image, pixel
scale in mas, header values) for quantitative comparison.
"""

import argparse
import json
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
        npz = save_organic_fits(t, args.out)
        print("ORGANIC FITS ->", npz or "not staged (run postagb_data.py fetch)")
    manifest_path.write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
