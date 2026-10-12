"""Render the "Imaging contests: gallery" page (docs/contest_gallery.md).

    python scripts/contest_gallery.py [--data DIR] [--private DIR]

For each dataset in contests/gallery.yml, one row: virgil's best-fit image
beside the truth, where we hold one, at a matched field of view, orientation
(North up, East left), colour map and stretch. The images come from the
campaign members' npz files (scripts/contest_images.py) and from truths
rebuilt from the contest papers' models (scripts/score_2004.py). The caption
gives the raw χ²/N against the quoted errors and, where the truth is held, our
score with the contest's own metric.

Paper figures (contests/figures.yml) are linked, not shown, unless their
licence allows republication. ``--private DIR`` also writes DIR/index.html,
outside the repository, which embeds the crops made by
scripts/contest_figures.py for local viewing only.

NumPy, SciPy and matplotlib only: no JAX.
"""

import argparse
import base64
import html
import importlib.util
import io
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets" / "contests"
PAGE = ROOT / "docs" / "contest_gallery.md"

NPIX = 161  # display grid, odd so that the centre is a pixel
PANEL_IN = 2.3  # inches per panel
DPI = 80
CMAP = "magma"
SOFT = 0.02  # stretch: asinh(I / (SOFT peak)) / asinh(1 / SOFT), the same for every panel
ENCLOSED = 0.97  # the field holds this fraction of each image's flux
FAILED_SCALE = 2.0  # error scale above which a fit counts as failed


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- members ---------------------------------------------------------------


def members(results, label):
    """The campaign members of ``label``: npz contents plus the V² and closure
    χ²/N parsed from the member's text log. Later directories win."""
    latest = {}
    for directory in results:
        for path in pathlib.Path(directory).glob(f"{label}_m[0-9]*.npz"):
            if re.fullmatch(rf"{re.escape(label)}_m\d+\.npz", path.name):
                latest[path.name] = path
    out = []
    for path in sorted(latest.values()):
        d = np.load(path)
        if "ref_image" not in d:
            continue
        m = {
            "member": int(re.search(r"_m(\d+)\.npz$", path.name).group(1)),
            "image": np.asarray(d["ref_image"], float),
            "fov": float(d["ref_fov"]),
            "star": bool(d["star"]),
            "log_z": float(d["best_log_z"]),
            "chi2": float(d["best_chi2_red"]),
            "scale": float(d["error_scale"]),
            "path": path,
        }
        log = path.with_suffix(".txt")
        if log.exists():
            text = log.read_text()
            for key in ("chi2_v2", "chi2_cp"):
                found = re.search(rf"'{key}': ([0-9.eE+-]+)", text)
                m[key] = float(found.group(1)) if found else None
        out.append(m)
    return out


def best_member(group, keep=1.5):
    """The kept member (χ²/N within ``keep`` x max(best, 1), as
    combine_ensemble) with the highest evidence; returns (member, n_kept)."""
    chi2 = np.array([m["chi2"] for m in group])
    best = np.nanmin(chi2)
    kept = [m for m in group if np.isfinite(m["chi2"]) and m["chi2"] <= keep * max(best, 1.0)]
    return max(kept, key=lambda m: m["log_z"]), len(kept)


# --- grids -------------------------------------------------------------------


def on_grid(image, fov, half, npix=NPIX):
    """``image`` (``fov`` mas across; column 0 East, row 0 North, centre at
    (n − 1)/2) sampled on an ``npix`` grid spanning ±``half`` mas in the same
    orientation, bilinearly, smoothing first when the new pixels are larger.
    Flux per new pixel, so unit sum in gives unit sum over the full field."""
    from scipy.ndimage import gaussian_filter, map_coordinates

    n = image.shape[0]
    old, new = fov / n, 2 * half / npix
    if new > old:
        image = gaussian_filter(image, sigma=0.5 * new / old, mode="constant")
    k = (n - 1) / 2 + (np.arange(npix) - (npix - 1) / 2) * new / old
    rows, cols = np.meshgrid(k, k, indexing="ij")
    return map_coordinates(image, [rows, cols], order=1, mode="constant", cval=0.0) * (new / old) ** 2


def enclosing_radius(image, fov, fraction=ENCLOSED):
    """Radius (mas) about the image centre holding ``fraction`` of its
    positive flux."""
    n = image.shape[0]
    c = (np.arange(n) - (n - 1) / 2) * fov / n
    r = np.hypot(*np.meshgrid(c, c))
    flux = np.clip(image, 0, None).ravel()
    order = np.argsort(r.ravel())
    cum = np.cumsum(flux[order]) / flux.sum()
    return float(r.ravel()[order][np.searchsorted(cum, fraction)])


def nice(x):
    """The next of 1, 1.5, 2, 3, 5 x 10^k at or above ``x``."""
    k = 10 ** np.floor(np.log10(x))
    return float(next(s * k for s in (1, 1.5, 2, 3, 5, 10) if s * k >= x))


# --- truths ------------------------------------------------------------------


def truth_2004_data2(entry, fov):
    """The data2 truth rebuilt from the paper with the parameters fitted to
    the data (score_2004.FITTED), and our σ/peak against it."""
    s = _load_script("score_2004")
    f = s.FITTED
    ref = s.truth_image(f["ratio"], f["u_ld"], f["spot_contrast"])
    box = np.zeros_like(ref, bool)
    rows, cols = np.nonzero(ref > 0)
    box[rows.min() - 10 : rows.max() + 11, cols.min() - 10 : cols.max() + 11] = True
    moved, _ = s.aligned(s.resample(entry, fov, s.NPIX + 2 * s.MAX_SHIFT), ref)
    sp = s.sigma_over_peak(moved, ref, box)[0]
    published = ", ".join(f"{k} {v:.3f}" for k, v in sorted(s.PUBLISHED.items(), key=lambda kv: kv[1]))
    score = f"σ/peak {sp:.3f} against the rebuilt truth (Lawson et al. 2004, Eq. 2; lower is better; published: {published})"
    return ref, s.NPIX * s.PIXEL, score


def markers_2018(ax):
    """The published planet position and disk major-axis PA (Mérand et al.
    2018, §2), drawn on our image: the ring radii are not published."""
    pa, sep = np.radians(27.0), 3.75
    ax.plot(sep * np.sin(pa), sep * np.cos(pa), "o", mfc="none", mec="c", ms=9, mew=1.2)
    t = np.radians(44.2)
    r = np.array([-20, 20])
    ax.plot(r * np.sin(t), r * np.cos(t), "c--", lw=0.8)


# --- rendering -----------------------------------------------------------------


def panel(ax, image, half, title, peak=None):
    img = np.clip(image, 0, None)
    peak = img.max() if peak is None else peak
    ax.imshow(np.arcsinh(img / (SOFT * peak)) / np.arcsinh(1 / SOFT), origin="upper", extent=[half, -half, -half, half], cmap=CMAP, vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_xlim(half, -half)
    ax.set_ylim(-half, half)


def save_png(fig, path):
    """A compact PNG: 256-colour palette, optimised."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    plt.close(fig)
    buf.seek(0)
    PILImage.open(buf).convert("RGB").quantize(256).save(path, optimize=True)


def render_row(row, best, truth):
    """virgil's image, and the truth beside it when we hold one, on one field
    (``row['half']`` mas, or one holding ENCLOSED of each image's flux)."""
    image, fov = best["image"], best["fov"]
    radius = enclosing_radius(image, fov)
    if truth is not None:
        radius = max(radius, enclosing_radius(truth[0], truth[1]))
    half = float(row.get("half") or nice(1.1 * radius))
    n = 1 if truth is None else 2
    fig, axes = plt.subplots(1, n, figsize=(n * PANEL_IN, PANEL_IN + 0.35), sharey=True, squeeze=False)
    axes = axes[0]
    panel(axes[0], on_grid(image, fov, half), half, "virgil")
    if row["truth"] == "markers_2018":
        markers_2018(axes[0])
    if truth is not None:
        panel(axes[1], on_grid(truth[0], truth[1], half), half, "truth (rebuilt)")
    axes[0].set_ylabel("ΔDec (mas)", fontsize=8)
    for ax in axes:
        ax.set_xlabel("ΔRA (mas)", fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(5, symmetric=True))
        ax.yaxis.set_major_locator(MaxNLocator(5, symmetric=True))
    fig.tight_layout(pad=0.4)
    return fig, half


# --- page --------------------------------------------------------------------

BADGES = {
    "imaged": "imaged",
    "failed": "failed fit (s > 2)",
    "not-imaged": "not imaged",
    "no-truth": "no truth",
    "truth": "truth rebuilt",
    "partial": "truth partial",
}


def badge(key):
    return f'<span class="cg-badge cg-{key}">{BADGES[key]}</span>'


def row_panels(paper, row):
    """The paper panels listed for ``row`` (by label, or by name for a row
    not imaged); a panel without ``rows`` belongs to every row of its year."""
    keys = {row.get("label"), row["name"]}
    return [p for p in (paper or {}).get("panels") or [] if not p.get("rows") or keys & set(p["rows"])]


def panel_ref(p):
    what = p["role"] if p["role"] == "truth" else f"{p['role']} {p.get('method') or ''}".strip()
    where = f"{p['figure']} ({p['panel']})" if p.get("panel") else p["figure"]
    return f"{what}, {where}"


def paper_links(year, figures, manifest):
    paper = figures["papers"].get(year)
    contest = manifest["contests"].get(year, {})
    if not paper:
        return ""
    parts = [f"[{contest.get('bibcode', 'paper')}]({paper['ads']})"]
    if paper.get("arxiv"):
        parts.append(f"[arXiv:{paper['arxiv']}](https://arxiv.org/abs/{paper['arxiv']})")
    if contest.get("winner"):
        parts.append(f"winner: {contest['winner']}")
    return "Paper: " + "; ".join(parts) + "."


def chi2_line(best, kept, total):
    split = ""
    if best.get("chi2_v2") is not None and best.get("chi2_cp") is not None:
        split = f" (V² {best['chi2_v2']:.3g}, closure phase {best['chi2_cp']:.3g})"
    return (f"χ²/N {best['chi2']:.3g} against the quoted errors{split}; error scale {best['scale']:.2f}; "
            f"member m{best['member']} ({'star' if best['star'] else 'no star'}), {kept}/{total} kept.")


def build(data, private=None):
    config = yaml.safe_load((ROOT / "contests" / "gallery.yml").read_text())
    figures = yaml.safe_load((ROOT / "contests" / "figures.yml").read_text())
    manifest = yaml.safe_load((ROOT / "contests" / "manifest.yml").read_text())
    results = [data / "results" / r for r in config["campaign"]["results"]]
    ASSETS.mkdir(parents=True, exist_ok=True)
    camp = config["campaign"]
    lines = [
        "# Imaging contests: gallery",
        "",
        "virgil's images of the SPIE interferometric imaging contest data, beside the truth",
        "where we hold one. Every panel has North up and East left, the same colour map",
        f"and the same stretch, asinh(I / {SOFT:g} I<sub>peak</sub>), which keeps faint emission beside a",
        "bright central star visible; the panels in a row share",
        f"one field, which holds {ENCLOSED:.0%} of the flux of each image.",
        "",
        f"**Method.** virgil {camp['virgil']}, campaign 1. {camp['method']} "
        "Truths are used for scoring and display only, never for fitting.",
        "",
        "**Captions.** The raw χ²/N against the quoted errors comes first; the error scale is the",
        f"factor that brings it to 1, and a fit with a scale above {FAILED_SCALE:g} is marked failed.",
        "Where we hold the truth, the score uses that contest's own metric",
        "([scoring sources](contest_scoring_sources.md)).",
        "",
        "**Paper figures.** The contest papers' truth panels and entries are not shown: no copy",
        "we found carries a licence that allows republication. Each row links to the paper on ADS.",
        "See the [plan](plan_imaging_contests.md) for the status of each dataset.",
        "",
    ]
    private_rows = []
    for year, block in config["years"].items():
        lines += [f"## {year}: {block['title']}", ""]
        for row in block["rows"]:
            badges, caption, png, best = [], [], None, None
            group = members(results, row["label"]) if row.get("label") else []
            if group:
                best, kept = best_member(group)
                badges.append(badge("failed" if best["scale"] > FAILED_SCALE else "imaged"))
                truth, score = None, None
                if row["truth"] == "score_2004":
                    ref, ref_fov, score = truth_2004_data2(best["image"], best["fov"])
                    truth = (ref, ref_fov)
                fig, half = render_row(row, best, truth)
                png = ASSETS / f"{row['label']}.png"
                save_png(fig, png)
                caption.append(chi2_line(best, kept, len(group)))
                if score:
                    caption.append(score + ".")
            else:
                badges.append(badge("not-imaged"))
            badges.append(badge({"score_2004": "truth", "markers_2018": "partial"}.get(row["truth"], "no-truth")))
            if row.get("note"):
                caption.append(row["note"])
            refs = row_panels(figures["papers"].get(year), row)
            if refs:
                caption.append("Paper figures: " + "; ".join(panel_ref(p) for p in refs) + ".")
            lines += [f"### {year} {row['name']}", "", " ".join(badges) + f" {row['target']}.", ""]
            if png:
                lines += [f"![{year} {row['name']}: virgil and truth](assets/contests/{png.name})", ""]
            lines += [" ".join(caption), ""] if caption else []
            private_rows.append((year, row, png, caption, best))
        links = paper_links(year, figures, manifest)
        if links:
            lines += [links, ""]
    PAGE.write_text("\n".join(lines).rstrip() + "\n")
    if private:
        out = pathlib.Path(private).expanduser().resolve()
        if out.is_relative_to(ROOT):
            raise SystemExit(f"--private must be outside the repository: {out}")
        write_private(out, private_rows, figures, data)


def write_private(out, rows, figures, data):
    """A standalone HTML page, outside the repository, embedding our renders
    and the paper crops listed in contests/figures.yml. For local viewing
    only: the crops are copyrighted."""
    def img(path):
        b = base64.b64encode(pathlib.Path(path).read_bytes()).decode()
        return f'<img src="data:image/png;base64,{b}" style="height:220px;margin:2px">'

    crops = data / "figures"
    body = ["<h1>Imaging contests: gallery (private, with paper crops)</h1>"]
    for year, row, png, caption, best in rows:
        cells = [img(png)] if png else []
        paper = figures["papers"].get(year) or {}
        # Our image again on the paper's field, where the figure's axes give it.
        extents = {abs(e) for p in row_panels(paper, row) for e in (p.get("extent_mas") or [])}
        if best is not None and extents:
            half = float(max(extents))
            fig, ax = plt.subplots(figsize=(PANEL_IN, PANEL_IN + 0.35))
            panel(ax, on_grid(best["image"], best["fov"], half), half, f"virgil, paper's field ±{half:g} mas")
            ax.set_xlabel("ΔRA (mas)", fontsize=8)
            fig.tight_layout(pad=0.4)
            matched = out / f"{row['label']}_paper_field.png"
            out.mkdir(parents=True, exist_ok=True)
            save_png(fig, matched)
            cells.append(img(matched))
        for p in row_panels(paper, row):
            crop = crops / str(paper.get("sha256", "")) / f"{p['id']}.png"
            if crop.exists():
                cells.append(f"<figure style='display:inline-block;margin:2px'>{img(crop)}"
                             f"<figcaption>{html.escape(panel_ref(p))}</figcaption></figure>")
        body.append(f"<h3>{year} {html.escape(row['name'])}</h3><div>{''.join(cells)}</div><p>{html.escape(' '.join(caption))}</p>")
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text("<!doctype html><meta charset=utf-8><body style='font-family:sans-serif'>" + "".join(body))
    print(f"wrote {out / 'index.html'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="~/data/imaging_contests")
    parser.add_argument("--private", default=None, help="also write a local-only page with paper crops here")
    args = parser.parse_args()
    build(pathlib.Path(args.data).expanduser(), args.private)


if __name__ == "__main__":
    main()
