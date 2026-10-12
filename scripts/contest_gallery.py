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
licence allows republication. ``--private DIR`` also writes a small Zensical site
in DIR, outside the repository, which shows the crops made by
scripts/contest_figures.py, for local viewing only.

NumPy, SciPy and matplotlib only: no JAX.
"""

import argparse
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

INK, PAPER = "#d8d8e0", "#15151c"  # panel text and background: dark cards suit magma


def render_panel(image, fov, half, title, markers=None):
    """One square panel: ``image`` on a ±``half`` mas field, North up, East
    left, asinh stretch to its own peak. Returns the figure."""
    img = np.clip(on_grid(image, fov, half), 0, None)
    fig, ax = plt.subplots(figsize=(PANEL_IN, PANEL_IN), facecolor=PAPER)
    ax.imshow(np.arcsinh(img / (SOFT * img.max())) / np.arcsinh(1 / SOFT), origin="upper",
              extent=[half, -half, -half, half], cmap=CMAP, vmin=0, vmax=1)
    if markers:
        markers(ax)
    ax.set_xlim(half, -half)
    ax.set_ylim(-half, half)
    ax.set_title(title, fontsize=9, color=INK)
    ax.set_xlabel("ΔRA (mas)", fontsize=8, color=INK)
    ax.set_ylabel("ΔDec (mas)", fontsize=8, color=INK)
    ax.tick_params(labelsize=7, colors=INK)
    for spine in ax.spines.values():
        spine.set_color(INK)
    ax.xaxis.set_major_locator(MaxNLocator(5, symmetric=True))
    ax.yaxis.set_major_locator(MaxNLocator(5, symmetric=True))
    fig.tight_layout(pad=0.3)
    return fig


def save_png(fig, path):
    """A compact PNG: 256-colour palette, optimised."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    PILImage.open(buf).convert("RGB").quantize(256).save(path, optimize=True)


def field(row, best, truth):
    """Half-width (mas) shared by a row's panels: ``row['half']``, or one
    holding ENCLOSED of each image's flux."""
    if row.get("half"):
        return float(row["half"])
    radius = enclosing_radius(best["image"], best["fov"])
    if truth is not None:
        radius = max(radius, enclosing_radius(truth[0], truth[1]))
    return nice(1.1 * radius)


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
    parts = [f'<a href="{paper["ads"]}">{contest.get("bibcode", "paper")}</a>']
    if paper.get("arxiv"):
        parts.append(f'<a href="https://arxiv.org/abs/{paper["arxiv"]}">arXiv:{paper["arxiv"]}</a>')
    if contest.get("winner"):
        parts.append(f"winner: {html.escape(contest['winner'])}")
    return "Paper: " + "; ".join(parts) + "."


def chi2_line(best, kept, total):
    split = ""
    if best.get("chi2_v2") is not None and best.get("chi2_cp") is not None:
        split = f" (V² {best['chi2_v2']:.3g}, closure phase {best['chi2_cp']:.3g})"
    return (f"χ²/N {best['chi2']:.3g} against the quoted errors{split}; error scale {best['scale']:.2f}; "
            f"member m{best['member']} ({'star' if best['star'] else 'no star'}), {kept}/{total} kept.")


def card(title, src=None, text=None, kind="ours"):
    """One figure card: an image (linked to itself) or a placeholder text."""
    body = (f'<a href="{src}"><img src="{src}" alt="{html.escape(title)}" loading="lazy"></a>' if src
            else f'<div class="cg-placeholder">{text}</div>')
    return f'<figure class="cg-card cg-{kind}">{body}<figcaption>{html.escape(title)}</figcaption></figure>'


def intro(camp, private):
    lines = [
        "# Imaging contests: gallery" + (" (private)" if private else ""),
        "",
        "virgil's images of the SPIE interferometric imaging contest data, beside the truth where we",
        "hold one" + (" and the truth and winning entries from the contest papers." if private else "."),
        f"Our panels have North up and East left, the `{CMAP}` colour map and one stretch,",
        f"asinh(I / {SOFT:g} I<sub>peak</sub>), which keeps faint emission beside a bright central star",
        f"visible; the panels in a row share one field, holding {ENCLOSED:.0%} of each image's flux.",
        "",
        f"**Method.** virgil {camp['virgil']}, campaign 1. {camp['method']} "
        "Truths are used for scoring and display only, never for fitting.",
        "",
        "**Captions.** The raw χ²/N against the quoted errors comes first; the error scale is the",
        f"factor that brings it to 1, and a fit with a scale above {FAILED_SCALE:g} is marked failed.",
        "Where we hold the truth, the score uses that contest's own metric.",
        "",
    ]
    if private:
        lines += ['!!! warning "Local use only"',
                  "    The paper crops are copyrighted (contests/figures.yml records each licence).",
                  "    This site is built outside the repository and must not be published.",
                  "    Paper panels keep their own orientation, scale and stretch: read each card's caption.", ""]
    else:
        lines += ["**Paper figures.** The contest papers' truth panels and entries are cited, not shown: no copy",
                  "we found carries a licence that allows republication. See the",
                  "[scoring sources](contest_scoring_sources.md) and the [plan](plan_imaging_contests.md).", ""]
    return lines


def build(data, private=None):
    config = yaml.safe_load((ROOT / "contests" / "gallery.yml").read_text())
    figures = yaml.safe_load((ROOT / "contests" / "figures.yml").read_text())
    manifest = yaml.safe_load((ROOT / "contests" / "manifest.yml").read_text())
    results = [data / "results" / r for r in config["campaign"]["results"]]
    if private:
        out = pathlib.Path(private).expanduser().resolve()
        if out.is_relative_to(ROOT):
            raise SystemExit(f"--private must be outside the repository: {out}")
        img_dir = out / "docs" / "assets" / "img"
        img_dir.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    public, secret = intro(config["campaign"], False), intro(config["campaign"], True)
    for year, block in config["years"].items():
        for lines in (public, secret):
            lines += [f"## {year}: {block['title']}", ""]
        paper = figures["papers"].get(year) or {}
        for row in block["rows"]:
            badges, caption, best = [], [], None
            ours = []  # (title, file name) of our renders
            group = members(results, row["label"]) if row.get("label") else []
            if group:
                best, kept = best_member(group)
                badges.append(badge("failed" if best["scale"] > FAILED_SCALE else "imaged"))
                truth, score = None, None
                if row["truth"] == "score_2004":
                    ref, ref_fov, score = truth_2004_data2(best["image"], best["fov"])
                    truth = (ref, ref_fov)
                half = field(row, best, truth)
                marks = markers_2018 if row["truth"] == "markers_2018" else None
                save_png(render_panel(best["image"], best["fov"], half, "virgil", marks), ASSETS / f"{row['label']}_virgil.png")
                ours.append(("virgil", f"{row['label']}_virgil.png"))
                if truth is not None:
                    save_png(render_panel(truth[0], truth[1], half, "truth (rebuilt)"), ASSETS / f"{row['label']}_truth.png")
                    ours.append(("truth, rebuilt from the paper's model", f"{row['label']}_truth.png"))
                caption.append(chi2_line(best, kept, len(group)))
                if score:
                    caption.append(score + ".")
            else:
                badges.append(badge("not-imaged"))
            badges.append(badge({"score_2004": "truth", "markers_2018": "partial"}.get(row["truth"], "no-truth")))
            if row.get("note"):
                caption.append(row["note"])
            refs = row_panels(paper, row)
            head = [f"### {year} {row['name']}", "", f'<p class="cg-meta">{" ".join(badges)} {html.escape(row["target"])}.</p>', ""]
            empty = [] if group else [card("virgil", text="Not imaged yet", kind="empty")]
            # Public: our renders, and the paper panels as citations.
            cards = [card(t, f"assets/contests/{f}") for t, f in ours] + empty
            cards += [card(panel_ref(p), text=f'<a href="{paper["ads"]}">{html.escape(p["figure"])}</a><br>'
                           f'{html.escape(p.get("panel") or "")}', kind="cite") for p in refs]
            public += head + ['<div class="cg-grid">', *cards, "</div>", ""]
            if caption:
                public += [f'<p class="cg-caption">{html.escape(" ".join(caption))}</p>', ""]
            if private:
                cards = []
                for t, f in ours:
                    (img_dir / f).write_bytes((ASSETS / f).read_bytes())
                    cards.append(card(t, f"assets/img/{f}"))
                cards += empty
                # Our image again on the paper's field, where the figure's axes give it.
                extents = {abs(e) for p in refs for e in (p.get("extent_mas") or [])}
                if best is not None and extents:
                    h = float(max(extents))
                    name = f"{row['label']}_paper_field.png"
                    save_png(render_panel(best["image"], best["fov"], h, f"virgil, ±{h:g} mas"), img_dir / name)
                    cards.append(card(f"virgil on the paper's field (±{h:g} mas)", f"assets/img/{name}"))
                for p in refs:
                    crop = data / "figures" / str(paper.get("sha256", "")) / f"{p['id']}.png"
                    if crop.exists():
                        (img_dir / crop.name).write_bytes(crop.read_bytes())
                        kind = "truth" if p["role"] == "truth" else "paper"
                        cards.append(card(panel_ref(p), f"assets/img/{crop.name}", kind=kind))
                secret += head + ['<div class="cg-grid">', *cards, "</div>", ""]
                notes = [p["orientation"] for p in refs if "RIGHT" in (p.get("orientation") or "")]
                text = " ".join(caption) + (" Paper orientation: " + "; ".join(sorted(set(notes))) + "." if notes else "")
                if text:
                    secret += [f'<p class="cg-caption">{html.escape(text)}</p>', ""]
        links = paper_links(year, figures, manifest)
        if links:
            for lines in (public, secret):
                lines += [f'<p class="cg-paper">{links}</p>', ""]
    PAGE.write_text("\n".join(public).rstrip() + "\n")
    if private:
        write_private(out, secret)


def write_private(out, lines):
    """A small Zensical project in ``out`` (outside the repository): the
    gallery with the paper crops, styled as the docs site. Build or serve it
    with ``zensical serve -f OUT/mkdocs.yml``. For local viewing only."""
    docs = out / "docs"
    (docs / "index.md").write_text("\n".join(lines).rstrip() + "\n")
    (docs / "assets" / "contests.css").write_text((ROOT / "docs" / "assets" / "contests.css").read_text())
    config = {
        "site_name": "Imaging contests: private gallery",
        "site_description": "Local only: contains copyrighted paper figures",
        "theme": {"name": "material", "features": ["navigation.sections", "toc.follow"],
                  "palette": [{"media": "(prefers-color-scheme: light)", "scheme": "default",
                               "toggle": {"icon": "material/brightness-7", "name": "Dark mode"}},
                              {"media": "(prefers-color-scheme: dark)", "scheme": "slate",
                               "toggle": {"icon": "material/brightness-4", "name": "Light mode"}}]},
        "markdown_extensions": ["tables", "admonition", {"toc": {"permalink": True}}],
        "extra_css": ["assets/contests.css"],
        "nav": [{"Gallery": "index.md"}],
    }
    (out / "mkdocs.yml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    print(f"wrote {out}: serve with  zensical serve -f {out / 'mkdocs.yml'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="~/data/imaging_contests")
    parser.add_argument("--private", default=None, help="also write a local-only Zensical site with paper crops here")
    args = parser.parse_args()
    build(pathlib.Path(args.data).expanduser(), args.private)


if __name__ == "__main__":
    main()
