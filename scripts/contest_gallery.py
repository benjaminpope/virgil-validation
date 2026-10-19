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
import csv
import io
import json
import pathlib
import re
import sys

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
OVERFIT = 0.5  # raw χ²/N below this: errors overestimated, or an overfit
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
        m.update(log_split(path.with_suffix(".txt")))
        out.append(m)
    return out


def log_split(log):
    """The V² and closure-phase χ²/N from a run's text log (None when absent)."""
    text = log.read_text() if log.exists() else ""
    out = {}
    for key in ("chi2_v2", "chi2_cp"):
        found = re.search(rf"'{key}': ([0-9.eE+-]+)", text)
        out[key] = float(found.group(1)) if found else None
    return out


def best_member(group, keep=1.5):
    """The kept member (χ²/N within ``keep`` x max(best, 1), as
    combine_ensemble) with the highest evidence; returns (member, n_kept)."""
    chi2 = np.array([m["chi2"] for m in group])
    best = np.nanmin(chi2)
    kept = [m for m in group if np.isfinite(m["chi2"]) and m["chi2"] <= keep * max(best, 1.0)]
    return max(kept, key=lambda m: m["log_z"]), len(kept)


# --- informed alternatives ------------------------------------------------------


def closeness(chi2):
    """How far a raw χ²/N is from the expected 1, as a factor ≥ 1: a fit at 0.3
    overfits or has overestimated errors, and is no better than one at 3."""
    return max(chi2, 1 / chi2)


def informed_runs(directory, prefix):
    """The advice-seeded runs ``<prefix>_*`` in ``directory`` that left an image
    (the GP, regularised and CLEAN runs; the sweeps are ensemble means on their
    own fields and are skipped), each with its raw χ²/N. Informed: the method
    was chosen after seeing the truth or the contest page, so none is blind."""
    out = []
    for run in sorted(pathlib.Path(directory).glob(f"{prefix}_*")):
        path = run / f"{run.name}.npz"
        if not path.exists():
            continue
        d = np.load(path)
        if "image" in d:
            image = d["image"]
        elif "image_iso" in d:  # the GP's isotropic and anisotropic kernels: the higher peak evidence
            aniso = "image_aniso" in d and np.max(d["log_z_aniso"]) > np.max(d["log_z_iso"])
            image = d["image_aniso" if aniso else "image_iso"]
        elif "restored" in d:
            image = d["restored"]
        elif "images" in d and "index" in d:
            image = d["images"][json.loads(str(d["index"]))["discrepancy"]]
        else:
            continue
        log = run / f"{run.name}.txt"
        text = log.read_text() if log.exists() else ""
        if "raw_chi2_red" in d:
            chi2 = float(d["raw_chi2_red"])
        elif "chi2_red" in d and d["chi2_red"].ndim == 0:
            chi2 = float(d["chi2_red"])
        elif found := re.search(r"raw chi2/N=([0-9.eE+-]+)", text):
            chi2 = float(found.group(1))
        else:
            continue
        name = re.sub(r"_r\d+$", "", run.name[len(prefix) + 1 :])
        out.append({"name": name, "image": np.asarray(image, float), "chi2": chi2, **log_split(log),
                    "fov": float(d["npix"]) * float(d["pixel_scale_mas"])})
    return out


def choose_informed(runs, blind):
    """The informed run closest to χ²/N = 1, if the row has no blind image or
    it is closer to 1 than the blind fit; else None. Against a blind fit, a run
    below ``OVERFIT`` is not preferred: both overfit or have overestimated errors."""
    runs = [r for r in runs if blind is None or r["chi2"] >= OVERFIT]
    if not runs:
        return None
    best = min(runs, key=lambda r: closeness(r["chi2"]))
    return best if blind is None or closeness(best["chi2"]) < closeness(blind["chi2"]) else None


def ladder(directory, label):
    """The 2004 informed parametric ladder's winner for ``label`` (its image,
    raw χ²/N split, rung), or None when there is no result."""
    path = pathlib.Path(directory) / f"{label}_parametric"
    if not path.with_suffix(".json").exists():
        return None
    info = json.loads(path.with_suffix(".json").read_text())
    win = next(r for r in info["ladder"] if r["winner"])
    d = np.load(path.with_suffix(".npz"))
    return {"image": np.asarray(d["ref_image"], float), "fov": float(d["ref_fov"]), "chi2": win["chi2_red"],
            "chi2_v2": win["v2"], "chi2_cp": win["cp"], "rung": win["label"]}


def advice_notes():
    """contests/fit_advice.csv keyed by (year, name): (advice, date)."""
    with (ROOT / "contests" / "fit_advice.csv").open(newline="", encoding="utf-8") as f:
        return {(r["year"], r["name"]): (r["advice"], r["updated"][:10]) for r in csv.DictReader(f)}


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
    the data (score_2004.FITTED), our σ/peak against it, and the published
    entries' scores as text."""
    s = _load_script("score_2004")
    f = s.FITTED
    ref = s.truth_image(f["ratio"], f["u_ld"], f["spot_contrast"])
    box = np.zeros_like(ref, bool)
    rows, cols = np.nonzero(ref > 0)
    box[rows.min() - 10 : rows.max() + 11, cols.min() - 10 : cols.max() + 11] = True
    moved, _ = s.aligned(s.resample(entry, fov, s.NPIX + 2 * s.MAX_SHIFT), ref)
    sp = s.sigma_over_peak(moved, ref, box)[0]
    published = ", ".join(f"{k} {v:.3f}" for k, v in sorted(s.PUBLISHED.items(), key=lambda kv: kv[1]))
    return ref, s.NPIX * s.PIXEL, sp, published


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


def field(row, shown):
    """Half-width (mas) shared by a row's panels: ``row['half']``, or one
    holding ENCLOSED of the flux of every ``(image, fov)`` in ``shown``."""
    if row.get("half"):
        return float(row["half"])
    return nice(1.1 * max(enclosing_radius(image, fov) for image, fov in shown))


# --- page --------------------------------------------------------------------

BADGES = {
    "imaged": "imaged",
    "failed": "failed fit (s > 2)",
    "not-imaged": "not imaged",
    "no-truth": "no truth",
    "truth": "truth rebuilt",
    "partial": "truth partial",
    "informed": "informed fit, not blind",
    "not-imaged-blind": "not imaged blind",
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
        "Every two years from 2004, the interferometry community has run a blind image-reconstruction",
        "contest: simulated or real optical interferometer data are released without the true image,",
        "and teams submit their reconstructions. This page runs virgil on those data and shows what it",
        "recovers, beside the truth where we hold one" + (" and the truth and winning entries from the contest papers." if private else "."),
        "A fit counts only as far as it reaches the data (the raw χ²/N against the quoted errors) and,",
        "where there is a truth, its score with the contest's own metric.",
        "",
        "**How to read the panels.** North up and East left, the "
        f"`{CMAP}` colour map and one stretch,",
        f"asinh(I / {SOFT:g} I<sub>peak</sub>), which keeps faint emission beside a bright central star",
        f"visible; the panels in a row share one field, holding {ENCLOSED:.0%} of every image's flux.",
        "",
        f"**Blind fits.** virgil {camp['virgil']}, campaign 1. {camp['method']} "
        "Truths are used for scoring and display only, never for fitting.",
        "",
        "**Informed fits.** Where Ben's visual review of the blind images asked for a different model "
        "(smoother or sparser priors, a halo, point sources), those runs and the 2004 parametric ladder "
        "use knowledge of what the object is. They are badged *informed*: the method was chosen after seeing "
        "the truth or the contest page, so they are not blind reconstructions, and they are shown only where "
        "their raw χ²/N is closer to 1 than the blind fit's, or where the dataset has no blind fit.",
        "",
        "**Captions.** The raw χ²/N against the quoted errors comes first; the error scale is the",
        f"factor that brings it to 1, and a fit with a scale above {FAILED_SCALE:g} is marked failed.",
        "The images were made before virgil's field and beam scaling change (virgil#361, in the "
        "current pin); a rerun may change them.",
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


def verdict(best):
    if best is None:
        return "not imaged blind"
    if best["scale"] > FAILED_SCALE:
        return "failed"
    return "fits, χ²/N below 0.5" if best["chi2"] < OVERFIT else "fits"


def status_table(status):
    """The page's summary: one line per dataset with the blind raw χ²/N and
    verdict, and the informed fit that is closer to 1, if any."""
    imaged = [s for s in status if s[2] is not None]
    failed = [s for s in imaged if s[2]["scale"] > FAILED_SCALE]
    lines = ["## Where we stand", "",
             f"{len(imaged)} of {len(status)} datasets are imaged blind; {len(imaged) - len(failed)} of those fit the "
             f"quoted errors (error scale at most {FAILED_SCALE:g}) and {len(failed)} fail. "
             "A fit that reaches the data is not yet a good image: the visual review under each dataset "
             "notes where the recovery breaks up (chains of dots, blobs).", "",
             "| Contest | Dataset | Blind raw χ²/N | Blind | Informed raw χ²/N |", "|---|---|---|---|---|"]
    for year, row, best, alt, lad in status:
        informed = ", ".join(f"{x['chi2']:.3g} ({what})" for x, what in ((alt, "advice-seeded"), (lad, "parametric ladder")) if x)
        chi2 = f"{best['chi2']:.3g}" if best else "–"
        lines.append(f"| {year} | {row['name']} | {chi2} | {verdict(best)} | {informed or '–'} |")
    return lines + [""]


def build(data, private=None):
    config = yaml.safe_load((ROOT / "contests" / "gallery.yml").read_text())
    figures = yaml.safe_load((ROOT / "contests" / "figures.yml").read_text())
    manifest = yaml.safe_load((ROOT / "contests" / "manifest.yml").read_text())
    results = [data / "results" / r for r in config["campaign"]["results"]]
    adv = config["informed"]["advice"]
    adv_dir = data / "results" / adv["results"]
    ladder_dir = data / "results" / config["informed"]["ladder"]["results"]
    reviews, status = advice_notes(), []
    if private:
        out = pathlib.Path(private).expanduser().resolve()
        if out.is_relative_to(ROOT):
            raise SystemExit(f"--private must be outside the repository: {out}")
        img_dir = out / "docs" / "assets" / "img"
        img_dir.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    public, secret = intro(config["campaign"], False), intro(config["campaign"], True)
    n_intro = len(public)
    for year, block in config["years"].items():
        for lines in (public, secret):
            lines += [f"## {year}: {block['title']}", ""]
        paper = figures["papers"].get(year) or {}
        for row in block["rows"]:
            badges, caption, best = [], [], None
            ours = []  # (title, file name) of our renders
            group = members(results, row["label"]) if row.get("label") else []
            prefix = row.get("advice") or row.get("label")
            runs = informed_runs(adv_dir, prefix) if prefix else []
            lad = ladder(ladder_dir, row["label"]) if row.get("label", "").startswith("2004_") else None
            lad_ok = lad is not None and lad["chi2"] ** 0.5 <= FAILED_SCALE
            truth, sigma, kept = None, None, 0
            if group:
                best, kept = best_member(group)
            alt = choose_informed(runs, best)
            shown = [(m["image"], m["fov"]) for m in (best, alt) if m] + ([(lad["image"], lad["fov"])] if lad_ok else [])
            if row["truth"] == "score_2004":
                src = best or alt or lad
                ref, ref_fov, sigma, published = truth_2004_data2(src["image"], src["fov"])
                truth = (ref, ref_fov)
                shown.append(truth)
            half = field(row, shown) if shown else None
            marks = markers_2018 if row["truth"] == "markers_2018" else None
            key = row.get("label") or prefix
            if best:
                badges.append(badge("failed" if best["scale"] > FAILED_SCALE else "imaged"))
                save_png(render_panel(best["image"], best["fov"], half, "virgil, blind", marks), ASSETS / f"{key}_virgil.png")
                ours.append(("virgil, blind", f"{key}_virgil.png"))
                caption.append("Blind: " + chi2_line(best, kept, len(group)))
                if sigma is not None:
                    caption.append(f"σ/peak {sigma:.3f} against the rebuilt truth (Lawson et al. 2004, Eq. 2; lower is better).")
            elif alt:
                badges.append(badge("not-imaged-blind"))
            else:
                badges.append(badge("not-imaged"))
            if alt:
                badges.append(badge("informed"))
                save_png(render_panel(alt["image"], alt["fov"], half, "virgil, informed", marks), ASSETS / f"{key}_informed.png")
                ours.append((f"virgil, informed ({alt['name'].replace('_', ' ')})", f"{key}_informed.png"))
                split = (f" (V² {alt['chi2_v2']:.3g}, closure phase {alt['chi2_cp']:.3g})"
                         if alt.get("chi2_v2") is not None and alt.get("chi2_cp") is not None else "")
                versus = (f"; closer to 1 than the blind fit's {best['chi2']:.3g}" if best else "")
                failed = " Failed (error scale above 2)." if alt["chi2"] ** 0.5 > FAILED_SCALE else ""
                caption.append(f"Informed ({alt['name'].replace('_', ' ')}): raw χ²/N {alt['chi2']:.3g}{split}{versus}.{failed} "
                               "Its method was chosen after seeing the truth or the contest page, so it is not a blind result.")
            if runs and not alt and best:
                caption.append(f"{len(runs)} informed run{'s' * (len(runs) > 1)} (advice-seeded, {adv['virgil']}) came no closer to 1 than the blind fit.")
            if lad_ok:
                save_png(render_panel(lad["image"], lad["fov"], half, "virgil, informed ladder"), ASSETS / f"{key}_ladder.png")
                ours.append((f"virgil, informed parametric ladder ({lad['rung']})", f"{key}_ladder.png"))
                badges.append(badge("informed"))
                text = (f"Informed parametric ladder ({lad['rung']}): raw χ²/N {lad['chi2']:.3g} "
                        f"(V² {lad['chi2_v2']:.3g}, closure phase {lad['chi2_cp']:.3g}).")
                if row["truth"] == "score_2004":
                    lad_sigma = truth_2004_data2(lad["image"], lad["fov"])[2]
                    text += f" σ/peak {lad_sigma:.3f} against the rebuilt truth (informed; the blind fit's is {sigma:.3f})."
                caption.append(text)
            elif lad is not None:
                caption.append(f"The informed parametric ladder does not reach this data set: its best rung ({lad['rung']}) has raw χ²/N {lad['chi2']:.3g}.")
            if truth is not None:
                save_png(render_panel(truth[0], truth[1], half, "truth (rebuilt)"), ASSETS / f"{key}_truth.png")
                ours.append(("truth, rebuilt from the paper's model", f"{key}_truth.png"))
            badges.append(badge({"score_2004": "truth", "markers_2018": "partial"}.get(row["truth"], "no-truth")))
            if row.get("note"):
                caption.append(row["note"])
            if sigma is not None:
                caption.append(f"Published σ/peak: {published}.")
            status.append((year, row, best, alt, lad if lad_ok else None))
            refs = row_panels(paper, row)
            head = [f"### {year} {row['name']}", "", f'<p class="cg-meta">{" ".join(badges)} {html.escape(row["target"])}.</p>', ""]
            empty = [] if ours else [card("virgil", text="Not imaged yet", kind="empty")]
            # Public: our renders, and the paper panels as citations.
            cards = [card(t, f"assets/contests/{f}") for t, f in ours] + empty
            cards += [card(panel_ref(p), text=f'<a href="{paper["ads"]}">{html.escape(p["figure"])}</a><br>'
                           f'{html.escape(p.get("panel") or "")}', kind="cite") for p in refs]
            public += head + ['<div class="cg-grid">', *cards, "</div>", ""]
            if caption:
                public += [f'<p class="cg-caption">{html.escape(" ".join(caption))}</p>', ""]
            if (year, row["name"]) in reviews:
                advice, when = reviews[(year, row["name"])]
                public += [f'<details class="cg-review"><summary>Visual review ({when})</summary>{html.escape(advice)}</details>', ""]
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
    public[n_intro:n_intro] = status_table(status)
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
    zensical = pathlib.Path(sys.executable).parent / "zensical"  # the venv's, often not on PATH
    print(f"wrote {out}: serve with  {zensical} serve -f {out / 'mkdocs.yml'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="~/data/imaging_contests")
    parser.add_argument("--private", default=None, help="also write a local-only Zensical site with paper crops here")
    args = parser.parse_args()
    build(pathlib.Path(args.data).expanduser(), args.private)


if __name__ == "__main__":
    main()
