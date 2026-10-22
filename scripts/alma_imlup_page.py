"""IM Lup: virgil against MPoL and CLEAN. Builds the figures and the generated blocks of docs/imaging/imlup.md.

    python scripts/alma_imlup_page.py --r DIR [--r-mpol DIR] [--r-mpolwf DIR] --mpol DIR [--compare DIR]
        [--clean DIR] [--dsharp FITS] [--dsharp-profile TXT] [--cv DIR] [--job-id NAME=ID ...]
        [--assets docs/imaging/imlup] [--page docs/imaging/imlup.md]

Every input is a directory (or file) argument and every one is optional: a missing input leaves a clearly marked
"pending" block (the generated blocks of the page sit between ``<!-- gen:NAME -->`` markers, as on the post-AGB page).
NumPy, matplotlib and astropy only, on the small products the fits and ``compare`` save; nothing here grids visibilities
or imports JAX. Whatever needs the 42 M visibilities (the residual dirty images of MPoL's solution, the per-ddid chi^2 of
both solutions in virgil) comes out of ``scripts/alma_imlup_maxent.py compare`` on OzSTAR.

Inputs, as the OzSTAR jobs write them:

    --r, --r-mpol, --r-mpolwf   a ``fit --config R`` directory (result.json, values.npz, image.npy, resid_dirty_obs*.npy)
    --mpol                      MPoL's export (``export/export.npz``, ``export.json``, ``objective.json``; ``summary.md``)
    --compare                   ``compare`` output (compare.json, resid_dirty_mpol_obs*.npy)
    --clean                     our CLEAN (``alma_imlup_clean.py run``: image.fits, clean.json)
    --dsharp, --dsharp-profile  DSHARP's IMLup_continuum.fits and IMLup.profile.txt
    --cv                        ``cv-aggregate`` output (cv_summary.json, cv_curve.png, cv_annulus.png)

Images follow ``crosscheck.alma_maxent``: ``image[row, col]`` with row 0 North and column 0 East, so ``imshow`` with its
default origin has North up and East left. Every panel is drawn with an explicit extent (``grid_extent``) in arcsec East
and North of the phase centre, and carries a marker at the phase centre. Both fits and MPoL's image put the disc on the phase
centre to a few mas, so a reflection through it cannot move the disc: a mirror is caught by the position angle of the emission's
major axis (``major_axis_pa``: 144.5° becomes 35.5°), and a 180° rotation only by comparing with the DSHARP image, which
``--dsharp`` supplies (the page then lists each image's centroid and position angle).
"""

import argparse
import json
import math
import pathlib
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crosscheck import alma_maxent as cx  # noqa: E402  (NumPy and astropy only)

# The MS's phase centre (docs/alma_conventions.md).
RA_DEG, DEC_DEG = (15 + 56 / 60 + 9.20 / 3600) * 15, -(37 + 56 / 60 + 6.42 / 3600)
INCL_DEG, PA_DEG = 47.5, 144.5  # IM Lup, Huang et al. 2018 (DSHARP II), as in design/alma_clean.md
DSHARP_BEAM = (44.0, 43.0, 115.0)  # DSHARP I Table 4 (mas, mas, deg): the nominal beam when no CLEAN image is given
N_VIS = 42_421_131
OBS_LABELS = ["SB1", "SB2", "SB3", "SB4", "SB5", "LB1 EB1", "LB1 EB2"]
OBS_OF_DDID = [0] * 5 + [1] * 6 + [2] * 2 + [3] * 2 + [4] * 3 + [5] * 4 + [6] * 4  # MPoL's obsid_dict
# categorical slots 1-7 of the reference palette (dataviz skill); markers differ per EB as the secondary encoding
EB_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
EB_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
C_VIRGIL, C_MPOL = "#2a78d6", "#eb6834"
RUNS = [("R", "R", "our P3 workflow"), ("R_mpol", "R_mpol", "warm-started at MPoL's solution"), ("R_mpolwf", "R_mpolwf", "MPoL's own start")]
HOME = str(pathlib.Path.home())


# ------------------------------------------------------------------ pure functions: stretch, geometry, placeholders


def asinh_stretch(x, vmax, soft):
    """Display stretch asinh(x/soft)/asinh(vmax/soft): odd, 0 at 0, 1 at ``vmax``, linear for |x| << soft and logarithmic for |x| >> soft."""
    return np.arcsinh(np.asarray(x, float) / soft) / np.arcsinh(vmax / soft)


def asinh_inverse(y, vmax, soft):
    """Inverse of ``asinh_stretch``."""
    return soft * np.sinh(np.asarray(y, float) * np.arcsinh(vmax / soft))


def asinh_norm(vmax, soft, vmin=0.0):
    """A matplotlib norm applying the asinh stretch between ``vmin`` and ``vmax`` (the colour bar is then in data units)."""
    from matplotlib.colors import FuncNorm

    return FuncNorm((lambda x: np.arcsinh(np.asarray(x, float) / soft), lambda y: soft * np.sinh(np.asarray(y, float))), vmin=vmin, vmax=vmax)


@dataclass(frozen=True)
class Grid:
    """Where the phase centre sits in an image of this repository's layout (row 0 North, column 0 East): the (fractional)
    cell ``(cr, cc)``, and the pixel size in arcsec. Columns run West, rows run South."""

    cr: float
    cc: float
    dx: float


def grid_extent(shape, grid):
    """``imshow`` extent (left, right, bottom, top) in arcsec East and North of the phase centre, East positive: with ``left > right``
    the x axis runs East to the left, and the image keeps row 0 on top (North up)."""
    ny, nx = shape
    return ((grid.cc + 0.5) * grid.dx, (grid.cc - nx + 0.5) * grid.dx, (grid.cr - ny + 0.5) * grid.dx, (grid.cr + 0.5) * grid.dx)


def sky_to_pixel(east, north, grid):
    """(row, column) of a sky offset (arcsec East, North of the phase centre) in an image on ``grid``."""
    return grid.cr - north / grid.dx, grid.cc - east / grid.dx


def pixel_to_sky(row, col, grid):
    """(East, North) offset in arcsec of a pixel (the inverse of ``sky_to_pixel``)."""
    return (grid.cc - col) * grid.dx, (grid.cr - row) * grid.dx


def crop_window(image, grid, half):
    """The part of ``image`` within ``half`` arcsec of the phase centre, and its ``Grid``."""
    ny, nx = image.shape
    r0, r1 = max(0, math.ceil(grid.cr - half / grid.dx)), min(ny, math.floor(grid.cr + half / grid.dx) + 1)
    c0, c1 = max(0, math.ceil(grid.cc - half / grid.dx)), min(nx, math.floor(grid.cc + half / grid.dx) + 1)
    return image[r0:r1, c0:c1], Grid(grid.cr - r0, grid.cc - c0, grid.dx)


def beam_ellipse_angle(pa_deg):
    """``matplotlib.patches.Ellipse`` angle for a beam whose major axis lies at position angle ``pa_deg`` (East of North), when the
    patch is drawn in (East, North) data coordinates: counter-clockwise from +East, 90 - PA. The inverted East axis then mirrors it on screen."""
    return 90.0 - pa_deg


def draw_beam(ax, east, north, bmaj_mas, bmin_mas, pa_deg, **kw):
    """The CLEAN beam (FWHM) as an ellipse centred on the sky offset (``east``, ``north``) of ``ax``, whose data coordinates are (East, North) arcsec."""
    from matplotlib.patches import Ellipse

    e = Ellipse((east, north), bmaj_mas / 1000.0, bmin_mas / 1000.0, angle=beam_ellipse_angle(pa_deg), **kw)
    ax.add_patch(e)
    return e


def centroid_en(image, grid, centre_en=(0.0, 0.0), radius=2.0, floor=0.02):
    """(East, North) offset in arcsec of the flux-weighted centroid of the pixels brighter than ``floor`` x peak, within ``radius`` arcsec of ``centre_en``."""
    rr, cc = np.indices(image.shape)
    e, n = pixel_to_sky(rr, cc, grid)
    sel = (np.hypot(e - centre_en[0], n - centre_en[1]) <= radius) & (image > floor * np.max(image))
    w = np.where(sel, image, 0.0)
    return float((w * e).sum() / w.sum()), float((w * n).sum() / w.sum())


def major_axis_pa(image, grid, centre_en=(0.0, 0.0), radius=1.5, floor=0.05, clip_percentile=99.5):
    """Position angle (degrees East of North, in [0, 180)) of the major axis of the flux-weighted second moments of the pixels brighter than ``floor`` x peak
    within ``radius`` arcsec of ``centre_en``: 144.5 for IM Lup, 35.5 for its East-West (or North-South) mirror image. The image is clipped at the
    ``clip_percentile`` of the window first, so that the point-like spike a MaxEnt image keeps on the star does not set the axis."""
    rr, cc = np.indices(image.shape)
    e, n = pixel_to_sky(rr, cc, grid)
    e, n = e - centre_en[0], n - centre_en[1]
    win = np.hypot(e, n) <= radius
    image = np.minimum(image, np.percentile(image[win], clip_percentile))
    sel = win & (image > floor * np.max(image[win]))
    w = np.where(sel, image, 0.0)
    ce, cn = (w * e).sum() / w.sum(), (w * n).sum() / w.sum()
    cov = np.array([[(w * (e - ce) ** 2).sum(), (w * (e - ce) * (n - cn)).sum()], [(w * (e - ce) * (n - cn)).sum(), (w * (n - cn) ** 2).sum()]]) / w.sum()
    vals, vecs = np.linalg.eigh(cov)
    ve, vn = vecs[:, np.argmax(vals)]
    return float(math.degrees(math.atan2(ve, vn)) % 180.0)


def radial_profile(image, grid, edges, centre_en=(0.0, 0.0), incl_deg=INCL_DEG, pa_deg=PA_DEG):
    """Azimuthal mean of ``image`` in bins of deprojected radius (arcsec) about ``centre_en``: a disc of inclination ``incl_deg`` whose
    major axis lies at position angle ``pa_deg`` (East of North). Returns (bin centres, mean, pixel count)."""
    rr, cc = np.indices(image.shape)
    e, n = pixel_to_sky(rr, cc, grid)
    e, n = e - centre_en[0], n - centre_en[1]
    p, i = math.radians(pa_deg), math.radians(incl_deg)
    along, across = e * math.sin(p) + n * math.cos(p), e * math.cos(p) - n * math.sin(p)
    r = np.hypot(along, across / math.cos(i))
    idx = np.digitize(r.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    cnt = np.bincount(idx[ok], minlength=len(edges) - 1).astype(float)
    tot = np.bincount(idx[ok], weights=image.ravel()[ok], minlength=len(edges) - 1)
    return 0.5 * (edges[1:] + edges[:-1]), np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan), cnt


IM_LUP_PC = 158.0  # pc, Huang et al. 2018 (DSHARP II) distance, to turn an au column into arcsec


def load_profile(path):
    """DSHARP's ``IMLup.profile.txt``, with the units read from its ``#`` header rather than assumed: the first column named in arcsec (or au,
    divided by ``IM_LUP_PC`` pc) is the radius, the first named in Jy/beam, mJy/beam or uJy/beam is the brightness (scaled to Jy/beam), and the
    next column is its error when it carries the same unit. Returns (r in arcsec, I in Jy/beam, dI in Jy/beam or None); ValueError if the
    header does not name them, so a wrong file stops the page rather than plotting a curve that is off by a unit."""
    lines = pathlib.Path(path).read_text().splitlines()
    header = [ln.lstrip("#").strip() for ln in lines if ln.lstrip().startswith("#")]
    names = None
    for ln in reversed(header):  # the last comment line that splits into as many labels as there are data columns
        ncol = len(np.atleast_2d(np.loadtxt(path, comments="#")).T)
        parts = [x.strip() for x in re.split(r"\s{2,}|\t|,", ln) if x.strip()] if re.search(r"\s{2,}|\t|,", ln) else ln.split()
        if len(parts) == ncol:
            names = parts
            break
    if names is None:
        raise ValueError(f"{path}: no '#' header line naming the columns; cannot tell the radius unit (au or arcsec) or the brightness unit (mJy/beam or Jy/beam)")
    low = [x.lower() for x in names]
    rcol = next((k for k, x in enumerate(low) if "arcsec" in x or '"' in x), None)
    r_scale = 1.0
    if rcol is None:
        rcol = next((k for k, x in enumerate(low) if re.search(r"(^|[^a-z])au([^a-z]|$)", x)), None)
        r_scale = 1.0 / IM_LUP_PC
    icol = next((k for k, x in enumerate(low) if "/beam" in x and ("jy" in x)), None)
    if rcol is None or icol is None:
        raise ValueError(f"{path}: header {names} does not name a radius (arcsec or au) and a brightness (Jy/beam) column")
    u = low[icol]
    i_scale = 1e-3 if "mjy" in u else 1e-6 if ("ujy" in u or "µjy" in u) else 1.0
    a = np.atleast_2d(np.loadtxt(path, comments="#"))
    err = a[:, icol + 1] * i_scale if a.shape[1] > icol + 1 and "/beam" in low[icol + 1] else None
    return a[:, rcol] * r_scale, a[:, icol] * i_scale, err


def pending(what, needs):
    """The block that stands for a product that does not exist yet: an admonition naming what is missing and what produces it."""
    return f'!!! warning "Pending: {what}"\n    {needs}\n'


def replace_block(text, name, body):
    """``text`` with the generated block ``<!-- gen:NAME --> ... <!-- /gen:NAME -->`` replaced by ``body`` (KeyError if it is not there)."""
    pat = re.compile(rf"(<!-- gen:{re.escape(name)} -->\n)(.*?)(<!-- /gen:{re.escape(name)} -->)", re.S)
    if not pat.search(text):
        raise KeyError(f"no gen:{name} block")
    return pat.sub(lambda m: m.group(1) + "\n" + body.strip("\n") + "\n\n" + m.group(3), text, count=1)


def gate_result(ours_objective, mpol_objective, ours_chi2n, mpol_chi2n, converged):
    """The gate of design/alma_imlup_p3.md: our objective <= MPoL's (lower is better), with raw chi^2/N (s = 1) beside it and whether C8 held."""
    d = ours_objective - mpol_objective
    return dict(passed=bool(d <= 0), delta=float(d), ours_chi2n=float(ours_chi2n), mpol_chi2n=float(mpol_chi2n), chi2n_delta=float(ours_chi2n - mpol_chi2n), c8=bool(converged))


def data_column(result):
    """The MS column a fit read, from the data directory it recorded: the ``-cd`` tags of ``alma_imlup_read`` are CORRECTED_DATA, the older ones DATA."""
    d = str((result.get("args") or {}).get("data", "")).rstrip("/")
    return "CORRECTED_DATA" if d.endswith("-cd") else "DATA"


def tag_of(result):
    """The pin of a run: the directory ``<tag>`` in ``.../out/<tag>/R`` of the output path it recorded (None if there is none)."""
    parts = pathlib.PurePosixPath(str((result.get("args") or {}).get("out", ""))).parts
    return parts[-2] if len(parts) >= 2 else None


def short(path):
    return str(path).replace(HOME, "~")


def save_small(fig, path, max_bytes=900_000, dpi=110):
    """Save a figure as PNG, lowering the resolution until the file is at most ``max_bytes`` (the repository keeps assets under about 1 MB)."""
    for d in (dpi, 90, 75, 60, 48):
        fig.savefig(path, dpi=d, facecolor="white")
        if pathlib.Path(path).stat().st_size <= max_bytes:
            return d
    raise RuntimeError(f"{path} is larger than {max_bytes} bytes even at 48 dpi")


# ------------------------------------------------------------------ loaders (each returns None for a missing input)


def load_fit(path):
    p = pathlib.Path(path) if path else None
    if p is None or not (p / "result.json").exists():
        return None
    res = json.loads((p / "result.json").read_text())
    return dict(dir=p, result=res, image=(np.load(p / "image.npy") if (p / "image.npy").exists() else None))


def load_mpol(path):
    p = pathlib.Path(path) if path else None
    if p is None:
        return None
    ex = p / "export" if (p / "export" / "export.npz").exists() else p
    if not (ex / "export.npz").exists():
        return None
    z = np.load(ex / "export.npz")
    out = dict(dir=p, z={k: z[k] for k in z.files}, export=json.loads((ex / "export.json").read_text()) if (ex / "export.json").exists() else {},
               objective=json.loads((ex / "objective.json").read_text()) if (ex / "objective.json").exists() else None, pins=None, versions={})
    for name in ("summary.md", "../summary.md"):
        if (ex / name).exists():
            m = re.search(r"^Pins: (.+)$", (ex / name).read_text(), re.M)
            out["pins"] = m.group(1).strip() if m else None
            break
    if (p / "versions.txt").exists():
        out["versions"] = dict(line.split("==", 1) for line in (p / "versions.txt").read_text().split() if "==" in line)
    return out


def load_compare(path):
    p = pathlib.Path(path) if path else None
    if p is None or not (p / "compare.json").exists():
        return None
    return dict(dir=p, result=json.loads((p / "compare.json").read_text()))


def load_sky_fits(path):
    """A CLEAN image: (image in this repository's layout, Jy/beam; its Grid; beam (bmaj mas, bmin mas, PA deg); the array in FITS layout and its WCS)."""
    p = pathlib.Path(path) if path else None
    if p is None or not p.exists():
        return None
    data, wcs, hdr = cx.read_fits(p)
    x, y = wcs.all_world2pix(RA_DEG, DEC_DEG, 0)
    ny = data.shape[0]
    grid = Grid(cr=float(ny - 1 - y), cc=float(x), dx=abs(float(hdr["CDELT2"])) * 3600.0)
    return dict(path=p, image=np.nan_to_num(data[::-1]), grid=grid, beam=cx.beam_of(hdr), fits=np.nan_to_num(data), wcs=wcs)


def load_clean(path):
    p = pathlib.Path(path) if path else None
    c = load_sky_fits(p / "image.fits") if p else None
    if c is not None:
        c["run"] = json.loads((p / "clean.json").read_text()) if (p / "clean.json").exists() else {}
    return c


# ------------------------------------------------------------------ figures


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False})
    return plt


def _blank(ax, text):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(1)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linestyle(":")
        s.set_color("0.6")
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color="0.35", fontsize=8, wrap=True)


def _sky_axes(ax, half):
    ax.set_xlim(half, -half)  # East left
    ax.set_ylim(-half, half)  # North up
    ax.set_xlabel("ΔRA (arcsec, East left)")
    ax.set_ylabel("ΔDec (arcsec, North up)")


def sky_image(ax, image, grid, norm, cmap, half, title, marker=True):
    im = ax.imshow(image, extent=grid_extent(image.shape, grid), norm=norm, cmap=cmap, interpolation="nearest")  # origin 'upper': row 0 North
    _sky_axes(ax, half)
    ax.set_title(title)
    if marker:
        ax.plot(0.0, 0.0, marker="+", color="cyan", ms=9, mew=1.2)
    return im


def to_jy_arcsec2(clean_like):
    """A CLEAN image (Jy/beam) in Jy/arcsec^2, the brightness unit of the MaxEnt images, by its own beam area."""
    b = clean_like["beam"]
    return clean_like["image"] / cx.beam_area_arcsec2(b[0], b[1])


def on_grid(clean_like, n, px_mas):
    """A CLEAN image (Jy/arcsec^2) resampled onto the n x n grid of a MaxEnt image of ``px_mas`` (this repository's layout, phase centre in cell n/2 - 1)."""
    b = clean_like["beam"]
    out = cx.resample(clean_like["fits"], clean_like["wcs"], cx.sky_wcs(n, px_mas, RA_DEG, DEC_DEG), (n, n))[::-1]
    return out / cx.beam_area_arcsec2(b[0], b[1])


def fig_images(assets, head, mpol, clean, dsharp, half=2.5, zoom=0.5, soft_frac=0.03, diff_frac=0.1, vmax_percentile=99.95):
    """Row 1: virgil | MPoL | our CLEAN | DSHARP in Jy/arcsec^2 on one asinh stretch, with an inset on the inner rings and the CLEAN beam. Row 2: the
    MaxEnt images convolved to the CLEAN beam, and their differences. Returns (file name or None, metrics)."""
    plt = _mpl()
    if head is None and mpol is None and clean is None and dsharp is None:
        return None, {}
    px = None
    panels = {}  # name -> (image Jy/arcsec2, Grid)
    if head is not None and head["image"] is not None:
        px = float(head["result"]["pixel_mas"])
        n = head["image"].shape[0]
        r0 = n // 2 - 1
        panels["virgil"] = (head["image"].astype(float), Grid(r0, r0, px / 1000.0))
    if mpol is not None:
        z = mpol["z"]
        n = z["base_image"].shape[0]
        px = px or float(z["cell_size_arcsec"]) * 1000.0
        panels["MPoL"] = (np.asarray(z["base_image"], float), Grid(n // 2 - 1, n // 2 - 1, float(z["cell_size_arcsec"])))
    for name, c in (("our CLEAN", clean), ("DSHARP CLEAN", dsharp)):
        if c is not None:
            panels[name] = (to_jy_arcsec2(c), c["grid"])
    beam_src = dsharp or clean
    beam = beam_src["beam"] if beam_src else DSHARP_BEAM
    beam_note = f"{beam[0]:.1f} × {beam[1]:.1f} mas, PA {beam[2]:.0f}°" + ("" if beam_src else " (DSHARP Table 4, nominal)")
    maxent = [k for k in ("virgil", "MPoL") if k in panels]
    # the MaxEnt images keep a point-like spike on the star: the scale saturates at the ``vmax_percentile`` of their pixels, not at their maximum
    vmax = float(max(np.percentile(panels[k][0], vmax_percentile) for k in (maxent or panels)))
    norm = asinh_norm(vmax, soft_frac * vmax)
    names = ["virgil", "MPoL", "our CLEAN", "DSHARP CLEAN"]
    titles = {"virgil": "virgil MaxEnt (R)", "MPoL": "MPoL (MaxEnt)", "our CLEAN": "our CLEAN (virgil.radio_clean)", "DSHARP CLEAN": "DSHARP CLEAN"}
    pend = {"virgil": "pending: our R fit\n(--r)", "MPoL": "pending: MPoL's export\n(--mpol)", "our CLEAN": "pending: our CLEAN\n(ozstar_scripts#60; --clean)", "DSHARP CLEAN": "pending: DSHARP's FITS\n(--dsharp)"}
    fig, axes = plt.subplots(2, 5, figsize=(20, 8.6), gridspec_kw=dict(wspace=0.3, hspace=0.28))
    metrics = {"beam_mas_deg": beam, "beam_note": beam_note, "centroid_offset_mas": {}, "major_axis_pa_deg": {}, "vmax_jy_arcsec2": vmax}
    ims = []
    for j, nm in enumerate(names):
        ax = axes[0, j]
        if nm not in panels:
            _blank(ax, pend[nm])
            continue
        img, g = panels[nm]
        c, cg = crop_window(img, g, half + 0.1)
        ims.append(sky_image(ax, c, cg, norm, "magma", half, titles[nm]))
        east_c, north_c = centroid_en(c, cg, (0.0, 0.0), 2.0)
        metrics["centroid_offset_mas"][nm] = [1000 * east_c, 1000 * north_c]
        metrics["major_axis_pa_deg"][nm] = major_axis_pa(c, cg, (east_c, north_c), 1.5)
        ax.plot(east_c, north_c, marker="x", color="white", ms=6, mew=1.2)
        if "CLEAN" in nm:
            b = (clean if nm == "our CLEAN" else dsharp)["beam"]
            draw_beam(ax, -half * 0.8, -half * 0.8, *b, fc="white", ec="black", lw=0.6)
        else:
            draw_beam(ax, -half * 0.8, -half * 0.8, *beam, fc="none", ec="white", lw=0.8, ls="--")
        zc, zg = crop_window(img, g, zoom + 0.05)
        ia = ax.inset_axes([0.0, 0.62, 0.36, 0.36])
        ia.imshow(zc, extent=grid_extent(zc.shape, zg), norm=norm, cmap="magma", interpolation="nearest")
        ia.set_xlim(zoom, -zoom)
        ia.set_ylim(-zoom, zoom)
        ia.set_xticks([])
        ia.set_yticks([])
        for s in ia.spines.values():
            s.set_visible(True)
            s.set_color("white")
        ax.add_patch(plt.Rectangle((zoom, -zoom), -2 * zoom, 2 * zoom, fill=False, ec="white", lw=0.6, ls=":"))
        ia.text(0.97, 0.03, f"±{zoom:g}″", transform=ia.transAxes, color="white", fontsize=6, ha="right", va="bottom")
    if ims:
        cb = fig.colorbar(ims[0], cax=axes[0, 4].inset_axes([0.15, 0.05, 0.08, 0.9]), ticks=[t for t in (0, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 4) if t <= vmax])
        cb.set_label("Jy/arcsec²  (asinh stretch; saturates at the 99.95th percentile of the MaxEnt pixels)", fontsize=7)
    axes[0, 4].axis("off")
    # row 2: convolved MaxEnt images and differences, all on the MaxEnt grid
    conv = {}
    for k in maxent:
        img, g = panels[k]
        conv[k] = cx.convolve(img, g.dx * 1000.0, *beam)
        c, cg = crop_window(conv[k], g, half + 0.1)
        sky_image(axes[1, 0 if k == "virgil" else 1], c, cg, norm, "magma", half, f"{titles[k].split(' (')[0]} ⊛ CLEAN beam")
        draw_beam(axes[1, 0 if k == "virgil" else 1], -half * 0.8, -half * 0.8, *beam, fc="white", ec="black", lw=0.6)
    for j in (0, 1):
        if ("virgil", "MPoL")[j] not in conv:
            _blank(axes[1, j], pend[("virgil", "MPoL")[j]])
    if "virgil" in conv:
        g = panels["virgil"][1]
        n = conv["virgil"].shape[0]
        lim = diff_frac * float(np.max(conv["virgil"]))
        others = [("MPoL", "virgil − MPoL", conv.get("MPoL")), ("DSHARP CLEAN", "virgil − DSHARP", on_grid(dsharp, n, px) if dsharp else None),
                  ("our CLEAN", "virgil − our CLEAN", on_grid(clean, n, px) if clean else None)]
        dims = []
        for j, (nm, title, other) in enumerate(others):
            ax = axes[1, 2 + j]
            if other is None:
                _blank(ax, pend[nm])
                continue
            d = conv["virgil"] - other
            c, cg = crop_window(d, g, half + 0.1)
            dims.append(sky_image(ax, c, cg, plt.Normalize(-lim, lim), "RdBu_r", half, title + f" (±{100 * diff_frac:.0f}% of peak)"))
            cb = fig.colorbar(dims[-1], ax=ax, fraction=0.046, pad=0.03)
            cb.set_label("Jy/arcsec²")
            metrics.setdefault("difference", {})[title] = dict(rms_over_peak=float(np.sqrt(np.mean(d[np.hypot(*pixel_grid(d.shape, g)) < half] ** 2)) / np.max(conv["virgil"])),
                                                              max_abs_over_peak=float(np.max(np.abs(d)) / np.max(conv["virgil"])))
    else:
        for j in range(2, 5):
            _blank(axes[1, j], pend["virgil"])
    fig.subplots_adjust(top=0.92, bottom=0.06, left=0.04, right=0.98)
    name = "images.png"
    save_small(fig, assets / name)
    plt.close(fig)
    return name, metrics


def pixel_grid(shape, grid):
    """(East, North) offsets in arcsec of every pixel of an image on ``grid``."""
    rr, cc = np.indices(shape)
    return pixel_to_sky(rr, cc, grid)


def cal_arrays(fit, mpol):
    """Per-EB gain and offsets (mas) and per-ddid s, ours (``fit``) and MPoL's, in the shared convention (docs/alma_conventions.md, Part 3):
    gains are directly comparable and s = g s_MPoL. Either may be None."""
    obs_of = np.asarray(OBS_OF_DDID)
    out = {}
    if fit is not None:
        r = fit["result"]
        out["ours"] = dict(gain=np.concatenate([[1.0], np.asarray(r["gain"], float)]), offset=np.concatenate([np.zeros((1, 2)), np.asarray(r["offset_mas"], float)]),
                           sigma=np.asarray(r["sigma"], float))
    if mpol is not None:
        z = mpol["z"]
        amp = np.asarray(z["amp"], float)
        out["mpol"] = dict(gain=amp, offset=1000.0 * np.stack([z["delta_ra_arcsec"], z["delta_dec_arcsec"]], -1), sigma=np.asarray(z["sigma"], float) * amp[obs_of])
    return out


def fig_calibration(assets, fit, mpol):
    plt = _mpl()
    cal = cal_arrays(fit, mpol)
    if "ours" not in cal or "mpol" not in cal:
        return None
    o, m = cal["ours"], cal["mpol"]
    obs_of = np.asarray(OBS_OF_DDID)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), gridspec_kw=dict(wspace=0.32))
    specs = [("s per ddid (ours vs g·s_MPoL)", o["sigma"], m["sigma"], obs_of, "s_ours", "g · s_MPoL"),
             ("gain g per EB", o["gain"], m["gain"], np.arange(7), "g (ours)", "g (MPoL)"),
             ("RA offset δRA per EB (mas)", o["offset"][:, 0], m["offset"][:, 0], np.arange(7), "δRA (ours, mas)", "δRA (MPoL, mas)"),
             ("Dec offset δDec per EB (mas)", o["offset"][:, 1], m["offset"][:, 1], np.arange(7), "δDec (ours, mas)", "δDec (MPoL, mas)")]
    for ax, (title, ours, theirs, eb, lx, ly) in zip(axes, specs):
        both = np.concatenate([ours, theirs])
        pad = 0.06 * np.ptp(both) + 1e-9
        lo, hi = both.min() - pad, both.max() + pad
        ax.plot([lo, hi], [lo, hi], color="0.55", lw=1, zorder=0)
        for k in range(7):
            sel = eb == k
            fixed = len(ours) == 7 and k == 0 and "per EB" in title
            ax.scatter(theirs[sel], ours[sel], s=46, marker=EB_MARKERS[k], facecolor="none" if fixed else EB_COLORS[k], edgecolor=EB_COLORS[k] if fixed else "white",
                       linewidth=1.4 if fixed else 0.6, label=OBS_LABELS[k] + (" (fixed)" if fixed else ""), zorder=3)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel(ly)
        ax.set_ylabel(lx)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left", title="execution block")
    fig.suptitle("Calibration nuisances: virgil against MPoL (grey line: 1:1; hollow marker: held fixed at 1 or 0 in both)", y=1.02, fontsize=10)
    save_small(fig, assets / "calibration.png")
    plt.close(fig)
    return "calibration.png"


def calibration_table(fit, mpol):
    cal = cal_arrays(fit, mpol)
    if "ours" not in cal or "mpol" not in cal:
        return None
    o, m = cal["ours"], cal["mpol"]
    rows = ["| EB | g (virgil) | g (MPoL) | δRA virgil / MPoL (mas) | δDec virgil / MPoL (mas) |", "| --- | ---: | ---: | ---: | ---: |"]
    for k in range(7):
        rows.append(f"| {OBS_LABELS[k]} | {o['gain'][k]:.4f} | {m['gain'][k]:.4f} | {o['offset'][k, 0]:+.2f} / {m['offset'][k, 0]:+.2f} | {o['offset'][k, 1]:+.2f} / {m['offset'][k, 1]:+.2f} |")
    ratio = o["sigma"] / m["sigma"]
    rows += ["", f"s_ours / (g s_MPoL) over the 26 ddids: min {ratio.min():.3f}, max {ratio.max():.3f}; max |gain difference| {np.max(np.abs(o['gain'] - m['gain'])):.4f}; "
             f"max |offset difference| {np.max(np.abs(o['offset'] - m['offset'])):.2f} mas.", "",
             '??? note "s per ddid"', "", "    | ddid | EB | s virgil | g · s MPoL | ratio |", "    | ---: | --- | ---: | ---: | ---: |"]
    for d in range(len(ratio)):
        rows.append(f"    | {d} | {OBS_LABELS[OBS_OF_DDID[d]]} | {o['sigma'][d]:.4f} | {m['sigma'][d]:.4f} | {ratio[d]:.3f} |")
    return "\n".join(rows)


def chi2_arrays(fit, mpol):
    """Raw chi^2/N (s = 1, quoted weights; chi^2 / 2N) per ddid for virgil's fit (its own report) and for MPoL's solution (MPoL's own evaluation).
    Returns (n per ddid, virgil, MPoL, MPoL label) or None."""
    if fit is None or mpol is None or mpol["objective"] is None:
        return None
    t = fit["result"].get("chi2_s1")
    if not t:
        return None
    n = np.asarray(t["n"], float)
    v = np.asarray(t["chi2_raw_over_n"], float)
    o = mpol["objective"]
    m = np.asarray(o["chi2_raw"], float) / (2 * np.asarray(o["n_ddid"], float))
    return n, v, m, "MPoL's solution (MPoL's own evaluation)"


def fig_chi2(assets, fit, mpol):
    plt = _mpl()
    c = chi2_arrays(fit, mpol)
    if c is None:
        return None
    n, v, m, mlabel = c
    fig, ax = plt.subplots(figsize=(12, 3.8))
    x = np.arange(len(v))
    for k in range(7):
        sel = np.asarray(OBS_OF_DDID) == k
        if k % 2 == 0:
            ax.axvspan(x[sel][0] - 0.5, x[sel][-1] + 0.5, color="0.94", zorder=0)
        ax.text(x[sel].mean(), 4.95, OBS_LABELS[k], ha="center", va="top", fontsize=7, color="0.35")
    ax.bar(x - 0.2, v, 0.38, color=C_VIRGIL, label="virgil R", edgecolor="white", linewidth=0.8)
    ax.bar(x + 0.2, m, 0.38, color=C_MPOL, label=mlabel, edgecolor="white", linewidth=0.8)
    ax.axhline(1.0, color="0.4", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xlabel("ddid")
    ax.set_ylabel("raw χ²/N  (s = 1, quoted weights)")
    ax.set_ylim(0, 5.0)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    save_small(fig, assets / "chi2.png")
    plt.close(fig)
    return "chi2.png"


def fig_residuals(assets, fit, compare):
    """Residual dirty images per EB: virgil's (``resid_dirty_obs*.npy`` of the fit) above MPoL's (``resid_dirty_mpol_obs*.npy`` of ``compare``)."""
    plt = _mpl()
    fd = fit["dir"] if fit else None
    cd = compare["dir"] if compare else None
    have_v = fd is not None and (fd / "resid_dirty_obs0.npy").exists()
    have_m = cd is not None and (cd / "resid_dirty_mpol_obs0.npy").exists()
    if not (have_v or have_m):
        return None, {}
    fig, axes = plt.subplots(2, 7, figsize=(21, 6.2), gridspec_kw=dict(wspace=0.08, hspace=0.12))
    stats = {}
    for e in range(7):
        imgs = {}
        for row, (have, path) in enumerate(((have_v, fd and fd / f"resid_dirty_obs{e}.npy"), (have_m, cd and cd / f"resid_dirty_mpol_obs{e}.npy"))):
            if have and path.exists():
                imgs[row] = np.load(path).astype(float)
        ref = imgs.get(0, imgs.get(1))
        if ref is None:
            continue
        sig = 1.4826 * np.median(np.abs(ref - np.median(ref)))
        lim = 5 * sig
        for row in (0, 1):
            ax = axes[row, e]
            if row not in imgs:
                _blank(ax, "pending:\nvirgil fit" if row == 0 else "pending: compare stage\n(resid_dirty_mpol_obs*.npy)")
                continue
            img = imgs[row]
            n = img.shape[0]
            g = Grid(n // 2, n // 2, 0.003)  # crosscheck.alma's layout: centre at npix // 2, 3 mas pixels
            ax.imshow(img * 1e3, extent=grid_extent(img.shape, g), cmap="RdBu_r", vmin=-lim * 1e3, vmax=lim * 1e3, interpolation="nearest")
            ax.plot(0.0, 0.0, marker="+", color="k", ms=6, mew=0.8)
            ax.set_xlim(1.0, -1.0)
            ax.set_ylim(-1.0, 1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            r = float(np.sqrt(np.mean((img[n // 2 - 170:n // 2 + 170, n // 2 - 170:n // 2 + 170]) ** 2)))
            stats.setdefault(OBS_LABELS[e], {})["virgil" if row == 0 else "MPoL"] = r
            ax.text(0.03, 0.03, f"rms {r * 1e6:.0f} µJy", transform=ax.transAxes, fontsize=7, color="0.2")
            if row == 0:
                ax.set_title(OBS_LABELS[e])
        axes[0, e].set_title(OBS_LABELS[e])
    axes[0, 0].set_ylabel("virgil R")
    axes[1, 0].set_ylabel("MPoL")
    fig.suptitle("Residual dirty images per execution block (data − model, natural weights w/s², ±5 σ_MAD per column, ±1″ field, North up, East left)", y=0.97, fontsize=10)
    save_small(fig, assets / "residuals.png")
    plt.close(fig)
    return "residuals.png", stats


def fig_profile(assets, head, mpol, clean, dsharp, dsharp_profile, half=2.5):
    """Deprojected radial profiles (i = 47.5°, PA = 144.5°, each about its own emission centroid) of the images at the CLEAN beam, in Jy/arcsec²."""
    plt = _mpl()
    series = {}
    beam = (dsharp or clean or {"beam": DSHARP_BEAM})["beam"]
    edges = np.arange(0.0, half, 0.02)
    srcs = []
    if head is not None and head["image"] is not None:
        px = float(head["result"]["pixel_mas"])
        n = head["image"].shape[0]
        srcs.append(("virgil R ⊛ beam", C_VIRGIL, head["image"].astype(float), Grid(n // 2 - 1, n // 2 - 1, px / 1000.0), True))
    if mpol is not None:
        z = mpol["z"]
        n = z["base_image"].shape[0]
        srcs.append(("MPoL ⊛ beam", C_MPOL, np.asarray(z["base_image"], float), Grid(n // 2 - 1, n // 2 - 1, float(z["cell_size_arcsec"])), True))
    if clean is not None:
        srcs.append(("our CLEAN", "#1baf7a", to_jy_arcsec2(clean), clean["grid"], False))
    if dsharp is not None:
        srcs.append(("DSHARP CLEAN image", "#6250d6", to_jy_arcsec2(dsharp), dsharp["grid"], False))
    if not srcs and dsharp_profile is None:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for label, col, img, g, smooth in srcs:
        if smooth:
            img = cx.convolve(img, g.dx * 1000.0, *beam)
        centre = centroid_en(img, g, (0.0, 0.0), 2.0)  # each image about its own disc centre, which sits within a few mas of the phase centre
        r, p, _ = radial_profile(img, g, edges, centre_en=centre)
        series[label] = dict(r=r.tolist(), i=p.tolist())
        ax.plot(r, p, color=col, lw=1.6, label=label)
    if dsharp_profile is not None:
        r, i, di = load_profile(dsharp_profile)
        ax.plot(r, i / cx.beam_area_arcsec2(beam[0], beam[1]), color="k", lw=1.0, ls="--", label="DSHARP II IMLup.profile.txt (Jy/beam ÷ beam area)")
    ax.set_xlabel("deprojected radius (arcsec)")
    ax.set_ylabel("azimuthal mean brightness (Jy/arcsec²)")
    ax.set_xlim(0, half)
    ax.legend(frameon=False, fontsize=8)
    save_small(fig, assets / "profile.png")
    plt.close(fig)
    return "profile.png"


# ------------------------------------------------------------------ generated blocks


def fmt_int(x):
    return f"{x:,.0f}"


def verdict_block(runs, mpol, compare, job_ids):
    """The verdict card and the table of runs. ``runs`` maps run name -> loaded fit (or None)."""
    mo = (mpol or {}).get("objective")
    out = []
    present = {k: v for k, v in runs.items() if v is not None}
    if mo is None or not present:
        what = "the fits and MPoL's export" if not present and mo is None else ("our R fits" if not present else "MPoL's export")
        return pending(f"the gate result ({what})", "`--r DIR` (a `fit --config R` directory) and `--mpol DIR` (MPoL's export) feed this card; see *How to rerun* below.")
    m_own = float(mo["objective_R"])
    m_chi = float(np.sum(mo["chi2_raw"]) / (2 * np.sum(mo["n_ddid"])))
    co = (compare["result"].get("objective") if compare is not None else None) or None
    m_obj = float(co["mpol_solution_in_virgil"]["objective"]) if co else m_own  # the same evaluator as ours, when ``compare`` has run
    gates = {}
    for k, f in present.items():
        r = f["result"]
        n = r["n"]
        ours = float(co["ours"]["objective"]) if co and k == next(iter(present)) else r["objective_R"]  # ``compare`` re-evaluates the head run at eps 1e-12
        gates[k] = gate_result(ours, m_obj, r["chi2_raw"] / (2 * n), m_chi, r["converged"])
    head_name = next(iter(present))
    g = gates[head_name]
    r = present[head_name]["result"]
    col = data_column(r)
    stale = col != "CORRECTED_DATA"
    kind = "warning" if stale else ("success" if g["passed"] and g["c8"] else "failure" if not g["passed"] else "info")
    if stale:
        title = f"Placeholder: the fit shown was read from the {col} column, which MPoL did not use; rerun on CORRECTED_DATA"
    elif g["passed"]:
        title = "The gate is reached: our objective is at or below MPoL's" + ("" if g["c8"] else ", but the fit has not converged (C8)")
    else:
        title = "The gate is not reached: our objective is above MPoL's"
    if not stale and not co:
        title += " (provisional: against MPoL's own evaluation, until `compare` has run)"
    if co:
        ref = f"MPoL's solution evaluated in virgil, {m_obj:,.1f} (`compare`: the same evaluator as ours)"
    else:
        ref = (f"MPoL's own number, {m_obj:,.0f}. This is a different evaluator from ours, so the sign of the difference is provisional until the `compare` stage "
               f"re-evaluates MPoL's solution in virgil (`--compare`)")
    body = [f"**Gate**: our R objective −ln L + λ S_M must be ≤ {ref}. "
            f"Raw χ²/N (s = 1, the quoted weights) is the honest fit measure; a χ² rescaled to 1 by the fitted s is tautological.", ""]
    body += ["| run | objective | − MPoL's | raw χ²/N | − MPoL's | gate | C8 (converged) | stopped because | data column | pin | job |", "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |"]
    body.append(f"| **MPoL** (`sgd.py`, 2 × 10 epochs; its own evaluation) | {m_own:,.0f} | | {m_chi:.4f} | | | | 20 epochs | CORRECTED_DATA | {mpol.get('pins') or 'n/a'} | {job_ids.get('MPoL', 'not recorded')} |")
    for k, label, what in RUNS:
        f = runs.get(k)
        if f is None:
            body.append(f"| {label}: {what} | *pending* | | | | | | | | | {job_ids.get(k, '')} |")
            continue
        gg, rr = gates[k], f["result"]
        body.append(f"| {label}: {what} | {rr['objective_R']:,.0f} | {gg['delta']:+,.0f} | {gg['ours_chi2n']:.4f} | {gg['chi2n_delta']:+.4f} | "
                    f"{'**reached**' if gg['passed'] else '**not reached**'} | {'yes' if gg['c8'] else 'no'} | {rr.get('stopped_because')} | {data_column(rr)} | {tag_of(rr) or 'n/a'} | {job_ids.get(k, 'not recorded')} |")
    body += ["", f"**Data:** 7 execution blocks, 26 spectral windows (ddids), {r['n']:,} visibilities ({r['n'] / 1e6:.1f} M); MPoL's export has {mo['n']:,}. "
             f"Fit: 1028² pixels of 5 mas, 30 mas taper, λ_R = {r['lam']:.6g}."]
    if co:
        body.append(f"**Cross-code agreement.** MPoL's solution evaluated in virgil minus MPoL's own evaluation of it: {m_obj - m_own:+,.1f} on {m_own:,.0f} "
                    f"({(m_obj - m_own) / abs(m_own):+.1e} relative). That is the check that the two objectives are the same function; the gate above does not depend on it.")
    if stale:
        body.append(f"The pinned fit ran on the {col} column; the DSHARP measurement set also holds CORRECTED_DATA, which MPoL's export read, and the two differ by up to 1.6 Jy on the long baselines "
                    f"(design note, decision D1). Every number and image in the fit panels below is therefore a placeholder until the three R runs on CORRECTED_DATA land.")
    out.append(f'!!! {kind} "{title}"\n' + "\n".join("    " + line if line else "" for line in body) + "\n")
    return "\n".join(out)


def images_block(name, metrics):
    if name is None:
        return pending("the image comparison", "Needs at least one of `--r`, `--mpol`, `--clean`, `--dsharp`.")
    txt = [f"![virgil, MPoL and CLEAN images]({'imlup/' + name})", "",
           f"*Top row:* virgil's R fit, MPoL's MaxEnt image, our CLEAN and DSHARP's CLEAN on one asinh stretch (colour bar in Jy/arcsec²; the CLEAN images are divided by their beam area), North up, East left. "
           f"The cyan **+** is the phase centre, the white **×** the centroid of each panel's emission within 2″ of it. The inset is the inner ±0.5″. The solid ellipse is the CLEAN beam "
           f"({metrics['beam_note']}). *Bottom row:* the two MaxEnt images convolved to that beam, and virgil minus MPoL, DSHARP and our CLEAN (diverging scale, ±10% of the convolved peak).", ""]
    cen, pas = metrics.get("centroid_offset_mas") or {}, metrics.get("major_axis_pa_deg") or {}
    if cen:
        txt += [f"**Orientation check.** The fitted discs sit on the phase centre to within a few mas, so a reflection through it cannot move them; "
                f"the test of a mirrored image is the position angle of the emission's major axis ({PA_DEG}° for IM Lup, Huang et al. 2018; its East-West or North-South mirror "
                f"gives {180 - PA_DEG}°). A 180° rotation leaves both numbers unchanged: only the comparison with DSHARP's image tells it, and that panel is pending until `--dsharp` lands. "
                "Centroid offset from the phase centre (mas, East and North) and major-axis position angle of each image's emission:", "",
                "| image | ΔEast | ΔNorth | major-axis PA (°) |", "| --- | ---: | ---: | ---: |"]
        txt += [f"| {k} | {v[0]:+.0f} | {v[1]:+.0f} | {pas.get(k, float('nan')):.1f} |" for k, v in cen.items()]
    if metrics.get("difference"):
        txt += ["", "| difference | rms over the ±2.5″ field / convolved peak | max abs / peak |", "| --- | ---: | ---: |"]
        txt += [f"| {k} | {v['rms_over_peak']:.4f} | {v['max_abs_over_peak']:.4f} |" for k, v in metrics["difference"].items()]
    return "\n".join(txt)


def cv_block(cv, assets):
    p = pathlib.Path(cv) if cv else None
    if p is None or not (p / "cv_summary.json").exists():
        return pending("cross-validation", "Dartboard cross-validation of config F (`alma_imlup_cv`, 45 tasks, then `cv-aggregate`) has not run. `--cv DIR` takes the `cv-aggregate` output "
                       "(`cv_summary.json`, `cv_curve.png`, `cv_annulus.png`) and shows the held-out −ln L against λ and per uv annulus here.")
    s = json.loads((p / "cv_summary.json").read_text())
    txt = []
    for f in ("cv_curve.png", "cv_annulus.png"):
        if (p / f).exists():
            shutil.copy(p / f, assets / f)
            txt.append(f"![{f}](imlup/{f})")
    txt += ["", f"λ* = {s['lam_star']:.6g}, one-standard-error λ = {s['lam_one_se']:.6g}; {s['n_records']} tasks over {len(s['folds'])} folds; at the grid edge: {s['at_edge']}."]
    return "\n".join(txt)


def science_block(profile_name, dsharp_profile):
    txt = []
    if profile_name is None:
        txt.append(pending("the radial profile", "Needs `--r` or `--mpol` (the images) and, for DSHARP, `--dsharp-profile` (`IMLup.profile.txt` of DSHARP II)."))
    else:
        txt += [f"![radial profiles](imlup/{profile_name})", "",
                f"Azimuthal means in deprojected radius (i = {INCL_DEG}°, PA = {PA_DEG}°, DSHARP II) about the mask centre; the MaxEnt images are convolved to the CLEAN beam, the CLEAN images are as released."]
        if dsharp_profile is None:
            txt.append(pending("the DSHARP II profile", "`--dsharp-profile IMLup.profile.txt` adds DSHARP's own profile (columns: radius in arcsec, brightness in Jy/beam, error)."))
    txt += ["", pending("spiral pitch angle and arm phase", "Metric 4 of the design (P5): the m = 2 trailing spiral inside about 1″, against DSHARP III, compared in sky coordinates. Not yet measured.")]
    return "\n".join(txt)


def provenance_block(runs, mpol, clean, dsharp, job_ids, inputs, vv_commit):
    rows = ["| item | value |", "| --- | --- |", f"| virgil-validation commit (generator) | `{vv_commit}` |"]
    vers = None
    for k, label, _ in RUNS:
        f = runs.get(k)
        if f is None:
            rows.append(f"| {label} | pending |")
            continue
        r = f["result"]
        v = r.get("versions", {})
        vers = vers or v
        rows.append(f"| {label} | tag `{tag_of(r) or 'n/a'}`, vv `{v.get('vv') or 'not recorded'}`, virgil {v.get('virgil')}, jax {v.get('jax')}, {', '.join(v.get('devices', []))}; workflow `{r.get('workflow', 'p3')}`; "
                    f"{data_column(r)}; {r.get('stopped_because')}, {r['stages'][-1]['steps']:,} steps in the last stage; wall {r.get('wall_s', 0) / 3600:.1f} h |")
    if mpol is not None:
        v = mpol["versions"]
        rows.append(f"| MPoL | MPoL {v.get('MPoL', '?')} (torch {v.get('torch', '?')}, torchkbnufft {v.get('torchkbnufft', '?')}, visread {v.get('visread', '?')}); {mpol.get('pins') or 'pins not recorded'}; `sgd.py` unmodified, "
                    f"λ_ent 1e-9, 30 mas, AdamW 1e-2, 2 × 10 epochs |")
    else:
        rows.append("| MPoL | pending |")
    rows.append("| our CLEAN | " + (f"`{short(clean['path'])}`; engine {clean['run'].get('args', {}).get('engine', 'n/a')}" if clean else "pending (ozstar_scripts#60)") + " |")
    rows.append("| DSHARP image | " + (f"`{dsharp['path'].name}`; beam {dsharp['beam'][0]:.1f} × {dsharp['beam'][1]:.1f} mas, PA {dsharp['beam'][2]:.0f}°" if dsharp else "pending") + " |")
    rows.append("| job ids | " + (", ".join(f"{k} {v}" for k, v in job_ids.items()) or "none given (`--job-id NAME=ID`)") + " |")
    notes = ["", "**Conventions** (`docs/alma_conventions.md`, Part 3):", "",
             "- Gains: both codes multiply the model of an observation by its gain g and rescale its weights by g², with observation 0 held at 1, so gains are directly comparable. "
             "virgil's per-ddid scale absorbs the weight scaling, s = g · s_MPoL, which is what the s panel plots.",
             "- Offsets: the phase exp(−2πi(δRA·u + δDec·v)) on the model, in mas for virgil and arcsec for MPoL (×1000), the same sign; observation 0 is fixed at 0 in both.",
             "- Images: row 0 North, column 0 East; the phase centre sits in cell (N/2 − 1, N/2 − 1) of MPoL's 1028² grid (the half-pixel map D11). CLEAN images are read from FITS (row 0 South) and flipped.",
             "- Raw χ²/N is χ²/(2N) with the quoted weights (s = 1); the fitted-s version is 1 by construction and is not shown as a result.",
             "- MaxEnt images are shown untapered (the 30 mas taper is part of the likelihood, not of the sky); MPoL's `base_image` is the same quantity.",
             "- CLEAN images are converted from Jy/beam to Jy/arcsec² by their own beam area."]
    inp = ["", "**Inputs of this build:**", ""] + [f"- `{k}`: " + (f"`{short(v)}`" if v else "*not given*") for k, v in inputs.items()]
    return "\n".join(rows + notes + inp)


# ------------------------------------------------------------------ main


def git_head():
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "scripts", "src"], capture_output=True, text=True).stdout.strip())
        return sha + (" (+ uncommitted changes)" if dirty else "")
    except OSError:
        return "unknown"


def build(args):
    assets = pathlib.Path(args.assets)
    assets.mkdir(parents=True, exist_ok=True)
    runs = {"R": load_fit(args.r), "R_mpol": load_fit(args.r_mpol), "R_mpolwf": load_fit(args.r_mpolwf)}
    head = next((v for v in runs.values() if v is not None), None)
    mpol, compare = load_mpol(args.mpol), load_compare(args.compare)
    clean, dsharp = load_clean(args.clean), load_sky_fits(args.dsharp)
    job_ids = dict(j.split("=", 1) for j in args.job_id)
    inputs = {"--r": args.r, "--r-mpol": args.r_mpol, "--r-mpolwf": args.r_mpolwf, "--mpol": args.mpol, "--compare": args.compare, "--clean": args.clean,
              "--dsharp": args.dsharp, "--dsharp-profile": args.dsharp_profile, "--cv": args.cv}
    images, metrics = fig_images(assets, head, mpol, clean, dsharp, half=args.half, zoom=args.zoom)
    cal, chi2 = fig_calibration(assets, head, mpol), fig_chi2(assets, head, mpol)
    resid, rstats = fig_residuals(assets, head, compare)
    prof = fig_profile(assets, head, mpol, clean, dsharp, args.dsharp_profile, half=args.half)
    blocks = {"verdict": verdict_block(runs, mpol, compare, job_ids), "images": images_block(images, metrics)}
    if cal:
        blocks["calibration"] = f"![calibration nuisances](imlup/{cal})\n\n" + calibration_table(head, mpol)
    else:
        blocks["calibration"] = pending("calibration nuisances", "Needs `--r` (virgil's fit) and `--mpol` (MPoL's export).")
    if chi2:
        c = chi2_arrays(head, mpol)
        blocks["fit"] = (f"![raw chi2 per ddid](imlup/{chi2})\n\nRaw χ²/N per ddid (s = 1, quoted weights) at each solution's own gains, offsets and image: virgil R "
                         f"{np.sum(c[1] * c[0]) / np.sum(c[0]):.4f} overall against {c[3].split(' (')[0]} {np.sum(c[2] * c[0]) / np.sum(c[0]):.4f}. Both are evaluated at their own fitted nuisances, by "
                         f"each code's own likelihood; the verdict card gives the like-for-like comparison once `compare` has run.")
    else:
        blocks["fit"] = pending("fit quality", "Needs `--r` and `--mpol`.")
    if resid:
        blocks["residuals"] = f"![residual dirty images](imlup/{resid})"
        if not (compare and (compare['dir'] / 'resid_dirty_mpol_obs0.npy').exists()):
            blocks["residuals"] += "\n\n" + pending("MPoL's residual dirty images", "The `compare` stage writes `resid_dirty_mpol_obs{0..6}.npy` when run with `--data` (OzSTAR, A100); pass its directory as `--compare`.")
    else:
        blocks["residuals"] = pending("residual dirty images", "Written by `fit` (`resid_dirty_obs*.npy`) for virgil and by `compare` (`resid_dirty_mpol_obs*.npy`) for MPoL.")
    blocks["cv"] = cv_block(args.cv, assets)
    blocks["science"] = science_block(prof, args.dsharp_profile)
    blocks["provenance"] = provenance_block(runs, mpol, clean, dsharp, job_ids, inputs, git_head())
    page = pathlib.Path(args.page)
    text = page.read_text()
    for name, body in blocks.items():
        text = replace_block(text, name, body)
    page.write_text(text)
    summary = dict(inputs={k: short(v) if v else None for k, v in inputs.items()}, found={k: v is not None for k, v in runs.items()}, mpol=mpol is not None, compare=compare is not None,
                   clean=clean is not None, dsharp=dsharp is not None, image_metrics=metrics, residual_rms_jy=rstats, figures=[x for x in (images, cal, chi2, resid, prof) if x])
    (assets / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"wrote {page} and {assets}: {summary['figures']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for a in ("r", "r-mpol", "r-mpolwf", "mpol", "compare", "clean", "dsharp", "dsharp-profile", "cv"):
        ap.add_argument(f"--{a}", default=None)
    ap.add_argument("--job-id", action="append", default=[], metavar="NAME=ID", help="job id to show (names: R, R_mpol, R_mpolwf, MPoL), repeatable")
    ap.add_argument("--assets", default=str(ROOT / "docs" / "imaging" / "imlup"))
    ap.add_argument("--page", default=str(ROOT / "docs" / "imaging" / "imlup.md"))
    ap.add_argument("--half", type=float, default=2.5, help="half-width of the image field, arcsec")
    ap.add_argument("--zoom", type=float, default=0.5, help="half-width of the inset on the inner rings, arcsec")
    build(ap.parse_args(argv))


if __name__ == "__main__":
    main()
