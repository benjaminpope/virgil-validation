"""Report stages of the post-AGB parametric disc campaign (WP2): render, compare, table, geometry.

Reads the JSON/NPZ files that ``postagb_parametric.py`` (WP1) writes under ``OUT/<T>/`` and never
fits anything. Missing P2/P3 files give "not run" cells, not errors. Numpy and matplotlib only:
the ring is rendered here from the fitted values (a cross-check against ``virgil`` is in the tests).

Layout assumed for the files (plan section 8; the plan fixes the keys of the model files, and this
module is tolerant about the rest):

``<T>_param_<model>_<arm>[_<variant>].json``
    ``best.values``: dict of path -> number or list (``env.inc``, ``env.pa``, ``env.profile.radius``
    [absent for M0: 0.01 mas], ``env.profile.sigma``, ``env.harmonics.re``/``.im`` lists, ``env.dra``,
    ``env.ddec``); ``best.sd``-like ``sd``; ``best.flags``; ``best.lnZ_use``, ``best.bic`` ...;
    ``derived``; ``modes`` (best first).
``<T>_param_selected.json``
    ``arms.<arm>`` = ``{chain: [{pair, dlnZ, dbic, verdict, flags}], selected, is_ring, flags}``;
    ``jackknife`` = ``{inc, pa, r0, sigma, c1, s1, ...: sd}`` (or ``{sd: {...}}``);
    ``gp_polar.<arm>`` = ``{stored, rule, delta, mismatch, ...}``.
``<T>_param_residuals.npz`` (optional, for the lower panels of the figure)
    ``v2_baseline_m, v2_resid`` (data - model in sigma), ``cp_baseline_m, cp_resid``.
The stored gp-polar/ORGANIC products are read from ``--stored-dir`` (``<T>_gp-polar*_compare_images.npz``).

Angles are degrees, PA East of North, lengths mas; the image grid has row 0 North and column 0 East.

Frames: the core writes ``derived.c`` / ``derived.s`` (the linear harmonic coefficients) in the frame of the *fitted*
``env.pa``, which is not limited to [0, 180). Wherever this module reports a PA wrapped to [0, 180) (table, geometry
JSON) it reports ``c, s`` in that wrapped frame too (odd orders negated when the wrap shifts by an odd number of half
turns, see ``wrap_pa``); ``bright_pa`` is an on-sky angle and is the same in both. ``<T>_param_image.npz`` stores the
wrapped ``pa`` with ``dra``/``ddec``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import sys

import numpy as np

TARGETS = ["AI_Sco", "EN_TrA", "IRAS15469-5311", "IW_Car", "HD95767", "HD108015", "HR4049", "PS_Gem"]
ARMS = ("N0", "N1", "N2")
MODELS = ("M0", "M1", "M2", "M2o", "M3", "M3o", "M5")
PREFERRED_STORED_RUN = "15b154a+6fa8afe"  # the stored gp-polar run to prefer when several hold a target
M0_RADIUS = 0.01  # mas, the fixed radius of the centred Gaussian

# IW Car is in neither postagb_paper dict (its image values are in postagb_imaging.TARGETS).
_IW_CAR_IMAGE = dict(theta_rim=(6.9, None, None), i_rim=(48.0, None, None), pa_rim=(144.0, None, None))


# --- literature values -------------------------------------------------------------------


def literature():
    """``(IMAGE, PMOIRED)`` dicts of ``postagb_paper`` (read at run time), with IW Car added."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    try:
        import postagb_paper as pp

        image, pmoired = dict(pp.IMAGE), dict(pp.PMOIRED)
    except Exception:  # pragma: no cover - the module is a plain dict file; this is a safety net
        image, pmoired = {}, {}
    image.setdefault("IW_Car", dict(_IW_CAR_IMAGE))
    return image, pmoired


def _lit(entry, key):
    """``(value, sigma)`` from a literature tuple ``(v, +s, -s)`` or a bare number."""
    t = entry.get(key) if entry else None
    if t is None:
        return None, None
    if not isinstance(t, (tuple, list)):
        return float(t), None
    v = float(t[0])
    sig = [float(x) for x in t[1:3] if x is not None]
    return v, (max(sig) if sig else None)


# --- small helpers -----------------------------------------------------------------------


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _angdiff180(a, b):
    """Signed difference a - b of two axis angles, in (-90, 90]."""
    return (a - b + 90.0) % 180.0 - 90.0


def _fmt(x, nd=2):
    x = _num(x)
    return "not run" if x is None else f"{x:.{nd}f}"


def _fmt_na(x, nd=2):
    x = _num(x)
    return "n/a" if x is None else f"{x:.{nd}f}"


def _derived(fit):
    """The core's derived quantities of a model file: ``best["derived"]`` (the real layout), else a top-level ``derived``."""
    fit = fit or {}
    return (fit.get("best") or {}).get("derived") or fit.get("derived") or {}


def wrap_pa(pa, c=(), s=()):
    """PA of the major axis wrapped to [0, 180) (the plan's convention), with the odd-order ``c_m, s_m`` negated when the
    wrap shifts by an odd multiple of 180 deg, so the same model is described. Returns ``(pa, c, s)`` (lists)."""
    pa = _num(pa)
    c, s = list(c or []), list(s or [])
    if pa is None:
        return None, c, s
    wrapped = pa % 180.0
    k = int(round((pa - wrapped) / 180.0))
    if k % 2:
        c = [-x if i % 2 == 0 else x for i, x in enumerate(c)]  # index 0 is order 1
        s = [-x if i % 2 == 0 else x for i, x in enumerate(s)]
    return wrapped, c, s


def _reason_tags(fit):
    tags = []
    for m in (fit or {}).get("modes") or []:
        for w in m.get("why_not") or []:
            w = str(w)
            t = "bound" if w.startswith("E2") else "Hessian" if w.startswith("E3") else "ineligible"
            if t not in tags:
                tags.append(t)
    return tags


def fit_status(tdata, model, arm, variant=None):
    """``None`` if the fit has an eligible best; else why not: ``not run`` (no file) or ``no eligible fit (bound, ...)``."""
    fit = tdata["models"].get((model, arm, variant))
    if fit is None:
        return "not run"
    if (fit.get("best") or {}).get("eligible", True) and fit.get("best"):
        return None
    tags = _reason_tags(fit)
    return "no eligible fit" + (f" ({', '.join(tags)})" if tags else "")


def _value(values, path, default=None):
    """A fitted value by path; exact key first, then a key ending in the path."""
    if not values:
        return default
    if path in values:
        return values[path]
    for k, v in values.items():
        if k.endswith("." + path) or k.endswith(path):
            return v
    return default


def _harmonic_w(values):
    """Complex latents ``w_j`` (order 1..M) from ``env.harmonics.re/im`` (lists, scalars or indexed keys)."""
    out = {}
    for part in ("re", "im"):
        base = f"env.harmonics.{part}"
        v = _value(values, base)
        if v is None:
            idx = {}
            for k, x in (values or {}).items():
                m = re.fullmatch(re.escape(base) + r"[\[.](\d+)\]?", k)
                if m:
                    idx[int(m.group(1))] = x
            v = [idx[i] for i in sorted(idx)] if idx else []
        out[part] = np.atleast_1d(np.asarray(v, float))
    n = max(len(out["re"]), len(out["im"]))
    r = np.zeros(n)
    i = np.zeros(n)
    r[: len(out["re"])] = out["re"]
    i[: len(out["im"])] = out["im"]
    return r + 1j * i


# --- the ring model (numpy) --------------------------------------------------------------


def reflection_coefficients(w):
    """``k = w tanh|w| / |w|`` (virgil's positive form), ``|k| < 1``."""
    w = np.asarray(w, complex)
    a = np.abs(w)
    ratio = np.where(a < 1e-8, 1.0, np.tanh(a) / np.where(a < 1e-8, 1.0, a))
    return w * ratio


def positive_factor(theta, w):
    """Azimuthal factor of mean one: ``|h(e^{i theta})|^2 / sum|h_j|^2`` from latents ``w`` (any order count)."""
    theta = np.asarray(theta, float)
    w = np.atleast_1d(np.asarray(w, complex))
    if w.size == 0 or not np.any(w):
        return np.ones_like(theta)
    k = reflection_coefficients(w)
    h = [1.0 + 0j]
    for j in range(1, len(k) + 1):  # step-up recursion of the minimum-phase polynomial
        prev = h + [0j]
        h = [prev[i] + k[j - 1] * np.conj(prev[j - i]) for i in range(j + 1)]
    h = np.asarray(h)
    z = np.exp(1j * theta)
    poly = sum(h[j] * z**j for j in range(len(h)))
    return np.abs(poly) ** 2 / np.sum(np.abs(h) ** 2)


def linear_factor(theta, c, s):
    """``1 + sum_m c_m cos m theta + s_m sin m theta`` (from the equivalent ``derived`` coefficients)."""
    theta = np.asarray(theta, float)
    out = np.ones_like(theta)
    for m, (cm, sm) in enumerate(zip(c, s), start=1):
        out = out + cm * np.cos(m * theta) + sm * np.sin(m * theta)
    return out


def ring_params(best, derived=None):
    """Geometry dict (inc, pa, r0, sigma, dra, ddec, w, c, s) from a model file's ``best`` and ``derived``."""
    values = (best or {}).get("values") or {}
    derived = derived or {}
    inc = _num(_value(values, "env.inc"))
    pa = _num(_value(values, "env.pa"))
    r0 = _num(_value(values, "env.profile.radius"))
    sig = _num(_value(values, "env.profile.sigma"))
    dra = _num(_value(values, "env.dra"))
    ddec = _num(_value(values, "env.ddec"))
    centre = derived.get("centre")
    if inc is None:
        inc = _num(derived.get("inc"))
    if pa is None:
        pa = _num(derived.get("pa_mod180"))
    if r0 is None:
        r0 = _num(derived.get("r0")) if derived.get("r0") is not None else M0_RADIUS
    if sig is None:
        sig = _num(derived.get("sigma"))
    if (dra is None or ddec is None) and centre is not None and len(centre) == 2:
        dra, ddec = _num(centre[0]), _num(centre[1])
    w = _harmonic_w(values)
    c = np.atleast_1d(np.asarray(derived.get("c", []), float)) if derived.get("c") is not None else np.zeros(0)
    s = np.atleast_1d(np.asarray(derived.get("s", []), float)) if derived.get("s") is not None else np.zeros(0)
    return dict(inc=inc, pa=pa, r0=r0, sigma=sig, dra=dra or 0.0, ddec=ddec or 0.0, w=w, c=c, s=s)


def azimuth_factor(p, theta):
    """The azimuthal factor of ring parameters ``p``: from the latents ``w`` if present, else ``c, s``."""
    if p["w"].size:
        return positive_factor(theta, p["w"])
    if p["c"].size:
        return linear_factor(theta, p["c"], p["s"])
    return np.ones_like(np.asarray(theta, float))


def render_ring(p, npix, pixel_mas):
    """Unit-sum image of the disc (row 0 North, column 0 East; star-centred grid, ring centre at (dra, ddec))."""
    idx = np.arange(npix) - (npix - 1) / 2.0
    east = -idx[None, :] * pixel_mas * np.ones((npix, 1))  # column 0 is the most positive East offset
    north = -idx[:, None] * pixel_mas * np.ones((1, npix))
    e, n = east - p["dra"], north - p["ddec"]
    pa, inc = np.radians(p["pa"]), np.radians(p["inc"])
    x = e * np.sin(pa) + n * np.cos(pa)  # along the projected major axis
    y = (e * np.cos(pa) - n * np.sin(pa)) / max(np.cos(inc), 1e-3)  # towards PA + 90, deprojected
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    img = np.exp(-0.5 * ((r - p["r0"]) / p["sigma"]) ** 2) * azimuth_factor(p, theta)
    tot = img.sum()
    return img / tot if tot > 0 else img


def bright_pa(p, n_theta=7200):
    """On-sky PA (0-360, East of North, about the ring centre) of the brightest point of the ring; ``None`` if axisymmetric.

    The profile is separable, so the maximum is on the ring radius at the azimuth maximising the factor.
    """
    if not (np.any(p["w"]) or np.any(p["c"]) or np.any(p["s"])):
        return None  # axisymmetric (M0, M1): there is no brightest azimuth
    theta = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    f = azimuth_factor(p, theta)
    t = theta[int(np.argmax(f))]
    x, y = p["r0"] * math.cos(t), p["r0"] * math.sin(t) * math.cos(math.radians(p["inc"]))
    pa = math.radians(p["pa"])
    east = x * math.sin(pa) + y * math.cos(pa)
    north = x * math.cos(pa) - y * math.sin(pa)
    return math.degrees(math.atan2(east, north)) % 360.0


def hole_contrast(p):
    """``1 - I(0)/I(r0)`` of the radial profile (azimuthal mean), the "is it a ring" number."""
    return 1.0 - math.exp(-0.5 * (p["r0"] / p["sigma"]) ** 2)


# --- beam and convolution ----------------------------------------------------------------


def beam_kernel(shape, sx, sy, phi):
    """Unit-sum elliptical Gaussian on an odd pixel grid centred on a pixel (like ``virgil.imaging.convolve_beam``).

    ``sx, sy`` are sigmas in pixels along and across the direction ``(cos phi, sin phi)`` in (column, row) pixel axes.
    """
    ny, nx = shape[0] + 1 - shape[0] % 2, shape[1] + 1 - shape[1] % 2
    x = (np.arange(nx) - nx // 2)[None, :]
    y = (np.arange(ny) - ny // 2)[:, None]
    c, s = math.cos(phi), math.sin(phi)
    k = np.exp(-0.5 * (((x * c + y * s) / max(sx, 1e-6)) ** 2 + ((-x * s + y * c) / max(sy, 1e-6)) ** 2))
    return k / k.sum()


def convolve(img, beam):
    """Convolve with a beam ``(sx, sy, phi)`` in pixels (linear: flux beyond the edge is zero, nothing wraps)."""
    from scipy.signal import fftconvolve

    img = np.asarray(img, float)
    return fftconvolve(img, beam_kernel(img.shape, *beam), mode="same")


def beam_from_mas(major_mas, minor_mas, pa_deg, pixel_mas):
    """``(sx, sy, phi)`` in pixels of a beam given as FWHMs (mas) and PA (East of North) on the row-0-North grid."""
    fw = 2.0 * math.sqrt(2.0 * math.log(2.0))
    return (major_mas / fw / pixel_mas, minor_mas / fw / pixel_mas, math.pi / 2 - math.radians(pa_deg))


def stored_beam(stored, pixel_mas):
    """The restoring beam in pixels from the sibling ``<prefix>_compare.json["beam"]``, or ``None`` if absent."""
    name = os.path.basename(str(stored))
    suffix = "_compare_images.npz"
    if not name.endswith(suffix):
        return None
    f = pathlib.Path(stored).with_name(name[: -len(suffix)] + "_compare.json")
    try:
        b = _load(f).get("beam")
    except (OSError, ValueError, AttributeError):
        return None
    try:
        if isinstance(b, dict):
            vals = [float(b[k]) for k in ("major_mas", "minor_mas", "pa_deg")]
        else:
            vals = [float(x) for x in b[:3]]
    except (KeyError, TypeError, ValueError):
        return None
    if not (all(math.isfinite(v) for v in vals) and vals[0] > 0 and vals[1] > 0):
        return None
    return beam_from_mas(*vals, pixel_mas)


def fit_beam(sharp, blurred):
    """Elliptical Gaussian ``(sx, sy, phi)`` in pixels that maps ``sharp`` to ``blurred``, and the relative rms.

    Fallback when no ``_compare.json`` beam is stored. The starts break the ``sx == sy`` symmetry (on that manifold the
    gradient in ``phi`` vanishes and an elliptical beam would stay circular).
    """
    from scipy.optimize import least_squares

    sharp = np.asarray(sharp, float)
    blurred = np.asarray(blurred, float)

    def res(q):
        return (convolve(sharp, (abs(q[0]), abs(q[1]), q[2])) - blurred).ravel()

    best = None
    for s0 in (1.0, 3.0, 6.0):
        for phi0 in (0.0, math.pi / 4):
            sol = least_squares(res, [1.3 * s0, 0.7 * s0, phi0])
            if best is None or sol.cost < best.cost:
                best = sol
    q = best.x
    rms = float(np.sqrt(2 * best.cost / blurred.size) / max(np.abs(blurred).max(), 1e-300))
    return (abs(q[0]), abs(q[1]), float(q[2])), rms


# --- reading one target ------------------------------------------------------------------


_MODEL_FILE = re.compile(r"^(?P<t>.+)_param_(?P<model>M\d+o?)_(?P<arm>N[012])(?:_(?P<variant>[A-Za-z0-9]+))?\.json$")


def target_dir(out, target):
    return pathlib.Path(out) / target


def model_files(out, target):
    """``{(model, arm, variant): path}`` of the model JSONs of a target (variant None for the ladder)."""
    d = target_dir(out, target)
    found = {}
    for f in sorted(d.glob(f"{target}_param_*.json")):
        m = _MODEL_FILE.match(f.name)
        if m and m["t"] == target:
            found[(m["model"], m["arm"], m["variant"])] = f
    return found


def read_target(out, target):
    """Everything the report stages need for one target (missing files give ``None``/empty)."""
    d = target_dir(out, target)
    files = model_files(out, target)
    models = {k: _load(f) for k, f in files.items()}
    sel_path = d / f"{target}_param_selected.json"
    selected = _load(sel_path) if sel_path.exists() else None
    return dict(target=target, dir=d, models=models, selected=selected)


def _accepted(step):
    """Whether a chain step was accepted: its ``accepted`` flag if it has one, else a verdict starting "accept"."""
    if step.get("accepted") is not None:
        return bool(step["accepted"])
    return str(step.get("verdict", "")).lower().startswith("accept")


def selected_model(tdata, arm="N2"):
    """Selected model id for ``arm``: the ``selected`` key, else the highest accepted rung of the chain."""
    sel = tdata["selected"] or {}
    a = (sel.get("arms") or {}).get(arm) or {}
    if a.get("selected"):
        return a["selected"]
    if a.get("chain") is not None:
        order = list(MODELS)
        chosen = "M0"
        for step in a["chain"]:
            if _accepted(step):
                child = str(step.get("pair", "")).split("/")[0].split(" ")[0]
                if child in order and order.index(child) > order.index(chosen):
                    chosen = child
        return chosen
    return None


def selected_fit(tdata, arm="N2"):
    """The model JSON of the selected model under ``arm`` (no variant), or ``None`` if absent or without an eligible best."""
    m = selected_model(tdata, arm)
    fit = tdata["models"].get((m, arm, None)) if m else None
    return fit if (fit or {}).get("best") else None


# --- stage: render -----------------------------------------------------------------------


def find_stored_compare(stored_dir, target):
    """Stored ``compare_images.npz``: nightw unbounded if it exists, else the newest of the stored ones."""
    if not stored_dir:
        return None
    pats = [f"{target}_gp-polar_nightw_unbounded_compare_images.npz", f"{target}_gp-polar_unbounded_compare_images.npz",
            f"{target}_gp-polar_compare_images.npz", f"{target}_gp-polar*_compare_images.npz"]
    root = pathlib.Path(stored_dir)
    for pat in pats:  # the named products first (the default-noise run for targets without nightw), then any gp-polar one
        hits = sorted(root.rglob(pat))
        if not hits:
            continue
        hits = [h for h in hits if PREFERRED_STORED_RUN in h.parts] or hits
        return str(max(hits, key=os.path.getmtime))
    return None


def render_stage(out, target, stored_dir=None, npix=128, pixel_mas=0.3):
    """Write ``<T>_param_image.npz`` from the selected N2 model; returns the path or ``None`` if there is no fit."""
    t = read_target(out, target)
    fit = selected_fit(t, "N2")
    if fit is None:
        return None
    p = ring_params(fit.get("best"), _derived(fit))
    stored = find_stored_compare(stored_dir, target)
    beam = beam_rms = None
    grid_src = "default"
    if stored:
        z = np.load(stored, allow_pickle=True)
        pixel_mas = float(z["pixel_mas"])
        npix = int(z["ours"].shape[0])
        beam = stored_beam(stored, pixel_mas)
        if beam is None:
            beam, beam_rms = fit_beam(z["ours"], z["ours_conv"])
        else:  # how well the stored beam maps the stored sharp image onto the stored convolved one
            ref = np.asarray(z["ours_conv"], float)
            beam_rms = float(np.sqrt(np.mean((convolve(z["ours"], beam) - ref) ** 2)) / max(np.abs(ref).max(), 1e-300))
        grid_src = os.path.basename(stored)
    img = render_ring(p, npix, pixel_mas)
    if beam is None:
        beam = (1.5 / pixel_mas / 2.355,) * 2 + (0.0,)
    conv = convolve(img, beam)
    conv = conv / conv.sum()
    d = _derived(fit)
    bp = bright_pa(p)
    best = fit.get("best") or {}
    values = best.get("values") or {}
    secondary = dict(
        dra=_num(_value(values, "sec.dra")),
        ddec=_num(_value(values, "sec.ddec")),
        f_sec_pct=_num(d.get("f_sec_pct")),
    )
    path = t["dir"] / f"{target}_param_image.npz"
    np.savez(
        path,
        native=img,
        convolved=conv,
        pixel_mas=pixel_mas,
        shape=np.array(img.shape),
        beam_pix=np.array(beam),
        beam_fit_rel_rms=np.nan if beam_rms is None else beam_rms,
        grid_source=grid_src,
        model=selected_model(t, "N2"),
        inc=p["inc"],
        pa=p["pa"] % 180.0,
        dra=p["dra"],
        ddec=p["ddec"],
        r0=p["r0"],
        sigma=p["sigma"],
        bright_pa=np.nan if bp is None else bp,
        star_dra=0.0,
        star_ddec=0.0,
        star_f_pct=_num(d.get("f_prim_pct")) if d.get("f_prim_pct") is not None else np.nan,
        sec_dra=np.nan if secondary["dra"] is None else secondary["dra"],
        sec_ddec=np.nan if secondary["ddec"] is None else secondary["ddec"],
        sec_f_pct=np.nan if secondary["f_sec_pct"] is None else secondary["f_sec_pct"],
    )
    return path


# --- stage: compare ----------------------------------------------------------------------


def ncc(a, b):
    """Normalised cross-correlation (zero-mean) of two images on one grid."""
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def _unit(img):
    img = np.asarray(img, float)
    s = img.sum()
    return img / s if s != 0 else img


def _draw_beam(ax, z, field):
    """The restoring beam (FWHM ellipse) in the lower-left corner of a sky panel."""
    from matplotlib.patches import Ellipse

    sx, sy, phi = (float(v) for v in z["beam_pix"])
    pix = float(z["pixel_mas"])
    fw = 2.0 * math.sqrt(2.0 * math.log(2.0))
    # an axis at (cos phi, sin phi) in (column, row) pixels is, in (East, North) data axes, the axis at angle phi
    ax.add_patch(Ellipse((0.8 * field, -0.8 * field), fw * sx * pix, fw * sy * pix, angle=math.degrees(phi), fill=False,
                         ec="w", lw=1))


def compare_stage(out, target, stored_dir=None):
    """Write ``<T>_param_compare.png``: parametric / gp-polar / ORGANIC on one scale, residuals below."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = read_target(out, target)
    img_path = t["dir"] / f"{target}_param_image.npz"
    if not img_path.exists():
        return None
    z = np.load(img_path, allow_pickle=True)
    stored = find_stored_compare(stored_dir, target)
    zs = np.load(stored, allow_pickle=True) if stored else None
    pix = float(z["pixel_mas"])
    panels = [("ring (this work)", _unit(z["convolved"]))]
    if zs is not None and zs["ours_conv"].shape == z["convolved"].shape and np.isclose(float(zs["pixel_mas"]), pix):
        panels += [("gp-polar", _unit(zs["ours_conv"])), ("ORGANIC", _unit(zs["published_conv"]))]
    elif zs is not None:  # render was run against another stored grid: do not compare images on different grids
        print(f"{target}: stored products are on a different grid from the rendered image; not plotted", file=sys.stderr)
    half = 0.5 * z["convolved"].shape[0] * pix
    r_out = float(z["r0"]) + 3 * float(z["sigma"])
    field = min(half, max(1.6 * r_out, 6.0))
    vmax = max(float(np.max(p[1])) for p in panels)
    resid = t["dir"] / f"{target}_param_residuals.npz"
    res = np.load(resid) if resid.exists() else None

    fig = plt.figure(figsize=(max(3.6 * len(panels), 7.0), 7.2), constrained_layout=True)
    npan = len(panels)
    gs = fig.add_gridspec(2, 2 * npan, height_ratios=[1.0, 0.6])
    for i, (name, im) in enumerate(panels):
        ax = fig.add_subplot(gs[0, 2 * i : 2 * i + 2])
        mappable = ax.imshow(im, origin="upper", extent=[half, -half, -half, half], vmin=0, vmax=vmax, cmap="inferno")
        ax.set_xlim(field, -field)  # East left
        ax.set_ylim(-field, field)
        ax.plot(0, 0, "c+", ms=10)
        if np.isfinite(z["sec_dra"]) and np.isfinite(z["sec_ddec"]):
            ax.plot(float(z["sec_dra"]), float(z["sec_ddec"]), "wx", ms=7)
        _draw_beam(ax, z, field)
        label = name if i == 0 else f"{name}  NCC {ncc(panels[0][1], im):.2f}"
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("East offset (mas)")
        if i == 0:
            ax.set_ylabel("North offset (mas)")
    fig.colorbar(mappable, ax=fig.axes[:npan], shrink=0.7, label="fraction of flux per pixel")
    axv = fig.add_subplot(gs[1, :npan])
    axc = fig.add_subplot(gs[1, npan:])
    for ax, (bk, rk, ylab) in ((axv, ("v2_baseline_m", "v2_resid", r"V$^2$ residual (quoted $\sigma$)")),
                               (axc, ("cp_baseline_m", "cp_resid", r"closure residual (quoted $\sigma$)"))):
        if res is not None and bk in res.files and rk in res.files:
            ax.plot(res[bk], res[rk], ".", ms=3)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel("baseline (m)")
            ax.set_ylabel(ylab)
        else:
            ax.text(0.5, 0.5, "residuals not available", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
    bright = float(z["bright_pa"])
    fig.suptitle(
        f"{target}: {z['model']}  inc {float(z['inc']):.1f}  PA {float(z['pa']):.1f}  r0 {float(z['r0']):.2f} mas"
        + (f"  brightest PA {bright:.0f}" if np.isfinite(bright) else ""),
        fontsize=10,
    )
    path = t["dir"] / f"{target}_param_compare.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --- jackknife, flags, usable_for_gp (plan section 9) ------------------------------------


def jackknife_sds(tdata):
    j = (tdata["selected"] or {}).get("jackknife") or {}
    if isinstance(j.get("sd"), dict):
        j = j["sd"]
    return {k: _num(v) for k, v in j.items() if _num(v) is not None}


def _fit_flags(fit):
    return list(((fit or {}).get("best") or {}).get("flags") or [])


def _accepted_steps(tdata, arm="N2"):
    a = ((tdata["selected"] or {}).get("arms") or {}).get(arm) or {}
    return [s for s in a.get("chain") or [] if _accepted(s)]


def _selected_vs_m0_steps(tdata):
    """Accepted N2 chain steps that test the *selected* model against M0 (``M1/M0``, or ``M2/M0``, ``M2o/M0`` when M1 has
    no eligible fit). Steps between unselected intermediate rungs do not bear on the selected evidence: their flags (an
    unselected rung with a low ESS, say) say nothing about a selected fit and M0 that are themselves verified."""
    model = selected_model(tdata, "N2")
    out = []
    for s in _accepted_steps(tdata):
        child, _, parent = str(s.get("pair", "")).replace(" ", "").partition("/")
        if child == model and (s.get("against") or s.get("parent") or parent) == "M0":
            out.append(s)
    return out


def _unverified_sources(tdata, fit):
    """Where ``evidence_unverified`` is flagged under N2: M0, the selected fit, or the selected-versus-M0 chain step."""
    out = []
    if "evidence_unverified" in _fit_flags(tdata["models"].get(("M0", "N2", None))):
        out.append("M0")
    if "evidence_unverified" in _fit_flags(fit):
        out.append("the selected fit")
    for step in _selected_vs_m0_steps(tdata):
        if "evidence_unverified" in (step.get("flags") or []):
            out.append(f"accepted step {str(step.get('pair', '?')).replace(' ', '')}")
    return out


def _chain_flags(tdata):
    """Flags of the accepted N2 step of the selected model against M0."""
    return {f for s in _selected_vs_m0_steps(tdata) for f in s.get("flags") or []}


def multimodal_competitor(fit, p):
    """A distinct eligible mode within 5 of the best ``lnZ_lap`` that differs by > 15 deg in inc or > 30 in PA."""
    modes = (fit or {}).get("modes") or []
    if len(modes) < 2:
        return False
    best_lap = _num(modes[0].get("lnZ_lap"))
    for m in modes[1:]:
        if not m.get("eligible", True):
            continue
        lap = _num(m.get("lnZ_lap"))
        if best_lap is None or lap is None or best_lap - lap > 5:
            continue
        v = m.get("values") or {}
        inc, pa = _num(_value(v, "env.inc")), _num(_value(v, "env.pa"))
        if inc is None or pa is None:
            continue
        if abs(inc - p["inc"]) > 15 or abs(_angdiff180(pa, p["pa"])) > 30:
            return True
    return False


def usable_for_gp(tdata):
    """``(usable, reasons, details)`` by the plan's section 9 rule, under N2."""
    reasons = []
    fit = selected_fit(tdata, "N2")
    model = selected_model(tdata, "N2")
    if fit is None:
        return False, ["no N2 selected fit"], {}
    best = fit.get("best") or {}
    derived = _derived(fit)
    p = ring_params(best, derived)
    hc = _num(derived.get("hole_contrast"))
    if hc is None and p["sigma"]:
        hc = hole_contrast(p)
    a = ((tdata["selected"] or {}).get("arms") or {}).get("N2") or {}
    is_ring = a.get("is_ring")
    if is_ring is None:
        is_ring = model not in (None, "M0") and hc is not None and hc >= 0.5
    if not is_ring:
        reasons.append("selected model is not a ring")
    modes = fit.get("modes") or []
    if modes and not modes[0].get("eligible", True):
        reasons.append("selected mode is not eligible")
    unverified = _unverified_sources(tdata, fit)
    if unverified:
        reasons.append("evidence_unverified on " + ", ".join(unverified))
    jk = jackknife_sds(tdata)
    if not jk:
        reasons.append("no jackknife")
    else:
        if jk.get("inc") is not None and jk["inc"] > 10:
            reasons.append(f"jackknife sd(inc) {jk['inc']:.1f} > 10 deg")
        low_inc = p["inc"] is not None and p["inc"] < 20
        if jk.get("pa") is not None and jk["pa"] > 15 and not low_inc:
            reasons.append(f"jackknife sd(PA) {jk['pa']:.1f} > 15 deg")
    if multimodal_competitor(fit, p):
        reasons.append("multimodal competitor differing by > 15 deg in inc or > 30 deg in PA")
    return (not reasons), reasons, dict(model=model, is_ring=bool(is_ring), hole_contrast=hc, params=p)


# --- stage: geometry ---------------------------------------------------------------------


def _de_prins(target, image):
    e = image.get(target) or {}
    th, _ = _lit(e, "theta_rim")
    inc, _ = _lit(e, "i_rim")
    pa, _ = _lit(e, "pa_rim")
    return dict(inc=inc, pa=pa, r0=None if th is None else th / 2.0)


def geometry_entry(tdata, image):
    """The entry of ``postagb_parametric_geometry.json`` for one target (rule of plan section 9)."""
    target = tdata["target"]
    dp = _de_prins(target, image)
    jk = jackknife_sds(tdata)
    usable, reasons, det = usable_for_gp(tdata)
    entry = dict(
        target=target,
        usable_for_gp=bool(usable),
        jackknife_sd=jk or None,
        model=det.get("model"),
        reasons=reasons,
    )
    ours = None
    if det:
        p = det["params"]
        d = _derived(selected_fit(tdata, "N2"))
        pa, c, s = wrap_pa(p["pa"], list(p["c"]) if p["c"].size else d.get("c"),
                           list(p["s"]) if p["s"].size else d.get("s"))
        ours = dict(
            inc=p["inc"], pa=pa, r0=p["r0"], sigma=p["sigma"],
            c=c, s=s, dra=p["dra"], ddec=p["ddec"], d_env=_num(d.get("d_env")),
        )
    use_ours = bool(usable and ours)
    branch = "ours"
    if target == "AI_Sco":
        if usable and ours:
            near = abs(ours["inc"] - dp["inc"]) <= 15 and abs(_angdiff180(ours["pa"], dp["pa"])) <= 25
            if not near:
                use_ours = False
                branch = "ours usable but elsewhere: De Prins image geometry"
                reasons = reasons + [
                    f"ours (inc {ours['inc']:.1f}, PA {ours['pa']:.1f}) is more than 15 deg in inc or 25 deg in PA "
                    f"from the De Prins image ellipse ({dp['inc']:.0f}, {dp['pa']:.0f})"
                ]
        else:
            branch = "no ring or unstable: De Prins image geometry"
    if use_ours:
        entry.update(ours)
        entry.update(source="ours", branch=branch if target == "AI_Sco" else None)
    else:
        if not reasons:
            reasons = ["no parametric result"]
        entry.update(
            inc=dp["inc"], pa=dp["pa"], r0=dp["r0"], sigma=None, c=None, s=None, dra=None, ddec=None, d_env=None,
            source="De Prins image", branch=branch if target == "AI_Sco" else None,
            reasons=reasons, ours=ours,
        )
    entry["reasons"] = reasons
    return entry


def geometry_stage(out, targets=None, outfile=None):
    image, _ = literature()
    result = {}
    for T in targets or present_targets(out):
        result[T] = geometry_entry(read_target(out, T), image)
    path = pathlib.Path(outfile) if outfile else pathlib.Path(out) / "postagb_parametric_geometry.json"
    path.write_text(json.dumps(result, indent=2, default=_jsonable))
    return path


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def present_targets(out):
    out = pathlib.Path(out)
    # only directories holding this campaign's files (a stray backup directory is not a target)
    have = {p.name for p in out.iterdir() if p.is_dir() and (any(p.glob(f"{p.name}_param_*.json")) or not any(p.iterdir()))} \
        if out.exists() else set()
    return [T for T in TARGETS if T in have] + sorted(have - set(TARGETS))


# --- stage: table ------------------------------------------------------------------------


def _best(fit, key):
    return ((fit or {}).get("best") or {}).get(key)


def table_row(tdata, image, pmoired):
    """One row (dict) of the first table; cells that were not run are ``None``."""
    T = tdata["target"]
    fit = selected_fit(tdata, "N2")
    model = selected_model(tdata, "N2")
    sel_arms = (tdata["selected"] or {}).get("arms") or {}
    d = _derived(fit)
    p = ring_params((fit or {}).get("best"), d) if fit else None
    jk = jackknife_sds(tdata)
    usable, reasons, det = usable_for_gp(tdata)
    row = dict(
        target=T,
        selected=model,
        is_ring=det.get("is_ring") if det else None,
        flags=sorted(set(_fit_flags(fit)) | set((sel_arms.get("N2") or {}).get("flags") or []) | _chain_flags(tdata)),
        arm_dependent=None,
        usable_for_gp=bool(usable) if fit else None,
    )
    if fit:
        pa_w, c, s = wrap_pa(p["pa"], d.get("c") or [], d.get("s") or [])
        if pa_w is None:
            pa_w = _num(d.get("pa_mod180"))
        row.update(
            inc=_num(d.get("inc", p["inc"])), pa=pa_w,
            theta_rim=None if p["r0"] is None else 2 * p["r0"],
            sigma_over_r0=None if not p["r0"] else p["sigma"] / p["r0"],
            hole_contrast=_num(d.get("hole_contrast")) if d.get("hole_contrast") is not None else hole_contrast(p),
            c1=_num(c[0]) if len(c) else None, s1=_num(s[0]) if len(s) else None, a1=_num(d.get("a1")),
            bright_pa=_num(d.get("bright_pa")) if d.get("bright_pa") is not None else bright_pa(p),
            centre=d.get("centre") or [p["dra"], p["ddec"]],
            f_prim_pct=_num(d.get("f_prim_pct")), f_sec_pct=_num(d.get("f_sec_pct")),
            f_disc_pct=_num(d.get("f_disc_pct")), f_bg_pct=_num(d.get("f_bg_pct")), d_env=_num(d.get("d_env")),
            raw_chi2=_num(_best(fit, "raw_chi2")), raw_chi2_v2=_num(_best(fit, "raw_chi2_v2")),
            raw_chi2_cp=_num(_best(fit, "raw_chi2_cp")), own_chi2=_num(_best(fit, "own_chi2")),
        )
    sel_n1 = selected_model(tdata, "N1")
    sel_n2 = model
    if sel_n1 and sel_n2:
        f1 = selected_fit(tdata, "N1")
        flagged = bool("arm_dependent" in _fit_flags(fit) or (tdata["selected"] or {}).get("arm_dependent")
                       or sel_n1 != sel_n2 or (f1 is None) != (fit is None))
        if f1:
            a = ring_params(f1.get("best"), _derived(f1))
            if p and None not in (a["inc"], a["pa"], p["inc"], p["pa"]):
                flagged = flagged or abs(a["inc"] - p["inc"]) > 15 or abs(_angdiff180(a["pa"], p["pa"])) > 25
        row["arm_dependent"] = flagged
    row["lnZ_use"] = {
        arm: {m: _num(_best(tdata["models"].get((m, arm, None)), "lnZ_use")) for m in MODELS
              if (m, arm, None) in tdata["models"]}
        for arm in ("N1", "N2")
    }
    row["status"] = {
        arm: {m: fit_status(tdata, m, arm) for m in MODELS if m in ("M0", "M1", "M2", "M2o") or (m, arm, None) in tdata["models"]}
        for arm in ("N0", "N1", "N2")
    }
    bic0 = _num(_best(tdata["models"].get(("M0", "N0", None)), "bic"))
    row["dbic_n0"] = {
        m: (None if bic0 is None or _num(_best(tdata["models"][(m, "N0", None)], "bic")) is None
            else _num(_best(tdata["models"][(m, "N0", None)], "bic")) - bic0)
        for m in MODELS if (m, "N0", None) in tdata["models"] and m != "M0"
    }
    gp = (tdata["selected"] or {}).get("gp_polar") or {}
    row["gp_delta"] = {arm: _num((gp.get(arm) or {}).get("delta")) for arm in ("N1", "N2")}
    row["gp_notes"] = {arm: _gp_note(gp.get(arm)) for arm in ("N1", "N2")}
    row["gp_provisional"] = {arm: bool((gp.get(arm) or {}).get("provisional")) for arm in ("N1", "N2")}
    row["variants"] = {
        f"{m}/{arm}/{v}": (fit_status(tdata, m, arm, v) or f"lnZ_use {_fmt(_best(f, 'lnZ_use'), 1)}")
        for (m, arm, v), f in sorted(tdata["models"].items(), key=lambda kv: tuple(str(x) for x in kv[0])) if v
    }
    row["jackknife_sd"] = jk or None
    # literature comparison
    lit = {}
    for key, ours in (("theta_rim", row.get("theta_rim")), ("i_rim", row.get("inc")), ("pa_rim", row.get("pa"))):
        v_im, s_im = _lit(image.get(T), key)
        v_pm, s_pm = _lit(pmoired.get(T), key)
        jkey = {"theta_rim": None, "i_rim": "inc", "pa_rim": "pa"}[key]
        lit[key] = dict(
            image=dict(value=v_im, sigma=s_im), pmoired=dict(value=v_pm, sigma=s_pm), ours=ours,
            ours_sd=jk.get(jkey) if jkey else (None if jk.get("r0") is None else 2 * jk["r0"]),
        )
    row["literature"] = lit
    return row


def _gp_note(g):
    """Why a gp-polar comparison is n/a: the core's ``note`` (``mismatch`` is a boolean), else a bare mismatch string."""
    g = g or {}
    if g.get("note"):
        return str(g["note"])
    mm = g.get("mismatch")
    if isinstance(mm, str):
        return mm
    return "mismatch" if mm is True else None


def _nsig(ours, lit, sigma, axis=False):
    if ours is None or lit is None or not sigma:
        return None
    d = _angdiff180(ours, lit) if axis else ours - lit
    return d / sigma


def table_stage(out, targets=None, outdir=None):
    image, pmoired = literature()
    rows = [table_row(read_target(out, T), image, pmoired) for T in targets or present_targets(out)]
    outdir = pathlib.Path(outdir or out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "postagb_parametric_table.json").write_text(json.dumps(rows, indent=2, default=_jsonable))
    (outdir / "postagb_parametric_table.md").write_text(render_markdown(rows))
    return outdir / "postagb_parametric_table.md"


def _lnz_cell(d, status=None):
    """``M0 -123.4 / M1 no eligible fit (bound) / ...``; a missing value shows the fit's own status."""
    status = status or {}
    models = [m for m in MODELS if m in (d or {}) or m in status]
    if not models:
        return "not run"
    return " / ".join(
        f"{m} {_fmt(d[m], 1)}" if _num((d or {}).get(m)) is not None else f"{m} {status.get(m) or 'not run'}" for m in models
    )


def _dbic_cell(d, status=None):
    status = status or {}
    models = [m for m in MODELS if m != "M0" and (m in (d or {}) or m in status)]
    if not models:
        return "not run"
    out = []
    for m in models:
        v = _num((d or {}).get(m))
        if v is not None:
            out.append(f"{m} {v:.0f}{'*' if v <= -100 else ''}")
        else:
            out.append(f"{m} {status.get(m) or (status.get('M0') and 'n/a (M0: ' + status['M0'] + ')') or 'not run'}")
    return " / ".join(out)


def render_markdown(rows):
    L = ["# Parametric ring fits: results table", "",
         "Selection under N2 (night gains and offsets over the MacKay-scaled errors; it counts the night scatter twice and "
         "so under-selects). Raw chi2 uses the quoted errors with no gains; own chi2 is the arm's likelihood. "
         "`*` marks Delta BIC <= -100 (De Prins threshold). Tables only; no statement for the paper.", ""]
    h = ["target", "selected", "ring", "flags", "usable_for_gp", "inc", "PA", "theta_rim", "sigma/r0", "hole", "c1", "s1",
         "a1", "bright PA", "centre (E, N)", "f_prim %", "f_sec %", "f_disc %", "f_bg %", "d_env",
         "raw chi2 (all/V2/CP)", "own chi2", "lnZ_use N1", "lnZ_use N2", "dBIC N0", "dlnZ gp N1", "dlnZ gp N2"]
    L += ["| " + " | ".join(h) + " |", "|" + "---|" * len(h)]
    for r in rows:
        if r["selected"] is None:
            L.append(f"| {r['target']} | not run |" + " not run |" * (len(h) - 2))
            continue
        centre = r.get("centre")
        cen = "n/a" if not centre else f"({_fmt_na(centre[0])}, {_fmt_na(centre[1])})"
        raw = "/".join(_fmt_na(r.get(k)) for k in ("raw_chi2", "raw_chi2_v2", "raw_chi2_cp"))
        flags = ", ".join(r["flags"] + (["arm_dependent"] if r.get("arm_dependent") else [])) or "none"
        L.append(
            "| " + " | ".join([
                r["target"], str(r["selected"]), "yes" if r["is_ring"] else "no", flags,
                "yes" if r["usable_for_gp"] else "no",
                _fmt_na(r.get("inc"), 1), _fmt_na(r.get("pa"), 1), _fmt_na(r.get("theta_rim")),
                _fmt_na(r.get("sigma_over_r0")), _fmt_na(r.get("hole_contrast")), _fmt_na(r.get("c1"), 3),
                _fmt_na(r.get("s1"), 3), _fmt_na(r.get("a1"), 3), _fmt_na(r.get("bright_pa"), 0), cen,
                _fmt_na(r.get("f_prim_pct"), 1), _fmt_na(r.get("f_sec_pct"), 1), _fmt_na(r.get("f_disc_pct"), 1),
                _fmt_na(r.get("f_bg_pct"), 1), _fmt_na(r.get("d_env")), raw, _fmt_na(r.get("own_chi2"), 3),
                _lnz_cell(r["lnZ_use"]["N1"], r["status"]["N1"]), _lnz_cell(r["lnZ_use"]["N2"], r["status"]["N2"]),
                _dbic_cell(r["dbic_n0"], r["status"]["N0"]),
                _gp_cell(r, "N1"), _gp_cell(r, "N2"),
            ]) + " |"
        )
    var = [(r["target"], k, v) for r in rows for k, v in (r.get("variants") or {}).items()]
    if var:
        L += ["", "## Variant fits (P2: no secondary, no background, ...)", "", "| target | model/arm/variant | result |", "|---|---|---|"]
        L += [f"| {a} | {b} | {c} |" for a, b, c in var]
    L += ["", "## Literature comparison (De Prins et al. image ellipse / PMOIRED rim / this work)", "",
          "theta_rim is a diameter (mas), inc and PA in degrees. Ours is the selected N2 model with its jackknife sd; "
          "n_sigma is the difference to the quoted literature sigma.", ""]
    h2 = ["target", "quantity", "De Prins image", "PMOIRED", "ours (N2)", "n_sigma image", "n_sigma PMOIRED"]
    L += ["| " + " | ".join(h2) + " |", "|" + "---|" * len(h2)]
    names = {"theta_rim": "theta_rim", "i_rim": "inc", "pa_rim": "PA"}
    for r in rows:
        for key, nm in names.items():
            c = r["literature"][key]
            ax = key == "pa_rim"

            def cell(e):
                return "n/a" if e["value"] is None else (
                    f"{e['value']:g}" + (f" +/- {e['sigma']:g}" if e["sigma"] else "")
                )

            ours = c["ours"]
            ours_s = "not run" if ours is None else f"{ours:.2f}" + (
                f" +/- {c['ours_sd']:.2f}" if c["ours_sd"] is not None else ""
            )
            n_im = _nsig(ours, c["image"]["value"], c["image"]["sigma"], ax)
            n_pm = _nsig(ours, c["pmoired"]["value"], c["pmoired"]["sigma"], ax)
            L.append(f"| {r['target']} | {nm} | {cell(c['image'])} | {cell(c['pmoired'])} | {ours_s} | "
                     f"{_fmt_na(n_im, 1)} | {_fmt_na(n_pm, 1)} |")
    return "\n".join(L) + "\n"


def _gp_cell(r, arm):
    d = r["gp_delta"].get(arm)
    note = r["gp_notes"].get(arm)
    if d is None:
        return f"n/a ({note})" if note else "n/a"
    tag = " (not decisive)" if abs(d) < 20 else ""
    prov = " (provisional)" if (r.get("gp_provisional") or {}).get(arm) else ""
    return f"{d:+.1f}{tag}{prov}"


# --- CLI ---------------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=["render", "compare", "table", "geometry"])
    ap.add_argument("out", help="OUT directory holding <T>/ subdirectories")
    ap.add_argument("--target", help="one target (render, compare); default all present")
    ap.add_argument("--stored-dir", help="directory of the stored gp-polar/ORGANIC products (compare_images.npz)")
    ap.add_argument("--outdir", help="table output directory (default OUT)")
    a = ap.parse_args(argv)
    targets = [a.target] if a.target else present_targets(a.out)
    if a.stage == "render":
        for T in targets:
            print(T, render_stage(a.out, T, a.stored_dir))
    elif a.stage == "compare":
        for T in targets:
            print(T, compare_stage(a.out, T, a.stored_dir))
    elif a.stage == "table":
        print(table_stage(a.out, targets if a.target else None, a.outdir))
    else:
        print(geometry_stage(a.out, targets if a.target else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
