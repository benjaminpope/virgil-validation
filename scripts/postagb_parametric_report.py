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
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pathlib
import re
import sys

import numpy as np

TARGETS = ["AI_Sco", "EN_TrA", "IRAS15469-5311", "IW_Car", "HD95767", "HD108015", "HR4049", "PS_Gem"]
ARMS = ("N0", "N1", "N2")
MODELS = ("M0", "M1", "M2", "M2o", "M3", "M5")
M0_RADIUS = 0.01  # mas, the fixed radius of the centred Gaussian

# IW Car is in neither postagb_paper dict (its image values are in postagb_imaging.TARGETS).
_IW_CAR_IMAGE = dict(theta_rim=(6.9, None, None), i_rim=(48.0, None, None), pa_rim=(144.0, None, None))
_AI_SCO_CENTRE = (0.51, -1.4)  # De Prins image centre, mas (East, North)


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
    """On-sky PA (0-360, East of North, about the ring centre) of the brightest point of the ring.

    The profile is separable, so the maximum is on the ring radius at the azimuth maximising the factor.
    """
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


def _gauss_otf(shape, sx, sy, phi):
    """Fourier transform (sum-preserving) of an elliptical Gaussian, sigmas ``sx, sy`` pixels, rotated by ``phi``."""
    ky = np.fft.fftfreq(shape[0])[:, None]
    kx = np.fft.fftfreq(shape[1])[None, :]
    c, s = math.cos(phi), math.sin(phi)
    u = kx * c + ky * s
    v = -kx * s + ky * c
    return np.exp(-2 * np.pi**2 * ((sx * u) ** 2 + (sy * v) ** 2))


def convolve(img, beam):
    """Convolve with a beam ``(sx, sy, phi)`` in pixels."""
    return np.real(np.fft.ifft2(np.fft.fft2(img) * _gauss_otf(img.shape, *beam)))


def fit_beam(sharp, blurred):
    """Elliptical Gaussian ``(sx, sy, phi)`` in pixels that maps ``sharp`` to ``blurred``, and the relative rms."""
    from scipy.optimize import least_squares

    sharp = np.asarray(sharp, float)
    blurred = np.asarray(blurred, float)

    def res(q):
        return (convolve(sharp, (abs(q[0]), abs(q[1]), q[2])) - blurred).ravel()

    best = None
    for s0 in (1.0, 3.0, 6.0):
        sol = least_squares(res, [s0, s0, 0.0])
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
            if str(step.get("verdict", "")).lower().startswith("accept"):
                child = str(step.get("pair", "")).split("/")[0].split(" ")[0]
                if child in order and order.index(child) > order.index(chosen):
                    chosen = child
        return chosen
    return None


def selected_fit(tdata, arm="N2"):
    """The model JSON of the selected model under ``arm`` (no variant), or ``None``."""
    m = selected_model(tdata, arm)
    return tdata["models"].get((m, arm, None)) if m else None


# --- stage: render -----------------------------------------------------------------------


def find_stored_compare(stored_dir, target):
    """Stored ``compare_images.npz``: nightw unbounded if it exists, else the newest of the stored ones."""
    if not stored_dir:
        return None
    pats = [f"{target}_gp-polar_nightw_unbounded_compare_images.npz", f"{target}_gp-polar*_compare_images.npz"]
    for pat in pats:
        hits = []
        for root in {stored_dir, *glob.glob(os.path.join(stored_dir, "*"))}:
            hits += glob.glob(os.path.join(root, pat))
        if hits:
            return max(hits, key=os.path.getmtime) if pat == pats[-1] else hits[0]
    return None


def render_stage(out, target, stored_dir=None, npix=128, pixel_mas=0.3):
    """Write ``<T>_param_image.npz`` from the selected N2 model; returns the path or ``None`` if there is no fit."""
    t = read_target(out, target)
    fit = selected_fit(t, "N2")
    if fit is None:
        return None
    p = ring_params(fit.get("best"), fit.get("derived"))
    stored = find_stored_compare(stored_dir, target)
    beam = beam_rms = None
    grid_src = "default"
    if stored:
        z = np.load(stored, allow_pickle=True)
        pixel_mas = float(z["pixel_mas"])
        npix = int(z["ours"].shape[0])
        beam, beam_rms = fit_beam(z["ours"], z["ours_conv"])
        grid_src = os.path.basename(stored)
    img = render_ring(p, npix, pixel_mas)
    if beam is None:
        beam = (1.5 / pixel_mas / 2.355,) * 2 + (0.0,)
    conv = convolve(img, beam)
    conv = conv / conv.sum()
    d = fit.get("derived") or {}
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
        pa=p["pa"],
        r0=p["r0"],
        sigma=p["sigma"],
        bright_pa=bright_pa(p),
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
    if zs is not None:
        panels += [("gp-polar", _unit(zs["ours_conv"])), ("ORGANIC", _unit(zs["published_conv"]))]
    half = 0.5 * z["convolved"].shape[0] * pix
    r_out = float(z["r0"]) + 3 * float(z["sigma"])
    field = min(half, max(1.6 * r_out, 6.0))
    vmax = max(float(np.max(p[1])) for p in panels)
    resid = t["dir"] / f"{target}_param_residuals.npz"
    res = np.load(resid) if resid.exists() else None

    fig = plt.figure(figsize=(3.6 * len(panels), 7.2), constrained_layout=True)
    npan = len(panels)
    gs = fig.add_gridspec(2, 2 * npan, height_ratios=[1.0, 0.6])
    for i, (name, im) in enumerate(panels):
        ax = fig.add_subplot(gs[0, 2 * i : 2 * i + 2])
        ax.imshow(im, origin="upper", extent=[half, -half, -half, half], vmin=0, vmax=vmax, cmap="inferno")
        ax.set_xlim(field, -field)  # East left
        ax.set_ylim(-field, field)
        ax.plot(0, 0, "c+", ms=10)
        if np.isfinite(z["sec_dra"]) and np.isfinite(z["sec_ddec"]):
            ax.plot(float(z["sec_dra"]), float(z["sec_ddec"]), "wx", ms=7)
        label = name if i == 0 else f"{name}  NCC {ncc(panels[0][1], im):.2f}"
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("East offset (mas)")
        if i == 0:
            ax.set_ylabel("North offset (mas)")
    axv = fig.add_subplot(gs[1, :npan])
    axc = fig.add_subplot(gs[1, npan:])
    for ax, (bk, rk, ylab) in ((axv, ("v2_baseline_m", "v2_resid", r"V$^2$ residual ($\sigma$)")),
                               (axc, ("cp_baseline_m", "cp_resid", r"closure residual ($\sigma$)"))):
        if res is not None and bk in res.files and rk in res.files:
            ax.plot(res[bk], res[rk], ".", ms=3)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel("baseline (m)")
            ax.set_ylabel(ylab)
        else:
            ax.text(0.5, 0.5, "residuals not available", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
    fig.suptitle(
        f"{target}: {z['model']}  inc {float(z['inc']):.1f}  PA {float(z['pa']):.1f}  "
        f"r0 {float(z['r0']):.2f} mas  brightest PA {float(z['bright_pa']):.0f}",
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


def _m1_over_m0_unverified(tdata):
    """True if ``evidence_unverified`` is flagged for the M1-over-M0 pair under N2."""
    a = ((tdata["selected"] or {}).get("arms") or {}).get("N2") or {}
    for step in a.get("chain") or []:
        if str(step.get("pair", "")).replace(" ", "").startswith("M1/M0") and "evidence_unverified" in (
            step.get("flags") or []
        ):
            return True
    return any(
        "evidence_unverified" in _fit_flags(tdata["models"].get((m, "N2", None))) for m in ("M0", "M1")
    )


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
    derived = fit.get("derived") or {}
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
    if _m1_over_m0_unverified(tdata):
        reasons.append("evidence_unverified for the M1-over-M0 pair")
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
        d = (selected_fit(tdata, "N2") or {}).get("derived") or {}
        c = list(p["c"]) if p["c"].size else d.get("c")
        s = list(p["s"]) if p["s"].size else d.get("s")
        ours = dict(
            inc=p["inc"], pa=p["pa"] % 180.0 if p["pa"] is not None else None, r0=p["r0"], sigma=p["sigma"],
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
    have = {p.name for p in out.iterdir() if p.is_dir()} if out.exists() else set()
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
    d = (fit or {}).get("derived") or {}
    p = ring_params((fit or {}).get("best"), d) if fit else None
    jk = jackknife_sds(tdata)
    usable, reasons, det = usable_for_gp(tdata)
    row = dict(
        target=T,
        selected=model,
        is_ring=det.get("is_ring") if det else None,
        flags=sorted(set(_fit_flags(fit)) | set((sel_arms.get("N2") or {}).get("flags") or [])),
        arm_dependent=None,
        usable_for_gp=bool(usable) if fit else None,
    )
    if fit:
        c, s = d.get("c") or [], d.get("s") or []
        row.update(
            inc=_num(d.get("inc", p["inc"])), pa=_num(d.get("pa_mod180", p["pa"])),
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
        flagged = "arm_dependent" in _fit_flags(fit)
        if f1:
            a = ring_params(f1.get("best"), f1.get("derived"))
            flagged = flagged or sel_n1 != sel_n2
            if p:
                flagged = flagged or abs(a["inc"] - p["inc"]) > 15 or abs(_angdiff180(a["pa"], p["pa"])) > 25
        row["arm_dependent"] = flagged
    row["lnZ_use"] = {
        arm: {m: _num(_best(tdata["models"].get((m, arm, None)), "lnZ_use")) for m in MODELS
              if (m, arm, None) in tdata["models"]}
        for arm in ("N1", "N2")
    }
    bic0 = _num(_best(tdata["models"].get(("M0", "N0", None)), "bic"))
    row["dbic_n0"] = {
        m: (None if bic0 is None or _num(_best(tdata["models"][(m, "N0", None)], "bic")) is None
            else _num(_best(tdata["models"][(m, "N0", None)], "bic")) - bic0)
        for m in MODELS if (m, "N0", None) in tdata["models"] and m != "M0"
    }
    gp = (tdata["selected"] or {}).get("gp_polar") or {}
    row["gp_delta"] = {arm: _num((gp.get(arm) or {}).get("delta")) for arm in ("N1", "N2")}
    row["gp_notes"] = {arm: (gp.get(arm) or {}).get("mismatch") or (gp.get(arm) or {}).get("note") for arm in ("N1", "N2")}
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


def _lnz_cell(d):
    if not d:
        return "not run"
    return " / ".join(f"{m} {_fmt(v, 1)}" for m, v in d.items())


def _dbic_cell(d):
    if not d:
        return "not run"
    return " / ".join(f"{m} {_fmt(v, 0)}{'*' if _num(v) is not None and v <= -100 else ''}" for m, v in d.items())


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
                _lnz_cell(r["lnZ_use"]["N1"]), _lnz_cell(r["lnZ_use"]["N2"]), _dbic_cell(r["dbic_n0"]),
                _gp_cell(r, "N1"), _gp_cell(r, "N2"),
            ]) + " |"
        )
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
    return f"{d:+.1f}{tag}"


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
