"""The IM Lup report of the registered weight rule (``lambda-report`` of ``scripts/alma_imlup_maxent.py``).

Reads the stage records of a ``lambda-rule`` run directory (``sigma.json``, ``d4.json``, ``sweep.json``, ``lambda_final.json``, ``final_values.npz``) and writes ``report.md`` and its figures.
Nothing is refitted and no threshold is chosen here: every one is the constant of ``alma_lambda`` that implements ``lambda_selection`` of ``design/alma_imlup_gate_criteria.yml``.
A stage that failed shows its reason and the stages after it are skipped. Numpy, SciPy and matplotlib only (no virgil, no JAX)."""

import json
import pathlib
import re

import numpy as np

from . import alma_lambda as vl

STAGES = (("sigma", "sigma.json"), ("d4", "d4.json"), ("sweep", "sweep.json"), ("final", "lambda_final.json"))
VALUES = "final_values.npz"
RA_DEG, DEC_DEG = (15 + 56 / 60 + 9.20 / 3600) * 15, -(37 + 56 / 60 + 6.42 / 3600)  # the MS's phase centre (docs/alma_conventions.md)
DISC_PA_DEG = 144.5  # IM Lup, Huang et al. 2018 (DSHARP II): an orientation check, not a registered threshold
DSHARP_BEAM = (44.0, 43.0, 115.0)  # DSHARP I Table 4 (mas, mas, deg), when no CLEAN image is found
MPOL_PIXEL_MAS = 5.0
CONTRAST_CAVEAT = (
    "**Known bias (item 9, stated in advance).** On synthetic spiral discs this arm under-recovers spiral and ring contrast by about 20% at the chosen w "
    "(m = 2 amplitude ratio 0.77 and 0.79). Any contrast claim made from this image carries that caveat."
)
HALF_ARCSEC = 1.5  # radius of the displayed field
BEGIN, END = "<!-- gen:lambda-report -->", "<!-- /gen:lambda-report -->"


# ---------------------------------------------------------------- reading


def read_run(run):
    """``(records, values)``: the stage records found (``None`` for a stage never run) and ``final_values.npz`` as a dict (or ``None``)."""
    run = pathlib.Path(run)
    rec = {name: (json.loads((run / f).read_text()) if (run / f).exists() else None) for name, f in STAGES}
    z = run / VALUES
    return rec, ({k: np.load(z)[k] for k in np.load(z).files} if z.exists() else None)


def locate_mpol(path):
    """MPoL's export (``export.npz`` with ``image`` in Jy/arcsec^2, row 0 North) as ``(image, pixel_mas, note)``; ``image`` is ``None`` with the reason in ``note`` when it is not there."""
    if not path:
        return None, MPOL_PIXEL_MAS, "no MPoL export given"
    p = pathlib.Path(path).expanduser()
    p = p / "export" if (p / "export" / "export.npz").exists() else p
    if not (p / "export.npz").exists():
        return None, MPOL_PIXEL_MAS, f"no export.npz under {path}"
    z = np.load(p / "export.npz")
    px = float(z["cell_size_arcsec"]) * 1e3 if "cell_size_arcsec" in z.files else MPOL_PIXEL_MAS
    return np.asarray(z["image"], float), px, f"{p / 'export.npz'}"


def locate_clean(path):
    """Our CLEAN image (``image.fits``) in Jy/arcsec^2 on its own grid, with its FITS array and WCS: ``(dict or None, note)``. ``beam`` is ``(bmaj, bmin, PA)`` in mas and degrees."""
    if not path:
        return None, "no CLEAN directory given"
    p = pathlib.Path(path).expanduser()
    if not (p / "image.fits").exists():
        return None, f"no image.fits under {path}"
    from crosscheck import alma_maxent as cx

    data, wcs, hdr = cx.read_fits(p / "image.fits")
    beam = cx.beam_of(hdr)
    return dict(fits=np.nan_to_num(data), wcs=wcs, beam=beam, area=cx.beam_area_arcsec2(beam[0], beam[1])), f"{p / 'image.fits'}"


def clean_on_grid(clean, npix, pixel_mas):
    """The CLEAN image (Jy/arcsec^2) on the fit's ``npix`` x ``npix`` grid (this repository's layout: row 0 North, phase-centre cell ``npix // 2 - 1``)."""
    from crosscheck import alma_maxent as cx

    out = cx.resample(clean["fits"], clean["wcs"], cx.sky_wcs(npix, pixel_mas, RA_DEG, DEC_DEG), (npix, npix))[::-1]
    return out / clean["area"]


def mpol_on_grid(image, src_pixel_mas, npix, pixel_mas):
    """MPoL's image on the fit's grid, flux-preserving (``alma_start.regrid``; MPoL's phase-centre cell is ``npix // 2 - 1`` as ours)."""
    from crosscheck import alma_maxent as cx

    from .alma_start import regrid

    return regrid(image, src_pixel_mas, pixel_mas, npix, cx.phase_centre_cell(image.shape[0]))[0]


# ---------------------------------------------------------------- images


def blur(image, pixel_mas, beam, fraction=1.0):
    """``image`` convolved with the Gaussian beam ``(bmaj, bmin, PA)`` (FWHM in mas, PA East of North) scaled by ``fraction``; row 0 North, column 0 East (column increases towards West)."""
    bmaj, bmin, pa = (fraction * beam[0], fraction * beam[1], np.radians(beam[2]))
    fr = np.fft.fftfreq(image.shape[0], d=pixel_mas)[:, None]  # cycles/mas along rows (rows run towards South)
    fc = np.fft.fftfreq(image.shape[1], d=pixel_mas)[None, :]  # along columns (towards West): the quadratic form below is even in the common sign
    sig = np.array([bmaj, bmin]) / 2.3548
    f_maj = fc * (-np.sin(pa)) + fr * (-np.cos(pa))  # major axis direction (East, North) = (sin PA, cos PA), in (col, row) the opposite signs
    f_min = fc * (-np.cos(pa)) + fr * np.sin(pa)
    kernel = np.exp(-2 * np.pi**2 * ((sig[0] * f_maj) ** 2 + (sig[1] * f_min) ** 2))
    return np.fft.ifft2(np.fft.fft2(image) * kernel).real


def extent(shape, pixel_mas):
    """``imshow`` extent in arcsec (East left, North up) of an image with row 0 North whose phase-centre cell is ``n // 2 - 1``."""
    (n, m), dx = shape, pixel_mas / 1000.0
    r0, c0 = n // 2 - 1, m // 2 - 1
    return ((c0 + 0.5) * dx, -(m - 1 - c0 + 0.5) * dx, -(n - 1 - r0 + 0.5) * dx, (r0 + 0.5) * dx)


def moments(image, pixel_mas, radius_arcsec=1.2):
    """``(east_mas, north_mas, pa_deg)``: flux-weighted centroid offset from the phase centre and the major-axis PA (East of North, mod 180) of the emission within ``radius_arcsec``."""
    n, m = image.shape
    rr, cc = np.indices(image.shape)
    north, east = -(rr - (n // 2 - 1)) * pixel_mas, -(cc - (m // 2 - 1)) * pixel_mas
    keep = np.hypot(east, north) <= radius_arcsec * 1e3
    w = np.where(keep, np.maximum(image, 0.0), 0.0)
    e0, n0 = (w * east).sum() / w.sum(), (w * north).sum() / w.sum()
    cov = np.cov(np.stack([east[keep], north[keep]]), aweights=w[keep], ddof=0)
    vals, vecs = np.linalg.eigh(cov)
    major = vecs[:, 1]
    return float(e0), float(n0), float(np.degrees(np.arctan2(major[0], major[1])) % 180.0)


def ellipse(ax, beam, fraction, xy, **kw):
    """The beam (FWHM ellipse, PA East of North) at ``xy`` of an axis whose data coordinates are (East, North) arcsec."""
    from matplotlib.patches import Ellipse

    ax.add_patch(Ellipse(xy, fraction * beam[0] / 1000.0, fraction * beam[1] / 1000.0, angle=90.0 - beam[2], **kw))


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _sky(ax, img, px, title, cmap, vmin=None, vmax=None, norm=None):
    kw = dict(norm=norm) if norm is not None else dict(vmin=vmin, vmax=vmax)
    im = ax.imshow(img, extent=extent(img.shape, px), cmap=cmap, interpolation="nearest", origin="upper", **kw)  # row 0 North
    half = min(HALF_ARCSEC, abs(extent(img.shape, px)[1]))
    ax.set_xlim(half, -half)
    ax.set_ylim(-half, half)
    ax.set_title(title)
    ax.set_xlabel("ΔRA (arcsec, East left)")
    ax.plot(0, 0, "+", color="cyan", ms=8, mew=1)  # the phase centre
    return im


def _blank(ax, text):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color="0.35", fontsize=8, wrap=True)


def fig_half_beam(path, img, px, beam):
    """The image raw and blurred to 0.5 of the CLEAN beam, with that beam."""
    plt = _plt()
    from matplotlib.colors import PowerNorm

    half = blur(img, px, beam, 0.5)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    for ax, a, t in zip(axs, (img, half), ("raw image", "blurred to 0.5 of the CLEAN beam")):
        im = _sky(ax, a, px, t, "inferno", norm=PowerNorm(0.5, vmin=0, vmax=float(a.max())))
        fig.colorbar(im, ax=ax, shrink=0.8, label="Jy/arcsec²")
    ellipse(axs[1], beam, 0.5, (HALF_ARCSEC * 0.8, -HALF_ARCSEC * 0.8), fill=False, ec="cyan", lw=1)
    axs[0].set_ylabel("ΔDec (arcsec, North up)")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_at_beam(path, img, px, beam, mpol, clean, beam_note):
    """Our image, MPoL's and CLEAN at the CLEAN beam, with the residual-difference maps on a symmetric diverging scale (``mpol``, ``clean``: Jy/arcsec^2 on our grid, or ``None``)."""
    plt = _plt()
    ours = blur(img, px, beam)
    mp = None if mpol is None else blur(mpol, px, beam)
    top = [("ours (blurred to the CLEAN beam)", ours), ("MPoL (blurred)", mp), ("CLEAN", clean)]
    diffs = [("ours − MPoL", None if mp is None else ours - mp), ("ours − CLEAN", None if clean is None else ours - clean), ("MPoL − CLEAN", None if mp is None or clean is None else mp - clean)]
    peak = float(max(a.max() for _, a in top if a is not None))
    lim = max([float(np.abs(d).max()) for _, d in diffs if d is not None] + [1e-30])
    fig, axs = plt.subplots(2, 3, figsize=(11, 7.4), constrained_layout=True)
    for ax, (t, a) in zip(axs[0], top):
        if a is None:
            _blank(ax, f"{t}: not found")
            continue
        im = _sky(ax, a, px, t, "inferno", 0.0, peak)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Jy/arcsec²")
        ellipse(ax, beam, 1.0, (HALF_ARCSEC * 0.75, -HALF_ARCSEC * 0.75), fill=False, ec="cyan", lw=1)
    for ax, (t, d) in zip(axs[1], diffs):
        if d is None:
            _blank(ax, f"{t}: not available")
            continue
        im = _sky(ax, d, px, t, "RdBu_r", -lim, lim)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Jy/arcsec² (symmetric scale)")
    axs[0, 0].set_ylabel("ΔDec (arcsec, North up)")
    fig.suptitle(f"CLEAN beam: {beam[0]:.1f} × {beam[1]:.1f} mas, PA {beam[2]:.0f}° ({beam_note})", fontsize=9)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_sweep(path, sweep):
    """m(w) and G(w) against w/s^2, the chosen root, every crossing and the s^2 +/- 2 sigma_s band marked."""
    plt = _plt()
    pts = sweep["points"]
    x = np.array([p["w_over_s2"] for p in pts])
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    for ax, key, lab in zip(axs, ("m", "g"), ("m(w) = χ²_q/s² − (n_real − G)", "G(w)")):
        ax.plot(x, [p[key] for p in pts], "o-", ms=3, color="#2a78d6")
        ax.set_xscale("log")
        ax.set_xlabel("w / s²")
        ax.set_ylabel(lab)
        for c in sweep.get("crossings_w_over_s2", []):
            ax.axvline(c, color="0.6", ls=":", lw=1)
        if sweep.get("w_over_s2"):
            ax.axvline(sweep["w_over_s2"], color="#eb6834", lw=1.2, label="chosen")
        band = [b for b in (sweep.get("weight_band") or []) if b]
        if len(band) == 2:
            ax.axvspan(min(band) / sweep["s2"], max(band) / sweep["s2"], color="#eb6834", alpha=0.15, label="s² ± 2σ_s")
    axs[0].axhline(0.0, color="k", lw=0.6)
    axs[0].set_yscale("symlog", linthresh=max(1.0, float(sweep["n_real"]) * 1e-4))
    axs[0].legend(fontsize=7)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_psi(path, pixels, px, stats):
    """The psi_hp map: high-passed ln b on the disc (the dot metric), with the 1e-2 and 1e-1 brightness contours."""
    plt = _plt()
    psi, rel = vl.psi_hp_map(pixels, px)
    lim = float(np.abs(psi).max()) or 1.0
    fig, ax = plt.subplots(figsize=(5, 4.4), constrained_layout=True)
    im = _sky(ax, psi, px, f"ψ_hp: disc {stats['disc']:.3g}, lo {stats['lo']:.3g}, hi {stats['hi']:.3g}", "RdBu_r", -lim, lim)
    ex = extent(psi.shape, px)
    ax.contour(np.linspace(ex[0], ex[1], psi.shape[1]), np.linspace(ex[3], ex[2], psi.shape[0]), rel, levels=[1e-2, 1e-1], colors="k", linewidths=0.5)
    fig.colorbar(im, ax=ax, shrink=0.8, label="high-passed ln b")
    ax.set_ylabel("ΔDec (arcsec, North up)")
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- tables and checks


def g(x, spec=".4g"):
    return "—" if x is None else (f"{x:{spec}}" if np.isfinite(x) else "nan")


def table(head, rows, align=None):
    align = align or ["---"] + ["---:"] * (len(head) - 1)
    return "\n".join(["| " + " | ".join(head) + " |", "| " + " | ".join(align) + " |"] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def verdict(ok):
    return "not evaluated" if ok is None else ("PASS" if ok else "**FAIL**")


def check(item, what, threshold, value, ok):
    return dict(item=item, what=what, threshold=threshold, value=value, ok=None if ok is None else bool(ok))


def _has(rec, *keys):
    return rec is not None and all(rec.get(k) is not None for k in keys)


def checks_of(rec):
    """The pre-registered checks (items 1, 6, 7 and D1-D6) evaluated on the records found; a check whose numbers are missing is ``not evaluated``."""
    sg, d4, sw, fi = (rec[n] for n in ("sigma", "d4", "sweep", "final"))
    out = []
    out.append(check("1", "σ_s / s² (noise scale known well enough)", f"≤ {vl.SIGMA_REL_MAX:g}", g(sg["sigma_rel"]) if _has(sg, "sigma_rel") else "—", sg["sigma_rel"] <= vl.SIGMA_REL_MAX if _has(sg, "sigma_rel") else None))
    out.append(check("D3", "reference observation = largest Σw, from the weights alone", "argmax Σw",
                     f"obs {sg['ref_obs']} (Σw argmax {int(np.argmax(sg['sum_weights']))})" if _has(sg, "ref_obs", "sum_weights") else "—",
                     sg["ref_obs"] == int(np.argmax(sg["sum_weights"])) if _has(sg, "ref_obs", "sum_weights") else None))
    if _has(d4, "bias", "se_mean", "s2_injected"):
        ok4 = abs(d4["bias"]) <= 2 * d4["se_mean"] and abs(d4["bias"]) <= vl.D4_REL_MAX * d4["s2_injected"]
        val4 = f"bias {g(d4['bias'])} = {g(d4['bias_in_se'], '.2f')} se; {g(abs(d4['bias_rel']))} of s²"
    else:
        ok4, val4 = None, "—"
    out.append(check("D4", f"scatter estimator unbiased on CLEAN + noise of known s² ({vl.D4_REALISATIONS} realisations)", f"|mean − s²| ≤ 2 sd/√{vl.D4_REALISATIONS} and ≤ {vl.D4_REL_MAX:g} s²", val4, ok4))
    pts = sw["points"] if _has(sw, "points") else None
    out.append(check("D1", "m > 0 at the first grid point", f"w/s² = {vl.GRID_MAX:g}", g(pts[0]["m"]) if pts else "—", pts[0]["m"] > 0 if pts else None))
    out.append(check("D1", "m(w) changes sign on the fixed grid", f"w/s² {vl.GRID_MAX:g} to {vl.GRID_MIN:g}, steps ×√10", f"{len(sw['crossings'])} crossing(s)" if _has(sw, "crossings") else "—", len(sw["crossings"]) > 0 if _has(sw, "crossings") else None))
    out.append(check("D2", "|m| at the root within 3 se(G)", f"≤ 3 se(G); ≤ {vl.MAX_SECANT} secant steps",
                     f"|m| = {g(abs(sw['m']))}, 3 se = {g(3 * sw['g_se'])}, {sw['n_secant']} step(s)" if _has(sw, "m", "g_se", "n_secant") else "—",
                     (abs(sw["m"]) <= 3 * sw["g_se"] and sw["n_secant"] <= vl.MAX_SECANT) if _has(sw, "m", "g_se", "n_secant") else None))
    if _has(sw, "weight", "crossings") and sw["crossings"]:
        step = vl.GRID_STEP
        ok5 = max(sw["crossings"]) / step <= sw["weight"] <= max(sw["crossings"]) * step
        val5 = f"chosen w/s² = {g(sw['w_over_s2'])}; largest crossing {g(max(sw['crossings_w_over_s2']))}"
    else:
        ok5, val5 = None, "—"
    out.append(check("D5", "the crossing at the largest w is taken", "largest w (within one grid step)", val5, ok5))
    out.append(check("D6", "fast loop on the full data (stride 1)", "n_rows = n_vis", f"{sw['n_rows']:,} of {sg['n_vis']:,}" if _has(sw, "n_rows") and _has(sg, "n_vis") else "—", sw["n_rows"] == sg["n_vis"] if _has(sw, "n_rows") and _has(sg, "n_vis") else None))
    out.append(check("6", "fast-loop fits converged", f"kkt_w < {vl.KKT_W_MAX:g} at every fit", f"{len(sw['unconverged_fits'])} unconverged" if _has(sw, "unconverged_fits") else "—", not sw["unconverged_fits"] if _has(sw, "unconverged_fits") else None))
    out.append(check("6", "final fit: kkt_w", f"< {vl.KKT_W_MAX:g}", g(fi["kkt_w"]) if _has(fi, "kkt_w") else "—", fi["kkt_w"] < vl.KKT_W_MAX if _has(fi, "kkt_w") else None))
    out.append(check("6", "final fit: band limit f_eta/f_floor", f"< {vl.BAND_MAX:g}", g(fi["band_limit"]) if _has(fi, "band_limit") else "—", fi["band_limit"] < vl.BAND_MAX if _has(fi, "band_limit") else None))
    if _has(fi, "nuisances"):
        worst = min(float(np.min(v["edge_frac"])) for v in fi["nuisances"].values())
        ok7 = not any(np.any(v["at_edge"]) for v in fi["nuisances"].values()) and worst > vl.EDGE
        val7 = f"smallest distance from an edge {g(worst, '.3f')} of the box"
    else:
        ok7, val7 = None, "—"
    out.append(check("7", "no nuisance within 5% of its prior box edge", f"> {vl.EDGE:.0%} of the box", val7, ok7))
    return out


def checks_table(checks):
    return table(["item", "check", "threshold", "value", "verdict"], [(c["item"], c["what"], c["threshold"], c["value"], verdict(c["ok"])) for c in checks], ["---", "---", "---", "---", "---"])


def sigma_section(r):
    rows = [("σ_s", g(r["sigma_s"])), ("σ_s / s²", f"{g(r['sigma_rel'])} (≤ {vl.SIGMA_REL_MAX:g})"), ("pairs", f"{r['n_pairs']:,}"),
            ("lag-1 correlation of successive differences", f"{g(r['lag1_correlation'], '.3f')} (white noise: −0.5; {r['n_lag1']:,} products)"), ("reference observation (D3)", r["ref_obs"]),
            ("visibilities / real data", f"{r['n_vis']:,} / {r['n_real']:,}"), ("F0 (Jy)", g(r["flux_jy"]))]
    return table(["", ""], rows, ["---", "---"])


def d4_section(r):
    rows = [("injected s²", g(r["s2_injected"], ".6g")), ("mean of the estimates", g(r["mean"], ".6g")), ("sd", g(r["sd"])), ("bias", f"{g(r['bias'])} ({g(r['bias_rel'])} of s², {g(r['bias_in_se'], '.2f')} se)"),
            ("lag-1 correlation, real data", g(r["lag1_correlation_real"], ".3f")), ("verdict", "pass" if r.get("pass") else "**fail**")]
    return table(["", ""], rows, ["---", "---"])


def grid_section(sw):
    pts = sw["points"]
    rows = []
    for i, p in enumerate(pts):
        cross = i + 1 < len(pts) and (p["m"] > 0) != (pts[i + 1]["m"] > 0)
        rows.append((g(p["w_over_s2"]), g(p["chi2"], ".6g"), g(p["m"]), g(p["g"]), g(p["kkt_w"], ".2g"), g(p["band_limit"]), "crossing below" if cross else ""))
    return table(["w/s²", "χ²", "m", "G", "kkt_w", "band limit", ""], rows)


def sweep_section(sw):
    out = [grid_section(sw)]
    if "crossings" in sw:
        out.append("Crossings (w/s²; the largest is taken, D5): " + (", ".join(g(c) for c in sw["crossings_w_over_s2"]) or "none") + ".")
    if sw.get("weight") is not None:
        s2 = sw["s2"]
        band = sw.get("weight_band") or [None, None]
        out.append(table(["", ""], [
            ("chosen w / w/s²", f"{g(sw['weight'], '.6g')} / {g(sw['w_over_s2'], '.6g')}"),
            ("m, G ± se, secant steps", f"{g(sw['m'])}, {g(sw['g'])} ± {g(sw['g_se'])}, {sw['n_secant']}"),
            ("w at s² + 2σ_s, s² − 2σ_s", ", ".join("off the grid" if b is None else f"{g(b, '.6g')} (w/s² {g(b / s2)})" for b in band)),
            ("h(w) = 2 (w/s²) KL − G (diagnostic only)", g(sw.get("h"))),
        ], ["---", "---"]))
    return "\n\n".join(out)


def nuisance_table(fi):
    rows = []
    for name, v in fi["nuisances"].items():
        val, lo, hi, ef, edge = (np.asarray(v[k]) for k in ("value", "lower", "upper", "edge_frac", "at_edge"))
        for idx in np.ndindex(val.shape):
            tag = "".join(f"[{i}]" for i in idx) if idx else ""
            ref = " (reference)" if name.endswith("gain") and idx and idx[0] == fi["ref_obs"] else ""
            rows.append((f"{name}{tag}{ref}", g(val[idx]), g(lo[idx]), g(hi[idx]), g(ef[idx], ".3f"), "**AT EDGE**" if edge[idx] else "ok"))
    return table(["nuisance", "value", "box lower", "box upper", "distance from nearest edge / box", ""], rows)


def final_section(fi, values, px):
    ps = fi["psi_hp"]
    out = [table(["", ""], [
        ("chosen w / w/s²", f"{g(fi['weight'], '.6g')} / {g(fi['w_over_s2'], '.6g')}"),
        ("G (the sweep's, on the same data)", f"{g(fi['g'])} ± {g(fi['g_se'])}"),
        ("kkt_w (< 1e-4) and f_eta/f_floor (< 2)", f"{g(fi['kkt_w'], '.3g')}; {g(fi['band_limit'], '.3g')} (f_eta/f_floor = {g(fi['f_eta_over_f_floor'], '.3g')})"),
        ("ψ_hp: disc, 1e-2 to 1e-1 of the peak, above 1e-1", f"{g(ps['disc'], '.3g')}, {g(ps['lo'], '.3g')}, {g(ps['hi'], '.3g')}"),
        ("flux (Jy) against the sigma stage's F0", f"{g(fi['flux_jy'], '.5g')} against {g(fi['flux_reference_jy'], '.5g')}"),
        ("σ factors", fi["sigma_factors"]),
    ], ["---", "---"]), "**Nuisances against their box edges** (the fit fails inside 5%):", nuisance_table(fi)]
    if values is not None and "pixels" in values:
        again = vl.psi_hp(values["pixels"], px)
        off = max(abs(again[k] - ps[k]) for k in ps)
        if off > 1e-6 * max(ps.values()):
            out.append(f"**Inconsistent:** ψ_hp recomputed from `{VALUES}` differs from the record by {off:.2g}.")
    return "\n\n".join(out)


# ---------------------------------------------------------------- the report


def build(run, mpol=None, clean=None, figures=None, fig_prefix="figures"):
    """The report of run directory ``run`` as Markdown; its figures go to ``figures`` (default ``run/figures``) and are linked under ``fig_prefix``. ``mpol``, ``clean``: MPoL export and CLEAN output directories."""
    run = pathlib.Path(run)
    rec, values = read_run(run)
    figdir = pathlib.Path(figures) if figures else run / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    first = next((r for r in rec.values() if r), {})
    md = [f"# IM Lup: the registered weight rule, run `{run.name}`", "",
          "Pre-registered rule `lambda_selection` (`design/alma_imlup_gate_criteria.yml`), primary arm maximum entropy on √I pixels. This report reads the stage records and refits nothing; "
          "every threshold is the constant of `virgil_bridge.alma_lambda` that implements the registered text.", "",
          f"Registered criteria: {first.get('registered', 'not recorded')}; criteria file hash: `{str(first.get('criteria_sha256', first.get('criteria_hash', 'not recorded')))[:16]}`.", ""]
    checks = checks_of(rec)
    stopped = None
    sections = []
    for name, fname in STAGES:
        r = rec[name]
        if stopped:
            sections.append(f"## {name}\n\nSkipped: stage `{stopped}` did not complete.")
            continue
        if r is None:
            sections.append(f"## {name}\n\nNot run (no `{fname}`).")
            stopped = name
            continue
        head = f"## {name}" + ("" if r.get("status") == "ok" else " — FAILED")
        body = []
        if r.get("status") != "ok":
            body.append(f"**Reason:** {r.get('reason', 'not recorded')}")
            stopped = name
        try:
            if name == "sigma":
                body.append("**Raw χ²/N with the quoted weights** (successive-difference pairs, no image): " + g(r["raw_chi2_over_N_pairs"], ".6g") + f"; **s² = {g(r['s2'], '.6g')}**.\n\n" + sigma_section(r))
            elif name == "d4":
                body.append(d4_section(r))
            elif name == "sweep":
                body.append(sweep_section(r))
            else:
                body.append(f"**Raw χ²/N with the quoted weights:** {g(r['raw_chi2_over_N'], '.6g')} (nuisances fitted; {g(r['raw_chi2_over_N_nominal'], '.6g')} with them at nominal); s² = {g(r['s2'], '.6g')}, σ_s/s² = {g(r['sigma_s'] / r['s2'])}.")
                body.append(final_section(r, values, r["pixel_mas"]))
        except KeyError as e:  # a failed stage's record may stop short of its table
            body.append(f"_(record incomplete: no {e})_")
        sections.append(head + "\n\n" + "\n\n".join(body))
    bad = [f"{c['item']} ({c['what']})" for c in checks if c["ok"] is False]
    summary = "FAIL: " + "; ".join(bad) if bad else ("Not complete: some checks are not evaluated" if any(c["ok"] is None for c in checks) else "All checks pass")
    md += ["## Pre-registered checks", "", f"**{summary}.** Items 2 to 5 and 8 are procedures or reported numbers (below); item 9 is the caveat at the end.", "", checks_table(checks), ""]
    md += ["\n\n".join(sections), ""]
    if stopped is None and values is not None:
        md += images_section(rec["final"], values, rec["sweep"], figdir, fig_prefix, mpol, clean)
    else:
        md += ["## Figures", "", "Skipped: " + ("a stage did not complete." if stopped else f"`{VALUES}` is missing."), ""]
    md += ["## Contrast caveat", "", CONTRAST_CAVEAT, ""]
    return "\n".join(md)


def images_section(fi, values, sw, figdir, prefix, mpol_dir, clean_dir):
    px, npix = float(fi["pixel_mas"]), int(fi["npix"])
    img = np.asarray(values["pixels"], float) ** 2 / (px / 1000.0) ** 2  # Jy/pixel to Jy/arcsec^2, row 0 North
    mp_img, mp_px, mp_note = locate_mpol(mpol_dir)
    clean, cl_note = locate_clean(clean_dir)
    beam, beam_note = (clean["beam"], "our CLEAN") if clean else (DSHARP_BEAM, "DSHARP's nominal beam: no CLEAN image found")
    mp = None if mp_img is None else mpol_on_grid(mp_img, mp_px, npix, px)
    cl = None if clean is None else clean_on_grid(clean, npix, px)
    fig_half_beam(figdir / "image_half_beam.png", img, px, beam)
    fig_at_beam(figdir / "image_at_beam.png", img, px, beam, mp, cl, beam_note)
    fig_sweep(figdir / "sweep.png", sw)
    fig_psi(figdir / "psi_hp.png", values["pixels"], px, fi["psi_hp"])
    rows = []
    for name, a in (("ours", img), ("MPoL", mp), ("CLEAN", cl)):
        if a is None:
            rows.append((name, "not found", "", ""))
        else:
            e, n, pa = moments(a, px)
            rows.append((name, f"{e:+.0f}", f"{n:+.0f}", f"{pa:.1f}"))
    return ["## Figures", "",
            f"Row 0 is North and East is left (`imshow`, origin upper); the cyan cross is the phase centre. MPoL: {mp_note}. CLEAN: {cl_note}. Beam: {beam[0]:.1f} × {beam[1]:.1f} mas, PA {beam[2]:.0f}° ({beam_note}).", "",
            f"![the image raw and blurred to 0.5 of the CLEAN beam]({prefix}/image_half_beam.png)", "",
            f"![at the CLEAN beam, with MPoL and CLEAN and their differences]({prefix}/image_at_beam.png)", "",
            f"![m(w) and G(w)]({prefix}/sweep.png)", "",
            f"![psi_hp map]({prefix}/psi_hp.png)", "",
            f"**Orientation check** (not a registered threshold): the disc's major-axis PA should be near {DISC_PA_DEG}° (Huang et al. 2018), and its centroid near the phase centre; a mirrored image would not be.", "",
            table(["image", "ΔEast (mas)", "ΔNorth (mas)", "major-axis PA (°)"], rows), ""]


def splice(page, body):
    """``page`` text with the block between the ``gen:lambda-report`` markers replaced by ``body`` (its first heading dropped)."""
    if BEGIN not in page or END not in page:
        raise ValueError(f"the page has no {BEGIN} ... {END} block")
    body = re.sub(r"\A# .*\n+", "", body)
    head, rest = page.split(BEGIN, 1)
    return head + BEGIN + "\n\n" + body.strip() + "\n\n" + END + rest.split(END, 1)[1]


def write(run, *, mpol=None, clean=None, page=None, assets=None):
    """Write ``run/report.md`` and its figures; with ``page`` (a docs page with the markers) and ``assets`` (a folder next to it) also splice the report into the page. Returns the report text."""
    run = pathlib.Path(run)
    md = build(run, mpol, clean)
    (run / "report.md").write_text(md + "\n")
    if page:
        import shutil

        page = pathlib.Path(page)
        assets = pathlib.Path(assets) if assets else page.with_suffix("")
        assets.mkdir(parents=True, exist_ok=True)
        for f in (run / "figures").glob("*.png"):
            shutil.copy(f, assets / f.name)
        page.write_text(splice(page.read_text(), md.replace("](figures/", f"]({assets.name}/")))
    return md
