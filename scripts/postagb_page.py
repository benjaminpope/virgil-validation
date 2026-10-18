"""Regenerate every number on docs/imaging/postagb.md from the saved result JSON, so that none can drift.

    python scripts/postagb_page.py [--check]

The page carries marker pairs

    <!-- gen:NAME KEY -->  ...  <!-- /gen:NAME -->

and everything between a pair is rewritten here from ``docs/imaging/postagb/<KEY>_*.json`` (the headline ``image``/``compare`` files, the
per-strategy ``<KEY>_<strategy>_image.json`` files, ``<KEY>_strategies.json`` and ``<KEY>_parametric.json``) and from the paper's numbers in
``scripts/postagb_paper.py``. Regions: ``summary``, ``methods``, ``parity`` (page level) and, per tab, ``figure``, ``verdict``,
``strategies``, ``parameters``, ``metrics``, ``model`` (IRAS 08544-4431: ``iras_binary``, ``iras_model``). Text outside the markers is
hand-written and carries no result. Missing keys print as "-"; the run is idempotent: running it twice changes nothing.
``--check`` exits 1 if the page is out of date instead of writing it.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from postagb_paper import IMAGE, PMOIRED  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/imaging"
PAGE = DOCS / "postagb.md"
OUT = DOCS / "postagb"
TABS = {  # tab title -> target key
    "IW Car": "IW_Car", "HR 4049": "HR4049", "HD 108015": "HD108015", "AI Sco": "AI_Sco", "EN TrA": "EN_TrA",
    "HD 95767": "HD95767", "IRAS 15469-5311": "IRAS15469-5311", "PS Gem": "PS_Gem", "IRAS 08544-4431": "IRAS08544",
}
TITLE = {v: k for k, v in TABS.items()}
DE_PRINS = [k for k in TABS.values() if k != "IRAS08544"]
STRATS = ("mem", "gp-iso", "gp-aniso", "gp-polar", "disc+res")
OTHERS = ("mem", "gp-iso", "gp-aniso")  # the strategies that can stand next to the headline (disc+res is a diagnostic)
HEADLINE = "gp-polar"
SNAME = {"mem": "MEM (MaxEntropy)", "gp-iso": "gp-iso", "gp-aniso": "gp-aniso", "gp-polar": "gp-polar", "disc+res": "disc+res"}
MARGIN = 0.1  # a verdict may only say "agree" if the NCC exceeds its best control by this much
FLOOR = 0.03  # below this margin the images are not distinguishable from the controls
MIN_NCC = 0.5  # and an image that correlates worse than this with the published one does not agree whatever its controls do
OBS = {  # observation log lines of the captions (data facts, not results)
    "IW_Car": "23 PIONIER exposures (MJD 58812 to 58893, up to 2020-02-14)", "HR4049": "19 PIONIER exposures (MJD 58884 to 58899)",
    "HD108015": "35 PIONIER exposures (MJD 59576 to 59621)", "AI_Sco": "28 PIONIER exposures (MJD 58693 to 58726)",
    "EN_TrA": "20 PIONIER exposures (MJD 59267 to 59295)", "HD95767": "36 PIONIER exposures (MJD 59577 to 59616)",
    "IRAS15469-5311": "20 PIONIER exposures (MJD 59266 to 59296)", "PS_Gem": "12 PIONIER exposures (MJD 58817 to 58928)",
}
# De Prins et al. (2026) numbers for IW Car, which has no PMOIRED table (no single-disc fit works): SPARCO values from the ORGANIC FITS
# header and the inner-arc and outer-arc ellipses of Table 2, as already quoted on the page.
IW_PAPER = dict(f_prim=63.6, f_sec=1.99, sec=(1.12, -1.90), ud=0.50, index=0.89, rim_inner=(6.9, 0.5), inc_inner=(48, 5), inc_outer=(57, 2),
                pa_inner=(144, 7, 8), pa_outer=(176, 3, 3))
IW_DATA = (828, 552)  # V2 rows and closure-phase rows (the Data paragraph)
IW_SQUEEZE_CHECK = "SPARCO + ORGANIC (SQUEEZE check)"
PUBLISHED = {k: "SPARCO + ORGANIC" for k in TABS.values()}
PUBLISHED["IW_Car"] = IW_SQUEEZE_CHECK
PUBLISHED["IRAS08544"] = "SPARCO, binary subtracted (Hillen+2016)"
HILLEN = dict(f_prim=(59.7, 0.6), f_sec=(3.9, 0.7), sep=(0.81, 0.05), pa=(236.0, 3.0), env_index=(0.42, None))  # Table 1; the PA is 56 + 180
IRAS_STAR_READING = "about 0.86 mas and 213° (read by hand off the star markers on the published figure; reading uncertainty about 0.1 mas and 8°; not a measurement)"
MINUS = "\N{MINUS SIGN}"


# ---------------------------------------------------------------------------------------------------------------------------------
# reading the results


def J(key, what, strat=None):
    p = OUT / f"{key}_{strat + '_' if strat else ''}{what}.json"
    return json.load(open(p)) if p.exists() else None


def strategies(key):
    s = J(key, "strategies")
    return s["strategies"] if s else {}


def best_control(c):
    return max(c["ncc_controls"].values())


def n_(x, spec=".2f", dash="-"):
    """Format a number, '-' for None or NaN."""
    return dash if x is None or x != x else format(x, spec)


def sgn(x, spec=".2f"):
    return "-" if x is None or x != x else format(x, "+" + spec).replace("-", MINUS)


def pct(x, spec=".0f"):
    return "-" if x is None else f"{100 * x:{spec}} %"


def headline_label(i):
    return i.get("strategy") or i["regulariser"]


def reg_text(i):
    r = i["regulariser"]
    if r == "MultipoleField":
        return (f"the polar Gaussian-process prior (σ = {i['gp_sigma']:g}, ℓ_r = {i['gp_length_r_mas']:.2f} mas, ℓ_θ = {i['gp_length_theta']:g} rad), "
                "our recommended model")
    if r == "GaussianField":
        return f"a Gaussian-process prior (σ = {i['gp_sigma']:g}, ℓ = {i['gp_length_mas']:.2f} mas)"
    if r == "MultipoleDisk":
        return "a disc-plus-residual prior"
    if i.get("weight_rule") == "discrepancy":
        return "MaxEntropy with the weight at the discrepancy principle"
    return "MaxEntropy with the weight at the L-curve corner"


def ctl(c):
    k = c["ncc_controls"]
    return (f"E-W mirror {k['mirrored_East-West']:.2f}, 180° rotation {k['rotated_180']:.2f}, "
            f"azimuthal average {k['azimuthal_average_of_published']:.2f}")


def indent(text, ind):
    return "\n".join((ind + ln) if ln.strip() else "" for ln in text.rstrip("\n").split("\n")) + "\n"


def wrap(text, width=118):
    lines, cur = [], ""
    for w in text.split(" "):
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + [cur]


def admonition(kind, title, body, opener="!!!"):
    return f'{opener} {kind} "{title}"\n' + "\n".join("    " + ln for ln in wrap(body)) + "\n"


# ---------------------------------------------------------------------------------------------------------------------------------
# verdicts, gated on the NCC controls


def rating(c):
    """('agree' | 'inconclusive' | 'does not agree', margin): the NCC must beat its best control by MARGIN to count as agreement."""
    m = c["ncc_zero_shift"] - best_control(c)
    if c["ncc_zero_shift"] < MIN_NCC:
        return "does not agree", m
    return ("agree" if m >= MARGIN else "inconclusive" if m >= FLOOR else "does not agree"), m


def morph(key, c):
    """The morphology sentence: flux-weighted radius, centroid and major axis, ours against the published image."""
    o, p = c["ours"], c["published"]
    if "disc" in o:
        d, q = o["disc"], p["disc"]
        return (f"Flux-weighted radius {d['mean_radius_mas']:.1f} against {q['mean_radius_mas']:.1f} mas, flux centroid {d['centroid_pa_deg']:.0f}° "
                f"against {q['centroid_pa_deg']:.0f}° East of North, major-axis PA {d['major_pa_deg']:.0f}° against {q['major_pa_deg']:.0f}° "
                f"(axis ratios {d['axis_ratio']:.2f} and {q['axis_ratio']:.2f}).")
    return (f"The inner arcs: flux-weighted radius {o['inner']['mean_radius_mas']:.1f} against {q_(p, 'inner')} mas, major-axis PA "
            f"{o['inner']['major_pa_deg']:.0f}° against {p['inner']['major_pa_deg']:.0f}°. The outer arc: radius {o['outer']['mean_radius_mas']:.1f} against "
            f"{p['outer']['mean_radius_mas']:.1f} mas, PA {o['outer']['major_pa_deg']:.0f}° against {p['outer']['major_pa_deg']:.0f}°.")


def q_(p, reg):
    return f"{p[reg]['mean_radius_mas']:.1f}"


def verdict(key, c, i):
    n, b = c["ncc_zero_shift"], c["ncc_best_shift"]
    k = c["ncc_controls"]
    label, m = rating(c)
    beam = (f"the beam drawn on the published figure ({c['beam']['major_mas']:.1f} × {c['beam']['minor_mas']:.1f} mas)" if key == "IRAS08544"
            else f"the λ_min/2B_max beam ({c['beam']['major_mas']:.2f} mas)")
    base = (f"At {beam} the headline image ({headline_label(i)}) correlates with the published one at {n:.2f} ({b:.2f} after the best sub-beam shift), "
            f"against {k['mirrored_East-West']:.2f} for the East-West mirror of the published image, {k['rotated_180']:.2f} for its 180° rotation and "
            f"{k['azimuthal_average_of_published']:.2f} for its azimuthal average (the same flux at the same radii with no azimuthal structure): "
            f"{sgn(m)} over the best control. A verdict says 'agree' only above +{MARGIN:g}.")
    st = strategies(key)
    other = [(s, st[s]["ncc"]) for s in OTHERS if s in st and st[s].get("ncc") is not None]
    cmp_ = ""
    if other:
        bs, bn = max(other, key=lambda t: t[1])
        mem = st.get("mem", {}).get("ncc")
        cmp_ = (f" Of the other strategies, {SNAME[bs]} is closest to the published image at {bn:.2f}" + (f" (MEM {mem:.2f})" if mem is not None and bs != "mem" else "") + ".")
    if key == "IRAS08544":
        o, q = c["ours"]["outer"], c["published"]["outer"]
        detail = (f" With the binary subtracted in both, the flux inside 3.5 mas is {100 * c['ours']['inner']['flux_fraction']:.1f} % against "
                  f"{100 * c['published']['inner']['flux_fraction']:.1f} %, and the ring has flux-weighted radius {o['mean_radius_mas']:.2f} against "
                  f"{q['mean_radius_mas']:.2f} mas and flux centroid {o['centroid_pa_deg']:.0f}° against {q['centroid_pa_deg']:.0f}°. "
                  "The binary table below is not a blind recovery: the fit starts at Hillen's values and the priors sit on them.")
    else:
        detail = " " + morph(key, c)
    extra = " SQUEEZE also failed to reconstruct this target in the paper (Appendix D)." if key in ("HR4049", "PS_Gem") else ""
    if label == "agree":
        kind, title = "success", f"Verdict: the reconstructions agree (NCC {sgn(m)} over its best control)"
    elif label == "inconclusive":
        kind, title = "warning", f"Verdict: inconclusive (NCC only {sgn(m)} over its best control)"
    else:
        kind, title = "failure", f"Verdict: does not agree (NCC {sgn(m)} against its best control)"
    return kind, title, base + detail + cmp_ + extra + " Conventions: see the [Methods note](#methods-note)."


def short_verdict(key, c):
    label, m = rating(c)
    return {"agree": "agrees", "inconclusive": "inconclusive", "does not agree": "does not agree"}[label] + f" ({sgn(m)} over its best control)"


# ---------------------------------------------------------------------------------------------------------------------------------
# regions: figure, verdict, strategies


def gen_figure(key):
    i, c = J(key, "image"), J(key, "compare")
    if not i or not c:
        return ""
    reg = reg_text(i)
    t = TITLE[key]
    if key == "IRAS08544":
        b = c["beam"]
        return (f"<figure markdown>\n![{t}: Hillen et al. image (left) and virgil image (right)](postagb/{key}_images.png)\n"
                f"<figcaption>{t} in H band, 27 PIONIER observing blocks (2015-01-21, 2015-01-24 and 2015-02-23). Left: the binary-subtracted SPARCO image of "
                "Hillen et al. (2016) as published (their Fig. 4, lower panel; the blue contour is their 5σ level). Right: virgil, "
                f"{reg}, primary and secondary subtracted, drawn in the published figure's own colour map, on the same field of view, orientation (North up, "
                f"East left), linear 0 to 1 peak-normalised stretch, and convolved with the beam ellipse drawn on the published figure ({b['major_mas']:.1f} × "
                f"{b['minor_mas']:.1f} mas, PA {b['pa_deg']:.0f}°). The paper's image exists only as a bitmap, so its stretch and its beam cannot be matched "
                "exactly; for this target only, virgil is convolved with the beam the paper draws rather than with the λ_min/2B_max beam of the other tabs, and "
                "the published image, already a restored image, is not convolved again. The primary is the marker at the origin.</figcaption>\n</figure>\n")
    sub = "star-and-secondary-subtracted" if i.get("f_sec_pct") is not None else "star-subtracted"
    return (f"<figure markdown>\n![{t}: ORGANIC image (left) and virgil image (right) at the comparison beam](postagb/{key}_images.png)\n"
            f"<figcaption>{t} in H band, {OBS[key]}. Left: the ORGANIC image of De Prins et al. (2026), re-rendered here from their Zenodo FITS "
            f"(10.5281/zenodo.20155600). Right: virgil, {reg}. Both are the {sub} disc emission, convolved to a circular λ_min/2B_max beam of "
            f"{c['beam']['major_mas']:.2f} mas FWHM (grey disc), peak-normalised, North up, East left; the star is the gold marker.</figcaption>\n</figure>\n")


def gen_verdict(key):
    i, c = J(key, "image"), J(key, "compare")
    if not i or not c:
        return ""
    kind, title, body = verdict(key, c, i)
    return admonition(kind, title, body)


def edge_text(s):
    e = s.get("gp_on_edge")
    return "-" if e is None else "yes" if e else "no"


def gen_strategies(key):
    st = strategies(key)
    i = J(key, "image")
    if not st:
        return (f"The five-strategy comparison (MEM, gp-iso, gp-aniso, gp-polar, disc+res) has not been run for {TITLE[key]} yet; the headline above is "
                f"{reg_text(i) if i else 'not available'}.\n")
    pref = J(key, "strategies")
    ev = pref.get("evidence_preferred")
    zbest = st.get(ev, {}).get("log_evidence") if ev else None
    if zbest is None:
        zs = [s["log_evidence"] for s in st.values() if s.get("log_evidence") is not None]
        zbest = max(zs) if zs else None
    paper_d = IMAGE.get(key, {}).get("image_index", IW_PAPER["index"] if key == "IW_Car" else None)
    lines = [f"**Strategies** (headline {HEADLINE}; the same SPARCO scene, hole and error rescaling in every row; the Methods note says what each is). "
             f"ΔlnZ is relative to the evidence-best strategy ({SNAME.get(ev, '-')}); MEM has no evidence.", "",
             f"| Strategy | NCC (best control) | χ²/pt (rescaled errors) | ΔlnZ | f★ / f_env / f_bg+outer (%) | d_env (paper SIND0 = {n_(paper_d)}) | "
             "grid edge | bound warnings | disc+res render guard, n_r |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in STRATS:
        if s not in st:
            continue
        r, si = st[s], J(key, "image", s) or {}
        z = r.get("log_evidence")
        dz = "-" if z is None or zbest is None else sgn(z - zbest, ".1f")
        bw = si.get("bound_warnings")
        warn = "-" if bw is None else (str(len(bw)) + (": " + "; ".join(bw) if bw else ""))
        edge = edge_text(r)
        if s == "disc+res" and si.get("d_env_on_edge") is not None:
            edge += f" (d_env: {'yes' if si['d_env_on_edge'] else 'no'})"
        guard = "-"
        if s == "disc+res":
            guard = (f"χ² {n_(si.get('chi2_per_point'))} model, {n_(si.get('chi2_rendered'))} rendered: "
                     f"{'pass' if si.get('render_ok') else 'FAIL'}; n_r = {si.get('n_r', '-')}")
        name = f"**{SNAME[s]}** (headline)" if s == HEADLINE else SNAME[s] + (" (diagnostic)" if s == "disc+res" else "")
        if s == ev:
            name += " \\*"
        fr = (f"{n_(r.get('f_prim_pct'), '.1f')} / {n_(r.get('f_env_pct'), '.1f')} / {n_(r.get('f_overresolved_pct'), '.1f')}")
        lines.append(f"| {name} | {n_(r.get('ncc'))} ({n_(r.get('ncc_best_control'))}) | {n_(r.get('chi2_per_point'))} | {dz} | {fr} | "
                     f"{n_(r.get('env_index'))} | {edge} | {warn} | {guard} |")
    lines += ["", "\\* the evidence-best strategy. f★ is the primary's share of the 1.65 µm flux, f_env the environment image's, f_bg+outer the grey over-resolved "
              "flux (the background plus the environment's outer region). d_env is the environment's spectral index against De Prins's image index.", ""]
    gp = [s for s in ("gp-iso", "gp-aniso", "gp-polar", "disc+res") if st.get(s, {}).get("gp_on_edge") is not None]
    edge = [SNAME[s] for s in gp if st[s]["gp_on_edge"]]
    d = st.get(HEADLINE, {}).get("log_evidence")
    sent = []
    if ev and ev != HEADLINE and d is not None and zbest is not None:
        gap = d - zbest
        sent.append(f"The evidence prefers {SNAME[ev]}, by ΔlnZ = {sgn(gap, '.1f')} for {HEADLINE}; " + (
            "a gap of that size does not separate them" if abs(gap) <= 5 else "that is a real preference on this grid, but the strategies have very different latent dimensions (see the Methods note), so we do not follow it blindly"
        ) + f", and {HEADLINE} stays the headline.")
    elif ev == HEADLINE:
        sent.append(f"{HEADLINE} is also the evidence-best strategy.")
    if gp:
        sent.append("Selected on the edge of the hyperparameter grid: " + (", ".join(edge) if edge else "none") + (
            f" (of {len(gp)} GP-based fits); an edge value means the grid did not bracket the optimum." if edge else f" (of {len(gp)} GP-based fits)."))
    dr = st.get("disc+res")
    if dr:
        si = J(key, "image", "disc+res") or {}
        sent.append("disc+res is a diagnostic column only, never a headline candidate: " + (
            "it passes the render guard here." if si.get("render_ok") else
            "it fails the render guard here (its rendered pixel image, put back into the scene, does not reproduce the model's χ²/pt), so its image is not a valid reconstruction."))
    lines.append(" ".join(sent))
    strip = OUT / f"{key}_strategies.png"
    if strip.exists():
        lines += ["", '??? note "All strategies side by side"', f"    ![Published image and all five strategies for {TITLE[key]}](postagb/{key}_strategies.png)"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------------------------------
# parameters


def fmt(v, e=None):
    return f"{v:.3g}" if e is None else f"{v:.3g} ± {e:.2g}"


def parametric_table(key, par):
    paper = PMOIRED[key]
    rows = [("Primary flux fraction, H (%)", "f_prim", par.get("f_prim_pct"), None), ("Primary UD diameter (mas)", "ud", par.get("ud_prim_mas"), None),
            ("Secondary flux fraction, H (%)", "f_sec", par.get("f_sec_pct"), None),
            ("Resolved-background flux fraction, H (%)", "f_bg", par.get("f_bg_pct"), None),
            ("Secondary offset Δα (mas)", "sec_dra", par.get("sec_dra"), None), ("Secondary offset Δδ (mas)", "sec_ddec", par.get("sec_ddec"), None),
            ("Rim diameter (mas)", "theta_rim", par.get("rim_diam_mas"), None), ("Rim inclination (°)", "i_rim", par.get("rim_inc_deg"), None),
            ("Rim position angle (°)", "pa_rim", par.get("rim_pa_deg"), 180.0), ("Rim modulation amplitude (m = 1)", "a_mod", par.get("az_amp"), None),
            ("Rim modulation phase (°)", "psi_mod", None if par.get("az_pa_deg") is None else par["az_pa_deg"] - par["rim_pa_deg"], 360.0)]
    lines = ["| Parameter | De Prins+2026 (PMOIRED) | virgil (parametric) | Difference |", "| --- | --- | --- | --- |"]
    for label, k, mine, period in rows:
        if k not in paper or mine is None:
            continue
        val, lo, hi = paper[k]
        d = mine - val
        if period:
            d = (d + period / 2) % period - period / 2
        sig = d / (lo if d < 0 else hi)
        lines.append(f"| {label} | {fmt(val, max(lo, hi))} | {fmt(mine)} | {d:+.3g} ({sig:+.1f}σ) |")
    lines.append(f"| χ² per point | {paper['chi2']:.2f} | {par['chi2_per_point']:.2f} | |")
    return "\n".join(lines)


def image_table(key, i):
    """The SPARCO scene of the ORGANIC image against that of the headline image fit; every virgil number is from image.json."""
    ip = IMAGE[key]
    pm = PMOIRED[key]
    st = i.get("strategy")
    rows = [("Primary flux fraction, H (%)", ip.get("f_prim"), i.get("f_prim_pct"), ".3g")]
    if i.get("f_sec_pct") is not None and ip.get("f_sec") is not None:
        rows.append(("Secondary flux fraction, H (%)", ip["f_sec"], i["f_sec_pct"], ".3g"))
    rows.append(("Primary UD diameter (mas; fixed in virgil)", ip.get("ud"), i.get("ud_fixed_mas"), ".3g"))
    lines = [f"| Parameter | De Prins+2026 (ORGANIC) | virgil ({st or 'image fit'}) | Difference |", "| --- | --- | --- | --- |"]
    for label, pv, mv, spec in rows:
        dif = "-" if pv is None or mv is None else f"{mv - pv:+.3g}"
        lines.append(f"| {label} | {n_(pv, spec)} | {n_(mv, spec)} | {dif} |")
    if i.get("sec_dra") is not None and "sec_dra" in pm:
        lines.append(f"| Secondary offset (Δα, Δδ) (mas; PMOIRED Table 3) | ({pm['sec_dra'][0]:.2f}, {pm['sec_ddec'][0]:.2f}) | ({i['sec_dra']:.2f}, {i['sec_ddec']:.2f}) | "
                     f"≤ {max(abs(i['sec_dra'] - pm['sec_dra'][0]), abs(i['sec_ddec'] - pm['sec_ddec'][0])):.2f} mas |")
    pv = ip.get("image_index")
    lines.append(f"| Primary spectral index (fixed) | ORGANIC SIND1 | {n_(i.get('star_index'), '.3g')} ({i.get('star_index_source', '-')}) | same by construction |")
    lines.append(f"| Disc spectral index d_env (paper's image index SIND0) | {n_(pv, '.3g')} | {n_(i.get('env_index'), '.3g')} | {sgn(None if pv is None or i.get('env_index') is None else i['env_index'] - pv, '.2f')} |")
    return "\n".join(lines)


def gen_parameters(key):
    i = J(key, "image")
    par = J(key, "parametric")
    if not i or not par:
        return ""
    nfiles = int(re.match(r"\d+", OBS[key]).group(0))
    starts = ", ".join(f"{x:.1f}" for x in par.get("chi2_starts", []))
    if key == "IW_Car":
        return iw_parameters(i, par, nfiles, starts)
    return (f"**Parameters.** The first table compares the PMOIRED geometric model of De Prins et al. (Table 3) with a PMOIRED-style parametric fit of virgil "
            f"to the same {nfiles} files (UD star, power-law rim with an m = 1 modulation, plus the paper's optional secondary or resolved background). The σ "
            "column uses the paper's uncertainty only; virgil's are point estimates, so large σ values mostly say how tight the paper's errors are. The rim "
            "diameter is that of a Gaussian ring in virgil and of a power-law rim in PMOIRED, so that row is not like for like; the modulation phase is "
            "virgil's `az_pas` minus the rim position angle, which is PMOIRED's `az projang` (docs/method/pmoired.md). The parametric fit takes the lowest "
            f"χ² of {len(par.get('chi2_starts', []))} starts ({starts}).\n\n" + parametric_table(key, par) + "\n\n"
            f"The SPARCO scene of the headline image fit ({headline_label(i)}) against that of the ORGANIC image (Table 2 and FITS header). The star's "
            "spectral index is not fitted: it is De Prins's own SIND1, so d_env is comparable with the paper's. The secondary's offset and flux carry "
            "PMOIRED priors (Methods note), so agreement on those rows shows the data do not pull them away, not a blind recovery.\n\n"
            + image_table(key, i) + "\n")


def iw_parameters(i, par, nfiles, starts):
    P = IW_PAPER
    v = par["values"]
    d = lambda a, b: sgn(a - b)  # noqa: E731
    pm_ud = par.get("ud_prim_mas")
    az_pa = par["az_pa_deg"] % 360 if par.get("az_pa_deg") is not None else None
    s_out = (par["rim_inc_deg"] - P["inc_outer"][0]) / P["inc_outer"][1]
    pa_o = ((par["rim_pa_deg"] - P["pa_outer"][0] + 90) % 180 - 90) / P["pa_outer"][1]
    pa_i = ((par["rim_pa_deg"] - P["pa_inner"][0] + 90) % 180 - 90) / (P["pa_inner"][2] if par["rim_pa_deg"] > P["pa_inner"][0] else P["pa_inner"][1])
    rows = [
        ("Primary flux fraction, H (%)", f"{P['f_prim']:.1f}", n_(i.get("f_prim_pct"), ".1f"), n_(par.get("f_prim_pct"), ".1f"),
         f"{d(i['f_prim_pct'], P['f_prim'])} / {d(par['f_prim_pct'], P['f_prim'])} points"),
        ("Secondary flux fraction, H (%)", f"{P['f_sec']:.2f}", n_(i.get("f_sec_pct")), n_(par.get("f_sec_pct")),
         f"{d(i['f_sec_pct'], P['f_sec'])} / {d(par['f_sec_pct'], P['f_sec'])} points"),
        ("Secondary offset (Δα, Δδ) (mas)", f"({P['sec'][0]:.2f}, {P['sec'][1]:.2f})", f"({i['sec_dra']:.2f}, {i['sec_ddec']:.2f})",
         f"({par['sec_dra']:.2f}, {par['sec_ddec']:.2f})",
         f"≤ {max(abs(i['sec_dra'] - P['sec'][0]), abs(i['sec_ddec'] - P['sec'][1]), abs(par['sec_dra'] - P['sec'][0]), abs(par['sec_ddec'] - P['sec'][1])):.1f} mas"),
        ("Primary UD diameter (mas)", f"{P['ud']:.2f}", f"{i['ud_fixed_mas']:.2f} (fixed)", n_(pm_ud),
         f"{d(i['ud_fixed_mas'], P['ud'])} / {d(pm_ud, P['ud'])} mas" if pm_ud is not None else "-"),
        ("Disc spectral index d_env (image)", f"{P['index']:.2f}", n_(i.get("env_index"), ".3g"), f"not comparable (rim {par['rim_index']:.1f})",
         d(i["env_index"], P["index"])),
        ("Inner-arc rim diameter (mas)", f"{P['rim_inner'][0]} ± {P['rim_inner'][1]}", "see metrics", f"{par['rim_diam_mas']:.1f} (outer-arc scale)", "not the same structure"),
        ("Rim inclination (°)", f"{P['inc_inner'][0]} ± {P['inc_inner'][1]} (inner), {P['inc_outer'][0]} ± {P['inc_outer'][1]} (outer)", "n/a",
         f"{par['rim_inc_deg']:.1f}", f"{sgn(s_out, '.1f')}σ against the outer arc"),
        ("Rim position angle (°)", f"{P['pa_inner'][0]} (+{P['pa_inner'][2]}, {MINUS}{P['pa_inner'][1]}) (inner), {P['pa_outer'][0]} ± {P['pa_outer'][1]} (outer)", "n/a",
         f"{par['rim_pa_deg']:.1f}", f"{sgn(pa_o, '.1f')}σ against the outer arc, {sgn(pa_i, '.1f')}σ against the inner"),
        ("Rim azimuthal modulation (m = 1)", "not reported", "n/a", f"amplitude {par['az_amp']:.2f} at {az_pa:.0f}°" if az_pa is not None else "-", "none to compare"),
    ]
    tab = "\n".join(f"| {a} | {b} | {c} | {e} | {f} |" for a, b, c, e, f in rows)
    return (f"**Parameters.** De Prins et al. could not fit a PMOIRED geometric model to IW Car (a single disc does not reproduce it), so there is no PMOIRED "
            "row to compare with. The comparison is with what the paper does report for IW Car: the SPARCO parameters of the ORGANIC reconstruction (FITS "
            "headers), and the inner-rim ellipses fitted to the image (Table 2; inner arcs and outer arc separately). The \"virgil\" columns are two fits to "
            f"the same {nfiles} files: the SPARCO parameters fitted together with the headline image ({headline_label(i)}), and a PMOIRED-style parametric "
            "model (UD star + point secondary + power-law rim with an m = 1 modulation + resolved background; "
            + (f"lowest χ² of {len(par['chi2_starts'])} starts: {starts}" if starts else "one fit") + "). virgil's fits are point estimates; the σ column uses the paper's "
            "uncertainty, where it gives one. The image fit keeps De Prins's stellar index SIND1 and secondary index SIND2 = 0 fixed, and its secondary "
            "carries PMOIRED position and flux priors (Methods note), so agreement in this table shows that the data do not pull the values away from the "
            "published ones, not that virgil recovers them blind.\n\n"
            "| Parameter | De Prins+2026 | virgil (image fit) | virgil (parametric) | Difference |\n| --- | --- | --- | --- | --- |\n" + tab + "\n\n"
            "The parametric rim lands between the two orientations the paper finds for the inner and outer arcs (and near the SPHERE outer disc, i = 41.7°, "
            "PA = 161°, Andrych et al. 2023), which says the single-ring model is describing a blend of structures, as the paper concluded. Its χ² per point is "
            f"{par['chi2_per_point']:.1f}, against {i['chi2_per_point']:.2f} for the image fit, for the same reason.\n")


# ---------------------------------------------------------------------------------------------------------------------------------
# metrics


def metrics_table(key, c, i):
    o, p = c["ours"], c["published"]
    name = "Hillen+2016" if key == "IRAS08544" else "ORGANIC"
    mine = f"virgil ({i['strategy']})" if i and i.get("strategy") else "virgil"
    head = (f"| Metric | {name} | {mine} |\n| --- | --- | --- |\n| Normalised cross-correlation (controls: {ctl(c)}) | 1 | "
            f"{c['ncc_zero_shift']:.2f} ({c['ncc_best_shift']:.2f} after the best sub-beam shift) |\n")
    if key == "IRAS08544":
        return head + (
            f"| Inside 3.5 mas / ring (3.5 to 16 mas) flux fraction | {100 * p['inner']['flux_fraction']:.1f} % / {100 * p['outer']['flux_fraction']:.1f} % | "
            f"{100 * o['inner']['flux_fraction']:.1f} % / {100 * o['outer']['flux_fraction']:.1f} % |\n"
            f"| Ring flux-weighted radius (mas) | {p['outer']['mean_radius_mas']:.2f} | {o['outer']['mean_radius_mas']:.2f} |\n"
            f"| Ring major-axis PA (°) | {p['outer']['major_pa_deg']:.0f} | {o['outer']['major_pa_deg']:.0f} (the ring is nearly circular, axis ratio "
            f"{p['outer']['axis_ratio']:.2f} and {o['outer']['axis_ratio']:.2f}, so this is ill-defined) |\n"
            f"| Asymmetry angle, ring flux centroid (° East of North), offset (mas) | {p['outer']['centroid_pa_deg']:.0f}, {p['outer']['centroid_offset_mas']:.2f} | "
            f"{o['outer']['centroid_pa_deg']:.0f}, {o['outer']['centroid_offset_mas']:.2f} |\n")
    if key == "IW_Car":
        rows = []
        for reg, nm in (("inner", "Inner"), ("outer", "Outer")):
            rows.append((f"{nm} flux fraction (%)", f"{100 * p[reg]['flux_fraction']:.1f}", f"{100 * o[reg]['flux_fraction']:.1f}"))
            rows.append((f"{nm} flux-weighted radius (mas)", f"{p[reg]['mean_radius_mas']:.2f}", f"{o[reg]['mean_radius_mas']:.2f}"))
            rows.append((f"{nm} major-axis PA (°), axis ratio", f"{p[reg]['major_pa_deg']:.0f}, {p[reg]['axis_ratio']:.2f}", f"{o[reg]['major_pa_deg']:.0f}, {o[reg]['axis_ratio']:.2f}"))
            rows.append((f"{nm} flux centroid (° East of North), offset (mas)", f"{p[reg]['centroid_pa_deg']:.0f}, {p[reg]['centroid_offset_mas']:.2f}",
                         f"{o[reg]['centroid_pa_deg']:.0f}, {o[reg]['centroid_offset_mas']:.2f}"))
    else:
        d, q = o["disc"], p["disc"]
        rows = [("Flux fraction in the image field", f"{100 * q['flux_fraction']:.0f} %", f"{100 * d['flux_fraction']:.0f} %"),
                ("Flux-weighted radius (mas)", f"{q['mean_radius_mas']:.2f}", f"{d['mean_radius_mas']:.2f}"),
                ("Major-axis PA (°), axis ratio", f"{q['major_pa_deg']:.0f}, {q['axis_ratio']:.2f}", f"{d['major_pa_deg']:.0f}, {d['axis_ratio']:.2f}"),
                ("Asymmetry angle, flux centroid (° East of North), offset (mas)", f"{q['centroid_pa_deg']:.0f}, {q['centroid_offset_mas']:.2f}",
                 f"{d['centroid_pa_deg']:.0f}, {d['centroid_offset_mas']:.2f}")]
    return head + "".join(f"| {a} | {b} | {c_} |\n" for a, b, c_ in rows)


def gen_metrics(key):
    c, i = J(key, "compare"), J(key, "image")
    if not c or not i:
        return ""
    bm = c["beam"]["major_mas"]
    if key == "IRAS08544":
        intro = (f"**Image metrics** (both images on one {c['pixel_mas']:.2f} mas grid, virgil convolved to the published beam, the published bitmap used as it "
                 "stands; the binary is subtracted in both, so there is no central source to compare; both images get the same moments):")
    elif key == "IW_Car":
        pa = c["published"]["outer"]["major_pa_deg"]
        intro = (f"**Image metrics** (both images on one {c['pixel_mas']:.2f} mas grid, convolved to the λ_min/2B_max beam ({bm:.2f} mas FWHM); inner = within 5 mas of "
                 "the star, outer = beyond; the paper's own rim fit needs per-pixel significances, so both images get the same moments. As an external check, the moments "
                 f"of the ORGANIC image give an outer major axis at PA {pa:.0f}° against the paper's {IW_PAPER['pa_outer'][0]} ± {IW_PAPER['pa_outer'][1]}° from its rim fit, "
                 f"within {abs((pa - IW_PAPER['pa_outer'][0] + 90) % 180 - 90):.0f}°, so the two definitions are consistent):")
    else:
        intro = (f"**Image metrics** (both images on one {c['pixel_mas']:.2f} mas grid, convolved to the λ_min/2B_max beam ({bm:.2f} mas FWHM); the paper's own rim fit "
                 "needs per-pixel significances, so both images get the same moments):")
    return intro + "\n\n" + metrics_table(key, c, i)


# ---------------------------------------------------------------------------------------------------------------------------------
# model and fit quality (the "Data and method" tail)


def hole_text(i):
    rule = i.get("hole_rule")
    if rule:
        why = f"{rule}"
    elif i.get("hole_beams") is not None:
        why = f"{i['hole_beams']:g} of the data beam's minor axis"
    else:
        why = "rule not recorded"
    return f"{i['hole_mas']:.2f} mas ({why}; {i['n_blanked']} pixels)"


def scene_text(key, i):
    sec = ""
    if i.get("f_sec_pct") is not None and i.get("sec_dra") is not None:
        sec = ("a point-source secondary "
               + ("(position and flux with PMOIRED priors, index fixed at the paper's SIND2, no image window inside the primary's hole)"
                  if i.get("sec_treatment") else "(free position, flux and power-law index, started at the published binary)") + ", ")
    return (f"a uniform-disc primary of fixed diameter {i['ud_fixed_mas']:.3g} mas with flux ∝ λ^{n_(i.get('star_index'), 'g')} "
            f"({i.get('star_index_source', 'index as in the paper')}), {sec}and a `virgil.models.Image` environment on a {i['grid'][0]} × {i['grid'][1]} grid of "
            f"{i['pixel_mas']:.3g} mas pixels (its own power-law index d_env = {i['env_index']:.2f}), plus a grey over-resolved background where the strategy fits one")


def gen_model(key):
    i, c = J(key, "image"), J(key, "compare")
    if not i or not c:
        return ""
    b = c["virgil_data_beam"]
    s = i.get("error_scale")
    flips = c.get("published_orientation_chi2") or {}
    pub = c.get("chi2_per_point_published_on_our_data")
    out = [f"**Beam and hole.** The PIONIER dirty beam is {b['major_mas']:.2f} × {b['minor_mas']:.2f} mas at PA {b['pa_deg']:.0f}°. The comparison beam is circular, "
           f"FWHM λ_min/(2B_max) = {c['beam']['major_mas']:.2f} mas, for both images. The environment image has a window of radius {hole_text(i)} blanked around the primary"
           + (", with no window inside the primary's hole around the secondary (its position and flux are set by the PMOIRED priors)" if i.get("sec_treatment") else "")
           + (f", inside a circular support of radius {i['support_radius_mas']:.1f} mas on a square grid reaching ±{grid_half(i):.1f} mas"
              + (", which clips the support (this fit predates the padded grid)" if grid_half(i) < i["support_radius_mas"] - 0.5 * i["pixel_mas"] else "") + ")" if i.get("support_radius_mas") and grid_half(i) else "")
           + ", so that its pixels cannot trade flux with the stellar point sources."]
    chi = f"{i['chi2_per_point']:.2f}"
    err = (f" on errors rescaled by MacKay's evidence (s = {s:.2f}; the Methods note explains why χ²/pt then sits below 1)" if s else "")
    if key == "IW_Car":
        # IW Car: the whitened-residual vector is longer than the number of data (the Data paragraph's counts)
        v2, t3 = IW_DATA
        nd = v2 + t3
        resid = (f" The {i['n_points']} is the length of virgil's whitened-residual vector, not the number of data: the {v2} V² and {t3} closure phases ({nd} samples) become "
                 f"{i['n_points']} residuals because each closure phase of four or more telescopes also adds a periodic penalty term. Dividing by {nd} instead would give "
                 f"about {i['chi2_per_point'] * i['n_points'] / nd:.2f} and {pub * i['n_points'] / nd:.2f}.") if pub else ""
    else:
        resid = ""
    out.append(f"**Model.** SPARCO (Methods note): {scene_text(key, i)}. Headline strategy {headline_label(i)}: {reg_text(i)}"
               + (f", selected by the Laplace evidence (ln Z = {i['log_evidence']:.0f})" if i.get("log_evidence") is not None else "")
               + (", at the edge of its hyperparameter grid" if i.get("gp_on_edge") else "")
               + (f", on a polar grid deprojected by the parametric disc's inclination {i['mp_inc_deg']:.0f}° and PA {i['mp_pa_deg']:.0f}°" if i.get("mp_inc_deg") is not None and i["regulariser"] == "MultipoleField" else "")
               + ". ORGANIC differs: a GAN generator as its regulariser, with the primary's parameters tuned with pyBADS. Neither is a ground truth.")
    orient = (f" Flipping the published image East-West or North-South gives χ² {flips['mirrored_East-West']:.1f} and {flips['mirrored_North-South']:.1f} "
              f"against {flips['as_published']:.2f} as published, so the published orientation is the better one.") if flips else ""
    out.append(f"**Fit quality.** On the same data the virgil {headline_label(i)} image has χ² per point {chi}{err} and the ORGANIC image, with its published SPARCO "
               f"parameters unchanged, {n_(pub)} ({c['n_points']} residuals, same rescaled errors). This does not say virgil's image is better: it has thousands of free "
               f"pixels for those points and its prior is the only regularisation.{resid}{orient}")
    return "\n\n".join(out) + "\n"


def gen_iras_model(key):
    i, c = J(key, "image"), J(key, "compare")
    if not i or not c:
        return ""
    b, v = c["beam"], c["virgil_data_beam"]
    out = [f"virgil's own dirty beam is {v['major_mas']:.1f} × {v['minor_mas']:.1f} mas at PA {v['pa_deg']:.0f}°; the images above use the published ellipse "
           f"({b['major_mas']:.1f} × {b['minor_mas']:.1f} mas, PA {b['pa_deg']:.0f}°) instead, so that both are shown at the resolution of the figure in the paper. This target "
           "does not use the λ_min/2B_max beam of the other tabs, because the published image is already restored at its own beam and cannot be sharpened. The "
           f"environment image has a window of radius {hole_text(i)} blanked around the primary and around the published position of the secondary."]
    reg = i["regulariser"]
    weight = f" (weight {i['weight']:.3g})" if reg == "MaxEntropy" and i.get("weight") else ""
    out.append("**Model.** The paper's binary-subtracted scene, Hillen's own setup: a uniform-disc primary "
               f"({i['ud']:.1f} mas fixed, flux ∝ λ^{n_(i.get('star_index', -4.0), 'g')} as in the paper)"
               + ", a point-source secondary (free position, flux and power-law index, started at the published binary) and a `virgil.models.Image` environment with a "
               f"power-law spectrum from `virgil.spectra` on the paper's grid ({i['grid'][0]} × {i['grid'][1]} pixels of {i['pixel_mas']:.2f} mas), starting from "
               f"`starting_image`. Regulariser: {reg_text(i)}{weight}. The paper instead fixes the primary's flux fraction and the environment index from a 20 × 20 grid and uses a "
               "regularisation weight of 10¹² in its own units, so the two are not the same optimisation. Neither image is a ground truth.")
    out.append(f"**Fit quality.** The virgil image has χ² per point {i['chi2_per_point']:.2f} on {i['n_points']} residuals. That is low because there are about "
               f"{i['grid'][0] * i['grid'][1]:,} free pixels for those residuals, so it says little about the image's quality, and the error basis is not settled (see the "
               "parametric fit above); the paper does not tabulate a χ² for its image.")
    return "\n\n".join(out) + "\n"


def sigma(x, ref, e):
    return f" ({(x - ref) / e:+.1f}σ)" if e and x is not None else ""


def gen_iras_binary(key):
    i = J(key, "image")
    par = J(key, "parametric")
    if not i:
        return ""

    def row(label, k, mine, spec=".1f", note=""):
        v, e = HILLEN[k]
        if mine is None:
            return f"| {label} | {v:{spec}}{f' ± {e:g}' if e else ''} | - | - |\n"
        return f"| {label} | {v:{spec}}{f' ± {e:g}' if e else ''} | {mine:{spec}} | {mine - v:+.2g}{sigma(mine, v, e)}{note} |\n"
    si = i.get("sec_index")
    pinned = "" if si is None else (" at the prior's lower bound (−5)" if si <= -4.9 else "")
    tab = ("| Parameter | Hillen+2016 | virgil (image fit) | Difference |\n| --- | --- | --- | --- |\n"
           + row("Primary flux fraction, H (%)", "f_prim", i.get("f_prim_pct")) + row("Secondary flux fraction, H (%)", "f_sec", i.get("f_sec_pct"))
           + row("Secondary separation (mas)", "sep", i.get("sec_sep_mas"), ".2f") + row("Secondary position angle from the primary (°)", "pa", i.get("sec_pa_deg"), ".0f")
           + f"| Secondary spectral index (power law at 1.65 µm) | −2.5 (−3.1 to −0.6; from 4000 ± 2000 K) | {n_(si, '.1f')} |{pinned or ' -'} |\n"
           + row("Environment spectral index", "env_index", i.get("env_index"), ".2f"))
    sep_d = None if i.get("sec_sep_mas") is None else (i["sec_sep_mas"] - HILLEN["sep"][0]) / HILLEN["sep"][1]
    pa_d = None if i.get("sec_pa_deg") is None else i["sec_pa_deg"] - HILLEN["pa"][0]
    text = (tab + "\n**This binary table is not a blind recovery.** The image fit starts at Hillen's separation, position angle and flux ratio, and its priors "
            "(±1 mas on each coordinate, the flux ratio, an index range of −5 to 2) sit on Hillen's values, so agreement here shows only that the data do not move "
            "the binary away from them. The blind test is the parametric fit above (four generic starting points).\n\n"
            f"The secondary's separation differs from Hillen's table value by {sgn(sep_d, '.1f')}σ and its angle by {sgn(pa_d, '.0f')}° from the table's 236°. That table value is itself in doubt: the star markers drawn on the published figure sit at {IRAS_STAR_READING}. "
            "A faint companion this close to the primary (inside the hole) trades flux and slope with the primary, so the secondary's index is not measured"
            + (" (it runs to the lower bound of its prior)" if pinned else "") + ".\n")
    if par:
        c2 = par["chi2_per_point"]
        text += (f"\nThe reduced χ² of the parametric fit is {c2:.1f} and the paper quotes 2.3 after inflating its uncertainties by √6, but the two are not shown to be on "
                 "the same error basis. The files' own errors (median V² error 2.4 % of V², closure phase about 1°) look like pipeline errors, and nothing in them says "
                 f"whether the √6 inflation is already applied. If it is not, the paper's χ² on these errors would be about {2.3 * 6:.0f}, {2.3 * 6 / c2:.0f} times virgil's "
                 f"for the same geometry, which would be an unexplained disagreement (the χ² counts also depend on how spectral channels are binned). If it is, {c2:.1f} against "
                 "2.3 compares like with like and virgil fits slightly better. This is unresolved here, so the χ² values are not used as evidence.\n")
    return text


# ---------------------------------------------------------------------------------------------------------------------------------
# page-level regions


def gen_summary(_):
    lines = ["| Target | Published method | Headline (gp-polar) | Verdict | NCC (best control) | Best other strategy NCC | Comparison beam (mas) |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for t, key in TABS.items():
        c, i = J(key, "compare"), J(key, "image")
        if not c or not i:
            lines.append(f"| [{t}](#{anchor(t)}) | {PUBLISHED[key]} | - | not yet available | - | - | - |")
            continue
        st = strategies(key)
        other = [(s, st[s]["ncc"]) for s in OTHERS if s in st and st[s].get("ncc") is not None]
        bo = "-" if not other else "{:.2f} ({})".format(*(lambda t_: (t_[1], t_[0]))(max(other, key=lambda t_: t_[1])))
        head = headline_label(i) + (" (grid edge)" if i.get("gp_on_edge") else "") + ("" if i.get("strategy") else " (single run, strategies not yet fitted)")
        beam = f"{c['beam']['major_mas']:.2f}" + (f" (Hillen: drawn {c['beam']['major_mas']:.1f} × {c['beam']['minor_mas']:.1f} ellipse)" if key == "IRAS08544" else "")
        lines.append(f"| [{t}](#{anchor(t)}) | {PUBLISHED[key]} | {head} | {short_verdict(key, c)} | {c['ncc_zero_shift']:.2f} ({best_control(c):.2f}) | {bo} | {beam} |")
    return "\n".join(lines) + "\n"


def anchor(t):
    return t.lower().replace(" ", "-")


def rng(vals, spec=".2f"):
    vals = [v for v in vals if v is not None]
    return "-" if not vals else f"{min(vals):{spec}} to {max(vals):{spec}}"


def grid_half(i):
    """Half-width of the fitted square grid, measured from the saved grid and pixel (not the configured field radius)."""
    return i["grid"][0] * i["pixel_mas"] / 2 if i.get("grid") and i.get("pixel_mas") else None


def grid_text(i):
    """'half-width / support radius (mas)', flagged when the square grid is narrower than the support disc."""
    gh, sr = grid_half(i), i.get("support_radius_mas")
    if gh is None:
        return "-"
    return f"{gh:.1f} / {sr:.1f}" + (" (clips the support)" if sr and gh < sr - 0.5 * i["pixel_mas"] else "") if sr else f"{gh:.1f}"


def gen_methods(_):
    imgs = {k: J(k, "image") for k in TABS.values()}
    rows = ["| Target | Stellar index (SIND1) | UD fixed (mas) | Error scale s | Hole (mas) | Grid half-width / support radius (mas) | Headline σ, ℓ | Headline χ²/pt |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for t, k in TABS.items():
        i = imgs[k]
        if not i:
            continue
        if i["regulariser"] == "MultipoleField":
            gp = f"σ = {i['gp_sigma']:g}, ℓ_r = {i['gp_length_r_mas']:.2f} mas, ℓ_θ = {i['gp_length_theta']:g} rad"
        elif i["regulariser"] == "GaussianField":
            gp = f"σ = {i['gp_sigma']:g}, ℓ = {i['gp_length_mas']:.2f} mas"
        else:
            gp = i["regulariser"]
        rows.append(f"| {t} | {n_(i.get('star_index'), '.3g')} | {n_(i.get('ud_fixed_mas'), '.3g')} | {n_(i.get('error_scale'))} | {i['hole_mas']:.2f} | "
                    f"{grid_text(i)} | {gp} | {i['chi2_per_point']:.2f} |")
    de = [imgs[k] for k in DE_PRINS if imgs[k]]
    sidx = [i["star_index"] for i in de if i.get("star_index") is not None]
    scl = [i["error_scale"] for i in de if i.get("error_scale") is not None]
    chis = [i["chi2_per_point"] for i in de if i.get("regulariser") == "MultipoleField"]
    ps, hr = imgs.get("PS_Gem"), imgs.get("HR4049")
    # strategy-level facts
    allst = {k: strategies(k) for k in DE_PRINS}
    gp_fits = [(k, s) for k, st in allst.items() for s in ("gp-iso", "gp-aniso", "gp-polar") if st.get(s, {}).get("gp_on_edge") is not None]
    on_edge = [(k, s) for k, s in gp_fits if allst[k][s]["gp_on_edge"]]
    pol_edge = [TITLE[k] for k in DE_PRINS if allst[k].get(HEADLINE, {}).get("gp_on_edge")]
    ev_best = {k: J(k, "strategies").get("evidence_preferred") for k in DE_PRINS if J(k, "strategies")}
    polar_best = [TITLE[k] for k, v in ev_best.items() if v == HEADLINE]
    non_polar = "; ".join(f"{TITLE[k]}: {v}" for k, v in ev_best.items() if v != HEADLINE)
    bgb = [TITLE[k] for k in DE_PRINS if imgs[k] and any(w.startswith("bg.") for w in (imgs[k].get("bound_warnings") or []))]
    guard = [(k, J(k, "image", "disc+res")) for k in DE_PRINS if J(k, "image", "disc+res")]
    g_fail = [TITLE[k] for k, j in guard if not j.get("render_ok")]
    nr = sorted({j.get("n_r") for _, j in guard if j.get("n_r") is not None})
    # optional one-off comparisons, only if their files exist
    si4 = ""
    a, b_ = J("HD95767", "image", "gp-polar_si4"), J("HD95767", "compare", "gp-polar_si4")
    h, hc = imgs.get("HD95767"), J("HD95767", "compare")
    if a and b_ and h and hc:
        si4 = (f" A one-off check on HD 95767 (gp-polar) with the earlier −4 index instead of SIND1 = {n_(h.get('star_index'), 'g')} changes the NCC from "
               f"{hc['ncc_zero_shift']:.2f} to {b_['ncc_zero_shift']:.2f} and d_env from {h['env_index']:.2f} to {a['env_index']:.2f}, with the log evidence "
               f"{h['log_evidence'] - a['log_evidence']:+.0f} in favour of SIND1: the disc index is degenerate with the fixed star colour.")
    bg0 = ""
    a, b_ = J("IW_Car", "image", "gp-polar_bg0"), J("IW_Car", "compare", "gp-polar_bg0")
    h, hc = imgs.get("IW_Car"), J("IW_Car", "compare")
    if a and b_ and h and hc:
        bg0 = (f" Fixing the background's index at 0 (grey) instead of fitting it, for IW Car (gp-polar, bg0 run), leaves the image alike (NCC {b_['ncc_zero_shift']:.2f}, against "
               f"{hc['ncc_zero_shift']:.2f} with it free) but fits worse (χ²/pt {a['chi2_per_point']:.2f} against {h['chi2_per_point']:.2f}, log evidence "
               f"{a['log_evidence']:.0f} against {h['log_evidence']:.0f}) and moves d_env from {h['env_index']:.2f} to {a['env_index']:.2f}: the background index and d_env are degenerate, "
               "so quoted d_env values carry that systematic uncertainty. We keep the index free, as the data prefer.")
    psc, pbeam = J("PS_Gem", "scale"), J("PS_Gem", "compare")
    ud_only = (f" For PS Gem the hole is the UD diameter only, {psc['hole_mas']:.2f} mas, because the beam-size hole blanked compact emission there; "
               "the table shows the hole of each stored fit." if psc and pbeam and psc["hole_mas"] < pbeam["beam"]["major_mas"] else "")
    sections = [
        "**The common scene (SPARCO).** Every strategy fits the same scene (Kluska et al. 2014): a uniform-disc primary of fixed diameter whose flux goes as "
        "λ^index (normalised at 1.65 µm); a point-source secondary where the paper has one; an environment image with a free flux fraction and its own spectral index "
        "d_env; and a grey (index-free) over-resolved background. The strategies differ only in how the environment image is regularised.",
        f"**Spectral indices.** We adopt De Prins's own per-target stellar index SIND1 from the ORGANIC FITS header, held fixed ({rng(sidx, '.2f')} across the eight "
        "targets), and the companion index SIND2 = 0, also fixed, so that our d_env values are comparable with the paper's image index (SIND0). An earlier convention "
        "of −4 for every star was wrong for this paper and has been removed." + si4,
        "**The secondary.** Its position and flux ratio carry PMOIRED priors (Table 3 of the paper), and its index is fixed at SIND2. The image has no window inside "
        "the primary's hole, so the secondary's own flux is not traded against pixels there; for HD 95767 this removed inner knots that the earlier treatment produced.",
        f"**Error rescaling.** The data errors are multiplied by a single factor s chosen by MacKay's evidence argument on an isotropic GP pilot "
        f"(s = {rng(scl)} across the targets, see the table). All χ²/pt on this page are on the rescaled errors. The headline χ²/pt is "
        f"{rng(chis)} on the targets fitted with gp-polar: well below 1 after rescaling, which says the PIONIER errors are conservative or that the latent field is "
        "flexible enough to absorb noise, so χ²/pt cannot be used to rank strategies.",
        f"**Hole, support and field.** The environment image has no pixels within a hole of radius max(λ_min/2B_max, UD diameter) around the primary.{ud_only} The image "
        "support is a circle, with a per-target radius; the table gives the measured half-width of each square grid next to it. A grid narrower than its support clips the "
        "image: the targets so marked were fitted before the grid was padded to the requested field and will be refitted.",
        "**Strategies.** MEM is MaxEntropy with the weight at the discrepancy principle (no evidence). gp-iso is an isotropic Gaussian-process prior on the log-brightness "
        "(σ over 0.25 to 16, ℓ from one pixel to twice the comparison beam; up to 8 beams for HR 4049). gp-aniso stretches the correlation length along the disc's major axis. gp-polar is "
        "`virgil.fields.MultipoleField`, a polar GP (Matérn in radius times a periodic kernel in azimuth) deprojected by the parametric fit's inclination and PA "
        "(σ over 0.125 to 2; ℓ_θ over 0.25 to 2 rad and ℓ_r from 0.5 to 4 beams as fitted for most targets, ℓ_θ from 0.125 and ℓ_r up to 8 beams for HR 4049, the only target refitted on the finer grid). In each GP strategy the hyperparameters are the cell of the grid with the largest "
        "Laplace evidence. A radial cap on the polar grid (the parametric rim radius plus the larger of three comparison beams and half a radius, the rest left to the grey background) was tested on IW Car and HD 95767: it lowered the log evidence by hundreds and degraded the images (IW Car NCC 0.74 against 0.90, HD 95767 0.88 against 0.94), so the faint outer arcs carry real, partly resolved flux that the fully resolved background cannot absorb. The polar grid therefore runs to the whole field. Its cells are uniform in radius, so at the field edge they can form concentric arcs: read the outermost emission with that in mind. "
        "gp-polar is our recommended model and the headline of every tab.",
        "**Why the evidence is not followed blindly.** The strategies have very different latent dimensions (a pixel field, a polar grid, a disc plus residual), so "
        f"their Occam factors are not like for like; and with headline χ²/pt of {rng(chis)} every fit is already well inside the noise. "
        f"gp-polar is the evidence-best strategy on {', '.join(polar_best) if polar_best else 'none of the targets'}"
        + (f"; elsewhere the evidence prefers {non_polar}" if non_polar else "") + ". "
        "A small ΔlnZ (a few units) means the data do not choose between strategies.",
        f"**Grid edges.** {len(on_edge)} of {len(gp_fits)} GP-based selections sit on the edge of their hyperparameter grid (the flags are in each tab's strategy table, "
        f"generated from `gp_on_edge`). For the gp-polar headline the edge flag is set on: {', '.join(pol_edge) if pol_edge else 'none'}. An edge value means the grid "
        "did not bracket the optimum, so the hyperparameters are not measured.",
        f"**disc+res and the render guard.** disc+res (a parametric disc plus a coarse residual field, n_r = {', '.join(map(str, nr)) if nr else '-'} radial knots) is shown "
        "as a diagnostic column only. Its render guard puts the disc's rendered pixel image back into the fitted scene and requires the rendered χ²/pt to reproduce the "
        "model's within 10 %: |rendered − model| ≤ 0.1 × model + 0.02. Cells that fail are excluded from its selection, and the guard fails "
        f"for the final disc+res image on {', '.join(g_fail) if g_fail else 'none of the targets'} ({len(g_fail)} of {len(guard)} run). It is never a headline candidate.",
        f"**The grey background.** The background's index and flux sit at a prior bound on most targets ({len(bgb)} of {len(de)}: {', '.join(bgb) if bgb else 'none'}); the grey "
        "background is a few per cent of the flux and its index is unconstrained, so we report f_bg plus the outer environment flux as one total over-resolved flux." + bg0,
        "**Verdicts.** Normalised cross-correlation is dominated by the radial profile, so each tab lists three controls (the East-West mirror, the 180° rotation and the "
        f"azimuthal average of the published image). A verdict says 'agree' only if the NCC exceeds its best control by at least {MARGIN:g} and is itself at least {MIN_NCC:g}; between {FLOOR:g} and "
        f"{MARGIN:g} it says 'inconclusive'; below {FLOOR:g} it says 'does not agree'. SQUEEZE also failed on HR 4049 and PS Gem in the paper (Appendix D), so "
        "those two are documented hard cases, not only virgil's.",
        "**IRAS 08544-4431 is different.** It follows Hillen et al. (2016): stellar index −4, secondary index free (it runs to its −5 lower bound), beam as drawn on their "
        "figure, and its binary table is not a blind recovery. It is compared with their published bitmap, not with De Prins's ORGANIC image.",
    ]
    return "\n\n".join(sections) + "\n\nSetup of each target, from the stored fits:\n\n" + "\n".join(rows) + "\n"


def gen_parity(_):
    cs = {k: J(k, "compare") for k in DE_PRINS if k != "IW_Car"}
    cs = {k: c for k, c in cs.items() if c}
    az = [c["ncc_controls"]["azimuthal_average_of_published"] for c in cs.values()]
    wr = [max(c["ncc_controls"]["mirrored_East-West"], c["ncc_controls"]["rotated_180"]) for c in cs.values()]
    flips = {k: c.get("published_orientation_chi2") for k, c in cs.items() if c.get("published_orientation_chi2")}
    wins = sum(f["as_published"] < min(f["mirrored_East-West"], f["mirrored_North-South"]) for f in flips.values())
    return (f"A caution on the cross-correlation for the {len(cs)} near-round discs of De Prins et al.: it is dominated by the radial profile. The azimuthal average of "
            f"the published image already correlates at {rng(az)} with virgil's headline images, and the East-West and 180° controls at up to {rng(wr)}, so a high NCC "
            "means the radial structure agrees and says less about the azimuthal structure; that is why the verdicts are gated on the best control. The orientation-flip χ² "
            f"of the published image on our data is the sharper test of orientation, and the published orientation wins it for {wins} of {len(flips)} targets.\n")


GEN = dict(figure=gen_figure, verdict=gen_verdict, strategies=gen_strategies, parameters=gen_parameters, metrics=gen_metrics, model=gen_model,
           iras_model=gen_iras_model, iras_binary=gen_iras_binary, summary=gen_summary, methods=gen_methods, parity=gen_parity)
REGION = re.compile(r"(?ms)^(?P<ind> *)<!-- gen:(?P<name>[\w-]+)(?: (?P<key>\S+))? -->\n.*?^(?P=ind)<!-- /gen:(?P=name) -->\n")


def build(text):
    def sub(m):
        ind, name, key = m.group("ind"), m.group("name"), m.group("key")
        body = GEN[name](key)
        head = f"{ind}<!-- gen:{name}{' ' + key if key else ''} -->\n"
        tail = f"{ind}<!-- /gen:{name} -->\n"
        return head + "\n" + (indent(body, ind) + "\n" if body.strip() else "") + tail
    return REGION.sub(sub, text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    old = PAGE.read_text()
    new = build(old)
    if args.check:
        return 0 if new == old else 1
    PAGE.write_text(new)
    print("page updated" if new != old else "page already current")
    return 0


if __name__ == "__main__":
    sys.exit(main())
