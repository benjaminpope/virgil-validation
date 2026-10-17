"""Regenerate the numbers on docs/imaging/postagb.md from the saved compare/image JSON, so they cannot drift.

    python scripts/postagb_page.py [--check]

For each tab this rewrites, in place: the comparison-beam wording in the figure caption, the verdict, the image-metrics table, and the
beam, hole, regulariser and chi2 sentences of "Data and method"; it also rewrites the summary table. The prose of the verdicts is written
here (``VERDICTS``), with every number taken from ``docs/imaging/postagb/<target>_compare.json`` and ``_image.json``. The IRAS 08544
parameter tables are hand-maintained next to their parametric fit (the fit's own numbers are in ``IRAS08544_parametric.json``).
``--check`` exits 1 if the page is out of date instead of writing it.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/imaging"
PAGE = DOCS / "postagb.md"
OUT = DOCS / "postagb"
TABS = {  # tab title -> target key
    "IW Car": "IW_Car", "HR 4049": "HR4049", "HD 108015": "HD108015", "AI Sco": "AI_Sco", "EN TrA": "EN_TrA",
    "HD 95767": "HD95767", "IRAS 15469-5311": "IRAS15469-5311", "PS Gem": "PS_Gem", "IRAS 08544-4431": "IRAS08544",
}
SEC_TARGETS = {"IW_Car", "HR4049", "AI_Sco", "EN_TrA", "HD95767", "IRAS08544"}  # a point-source secondary in the scene


def get(key, what, mem=False):
    p = OUT / f"{key}_{'mem_' if mem else ''}{what}.json"
    return json.load(open(p)) if p.exists() else None


def ctl(c):
    k = c["ncc_controls"]
    return (f"E-W mirror {k['mirrored_East-West']:.2f}, 180° rotation {k['rotated_180']:.2f}, "
            f"azimuthal average {k['azimuthal_average_of_published']:.2f}")


def reg_text(i):
    if i["regulariser"] == "GaussianField":
        return f"a Gaussian-process prior (σ = {i['gp_sigma']:g}, ℓ = {i['gp_length_mas']:.2f} mas)"
    return "MaxEntropy at the L-curve corner"


def disc(c, who):
    return c[who]["disc"]


# One-line verdicts: (admonition kind, title, body). Numbers come from the JSON through ``c`` (compare), ``i`` (image), ``m`` (the
# MaxEntropy comparison, where there is one).
def verdict(key, c, i, m):
    n, b = c["ncc_zero_shift"], c["ncc_best_shift"]
    k = c["ncc_controls"]
    base = (f"At the λ/2B_max beam ({c['beam']['major_mas']:.2f} mas) the images correlate at {n:.2f} ({b:.2f} after the best sub-beam shift), "
            f"against {k['mirrored_East-West']:.2f} for the East-West mirror of the published image, {k['rotated_180']:.2f} for its 180° "
            f"rotation and {k['azimuthal_average_of_published']:.2f} for its azimuthal average (the same flux at the same radii with no azimuthal structure).")
    if key == "IW_Car":
        o, p = c["ours"], c["published"]
        return ("warning", "Verdict: the inner arcs agree, the outer arc does not",
                f"{base} The inner arcs match (flux-weighted radius {o['inner']['mean_radius_mas']:.1f} against {p['inner']['mean_radius_mas']:.1f} mas, major-axis PA "
                f"{o['inner']['major_pa_deg']:.0f}° against {p['inner']['major_pa_deg']:.0f}°). The outer arc sits further out and is oriented differently in virgil "
                f"(radius {o['outer']['mean_radius_mas']:.1f} against {p['outer']['mean_radius_mas']:.1f} mas, PA {o['outer']['major_pa_deg']:.0f}° against "
                f"{p['outer']['major_pa_deg']:.0f}°). De Prins et al. themselves note that the arcs' azimuthal profile is not well constrained (Appendix D, SQUEEZE).")
    if key == "IRAS08544":
        o, q = c["ours"]["outer"], c["published"]["outer"]
        base = base.replace(f"At the λ/2B_max beam ({c['beam']['major_mas']:.2f} mas)", f"At the beam drawn on the published figure ({c['beam']['major_mas']:.1f} × {c['beam']['minor_mas']:.1f} mas)")
        return ("success", "Verdict: the two reconstructions agree, with the binary core modelled",
                f"{base} With the binary subtracted in both, there is no residual central emission (inside 3.5 mas: {100 * c['ours']['inner']['flux_fraction']:.1f} % of "
                f"the flux against {100 * c['published']['inner']['flux_fraction']:.1f} %), and the ring matches: flux-weighted radius {o['mean_radius_mas']:.2f} against "
                f"{q['mean_radius_mas']:.2f} mas, flux centroid {o['centroid_pa_deg']:.0f}° against {q['centroid_pa_deg']:.0f}°. The secondary's parameters are in the table below.")
    d, p = disc(c, "ours"), disc(c, "published")
    r = f"flux-weighted radius {d['mean_radius_mas']:.1f} against {p['mean_radius_mas']:.1f} mas"
    if key == "HR4049":
        return ("failure", "Verdict: virgil's disc is still too compact; the Gaussian-process prior does not fix it",
                f"{base} virgil's disc is smaller than the published one ({r}); MaxEntropy gave {disc(m, 'ours')['mean_radius_mas']:.1f} mas, so the Gaussian-process "
                f"prior (σ = {i['gp_sigma']:g}, ℓ = {i['gp_length_mas']:.1f} mas, chosen by the evidence) does not close the gap, and the correlation "
                f"({n:.2f}, against {m['ncc_zero_shift']:.2f} for MaxEntropy) is barely above its azimuthal-average control, so it says little. "
                "The data prefer the published orientation (see Fit quality), but virgil's image does not reproduce the published extent.")
    if key == "PS_Gem":
        return ("failure", "Verdict: still poor: the GP fixes the size, not the structure",
                f"{base} With the Gaussian-process prior virgil's disc has the right size ({r}; MaxEntropy gave {disc(m, 'ours')['mean_radius_mas']:.1f} mas, "
                f"NCC {m['ncc_zero_shift']:.2f} against {n:.2f} now), but the structure is a scatter of clumps and the correlation is only just above its controls. "
                "ORGANIC's brightest feature is a clump at the star, inside the window virgil blanks around the star (radius "
                f"{i['hole_mas']:.2f} mas); that is one reason the two cannot agree there. Do not read anything into the clumps.")
    if key == "AI_Sco":
        return ("warning", "Verdict: the same orientation, but virgil's disc is larger",
                f"{base} The flux centroid points the same way ({d['centroid_pa_deg']:.0f}° against {p['centroid_pa_deg']:.0f}° East of North), but virgil's "
                f"disc is larger ({r}).")
    if key == "EN_TrA":
        return ("warning", "Verdict: the same size, a different orientation",
                f"{base} The size agrees ({r}), but the major-axis PA differs ({d['major_pa_deg']:.0f}° against {p['major_pa_deg']:.0f}°) and the flux "
                f"centroid is close to the star in both ({d['centroid_offset_mas']:.2f} and {p['centroid_offset_mas']:.2f} mas), so its direction is not meaningful.")
    if key == "IRAS15469-5311":
        return ("success", "Verdict: the two reconstructions agree in radius and centroid, less in the major axis",
                f"{base} The radius ({r}) and the flux centroid ({d['centroid_pa_deg']:.0f}° against {p['centroid_pa_deg']:.0f}°) agree; the major-axis "
                f"PA differs ({d['major_pa_deg']:.0f}° against {p['major_pa_deg']:.0f}°, the discs being nearly round).")
    kind = "success"
    return (kind, "Verdict: the two reconstructions agree", f"{base} Flux-weighted radius {d['mean_radius_mas']:.1f} against {p['mean_radius_mas']:.1f} mas, "
            f"flux centroid {d['centroid_pa_deg']:.0f}° against {p['centroid_pa_deg']:.0f}° East of North.")


def admonition(kind, title, body):
    return f'    !!! {kind} "{title}"\n' + "\n".join("        " + ln for ln in wrap(body)) + "\n"


def wrap(text, width=118):
    lines, cur = [], ""
    for w in text.split(" "):
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + [cur]


def metrics_table(key, c):
    o, p = c["ours"], c["published"]
    head = (f"    | Metric | ORGANIC | virgil |\n    | --- | --- | --- |\n    | Normalised cross-correlation (controls: {ctl(c)}) | 1 | "
            f"{c['ncc_zero_shift']:.2f} ({c['ncc_best_shift']:.2f} after the best sub-beam shift) |\n")
    if key == "IW_Car":
        rows = []
        for reg, name in (("inner", "Inner"), ("outer", "Outer")):
            rows.append((f"{name} flux fraction (%)", f"{100 * p[reg]['flux_fraction']:.1f}", f"{100 * o[reg]['flux_fraction']:.1f}"))
            rows.append((f"{name} flux-weighted radius (mas)", f"{p[reg]['mean_radius_mas']:.2f}", f"{o[reg]['mean_radius_mas']:.2f}"))
            rows.append((f"{name} major-axis PA (°), axis ratio", f"{p[reg]['major_pa_deg']:.0f}, {p[reg]['axis_ratio']:.2f}", f"{o[reg]['major_pa_deg']:.0f}, {o[reg]['axis_ratio']:.2f}"))
            rows.append((f"{name} flux centroid (° East of North), offset (mas)", f"{p[reg]['centroid_pa_deg']:.0f}, {p[reg]['centroid_offset_mas']:.2f}",
                         f"{o[reg]['centroid_pa_deg']:.0f}, {o[reg]['centroid_offset_mas']:.2f}"))
    else:
        d, q = o["disc"], p["disc"]
        rows = [("Flux fraction in the image field", f"{100 * q['flux_fraction']:.0f} %", f"{100 * d['flux_fraction']:.0f} %"),
                ("Flux-weighted radius (mas)", f"{q['mean_radius_mas']:.2f}", f"{d['mean_radius_mas']:.2f}"),
                ("Major-axis PA (°), axis ratio", f"{q['major_pa_deg']:.0f}, {q['axis_ratio']:.2f}", f"{d['major_pa_deg']:.0f}, {d['axis_ratio']:.2f}"),
                ("Asymmetry angle, flux centroid (° East of North), offset (mas)", f"{q['centroid_pa_deg']:.0f}, {q['centroid_offset_mas']:.2f}",
                 f"{d['centroid_pa_deg']:.0f}, {d['centroid_offset_mas']:.2f}")]
    return head + "".join(f"    | {a} | {b} | {c_} |\n" for a, b, c_ in rows)


def patch_tab(text, key):
    c, i, m = get(key, "compare"), get(key, "image"), get(key, "compare", mem=True) if key in ("HR4049", "PS_Gem") else None
    bm = c["beam"]["major_mas"]
    if key == "IRAS08544":
        return patch_iras(text, c, i)
    # caption
    text = re.sub(r"convolved to the PIONIER\s+beam \(grey ellipse\)", f"convolved to a circular λ_min/2B_max beam of {bm:.2f} mas FWHM (grey disc)", text)
    text = re.sub(r"virgil, MaxEntropy with the weight at the\s+L-curve corner", f"virgil, {reg_text(i)}", text)
    # verdict
    text = re.sub(r'    !!! \w+ "Verdict[^\n]*\n(?:        [^\n]*\n)+', lambda _: admonition(*verdict(key, c, i, m)), text, count=1)
    # metrics
    text = re.sub(r"    \| Metric \| ORGANIC \| virgil \|\n(?:    \|[^\n]*\n)+", lambda _: metrics_table(key, c), text, count=1)
    text = re.sub(r"convolved to the beam;", f"convolved to the λ_min/2B_max beam ({bm:.2f} mas FWHM);", text)
    # data paragraph: the comparison beam and the hole
    sec = " and around the secondary" if key in SEC_TARGETS else ""
    hole = (f" The comparison beam is circular, FWHM λ_min/(2B_max) = {bm:.2f} mas, for both images. The environment image has a window of radius "
            f"{i['hole_mas']:.2f} mas (0.5 of the data beam's minor axis; {i['n_blanked']} pixels) blanked around the primary{sec}, so that its pixels cannot "
            "trade flux with the stellar point sources.")
    text = re.sub(r"(The PIONIER beam is [\d.]+ × [\d.]+ mas at PA \d+°\.)(?: The comparison beam.*?point sources\.)?", lambda mo: mo.group(1) + hole, text, count=1)
    # model paragraph
    if i["regulariser"] == "GaussianField":
        mem = f" MaxEntropy comparison (the earlier run): radius {disc(m, 'ours')['mean_radius_mas']:.1f} mas, NCC {m['ncc_zero_shift']:.2f}." if m else ""
        new = (f"regulariser a `GaussianField` prior on the log-brightness about the Gaussian start image (σ and ℓ chosen on a grid of σ = 1, 2, 4 and ℓ = 0.25 to 2 data "
               f"beams by the Laplace evidence, after MacKay's error rescaling, s = {i['error_scale']:.2f}: σ = {i['gp_sigma']:g}, ℓ = {i['gp_length_mas']:.2f} mas, "
               f"log evidence {i['log_evidence']:.0f}).{mem}")
        text = re.sub(r"regulariser `MaxEntropy`; weight from `l_curve` \(corner at [^)]*\);? ?(?:start from `starting_image`\.)?", lambda _: new + " Start from `starting_image`.", text, count=1)
    else:
        text = re.sub(r"\(corner at [\d.e-]+\)", f"(corner at {i['corner_weight']:.3g})", text)
        text = re.sub(r"taking the corner \(weight [\d.e-]+\)", f"taking the corner (weight {i['corner_weight']:.3g})", text)
    # fit quality
    text = re.sub(r"(the virgil image has χ² per point )[\d.]+", lambda mo: f"{mo.group(1)}{i['chi2_per_point']:.2f}", text, count=1)
    text = re.sub(r"(published SPARCO parameters unchanged, )[\d.]+( \()[\d ]+( data points\))",
                  lambda mo: f"{mo.group(1)}{c['chi2_per_point_published_on_our_data']:.2f}{mo.group(2)}{c['n_points']}{mo.group(3)}", text, count=1)
    text = text.replace("This target was computed on OzSTAR (job postagb_imaging, library snapshot ea943fa + c80aed7); the parametric fit is a local multi-start rerun.",
                        "The images were computed on the laptop for this page; the parametric fit is a local multi-start rerun.")
    text = text.replace("This target was computed on the laptop; the parametric fit is a local multi-start rerun.",
                        "The images were computed on the laptop for this page; the parametric fit is a local multi-start rerun.")
    if key in ("PS_Gem", "HR4049"):
        text = re.sub(r"and is over-fitted at this weight, while ORGANIC is more strongly regularised\.",
                      "and the Gaussian-process prior is the only regularisation (the χ² is on the errors as given, not rescaled).", text)
        text = re.sub(r" \(a difference of [^)]*\)", "", text)
        text = re.sub(r"\(a difference of 440 and 701 in χ², so the data clearly prefer the published orientation; it is virgil's image[^)]*\)", "(so the data prefer the published orientation)", text)
    return text


def patch_iras(text, c, i):
    """Numbers that follow the image fit (the prose and the parameter tables of this tab are edited by hand)."""
    kind, title, body = verdict("IRAS08544", c, i, None)
    text = re.sub(r'    !!! \w+ "Verdict[^\n]*\n(?:        [^\n]*\n)+', lambda _: admonition(kind, title, body), text, count=1)
    text = re.sub(r"    \| Metric \| Hillen\+2016 \| virgil \|\n(?:    \|[^\n]*\n)+", lambda _: iras_metrics(c), text, count=1)
    text = re.sub(r"(    <!-- binary-table -->\n)(?:    \|[^\n]*\n)*(    <!-- /binary-table -->)", lambda mo: mo.group(1) + binary_table(i) + mo.group(2), text, count=1)
    text = re.sub(r"The comparison beam is circular.*?point sources\.", "", text)  # this tab keeps the drawn beam
    text = re.sub(r"(virgil's own dirty beam is [\d.]+ × [\d.]+ mas at PA \d+°)", lambda mo: mo.group(1), text)
    text = re.sub(r"(window of radius )[\d.]+( mas \(0\.5 of the data beam's minor axis; )\d+( pixels\))", lambda mo: f"{mo.group(1)}{i['hole_mas']:.2f}{mo.group(2)}{i['n_blanked']}{mo.group(3)}", text)
    text = re.sub(r"(the virgil image has χ² per point )[\d.]+( on )[\d ]+( residuals)", lambda mo: f"{mo.group(1)}{i['chi2_per_point']:.2f}{mo.group(2)}{i['n_points']}{mo.group(3)}", text)
    return text


def iras_metrics(c):
    o, p = c["ours"], c["published"]
    k = c["ncc_controls"]
    return ("    | Metric | Hillen+2016 | virgil |\n    | --- | --- | --- |\n"
            f"    | Normalised cross-correlation (controls: {ctl(c)}) | 1 | {c['ncc_zero_shift']:.2f} ({c['ncc_best_shift']:.2f} after the best sub-beam shift) |\n"
            f"    | Inside 3.5 mas / ring (3.5 to 16 mas) flux fraction | {100 * p['inner']['flux_fraction']:.1f} % / {100 * p['outer']['flux_fraction']:.1f} % | "
            f"{100 * o['inner']['flux_fraction']:.1f} % / {100 * o['outer']['flux_fraction']:.1f} % |\n"
            f"    | Ring flux-weighted radius (mas) | {p['outer']['mean_radius_mas']:.2f} | {o['outer']['mean_radius_mas']:.2f} |\n"
            f"    | Ring major-axis PA (°) | {p['outer']['major_pa_deg']:.0f} | {o['outer']['major_pa_deg']:.0f} (the ring is nearly circular, axis ratio "
            f"{p['outer']['axis_ratio']:.2f} and {o['outer']['axis_ratio']:.2f}, so this is ill-defined) |\n"
            f"    | Asymmetry angle, ring flux centroid (° East of North), offset (mas) | {p['outer']['centroid_pa_deg']:.0f}, {p['outer']['centroid_offset_mas']:.2f} | "
            f"{o['outer']['centroid_pa_deg']:.0f}, {o['outer']['centroid_offset_mas']:.2f} |\n")


HILLEN = dict(f_prim=(59.7, 0.6), f_sec=(3.9, 0.7), sep=(0.81, 0.05), pa=(236.0, 3.0), env_index=(0.42, None))  # Table 1; the PA is 56 + 180


def binary_table(i):
    def row(label, key, mine, fmt="{:.1f}", note=""):
        v, e = HILLEN[key]
        d = mine - v
        sig = f" ({d / e:+.1f}σ)" if e else ""
        return f"    | {label} | {fmt.format(v)}{f' ± {e:g}' if e else ''} | {fmt.format(mine)} | {d:+.2g}{sig}{note} |\n"
    return (row("Primary flux fraction, H (%)", "f_prim", i["f_prim_pct"]) + row("Secondary flux fraction, H (%)", "f_sec", i["f_sec_pct"])
            + row("Secondary separation (mas)", "sep", i["sec_sep_mas"], "{:.2f}") + row("Secondary position angle from the primary (°)", "pa", i["sec_pa_deg"], "{:.0f}")
            + f"    | Secondary spectral index (power law at 1.65 µm) | −2.5 (−3.1 to −0.6; from 4000 ± 2000 K) | {i['sec_index']:.1f} | at the prior's lower bound (−5) |\n"
            + row("Environment spectral index", "env_index", i["env_index"], "{:.2f}"))


def summary_table(rows):
    lines = ["| Target | Published method | virgil method | virgil verdict | NCC (best control) | Comparison beam (mas) |", "| --- | --- | --- | --- | --- | --- |"]
    for t, key in TABS.items():
        c, i = get(key, "compare"), get(key, "image")
        lines.append(f"| [{t}](#{anchor(t)}) | {PUBLISHED[key]} | {'GP prior' if i['regulariser'] == 'GaussianField' else 'MaxEntropy'} | {rows[key]} | "
                     f"{c['ncc_zero_shift']:.2f} ({max(c['ncc_controls'].values()):.2f}) | {c['beam']['major_mas']:.2f}{' (Hillen: drawn 2.5 × 1.8 ellipse)' if key == 'IRAS08544' else ''} |")
    return "\n".join(lines) + "\n"


def anchor(t):
    return t.lower().replace(" ", "-")


PUBLISHED = {k: "SPARCO + ORGANIC" for k in TABS.values()}
PUBLISHED["IW_Car"] = "SPARCO + ORGANIC (SQUEEZE check)"
PUBLISHED["IRAS08544"] = "SPARCO, binary subtracted (Hillen+2016)"


def one_line(key, c, i, m):
    return verdict(key, c, i, m)[1].removeprefix("Verdict: ")


def build(text):
    parts = re.split(r'(?m)^(?==== ")', text)
    out = [parts[0]]
    one = {}
    for part in parts[1:]:
        t = re.match(r'=== "([^"]+)"', part).group(1)
        if t not in TABS:  # the last block runs on to the next section
            out.append(part)
            continue
        key = TABS[t]
        tail = ""
        # the final tab runs on into the page's later sections: patch only up to the first top-level heading
        m_ = re.search(r"(?m)^## ", part)
        if m_:
            part, tail = part[:m_.start()], part[m_.start():]
        part = patch_tab(part, key)
        out.append(part + tail)
        c, i = get(key, "compare"), get(key, "image")
        one[key] = one_line(key, c, i, get(key, "compare", mem=True) if key in ("HR4049", "PS_Gem") else None)
    text = "".join(out)
    text = re.sub(r"\| Target \| Published method \| virgil verdict \| Image cross-correlation \|\n(?:\|[^\n]*\n)+|"
                  r"\| Target \| Published method \| virgil method \| virgil verdict \| NCC \(best control\) \| Comparison beam \(mas\) \|\n(?:\|[^\n]*\n)+",
                  lambda _: summary_table(one), text, count=1)
    return text


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
