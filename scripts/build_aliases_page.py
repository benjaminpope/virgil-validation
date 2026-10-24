"""The period-alias pages, one per system: docs/orbits/aliases/<system>.md, from
scripts/templates/aliases_page.md.tmpl and the per-system bands.json / meta.json in docs/assets/orbit_aliases/<system>/.
Each gallery tile on docs/orbits/index.md links to its system's page; scripts/build_orbits_page.py writes them.

Systems with no bands.json get a clearly marked placeholder. Run by
scripts/pull_back_orbit_results.py after it copies the results; safe to run on its own.

    python scripts/build_aliases_page.py [--assets DIR] [--out FILE]
"""

import json
import pathlib
import string
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from crosscheck.skyplane import GOOD_CHI2_N, SAME_FRACTION, SAME_SIGMA, VERDICT_RULE_VERSION, verdict_from_stats  # noqa: E402

TEMPLATE = REPO / "scripts" / "templates" / "aliases_page.md.tmpl"
ASSETS = REPO / "docs" / "assets" / "orbit_aliases"
MAX_ROWS = 12
PLOTS = [("sky.png", "Sky plane: per-epoch positions (red), posterior orbits in the winning band, reference orbit"),
         ("corner.png", "Posterior of the winning band"),
         ("residuals.png", "Closure-phase residuals at the best orbit, per epoch")]

SKIPPED = [
    ("kappa Vel, NN Del (Track B)", "span shorter than the period, so only N = 0 and 1 exist and nothing is tested"),
    ("9 Sgr, delta Vel, HD 136164 (NACO SAM)", "one epoch, or four nights in one run: not an orbit"),
    ("HR 4049 (PIONIER)", "no period; the analysis lives in notebooks, with no standalone loader"),
    ("9 Sgr, HD 152314, HD 168137, KQ Vel, CPD-71 172, TYC 1703-394-1 (Track B)", "one or two epochs"),
    ("zeta Boo, eta Oph (Track B)", "no tabulated reference period to set the prior window"),
    ("del Cir, TZ For (Track B)", "mixed GRAVITY and PIONIER epochs; a conflict between the manifest and the files"),
]


def fmt(x, spec):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, spec)


def current_verdict(row):
    """The verdict of a stored sky-plane row under the CURRENT rule (fraction AND sigma), recomputed from its
    statistics: the stored ``verdict`` string comes from whatever rule the run used (older runs: fraction only)."""
    return verdict_from_stats(row["frac_sep"], row.get("offset_over_spread"), row.get("mirror") == "flip",
                              row.get("flipped", False), row.get("labels_free", False), row.get("chi2_n"))


def finite(xs):
    return [x for x in (xs or []) if x is not None and np.isfinite(x)]


def flag_text(b):
    flags = b.get("flags")
    if flags is None:  # an older bands.json: the single flagged boolean
        return "flagged" if b.get("flagged") else ""
    return ", ".join(flags)


def table(bands):
    rows = ["| N | P (d) | χ²/N (raw) | error scales | log Z | log Z (IS) | p | ESS | flags |",
            "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |"]
    for b in bands[:MAX_ROWS]:
        sc = finite(b.get("scales"))
        s = f"{min(sc):.2f}–{max(sc):.2f}" if sc else "-"
        rows.append(f"| {b['n']} | {fmt(b.get('period'), '.4f')} | {fmt(b.get('chi2_red'), '.2f')} | {s} | "
                    f"{fmt(b.get('log_z'), '.2f')} | {fmt(b.get('log_z_is'), '.2f')} | {fmt(b.get('p'), '.3f')} | "
                    f"{fmt(b.get('ess'), '.0f')} | {flag_text(b)} |")
    if len(bands) > MAX_ROWS:
        rows.append(f"\n*{len(bands) - MAX_ROWS} further bands, each with p < {fmt(bands[MAX_ROWS - 1].get('p'), '.3g')}, are in `bands.json`.*")
    return "\n".join(rows)


def sky_table(sp):
    """The rows of a stored ``sky_plane`` dict as a table, verdicts recomputed under the current rule."""
    rows = [
        "| band | N | P (d) | rms | max | rms / sep | rms / spread | mirror | flipped | verdict |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |"
    ]
    for k, v in sp.items():
        rows.append(
            f"| {'winner' if k == 'top' else 'reference'}{' (warm-started)' if v.get('warm_started') else ''} | {v['band']} | {v['period']:.4f} | {v['rms']:.2f} | "
            f"{v['max']:.2f} | {v['frac_sep']:.0%} | {fmt(v['offset_over_spread'], '.1f')} | {v['mirror']} | "
            f"{'yes' if v.get('flipped') else 'no'} | {current_verdict(v)} |"
        )
    return "\n".join(rows) + "\n"


def chi2_warning(meta):
    chk = meta.get("chi2_check")
    if chk and not chk["agrees"]:
        return [f"!!! warning\n    Raw χ²/N re-summed here from virgil's own residuals ({chk['ours']:.2f}) differs from the band table ({chk['virgil']:.2f}).\n"]
    return []


def section(name, label, instrument, sysdir, rel):
    head = ""
    bj = sysdir / "bands.json"
    if not bj.exists():
        return head + (f"!!! warning \"Placeholder\"\n    No results yet for `{name}` ({instrument}). "
                       f"Expected: band table, `sky.png`, `corner.png`, `residuals.png`.\n")
    bands = json.load(open(bj))["bands"]
    meta = json.load(open(sysdir / "meta.json")) if (sysdir / "meta.json").exists() else {}
    out = [head]
    n_ep = len(meta.get("epochs", []))
    out.append(f"{instrument}, {n_ep} epochs, period range {meta.get('p_range', '?')} d; {len(bands)} alias bands.\n")
    v = meta.get("virgil") or {}
    out.append(f"virgil {v.get('version', '?')}, commit `{v.get('commit_id') or 'unknown'}`.\n")
    ref = meta.get("reference_band")
    if ref:
        out.append(f"Reference period {ref['period']:.4f} d lies in band N = {ref['n']} (p = {fmt(ref.get('p'), '.3f')}); "
                   f"{'it is the winner' if ref.get('is_winner') else 'it is not the winner'}.\n")
    if not is_cold(meta):
        out.append('!!! warning "Not a cold start"\n    This run is not a cold start (`args.no_init` is false, or an initial orbit was used): '
                   "agreement with the reference is not independent.\n")
        out.append(cold_line(name, sysdir, rel))
    out.append(table(bands) + "\n")
    sp = meta.get("sky_plane")
    if sp:
        out.append(
            "Orbits are compared on the sky, at the epochs, not by their elements (conventions and aliases give one sky orbit "
            "many element sets). Offsets are in mas; **mirror** = flip means the reference's point reflection "
            "matches, and **flipped** says whether the fit swapped the two stars' labels (flux ratio > 1): a mirror without a swap "
            "(and a fit that did not report its labels free) is UNEXPLAINED-FLIP, a sign error rather than agreement. A fit that is "
            f"good on its own data (raw chi2/N at most {GOOD_CHI2_N:g}) but far from the reference is REFERENCE-DISAGREES. Verdict rule "
            f"version {VERDICT_RULE_VERSION}. The verdict is SAME-SKY-ORBIT only if "
            f"the rms offset is under {SAME_FRACTION:.0%} of the separation and, with posterior samples, under {SAME_SIGMA:g} "
            "posterior spreads. Rows marked *warm-started* are in the band whose fit began at the reference orbit, so agreement "
            "there is not independent.\n"
        )
        out.append(sky_table(sp))
    fs = meta.get("fit_summary")
    if fs:
        amps = [a for a in fs.get("vis_amp") or [] if a is not None]
        bits = [f"flux ratio {fmt(fs.get('flux'), '.3f')}" + (" (stars' labels swapped)" if fs.get("flipped") else "")]
        if fs.get("diam") is not None:
            bits.append(f"primary diameter {fs['diam']:.3f} mas (fitted)")
        elif meta.get("primary_diam"):
            bits.append(f"primary diameter {meta['primary_diam']} mas (fixed)")
        if amps:
            bits.append(f"per-epoch V² amplitudes {min(amps):.2f}–{max(amps):.2f}")
        out.append("Winning band: " + "; ".join(bits) + ".\n")
    ic = meta.get("init_check")
    if meta.get("init") and ic:
        out.append(
            f"The reference band was warm-started at the reference orbit; the best orbit is {ic['offset_rms_mas']:.2f} mas rms "
            f"(max {ic['offset_max_mas']:.2f}) from that start, with χ² {ic['chi2_init']:.1f} at the start and "
            f"{ic['chi2_best']:.1f} at the best orbit (gain {ic['chi2_gain']:.1f}; V² amplitudes unscaled). A small offset "
            "with no χ² gain means the fit did not move.\n"
        )
    out += chi2_warning(meta)
    for png, cap in PLOTS:
        if (sysdir / png).exists():
            out.append(f"![{cap}]({rel}/{name}/{png})\n")
    return "\n".join(out)


def is_cold(meta):
    """A run is cold if its own record says so: ``args.no_init`` is true and no ``init`` orbit was used. Never from
    a directory or tag name; a record with no ``args`` is not known to be cold."""
    return (meta.get("args") or {}).get("no_init") is True and meta.get("init") is None


def cold_line(name, sysdir, rel, ref_n=None):
    """One line citing the stored cold run ``<name>-cold`` (winner and the reference band's rank), or "" if none."""
    cdir = sysdir.parent / f"{name}-cold"
    if not (cdir / "bands.json").exists() or not (cdir / "meta.json").exists():
        return ""
    meta = json.load(open(cdir / "meta.json"))
    if not is_cold(meta):
        return ""
    bands = json.load(open(cdir / "bands.json"))["bands"]
    ref = (json.load(open(sysdir / "meta.json")).get("reference_band") or {}) if (sysdir / "meta.json").exists() else {}
    n_ref = ref_n if ref_n is not None else ref.get("n")
    rank = next((i + 1 for i, b in enumerate(bands) if b["n"] == n_ref), None)
    tail = f"; the reference band N = {n_ref} ranks {rank} of {len(bands)}" if rank else ""
    return (f"A stored cold run (`--no-init`, [`{name}-cold`]({rel}/{name}-cold/bands.json)) ranks N = {bands[0]['n']} first{tail}. "
            "It is the independent result; the table below is not.\n")


def verdict_tally(assets, names):
    """Winning-band verdicts of every system with a stored sky-plane row, under the current rule. ``cold`` counts
    those whose run was a cold start; the rest were warm or reference-informed."""
    verdicts = {}
    cold = 0
    for n in names:
        mj = pathlib.Path(assets) / n / "meta.json"
        meta = json.load(open(mj)) if mj.exists() else {}
        top = (meta.get("sky_plane") or {}).get("top")
        if top:
            verdicts[n] = current_verdict(top)
            cold += is_cold(meta)
    return dict(total=len(verdicts), verdicts=verdicts, cold=cold,
                **{"pass": sum(v == "SAME-SKY-ORBIT" for v in verdicts.values())})


PAGES = "aliases"  # the system pages, under docs/orbits/
# Gallery tiles (scripts/orbit_gallery.py slugs) for each alias system; by default the name with "_" as "-".
TILES = {"gl229": ("gl229b", "gl229b-blind")}


def tiles(name):
    return TILES.get(name, (name.replace("_", "-"),))


def tally_sentence(assets, names):
    tally = verdict_tally(assets, names)
    return (f"Under the current sky-plane rule (rms offset under {SAME_FRACTION:.0%} of the separation and under "
            f"{SAME_SIGMA:g} posterior spreads), {tally['pass']} of {tally['total']} systems have a winning-band orbit on the "
            f"reference's sky orbit (warm or reference-informed starts; {tally['cold']} of them cold), "
            "recomputed from the stored statistics. This is not a blind result.")


def skipped():
    return "\n".join(f"- {a}: {b}." for a, b in SKIPPED)


def render(assets=ASSETS, template=TEMPLATE, rel="../../assets/orbit_aliases"):
    """{system name: its page's markdown}."""
    names = _registry()
    out = {}
    for n, v in names.items():
        body = section(n, v["label"], v["instrument"], pathlib.Path(assets) / n, rel)
        out[n] = string.Template(template.read_text()).substitute(TITLE=v["label"], SYSTEM=body)
    return out


def _registry():
    """System names, labels and instruments, from reanalyse_orbits.py without importing virgil."""
    import reanalyse_orbits as R

    return {k: dict(label=v.label, instrument=v.instrument) for k, v in R.SYSTEMS.items()}


def main(argv=None):
    import build_orbits_page

    build_orbits_page.main(argv)


if __name__ == "__main__":
    main()
