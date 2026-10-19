"""Generate docs/orbits/aliases.md from scripts/templates/aliases_page.md.tmpl and the
per-system bands.json / meta.json in docs/assets/orbit_aliases/<system>/.

Systems with no bands.json get a clearly marked placeholder. Run by
scripts/pull_back_orbit_results.py after it copies the results; safe to run on its own.

    python scripts/build_aliases_page.py [--assets DIR] [--out FILE]
"""

import argparse
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
OUT = REPO / "docs" / "orbits" / "aliases.md"
MAX_ROWS = 12
PLOTS = [("sky.png", "Sky plane: per-epoch positions (red), posterior orbits in the winning band, reference orbit"),
         ("corner.png", "Posterior of the winning band"),
         ("residuals.png", "Closure-phase residuals at the best orbit, per epoch")]

SKIPPED = [
    ("kappa Vel, NN Del (Track B)", "span shorter than the period, so only N = 0 and 1 exist and nothing is tested"),
    ("Apep (GRAVITY, 6 nights)", "calibration is done by a script outside this repository and the calibrated OIFITS are not on /fred"),
    ("Apep, 9 Sgr, delta Vel, HD 136164 (NACO SAM)", "one epoch, or four nights in one run: not an orbit"),
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


def section(name, label, instrument, sysdir, rel):
    head = f"## {label}\n\n"
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
        out.append(
            "| band | N | P (d) | rms | max | rms / sep | rms / spread | mirror | flipped | verdict |\n"
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |"
        )
        for k, v in sp.items():
            out.append(
                f"| {'winner' if k == 'top' else 'reference'}{' (warm-started)' if v.get('warm_started') else ''} | {v['band']} | {v['period']:.4f} | {v['rms']:.2f} | "
                f"{v['max']:.2f} | {v['frac_sep']:.0%} | {fmt(v['offset_over_spread'], '.1f')} | {v['mirror']} | "
                f"{'yes' if v.get('flipped') else 'no'} | {current_verdict(v)} |"
            )
        out.append("")
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
    chk = meta.get("chi2_check")
    if chk and not chk["agrees"]:
        out.append(f"!!! warning\n    Raw χ²/N re-summed here from virgil's own residuals ({chk['ours']:.2f}) differs from the band table ({chk['virgil']:.2f}).\n")
    for png, cap in PLOTS:
        if (sysdir / png).exists():
            out.append(f"![{cap}]({rel}/{name}/{png})\n")
    return "\n".join(out)


GATE_N, GATE_NATS = 34, 10.0  # virgil 0.4.0 release gate (CONTRIBUTING "Releasing", item 4)


def _rank34(bands):
    bands = sorted(bands, key=lambda b: -b["log_z"])
    n34 = next((b for b in bands if b["n"] == GATE_N), None)
    return bands, n34, ([b["n"] for b in bands].index(GATE_N) + 1 if n34 else None)


def _pipeline_gate(assets, rel):
    """The blind gate: virgil's OrbitPipeline on Gl 229 with no settings, on data prepared on the virgil-vlti side."""
    d = pathlib.Path(assets) / "gl229-pipeline"
    if not (d / "meta.json").exists():
        return []
    m = json.load(open(d / "meta.json"))
    g, s, prov = m["gate"], m["summary"], m.get("provenance") or {}
    prep = m.get("preparation") or {}
    win = s["modes"]["winner"]
    el = win.get("elements") or {}
    temper = (s.get("search") or {}).get("tempering") or {}
    out = [
        f"**{g['verdict']}.** `OrbitPipeline` on Gl 229 with no settings: no period window, start or published orbit "
        f"(the window, {fmt(g['p_range'][0], '.1f')} to {fmt(g['p_range'][1], '.0f')} d, comes from the data, with {g['n_bands']} bands). "
        f"It must rank band N = {GATE_N} first with Δlog Z > {GATE_NATS:g} over every other band.\n",
        f"Winner: N = {g['winner_n']}, P = {fmt(el.get('period'), '.3f')} d, e = {fmt(el.get('ecc'), '.3f')}, flux ratio {fmt(el.get('flux'), '.3f')}. "
        f"Δlog Z = {fmt(g['delta_log_z'], '.1f')} over the runner-up (N = {g['runner_up_n']}); the evidences are tempered "
        f"(effective-channel factor f = {fmt(temper.get('f'), '.0f')}), so this margin is on the tempered likelihood.\n",
        f"Information, not pass/fail: raw χ²/N {fmt(g['chi2_per_n_raw'], '.1f')} on the quoted errors (the pipeline's χ² check fails: "
        f"the errors are far too small, a noise-model matter for virgil-vlti, not the period search); "
        f"{(prep.get('n_files_kept'))} of {prep.get('n_files')} files, {len(prep.get('nights') or [])} nights. "
        f"virgil `{prov.get('virgil_commit')}`, virgil-vlti `{prov.get('virgil_vlti_commit')}` (channel `{prep.get('insname')}`, "
        f"window {prep.get('window_um')} µm, one bad exposure out), OzSTAR job {prov.get('slurm_job')}.\n",
    ]
    for png, cap in (("search_bands.png", "Pipeline band search: Δlog Z and raw χ²/N by period"),
                     ("modes_sky.png", "Pipeline modes on the sky")):
        if (d / png).exists():
            out.append(f"![{cap}]({rel}/gl229-pipeline/{png})\n")
    return out


def _alias_runs(assets):
    rows = ["| run | commit | start | search effort (candidates/refines/steps) | N = 34: rank, log Z, raw χ²/N | winner |",
            "| --- | --- | --- | --- | --- | --- |"]
    for name, label, commit in (("gl229", "warm, 2026-10-15", "a92b97c"), ("gl229-warm-2946041", "warm (job 18760999)", "2946041"),
                                ("gl229-cold", "cold (job 18758960)", "2946041"), ("gl229-cold-effort", "cold, high effort (job 18761000)", "2946041")):
        d = pathlib.Path(assets) / name
        if not (d / "bands.json").exists():
            continue
        bands, n34, rank = _rank34(json.load(open(d / "bands.json"))["bands"])
        a = (json.load(open(d / "meta.json")).get("args") or {}) if (d / "meta.json").exists() else {}
        start = "cold" if a.get("no_init") else "warm"
        eff = f"{a.get('n_candidates', '?')}/{a.get('n_refine', '?')}/{a.get('max_steps', '?')}"
        rows.append(f"| {label} | {commit} | {start} | {eff} | {rank}, {fmt(n34['log_z'], '.1f')}, {fmt(n34.get('chi2_red'), '.2f')} | "
                    f"N = {bands[0]['n']} ({fmt(bands[0]['log_z'], '.1f')}) |")
    return "\n".join(rows) + "\n"


def gate_section(assets, rel):
    """The Gl 229 release-gate runs, kept beside the warm result of 2026-10-15 in ``gl229/``."""
    sysdir = pathlib.Path(assets) / "gl229-cold"
    pipe = _pipeline_gate(assets, rel)
    if not pipe and not (sysdir / "bands.json").exists():
        return ""
    out = ["## Release gate: Gl 229\n"]
    if pipe:
        out += ["### Blind: OrbitPipeline, no settings\n"] + pipe
    if not (sysdir / "bands.json").exists():
        return "\n".join(out)
    bands, n34, rank = _rank34(json.load(open(sysdir / "bands.json"))["bands"])
    meta = json.load(open(sysdir / "meta.json")) if (sysdir / "meta.json").exists() else {}
    top, nxt = bands[0], bands[1]
    delta = top["log_z"] - nxt["log_z"]
    ok = top["n"] == GATE_N and delta > GATE_NATS
    prov, v = meta.get("provenance") or {}, meta.get("virgil") or {}
    out += [
        "### Cold orbit_aliases, reference-informed window\n",
        f"**{'PASS' if ok else 'FAIL'}.** `reanalyse_orbits.py --system gl229 --no-init`, no warm start, window 10 to 14 d "
        "(12 bands, set from the published period, so not blind). "
        f"virgil {v.get('version', '?')}, commit `{v.get('commit_id') or 'unknown'}`; virgil-validation `{prov.get('validation_commit') or 'unknown'}`; "
        f"options `{prov.get('extra_args') or '?'}`; OzSTAR job {prov.get('slurm_job') or '?'}. "
        "The release adds only #421 (a `starting_image` change in `imaging.py`) and the CHANGELOG to this commit.\n",
        f"Winner: N = {top['n']}, P = {fmt(top.get('period'), '.4f')} d, raw χ²/N {fmt(top.get('chi2_red'), '.2f')}, p = {fmt(top.get('p'), '.3f')}, "
        f"log Z {fmt(top.get('log_z'), '.2f')} (Laplace) vs {fmt(top.get('log_z_is'), '.2f')} (IS); "
        f"Δlog Z = {delta:.1f} over the runner-up (N = {nxt['n']}).\n",
    ]
    if n34 is not None and top["n"] != GATE_N:
        out.append(
            f"Band N = {GATE_N} ranks {rank} of {len(bands)} in this run. **Diagnosis: the cold search seeds the wrong modes; "
            "the likelihood is unchanged.** The warm rerun at the release commit reproduces the 2026-10-15 N = 34 optimum exactly, and a cold run with four times the "
            "candidates, more refinements and five times the steps converges to the same wrong N = 39 and N = 36 optima. More random effort "
            "does not help, so the problem is the starting points, not the search length.\n"
        )
        out.append(_alias_runs(assets))
    out.append(table(bands) + "\n")
    out.append("The warm-started result of 2026-10-15 (virgil a92b97c) is the Gl 229 section above.\n")
    for png, cap in PLOTS:
        if (sysdir / png).exists():
            out.append(f"![{cap}]({rel}/gl229-cold/{png})\n")
    return "\n".join(out)


def verdict_tally(assets, names):
    """Winning-band verdicts of every system with a stored sky-plane row, under the current rule."""
    verdicts = {}
    for n in names:
        mj = pathlib.Path(assets) / n / "meta.json"
        top = (json.load(open(mj)).get("sky_plane") or {}).get("top") if mj.exists() else None
        if top:
            verdicts[n] = current_verdict(top)
    return dict(total=len(verdicts), verdicts=verdicts,
                **{"pass": sum(v == "SAME-SKY-ORBIT" for v in verdicts.values())})


def render(assets=ASSETS, template=TEMPLATE, rel="../assets/orbit_aliases"):
    names = _registry()
    secs = [section(n, v["label"], v["instrument"], assets / n, rel) for n, v in names.items()]
    tally = verdict_tally(assets, names)
    summary = (f"Under the current sky-plane rule (rms offset under {SAME_FRACTION:.0%} of the separation AND under "
               f"{SAME_SIGMA:g} posterior spreads), {tally['pass']} of {tally['total']} systems have a winning-band orbit on the "
               "reference's sky orbit. Verdicts are recomputed here from the stored statistics, not copied from the run.\n")
    secs.insert(0, summary)
    gate = gate_section(assets, rel)
    if gate:
        secs.append(gate)
    skipped = "\n".join(f"- {a}: {b}." for a, b in SKIPPED)
    return string.Template(template.read_text()).substitute(SYSTEMS="\n".join(secs), SKIPPED=skipped)


def _registry():
    """System names, labels and instruments, from reanalyse_orbits.py without importing virgil."""
    import reanalyse_orbits as R

    return {k: dict(label=v.label, instrument=v.instrument) for k, v in R.SYSTEMS.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=str(ASSETS))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    text = render(pathlib.Path(a.assets))
    pathlib.Path(a.out).write_text(text)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
