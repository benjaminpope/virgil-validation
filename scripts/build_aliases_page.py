"""The Period aliases section of docs/orbits/index.md, from scripts/templates/aliases_page.md.tmpl and the
per-system bands.json / meta.json in docs/assets/orbit_aliases/<system>/. scripts/build_orbits_page.py
places it in the orbit page; run on its own, this writes the whole orbit page the same way.

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


GATE_N, GATE_NATS = 34, 10.0  # virgil release gate (CONTRIBUTING "Releasing", item 4)
GATE_VERSION = "0.4.0"  # the release being gated
BAD_N34_FLAGS = ("not-converged", "no-valid-mode")  # a band-34 evidence that cannot carry the gate
TOL_LOGZ = 0.1  # nats: two runs "reproduce" a band's log Z when they agree to this
NOT_RECORDED = "not recorded"
RUNS = (("gl229", "warm, first run"), ("gl229-warm-2946041", "warm"), ("gl229-cold", "cold"),
        ("gl229-cold-effort", "cold, high effort"))


def _load(path):
    path = pathlib.Path(path)
    return json.load(open(path)) if path.exists() else {}


def is_cold(meta):
    """A run is cold if its own record says so: ``args.no_init`` is true and no ``init`` orbit was used. Never from
    a directory or tag name; a record with no ``args`` is not known to be cold."""
    return (meta.get("args") or {}).get("no_init") is True and meta.get("init") is None


def short(commit):
    return commit[:7] if commit else NOT_RECORDED


def _rank34(bands):
    bands = sorted(bands, key=lambda b: -b["log_z"])
    n34 = next((b for b in bands if b["n"] == GATE_N), None)
    return bands, n34, ([b["n"] for b in bands].index(GATE_N) + 1 if n34 else None)


def pipeline_gate(m, reference_meta):
    """Recompute the blind gate from a ``gl229-pipeline/meta.json`` record, without trusting virgil's verdict string.

    PASS needs virgil's own verdict, the recomputed margin (winner_n == GATE_N and delta log Z > GATE_NATS) and the
    winner's orbit on the reference's sky orbit (`crosscheck.skyplane.score` at the pipeline's nights, with the
    reference of ``reference_meta``) to agree. Returns a dict with ``ok``, the three parts, the sky row, and the
    non-pass checks (all of them are shown beside the verdict).
    """
    from crosscheck import skyplane

    g, s = m["gate"], m["summary"]
    margin_ok = g["winner_n"] == GATE_N and g["delta_log_z"] > GATE_NATS
    row, sky_verdict = None, None
    ref = (reference_meta or {}).get("reference")
    if ref and g.get("winner_elements"):
        nights = s["nights"]
        el = g["winner_elements"]
        t_ref = (s.get("search") or {}).get("t_ref", 0.0)
        row = skyplane.score(dict(mjd=[n["mjd"] for n in nights], reference=ref), el, t_ref=t_ref,
                             flipped=el.get("flux", 0) > 1, labels_free=bool((s.get("search") or {}).get("labels_free")),
                             chi2_n=g.get("chi2_per_n_raw"))
        row = dict(row, band=g["winner_n"], period=el["period"], p=None)
        sky_verdict = current_verdict(row)
    virgil_ok = g.get("verdict") == "PASS"
    return dict(ok=virgil_ok and margin_ok and sky_verdict == "SAME-SKY-ORBIT", virgil_ok=virgil_ok, margin_ok=margin_ok,
                sky_verdict=sky_verdict, row=row,
                warnings=[c for c in s.get("checks") or [] if c.get("status") != "pass"],
                out_of_range=bool(((s.get("search") or {}).get("tempering") or {}).get("out_of_range")))


def _pipeline_gate(assets, rel):
    """The blind gate: virgil's OrbitPipeline on Gl 229 with no settings, on data prepared on the virgil-vlti side."""
    d = pathlib.Path(assets) / "gl229-pipeline"
    if not (d / "meta.json").exists():
        return []
    m = json.load(open(d / "meta.json"))
    ref_meta = _load(pathlib.Path(assets) / "gl229" / "meta.json")
    g, s, prov = m["gate"], m["summary"], m.get("provenance") or {}
    prep = m.get("preparation") or {}
    v = m.get("virgil") or {}
    cold_v = (_load(pathlib.Path(assets) / "gl229-cold" / "meta.json").get("virgil") or {})
    win = s["modes"]["winner"]
    el = win.get("elements") or {}
    temper = (s.get("search") or {}).get("tempering") or {}
    r = pipeline_gate(m, ref_meta)
    warn_bits = [f"{c['name']} ({c['status']})" for c in r["warnings"]]
    label = "PASS" if r["ok"] else "FAIL"
    if r["ok"] and warn_bits:
        label += f", with {len(warn_bits)} virgil warnings"
    out = [
        f"**{label}.** `OrbitPipeline` on Gl 229 with no settings: no period window, start or published orbit "
        f"(the window, {fmt(g['p_range'][0], '.1f')} to {fmt(g['p_range'][1], '.0f')} d, comes from the data, with {g['n_bands']} bands). "
        f"It must rank band N = {GATE_N} first with Δlog Z > {GATE_NATS:g} over every other band, and the winner must lie on the "
        "published orbit on the sky. vv recomputes this: "
        f"virgil's verdict is {g.get('verdict')}; winner N = {g['winner_n']} and Δlog Z = {fmt(g['delta_log_z'], '.1f')} "
        f"{'meet' if r['margin_ok'] else 'do not meet'} the margin test; the sky-plane verdict is {r['sky_verdict'] or 'not computed'}.\n",
        f"Winner: N = {g['winner_n']}, P = {fmt(el.get('period'), '.3f')} d, e = {fmt(el.get('ecc'), '.3f')}, flux ratio {fmt(el.get('flux'), '.3f')}. "
        f"Δlog Z = {fmt(g['delta_log_z'], '.1f')} over the runner-up (N = {g['runner_up_n']}), on the **tempered** likelihood "
        f"(effective-channel factor f = {fmt(temper.get('f'), '.1f')}"
        f"{', outside virgil\'s accepted range 0.5 to 2' if temper.get('out_of_range') else ''}). "
        "Tempering rescales the evidences to absorb errors that are too small; it is not a noise model, and this margin "
        "can move across the 10-nat line when f changes.\n",
    ]
    if r["row"]:
        out += ["Winner against the published orbit, on the sky at the pipeline's nights:\n", sky_table(dict(top=r["row"]))]
    if r["warnings"]:
        out.append("virgil's own checks that are not a pass:\n")
        out += [f"- **{c['name']}** ({c['status']}): {c.get('message', '')}" for c in r["warnings"]]
        out.append("")
    pv = v.get("version")
    out.append(
        f"Provenance: this blind run is virgil {pv or '?'} commit `{short(v.get('commit_id') or prov.get('virgil_commit'))}`, "
        + (f"not the {GATE_VERSION} release code (`{short(cold_v.get('commit_id'))}`): no `OrbitPipeline` run at the release commit "
           "is stored here, so this is a pre-release result. " if pv != GATE_VERSION else f"release version {GATE_VERSION}. ")
        + f"virgil-vlti `{short(prov.get('virgil_vlti_commit'))}`; OzSTAR job {prov.get('slurm_job') or NOT_RECORDED}.\n")
    nights = list(prep.get("nights") or [])
    alias_epochs = _load(pathlib.Path(assets) / "gl229" / "meta.json").get("epochs") or []
    extra, missing = sorted(set(nights) - set(alias_epochs)), sorted(set(alias_epochs) - set(nights))
    excl = prep.get("excluded") or {}
    flagged = [n["dataset"] for n in s["nights"] if n.get("flags")]
    out.append(
        f"Data: virgil-vlti's preparation, channel `{prep.get('insname')}`, window {prep.get('window_um')} µm, "
        f"{prep.get('n_files_kept')} of {prep.get('n_files')} files, {len(nights)} nights"
        + (f"; excluded: {'; '.join(f'{k} ({why})' for k, why in excl.items())}" if excl else "")
        + ". The alias runs below use this repository's loader (`reanalyse_orbits.py`), which is not the same preparation: "
        + ("the nights differ"
           + (f" (only in the pipeline: {', '.join(extra)}" if extra else " (")
           + (f"{'; ' if extra else 'only in the alias runs: '}{', '.join(missing)}" if missing else "") + ")" if extra or missing
           else "the recorded nights are the same")
        + f". Nights flagged by virgil: {', '.join(flagged) or 'none'}. The two results are therefore not on identical data "
        "(the files and errors differ even where the nights match).\n")
    out.append(f"Raw χ²/N {fmt(g['chi2_per_n_raw'], '.1f')} on the quoted errors: the errors are far too small, a noise-model matter for "
               "virgil-vlti, not the period search.\n")
    for png, cap in (("search_bands.png", "Pipeline band search: Δlog Z and raw χ²/N by period"),
                     ("modes_sky.png", "Pipeline modes on the sky")):
        if (d / png).exists():
            out.append(f"![{cap}]({rel}/gl229-pipeline/{png})\n")
    return out


def _alias_runs(assets):
    rows = ["| run | virgil | commit | start | job | search effort (candidates/refines/steps) | N = 34: rank, log Z, raw χ²/N | winner |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, label in RUNS:
        d = pathlib.Path(assets) / name
        if not (d / "bands.json").exists():
            continue
        bands, n34, rank = _rank34(json.load(open(d / "bands.json"))["bands"])
        meta = _load(d / "meta.json")
        a, v, prov = meta.get("args") or {}, meta.get("virgil") or {}, meta.get("provenance") or {}
        start = "cold" if is_cold(meta) else ("warm" if meta.get("args") else "unknown")
        eff = f"{a.get('n_candidates', '?')}/{a.get('n_refine', '?')}/{a.get('max_steps', '?')}"
        n34s = f"{rank}, {fmt(n34['log_z'], '.1f')}, {fmt(n34.get('chi2_red'), '.2f')}" if n34 else "no N = 34 band"
        rows.append(f"| {label} (`{name}`) | {v.get('version') or '?'} | {short(v.get('commit_id') or prov.get('virgil_commit'))} | {start} | "
                    f"{prov.get('slurm_job') or NOT_RECORDED} | {eff} | {n34s} | N = {bands[0]['n']} ({fmt(bands[0]['log_z'], '.1f')}) |")
    return "\n".join(rows) + "\n"


def _logz34(assets, name):
    p = pathlib.Path(assets) / name / "bands.json"
    n34 = _rank34(json.load(open(p))["bands"])[1] if p.exists() else None
    return None if n34 is None else n34["log_z"]


def diagnosis(assets):
    """The diagnosis of a cold failure, each claim computed from the stored runs; lines for what is not established."""
    A = pathlib.Path(assets)
    cold, eff, warm0, warm1 = (_load(A / n / "meta.json") for n in ("gl229-cold", "gl229-cold-effort", "gl229", "gl229-warm-2946041"))
    lines, missing = [], []
    z0, z1 = _logz34(A, "gl229"), _logz34(A, "gl229-warm-2946041")
    cold_commit = (cold.get("virgil") or {}).get("commit_id")
    if z0 is not None and z1 is not None and (warm1.get("virgil") or {}).get("commit_id") == cold_commit and cold_commit:
        if abs(z0 - z1) < TOL_LOGZ:
            lines.append(f"The likelihood is unchanged: the warm N = {GATE_N} log Z is {z0:.1f} in the first run and {z1:.1f} in the "
                         f"warm rerun at the cold run's commit `{short(cold_commit)}` (agreement within {TOL_LOGZ:g} nats).")
        else:
            lines.append(f"The likelihood changed: the warm N = {GATE_N} log Z is {z0:.1f} in the first run but {z1:.1f} at the cold run's commit.")
    else:
        missing.append("no warm rerun at the cold run's commit to compare the likelihood")
    cb, eb = A / "gl229-cold" / "bands.json", A / "gl229-cold-effort" / "bands.json"
    if cb.exists() and eb.exists() and is_cold(eff):
        top2 = lambda p: [b["n"] for b in sorted(json.load(open(p))["bands"], key=lambda b: -b["log_z"])[:2]]
        a_c, a_e = cold.get("args") or {}, eff.get("args") or {}
        ratio = lambda k: (a_e[k] / a_c[k]) if a_e.get(k) and a_c.get(k) else None
        rc, rs = ratio("n_candidates"), ratio("max_steps")
        same = top2(cb) == top2(eb)
        if rc and rs and rc > 1 and rs > 1:
            lines.append(f"A cold run with {rc:g} times the candidates and {rs:g} times the steps (refines {a_c.get('n_refine')} to "
                         f"{a_e.get('n_refine')}) ranks N = {top2(eb)[0]} then N = {top2(eb)[1]}, "
                         + ("the same two optima as the base cold run: more random effort does not move them."
                            if same else f"not the base cold run's N = {top2(cb)[0]}, N = {top2(cb)[1]}."))
    else:
        missing.append("no cold high-effort run to test whether more search effort helps")
    if len(lines) == 2 and "unchanged" in lines[0] and "does not move" in lines[1]:
        lines.append("So the cold search seeds the wrong modes: the likelihood supports N = 34 when started there, and more random "
                     "search does not find it from a cold start.")
    elif missing:
        lines.append("Diagnosis not established: " + "; ".join(missing) + ".")
    return lines


def gate_section(assets, rel):
    """The Gl 229 release-gate runs, kept beside the first warm result in ``gl229/``."""
    A = pathlib.Path(assets)
    sysdir = A / "gl229-cold"
    pipe = _pipeline_gate(assets, rel)
    if not pipe and not (sysdir / "bands.json").exists():
        return ""
    out = ["## Release gate: Gl 229\n"]
    if pipe:
        out += ["### Blind: OrbitPipeline, no settings\n"] + pipe
    if not (sysdir / "bands.json").exists():
        return "\n".join(out)
    bands, n34, rank = _rank34(json.load(open(sysdir / "bands.json"))["bands"])
    meta = _load(sysdir / "meta.json")
    top, nxt = bands[0], bands[1]
    delta = top["log_z"] - nxt["log_z"]
    sp = meta.get("sky_plane") or {}
    row34 = next((v for v in sp.values() if v.get("band") == GATE_N), None)
    sky34 = current_verdict(row34) if row34 else None
    cold = is_cold(meta)
    bad = [f for f in (n34 or {}).get("flags") or [] if f in BAD_N34_FLAGS]
    ok = cold and top["n"] == GATE_N and delta > GATE_NATS and sky34 == "SAME-SKY-ORBIT" and not bad
    prov, v = meta.get("provenance") or {}, meta.get("virgil") or {}
    wmeta = _load(A / "gl229" / "meta.json")
    pr = meta.get("p_range")
    out += [
        "### Cold orbit_aliases, reference-informed window\n",
        f"**{'PASS' if ok else 'FAIL'}.** `reanalyse_orbits.py --system gl229 --no-init`: "
        f"{'a cold start, from the run\'s own record (no_init set, no init orbit)' if cold else '**this run is not recorded as cold (args.no_init / init), so it cannot pass**'}; "
        f"window {pr[0]:g} to {pr[1]:g} d ({len(bands)} bands), set from the published period, so not blind. "
        f"PASS needs the cold record, N = {GATE_N} first by more than {GATE_NATS:g} nats, N = {GATE_N} on the reference's sky orbit, "
        f"and no {'/'.join(BAD_N34_FLAGS)} flag on it. "
        f"virgil {v.get('version', '?')}, commit `{short(v.get('commit_id'))}`"
        + (f" ({'the' if v.get('version') == GATE_VERSION else 'not the'} {GATE_VERSION} release)" if v.get("version") else "")
        + f"; virgil-validation `{short(prov.get('validation_commit'))}`; "
        f"options `{prov.get('extra_args') or '?'}`; OzSTAR job {prov.get('slurm_job') or NOT_RECORDED}.\n",
        f"Winner: N = {top['n']}, P = {fmt(top.get('period'), '.4f')} d, raw χ²/N {fmt(top.get('chi2_red'), '.2f')}, p = {fmt(top.get('p'), '.3f')}, "
        f"log Z {fmt(top.get('log_z'), '.2f')} (Laplace) vs {fmt(top.get('log_z_is'), '.2f')} (IS); "
        f"Δlog Z = {delta:.1f} over the runner-up (N = {nxt['n']}). "
        f"Band N = {GATE_N}: {f'rank {rank} of {len(bands)}, flags {flag_text(n34) or chr(8211)}' if n34 else 'absent from this run'}; "
        f"sky-plane verdict {sky34 or 'not computed'}.\n",
    ]
    if sp:
        out.append(sky_table(sp))
    out += chi2_warning(meta)
    if top["n"] != GATE_N or not ok:
        out += [line + "\n" for line in diagnosis(assets)] if top["n"] != GATE_N else []
        out.append(_alias_runs(assets))
    out.append(table(bands) + "\n")
    w = _logz34(A, "gl229")
    if w is not None:
        wv = wmeta.get("virgil") or {}
        out.append(f"The first warm result (virgil {wv.get('version', '?')}, commit `{short(wv.get('commit_id'))}`) is the Gl 229 section above. "
                   f"Its N = {GATE_N} log Z ({w:.1f}) was **warm-started at the published orbit** "
                   f"({'init recorded' if wmeta.get('init') else 'the record shows no init orbit, but args.no_init is false'}), so it is not a cold optimum "
                   "and not independent evidence for N = 34.\n")
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
    import build_orbits_page

    build_orbits_page.main(argv)


if __name__ == "__main__":
    main()
