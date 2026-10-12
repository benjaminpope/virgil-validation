"""Binary orbit gallery: trimmed fit summaries and the generated docs page.

    python scripts/orbit_gallery.py extract <dir of fit_<id>.json> [--out trust/campaigns/orbits]
    python scripts/orbit_gallery.py render --out docs/orbits/binaries.md --json docs/assets/orbits.json

``extract`` (run once, on the fits copied back from OzSTAR) keeps what the gallery needs of each
``fit_<collection>.json`` of scripts/oidb_fit.py (the MAP and its alternative modes, the NUTS
summaries, the per-epoch positions, the raw chi2, the provenance commits) and drops the bulky
per-dataset entries. The summaries have the shape of a fit file, so ``oidb_compare.compare`` reads
them directly (its ``fits=`` argument).

``render`` (in the docs CI, git-ignored output) reads the summaries and ``oidb/references/*.json``,
samples the sky-plane polylines with ``crosscheck.orbits`` (the independent reference code: the
browser never reimplements Kepler's equation), applies the convention fixes of
``oidb_compare.orbit_rows`` (the primary's omega + 180, a in au -> mas), takes the parameter table
from ``oidb_compare.compare`` and writes the JSON for ``docs/assets/orbits.js`` and a static
Markdown page (one section per target, readable without JavaScript). A target without a summary
gets an "awaiting" tile, so the page builds before the fits land.

Units are mas and degrees; sky paths are (dRA, dDec) with dRA east, closed polylines.
"""

import argparse
import json
import math
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from crosscheck import orbits  # noqa: E402
import oidb_compare as oc  # noqa: E402

SUMMARIES = ROOT / "trust" / "campaigns" / "orbits"
REFERENCES = ROOT / "oidb" / "references"
MANIFEST = ROOT / "oidb" / "manifest.yml"
N_PATH = 200  # points on a sky polyline (the epoch times are added to them)
N_PHASE = 120
TENSION_SIGMA = 3.0  # a headline published orbit further than this from our epoch position is a tension
HD45166_DISTANCE_PC = 991.0  # Deshmukh et al. 2025, adopted distance (the reference's 'other' notes)
INSTRUMENT = {"GRAVITY": "VLTI/GRAVITY", "MIRCX": "CHARA/MIRC-X", "PIONIER": "VLTI/PIONIER"}

# gallery order; ``kind`` orbit: we fit an orbit; positions: epoch positions or a diameter only
TARGETS = {
    "782185b2-0727-42b0-a185-b2072732b047": dict(slug="gl229b", name="Gliese 229 B (Ba-Bb)", kind="orbit"),
    "696baf06-6c3c-424d-abaf-066c3c324d99": dict(slug="hr6819", name="HR 6819", kind="orbit"),
    "fac164e1-d9d0-4500-8164-e1d9d0450099": dict(slug="iota-peg", name="iota Peg", kind="orbit"),
    "f4afc4cd-fd31-40d3-afc4-cdfd3150d340": dict(slug="hd45166", name="HD 45166", kind="positions"),
    "bda75673-61c6-49f0-a756-7361c699f0c4": dict(slug="a-stars", name="A-star companions (27 stars)",
                                                 kind="positions"),
    "19f7e2cf-2a03-4bb2-b7e2-cf2a03bbb245": dict(slug="pi1-gru", name="pi1 Gru", kind="positions"),
}
ELEMENTS = ("period_day", "t_peri_mjd", "ecc", "inc_deg", "omega_deg", "Omega_deg", "a_mas")


def _num(x):
    return x[0] if isinstance(x, (list, tuple)) else x


def _clean(x):
    """JSON-safe: numpy to Python, NaN/inf to None, floats to 10 significant digits (an MJD needs them)."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.floating, np.integer)):
        x = x.item()
    if isinstance(x, float):
        return float(f"{x:.10g}") if math.isfinite(x) else None
    return x


def load_manifest():
    lines = MANIFEST.read_text().splitlines()
    return json.loads("\n".join(x for x in lines if not x.lstrip().startswith("#")))


# ----------------------------------------------------------------------------- extract

def _keep(d, keys):
    return {k: d[k] for k in keys if k in d}


def trim_epoch(e):
    return _keep(e, ("label", "variant", "error", "mjd", "values", "cov_names", "cov", "chi2_raw", "scales",
                     "fractions", "flux", "resolved", "ud1", "ud2"))


def trim_orbit(o):
    out = _keep(o, ("t_ref", "map", "loss", "chi2_raw"))
    out["modes"] = [_keep(m, ("loss", "elements")) for m in o.get("modes", [])]
    out["positions"] = [_keep(p, ("name", "mjd", "dra", "ddec", "cov", "flux", "rho_mas", "pa_deg"))
                        for p in o.get("positions", [])]
    if o.get("nuts"):
        out["nuts"] = _keep(o["nuts"], ("elements", "r_hat", "divergences"))
    return out


def trim_fit(fit):
    """The part of a fit_<id>.json the gallery and ``oidb_compare`` use, in the same shape."""
    out = _keep(fit, ("model", "flags", "recipe", "quick"))
    if "orbit" in fit:
        out["orbit"] = trim_orbit(fit["orbit"])
    if "epochs" in fit:
        out["epochs"] = [trim_epoch(e) for e in fit["epochs"]]
    if "parametric" in fit:
        out["parametric"] = trim_epoch(fit["parametric"])
    if "variants" in fit:
        out["variants"] = [trim_epoch(v) for v in fit["variants"]]
    if "targets" in fit:
        out["targets"] = {name: dict(epochs=[trim_epoch(e) for e in t.get("epochs", [])],
                                     **({"variants": [trim_epoch(v) for v in t["variants"]]}
                                        if t.get("variants") else {}))
                          for name, t in fit["targets"].items()}
    return out


def extract(fits_dir, out_dir=SUMMARIES):
    """One ``<collection_id>.json`` summary per published-binary ``fit_*.json`` in ``fits_dir``."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {c["id"]: c for c in load_manifest()["collections"]}
    written = []
    for path in sorted(pathlib.Path(fits_dir).glob("fit_*.json")):
        fit = json.loads(path.read_text())
        cid = fit.get("collection") or path.stem[len("fit_"):]
        if cid not in TARGETS:  # the L2 workshop set and the collections the gallery does not show
            print(f"skip {path.name}: not a gallery target")
            continue
        prov = fit.get("provenance") or {}
        c = manifest.get(cid, {})
        summary = dict(collection_id=cid, target=TARGETS[cid]["name"],
                       instrument=" + ".join(INSTRUMENT.get(i, i) for i in c.get("instruments", [])),
                       provenance=_keep(prov, ("virgil_commit", "validation_commit")), **trim_fit(fit))
        dest = out_dir / f"{cid}.json"
        dest.write_text(json.dumps(_clean(summary), indent=1) + "\n")
        written.append(dest)
        print(f"wrote {dest} ({dest.stat().st_size / 1024:.0f} kB)")
    return written


# ----------------------------------------------------------------------------- orbits to polylines

def _orbit_at(el, t):
    return orbits.position(t, el["period_day"], el["t_peri_mjd"], el["ecc"], el["inc_deg"], el["omega_deg"],
                           el["Omega_deg"], el["a_mas"])


def sky_path(el, mjds=(), n=N_PATH):
    """Closed (dRA, dDec) polyline of an orbit: ``n`` points uniform in mean anomaly plus the orbit's
    position at each of ``mjds``, so the path passes exactly through them."""
    P, tp = el["period_day"], el["t_peri_mjd"]
    phase = np.concatenate([np.arange(n) / n, [((t - tp) / P) % 1.0 for t in mjds]])
    phase = np.unique(np.round(phase, 12))
    dra, ddec = _orbit_at(el, tp + phase * P)
    pts = np.column_stack([np.ravel(dra), np.ravel(ddec)])
    return _clean(np.round(np.vstack([pts, pts[:1]]), 6).tolist())


def phase_curve(el, n=N_PHASE):
    """[[phase, sep (mas), PA (deg)], ...] over one period from periastron."""
    phase = np.arange(n + 1) / n
    dra, ddec = _orbit_at(el, el["t_peri_mjd"] + phase * el["period_day"])
    dra, ddec = np.ravel(dra), np.ravel(ddec)
    return _clean(np.round(np.column_stack([phase, np.hypot(dra, ddec), np.degrees(np.arctan2(dra, ddec)) % 360.0]), 6).tolist())


def _sep_pa(dra, ddec):
    return math.hypot(dra, ddec), math.degrees(math.atan2(dra, ddec)) % 360.0


def _elements_of(d):
    return {k: float(d[k]) for k in ELEMENTS}


def _our_elements(orbit):
    return _elements_of(orbit["map"])


def published_orbits(cid, ref):
    """[(label, elements)] of the published orbits, headline first, in virgil's convention: the
    companion's omega (the primary's + 180, as ``oidb_compare.orbit_rows``) and a in mas."""
    if cid.startswith("782185b2"):
        out = []
        for label, key in (("PMOIRED (Xuan+24)", "pmoired_frequentist"), ("Octofitter (Xuan+24)", "octofitter_bayesian")):
            s = ref["orbit"][key]
            out.append((label, dict(period_day=s["P_day"][0], t_peri_mjd=s["T_peri_mjd"][0], ecc=s["e"][0],
                                    inc_deg=s["i_deg"][0], omega_deg=(s["omega_primary_deg"][0] + 180) % 360,
                                    Omega_deg=s["Omega_deg"][0] % 360,
                                    a_mas=s["a_au"][0] * oc.CRITERIA["gl229_parallax_mas"])))
        return out
    if cid.startswith("696baf06"):
        sols = ref["orbit"]["all_solutions"]
        table3 = next(v for k, v in sols.items() if k.startswith("Table 3"))
        out = []
        # Table 3 columns: P, T0 (RJD = MJD + 0.5), e, a (mas), i, Omega, omega (the Be star's: + 180)
        for key in ("eccentric_astro+RVs", "eccentric_astro+RVpre-sd+K_Be", "circular_astro+RVs",
                    "circular_astro+RVpre-sd+K_Be"):
            c = table3[key]
            out.append((key.replace("_", " ") + " (Klement+25)",
                        dict(period_day=c[0][0], t_peri_mjd=c[1][0] - 0.5, ecc=c[2][0], inc_deg=c[4][0],
                             omega_deg=(c[6][0] + 180) % 360, Omega_deg=c[5][0] % 360, a_mas=c[3][0])))
        astro = next(v for k, v in sols.items() if k.startswith("astrometry_only"))
        # omega here is read as the pre-sd star's (no shift). That holds for the circular row but not
        # obviously for the eccentric one: its 117.6 deg is close to Table 3's omega_Be = 111.9, so
        # the eccentric curve may be half an orbit out of phase with the headline.
        for key in ("eccentric", "circular"):
            c = astro[key]
            note = " (omega convention unclear)" if key == "eccentric" else ""
            out.append((f"{key} astrometry only{note} (Klement+25)",
                        dict(period_day=c["P_day"][0], t_peri_mjd=c["T0_RJD"][0] - 0.5, ecc=_num(c["e"]),
                             inc_deg=c["i_deg"][0], omega_deg=c["omega_deg"][0] % 360 if isinstance(c["omega_deg"], list)
                             else c["omega_deg"] % 360, Omega_deg=c["Omega_deg"][0] % 360, a_mas=c["a_mas"][0])))
        return out
    if cid.startswith("fac164e1"):
        o = ref["orbit"]
        out = [("Anugu+20 (omega component / T system unstated)", dict(period_day=o["P_day"][0], t_peri_mjd=o["T_peri_unstated_system"][0], ecc=o["e"][0],
                                 inc_deg=o["i_deg"][0], omega_deg=o["omega_deg"][0] % 360,
                                 Omega_deg=o["Omega_deg"][0] % 360, a_mas=o["a_mas"][0]))]
        k = o["comparison_orbit_Konacki2010"]
        out.append(("Konacki+10", dict(period_day=k["P_day"][0], t_peri_mjd=k["T"][0], ecc=k["e"][0],
                                       inc_deg=k["i_deg"][0], omega_deg=k["omega_deg"][0] % 360,
                                       Omega_deg=k["Omega_deg"][0] % 360, a_mas=k["a_mas"][0])))
        return out
    if cid.startswith("f4afc4cd"):
        o = ref["orbit"]
        M, P = o["masses"]["M_total_Msun"][0], o["P_day"][0]
        a_mas = (M * (P / 365.25) ** 2) ** (1 / 3) / HD45166_DISTANCE_PC * 1000.0  # Kepler III; au / pc = arcsec
        return [("Deshmukh+25 (a from Kepler III, M_tot, 991 pc)",
                 dict(period_day=P, t_peri_mjd=o["T_peri_mjd"][0], ecc=o["e"][0], inc_deg=o["i_deg"][0],
                      omega_deg=o["omega_deg"][0] % 360, Omega_deg=o["Omega_deg"][0] % 360, a_mas=a_mas))]
    return []


# ----------------------------------------------------------------------------- epochs

def _cov_of_epoch(e):
    names, cov = e.get("cov_names"), e.get("cov")
    if not names or cov is None or "dra" not in names or "ddec" not in names:
        return None
    i, j = names.index("dra"), names.index("ddec")
    return [[cov[i][i], cov[i][j]], [cov[j][i], cov[j][j]]]


def _ours_from_fit_epoch(e, label=None):
    v = e.get("values", {})
    if "dra" in v and "ddec" in v:
        dra, ddec = v["dra"][0], v["ddec"][0]
    elif "rho_mas" in v and "pa_deg" in v:
        p = math.radians(v["pa_deg"][0])
        dra, ddec = v["rho_mas"][0] * math.sin(p), v["rho_mas"][0] * math.cos(p)
    else:
        return None
    return dict(label=label or e.get("label"), mjd=e.get("mjd"), dra=dra, ddec=ddec, cov=_cov_of_epoch(e))


def ours_epochs(cid, summary):
    """Our per-epoch positions: the orbit fit's start-search positions, else the static epoch fits."""
    if summary.get("orbit", {}).get("positions"):
        return [dict(label=p["name"], mjd=p["mjd"], dra=p["dra"], ddec=p["ddec"], cov=p.get("cov"))
                for p in summary["orbit"]["positions"]]
    if cid.startswith("bda75673"):  # the detections: one epoch per star
        out = []
        for star, t in summary.get("targets", {}).items():
            e = next(iter(t.get("epochs", [])), None)
            o = e and _ours_from_fit_epoch(e, label=star)
            if o:
                out.append(o)
        return out
    return [o for o in (_ours_from_fit_epoch(e) for e in summary.get("epochs", [])) if o]


def _polar_cov(rho, pa, srho, spa):
    p = math.radians(pa)
    r, t = np.array([math.sin(p), math.cos(p)]), np.array([math.cos(p), -math.sin(p)])
    return (srho ** 2 * np.outer(r, r) + (rho * math.radians(spa)) ** 2 * np.outer(t, t)).tolist()


def _err(x):
    """The error of a [value, err, ...] entry when it is a number, else None."""
    if isinstance(x, (list, tuple)) and len(x) > 1 and isinstance(x[1], (int, float)):
        return float(x[1])
    return None


def published_position(r):
    """{dra, ddec, cov} of a reference epoch: dRA/dDec if the paper gives them, else rho/PA."""
    o = r.get("other") or {}
    cov = None
    if "dRA_mas" in o:
        dra, ddec = o["dRA_mas"], o["dDec_mas"]
    elif "dE_mas" in o:
        dra, ddec = o["dE_mas"][0], o["dN_mas"][0]
        cov = np.diag([o["dE_mas"][1] ** 2, o["dN_mas"][1] ** 2]).tolist()
    elif "rho_mas" in r:
        rho, pa = _num(r["rho_mas"]), _num(r["pa_deg"])
        dra, ddec = rho * math.sin(math.radians(pa)), rho * math.cos(math.radians(pa))
        if _err(r["rho_mas"]) and _err(r["pa_deg"]):
            cov = _polar_cov(rho, pa, _err(r["rho_mas"]), _err(r["pa_deg"]))
    else:
        return None
    if "err_ellipse_major_mas" in o:
        cov = oc.ellipse_cov(o["err_ellipse_major_mas"], o["err_ellipse_minor_mas"], o["err_ellipse_PA_deg"]).tolist()
    return dict(dra=dra, ddec=ddec, cov=cov)


def published_epochs(cid, ref, ours):
    """Published position for each of our epochs (None where the paper has no matching one)."""
    if cid.startswith("782185b2"):  # the PMOIRED orbit at our epoch times
        el = published_orbits(cid, ref)[0][1]
        return [dict(zip(("dra", "ddec"), (float(x) for x in np.ravel(_orbit_at(el, e["mjd"]))))) for e in ours]
    if cid.startswith("bda75673"):
        by_star = {r["star"]: dict(r, rho_mas=[r["rho_mas"], r["rho_err_mas"]], pa_deg=[r["pa_deg_E_of_N"], r["pa_err_deg"]])
                   for r in ref["epochs"]}
        return [published_position(by_star[e["label"]]) if e["label"] in by_star else None for e in ours]
    out = []
    for e in ours:
        probe = oc.by_epoch([dict(label=e["label"], mjd=e["mjd"])], ref.get("epochs", []))
        r = next((r for r, f in probe if f), None)
        out.append(published_position(r) if r else None)
    return out


# ----------------------------------------------------------------------------- one target

def _tension(ours_ep, headline):
    """Mahalanobis distance (our covariance) of our epoch positions from the headline published orbit."""
    worst = 0.0
    for e in ours_ep:
        if e.get("cov") is None or e.get("mjd") is None:
            continue
        pdra, pddec = (float(x) for x in np.ravel(_orbit_at(headline, e["mjd"])))
        d = np.array([e["dra"] - pdra, e["ddec"] - pddec])
        try:
            worst = max(worst, float(math.sqrt(d @ np.linalg.solve(np.asarray(e["cov"], float), d))))
        except np.linalg.LinAlgError:
            continue
    return worst


def paper_of(ref):
    p = ref["paper"]
    first = re.split(r"[,;]", p["authors"])[0].strip()
    who = first if first.endswith("et al.") else f"{first} et al."
    journal = re.sub(r"\s*\(.*", "", p.get("journal") or "")
    bib = (p.get("bibcode") or "").split(" (")[0] or None
    return dict(cite=f"{who} {p['year']}, {journal}", arxiv=p.get("arxiv"), bibcode=bib, doi=p.get("doi"))


def table_of(rows):
    out = []
    for r in rows:
        if r["quantity"] == "fit":
            continue
        pe = r["published_err"]
        q = r["quantity"] if r["epoch"] in ("orbit", None, "") else f"{r['epoch']}: {r['quantity']}"
        out.append(dict(quantity=q, ours=r["virgil"], err=r["virgil_err"], pub=r["published"],
                        pub_err=max(abs(x) for x in pe) if isinstance(pe, (list, tuple)) else pe,
                        pull=r["dev_sigma"], status=r["status"], note=r["note"]))
    return out


def build_target(cid, ref, summary, rows):
    cfg = TARGETS[cid]
    manifest = {c["id"]: c for c in load_manifest()["collections"]}.get(cid, {})
    pubs = published_orbits(cid, ref)
    t = dict(slug=cfg["slug"], collection_id=cid, target=cfg["name"],
             instrument=(summary or {}).get("instrument") or " + ".join(
                 INSTRUMENT.get(i, i) for i in manifest.get("instruments", [])),
             kind=cfg["kind"], status="awaiting", paper=paper_of(ref), ours=None, alternatives=[], published=[],
             epochs=[], phase=dict(ours=[], published=[]), table=[], provenance={})
    ep_mjds = []
    ours_ep = []
    if summary is not None:
        t["provenance"] = summary.get("provenance", {})
        ours_ep = ours_epochs(cid, summary)
        ep_mjds = [e["mjd"] for e in ours_ep if e.get("mjd") is not None]
        pub_ep = published_epochs(cid, ref, ours_ep)
        t["epochs"] = []
        for e, p in zip(ours_ep, pub_ep):
            sep, pa = _sep_pa(e["dra"], e["ddec"])
            d = dict(label=e["label"], mjd=e["mjd"], dra=e["dra"], ddec=e["ddec"], cov=e["cov"], sep=sep, pa=pa)
            if p:
                d["pub"] = p
            t["epochs"].append(d)
        t["table"] = table_of(rows)
        orb = summary.get("orbit")
        if cfg["kind"] == "orbit" and orb:
            el = _our_elements(orb)
            t["ours"] = dict(label="virgil MAP", path=sky_path(el, ep_mjds), elements=el)
            best = orb["modes"][0]["loss"] if orb.get("modes") else orb.get("loss")
            for k, m in enumerate(orb.get("modes", [])):
                if k == 0:
                    continue  # the best mode is the MAP
                mel = _elements_of(m["elements"])
                t["alternatives"].append(dict(label=f"mode {k + 1}", dloss=m["loss"] - best,
                                              path=sky_path(mel, ep_mjds), elements=mel))
            t["phase"]["ours"] = phase_curve(el)
    for label, el in pubs:
        t["published"].append(dict(label=label, path=sky_path(el, ep_mjds), elements=el))
    if pubs:
        t["phase"]["published"] = phase_curve(pubs[0][1])
    if summary is not None:
        scored = [r for r in rows if r["status"] in ("FAIL", "MISSING") and r["quantity"] != "fit"]
        # the badge follows the scored rows only: agrees needs one PASS, tension a FAIL/MISSING; rows
        # oidb_compare keeps as REPORT/NOT-CLEAN (orbit elements, per-night positions, iota Peg's
        # unstated conventions) give "reported". The distance from the headline orbit is shown, not scored.
        worst = _tension(ours_ep, pubs[0][1]) if pubs and cfg["kind"] == "orbit" else 0.0
        passed = any(r["status"] == "PASS" for r in rows)
        t["status"] = "tension" if scored else "agrees" if passed else "reported"
        t["tension_sigma"] = worst
    return _clean(t)


# ----------------------------------------------------------------------------- render

def load_summaries(summaries_dir):
    out = {}
    for cid in TARGETS:
        p = pathlib.Path(summaries_dir) / f"{cid}.json"
        if p.exists():
            out[cid] = json.loads(p.read_text())
    return out


def build(summaries_dir=SUMMARIES, refs_dir=REFERENCES):
    summaries = load_summaries(summaries_dir)
    rows, _ = oc.compare(None, refs_dir=refs_dir, fits=summaries)
    targets = []
    for cid in TARGETS:
        ref = json.loads((pathlib.Path(refs_dir) / f"{cid}.json").read_text())
        targets.append(build_target(cid, ref, summaries.get(cid), [r for r in rows if r["collection"] == cid]))
    return dict(targets=targets)


def _md_cell(x):
    return str(x).replace("|", "\\|").replace("\n", " ")


def _section(t):
    p = t["paper"]
    arx = f"[arXiv:{p['arxiv']}](https://arxiv.org/abs/{p['arxiv']})" if p.get("arxiv") else (
        f"[doi:{p['doi']}](https://doi.org/{p['doi']})" if p.get("doi") else "")
    lines = [f"## {t['target']}", "", f"{t['instrument']}. Reference: {p['cite']}" + (f", {arx}" if arx else "") + ".", ""]
    if t["status"] == "awaiting":
        lines += ["Awaiting fit: the multi-epoch fit of this collection has not been added yet.", ""]
    else:
        lines += [f"Status: {t['status']} (" + ("orbit: our epoch positions against the first published orbit; "
                                                 if t["kind"] == "orbit" else "") + "scored rows of the table).", ""]
    if t["table"]:
        lines += ["| quantity | virgil | published | pull | status | note |", "|---|---|---|---|---|---|"]
        for r in t["table"]:
            lines.append("| " + " | ".join(_md_cell(x) for x in (
                r["quantity"], f"{oc._fmt(r['ours'], 7)} ± {oc._fmt(r['err'], 2)}",
                f"{oc._fmt(r['pub'], 7)} ± {oc._fmt(r['pub_err'], 2)}", oc._fmt(r["pull"], 3), r["status"],
                r["note"] or "")) + " |")
        lines.append("")
    prov = t["provenance"]
    if prov and any(prov.values()):
        lines += ["Provenance: " + ", ".join(f"{k} `{v}`" for k, v in prov.items() if v) + ".", ""]
    return lines


def page(data):
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/").replace("<!--", "<\\!--")
    lines = ["# Binary orbit gallery", "",
             "Our multi-epoch orbit fits of archival binary data against the published orbits. Generated by "
             "`scripts/orbit_gallery.py render`; the plots are drawn in the browser from the embedded data, the "
             "sections below hold the same numbers without JavaScript.", "",
             '<div class="og"></div>', "",
             f'<script type="application/json" id="og-data">{blob}</script>', ""]
    for t in data["targets"]:
        lines += _section(t)
    return "\n".join(lines).rstrip() + "\n"


def render(out, json_out, summaries_dir=SUMMARIES, refs_dir=REFERENCES):
    data = build(summaries_dir, refs_dir)
    for path, text in ((out, page(data)), (json_out, json.dumps(data, separators=(",", ":")))):
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return data


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="trim fit_<id>.json files to committed summaries")
    e.add_argument("fits_dir")
    e.add_argument("--out", default=str(SUMMARIES))
    r = sub.add_parser("render", help="write the gallery page and its JSON")
    r.add_argument("--out", required=True, help="the Markdown page (docs/orbits/binaries.md)")
    r.add_argument("--json", required=True, dest="json_out", help="the data for the front end (docs/assets/orbits.json)")
    r.add_argument("--summaries", default=str(SUMMARIES))
    r.add_argument("--refs", default=str(REFERENCES))
    a = ap.parse_args(argv)
    if a.cmd == "extract":
        extract(a.fits_dir, a.out)
    else:
        data = render(a.out, a.json_out, a.summaries, a.refs)
        print(f"wrote {a.out} and {a.json_out}: " + ", ".join(f"{t['slug']} {t['status']}" for t in data["targets"]))


if __name__ == "__main__":
    main()
