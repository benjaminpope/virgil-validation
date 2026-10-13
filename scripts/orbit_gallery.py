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
Markdown page: an introduction, how to read a plot, the badges, the data attribution and a collapsible
method note around the interactive gallery, plus one plain-text section per target, readable without
JavaScript. A target without a summary gets an "awaiting" tile, so the page builds before the fits land.
One collection may hold several binaries (the CHARA workshop files): each gets its own tile (``key``, ``rows``).

Units are mas and degrees; sky paths are (dRA, dDec) with dRA east, closed polylines.
"""

import argparse
import collections
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
INSTRUMENT = {"GRAVITY": "VLTI/GRAVITY", "MIRCX": "CHARA/MIRC-X", "PIONIER": "VLTI/PIONIER"}

GL229 = "782185b2-0727-42b0-a185-b2072732b047"
HR6819 = "696baf06-6c3c-424d-abaf-066c3c324d99"
IOTA_PEG = "fac164e1-d9d0-4500-8164-e1d9d0450099"
ASTARS = "bda75673-61c6-49f0-a756-7361c699f0c4"
# The CHARA 2023 imaging workshop collection (L2), cleared for display by the project lead on 2026-10-13
# (DATA_ATTRIBUTION.md)
WORKSHOP = "647a22a9-5047-4220-ba22-a95047022072"

# One tile per binary, in gallery order. ``kind``: orbit = we fit an orbit; positions = epoch positions
# only (a published orbit may still be drawn for context). ``key`` picks one target out of a collection that
# holds several (the fit's ``targets`` and the reference's ``targets``); ``rows`` is the prefix of the
# ``oidb_compare`` row target for that tile; ``pub`` names the published-orbit reader.
# pi1 Gru (19f7e2cf) is a uniform-disk fit to an AGB star, not a binary: no tile;
# HD 45166 (f4afc4cd) is out until its calibration is solved (design/hd45166_diagnosis.md).
TILES = [
    dict(cid=GL229, slug="gl229b", name="Gliese 229 B (Ba-Bb)", kind="orbit", pub="gl229",
         tagline="Two brown dwarfs orbiting each other every 12 days",
         about="Xuan et al. (2024) used VLTI/GRAVITY to split the brown dwarf Gl 229 B into a tight pair, Ba and Bb. "
               "We fit the same nights of closure phases with a Keplerian orbit and draw it against their two orbits "
               "(PMOIRED and Octofitter).",
         notes=[dict(kind="convention", title="Conventions",
                     text="Xuan et al. give ω for the primary (Ba). virgil's ω is the companion's, so ours is the published "
                          "value + 180°. Their semi-major axis is in au; we convert it to mas with the parallax "
                          "(173.574 mas). The published curves drawn here already include both conversions."),
                dict(kind="caveat", title="Visibility contrast",
                     text="GRAVITY's V² loses contrast by a different amount each night, so the fit gives each night its own "
                          "V² amplitude rather than one for the whole run. The flux ratio, read from the same amplitudes, "
                          "depends on it."),
                dict(kind="caveat", title="Radial velocities",
                     text="The paper combines the astrometry with radial velocities; ours is astrometry only. Orbital "
                          "elements are therefore reported, not scored.")]),
    dict(cid=HR6819, slug="hr6819", name="HR 6819", kind="orbit", pub="hr6819",
         tagline="A Be star and a stripped companion on a 40-day orbit",
         about="Klement et al. (2025) combined VLTI/GRAVITY astrometry of the inner pair of HR 6819 with radial velocities "
               "to get dynamical masses. We fit the astrometry alone.",
         notes=[dict(kind="convention", title="Conventions",
                     text="No ω shift is needed. Table 3 of the paper labels its ω as the Be star's, but the number matches "
                          "the companion's ω in Table 2 (astrometry only), and virgil's ω is the companion's too, so the "
                          "published ω is used as it stands (settled in virgil-validation PR #102). The circular-orbit "
                          "curves are drawn for completeness: their T₀ is a time of superior conjunction, and in Table 3 "
                          "it looks half a period away from Table 2's."),
                dict(kind="caveat", title="Not a clean target",
                     text="A Be star's decretion disk adds light that a model of two point stars does not describe. Every "
                          "row is compared but never scored, so the badge reads 'reported'.")]),
    dict(cid=IOTA_PEG, slug="iota-peg", name="ι Peg", kind="orbit", pub="iota",
         tagline="A 10-day spectroscopic binary seen by CHARA/MIRC-X",
         about="Anugu et al. (2020) fitted five nights of CHARA/MIRC-X data on ι Peg (HD 210027) and compared the orbit "
               "with the spectroscopic one of Konacki et al. (2010). We fit the same files.",
         notes=[dict(kind="convention", title="Conventions",
                     text="The paper does not say which component its ω refers to, nor whether T_peri is an MJD or an RJD "
                          "(MJD + 0.5), and its Ω is epoch-of-date rather than J2000. We take ω and T_peri as published, so "
                          "the drawn curves may be out of phase by up to half a period or a degree-scale rotation: they "
                          "are a guide, and the elements are reported, not scored.")]),
    dict(cid=WORKSHOP, key="iot Peg", rows="iota Peg", slug="iota-peg-workshop", name="ι Peg (CHARA workshop)",
         kind="positions", pub="iota",
         tagline="The same star from the CHARA 2023 imaging-workshop files",
         about="The workshop data set holds the 2018-10-22 night of ι Peg, byte-identical to the files of the ι Peg tile. "
               "This tile compares virgil's position on the workshop copy with the paper's. The files are identical to the "
               "ι Peg tile's, so the two tiles should agree exactly; one night is not enough to fit an orbit, so the "
               "published orbit is drawn for context only.",
         notes=[dict(kind="data", title="Unpublished data",
                     text="An L2 (not yet published) collection from the CHARA 2023 imaging workshop, shown here with "
                          "clearance for display (data PI Gail Schaefer).")]),
    dict(cid=WORKSHOP, key="sig Ori", rows="sigma Ori", slug="sigma-ori", name="σ Ori (CHARA workshop)",
         kind="positions", pub="sigori",
         tagline="A pair of O stars on a 143-day eccentric orbit, one night",
         about="Schaefer et al. (2016) measured the σ Ori Aa-Ab orbit from MIRC data. The workshop set holds one night "
               "(2011-09-29), fitted with their scaled-binary model: two point-like stars plus the light of the B companion "
               "as resolved, incoherent flux. With one night there is no orbit to fit; we check that our position lies "
               "on the published orbit.",
         notes=[dict(kind="convention", title="Conventions",
                     text="Schaefer et al. give ω for Aa, the primary, so the companion's is +180° (199.83° becomes "
                          "19.83°). Their time of periastron is HJD − 2400000, converted to MJD by subtracting 0.5 d. "
                          "With both, the published orbit reproduces their 2011-09-29 position (6.29 mas, PA 206.2°)."),
                dict(kind="data", title="Unpublished data",
                     text="An L2 (not yet published) collection from the CHARA 2023 imaging workshop, shown here with "
                          "clearance for display (data PI Gail Schaefer).")]),
    dict(cid=ASTARS, slug="a-stars", name="A-star companions (5 detections)", kind="positions", pub=None,
         tagline="Faint companions to nearby A stars, one night each",
         about="De Furio et al. (2022) searched A stars for companions with CHARA/MIRC-X. We re-fit their five detections "
               "with a CANDID-style binary model (a uniform-disk primary, a companion and resolved flux). Each star has a "
               "single night, so there is no orbit: the plot compares our companion positions with the paper's.",
         notes=[dict(kind="caveat", title="Not fitted",
                     text="The 22 non-detections (detection limits) are not part of this comparison.")]),
]
TARGETS = {t["cid"]: t for t in TILES}  # the collections the gallery shows
# Single-night alias peaks (design/orbit_comparison.md section 5): a fit summary names them in
# ``orbit["epoch_checks"]`` (reasons starting "flagged"); for a summary that predates that field, these nights
# are flagged by label (Xuan et al. flag the same two, as 02-27 and 03-28).
ALIAS_NIGHTS = {GL229: {"2024-02-28", "2024-03-29"}}
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
    out = _keep(o, ("t_ref", "map", "loss", "chi2_raw", "epoch_checks"))
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
                                        if t.get("variants") else {}),
                                     **({"orbit": trim_orbit(t["orbit"])} if t.get("orbit") else {}))
                          for name, t in fit["targets"].items()}
    return out


def extract(fits_dir, out_dir=SUMMARIES):
    """One ``<collection_id>.json`` summary per gallery ``fit_*.json`` in ``fits_dir``. A collection with several
    targets (the workshop's iota Peg and sigma Ori) stays one file; ``render`` splits it into tiles."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {c["id"]: c for c in load_manifest()["collections"]}
    written = []
    for path in sorted(pathlib.Path(fits_dir).glob("fit_*.json")):
        fit = json.loads(path.read_text())
        cid = fit.get("collection") or path.stem[len("fit_"):]
        if cid not in TARGETS:  # collections the gallery does not show
            print(f"skip {path.name}: not a gallery target")
            continue
        prov = fit.get("provenance") or {}
        c = manifest.get(cid, {})
        summary = dict(collection_id=cid, target=" + ".join(t["name"] for t in TILES if t["cid"] == cid),
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


def published_orbits(pub, ref):
    """[(label, elements)] of the published orbits, headline first, in virgil's convention: the
    companion's omega (the primary's + 180, as ``oidb_compare.orbit_rows``) and a in mas."""
    if pub == "gl229":
        out = []
        for label, key in (("PMOIRED (Xuan+24)", "pmoired_frequentist"), ("Octofitter (Xuan+24)", "octofitter_bayesian")):
            s = ref["orbit"][key]
            out.append((label, dict(period_day=s["P_day"][0], t_peri_mjd=s["T_peri_mjd"][0], ecc=s["e"][0],
                                    inc_deg=s["i_deg"][0], omega_deg=(s["omega_primary_deg"][0] + 180) % 360,
                                    Omega_deg=s["Omega_deg"][0] % 360,
                                    a_mas=s["a_au"][0] * oc.CRITERIA["gl229_parallax_mas"])))
        return out
    if pub == "hr6819":
        shift = ref["conventions"]["omega_shift_deg"]  # 0: Table 3's omega is the companion's (PR #102)
        sols = ref["orbit"]["all_solutions"]
        table3 = next(v for k, v in sols.items() if k.startswith("Table 3"))
        out = []
        # Table 3 columns: P, T0 (RJD = MJD + 0.5), e, a (mas), i, Omega, omega (labelled the Be star's, but the companion's)
        for key in ("eccentric_astro+RVs", "eccentric_astro+RVpre-sd+K_Be", "circular_astro+RVs",
                    "circular_astro+RVpre-sd+K_Be"):
            c = table3[key]
            out.append((key.replace("_", " ") + " (Klement+25)",
                        dict(period_day=c[0][0], t_peri_mjd=c[1][0] - 0.5, ecc=c[2][0], inc_deg=c[4][0],
                             omega_deg=(c[6][0] + shift) % 360, Omega_deg=c[5][0] % 360, a_mas=c[3][0])))
        astro = next(v for k, v in sols.items() if k.startswith("astrometry_only"))
        # Table 2's omega is the pre-sd companion's, like Table 3's: no shift
        for key in ("eccentric", "circular"):
            c = astro[key]
            out.append((f"{key} astrometry only (Klement+25)",
                        dict(period_day=c["P_day"][0], t_peri_mjd=c["T0_RJD"][0] - 0.5, ecc=_num(c["e"]),
                             inc_deg=c["i_deg"][0], omega_deg=(c["omega_deg"][0] if isinstance(c["omega_deg"], list)
                                        else c["omega_deg"]) % 360, Omega_deg=c["Omega_deg"][0] % 360, a_mas=c["a_mas"][0])))
        return out
    if pub == "iota":
        o = ref["orbit"]
        out = [("Anugu+20 (omega component / T system unstated)", dict(period_day=o["P_day"][0], t_peri_mjd=o["T_peri_unstated_system"][0], ecc=o["e"][0],
                                 inc_deg=o["i_deg"][0], omega_deg=o["omega_deg"][0] % 360,
                                 Omega_deg=o["Omega_deg"][0] % 360, a_mas=o["a_mas"][0]))]
        k = o["comparison_orbit_Konacki2010"]
        out.append(("Konacki+10", dict(period_day=k["P_day"][0], t_peri_mjd=k["T"][0], ecc=k["e"][0],
                                       inc_deg=k["i_deg"][0], omega_deg=k["omega_deg"][0] % 360,
                                       Omega_deg=k["Omega_deg"][0] % 360, a_mas=k["a_mas"][0])))
        return out
    if pub == "sigori":  # Schaefer+16 Table 9: omega is Aa's (the primary's): + 180; T is HJD - 2400000
        o = ref["orbit"]
        return [("Schaefer+16", dict(period_day=o["P_day"][0], t_peri_mjd=o["T_peri_hjd_minus_2400000"][0] - 0.5,
                                     ecc=o["e"][0], inc_deg=o["i_deg"][0], omega_deg=(o["omega_deg"][0] + 180) % 360,
                                     Omega_deg=o["Omega_deg"][0] % 360, a_mas=o["a_mas"][0]))]
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
        return [dict(label=p["name"], mjd=p["mjd"], dra=p["dra"], ddec=p["ddec"], cov=p.get("cov"),
                     **_keep(p, ("alias", "note"))) for p in summary["orbit"]["positions"]]
    if cid == ASTARS:  # the detections: one epoch per star
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
    ell = r.get("error_ellipse")
    if ell:  # sigma Ori: the published ellipse over its chi2 inflation, as oidb_compare
        k = oc.CRITERIA["sigma_ori_ellipse_scale"]
        cov = oc.ellipse_cov(ell["sigma_maj_mas"] / k, ell["sigma_min_mas"] / k, ell["phi_deg"]).tolist()
    return dict(dra=dra, ddec=ddec, cov=cov)


def published_epochs(tile, ref, ours, headline):
    """Published position for each of our epochs (None where the paper has no matching one)."""
    if tile["pub"] == "gl229":  # the PMOIRED orbit at our epoch times
        return [dict(zip(("dra", "ddec"), (float(x) for x in np.ravel(_orbit_at(headline, e["mjd"]))))) for e in ours]
    if tile["cid"] == ASTARS:
        by_star = {r["star"]: dict(r, rho_mas=[r["rho_mas"], r["rho_err_mas"]], pa_deg=[r["pa_deg_E_of_N"], r["pa_err_deg"]])
                   for r in ref["epochs"]}
        return [published_position(by_star[e["label"]]) if e["label"] in by_star else None for e in ours]
    out = []
    for e in ours:
        probe = oc.by_epoch([dict(label=e["label"], mjd=e["mjd"])], _ref_epochs(ref))
        r = next((r for r, f in probe if f), None)
        out.append(published_position(r) if r else None)
    return out


def _ref_epochs(ref):
    """The reference's epochs: its ``epochs`` list, or the ``epoch_<date>`` entries of a one-night target."""
    return ref.get("epochs") or [v for k, v in ref.items() if k.startswith("epoch_") and isinstance(v, dict)]


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
    if not ref or "paper" not in ref:
        return None
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
        note = "; ".join(x for x in r["note"].split("; ") if not x.startswith("L2: contact"))
        out.append(dict(quantity=q, ours=r["virgil"], err=r["virgil_err"], pub=r["published"],
                        pub_err=max(abs(x) for x in pe) if isinstance(pe, (list, tuple)) else pe,
                        pull=r["dev_sigma"], status=r["status"], note=note))
    return out


def _phase_of(el, mjd):
    return ((mjd - el["t_peri_mjd"]) / el["period_day"]) % 1.0


def status_of(table, summary):
    """(status, counts, sentence) from the table rows. The badge follows the scored rows only: agrees needs
    one PASS and no FAIL/MISSING; tension a FAIL/MISSING; rows ``oidb_compare`` keeps as REPORT/NOT-CLEAN (orbit
    elements, per-night positions, iota Peg's unstated conventions) give "reported"."""
    if summary is None:
        return "awaiting", dict(passed=0, failed=0, reported=0), ""
    n = collections.Counter(r["status"] for r in table)
    failed = [r for r in table if r["status"] in ("FAIL", "MISSING")]
    counts = dict(passed=n["PASS"], failed=len(failed), reported=len(table) - n["PASS"] - len(failed))
    scored = counts["passed"] + counts["failed"]
    if failed:
        names = ", ".join(r["quantity"] + (f" ({r['pull']:+.1f}σ)" if r["pull"] is not None else " (missing)") for r in failed[:3])
        why = f"{counts['failed']} of {scored} scored comparisons fail: {names}" + (" and more" if len(failed) > 3 else "") + "."
        return "tension", counts, why
    if counts["passed"]:
        return "agrees", counts, f"All {scored} scored comparisons pass (limits fixed before the fits were run)."
    return "reported", counts, "Nothing is scored for this target: its comparisons are reported only."


def _split_summary(tile, summaries):
    """The fit summary of one tile: the collection's, or the ``targets[key]`` part of it, which gets the
    collection's provenance and instrument."""
    s = summaries.get(tile["cid"])
    if s is None or not tile.get("key"):
        return s
    sub = (s.get("targets") or {}).get(tile["key"])
    return None if sub is None else dict(sub, provenance=s.get("provenance", {}), instrument=s.get("instrument"))


def _load_ref(tile, refs_dir):
    path = pathlib.Path(refs_dir) / f"{tile.get('ref') or tile['cid']}.json"
    if not path.exists():
        return None
    ref = json.loads(path.read_text())
    return ref.get("targets", {}).get(tile["key"]) if tile.get("key") else ref


def _data_of(tile, manifest):
    c = manifest.get(tile["cid"], {})
    level = "L" + str(max(c["levels"])) if c.get("levels") else None
    return dict(level=level, datapi=", ".join(c.get("datapis", [])) or None, title=c.get("title"),
                collection=tile["cid"])


def build_target(tile, ref, summary, rows, manifest=None):
    manifest = manifest if manifest is not None else {c["id"]: c for c in load_manifest()["collections"]}
    pubs = published_orbits(tile["pub"], ref) if ref and tile["pub"] else []
    mc = manifest.get(tile["cid"], {})
    t = dict(slug=tile["slug"], collection_id=tile["cid"], target=tile["name"], tagline=tile["tagline"],
             about=tile["about"], notes=tile["notes"], data=_data_of(tile, manifest),
             instrument=(summary or {}).get("instrument") or " + ".join(
                 INSTRUMENT.get(i, i) for i in mc.get("instruments", [])),
             kind=tile["kind"], status="awaiting", awaiting=None if summary is not None else "fit",
             paper=paper_of(ref), ours=None, alternatives=[], published=[], epochs=[],
             phase=dict(ours=[], published=[]), table=[], provenance={}, scored=None, status_why="")
    if ref is None or t["paper"] is None:
        t["awaiting"] = "reference"
    ep_mjds, ours_ep = [], []
    if summary is not None:
        t["provenance"] = summary.get("provenance", {})
        ours_ep = ours_epochs(tile["cid"], summary)
        ep_mjds = [e["mjd"] for e in ours_ep if e.get("mjd") is not None]
        pub_ep = published_epochs(tile, ref, ours_ep, pubs[0][1] if pubs else None) if ref else [None] * len(ours_ep)
        t["epochs"] = []
        checks = (summary.get("orbit") or {}).get("epoch_checks")
        alias_labels = ({c["name"] for c in checks if any(str(r).startswith("flagged") for r in c.get("reasons", []))}
                        if checks is not None else ALIAS_NIGHTS.get(tile["cid"], ()))
        for e, p in zip(ours_ep, pub_ep):
            sep, pa = _sep_pa(e["dra"], e["ddec"])
            d = dict(label=e["label"], mjd=e["mjd"], dra=e["dra"], ddec=e["ddec"], cov=e["cov"], sep=sep, pa=pa)
            if p:
                d["pub"] = p
            if e.get("alias") or e["label"] in alias_labels:
                d["alias"] = True
            if e.get("note"):
                d["note"] = e["note"]
            t["epochs"].append(d)
        t["table"] = table_of(rows) if ref else []
        orb = summary.get("orbit")
        if tile["kind"] == "orbit" and orb:
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
    # the orbit the epoch points are placed on in the phase panels: ours, else the headline published one
    on = (t["ours"] or {}).get("elements") or (pubs[0][1] if pubs else None)
    if on:
        for e in t["epochs"]:
            if e.get("mjd") is not None:
                e["phase"] = _phase_of(on, e["mjd"])
        t["phase_on"] = "ours" if t["ours"] else "published"
    if summary is not None and ref is not None:
        t["status"], t["scored"], t["status_why"] = status_of(t["table"], summary)
        t["tension_sigma"] = _tension(ours_ep, pubs[0][1]) if pubs and tile["kind"] == "orbit" else 0.0
    elif summary is not None:
        t["status"] = "awaiting"
    t["awaiting"] = None if t["status"] != "awaiting" else (t["awaiting"] or "fit")
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
    manifest = {c["id"]: c for c in load_manifest()["collections"]}
    rows, _ = oc.compare(None, refs_dir=refs_dir, fits=summaries)
    targets = []
    for tile in TILES:
        mine = [r for r in rows if r["collection"] == tile["cid"]
                and (not tile.get("rows") or r["target"].startswith(tile["rows"]))]
        targets.append(build_target(tile, _load_ref(tile, refs_dir), _split_summary(tile, summaries), mine, manifest))
    return dict(targets=targets, guide=GUIDE, badges=badges())


# ----------------------------------------------------------------------------- the page text

# How to read a plot: one source for the page and for the help inside each detail view (docs/assets/orbits.js).
GUIDE = [
    dict(key="sky", title="The sky plane",
         text="North is up and east is to the left, as on the sky, so ΔRA increases towards the left. The primary star is "
              "the gold dot at the origin, both axes are in milliarcseconds (mas), and the scale bar gives the length on "
              "the sky."),
    dict(key="ours", title="virgil's orbit",
         text="The solid blue line is our maximum-a-posteriori (MAP) orbit: the single best Keplerian orbit fitted to "
              "all the nights at once."),
    dict(key="pub", title="Published orbits",
         text="Dashed lines are the orbits in the paper, converted to virgil's conventions (see each target's notes). "
              "Where the paper gives several solutions, the headline one is orange and the others follow in other hues."),
    dict(key="alt", title="Alternative modes",
         text="The faint grey lines are other local optima that the multi-start search found in the likelihood, each "
              "refitted to convergence. Δloss is how much higher its loss (the negative log posterior) is than the "
              "MAP's: a few units cannot be ruled out, hundreds are clearly worse. They are shown because an orbit fit "
              "can lock onto an alias (a different number of cycles between nights) or a mirror image, and the reader "
              "should see what else the data allow."),
    dict(key="eps", title="Epoch points",
         text="Each filled blue dot is the companion's position on one night, from fitting that night alone, with its "
              "1σ error ellipse. Open orange rings are the paper's own position for that night, where it gives one. "
              "Hover over a dot, or tab to it, for the date, separation and position angle."),
    dict(key="alias", title="Alias nights",
         text="A cross with a dashed ellipse marks a night whose best single-night peak is a known alias: a secondary "
              "maximum of the likelihood, not the companion's true position. Such nights are flagged, not used as "
              "constraints, and are not expected to sit on the orbit."),
    dict(key="phase", title="Phase panels",
         text="Separation and position angle against orbital phase, with phase 0 at periastron. Each curve is folded on "
              "its own period; the dots are our nights placed on our orbit. A target without an orbit shows the "
              "published orbit if there is one, else the positions against date."),
]


def badges():
    """What each badge means, with the pass limits read from ``oidb_compare.CRITERIA`` so they cannot go stale."""
    ep, par = oc.CRITERIA["epoch"]["max_dev_sigma"], oc.CRITERIA["parametric"]["max_dev_sigma"]
    items = [
        dict(key="agrees", title="agrees",
             text="At least one comparison was scored and every scored comparison passed."),
        dict(key="tension", title="tension",
             text="At least one scored comparison failed, or virgil did not deliver a quantity that is scored. Not a "
                  "verdict on virgil: a convention, the data or the model may be at fault, and the notes say what is known."),
        dict(key="reported", title="reported",
             text="Nothing is scored. Everything is compared and shown, but not as a pass or fail: orbital elements, "
                  "because the papers fit radial velocities that we do not; per-night positions with unstated "
                  "conventions; or a target the plan lists as not clean (HR 6819's Be-star disk)."),
        dict(key="awaiting", title="awaiting fit",
             text="The fit, or the published reference, is not in yet. The tile shows whatever exists, such as the "
                  "published orbit."),
    ]
    return dict(items=items, pull=(
             f"Each row of a table is one quantity, with virgil's value and its error beside the paper's. The pull is "
             f"(virgil − paper) divided by the paper's quoted error on the side facing virgil. The pass limits were "
             f"fixed before any fit was run: |pull| ≤ {ep:g} for per-night positions and flux ratios, which come from "
             f"the very same files and should nearly agree, and ≤ {par:g} for diameters and flux fractions. In the "
             f"tables the pull is shaded by size: below 1σ, 1 to 3σ, above 3σ."),
        criteria=getattr(oc, "CRITERIA_HASH", None))


def _md_cell(x):
    return str(x).replace("|", "\\|").replace("\n", " ")


def _paper_link(p):
    if not p:
        return ""
    if p.get("arxiv"):
        return f"[arXiv:{p['arxiv']}](https://arxiv.org/abs/{p['arxiv']})"
    return f"[doi:{p['doi']}](https://doi.org/{p['doi']})" if p.get("doi") else ""


def _section(t):
    """One target as plain Markdown, the part of the page that reads without JavaScript."""
    p = t["paper"]
    arx = _paper_link(p)
    ref = f"Reference: {p['cite']}" + (f", {arx}" if arx else "") if p else "Reference: not yet added"
    lines = [f"**{t['target']}**", "", f"{t['instrument']}. {t['tagline']}. {ref}.", "", t["about"], ""]
    if t["status"] == "awaiting":
        lines += [f"Awaiting {t.get('awaiting') or 'fit'}: it has not been added yet.", ""]
    else:
        lines += [f"Status: {t['status']}. {t['status_why']}", ""]
    for n in t["notes"]:
        lines += [f"- **{n['title']}.** {n['text']}"]
    lines += [""] if t["notes"] else []
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
        lines += ["Provenance: " + ", ".join(f"{k.replace('_', ' ')} `{str(v)[:10]}`" for k, v in prov.items() if v) + ".", ""]
    return lines


INTRO = """\
This gallery asks one question of [virgil](https://github.com/benjaminpope/virgil), our interferometry package: if we \
give it the archival data behind a published binary orbit, does it get the authors' answer? Each tile is one binary; \
open it for the orbit on the sky, the separation and position angle through the orbit, and a table comparing every \
fitted quantity with the paper.

**Why this tests virgil.** The data are the authors' own, so a difference comes from the analysis or from calibration \
choices the archive does not record: the model, the likelihood, the optimizer, a convention such as which star an angle \
belongs to, or the calibrators used. Agreement with the authors' independent pipelines on their own data is evidence \
that virgil's orbit and likelihood machinery works; disagreement is a lead to chase, and the notes under each target \
say what is known. Nothing is tuned towards the published numbers. The data come from the JMMC \
OiDB database ([attribution below](#where-the-data-come-from)).
"""

DATA = """\
Calibrated OIFITS files from the Jean-Marie Mariotti Center's optical interferometry database, \
[OiDB](https://oidb.jmmc.fr), used under its terms (CC BY-NC-SA 4.0). For published data (calibration level L3) we \
thank the dataPI and cite the paper, which every tile links. This research has made use of the Jean-Marie Mariotti \
Center OiDB service, available at <http://oidb.jmmc.fr>. The collections and their terms are in \
[`oidb/manifest.yml`](https://github.com/benjaminpope/virgil-validation/blob/main/oidb/manifest.yml) and \
[`DATA_ATTRIBUTION.md`](https://github.com/benjaminpope/virgil-validation/blob/main/DATA_ATTRIBUTION.md). The two \
workshop tiles come from an unpublished (L2) collection and are shown with clearance for display.
"""

METHOD = """\
For each collection virgil fits the calibrated OIFITS files (closure phases, and V² where they are usable). A \
multi-start search first finds the peaks of the likelihood on each night, then each mode is refitted as one Keplerian \
orbit across all nights, and the lowest loss is the MAP. Where sampling is run, NUTS gives the posterior medians and \
errors in the tables.

The curves are drawn from `crosscheck.orbits`, our own independent Kepler solver, never from virgil's code and never \
in the browser. The published orbits are converted to virgil's conventions first. The tables come from \
`scripts/oidb_compare.py`, whose pass limits were fixed before the fits.

Related pages: [Against orbitize!](orbitize.md) (virgil's Kepler code against orbitize!), \
[Multi-epoch orbits](epochs.md) (starting and scoring multi-night fits) and [Period aliases](aliases.md) (the cycle-count \
ambiguity behind the alias nights).
"""


def page(data):
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/").replace("<!--", "<\\!--")
    bd = data.get("badges") or badges()
    lines = ["# Binary orbit gallery", "", '<div class="og-intro" markdown>', "", INTRO.rstrip(), "", "</div>", "",
             '<div class="og"></div>', "",
             f'<script type="application/json" id="og-data">{blob}</script>', "",
             '<div class="og-about" markdown>', "", "## How to read a plot", ""]
    lines += [f"- **{g['title']}.** {g['text']}" for g in data["guide"]]
    lines += ["", "## The badges", "",
              "Each tile carries one badge, which comes from the comparison tables of `scripts/oidb_compare.py`:", ""]
    lines += [f"- **{b['title']}.** {b['text']}" for b in bd["items"]]
    lines += ["", bd["pull"], "", "## Where the data come from", "", DATA.rstrip(), "", "## Method", ""]
    lines += ['??? note "How the fits and the comparison are made"', ""]
    lines += [("    " + x).rstrip() for x in METHOD.rstrip().splitlines()]
    lines += ["", "</div>", "", '<div class="og-static" markdown>', "", "## The numbers without JavaScript", "",
              "The same tables as text, for readers without JavaScript.", ""]
    for t in data["targets"]:
        lines += _section(t)
    lines += ["</div>"]
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
