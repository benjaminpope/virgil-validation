"""Build the committed Track B orbit-target records, ``oidb/trackb/<slug>.json``.

    python scripts/build_trackb_targets.py [--trackb ~/data/eso_binaries/trackb] [--only ll_aqr ...]

One record per binary that a blind orbit solver (design/orbit_solver_v2.md) will be scored on. Each holds the
published elements and their conventions, the published per-epoch positions, and every ESO Phase 3 OIFITS file with its
provenance and its role in the held-out split. Nothing is fitted here.

Sources (never typed from memory):

* Gallenne et al. 2023 (arXiv:2302.12960) elements and positions: VizieR ``J/A+A/672/A119/table6`` and ``tableb1``,
  fetched live. The diameters (Table 3) and the kappa Vel and NN Del elements and positions are read from the Track B
  ``catalogue/`` and ``system.json`` transcriptions, which cite their tables.
* ESO Phase 3 products: ``ivoa.ObsCore`` by cone search (``dataproduct_type='visibility'``), with the file sizes and release
  dates it reports. Files are fetched by ``dataportal.eso.org/dataPortal/file/<dp_id>`` and checked by sha256.

The held-out split (``role`` of each file):

* ``paper``: a product whose start time is within 0.03 d of a published epoch (the data behind a published position);
* ``paper_night``: another product of a night with a published epoch (same night, so not an independent prediction);
* ``heldout``: a product of a night the paper has no epoch for. ``after_paper_span`` marks those later than the last
  published epoch; the others fall inside the published span or before it (earlier programmes).

A night is ``floor(MJD - 0.5)``. The blind solver is fitted on ``paper`` (and ``paper_night``) nights only and scored on
what the elements it finds predict for the ``heldout`` nights.
"""

import argparse
import csv
import datetime
import hashlib
import io
import json
import math
import pathlib
import sys

import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "oidb" / "trackb"
TRACKB = pathlib.Path.home() / "data" / "eso_binaries" / "trackb"
VIZIER = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
ESO_TAP = "https://archive.eso.org/tap_obs/sync"
MATCH_DAYS = 0.03  # a product this close to a published epoch is that epoch's data
LAMBDA_2B_MAS = {"GRAVITY": 1.75, "PIONIER": 1.31}  # K at B = 130 m, H at B = 130 m (design/trackb_criteria.md)

GALLENNE23 = dict(authors="Gallenne, A.; Mérand, A.; Kervella, P.; et al.", year=2023, journal="A&A 672, A119",
                  bibcode="2023A&A...672A.119G", arxiv="2302.12960", doi="10.1051/0004-6361/202245712")

# slug: (VizieR System name, Track B catalogue slug, kind). kind: sb2_subresolution = the Araucaria-I SB2s whose
# separations are mostly below lambda/2B; alias_track = already in Track B, later epochs held out; short_arc = the data
# span less than one period.
GALLENNE_SYSTEMS = {
    "ll_aqr": ("LL Aqr", "ll_aqr", "sb2_subresolution"),
    "hd_224974": ("HD 224974", "hd_224974", "sb2_subresolution"),
    "ak_for": ("AK For", "ak_for", "sb2_subresolution"),
    "v963_cen": ("V963 Cen", "v963_cen", "sb2_subresolution"),
    "hd_9312": ("HD 9312", "hd_9312", "sb2_subresolution"),
    "hd188088": ("HD 188088", "hd_188088", "alias_track"),
    "hd70937": ("HD 70937", "hd_70937", "alias_track"),
    "hd210763": ("HD 210763", "hd_210763", "alias_track"),
    "hd41255": ("HD 41255", "hd_41255", "alias_track"),
    "omi_leo": ("o Leo", "o_leo", "alias_track"),
}
SHORT_ARCS = {"kap_vel": "short_arc", "nn_del": "short_arc"}
SHORT_ARC_PAPERS = {
    "kap_vel": dict(authors="Rowan, D. M.; Kraus, S.; Thompson, T. A.", year=2026, journal="PASP 138, 024203",
                    bibcode="2026PASP..138b4203R", arxiv="2512.02116", doi="10.1088/1538-3873/ae3b75"),
    "nn_del": dict(authors="Gallenne, A.; Mérand, A.; Kervella, P.; et al.", year=2019, journal="A&A 632, A31",
                   bibcode="2019A&A...632A..31G", arxiv="1910.03393", doi="10.1051/0004-6361/201935837"),
}
# Time of periastron as the paper prints it -> JD. NN Del: "Tp: time passage through periastron (-2445000)", read as JD - 2445000
# and confirmed against the paper's own STELLA RVs (rms 0.10 km/s; a 0.5 d shift gives 1.4 km/s). kappa Vel: the paper
# gives T0 = 73.1 d with no epoch; its RV table is on JD - 2416500 and that offset (alone) fits the table's 33 RVs.
T_PERI_OFFSET_JD = {"nn_del": 2445000.0, "kap_vel": 2416500.0}

_DEGENERATE = (" Without radial velocities the sky positions are unchanged by (omega, Omega) -> (omega + 180, Omega + 180) "
               "degrees, so a blind astrometric solver's omega and Omega can only be compared with the published pair "
               "jointly, modulo that shift (the companion-convention pair is the published one plus 180 degrees in both).")
_CONV_COMMON = (
    "Omega is the position angle of the ascending node, measured east of north, with the equinox not stated; "
    "the time scale of T_peri (UTC, HJD or BJD) is not stated. Positions are the secondary relative to the primary "
    "(east and north positive). "
)
CONVENTIONS = {
    "gallenne2023": dict(
        omega_shift_deg=0.0,
        text=("Elements from Gallenne et al. 2023, Table 6 (joint radial-velocity and astrometry fit). omega is the "
              "primary's (the RV model is V1 = gamma + K1[cos(omega + nu) + e cos omega], V2 with -K2); T_peri is a full JD "
              "(here also as MJD = JD - 2400000.5). " + _CONV_COMMON + "Checked against the paper's own tables: the "
              "tabulated omega and Omega, used as they stand in the standard Campbell formulas, reproduce the published "
              "positions (Table B.1); changing either alone by 180 degrees does not. (Equivalent to the companion's "
              "omega + 180 degrees with Omega + 180 degrees.) Where e is near zero (AK For, o Leo) omega and T_peri are "
              "poorly determined."),
        omega_primary=True, degenerate_shift_deg=180.0,
        check="positions of Table B.1 reproduced to 0.01-0.4 mas (0.4 for AK For, e = 0.0002)"),
    "nn_del": dict(
        omega_shift_deg=0.0,
        text=("Elements from Gallenne et al. 2019, Table 2 (joint fit of PIONIER astrometry and STELLA radial velocities). "
              "omega is the primary's (RV model V1 = gamma + K1[cos(omega + nu) + e cos omega]; the V2 sign of K2 is "
              "opposite although the typeset formula does not show it). T_peri is printed as 5235.164 d with the note "
              "'-2445000': read as JD - 2445000. " + _CONV_COMMON + "The paper's four positions are reproduced to about "
              "0.003 mas by the standard Campbell formulas with these omega and Omega."),
        omega_primary=True, degenerate_shift_deg=180.0, check="the four Table 1 positions reproduced to about 0.003 mas"),
    "kap_vel": dict(
        omega_shift_deg=0.0,
        text=("Elements from Rowan, Kraus and Thompson 2026, Table 4 (joint SB1 radial-velocity and GRAVITY astrometry "
              "fit). omega is the photometric primary's. T0 (73.1 d) is printed without an epoch: JD - 2416500 is the "
              "scale of the paper's RV table and the only offset that fits it, so T_peri here is derived, not stated. "
              "The paper's Omega and i are in a form (Thiele-Innes constants with x = -A(cosE - e) - F sqrt(1-e^2) sinE) "
              "that does not give the published positions in the standard east-of-north Campbell formulas; "
              "the equivalent standard elements, found here by comparing the paper's Eqs. 6-7 with the standard form and then tried "
              "against Table 3 (so that check tests this mapping and is not independent of it), are Omega = 270 - 109.74 = "
              "160.26 degrees and i = 180 - 74.03 = 105.97 degrees (retrograde, as the position angles run). "
              "Use 'standard' for any comparison; the published values stay in 'orbit'. " + _CONV_COMMON),
        omega_primary=True, degenerate_shift_deg=180.0, standard=dict(Omega_deg=270.0 - 109.74, i_deg=180.0 - 74.03),
        check="Table 3 positions reproduced to 0.02-0.06 mas for 8 of 9 epochs; the ninth (2025-02-28) is flagged suspect"),
}


# Products that virgil cannot load alone (scripts/trackb_smoke.py, Track B recipe); the blind solver skips them.
UNREADABLE = {
    "ADP.2026-03-04T16:19:03.401": "no observables left after the Brgamma window and flag cuts (every closure phase flagged or NaN)",
    "ADP.2026-03-04T16:19:04.655": "closure-phase triangle (24, 23, 1) has a leg with no visibility row and zero OI_T3 coordinates",
    "ADP.2026-03-04T16:19:04.805": "no observables left after the Brgamma window and flag cuts (every closure phase flagged or NaN)",
    "ADP.2026-03-04T16:19:05.376": "no observables left after the Brgamma window and flag cuts (every closure phase flagged or NaN)",
}


# Published positions that the paper's own elements and the data disagree with. Kept as printed, flagged, and left out of
# the convention check (tests/test_trackb_targets.py).
SUSPECT = {("kap_vel", "2025-02-28"): (
    "Rowan+2026 Table 3 prints dRA = +0.510 mas under the date 2025-02-28, but its Table 1 gives MJD 60731.05 for that "
    "epoch, which is the night of 2025-02-24 (the date label and the MJD disagree). The paper's own orbit predicts about "
    "-0.53 mas, a blind single-night fit of the Phase 3 file gives about -0.51 mas, and dRA is falling through the "
    "neighbouring epochs: a sign error in the printed value is likely. The files are matched by MJD, so the data are not affected.")}


def _pair(x):
    return [x[0], x[1]] if x else None


def short_arc_orbit(slug, ro):
    """Track B ``reference_orbit`` ([value, error, note]) -> the record's orbit dict; T_peri in JD and MJD."""
    o = {k: _pair(ro[k]) for k in ("P_day", "e", "omega_deg", "Omega_deg", "i_deg", "a_mas", "K1")}
    o["K2"] = _pair(ro.get("K2"))
    for k, name in dict(parallax_mas="parallax_mas", M1_Msun="M1_Msun", M2_Msun="M2_Msun").items():
        o[k] = _pair(ro.get(name))
    t0 = ro["T0"]
    jd = t0[0] + T_PERI_OFFSET_JD[slug]
    o["T_peri_jd"] = [jd, t0[1]]
    o["T_peri_mjd"] = [jd - 2400000.5, t0[1]]
    o["source"] = ro["source"]
    return o


def short_arc_positions(s):
    out = []
    for r in s["reference_positions"]:
        out.append(dict(
            date=r["date"], mjd=r["mjd"], dra_mas=r["dra_mas"], ddec_mas=r["ddec_mas"],
            sigma_major_mas=r.get("sigma_major_mas"), sigma_minor_mas=r.get("sigma_minor_mas"),
            sigma_pa_deg=r.get("sigma_pa_deg"), flux_ratio_pct=r["flux_ratio"] * (100 if r["flux_ratio_unit"].startswith("ratio") else 1),
            sigma_flux_ratio_pct=r.get("sigma_flux_ratio") and r["sigma_flux_ratio"] * (100 if r["flux_ratio_unit"].startswith("ratio") else 1),
            note=r.get("note", ""), sep_mas=math.hypot(r["dra_mas"], r["ddec_mas"])))
        if (s["target"].split()[0].lower() == "kappa" and ("kap_vel", r["date"]) in SUSPECT):
            out[-1]["suspect"] = SUSPECT[("kap_vel", r["date"])]
    return out

def vizier(query):
    r = requests.get(VIZIER, params=dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=query), timeout=120)
    r.raise_for_status()
    return list(csv.DictReader(io.StringIO(r.text)))


def eso_products(ra, dec, radius_deg=0.005):
    q = ("SELECT dp_id, instrument_name, t_min, t_max, obs_release_date, proposal_id, access_estsize, "
         "dataproduct_subtype, obs_id FROM ivoa.ObsCore WHERE dataproduct_type='visibility' "
         "AND instrument_name IN ('GRAVITY','PIONIER') "
         f"AND CONTAINS(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))=1 ORDER BY t_min")
    r = requests.get(ESO_TAP, params=dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=q), timeout=120)
    r.raise_for_status()
    return list(csv.DictReader(io.StringIO(r.text)))


def night(mjd):
    return math.floor(mjd - 0.5)


def pair(row, k):
    v, e = row[k], row.get("e_" + k)
    return [float(v), float(e) if e not in (None, "") else None]


def gallenne_orbit(row):
    """Table 6 row -> the orbit dict, elements as published (omega is the RV primary's; T_peri is in JD)."""
    o = {k: pair(row, c) for k, c in dict(P_day="Prob", T_peri_jd="Tp", e="e", K1="K1", K2="K2", omega_deg="omega",
                                          Omega_deg="Omega", a_mas="a", i_deg="incl", parallax_mas="plx",
                                          M1_Msun="M1", M2_Msun="M2").items()}
    o["T_peri_mjd"] = [o["T_peri_jd"][0] - 2400000.5, o["T_peri_jd"][1]]
    o["source"] = f"VizieR J/A+A/672/A119/table6, row '{row['System'].strip()}' (Table 6; joint RV + astrometry)"
    return o


def gallenne_positions(rows):
    out = []
    for r in rows:
        dra, dde = float(r["DRA"]), float(r["DDE"])
        out.append(dict(
            date=_date(float(r["MJD"])), mjd=float(r["MJD"]), dra_mas=dra, ddec_mas=dde,
            sigma_major_mas=float(r["smaj"]), sigma_minor_mas=float(r["smin"]), sigma_pa_deg=float(r["sPA"]),
            flux_ratio_pct=float(r["f"]), sigma_flux_ratio_pct=float(r["e_f"]) if r["e_f"] else None,
            baselines=r["Baselines"].strip(), spectral_resolution=r["SpRes"].strip(), calibrator_id=int(r["Cal"]),
            sep_mas=math.hypot(dra, dde)))
    return out


def _date(mjd):
    return (datetime.datetime(1858, 11, 17) + datetime.timedelta(days=mjd)).strftime("%Y-%m-%d")


def classify(products, pub_mjds):
    """Role of each ObsCore product against the published epoch times."""
    pub_nights = {night(m) for m in pub_mjds}
    last = max(pub_mjds)
    items = []
    for r in products:
        t = float(r["t_min"])
        dm, idx = min((abs(t - m), i) for i, m in enumerate(pub_mjds))
        role = "paper" if dm < MATCH_DAYS else ("paper_night" if night(t) in pub_nights else "heldout")
        items.append(dict(
            dp_id=r["dp_id"], file=r["dp_id"].replace(":", "_") + ".fits", instrument=r["instrument_name"],
            t_min_mjd=round(t, 5), t_max_mjd=round(float(r["t_max"]), 5), night_mjd=night(t),
            programme=r["proposal_id"], release_date=r["obs_release_date"], size_kB=float(r["access_estsize"]),
            product=r["dataproduct_subtype"], role=role, epoch_index=idx if role == "paper" else None,
            after_paper_span=bool(role == "heldout" and t > last)))
    return items


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def locate(trackb, slug, item):
    """The on-disk directory of a file: ``oifits`` (what Track B reads) or ``oifits_extra`` (the rest)."""
    for sub in ("oifits", "oifits_extra"):
        p = trackb / slug / sub / item["file"]
        if p.exists():
            return sub, p
    return None, None


def templates(trackb):
    """slug -> everything ``build`` needs that is not the ESO file list: orbit, positions, paper, conventions."""
    out = {}
    elements = {r["System"].strip(): r for r in vizier('SELECT * FROM "J/A+A/672/A119/table6"')}
    positions = {}
    for r in vizier('SELECT * FROM "J/A+A/672/A119/tableb1"'):
        positions.setdefault(r["System"].strip(), []).append(r)
    cat = {s["system"]: s for s in json.loads((trackb / "catalogue" / "2302.12960.json").read_text())}
    for slug, (name, cat_slug, kind) in GALLENNE_SYSTEMS.items():
        c = cat[cat_slug]
        sj = trackb / slug / "system.json"  # the Track B position where there is one (VizieR's name resolution put o Leo in the wrong place)
        ra, dec = ((lambda d: (d["ra_deg"], d["dec_deg"]))(json.loads(sj.read_text())) if sj.exists()
                   else (float(elements[name]["_RA"]), float(elements[name]["_DE"])))
        out[slug] = dict(
            simbad=name, cat_slug=cat_slug, kind=kind, instrument="GRAVITY", band="K", paper=GALLENNE23,
            ra_deg=ra, dec_deg=dec,
            conventions=CONVENTIONS["gallenne2023"], orbit=gallenne_orbit(elements[name]),
            positions=gallenne_positions(positions[name]),
            diameters_mas=c["diameters_mas"],
            positions_source="Gallenne et al. 2023, Appendix B Table B.1 (VizieR J/A+A/672/A119/tableb1): CANDID on "
                             "the SC visibilities and closure phases, bootstrap errors with 8-50 uas systematic floors",
            third_light=c.get("third_light"),
            caveats=c["caveats"] + " The Phase 3 products are uncalibrated (SINGLE_SCI_VIS): use closure phases only.")
    for slug, kind in SHORT_ARCS.items():
        s = json.loads((trackb / slug / "system.json").read_text())
        ref = s["reference"]
        out[slug] = dict(
            simbad=s["target"], cat_slug=slug, kind=kind, instrument=s["instrument"], band=s["band"],
            ra_deg=s["ra_deg"], dec_deg=s["dec_deg"], paper=SHORT_ARC_PAPERS[slug], conventions=CONVENTIONS[slug],
            orbit=short_arc_orbit(slug, s["reference_orbit"]), positions=short_arc_positions(s),
            diameters_mas=s.get("diameters_mas"), positions_source=ref["positions_source"],
            third_light=s.get("third_light"), caveats=s["caveats"])
    return out


def build(slug, trackb, tmpl):
    simbad, cat_slug, kind = tmpl["simbad"], tmpl["cat_slug"], tmpl["kind"]
    tmpl["conventions"] = dict(tmpl["conventions"], text=tmpl["conventions"]["text"] + _DEGENERATE)
    rec = dict(schema="trackb-target/1", slug=slug, target=simbad, kind=kind, instrument=tmpl["instrument"],
               band=tmpl["band"], ra_deg=tmpl["ra_deg"], dec_deg=tmpl["dec_deg"], paper=tmpl["paper"],
               conventions=tmpl["conventions"], orbit=tmpl["orbit"], diameters_mas=tmpl.get("diameters_mas"),
               positions_source=tmpl["positions_source"], third_light=tmpl.get("third_light"),
               caveats=tmpl["caveats"])
    pub = tmpl["positions"]
    prods = classify(eso_products(rec["ra_deg"], rec["dec_deg"]), [p["mjd"] for p in pub])
    for p in pub:
        p["below_lambda_2B"] = p["sep_mas"] < LAMBDA_2B_MAS[rec["instrument"]]
        p["role"] = "paper"
    files = []
    for it in prods:
        sub, path = locate(trackb, slug, it)
        it["dir"] = sub
        it["sha256"] = sha256(path) if path else None
        it["bytes"] = path.stat().st_size if path else None
        if it["dp_id"] in UNREADABLE:
            it["unreadable"] = UNREADABLE[it["dp_id"]]
        files.append(it)
    for i, p in enumerate(pub):
        p["dp_ids"] = [f["dp_id"] for f in files if f["epoch_index"] == i]
    held_nights = sorted({f["night_mjd"] for f in files if f["role"] == "heldout"})
    rec["epochs"] = pub
    rec["files"] = files
    rec["split"] = dict(
        rule=" ".join(__doc__.split("The held-out split (``role`` of each file):")[1].split("A night is")[0].split()),
        paper_epochs=len(pub), paper_epochs_with_phase3=sum(1 for p in pub if p["dp_ids"]),
        paper_epochs_below_lambda_2B=sum(p["below_lambda_2B"] for p in pub),
        paper_files=sum(f["role"] == "paper" for f in files),
        paper_night_files=sum(f["role"] == "paper_night" for f in files),
        heldout_files=sum(f["role"] == "heldout" for f in files), heldout_nights=len(held_nights),
        heldout_nights_after_paper_span=sum(1 for n in held_nights if n > night(max(p["mjd"] for p in pub))),
        last_paper_mjd=max(p["mjd"] for p in pub))
    rec["data_root"] = f"~/data/eso_binaries/trackb/{slug}"
    rec["provenance"] = dict(
        eso_query="ivoa.ObsCore dataproduct_type='visibility', GRAVITY/PIONIER, 18 arcsec cone on the SIMBAD position",
        download="https://dataportal.eso.org/dataPortal/file/<dp_id> (public Phase 3 products, unchanged)",
        built=datetime.date.today().isoformat(), builder="scripts/build_trackb_targets.py")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trackb", default=str(TRACKB))
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args(argv)
    trackb = pathlib.Path(a.trackb)
    OUT.mkdir(parents=True, exist_ok=True)
    for slug, tmpl in templates(trackb).items():
        if a.only and slug not in a.only:
            continue
        rec = build(slug, trackb, tmpl)
        (OUT / f"{slug}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
        s = rec["split"]
        print(f"{slug}: {s['paper_epochs']} paper epochs, {s['paper_files']}+{s['paper_night_files']} paper files, "
              f"{s['heldout_files']} held-out files on {s['heldout_nights']} nights")


if __name__ == "__main__":
    main()
