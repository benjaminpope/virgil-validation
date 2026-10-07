"""Stage O1 comparison table: virgil's fits of the OiDB collections against the
published values (design/plan_oidb.md, stage O1).

    python scripts/oidb_compare.py <dir with fit_<id>.json> [--md table.md] [--csv table.csv] [--include-l2]

Reads ``fit_<collection>.json`` (scripts/oidb_fit.py, run on OzSTAR) and
``oidb/references/<collection>.json`` and writes one row per compared quantity:
virgil's value and error, the published value and error, the sigma used, the
deviation in that sigma, the rule and the status. NumPy and the standard library only;
the reference orbit positions come from ``crosscheck.orbits`` (our NumPy evaluator).

The pass/fail rules (``CRITERIA``) were fixed before any fit of the real files was
run; ``CRITERIA_HASH`` is printed with the table so a change to them shows.

Statuses: PASS / FAIL (scored), MISSING (a scored quantity, or a whole fit file, that
virgil did not deliver; counted as a failure), REPORT (compared, not scored: orbits, variants,
alternative published values), NOT-CLEAN (a target the plan lists as not clean:
compared, never scored), WITHHELD (L2 data: the row is left out of the table unless
``--include-l2``, and an L2 table is not published before the dataPI has been
contacted). Every row carries the raw chi2/N of its fit on the quoted errors and the
fitted error scales; chi2/N above ``chi2_flag`` or a scale above ``scale_flag`` is
flagged (a rescaled chi2/N of 1 is not evidence of a good fit, and s >> 1 means the
model or the errors have failed).
"""

import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from crosscheck import orbits  # noqa: E402

CRITERIA = {
    "epoch": dict(max_dev_sigma=0.25, applies="per-epoch binary in-sample (same files): separation, PA, flux ratio, "
                                               "each against the published statistical error"),
    "parametric": dict(max_dev_sigma=2.0, applies="diameters, resolved flux, flux ratios from orbit fits, and the "
                                                   "A-star flux ratios (CANDID's bandwidth smearing averages V^2 and "
                                                   "the bispectrum over 3 points, ours the complex visibility over 7: "
                                                   "docs/method/candid.md)"),
    "adopted": dict(max_dev_sigma=1.0, applies="HD 45166: the paper's adopted value is the mean over four calibrations "
                                               "and its error their dispersion; virgil fits one of them"),
    "orbit": dict(applies="orbital elements (Jeffreys priors, astrometry only): reported, not scored"),
    "not_clean": dict(applies="HR 6819 (Be decretion disk): compared, never scored"),
    "chi2_flag": 3.0,
    "scale_flag": 3.0,
    "sigma_ori_ellipse_scale": 2.24,
    "gl229_parallax_mas": 173.574,
}
CRITERIA_HASH = hashlib.sha256(json.dumps(CRITERIA, sort_keys=True).encode()).hexdigest()[:16]

L2 = {"647a22a9-5047-4220-ba22-a95047022072"}
FIELDS = ["collection", "target", "epoch", "quantity", "virgil", "virgil_err", "published", "published_err",
          "sigma_used", "dev_sigma", "rule", "status", "chi2_raw", "flags", "note"]


# ----------------------------------------------------------------------------- arithmetic

def wrap180(x):
    return (x + 180.0) % 360.0 - 180.0


def ellipse_cov(sigma_maj, sigma_min, phi_deg):
    """Covariance of (dRA, dDec) for an error ellipse with its major axis at position
    angle phi (degrees east of north)."""
    p = math.radians(phi_deg)
    u = np.array([math.sin(p), math.cos(p)])
    v = np.array([math.cos(p), -math.sin(p)])
    return sigma_maj ** 2 * np.outer(u, u) + sigma_min ** 2 * np.outer(v, v)


def project(dra, ddec, cov):
    """sigma of the separation (mas) and of the PA (degrees) from a (dRA, dDec) covariance."""
    rho = math.hypot(dra, ddec)
    r = np.array([dra, ddec]) / rho
    t = np.array([ddec, -dra]) / rho
    return math.sqrt(r @ cov @ r), math.degrees(math.sqrt(t @ cov @ t) / rho)


def err_toward(err, deviation):
    """A symmetric error, or the side of an asymmetric [-lo, +hi] error the deviation falls on."""
    if isinstance(err, (list, tuple)):
        return abs(err[1]) if deviation > 0 else abs(err[0])
    return err


def chi2_info(fit):
    """(raw chi2/N on the quoted errors, flags) of one epoch fit or orbit fit."""
    if fit is None:
        return None, ""
    c = fit.get("chi2_raw", {})
    red = c.get("all", c).get("chi2_red")
    scales = list((fit.get("scales") or {}).values())
    for d in fit.get("datasets", []):
        scales += list(d.get("scales", {}).values())
    flags = []
    if red is not None and red > CRITERIA["chi2_flag"]:
        flags.append(f"raw chi2/N {red:.1f} > {CRITERIA['chi2_flag']:g}")
    if scales and max(scales) > CRITERIA["scale_flag"]:
        flags.append(f"error scale {max(scales):.1f} > {CRITERIA['scale_flag']:g}")
    return red, "; ".join(flags)


def row(collection, target, epoch, quantity, value, err, pub, pub_err, rule, *, fit=None, status=None, angle=False,
        sigma=None, note=""):
    """One comparison. ``sigma`` defaults to the published error; the status follows
    the rule unless given."""
    diff = None if value is None or pub is None else (wrap180(value - pub) if angle else value - pub)
    s = sigma if sigma is not None else (None if diff is None else err_toward(pub_err, diff))
    dev = None if diff is None or not s else diff / s
    if status is None:
        limit = CRITERIA.get(rule, {}).get("max_dev_sigma")
        if limit is None:
            status = "REPORT"
        elif value is None:
            status = "MISSING"  # a scored quantity virgil did not deliver counts as a failure
        else:
            status = "REPORT" if dev is None else ("PASS" if abs(dev) <= limit else "FAIL")
    red, flags = chi2_info(fit)
    return dict(collection=collection, target=target, epoch=epoch, quantity=quantity, virgil=value, virgil_err=err,
                published=pub, published_err=pub_err, sigma_used=s, dev_sigma=dev, rule=rule, status=status,
                chi2_raw=red, flags=flags, note=note)


def val(fit, key):
    v = (fit or {}).get("values", {}).get(key)
    return (None, None) if v is None else (v[0], v[1])


def by_epoch(fits, ref_epochs, *, date_key="date"):
    """Pair each published epoch with virgil's epoch fit: nearest MJD within a day when
    both have one, else the same date label."""
    out = []
    for r in ref_epochs:
        mjd = r.get("mjd")
        if mjd is None and r.get("hjd_minus_2400000") is not None:
            mjd = r["hjd_minus_2400000"] - 0.5
        best = None
        if mjd is not None:
            near = [f for f in fits if f.get("mjd") is not None and abs(f["mjd"] - mjd) < 1.0]
            best = min(near, key=lambda f: abs(f["mjd"] - mjd)) if near else None
        if best is None:
            best = next((f for f in fits if f.get("label") == r.get(date_key)), None)
        out.append((r, best))
    return out


def orbit_values(orbit):
    """Elements with errors: NUTS medians and sds when sampled, else the MAP without errors."""
    if orbit.get("nuts"):
        return {k: (s["median"], s["sd"]) for k, s in orbit["nuts"]["elements"].items()}, "NUTS median, sd"
    return {k: (v, None) for k, v in orbit["map"].items()}, "MAP"


def fold(t, t_pub, period):
    """t moved by whole periods to the passage nearest t_pub."""
    return t_pub + ((t - t_pub + period / 2) % period - period / 2)


def orbit_rows(cid, target, orbit, pub, *, omega_shift=0.0, a_scale=1.0, twin=True, status="REPORT", note=""):
    """Elements against a published orbit. ``pub`` maps our element names to [value,
    err]; ``omega_shift`` converts the published omega to the companion's (+180 for
    the primary's); with astrometry only, (Omega, omega) and (Omega+180, omega+180) fit
    equally, and the twin nearer the published Omega is compared."""
    el, how = orbit_values(orbit)
    out = []
    Om, om = el["Omega_deg"][0], el["omega_deg"][0]
    if twin and "Omega_deg" in pub and abs(wrap180(Om + 180 - pub["Omega_deg"][0])) < abs(wrap180(Om - pub["Omega_deg"][0])):
        el["Omega_deg"] = ((Om + 180) % 360, el["Omega_deg"][1])
        el["omega_deg"] = ((om + 180) % 360, el["omega_deg"][1])
        how += "; the (Omega+180, omega+180) twin"
    for k, (p, pe) in pub.items():
        if k not in el and k != "a_au":
            continue
        v, e = el["a_mas"] if k == "a_au" else el[k]
        if k == "a_au":
            p, pe = p * a_scale, None if pe is None else pe * a_scale
        if k == "omega_deg":
            p = (p + omega_shift) % 360
        if k == "t_peri_mjd":
            v = fold(v, p, el["period_day"][0])
        out.append(row(cid, target, "orbit", "a_mas" if k == "a_au" else k, v, e, p, pe, "orbit", fit=orbit,
                       status=status, angle=k.endswith("_deg") and k != "inc_deg", note="; ".join(x for x in (how, note) if x)))
    return out


# ----------------------------------------------------------------------------- per collection

def cmp_gl229(cid, fit, ref):
    rows, orbit = [], fit["orbit"]
    par = ref["orbit"]["pmoired_frequentist"]
    el, how = orbit_values(orbit)
    rows.append(row(cid, "Gl 229 Ba-Bb", "orbit", "flux_ratio", *el["flux_ratio"], *par["flux_ratio_2um"],
                    "parametric", fit=orbit, note=f"PMOIRED (frequentist) f2/f1 at 2.0 um; virgil {how}"))
    octo = ref["orbit"]["octofitter_bayesian"]
    rows.append(row(cid, "Gl 229 Ba-Bb", "orbit", "flux_ratio", *el["flux_ratio"], *octo["flux_ratio_2um"],
                    "parametric", fit=orbit, status="REPORT", note="Octofitter posterior (reported)"))
    names = {"P_day": "period_day", "e": "ecc", "i_deg": "inc_deg", "Omega_deg": "Omega_deg",
             "omega_primary_deg": "omega_deg", "T_peri_mjd": "t_peri_mjd", "a_au": "a_au"}
    for label, src in (("PMOIRED", par), ("Octofitter", octo)):
        pub = {names[k]: src[k] for k in names}
        rows += orbit_rows(cid, "Gl 229 Ba-Bb", orbit, pub, omega_shift=180.0,
                           a_scale=CRITERIA["gl229_parallax_mas"],
                           note=f"vs {label}; theirs with RVs, ours astrometry only; omega_pub(primary)+180")
    # per-night positions against Xuan's PMOIRED orbit, evaluated with crosscheck.orbits
    for e in fit["epochs"]:
        dra, ddec = (float(x) for x in np.ravel(orbits.position(
            e["mjd"], par["P_day"][0], par["T_peri_mjd"][0], par["e"][0], par["i_deg"][0],
            (par["omega_primary_deg"][0] + 180) % 360, par["Omega_deg"][0],
            par["a_au"][0] * CRITERIA["gl229_parallax_mas"])))
        rho, pa = math.hypot(dra, ddec), math.degrees(math.atan2(dra, ddec)) % 360
        for q, p, ang in (("rho_mas", rho, False), ("pa_deg", pa, True)):
            v, s = val(e, q)
            rows.append(row(cid, "Gl 229 Ba-Bb", e["label"], q, v, s, p, None, "orbit", fit=e, status="REPORT",
                            angle=ang, sigma=s,
                            note="vs Xuan's PMOIRED orbit at the night's mean time (crosscheck.orbits); "
                                 "a static fit to a night of a binary moving ~0.1 mas/h; deviation in virgil's sigma"))
    return rows


def _position_rows(cid, target, r, e, rule, *, status=None, cov=None, note=""):
    """rho and PA rows for one epoch; ``cov`` (a published error ellipse) gives the
    sigmas when the paper has no rho/PA errors."""
    rows = []
    rho_p, rho_pe = r["rho_mas"][0], r["rho_mas"][1]
    pa_p, pa_pe = r["pa_deg"][0], r["pa_deg"][1]
    if cov is not None:
        p = math.radians(pa_p)
        rho_pe, pa_pe = project(rho_p * math.sin(p), rho_p * math.cos(p), cov)
    for q, p, pe, ang in (("rho_mas", rho_p, rho_pe, False), ("pa_deg", pa_p, pa_pe, True)):
        v, s = val(e, q)
        rows.append(row(cid, target, r.get("date"), q, v, s, p, pe, rule, fit=e, status=status, angle=ang, note=note))
    return rows


def cmp_hr6819(cid, fit, ref):
    rows = []
    for r, e in by_epoch(fit["epochs"], ref["epochs"]):
        o = r["other"]
        cov = ellipse_cov(o["err_ellipse_major_mas"], o["err_ellipse_minor_mas"], o["err_ellipse_PA_deg"])
        rows += _position_rows(cid, "HR 6819", r, e, "epoch", status="NOT-CLEAN", cov=cov,
                               note="sigma from the published error ellipse projected on rho and PA")
        f, fe = val(e, "flux")
        frac = None if f is None else (f / (1 + f), fe / (1 + f) ** 2)
        rows.append(row(cid, "HR 6819", r["date"], "f_pre-sd", *(frac or (None, None)), *r["flux_ratio"][:2], "epoch",
                        fit=e, status="NOT-CLEAN", note="fraction of the total K continuum, f = r/(1+r)"))
    o = ref["orbit"]
    pub = {"period_day": o["P_day"][:2], "a_mas": o["a_mas"][:2], "ecc": o["e"][:2], "inc_deg": o["i_deg"][:2],
           "Omega_deg": o["Omega_deg"][:2], "omega_deg": o["omega_deg"][:2], "t_peri_mjd": o["T_peri_mjd"][:2]}
    rows += orbit_rows(cid, "HR 6819", fit["orbit"], pub, omega_shift=180.0, status="NOT-CLEAN",
                       note="vs eccentric astro+RVs (headline); omega_Be+180")
    return rows


def _iota_rows(cid, epochs, ref_epochs, status=None, note=""):
    rows = []
    for r, e in by_epoch(epochs, ref_epochs):
        rows += _position_rows(cid, "iota Peg", r, e, "epoch", status=status, note=note)
        f, fe = val(e, "flux")
        inv = (None, None) if f is None else (1 / f, fe / f ** 2)
        rows.append(row(cid, "iota Peg", r["date"], "F_pri/sec", *inv, *r["flux_ratio"], "epoch", fit=e,
                        status=status, note="; ".join(x for x in ("1/flux", note) if x)))
        alt = r.get("candid_alt")
        if alt:
            for q, ang in (("rho_mas", False), ("pa_deg", True)):
                rows.append(row(cid, "iota Peg", r["date"], q, *val(e, q), *alt[q], "epoch", fit=e, angle=ang,
                                status="REPORT" if status is None else status, note="vs the paper's CANDID column"))
    return rows


def cmp_iota_peg(cid, fit, ref):
    rows = _iota_rows(cid, fit["epochs"], ref["epochs"])
    o = ref["orbit"]
    pub = {"period_day": o["P_day"][:2], "a_mas": o["a_mas"][:2], "ecc": o["e"][:2], "inc_deg": o["i_deg"][:2],
           "Omega_deg": o["Omega_deg"][:2], "omega_deg": o["omega_deg"][:2], "t_peri_mjd": o["T_peri_unstated_system"][:2]}
    rows += orbit_rows(cid, "iota Peg", fit["orbit"], pub,
                       note="the paper's omega is of an unstated component and its T_peri of an unstated time system")
    return rows


def cmp_workshop(cid, fit, ref):
    rows = _iota_rows(cid, fit["targets"]["iot Peg"]["epochs"], [ref["targets"]["iot Peg"]["epoch_2018-10-22"]],
                      note="files byte-identical to fac164e1's")
    s = fit["targets"]["sig Ori"]
    so = ref["targets"]["sig Ori"]
    in_collection = [r for r in so["epochs"] if r["date"] == "2011-09-29"]  # the one sigma Ori night in the workshop set
    for r, e in by_epoch(s["epochs"], in_collection):
        el = r["error_ellipse"]
        k = CRITERIA["sigma_ori_ellipse_scale"]
        cov = ellipse_cov(el["sigma_maj_mas"] / k, el["sigma_min_mas"] / k, el["phi_deg"])
        rows += _position_rows(cid, "sigma Ori Aa-Ab", r, e, "epoch", cov=cov,
                               note="sigma: the published ellipse / 2.24 (its chi2 inflation), projected")
        for q in ("fAa", "fAb", "fB"):
            rows.append(row(cid, "sigma Ori Aa-Ab", r["date"], q, (e or {}).get("fractions", {}).get(q), None,
                            *r["fractions"][q],
                            "epoch", fit=e, note="fraction of the total light"))
        for v in s.get("variants", []):
            for q, ang in (("rho_mas", False), ("pa_deg", True)):
                p = math.radians(r["pa_deg"][0])
                sr, sp = project(r["rho_mas"][0] * math.sin(p), r["rho_mas"][0] * math.cos(p), cov)
                rows.append(row(cid, "sigma Ori Aa-Ab", r["date"], q, *val(v, q), r[q][0], sr if q == "rho_mas" else sp,
                                "epoch", fit=v, angle=ang, status="REPORT", note=v.get("variant", "variant")))
    for x in rows:
        x["status"] = "WITHHELD" if x["status"] in ("PASS", "FAIL") else x["status"]
        x["note"] = "; ".join(y for y in (x["note"], "L2: contact the dataPI before presenting") if y)
    return rows


def cmp_astars(cid, fit, ref):
    rows = []
    for r in ref["epochs"]:
        star = r["star"]
        t = fit["targets"].get(star)
        e = t["epochs"][0] if t and t.get("epochs") else None  # a missing star gives MISSING rows
        rec = dict(date=star, rho_mas=[r["rho_mas"], r["rho_err_mas"]], pa_deg=[r["pa_deg_E_of_N"], r["pa_err_deg"]])
        rows += _position_rows(cid, star, rec, e, "epoch")
        f, fe = val(e, "flux")
        c = r["contrast"]
        rows.append(row(cid, star, star, "flux_%_primary", None if f is None else 100 * f, None if fe is None else 100 * fe,
                        c["flux_ratio_percent_of_primary"], c["err_percent"], "parametric", fit=e,
                        note="parametric, not epoch: smearing is defined differently from CANDID's"))
        rf, rfe = val(e, "resolved")
        rp = r["resolved_flux_percent_primary"]
        rows.append(row(cid, star, star, "resolved_%_primary", None if rf is None else 100 * rf,
                        None if rfe is None else 100 * rfe, rp["value"], rp["err"], "parametric", fit=e))
        rows.append(row(cid, star, star, "UD1_mas", *val(e, "ud1"), r["UD1_mas"], r["UD1_err_mas"], "parametric", fit=e))
        if r.get("UD2_mas") is not None:
            rows.append(row(cid, star, star, "UD2_mas", *val(e, "ud2"), r["UD2_mas"], None, "parametric", fit=e,
                            status="REPORT", note="no published error"))
    return rows


def cmp_hd45166(cid, fit, ref):
    rows = []
    r = ref["epochs"][0]
    e = fit["epochs"][0]
    o = r["other"]
    for q, ang in (("rho_mas", False), ("pa_deg", True)):
        rows.append(row(cid, "HD 45166", r["date"], q, *val(e, q), *r[q][:2], "adopted", fit=e, angle=ang,
                        note="adopted: mean and dispersion of the four calibrations"))
        per = o["rho_per_calibration_derived" if q == "rho_mas" else "pa_deg_per_calibration_derived"]
        v = val(e, q)[0]
        if v is not None:
            near = min(per, key=lambda p: abs(wrap180(v - p) if ang else v - p))
            rows.append(row(cid, "HD 45166", r["date"], q, *val(e, q), near, None, "adopted", fit=e, angle=ang,
                            status="REPORT", sigma=r[q][1], note="nearest single calibration; adopted sigma"))
    rows.append(row(cid, "HD 45166", r["date"], "flux_ratio", *val(e, "flux"), *r["flux_ratio"][:2], "adopted", fit=e,
                    note="f2/f1"))
    for v in fit.get("variants", []):
        if "values" not in v:
            continue
        for q, ang in (("rho_mas", False), ("pa_deg", True)):
            rows.append(row(cid, "HD 45166", r["date"], q, *val(v, q), *r[q][:2], "adopted", fit=v, angle=ang,
                            status="REPORT", note=v["variant"]))
    return rows


def cmp_pi1gru(cid, fit, ref):
    ud = next(x for x in ref["results"] if x["name"] == "equivalent uniform-disk diameter")
    rows = [row(cid, "pi1 Gru", "2014-09", "UD_mas", *val(fit["parametric"], "ud_mas"), ud["value"], ud["err"],
                "parametric", fit=fit["parametric"], note="all V^2 (LitPro's fit)")]
    for v in fit.get("variants", []):
        rows.append(row(cid, "pi1 Gru", "2014-09", "UD_mas", *val(v, "ud_mas"), ud["value"], ud["err"], "parametric",
                        fit=v, status="REPORT", note=v.get("variant", "")))
    return rows


COMPARE = {
    "782185b2-0727-42b0-a185-b2072732b047": cmp_gl229,
    "696baf06-6c3c-424d-abaf-066c3c324d99": cmp_hr6819,
    "647a22a9-5047-4220-ba22-a95047022072": cmp_workshop,
    "fac164e1-d9d0-4500-8164-e1d9d0450099": cmp_iota_peg,
    "bda75673-61c6-49f0-a756-7361c699f0c4": cmp_astars,
    "f4afc4cd-fd31-40d3-afc4-cdfd3150d340": cmp_hd45166,
    "19f7e2cf-2a03-4bb2-b7e2-cf2a03bbb245": cmp_pi1gru,
}


# ----------------------------------------------------------------------------- table

def compare(fits_dir, *, refs_dir=ROOT / "oidb" / "references", include_l2=False):
    rows, missing = [], []
    for cid, fn in COMPARE.items():
        path = os.path.join(fits_dir, f"fit_{cid}.json")
        if cid in L2 and not include_l2:
            continue
        if not os.path.exists(path):
            missing.append(cid)
            rows.append(row(cid, "", "", "fit", None, None, None, None, "epoch",
                            note=f"no fit_{cid}.json: the task failed or has not run"))
            continue
        with open(path) as f:
            fit = json.load(f)
        with open(os.path.join(refs_dir, f"{cid}.json")) as f:
            ref = json.load(f)
        rows += fn(cid, fit, ref)
    return rows, missing


def _fmt(x, nd=4):
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        return "-%s/+%s" % (_fmt(abs(x[0]), nd), _fmt(abs(x[1]), nd))
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def markdown(rows, missing=()):
    head = ["collection", "target", "epoch", "quantity", "virgil", "published", "dev (sigma)", "rule", "status",
            "raw chi2/N", "flags / note"]
    lines = [f"Criteria hash `{CRITERIA_HASH}`.", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        lines.append("| " + " | ".join([
            r["collection"][:8], r["target"], str(r["epoch"]), r["quantity"],
            f"{_fmt(r['virgil'], 7)} ± {_fmt(r['virgil_err'], 2)}", f"{_fmt(r['published'], 7)} ± {_fmt(r['published_err'], 2)}",
            _fmt(r["dev_sigma"], 3), r["rule"], r["status"], _fmt(r["chi2_raw"], 3),
            "; ".join(x for x in (r["flags"], r["note"]) if x)]) + " |")
    if missing:
        lines += ["", "No fit yet: " + ", ".join(missing)]
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    lines += ["", "Totals: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())),
              f"Failures (FAIL + MISSING): {counts.get('FAIL', 0) + counts.get('MISSING', 0)}"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("fits_dir")
    p.add_argument("--md", help="write the markdown table here (default: stdout)")
    p.add_argument("--csv", help="also write a CSV")
    p.add_argument("--include-l2", action="store_true",
                   help="include the L2 workshop collection (never publish before the dataPI is contacted)")
    a = p.parse_args(argv)
    rows, missing = compare(a.fits_dir, include_l2=a.include_l2)
    text = markdown(rows, missing)
    if a.md:
        pathlib.Path(a.md).write_text(text)
    else:
        sys.stdout.write(text)
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({k: json.dumps(v) if isinstance(v, (list, tuple)) else v for k, v in r.items()})


if __name__ == "__main__":
    main()
