"""Find archive calibrators for the GRAVITY science files of the gallery binaries.

ESO TAP query (``ivoa.ObsCore``), per science file: GRAVITY products within ``--window`` days
(default 0.25) of the science exposure, same resolution mode and polarisation mode, similar
angular resolution (``s_resolution``, i.e. the same baselines), not the science file. Candidates
from the same programme come first, then those of other programmes on the same night.

Excluded: any star within ``--exclude-arcsec`` of a science target in ``--root`` (every Track B
or gallery system, so programme binaries are never calibrators), and any star whose SIMBAD types (all of them,
for the nearest object) include a binary or multiple. Only standards are kept (target name "STD,SINGLE", the archive's name for ``DPR TYPE`` STD,
whose products are ``SINGLE_CAL_VIS``): a named star is a science target, also in the same
programme (several programmes here observe binaries).

Writes JSON: {system: [{science, t_min, candidates: [...], rejected: [...]}]}. No data are
downloaded; ``access_estsize`` (kB) gives the download size. Usage:

    python scripts/gravity_calibrator_query.py --out gravity_calibrators.json
"""

import argparse
import glob
import json
import os

import numpy as np
import pyvo
from astropy.io import fits

TAP = "https://archive.eso.org/tap_obs"
SIMBAD = "http://simbad.cds.unistra.fr/simbad/sim-tap"
ROOT = os.path.expanduser("~/data/eso_binaries/trackb")
# slug -> resolution of the template ("HIGH" or "MEDIUM"); the gallery GRAVITY systems.
SYSTEMS = {
    "ak_for": "HIGH", "hd188088": "HIGH", "hd210763": "HIGH", "hd41255": "HIGH",
    "hd70937": "HIGH", "hd_224974": "HIGH", "hd_9312": "HIGH", "omi_leo": "HIGH",
    "v963_cen": "HIGH", "kap_vel": "MEDIUM", "ll_aqr": "MEDIUM",
}  # fmt: skip
# SIMBAD main types of binaries and multiples (and their variable-star flavours).
BINARY_OTYPES = {
    "**",
    "SB*",
    "EB*",
    "El*",
    "Al*",
    "bL*",
    "WU*",
    "Sy*",
    "RS*",
    "CV*",
    "XB*",
    "Be*",
}


def dp_id_of(path):
    return os.path.basename(path)[: -len(".fits")].replace("_", ":")


def science_targets(root):
    """RA, Dec (deg) of every science file under ``root``: the stars never to use as calibrators."""
    ra, dec = [], []
    for f in glob.glob(f"{root}/*/oifits*/*.fits"):
        h = fits.getheader(f)
        if "RA" in h and "DEC" in h:
            ra.append(h["RA"]), dec.append(h["DEC"])
    return np.array(ra), np.array(dec)


def near(ra, dec, ras, decs, arcsec):
    d = np.hypot((ras - ra) * np.cos(np.radians(dec)), decs - dec) * 3600.0
    return bool(d.size and d.min() < arcsec)


SIMBAD_RADIUS_ARCSEC = 30.0
_SIMBAD_CACHE = {}


def binary_type(simbad, ra, dec):
    """The nearest SIMBAD object within SIMBAD_RADIUS_ARCSEC (proper motion moves the star from its J2000
    position by several arcsec: 64 Psc by 5.6 arcsec): ``(main_id, binary_otype or None)``.

    Every type the object carries (``otypes``, not only the main type) is checked: a
    spectroscopic binary can have a main type of ``*`` or ``PM*`` with ``SB*`` only among the others.
    """
    key = (round(ra, 4), round(dec, 4))
    if key not in _SIMBAD_CACHE:
        near = simbad.run_sync(
            f"SELECT TOP 1 oid, main_id, DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {ra}, {dec})) AS sep "
            "FROM basic WHERE CONTAINS(POINT('ICRS', ra, dec), "
            f"CIRCLE('ICRS', {ra}, {dec}, {SIMBAD_RADIUS_ARCSEC / 3600})) = 1 "
            "ORDER BY sep"
        ).to_table()
        rows = (
            simbad.run_sync(
                f"SELECT otype FROM otypes WHERE oidref = {near['oid'][0]}"
            ).to_table()
            if len(near)
            else near
        )
        if not len(near):
            _SIMBAD_CACHE[key] = (None, None)
        else:
            hits = sorted({str(t).strip() for t in rows["otype"]} & BINARY_OTYPES)
            _SIMBAD_CACHE[key] = (str(near["main_id"][0]), ",".join(hits) or None)
    return _SIMBAD_CACHE[key]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default=ROOT)
    p.add_argument("--out", default=f"{ROOT}/gravity_calibrators.json")
    p.add_argument("--window", type=float, default=0.25, help="days either side")
    p.add_argument("--exclude-arcsec", type=float, default=30.0)
    p.add_argument(
        "--resolution-tol", type=float, default=0.15, help="relative s_resolution"
    )
    args = p.parse_args()

    tap, simbad = pyvo.dal.TAPService(TAP), pyvo.dal.TAPService(SIMBAD)
    sra, sdec = science_targets(args.root)
    out = {}
    for slug in SYSTEMS:
        files = sorted(glob.glob(f"{args.root}/{slug}/oifits/*.fits"))
        ids = ",".join(f"'{dp_id_of(f)}'" for f in files)
        sci = tap.run_sync(
            "SELECT dp_id, t_min, proposal_id, em_res_power, obstech, s_resolution "
            f"FROM ivoa.ObsCore WHERE dp_id IN ({ids})"
        ).to_table()
        entries = []
        for s in sorted(sci, key=lambda r: r["t_min"]):
            t = float(s["t_min"])
            cand = tap.run_sync(
                "SELECT dp_id, target_name, t_min, access_estsize, proposal_id, obs_release_date, "
                "s_ra, s_dec, em_res_power, obstech, s_resolution FROM ivoa.ObsCore "
                f"WHERE instrument_name = 'GRAVITY' AND t_min BETWEEN {t - args.window} "
                f"AND {t + args.window} AND dp_id <> '{s['dp_id']}'"
            ).to_table()
            keep, rejected = [], []
            for c in cand:
                row = {
                    k: (v.item() if hasattr(v, "item") else v)
                    for k, v in zip(c.colnames, c)
                }
                reason = None
                if c["target_name"] != "STD,SINGLE":
                    reason = "not a standard (DPR TYPE STD): a science product, also in the same programme"
                elif c["obstech"] != s["obstech"]:
                    reason = "polarisation mode differs"
                elif (
                    abs(c["em_res_power"] - s["em_res_power"]) > 0.3 * s["em_res_power"]
                ):
                    reason = "spectral resolution differs"
                elif (
                    abs(c["s_resolution"] / s["s_resolution"] - 1) > args.resolution_tol
                ):
                    reason = "baselines differ (s_resolution)"
                elif near(c["s_ra"], c["s_dec"], sra, sdec, args.exclude_arcsec):
                    reason = "Track B / gallery science target"
                else:
                    main_id, otypes = binary_type(simbad, c["s_ra"], c["s_dec"])
                    row["simbad_main_id"] = main_id
                    reason = f"known binary ({main_id}: {otypes})" if otypes else None
                row["same_programme"] = bool(c["proposal_id"] == s["proposal_id"])
                if reason:
                    rejected.append({**row, "reason": reason})
                else:
                    keep.append(row)
            keep.sort(key=lambda r: not r["same_programme"])
            entries.append(
                {"science": s["dp_id"], "t_min": t, "proposal_id": s["proposal_id"],
                 "candidates": keep, "rejected": rejected}
            )  # fmt: skip
            print(
                slug,
                s["dp_id"],
                len(keep),
                "candidates",
                len(rejected),
                "rejected",
                flush=True,
            )
        out[slug] = entries
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    kb = {
        k: sum(c["access_estsize"] for e in v for c in e["candidates"])
        for k, v in out.items()
    }
    print({k: f"{v / 1e6:.2f} GB" for k, v in kb.items()})


if __name__ == "__main__":
    main()
