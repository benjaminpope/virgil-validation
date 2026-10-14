"""Register the provenance of the ESO Phase 3 GRAVITY products used as validation data.

    python scripts/register_eso_phase3.py zet_boo eta_oph [--root ~/data/eso_binaries/trackb] [--out oidb/eso_phase3.json]

For each system's ``system.json`` (the ESO Track B staging area, design/plan_eso_binaries.md) this writes one record
per Phase 3 product named by its ``reference_positions``: the dp_id, file name, size, SHA-256, the header's DATE-OBS,
MJD-OBS and programme, the AT stations, and the public release date, programme and product type that the ESO
archive's ``ivoa.ObsCore`` table gives for that dp_id (queried once, here; the tests read the JSON and need no
network). The files themselves are never committed (DATA_ATTRIBUTION.md).
"""

import argparse
import csv
import hashlib
import io
import json
import pathlib
import sys
import time

import requests
from astropy.io import fits

TAP = "https://archive.eso.org/tap_obs/sync"
ROOT = pathlib.Path(__file__).resolve().parents[1]


def obscore(dp_id):
    """The ObsCore row of a Phase 3 dp_id (release date, programme, product type), or {} if the archive has none."""
    q = ("SELECT dp_id, obs_release_date, proposal_id, dataproduct_type, instrument_name, target_name "
         f"FROM ivoa.ObsCore WHERE dp_id = '{dp_id}'")
    r = requests.get(TAP, params=dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=q), timeout=120)
    r.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(r.text)))
    return rows[0] if rows else {}


def record(path, dp_id):
    data = path.read_bytes()
    with fits.open(path) as h:
        hd = h[0].header
        stations = [hd.get(f"HIERARCH ESO ISS CONF STATION{i}") for i in (1, 2, 3, 4)]
        row = dict(dp_id=dp_id, file=path.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                   date_obs=hd.get("DATE-OBS"), mjd_obs=hd.get("MJD-OBS"), prog_id=hd.get("HIERARCH ESO OBS PROG ID"),
                   object=hd.get("OBJECT"), insname=hd.get("INSTRUME"), spec_res=hd.get("HIERARCH ESO INS SPEC RES"),
                   stations=stations, prodcatg=hd.get("PRODCATG"))
    o = obscore(dp_id)
    row["archive"] = dict(release_date=o.get("obs_release_date"), proposal_id=o.get("proposal_id"),
                          dataproduct_type=o.get("dataproduct_type"), target_name=o.get("target_name"))
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("systems", nargs="+")
    ap.add_argument("--root", default=str(pathlib.Path.home() / "data/eso_binaries/trackb"))
    ap.add_argument("--out", default=str(ROOT / "oidb" / "eso_phase3.json"))
    a = ap.parse_args(argv)
    root = pathlib.Path(a.root)
    out = dict(registered=time.strftime("%Y-%m-%d"), source="ESO Science Archive Facility, Phase 3 (ivoa.ObsCore, dataproduct_type='visibility')",
               query_service=TAP, systems={})
    for s in a.systems:
        sysjson = json.loads((root / s / "system.json").read_text())
        recs = []
        for ep in sysjson["reference_positions"]:
            for dp in ep.get("phase3_dp_ids") or []:
                recs.append(dict(record(root / s / "oifits" / (dp.replace(":", "_") + ".fits"), dp), epoch_date=ep["date"]))
        out["systems"][s] = dict(target=sysjson["target"], reference=sysjson["reference"]["bibcode"],
                                 programmes=sysjson["programmes"], products=recs)
        print(s, len(recs), "products", sum(r["bytes"] for r in recs) / 1e6, "MB", flush=True)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
