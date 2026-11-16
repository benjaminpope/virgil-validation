"""Describe a measurement set for the DSHARP campaign: columns, DATA_DESC_ID to observation map, rows per ddid and the field phase centre.

    dsharp_ms_info.py --ms PATH --out info.json            (runs in the CASA environment: casatools only)

The fit reads the phase centre from this file (``dsharp_fit.py --info``) and the read stage records which data column holds what.
Nothing here imports virgil.
"""

import argparse
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "external_bridge"))


def describe(ms):
    import alma_worker as aw
    import casatools

    tb = casatools.table()
    tb.open(str(ms))
    cols = tb.colnames()
    nrow = tb.nrows()
    tb.close()
    tb.open(f"{ms}/FIELD")
    names = list(tb.getcol("NAME"))
    phase = tb.getcol("PHASE_DIR")  # (2, 1, nfield) rad
    tb.close()
    dirs = [[math.degrees(float(phase[0, 0, i])) % 360.0, math.degrees(float(phase[1, 0, i]))] for i in range(phase.shape[2])]
    rows = aw.info({"path": str(ms)})["ddids"]
    return dict(
        ms=str(ms), columns=cols, has_corrected="CORRECTED_DATA" in cols, n_rows=int(nrow), fields=names,
        phase_centre=dict(field=names[0], ra_deg=dirs[0][0], dec_deg=dirs[0][1], note="PHASE_DIR of field 0; every field's centre is in `phase_dirs_deg`"),
        phase_dirs_deg=dirs, ddids=rows, obs_of_ddid={r["ddid"]: r["obsid"] for r in rows}, n_obs=len({r["obsid"] for r in rows}),
        casatools=casatools.version_string(),
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ms", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    info = describe(a.ms)
    pathlib.Path(a.out).write_text(json.dumps(info, indent=1, default=str))
    print(f"{info['n_rows']:,} rows, {len(info['ddids'])} ddids, {info['n_obs']} observations, corrected data: {info['has_corrected']}, phase centre {info['phase_centre']}")
    if len(info["fields"]) > 1:
        print(f"warning: {len(info['fields'])} fields: {info['fields']}; the read stage must pick one")


if __name__ == "__main__":
    main()
