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


def science_field(field_id, flag_row):
    """The one FIELD row the unflagged rows of MAIN point to. ``split`` may leave calibrator rows in FIELD, so FIELD's first row is not the target;
    several fields in use is refused (the read stage would mix them)."""
    import numpy as np

    used = np.unique(np.asarray(field_id)[~np.asarray(flag_row, bool)])
    if len(used) != 1:
        raise SystemExit(f"the unflagged rows of MAIN use {len(used)} fields {used.tolist()}: select one field (split) before reading")
    return int(used[0])


def describe(ms):
    import alma_worker as aw
    import casatools

    tb = casatools.table()
    tb.open(str(ms))
    cols = tb.colnames()
    nrow = tb.nrows()
    fid = science_field(tb.getcol("FIELD_ID"), tb.getcol("FLAG_ROW"))
    tb.close()
    tb.open(f"{ms}/FIELD")
    names = list(tb.getcol("NAME"))
    phase = tb.getcol("PHASE_DIR")  # (2, 1, nfield) rad
    tb.close()
    dirs = [[math.degrees(float(phase[0, 0, i])) % 360.0, math.degrees(float(phase[1, 0, i]))] for i in range(phase.shape[2])]
    rows = aw.info({"path": str(ms)})["ddids"]
    return dict(
        ms=str(ms), columns=cols, has_corrected="CORRECTED_DATA" in cols, n_rows=int(nrow), fields=[names[fid]], field_table=names, field_id=fid,
        phase_centre=dict(field=names[fid], field_id=fid, ra_deg=dirs[fid][0], dec_deg=dirs[fid][1], note="PHASE_DIR of the one field the unflagged rows of MAIN use"),
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
    if len(info["field_table"]) > 1:
        print(f"note: FIELD holds {len(info['field_table'])} rows {info['field_table']}; MAIN uses only {info['fields'][0]!r}")


if __name__ == "__main__":
    main()
