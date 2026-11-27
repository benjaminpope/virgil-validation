"""HL Tau (ALMA 2014 LBC science verification): apply the released self-calibration tables and write a self-calibrated measurement set.

    dsharp_hltau_applycal.py --target hltau_b6|hltau_b7 --root DIR --out DIR     (CASA environment with casatasks)

``--root`` is the directory ``dsharp_fetch.py ms`` filled (``<root>/<target>/`` holds the calavg MS and ``calibration/``). The steps are
those of the CASA imaging scripts the release points to (``HLTau_Band6_Cont_Imaging_5.1.py``, ``HLTau_Band7_Cont_Imaging_5.1.py``,
casaguides.nrao.edu): the flags the script sets before the final ``applycal``; ``applycal(gaintable=['pcal2', 'apcal'],
spwmap=[[0,0,0,0],[0,0,0,0]], calwt=False)``; the flag after it (Band 6); then ``split(datacolumn='corrected')`` to
``<out>/<target>/<target>_selfcal.ms``, whose DATA column is then the self-calibrated data (so the read stage uses ``DATA``). The table
names and flags are in ``crosscheck.dsharp`` per target. The record ``applycal.json`` lists the tables, the flags and the fraction of
samples flagged before and after.
"""

import argparse
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crosscheck import dsharp as ds  # noqa: E402


def flagged_fraction(ms):
    import casatools

    tb = casatools.table()
    tb.open(str(ms))
    f = tb.getcol("FLAG")
    tb.close()
    return float(f.mean())


def run(key, root, out):
    from casatasks import applycal, flagdata, split

    t = ds.target(key)
    if t["kind"] != "hltau":
        raise SystemExit(f"{key}: DSHARP measurement sets already hold the released self-calibration")
    src = pathlib.Path(root) / key / t["ms"]["member"]
    cal = pathlib.Path(root) / key / t["ms"]["calibration"]
    tables = [cal / n for n in t["gaintables"]]
    missing = [str(p) for p in tables if not p.exists()]
    if missing:
        found = sorted(p.name for p in cal.iterdir()) if cal.exists() else []
        raise SystemExit(f"missing self-calibration tables {missing}; {cal} holds {found}: check the names in crosscheck.dsharp against the release's imaging script")
    out = pathlib.Path(out) / key
    out.mkdir(parents=True, exist_ok=True)
    work, final = out / f"{key}_work.ms", out / f"{key}_selfcal.ms"
    for p in (work, final):
        shutil.rmtree(p, ignore_errors=True)
    shutil.copytree(src, work)  # the released MS stays as downloaded
    rec = dict(target=key, source=str(src), tables=[str(p) for p in tables], gaintables=t["gaintables"], spwmap=t["spwmap"], calwt=False,
               flags_before=t["flags_before"], flags_after=t["flags_after"], flagged_fraction_start=flagged_fraction(work))
    for fl in t["flags_before"]:
        flagdata(vis=str(work), flagbackup=False, **fl)
    rec["flagged_fraction_before_applycal"] = flagged_fraction(work)
    applycal(vis=str(work), spwmap=t["spwmap"], gaintable=[str(p) for p in tables], gainfield="", calwt=False, flagbackup=False)
    for fl in t["flags_after"]:
        flagdata(vis=str(work), flagbackup=False, **fl)
    rec["flagged_fraction_after"] = flagged_fraction(work)
    split(vis=str(work), outputvis=str(final), datacolumn="corrected")
    shutil.rmtree(work)
    rec["output"] = str(final)
    (out / "applycal.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", required=True, choices=ds.HL_TAU_HEADLINE)
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    run(a.target, a.root, a.out)


if __name__ == "__main__":
    main()
