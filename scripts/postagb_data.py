"""Stage the public post-AGB PIONIER data used by docs/imaging/postagb.md.

    python scripts/postagb_data.py fetch [--targets IW_Car ...] [--root DIR]
    python scripts/postagb_data.py check [--root DIR]

``fetch`` lists the Phase 3 PIONIER products of ESO programme 1103.D-0797 (PI
Kluska) from the ESO TAP service, downloads them into ``<root>/<target>/<collection>/``
(collection is ESO's ``PIONIER`` or ``INSPIRING`` Phase 3 release; the
INSPIRING one adds nights the first release lacks), and downloads the De Prins
et al. (2026) Zenodo record (10.5281/zenodo.20155600: ORGANIC images and
observation logs, CC-BY-4.0) into ``<root>/literature/``. Downloads go into
their own directories and are only read with ``python -I``.

``check`` confirms that, for each target with a local reduced copy (IW Car:
``~/code/nuHor/data/IW_Car_PIONIER_data_all``), the archive files cover exactly the
same exposures (MJD) with the same baselines and wavelengths.

Targets are keyed by name below; add a row to TARGETS to stage another one.
"""

import argparse
import csv
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path("~/data/postagb_pionier").expanduser()
TAP = "https://archive.eso.org/tap_obs/sync"
PORTAL = "https://dataportal.eso.org/dataPortal/file/"
ZENODO = "https://zenodo.org/api/records/20155600"  # concept DOI; redirects to the latest version
PROGRAMME = "1103.D-0797"

# ESO target_name patterns (observers spelled them several ways) and the
# local reduced copy, if we have one.
TARGETS = {
    "IW_Car": dict(like=["IW%Car"], local="~/code/nuHor/data/IW_Car_PIONIER_data_all"),
    "HR4049": dict(like=["HR%4049"], local="~/code/nuHor/data/HR4049_pionier_data_all"),
    "HD108015": dict(like=["HD%108015"], local=None),
    "AI_Sco": dict(like=["AI%Sco"], local=None),
    "EN_TrA": dict(like=["EN%TrA"], local=None),
    "HD95767": dict(like=["HD%95767"], local=None),
    "IRAS15469-5311": dict(like=["IRAS%15469%"], local=None),
    "PS_Gem": dict(like=["PS%Gem"], local=None),
    "IRAS08544-4431": dict(like=["IRAS%08544%"], local=None),
}


def _get(url, params=None, timeout=600):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    for attempt in range(5):  # the archive drops connections now and then
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except OSError:
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))


def list_products(target):
    """Phase 3 PIONIER rows of programme 1103.D-0797 for one target (list of dicts)."""
    like = " OR ".join(f"target_name LIKE '{p}'" for p in TARGETS[target]["like"])
    query = (
        "SELECT dp_id, obs_collection, target_name, t_min, proposal_id, obs_release_date "
        "FROM ivoa.ObsCore WHERE instrument_name='PIONIER' "
        f"AND proposal_id LIKE '{PROGRAMME}%' AND ({like}) ORDER BY t_min"
    )
    text = _get(TAP, dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=query)).decode()
    return list(csv.DictReader(io.StringIO(text)))


def download(dp_id, outdir):
    """One Phase 3 file to ``outdir`` (skipped when present); returns its path."""
    path = Path(outdir) / (dp_id.replace(":", "_") + ".fits")
    if path.exists():
        return path
    data = _get(PORTAL + dp_id)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.rename(path)
    return path


def fetch_target(target, root):
    rows = list_products(target)
    if not rows:
        print(f"{target}: no Phase 3 rows found")
        return
    by_collection = {}
    for r in rows:
        by_collection.setdefault(r["obs_collection"], []).append(r)
    for coll, rs in by_collection.items():
        outdir = Path(root) / target / coll
        outdir.mkdir(parents=True, exist_ok=True)
        for r in rs:
            download(r["dp_id"], outdir)
        (outdir / "manifest.csv").write_text(
            "dp_id,target_name,mjd,proposal_id,release\n"
            + "".join(
                f"{r['dp_id']},{r['target_name']},{r['t_min']},{r['proposal_id']},{r['obs_release_date']}\n"
                for r in rs
            )
        )
        print(f"{target}/{coll}: {len(rs)} files -> {outdir}")


def fetch_zenodo(root):
    outdir = Path(root) / "literature"
    outdir.mkdir(parents=True, exist_ok=True)
    rec = json.loads(_get(ZENODO))
    (outdir / "record.json").write_text(json.dumps(
        {k: rec[k] for k in ("doi", "conceptdoi", "created") if k in rec}
        | {"title": rec["metadata"]["title"], "license": rec["metadata"].get("license")}, indent=1))
    for f in rec["files"]:
        zpath = outdir / f["key"]
        if not zpath.exists():
            zpath.write_bytes(_get(f["links"]["self"]))
        if zpath.suffix == ".zip":
            dest = (outdir / zpath.stem).resolve()
            with zipfile.ZipFile(zpath) as z:
                for name in z.namelist():  # refuse paths that escape dest
                    if not (dest / name).resolve().is_relative_to(dest):
                        raise ValueError(f"unsafe path in zip: {name}")
                z.extractall(dest)
            print(f"zenodo {rec['doi']}: {len(z.namelist())} members -> {dest}")


def fetch(args):
    for t in args.targets:
        fetch_target(t, args.root)
    fetch_zenodo(args.root)


def _summary(path):
    """(MJD-OBS, sorted baseline station pairs, wavelengths, V2 table size) of one OIFITS file."""
    from astropy.io import fits
    import numpy as np

    with fits.open(path) as h:
        mjd = float(h[0].header["MJD-OBS"])
        v2 = h["OI_VIS2"].data
        wl = np.asarray(h["OI_WAVELENGTH"].data["EFF_WAVE"], float)
        sta = sorted(tuple(int(s) for s in row) for row in v2["STA_INDEX"])
        uv = np.sort(np.hypot(v2["UCOORD"], v2["VCOORD"]))
        return mjd, sta, wl, uv, np.asarray(v2["VIS2DATA"], float)


def check_target(target, root):
    import numpy as np

    local = TARGETS[target]["local"]
    if local is None:
        print(f"{target}: no local copy to compare with")
        return True
    lf = sorted(Path(local).expanduser().glob("*_oidataCalibrated.fits"))
    loc = {round(_summary(f)[0], 4): f for f in lf}
    arch = {}
    for coll in sorted((Path(root) / target).glob("*/")):
        for f in sorted(coll.glob("*.fits")):
            arch.setdefault(round(_summary(f)[0], 4), []).append(f)
    ok = set(loc) == set(arch)
    print(f"{target}: {len(lf)} local files, {len(arch)} archive exposures; "
          f"same MJDs: {set(loc) == set(arch)}")
    for m in sorted(set(loc) ^ set(arch)):
        print("  only in", "local" if m in loc else "archive", m)
    ndiff = 0
    for m in sorted(set(loc) & set(arch)):
        a = _summary(loc[m])
        for f in arch[m]:
            b = _summary(f)
            same = (a[1] == b[1] and np.allclose(a[2], b[2], rtol=1e-4)
                    and np.allclose(a[3], b[3], rtol=1e-6, atol=1e-3))
            vd = np.nanmax(np.abs(a[4] - b[4])) if a[4].shape == b[4].shape else np.inf
            if not same:
                ndiff += 1
                ok = False
                print(f"  MJD {m}: baselines/wavelengths differ in {f.parent.name}/{f.name}")
            elif vd > 1e-6:
                print(f"  MJD {m}: same geometry, V2 differs by up to {vd:.3g} in {f.parent.name}")
    print(f"  baselines and wavelengths: {'match' if ndiff == 0 else str(ndiff) + ' mismatches'}")
    return ok


def check(args):
    ok = all(check_target(t, args.root) for t in args.targets)
    sys.exit(0 if ok else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("fetch", fetch), ("check", check)):
        s = sub.add_parser(name)
        s.add_argument("--targets", nargs="+", default=["IW_Car"], choices=list(TARGETS))
        s.add_argument("--root", type=Path, default=ROOT)
        s.set_defaults(fn=fn)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
