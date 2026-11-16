"""Fetch the public ALMA data of the DSHARP / HL Tau campaign, with size and SHA-256 checks.

    dsharp_fetch.py list
    dsharp_fetch.py images --out DIR [--target KEY ...]    # the released CLEAN images (small: 9-64 MB each)
    dsharp_fetch.py ms --target KEY --out DIR              # the measurement set (0.7-1.9 GB); run on an OzSTAR data node

Files land in ``DIR/<target>/``. The first fetch of a URL records its SHA-256 in ``data/dsharp/checksums.json`` (commit it); later fetches
verify it (``crosscheck.dsharp``; terms in ``DATA_ATTRIBUTION.md``). Nothing downloaded is executed.
"""

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crosscheck import dsharp as ds  # noqa: E402


def cmd_list(args):
    for k in ds.CAMPAIGN:
        t = ds.TARGETS[k]
        print(f"{k:9s} {t['name']:26s} ms {t['ms']['bytes'] / 1e9:5.2f} GB  image {t['image']['bytes'] / 1e6:6.1f} MB  {t['ms']['url']}")


def cmd_images(args):
    for k in args.target or ds.CAMPAIGN:
        path, status = ds.fetch_image(k, args.out)
        print(f"{k}: {path} ({status})", flush=True)


def cmd_ms(args):
    t = ds.target(args.target)
    known = ds.read_checksums()  # the committed table: verified, never written here (a pinned snapshot is read-only)
    ms = t["ms"]
    root = pathlib.Path(args.out) / args.target
    done = root / ms["member"] / "table.dat"
    if done.exists():
        print(f"{done.parent} is already there: nothing to download")
        return
    tgz = root / pathlib.PurePosixPath(ms["url"]).name
    ds.download(ms["url"], tgz, ms["bytes"])
    if ms["url"] in known and ds.sha256_file(tgz) != known[ms["url"]]["sha256"]:
        raise SystemExit(f"{tgz}: sha256 differs from data/dsharp/checksums.json")
    print(f"{args.target}: {ds.record_checksum(ms['url'], tgz, ms['bytes'], args.checksums)}", flush=True)
    keep = [ms["member"]] + ([ms["calibration"]] if ms.get("calibration") else [])
    ds.safe_extract(tgz, root, members=keep)
    if not done.exists():
        raise SystemExit(f"no {done} after extracting {tgz}")
    tgz.unlink()
    print(json.dumps(dict(target=args.target, ms=str(done.parent), calibration=ms.get("calibration"))))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    q = sub.add_parser("images")
    q.add_argument("--out", required=True)
    q.add_argument("--target", nargs="*", choices=ds.CAMPAIGN)
    q.set_defaults(fn=cmd_images)
    q = sub.add_parser("ms")
    q.add_argument("--checksums", default=str(ds.CHECKSUMS), help="the table to record a first fetch in (on OzSTAR: a file on /fred, since the snapshot is read-only)")
    q.add_argument("--out", required=True)
    q.add_argument("--target", required=True, choices=ds.CAMPAIGN)
    q.set_defaults(fn=cmd_ms)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
