"""The sealed comparison of one system: scripts/gallery_pull_back.summary_of on this order's own orbit run.

Reads the work order for the run folder of the orbit it is given (not "the newest folder"), writes the gallery
summary into the order's run folder and a result.json. Publishing to trust/campaigns/orbits/ stays a person's step:
python scripts/gallery_pull_back.py RUNS_DIR, as before. A MIRC-X system has no trackb record and is skipped with a
warning. Needs what gallery_pull_back needs (NumPy; no JAX).
"""

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--order", required=True)
    order = json.loads(pathlib.Path(ap.parse_args(argv).order).read_text())
    out, slug = pathlib.Path(order["out"]), order["key"]["slug"]
    record = REPO / "oidb" / "trackb" / f"{slug}.json"
    result = {"status": "pass", "checks": {}}
    if not record.exists():
        result = {"status": "warn", "checks": {"summary_written": "warn"}, "problem": f"no published record for {slug}"}
    else:
        import gallery_pull_back as P
        import orbit_gallery as G

        folder = pathlib.Path(order["inputs"]["orbit"][0]["path"]) / slug
        summary = P.summary_of(folder, json.loads(record.read_text()))
        (out / "summary.json").write_text(json.dumps(G._clean(summary), indent=1) + "\n")
        result["checks"]["summary_written"] = "pass"
        result["collection_id"] = summary.get("collection_id")
    (out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
