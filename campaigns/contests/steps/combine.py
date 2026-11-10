"""Combine one dataset's ensemble members: scripts/combine_ensemble.py on exactly the members this order was given.

Reads the work order for the member run folders, runs the repository's combine script on them, and writes a
result.json from its table row: `members_kept` passes when at least one member survives the chi2 filter, and the
status is `warn` when the best chi2/N is above 1.5. Needs what combine_ensemble.py needs (NumPy, matplotlib).
"""

import argparse
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]


def row_of(table, label):
    """The cells of a dataset's row of ensemble.md: dataset | kept | best chi2/N | median | error scale | dlogZ."""
    for line in table.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0].strip("`* ") == label:
            return cells
    return None


def number(text):
    try:
        return float(str(text).split("/")[0].split()[0])
    except (ValueError, IndexError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--order", required=True)
    order = json.loads(pathlib.Path(ap.parse_args(argv).order).read_text())
    out, label = pathlib.Path(order["out"]), order["key"]["label"]
    folders = [m["path"] for m in order["inputs"]["member"]]
    subprocess.run([sys.executable, str(REPO / "scripts" / "combine_ensemble.py"), *folders, "--out", str(out)], check=True)
    cells = row_of((out / "ensemble.md").read_text(), label)
    kept, best = (number(cells[1]), number(cells[2])) if cells and len(cells) > 2 else (None, None)
    ok = bool(kept) and kept >= 1
    status = "fail" if not ok else ("warn" if best is None or best > 1.5 else "pass")
    result = {"status": status, "checks": {"members_kept": "pass" if ok else "fail"}, "members_given": len(folders),
              "members_kept": kept, "best_chi2_red": best, "row": cells}
    (out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
