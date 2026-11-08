"""Convert virgil's own JUnit evidence (its tests/conftest.py ``validates``
markers) into our evidence records, so virgil's tests count where they
already check something.

    python scripts/ingest_virgil_evidence.py junit-evidence.xml OUT.jsonl \
        --commit SHA [--run-url URL]

The first line of OUT is a run header for virgil's CI run; the records keep
virgil's roots, so the evidence page can show self-consistency as weaker.
"""

import argparse
import datetime
import json
import pathlib
import subprocess
import xml.etree.ElementTree as ET


def outcome(case):
    if case.find("failure") is not None or case.find("error") is not None:
        return "failed"
    skipped = case.find("skipped")
    if skipped is not None:
        kind = skipped.get("type", "")
        return "xfailed" if "xfail" in kind else "skipped"
    return "passed"


ROOT = pathlib.Path(__file__).resolve().parents[1]


def own_provenance(root=ROOT):
    """This repository's commit and whether tracked files have uncommitted
    changes, as the run header of our own evidence records them. Untracked
    files are ignored: in CI the evidence being built and the downloaded
    virgil artefact sit untracked in the checkout. (None, None) when it
    cannot be told, which the strict loader then refuses."""
    try:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
                                         stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                                         text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, any(not line[3:].startswith("trust/evidence/") for line in status.splitlines())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("junit")
    ap.add_argument("out")
    ap.add_argument("--commit", required=True)
    ap.add_argument("--run-url")
    args = ap.parse_args()

    root = ET.parse(args.junit).getroot()
    records = []
    for case in root.iter("testcase"):
        for prop in case.iter("property"):
            if prop.get("name") != "validates":
                continue
            claim = json.loads(prop.get("value"))
            records.append(
                {
                    "record": "test",
                    "test": f"virgil:{case.get('classname')}::{case.get('name')}",
                    **claim,
                    "outcome": outcome(case),
                    "duration_s": float(case.get("time", 0)),
                    "metrics": {},
                    "source": "virgil-ci",
                }
            )
    validation_commit, validation_dirty = own_provenance()
    header = {
        "record": "run",
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "virgil": {"version": None, "commit": args.commit, "url": "https://github.com/benjaminpope/virgil"},
        "validation_commit": validation_commit,
        "validation_dirty": validation_dirty,
        "versions": {},
        "python": None,
        "runner": "virgil-ci",
        "run_url": args.run_url,
    }
    with open(args.out, "w") as f:
        f.write(json.dumps(header) + "\n")
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"{len(records)} records from virgil CI at {args.commit[:10]}")


if __name__ == "__main__":
    main()
