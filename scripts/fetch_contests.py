"""Download the imaging-contest data listed in contests/manifest.yml.

    python scripts/fetch_contests.py [--dest ~/data/imaging_contests] [--year 2010 ...]
    python scripts/fetch_contests.py --record     # write sha256s into the manifest

Files land in <dest>/<year>/ and are checked against the sha256 in the
manifest; a mismatch is an error (the hosts have changed files before: the
OiDB copies of the 2004 data differ from the OLBIN originals in their
headers). Gzipped OIFITS are unpacked next to the archive. The data are
small (about 15 MB in all) and are never committed.
"""

import argparse
import gzip
import hashlib
import pathlib
import shutil
import urllib.request

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "contests" / "manifest.yml"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_url(manifest, contest, name):
    base = contest.get("base", manifest["jmmc"])
    parts = [base, contest.get("subdir"), name]
    return "/".join(p for p in parts if p)


def fetch(url, path):
    request = urllib.request.Request(url, headers={"User-Agent": "virgil-validation"})
    with urllib.request.urlopen(request, timeout=120) as response, open(path, "wb") as out:
        shutil.copyfileobj(response, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", default="~/data/imaging_contests")
    parser.add_argument("--year", nargs="*", help="only these contests")
    parser.add_argument("--record", action="store_true", help="write the sha256 of each file into the manifest")
    args = parser.parse_args()

    text = MANIFEST.read_text()
    manifest = yaml.safe_load(text)
    dest = pathlib.Path(args.dest).expanduser()
    hashes = {}
    for year, contest in manifest["contests"].items():
        if args.year and year not in args.year:
            continue
        for entry in contest.get("files", []):
            path = dest / year / entry["name"]
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                fetch(file_url(manifest, contest, entry["name"]), path)
            digest = sha256(path)
            hashes[entry["name"]] = digest
            if not args.record:
                if "sha256" not in entry:
                    raise SystemExit(f"{year}/{entry['name']}: no sha256 in the manifest (run with --record)")
                if digest != entry["sha256"]:
                    raise SystemExit(f"{year}/{entry['name']}: sha256 {digest} != manifest {entry['sha256']}")
            if path.suffix == ".gz":
                with gzip.open(path) as src, open(path.with_suffix(""), "wb") as out:
                    shutil.copyfileobj(src, out)
            print(f"{year}/{entry['name']}  {path.stat().st_size:>9d} B  ok")

    if args.record:
        # Edit the text, not the parsed YAML, so that comments and layout survive.
        lines = []
        for line in text.splitlines():
            for name, digest in hashes.items():
                if f"{{name: {name}," in line and "sha256" not in line:
                    line = line.replace(f"{{name: {name},", f"{{name: {name}, sha256: {digest},")
            lines.append(line)
        MANIFEST.write_text("\n".join(lines) + "\n")
        print(f"recorded {len(hashes)} sha256s in {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
