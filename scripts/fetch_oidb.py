#!/usr/bin/env python3
"""List, size, fetch and verify OIFITS collections from the JMMC OiDB.

Standard library only: the fetch runs on OzSTAR's trevor nodes, whose
Python has nothing else installed. See design/plan_oidb.md (Stage O0) and
oidb/manifest.yml.

How OiDB serves files (checked 2026-10-07):

* ``search.html?collection=~<id>&perpage=N&page=P`` is rendered on the
  server. Each granule row (``<tr data-template="app:each-row" ...>``)
  carries ``data-access_url`` (``get-data.html?id=<granule>&name=/<file>``),
  ``data-calib_level``, ``data-bib_reference``, ``data-target_name`` and
  ``data-id``; its cells give the instrument, the time and the dataPI.
* ``get-data.html`` answers HEAD with 400 and GET with ``303 See Other`` to
  ``/exist/apps/oidb-data/oifits/staging/<uuid>/<file>``, which answers HEAD
  with 200 and a Content-Length. No login or cookie is needed for public
  collections.
* OiDB refuses non-browser clients, so requests to it send a browser-like
  User-Agent (decided by Ben, 2026-10-07).
* Collections imported from VizieR (``J/A+A/...``) redirect instead to
  ``cdsarc.cds.unistra.fr``, which serves the file to a plain client and
  answers a browser User-Agent with a bot challenge; requests there send an
  honest User-Agent, and no challenge is ever answered.

Subcommands::

    fetch_oidb.py scan <collection-id>        # granules, levels, dataPIs, files and sizes (JSON)
    fetch_oidb.py ids --stage O1              # collection ids of a stage, one per line
    fetch_oidb.py fetch <collection-id> DEST  # download the manifest's files, check sizes and sha256
    fetch_oidb.py verify <collection-id> DEST # re-hash DEST without the network
    fetch_oidb.py record <collection-id> SUMS # copy a fetch's sha256 sums into the manifest

``scan`` and ``fetch`` touch the network; ``scan`` downloads no file
bodies (listing pages, redirects and HEAD requests only).
"""

import argparse
import hashlib
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://oidb.jmmc.fr"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126 Safari/537.36"
)
HONEST_AGENT = "virgil-validation-fetch_oidb (Python-urllib; github.com/benjaminpope/virgil-validation)"
MANIFEST = Path(__file__).resolve().parents[1] / "oidb" / "manifest.yml"
PAUSE = 0.5  # seconds between requests, to be gentle on OiDB
PERPAGE = 100


# ---------------------------------------------------------------- manifest


def load_manifest(path=MANIFEST):
    """The manifest is JSON (a subset of YAML) after a header of '#' comments."""
    text = Path(path).read_text()
    body = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return json.loads(body)


def save_manifest(manifest, path=MANIFEST):
    text = Path(path).read_text()
    header = []
    for line in text.splitlines():
        if not line.lstrip().startswith("#"):
            break
        header.append(line)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text("\n".join(header) + "\n" + json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def collection(manifest, cid):
    for c in manifest["collections"]:
        if c["id"] == cid:
            return c
    raise SystemExit(f"{cid}: not in the manifest")


# ---------------------------------------------------------------- network


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


_opener = urllib.request.build_opener()
_no_redirect = urllib.request.build_opener(_NoRedirect)


def _request(url, method="GET"):
    # OiDB needs a browser User-Agent. Collections imported from VizieR
    # (J/...) redirect to CDS, which serves files to plain clients but
    # answers a browser User-Agent with a bot challenge (not to be solved),
    # so other hosts get an honest one.
    oidb = urllib.parse.urlparse(url).hostname == urllib.parse.urlparse(BASE).hostname
    return urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT if oidb else HONEST_AGENT})


def _retry(fn, retries=4):
    """OiDB sometimes refuses or drops a connection; wait and try again."""
    for attempt in range(retries):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries - 1:
                raise
        time.sleep(10 * (attempt + 1))
    raise AssertionError


def get_text(url):
    def once():
        with _opener.open(_request(url), timeout=120) as r:
            return r.read().decode("utf-8", "replace")

    return _retry(once)


def staging_url(get_data_url):
    """Follow get-data.html's 303 by hand (no body is read)."""

    def once():
        try:
            with _no_redirect.open(_request(get_data_url), timeout=120) as r:
                raise RuntimeError(f"{get_data_url}: expected a redirect, got {r.status}")
        except urllib.error.HTTPError as e:
            if e.code not in (301, 302, 303, 307, 308):
                raise
            return urllib.parse.urljoin(BASE, e.headers["Location"])

    return _retry(once)


def content_length(url):
    def once():
        with _opener.open(_request(url, "HEAD"), timeout=120) as r:
            n = r.headers["Content-Length"]
            if n is None or r.headers.get("Content-Type", "").startswith("text/html"):
                raise RuntimeError(f"{url}: no Content-Length, or a web page ({r.headers.get('Content-Type')})")
            return int(n)

    return _retry(once)


# ---------------------------------------------------------------- scan

_ROW = re.compile(r'<tr data-template="app:each-row"(.*?)</tr>', re.S)
_ATTR = re.compile(r'data-([a-z_]+)="([^"]*)"')
_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S)


def _text(cell):
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", cell))).strip()


def parse_search_page(page):
    """Granule rows of one search.html page."""
    rows = []
    for m in _ROW.finditer(page):
        attrs = {k: html.unescape(v) for k, v in _ATTR.findall(m.group(1).split(">", 1)[0])}
        cells = [_text(c) for c in _CELL.findall(m.group(1))]
        # cells: [menu, L, target_name, access_url, t_min, instrument, wlen_min, wlen_max, nb_channels, datapi]
        rows.append(
            {
                "granule": int(attrs["id"]),
                "level": int(attrs["calib_level"]),
                "target": attrs.get("target_name", ""),
                "bibcode": attrs.get("bib_reference") or None,
                "get_data": attrs["access_url"],
                "file": urllib.parse.parse_qs(urllib.parse.urlparse(attrs["access_url"]).query)["name"][0].lstrip("/"),
                "t_min": cells[4] if len(cells) > 4 else None,
                "instrument": cells[5] if len(cells) > 5 else None,
                "datapi": cells[9] if len(cells) > 9 else None,
            }
        )
    pages = re.search(r"Page \d+ / (\d+)", page)
    return rows, int(pages.group(1)) if pages else 1


def scan(cid, sizes=True, log=sys.stderr):
    """Every granule of a collection, grouped into files, with sizes."""
    granules, page, npages = [], 1, 1
    while page <= npages:
        q = urllib.parse.urlencode({"collection": "~" + urllib.parse.unquote(cid), "perpage": PERPAGE, "page": page})
        rows, npages = parse_search_page(get_text(f"{BASE}/search.html?{q}"))
        granules += rows
        page += 1
        time.sleep(PAUSE)
    files = {}
    for g in granules:
        f = files.setdefault(g["file"], {"name": g["file"], "granules": [], "get_data": g["get_data"]})
        f["granules"].append(g["granule"])
    out = []
    for i, f in enumerate(sorted(files.values(), key=lambda f: f["name"])):
        if sizes:
            f["url"] = staging_url(f["get_data"])
            time.sleep(PAUSE)
            f["bytes"] = content_length(f["url"])
            time.sleep(PAUSE)
            print(f"  {i + 1}/{len(files)} {f['name']} {f['bytes']}", file=log)
        f["sha256"] = None
        out.append(f)
    return {
        "levels": sorted({g["level"] for g in granules}),
        "datapis": sorted({g["datapi"] for g in granules if g["datapi"]}),
        "bibcodes": sorted({g["bibcode"] for g in granules if g["bibcode"]}),
        "instruments": sorted({g["instrument"] for g in granules if g["instrument"]}),
        "targets": sorted({g["target"] for g in granules}),
        "n_granules": len(granules),
        "files": out,
        "total_bytes": sum(f.get("bytes", 0) for f in out) if sizes else None,
    }


# ---------------------------------------------------------------- fetch


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url, dest, expected_bytes):
    part = dest.with_name(dest.name + ".part")
    req = _request(url)
    with _opener.open(req, timeout=300) as r, open(part, "wb") as fh:
        for block in iter(lambda: r.read(1 << 20), b""):
            fh.write(block)
        fh.flush()
        os.fsync(fh.fileno())
    got = part.stat().st_size
    if expected_bytes is not None and got != expected_bytes:
        part.unlink()
        raise RuntimeError(f"{dest.name}: {got} bytes, manifest says {expected_bytes}")
    with open(part, "rb") as fh:
        if fh.read(6) != b"SIMPLE":
            part.unlink()
            raise RuntimeError(f"{dest.name}: not a FITS file")
    os.replace(part, dest)


def check(c, dest, fetch):
    """Download (if fetch) and hash every file; return (sums, problems)."""
    dest.mkdir(parents=True, exist_ok=True)
    sums, problems = {}, []
    for f in c["files"]:
        path = dest / f["name"]
        try:
            if fetch and not (path.exists() and path.stat().st_size == f["bytes"]):
                # staging URLs may expire: resolve get-data.html afresh
                _retry(lambda: download(staging_url(f["get_data"]), path, f.get("bytes")))
                time.sleep(PAUSE)
            if not path.exists():
                problems.append(f"{f['name']}: missing")
                continue
            digest = sha256(path)
            sums[f["name"]] = digest
            if f.get("sha256") and f["sha256"] != digest:
                problems.append(f"{f['name']}: sha256 {digest} != manifest {f['sha256']}")
        except Exception as e:  # report every file, then fail
            problems.append(f"{f['name']}: {e}")
    tmp = dest / "sha256sums.txt.tmp"
    tmp.write_text("".join(f"{d}  {n}\n" for n, d in sorted(sums.items())))
    os.replace(tmp, dest / "sha256sums.txt")
    return sums, problems


# ---------------------------------------------------------------- CLI


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="list a collection's granules and files, with sizes (no file bodies)")
    s.add_argument("collection")
    s.add_argument("--no-sizes", action="store_true")
    s = sub.add_parser("ids", help="collection ids of one stage")
    s.add_argument("--stage", required=True)
    for name in ("fetch", "verify"):
        s = sub.add_parser(name)
        s.add_argument("collection")
        s.add_argument("dest", type=Path)
    s = sub.add_parser("record", help="write a fetch's sha256sums.txt into the manifest")
    s.add_argument("collection")
    s.add_argument("sums", type=Path)
    for s in sub.choices.values():
        s.add_argument("--manifest", type=Path, default=MANIFEST)
    a = p.parse_args(argv)

    if a.cmd == "scan":
        json.dump(scan(a.collection, sizes=not a.no_sizes), sys.stdout, indent=2)
        print()
        return 0
    manifest = load_manifest(a.manifest)
    if a.cmd == "ids":
        for c in manifest["collections"]:
            if c["stage"] == a.stage:
                print(c["id"])
        return 0
    c = collection(manifest, a.collection)
    if a.cmd == "record":
        sums = dict(reversed(line.split(None, 1)) for line in a.sums.read_text().splitlines() if line.strip())
        sums = {n.strip(): d for n, d in sums.items()}
        for f in c["files"]:
            if f["name"] not in sums:
                raise SystemExit(f"{f['name']}: not in {a.sums}")
            if f.get("sha256") and f["sha256"] != sums[f["name"]]:
                raise SystemExit(f"{f['name']}: recorded sha256 differs; a changed file is a failure")
            f["sha256"] = sums[f["name"]]
        save_manifest(manifest, a.manifest)
        return 0
    sums, problems = check(c, a.dest, fetch=a.cmd == "fetch")
    total = sum((a.dest / n).stat().st_size for n in sums)
    print(f"{a.collection}: {len(sums)}/{len(c['files'])} files, {total} bytes, sums in {a.dest / 'sha256sums.txt'}")
    for line in problems:
        print("PROBLEM", line, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
