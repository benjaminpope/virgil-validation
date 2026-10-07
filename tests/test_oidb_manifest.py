"""The OiDB manifest and reference values (design/plan_oidb.md, stage O0).

No network: these check the manifest's shape, that every O1 collection has
published reference values with their sources, and the fetcher's parsing and
bookkeeping on hand-made inputs.
"""

import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ROOT / "oidb" / "references"

_spec = importlib.util.spec_from_file_location("fetch_oidb", ROOT / "scripts" / "fetch_oidb.py")
fetch_oidb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_oidb)

MANIFEST = fetch_oidb.load_manifest()
COLLECTIONS = MANIFEST["collections"]


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_manifest_shape():
    ids = [c["id"] for c in COLLECTIONS]
    assert len(ids) == len(set(ids))
    for c in COLLECTIONS:
        assert c["stage"] in ("O1", "O2"), c["id"]
        assert c["levels"] and set(c["levels"]) <= {1, 2, 3}, c["id"]
        assert c["datapis"], c["id"]
        assert len(c["terms"]) == len(c["levels"]), c["id"]
        pub = c["publication"]
        assert pub["status"] in ("published", "no paper"), c["id"]
        if pub["status"] == "published":
            assert pub["doi"] and pub["doi"] in pub["evidence"], c["id"]
        names = [f["name"] for f in c["files"]]
        assert len(names) == len(set(names)) == c["n_files"], c["id"]
        for f in c["files"]:
            assert f["name"].endswith((".fits", ".oifits")) and "/" not in f["name"], f
            assert f["get_data"].startswith("https://oidb.jmmc.fr/get-data.html?id="), f
            if f["bytes"] is None:  # listed but not yet sized: allowed only after O1
                assert c["stage"] != "O1" and c["total_bytes"] is None, f
                continue
            assert isinstance(f["bytes"], int) and f["bytes"] > 0 and f["bytes"] % 2880 == 0, f
            assert f["sha256"] is None or re.fullmatch(r"[0-9a-f]{64}", f["sha256"]), f
        if c["total_bytes"] is not None:
            assert c["total_bytes"] == sum(f["bytes"] for f in c["files"]), c["id"]


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_o1_references_have_sources():
    for c in COLLECTIONS:
        if c["stage"] != "O1":
            continue
        ref = json.loads((REFERENCES / f"{c['id']}.json").read_text())
        assert ref["collection_id"] == c["id"]
        papers = [ref["paper"]] if "paper" in ref else [t["paper"] for t in ref["targets"].values() if "paper" in t]
        assert papers, c["id"]
        for p in papers:
            assert p.get("doi") or p.get("bibcode"), c["id"]
        if c["publication"]["doi"]:
            assert c["publication"]["doi"] in json.dumps(ref), c["id"]


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_parse_search_page():
    row = (
        '<tr data-template="app:each-row" data-access_url="get-data.html?id=7&amp;name=/a_b.fits" '
        'data-bib_reference="2024Natur.634.1070X" data-calib_level="3" data-id="7" data-target_name="Gl 229 B">'
        "<td>m</td><td>L3</td><td>Gl 229 B</td><td>x</td><td>2023-12-26</td><td>GRAVITY_SC</td>"
        "<td>2.0</td><td>2.4</td><td>210</td><td>XUAN</td></tr>"
    )
    rows, npages = fetch_oidb.parse_search_page(f"<table>{row}{row}</table> Page 1 / 3")
    assert npages == 3 and len(rows) == 2
    assert rows[0] == {
        "granule": 7, "level": 3, "target": "Gl 229 B", "bibcode": "2024Natur.634.1070X",
        "get_data": "https://oidb.jmmc.fr/get-data.html?id=7&name=/a_b.fits", "file": "a_b.fits",
        "t_min": "2023-12-26", "instrument": "GRAVITY_SC", "datapi": "XUAN",
    }


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_verify_and_record(tmp_path):
    body = b"SIMPLE  =                    T" + b" " * (2880 - 30)
    digest = hashlib.sha256(body).hexdigest()
    c = {"id": "x", "stage": "O1", "files": [{"name": "a.fits", "get_data": "", "bytes": len(body), "sha256": None}]}
    manifest = tmp_path / "manifest.yml"
    manifest.write_text("# header\n" + json.dumps({"collections": [c]}))
    data = tmp_path / "x"
    data.mkdir()
    (data / "a.fits").write_bytes(body)

    assert fetch_oidb.main(["verify", "x", str(data), "--manifest", str(manifest)]) == 0
    assert (data / "sha256sums.txt").read_text() == f"{digest}  a.fits\n"
    assert fetch_oidb.main(["record", "x", str(data / "sha256sums.txt"), "--manifest", str(manifest)]) == 0
    assert manifest.read_text().startswith("# header\n")
    assert fetch_oidb.load_manifest(manifest)["collections"][0]["files"][0]["sha256"] == digest

    (data / "a.fits").write_bytes(body[:-1] + b"!")  # a changed file must fail
    assert fetch_oidb.main(["verify", "x", str(data), "--manifest", str(manifest)]) == 1


def _one_file_manifest(tmp_path, body, sha=None):
    c = {"id": "x", "stage": "O1",
         "files": [{"name": "a.fits", "get_data": "https://oidb.jmmc.fr/get-data.html?id=1&name=/a.fits",
                    "bytes": len(body), "sha256": sha}]}
    manifest = tmp_path / "manifest.yml"
    manifest.write_text("# header\n" + json.dumps({"collections": [c]}))
    data = tmp_path / "x"
    data.mkdir()
    return manifest, data


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
@pytest.mark.parametrize("bad", [b"SIMPLE  =" + b" " * 100, b"<html>" + b" " * 2874])
def test_verify_rejects_wrong_size_or_not_fits(tmp_path, bad):
    """With no recorded sum, a file of the wrong size or without a FITS header
    fails verify and is left out of sha256sums.txt, so record can't trust it."""
    manifest, data = _one_file_manifest(tmp_path, b"SIMPLE" + b" " * 2874)
    (data / "a.fits").write_bytes(bad)
    assert fetch_oidb.main(["verify", "x", str(data), "--manifest", str(manifest)]) == 1
    assert (data / "sha256sums.txt").read_text() == ""
    with pytest.raises(SystemExit):
        fetch_oidb.main(["record", "x", str(data / "sha256sums.txt"), "--manifest", str(manifest)])


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_fetch_redownloads_unrecorded_files(tmp_path, monkeypatch):
    """An existing file of the right size is fetched again unless its sha256
    matches one already recorded."""
    good = b"SIMPLE  =                    T" + b" " * (2880 - 30)
    stale = b"SIMPLE  =                    F" + b" " * (2880 - 30)
    calls = []

    def fake_download(url, dest, expected_bytes):
        calls.append(url)
        dest.write_bytes(good)

    monkeypatch.setattr(fetch_oidb, "staging_url", lambda u: u)
    monkeypatch.setattr(fetch_oidb, "download", fake_download)
    monkeypatch.setattr(fetch_oidb, "PAUSE", 0)

    manifest, data = _one_file_manifest(tmp_path, good)
    (data / "a.fits").write_bytes(stale)
    assert fetch_oidb.main(["fetch", "x", str(data), "--manifest", str(manifest)]) == 0
    assert len(calls) == 1 and (data / "a.fits").read_bytes() == good

    (tmp_path / "rec").mkdir()
    manifest, data = _one_file_manifest(tmp_path / "rec", good, sha=hashlib.sha256(good).hexdigest())
    (data / "a.fits").write_bytes(good)
    assert fetch_oidb.main(["fetch", "x", str(data), "--manifest", str(manifest)]) == 0
    assert len(calls) == 1  # recorded and matching: reused


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_scan_urls_are_absolute(monkeypatch):
    """parse_search_page makes get-data links absolute, so scan can open them."""
    row = ('<tr data-template="app:each-row" data-access_url="get-data.html?id=9&amp;name=/b.fits" '
           'data-calib_level="3" data-id="9" data-target_name="t"><td></td></tr>')
    seen = []
    monkeypatch.setattr(fetch_oidb, "get_text", lambda url: row)
    monkeypatch.setattr(fetch_oidb, "staging_url", lambda u: seen.append(u) or u)
    monkeypatch.setattr(fetch_oidb, "content_length", lambda u: 2880)
    monkeypatch.setattr(fetch_oidb, "PAUSE", 0)
    out = fetch_oidb.scan("x", log=open(os.devnull, "w"))
    assert seen == ["https://oidb.jmmc.fr/get-data.html?id=9&name=/b.fits"]
    assert out["files"][0]["get_data"] == seen[0] and out["total_bytes"] == 2880
