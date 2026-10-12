"""The advice dashboard (scripts/contest_advice.py): CSV round-trip, and the
server refuses paths outside the allowed directories."""

import importlib.util
import json
import pathlib
import threading
import urllib.error
import urllib.request

import pytest
from http.server import ThreadingHTTPServer

ROOT = pathlib.Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("contest_advice", ROOT / "scripts" / "contest_advice.py")
ca = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ca)


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_csv_round_trip(tmp_path):
    path = tmp_path / "a.csv"
    store = {("2004", "2004_data1", "data1"): {"advice": 'line one, "quoted"\nline two é', "updated": "t1"},
             ("2006", "", "MIR"): {"advice": "  ", "updated": "t2"}}
    ca.save_advice(store, path)
    back = ca.load_advice(path)
    assert back == {("2004", "2004_data1", "data1"): store[("2004", "2004_data1", "data1")]}
    assert ca.load_advice(tmp_path / "none.csv") == {}


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_safe_path_and_server(tmp_path):
    figs, secret = tmp_path / "figures", tmp_path / "secret"
    (figs / "abc").mkdir(parents=True)
    secret.mkdir()
    (figs / "abc" / "p.png").write_bytes(b"png")
    (secret / "s.png").write_bytes(b"nope")
    roots = [("crop", figs)]
    assert ca.safe_path(roots, "crop/abc/p.png") == (figs / "abc" / "p.png").resolve()
    for bad in ["crop/../secret/s.png", "crop/abc/../../secret/s.png", "/etc/passwd", "crop//etc/passwd",
                "other/abc/p.png", "crop/abc/missing.png"]:
        assert ca.safe_path(roots, bad) is None
    (figs / "link.png").symlink_to(secret / "s.png")
    assert ca.safe_path(roots, "crop/link.png") is None
    state = {"rows": [], "store": {}, "csv": tmp_path / "out.csv", "roots": roots, "keys": {("2004", "", "x")}}
    server = ThreadingHTTPServer(("127.0.0.1", 0), ca.make_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        assert urllib.request.urlopen(f"{base}/crop/abc/p.png").read() == b"png"
        for bad in ["/crop/../secret/s.png", "/crop/%2e%2e/secret/s.png", "/crop/link.png"]:
            try:
                urllib.request.urlopen(base + bad)
                raise AssertionError(bad)
            except urllib.error.HTTPError as e:
                assert e.code == 404
        post = urllib.request.Request(f"{base}/api/advice", method="POST",
                                      data=json.dumps({"year": "2004", "label": "", "name": "x", "advice": "hi"}).encode())
        urllib.request.urlopen(post)
        assert ca.load_advice(tmp_path / "out.csv")[("2004", "", "x")]["advice"] == "hi"
    finally:
        server.shutdown()
