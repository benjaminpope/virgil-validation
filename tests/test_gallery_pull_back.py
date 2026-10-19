"""scripts/gallery_pull_back.py turns an OrbitPipeline run folder into a gallery summary (no JAX)."""

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import gallery_pull_back as PB  # noqa: E402

pytestmark = pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")

BEST = dict(period=100.0, dt_peri=-5.0, ecc=0.3, inc=60.0, omega=40.0, Omega=120.0, a_mas=4.0, flux=0.5)


def _write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


@pytest.fixture
def folder(tmp_path):
    run = tmp_path / "hd_9312" / "run"
    flag = dict(code="aliased-peaks", reason="two peaks")
    _write(tmp_path / "hd_9312" / "gallery.json", dict(instrument="GRAVITY", virgil_commit="abc", validation_commit="def"))
    table = [dict(rank=0, n=2, period=100.0, p_lo=80.0, p_hi=120.0, log_z=-10.0, delta_log_z=0.0, p=0.6, chi2_red=1.0, labels_free=False, flags=[]),
             dict(rank=1, n=3, period=60.0, p_lo=50.0, p_hi=79.0, log_z=-12.0, delta_log_z=-2.0, p=0.4, chi2_red=1.1, labels_free=False, flags=[])]
    _write(run / "summary.json", dict(status="warn", worst_check="warn", chi2=dict(best_orbit_reduced=1.0), checks=[dict(name="r_hat", status="fail")],
           modes=dict(table=table, delta_log_z=2.0, winner=dict(n=2), runner_up=dict(n=3), no_orbit=dict(verdict="orbit"))))
    _write(run / "stages" / "search" / "report.json", dict(t_ref=59000.0, p_range=[1, 400], n_bands=9, primary_diam=None))
    _write(run / "stages" / "epochs" / "report.json", dict(nights=[
        dict(dataset="MJD 1", mjd=59000.2, dra_mas=1.0, ddec_mas=2.0, flux=0.5, flags=[]),
        dict(dataset="MJD 2", mjd=59010.2, dra_mas=1.5, ddec_mas=2.5, flux=0.5, flags=[flag])]))
    for n, z in ((2, -10.0), (3, -12.0)):
        _write(run / "stages" / "search" / "bands" / f"band_{n}.json", dict(n=n, log_z=z, best=BEST, flipped=False, labels_free=False, chi2_red=1.0))
    return tmp_path / "hd_9312"


def test_summary_of_a_run_folder(folder):
    record = json.loads((ROOT / "oidb" / "trackb" / "hd_9312.json").read_text())
    s = PB.summary_of(folder, record)
    assert s["collection_id"] == "trackb-hd_9312"
    o = s["orbit"]
    assert o["start"] == "blind" and o["map"]["t_peri_mjd"] == 58995.0
    assert [m["loss"] for m in o["modes"]] == [10.0, 12.0]  # best band first, loss = -log Z
    assert o["epoch_checks"][0]["name"] == "MJD 2" and o["positions"][1]["alias"]
    keys = {f["key"] for f in s["blind"]["flags"]}
    assert {"alias", "checks", "short-arc"} <= keys  # 2 nats between the bands; r_hat failed; #425 not in the run
    assert [r["quantity"] for r in s["blind"]["table"]][-1] == "period (d)"


def test_a_folder_without_bands_is_an_error(folder):
    for f in (folder / "run" / "stages" / "search" / "bands").glob("*.json"):
        f.unlink()
    with pytest.raises(ValueError):
        PB.summary_of(folder, json.loads((ROOT / "oidb" / "trackb" / "hd_9312.json").read_text()))
