"""scripts/gallery_pull_back.py turns an OrbitPipeline run folder into a gallery summary (no JAX)."""

import json
import pathlib
import sys

import numpy as np
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
    _write(run / "summary.json", dict(status="complete", worst_check="warn", chi2=dict(best_orbit_reduced=1.0), checks=[dict(name="r_hat", status="fail"), dict(name="ess", status="warn")],
           modes=dict(table=table, delta_log_z=2.0, winner=dict(n=2), runner_up=dict(n=3), no_orbit=dict(verdict="orbit"))))
    _write(run / "stages" / "search" / "report.json", dict(t_ref=59000.0, p_range=[1, 400], n_bands=9, primary_diam=None))
    _write(run / "stages" / "epochs" / "report.json", dict(nights=[
        dict(dataset="MJD 1", mjd=59000.2, dra_mas=1.0, ddec_mas=2.0, flux=0.5, flags=[]),
        dict(dataset="MJD 2", mjd=59010.2, dra_mas=1.5, ddec_mas=2.5, flux=0.5, flags=[flag])]))
    for n, z in ((2, -10.0), (3, -12.0)):
        _write(run / "stages" / "search" / "bands" / f"band_{n}.json", dict(n=n, p_lo=10.0 * (n + 1), p_hi=10.0 * (n + 2), log_z=z, best=BEST, flipped=False, labels_free=False, chi2_red=1.0))
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
    assert {"alias", "checks-fail", "short-arc"} <= keys  # 2 nats between the bands; r_hat failed; #425 not in the run
    assert any(f["key"] == "checks-warn" for f in s["blind"]["flags"])  # a warning is listed too
    assert [r["quantity"] for r in s["blind"]["table"]][-1] == "period (d)"


def test_a_folder_without_bands_is_an_error(folder):
    for f in (folder / "run" / "stages" / "search" / "bands").glob("*.json"):
        f.unlink()
    with pytest.raises(ValueError):
        PB.summary_of(folder, json.loads((ROOT / "oidb" / "trackb" / "hd_9312.json").read_text()))


def test_a_band_file_that_is_not_the_pipelines_winner_is_an_error(folder):
    s = json.loads((folder / "run" / "summary.json").read_text())
    s["modes"]["winner"]["n"] = 3
    _write(folder / "run" / "summary.json", s)
    with pytest.raises(ValueError, match="winner"):
        PB.summary_of(folder, json.loads((ROOT / "oidb" / "trackb" / "hd_9312.json").read_text()))


WINDOW = (3.0, 3000.0)


# --- the sky comparison. Positions are generated from hd_9312's tabulated orbit, so the match is exact by construction
# (the paper's own residuals are ~0.1 mas, 3% of the separation, above crosscheck's 2% bar).

@pytest.fixture(scope="module")
def hd9312():
    from crosscheck import skyplane as SP

    r = json.loads((ROOT / "oidb" / "trackb" / "hd_9312.json").read_text())
    o, t_ref = r["orbit"], 59500.0
    best = dict(period=o["P_day"][0], dt_peri=o["T_peri_mjd"][0] - t_ref, ecc=o["e"][0], inc=o["i_deg"][0],
                omega=o["omega_deg"][0], Omega=o["Omega_deg"][0], a_mas=o["a_mas"][0])
    mjd = np.array([e["mjd"] for e in r["epochs"] if e.get("mjd")])
    xy = SP.fit_pos(best, mjd, t_ref)
    r = dict(r, epochs=[dict(mjd=float(m), dra_mas=float(x), ddec_mas=float(y)) for m, (x, y) in zip(mjd, xy)])
    table = [dict(rank=0, n=5, p_lo=best["period"] - 1, p_hi=best["period"] + 1, delta_log_z=0.0),
             dict(rank=1, n=6, p_lo=best["period"] * 2, p_hi=best["period"] * 3, delta_log_z=-3.0)]
    return r, best, t_ref, table


@pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")
def test_an_orbit_on_the_measured_positions_is_the_same_sky_orbit(hd9312):
    r, best, t_ref, table = hd9312
    rows, verdict = PB.sky_rows(best, t_ref, r, table, WINDOW, False, False, 1.0)
    assert verdict == "SAME-SKY-ORBIT"
    assert rows[0]["ours"] < 1e-6
    assert "band N=5 (rank 1 of 2" in rows[1]["note"]


@pytest.mark.validates("evidence", roots=["self-consistency"], kind="control")
def test_a_point_reflection_without_a_label_swap_is_not_agreement(hd9312):
    r, best, t_ref, table = hd9312
    reflected = dict(best, Omega=best["Omega"] + 180.0)  # (dRA, dDec) -> (-dRA, -dDec)
    _, verdict = PB.sky_rows(reflected, t_ref, r, table, WINDOW, False, False, 1.0)
    assert verdict == "UNEXPLAINED-FLIP"
    _, accepted = PB.sky_rows(reflected, t_ref, r, table, WINDOW, True, False, 1.0)
    assert accepted == "SAME-SKY-ORBIT"


@pytest.mark.validates("evidence", roots=["self-consistency"], kind="control")
def test_another_alias_band_is_a_different_sky_orbit(hd9312):
    r, best, t_ref, table = hd9312
    far = dict(best, period=best["period"] * 2.5)
    rows, verdict = PB.sky_rows(far, t_ref, r, table, WINDOW, False, False, None)
    assert verdict == "DIFFERENT"  # without a raw chi2/N the paper is not blamed
    assert PB.sky_rows(far, t_ref, r, table, WINDOW, False, False, 1.0)[1] == "REFERENCE-DISAGREES"


def test_the_newest_run_folder_wins_by_time_not_by_name(tmp_path):
    import os

    for tag, when in (("run-9", 2000), ("run-10", 1000)):  # run-10 sorts first but is older
        d = tmp_path / tag / "nn_del"
        (d / "run").mkdir(parents=True)
        (d / "gallery.json").write_text("{}")
        (d / "run" / "summary.json").write_text("{}")
        os.utime(d / "run" / "summary.json", (when, when))
    assert PB.run_folders(tmp_path)["nn_del"].parent.name == "run-9"


def test_an_unfinished_run_is_flagged_not_hidden(folder):
    s = json.loads((folder / "run" / "summary.json").read_text())
    s["status"] = "running"
    _write(folder / "run" / "summary.json", s)
    out = PB.summary_of(folder, json.loads((ROOT / "oidb" / "trackb" / "hd_9312.json").read_text()))
    assert any(f["key"] == "incomplete" for f in out["blind"]["flags"])


@pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")
def test_the_period_row_says_where_the_published_period_ranked_or_that_it_is_outside_the_window(hd9312):
    r, best, t_ref, table = hd9312
    low = [dict(rank=0, n=9, p_lo=100.0, p_hi=200.0, delta_log_z=0.0)] + [
        dict(rank=i, n=i, p_lo=0.0, p_hi=0.0, delta_log_z=-1.0 * i) for i in range(1, 11)] + [
        dict(rank=11, n=5, p_lo=best["period"] - 1, p_hi=best["period"] + 1, delta_log_z=-30.0)]
    note = PB.sky_rows(best, t_ref, r, low, WINDOW, False, False, 1.0)[0][-1]["note"]
    assert "rank 12 of 12" in note  # below the tenth: the ranking's finding, not the window's
    note = PB.sky_rows(best, t_ref, r, low[:-1], WINDOW, False, False, 1.0)[0][-1]["note"]
    assert "no finite evidence" in note
    note = PB.sky_rows(best, t_ref, r, low[:-1], (100.0, 3000.0), False, False, 1.0)[0][-1]["note"]
    assert "outside the searched window" in note
