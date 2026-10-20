"""scripts/calibrated_load.py and its callers read ESO data only through virgil-vlti's loader (small synthetic files)."""

import json
import pathlib
import sys

import pytest
from astropy.io import fits

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("virgil_vlti")
import calibrated_load as C  # noqa: E402
import gallery_orbits as G  # noqa: E402
import reanalyse_orbits as R  # noqa: E402

pytestmark = pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")


def _file(path, cp_only=False):
    """A file with only a primary header: all that the night scan reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU()
    if cp_only:
        hdu.header["CPONLY"] = True
    hdu.writeto(path)
    return path


@pytest.fixture
def sysdir(tmp_path):
    d = tmp_path / "sys" / "calibrated"
    _file(d / "HIGH" / "2020-12-08" / "a.fits")
    _file(d / "HIGH" / "2020-12-08" / "b.fits", cp_only=True)  # same night: left out, noted
    _file(d / "HIGH" / "2020-12-13" / "c.fits", cp_only=True)  # no calibrator: closure phases only
    _file(d / "MEDIUM" / "2020-12-08" / "e.fits")
    return tmp_path / "sys"


def test_nights_by_mode_with_calibrated_or_cp_only_path(sysdir):
    nights = {(n["mode"], n["mjd"]): n for n in C.gravity_nights(sysdir)}
    assert set(nights) == {("HIGH", 59191), ("HIGH", 59196), ("MEDIUM", 59191)}  # folder 2020-12-08 is MJD 59191
    assert nights["HIGH", 59191]["path"] == C.CALIBRATED
    assert [f.name for f in nights["HIGH", 59191]["files"]] == ["a.fits"]
    assert "b.fits" in nights["HIGH", 59191]["notes"][0]
    assert nights["HIGH", 59196]["path"] == C.CP_ONLY
    assert C.system_data_path(["calibrated", "calibrated"]) == "calibrated"
    assert C.system_data_path(["calibrated", "cp-only"]) == "mixed"


def test_uncalibrated_gravity_is_refused_not_read(sysdir, monkeypatch):
    with pytest.raises(FileNotFoundError, match="calibrate the system first"):
        C.gravity_nights(sysdir.parent)  # no calibrated/ folder: no fallback to the raw products
    import trackb_fit as T

    read = R._refuse_gravity_sci_vis(T)
    with pytest.raises(ValueError, match="not used"):
        read([pathlib.Path("x.fits")], "GRAVITY")
    with pytest.raises(ValueError, match="oifits_extra"):
        G.load_nights("hd_9312", sysdir.parent.parent, include_extra=True)


def test_loader_asks_for_vis_and_phi_only_where_calibrated(sysdir, monkeypatch):
    import virgil_vlti

    calls = []
    monkeypatch.setattr(virgil_vlti, "to_oidata", lambda files, ins, **kw: calls.append((ins, kw)) or "data")
    nights = {n["mjd"]: n for n in C.gravity_nights(sysdir) if n["mode"] == "HIGH"}
    C.load_gravity_night(nights[59191])
    C.load_gravity_night(nights[59196])
    assert [c[0] for c in calls] == ["GRAVITY_SC"] * 2
    assert calls[0][1]["observables"] == ("vis", "phi") and calls[1][1]["observables"] == "phi"
    assert calls[0][1]["ranges"] == [(2.05e-6, 2.155e-6), (2.175e-6, 2.38e-6)]  # Brγ out


def test_gallery_provenance_records_the_data_path(tmp_path, monkeypatch):
    root = tmp_path / "root"
    for date, cp in (("2020-12-08", False), ("2020-12-13", True), ("2021-01-02", True)):
        _file(root / "trackb" / "hd_9312" / "calibrated" / "HIGH" / date / "f.fits", cp_only=cp)
    monkeypatch.setattr(C, "load_gravity_night", lambda n, ranges=None: n["path"])
    nights, prov = G.load_nights("hd_9312", root)
    assert list(nights) == ["MJD 59191", "MJD 59196", "MJD 59216"]  # one mode: no suffix
    assert prov["data_path"] == "mixed"
    assert {k: v["data_path"] for k, v in prov["nights"].items()} == {
        "MJD 59191": "calibrated", "MJD 59196": "cp-only", "MJD 59216": "cp-only"}
    assert len(prov["nights"]["MJD 59191"]["files"][0]["sha256"]) == 64
    json.dumps(prov)


def test_pionier_template_carries_the_pair_test_widths():
    t = C.pionier_template()
    assert t.provenance["widths"] == dict(telescope=0.011, baseline=0.009, offset=0.21)


def test_pull_back_keeps_the_legacy_flag_unless_a_path_was_recorded():
    import gallery_pull_back as PB

    src = pathlib.Path(PB.__file__).read_text()
    assert '"data_path"' in src and "absent for a run that predates it" in src
