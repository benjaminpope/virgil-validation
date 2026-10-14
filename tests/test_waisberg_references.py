"""The Waisberg+2025 references for zeta Boo and eta Oph (design/gravity_aberration_test.md) and the ESO Phase 3
provenance record (oidb/eso_phase3.json). No network and no data files: the JSON records only."""

import importlib.util
import json
import pathlib
import re

import numpy as np
import pytest

from crosscheck import orbits
from evidence.plugin import record

ROOT = pathlib.Path(__file__).resolve().parents[1]
REFS = {"zet_boo": ROOT / "oidb" / "references" / "waisberg2025-zet-boo.json",
        "eta_oph": ROOT / "oidb" / "references" / "waisberg2025-eta-oph.json"}
N_EPOCHS = {"zet_boo": 4, "eta_oph": 6}
# the published orbit and the paper's own positions: the largest miss allowed (mas), a little above what is seen
TOL_MAS = {"zet_boo": 0.12, "eta_oph": 0.06}
PHASE3 = json.loads((ROOT / "oidb" / "eso_phase3.json").read_text())


def _positions(ref, omega_shift=0.0):
    o = ref["orbit"]
    return np.array([[float(np.ravel(x)[0]) for x in orbits.position(
        e["mjd"], o["P_day"][0], o["T_peri_mjd"][0], o["e"][0], o["i_deg"][0], (o["omega_deg"][0] + omega_shift) % 360,
        o["Omega_deg"][0], o["a_mas"][0])] for e in ref["epochs"]])


def _published(ref):
    return np.array([[e["other"]["dE_mas"][0], e["other"]["dN_mas"][0]] for e in ref["epochs"]])


@pytest.mark.parametrize("system", ["zet_boo", "eta_oph"])
@pytest.mark.validates("pipeline:gravity-aberration-wide", roots=["literature"], kind="guard")
def test_published_orbit_reproduces_the_published_positions(system):
    """Table 7's ORB6+GRAVITY orbit, read as the companion's omega with Julian-year Tp (the reference's stated
    conventions), passes through the paper's own positions: this fixes the conventions of the reference files."""
    ref = json.loads(REFS[system].read_text())
    assert len(ref["epochs"]) == N_EPOCHS[system]
    miss = np.abs(_positions(ref) - _published(ref)).max()
    record(f"{system}_orbit_vs_positions_max_miss_mas", float(miss))
    assert miss < TOL_MAS[system]


@pytest.mark.parametrize("system", ["zet_boo", "eta_oph"])
@pytest.mark.validates("pipeline:gravity-aberration-wide", roots=["literature"], kind="control")
def test_primary_omega_reading_misses_by_tens_of_mas(system):
    """The other reading of omega (the primary's, +180 deg) must fail: it is the mapping the references avoid."""
    ref = json.loads(REFS[system].read_text())
    assert np.abs(_positions(ref, 180.0) - _published(ref)).max() > 30.0


@pytest.mark.parametrize("system", ["zet_boo", "eta_oph"])
@pytest.mark.validates("pipeline:gravity-aberration-wide", roots=["literature"], kind="guard")
def test_reference_shape_and_sources(system):
    ref = json.loads(REFS[system].read_text())
    assert ref["collection_id"] == REFS[system].stem
    assert ref["paper"]["bibcode"] == "2025OJAp....8E..63W" and ref["paper"]["arxiv"] == "2504.17858"
    sigma = {"zet_boo": 0.040, "eta_oph": 0.020}[system]
    for e in ref["epochs"]:
        dra, ddec = e["other"]["dE_mas"], e["other"]["dN_mas"]
        assert dra[1] == ddec[1] == sigma
        assert e["rho_mas"][0] == pytest.approx(np.hypot(dra[0], ddec[0]), abs=1e-3)
        assert e["pa_deg"][0] == pytest.approx(np.degrees(np.arctan2(dra[0], ddec[0])) % 360, abs=1e-3)
        assert re.fullmatch(r"ADP\.\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}", e["phase3_dp_ids"][0])
        assert "Table" in e["source"]
    seps = [e["rho_mas"][0] for e in ref["epochs"]]
    expected = {"zet_boo": (57, 88), "eta_oph": (10, 265)}[system]
    assert (min(seps), max(seps)) == pytest.approx(expected, abs=1.0)
    assert ref["orbit"]["P_day"][0] == pytest.approx(ref["orbit"]["P_yr"][0] * 365.25, abs=1e-3)


@pytest.mark.validates("pipeline:gravity-aberration-wide", roots=["literature"], kind="guard")
def test_eso_phase3_record_matches_the_references():
    """Every Phase 3 product a reference names is registered, once, with a SHA-256, its programme and a release date."""
    for system, path in REFS.items():
        ref = json.loads(path.read_text())
        prods = PHASE3["systems"][system]["products"]
        assert [p["dp_id"] for p in prods] == [e["phase3_dp_ids"][0] for e in ref["epochs"]]
        for p, e in zip(prods, ref["epochs"]):
            assert re.fullmatch(r"[0-9a-f]{64}", p["sha256"]) and p["bytes"] % 2880 == 0
            assert abs(p["mjd_obs"] - e["mjd"]) < 0.01
            assert p["insname"] == "GRAVITY" and p["archive"]["dataproduct_type"] == "visibility"
            assert p["archive"]["proposal_id"] == p["prog_id"] and p["archive"]["release_date"] < "2026-01-01"
            assert p["prog_id"] in PHASE3["systems"][system]["programmes"]


@pytest.mark.validates("pipeline:gravity-aberration-wide", roots=["literature"], kind="guard")
def test_smoke_residual_is_the_fit_minus_the_published_position_at_the_nearer_sign():
    spec = importlib.util.spec_from_file_location("aberration_smoke", ROOT / "scripts" / "aberration_smoke.py")
    smoke = importlib.util.module_from_spec(spec)
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    spec.loader.exec_module(smoke)
    # published (3, 4) mas (separation 5, PA 36.87 deg); the fit is the swapped pair (-3.03, -4.01), sign -1
    row = dict(dra=-3.03, ddec=-4.01, published=[3.0, 4.0], sign=-1.0)
    r = smoke.residual(row, 0.04)
    assert r["dra_uas"] == pytest.approx(-30.0) and r["ddec_uas"] == pytest.approx(-10.0)
    # radial axis along the nearer-sign separation, (-0.6, -0.8); tangential (-0.8, 0.6), towards increasing PA
    assert r["radial_uas"] == pytest.approx(26.0) and r["tangential_uas"] == pytest.approx(18.0)
    assert r["total_uas"] == pytest.approx(np.hypot(30, 10)) and r["total_sigma"] == pytest.approx(np.hypot(30, 10) / 40)
