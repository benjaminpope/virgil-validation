"""The Track B orbit-target records (``oidb/trackb/<slug>.json``, built by scripts/build_trackb_targets.py): the published
elements reproduce the published positions in crosscheck.orbits' convention, the held-out split is consistent, and every
file is registered with provenance. The OIFITS themselves are local (~/data/eso_binaries/trackb); these tests read only
the committed records."""

import importlib.util
import json
import pathlib
import re

import numpy as np
import pytest

from crosscheck import orbits as O
from evidence.plugin import record

ROOT = pathlib.Path(__file__).resolve().parents[1]
RECORDS = sorted((ROOT / "oidb" / "trackb").glob("*.json"))
# the ESO Track B systems: records built by the builder, with registered Phase 3 files; gallery-target/1 records (Gl 229 B,
# HR 6819, fitted on the same path but with other data) are tiles of their own and carry no file registry
SLUGS = [p.stem for p in RECORDS if json.loads(p.read_text())["schema"] == "trackb-target/1"]
KINDS = {"sb2_subresolution", "alias_track", "short_arc"}


def _rec(slug):
    return json.loads((ROOT / "oidb" / "trackb" / f"{slug}.json").read_text())


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location("build_trackb_targets", ROOT / "scripts" / "build_trackb_targets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.validates("pipeline:binary-orbits", roots=["standards"], kind="guard")
def test_twelve_systems_in_three_kinds():
    assert len(SLUGS) == 12
    kinds = [_rec(s)["kind"] for s in SLUGS]
    assert set(kinds) == KINDS
    assert kinds.count("sb2_subresolution") == 5 and kinds.count("short_arc") == 2
    record("systems", len(SLUGS))


@pytest.mark.validates("pipeline:binary-orbits", roots=["standards"], kind="guard")
@pytest.mark.parametrize("slug", SLUGS)
def test_record_cites_its_paper_and_elements(slug):
    r = _rec(slug)
    p = r["paper"]
    assert re.fullmatch(r"\d{4}\.\d{4,5}", p["arxiv"]) and p["doi"].startswith("10.") and p["bibcode"]
    o = r["orbit"]
    for k in ("P_day", "T_peri_jd", "T_peri_mjd", "e", "omega_deg", "Omega_deg", "i_deg", "a_mas"):
        assert o[k] and o[k][0] is not None, k
    assert o["T_peri_mjd"][0] == pytest.approx(o["T_peri_jd"][0] - 2400000.5)
    assert o["source"]
    assert r["conventions"]["text"] and "omega_shift_deg" in r["conventions"]


LOOSE_MAS = {"ak_for": 0.45}  # e = 0.0002: omega and T_peri are undetermined, so the orbit follows the positions less closely
FLOOR_MAS = 0.15  # the larger of this and 6 sigma_major is each epoch's tolerance (the short arcs print no sigma)


def _ti_position(t, o):
    """(dRA, dDec) from the Thiele-Innes constants: dDec = A X + F Y, dRA = B X + G Y (no angle convention involved)."""
    ti = o["thiele_innes_mas"]
    M = 2 * np.pi * (np.asarray(t) - o["T_peri_mjd"][0]) / o["P_day"][0]
    e = o["e"][0]
    E = M + e * np.sin(M)
    for _ in range(60):  # Newton on Kepler's equation
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    X, Y = np.cos(E) - e, np.sqrt(1 - e * e) * np.sin(E)
    return ti["B"] * X + ti["G"] * Y, ti["A"] * X + ti["F"] * Y


@pytest.mark.validates("pipeline:binary-orbits", roots=["literature"], kind="guard")
@pytest.mark.parametrize("slug", SLUGS)
def test_thiele_innes_reproduce_the_published_positions(slug):
    """The record's Thiele-Innes constants (the sky orbit, whatever the paper's angle convention) reproduce the paper's own
    positions at every epoch, to 6 sigma or the floor; none was fitted. A flagged ``suspect`` epoch is excluded, but only
    if it really is the outlier."""
    r = _rec(slug)
    ep = r["epochs"]
    x, y = _ti_position([e["mjd"] for e in ep], r["orbit"])
    m = np.hypot(x - [e["dra_mas"] for e in ep], y - [e["ddec_mas"] for e in ep])
    sigma = np.array([max(e["sigma_major_mas"] or 0, e["sigma_minor_mas"] or 0) for e in ep])
    tol = np.maximum(6 * sigma, LOOSE_MAS.get(slug, FLOOR_MAS))
    suspect = np.array(["suspect" in e for e in ep])
    assert suspect.sum() <= 1
    assert np.all(m[~suspect] <= tol[~suspect]), (slug, np.round(m, 3), np.round(tol, 3))
    if suspect.any():
        assert m[suspect][0] > 0.5 and np.sum(m > 0.5) == 1
    record("max_miss_mas", float(m[~suspect].max()))


@pytest.mark.validates("pipeline:binary-orbits", roots=["literature"], kind="guard")
def test_kappa_vel_standard_elements_are_the_papers_orbit_exactly():
    """kappa Vel's Omega and i are printed in the paper's own Thiele-Innes form (its Eqs. 6-7); the standard elements stored
    beside them give the same constants by algebra (no fit to Table 3)."""
    import math
    r = _rec("kap_vel")
    ti, o, st = r["orbit"]["thiele_innes_mas"], r["orbit"], r["conventions"]["standard"]
    w, O, i = (math.radians(x) for x in (o["omega_deg"][0], st["Omega_deg"], st["i_deg"]))
    a = o["a_mas"][0]
    A = a * (math.cos(O) * math.cos(w) - math.cos(i) * math.sin(O) * math.sin(w))
    B = a * (math.sin(O) * math.cos(w) + math.cos(i) * math.cos(O) * math.sin(w))
    F = a * (-math.cos(O) * math.sin(w) - math.cos(i) * math.sin(O) * math.cos(w))
    G = a * (-math.sin(O) * math.sin(w) + math.cos(i) * math.cos(O) * math.cos(w))
    assert (ti["A"], ti["B"], ti["F"], ti["G"]) == pytest.approx((A, B, F, G), abs=1e-9)


@pytest.mark.validates("pipeline:binary-orbits", roots=["literature"], kind="guard")
@pytest.mark.parametrize("slug", SLUGS)
def test_published_orbit_reproduces_the_published_positions(slug):
    """The convention check: the tabulated elements, with the record's convention, reproduce the paper's own positions at
    *every* epoch within 6 sigma (or the floor); never fitted. A point reflection (omega + 180 degrees alone) misses by
    milliarcseconds. An epoch flagged ``suspect`` (the paper's own table disagrees with itself) is excluded, but only if it
    really is the outlier, and at most one per system."""
    r = _rec(slug)
    o, c = r["orbit"], r["conventions"]
    std = c.get("standard", {})
    ep = r["epochs"]
    t = np.array([e["mjd"] for e in ep])

    def miss(shift):
        x, y = O.position(t, o["P_day"][0], o["T_peri_mjd"][0], o["e"][0], std.get("i_deg", o["i_deg"][0]),
                          o["omega_deg"][0] + shift, std.get("Omega_deg", o["Omega_deg"][0]), o["a_mas"][0])
        return np.hypot(x - [e["dra_mas"] for e in ep], y - [e["ddec_mas"] for e in ep])

    m = miss(c["omega_shift_deg"])
    sigma = np.array([max(e["sigma_major_mas"] or 0, e["sigma_minor_mas"] or 0) for e in ep])
    tol = np.maximum(6 * sigma, LOOSE_MAS.get(slug, FLOOR_MAS))
    suspect = np.array(["suspect" in e for e in ep])
    assert suspect.sum() <= 1
    assert np.all(m[~suspect] <= tol[~suspect]), (slug, np.round(m, 3), np.round(tol, 3))
    if suspect.any():
        assert m[suspect][0] > 0.5 and np.sum(m > 0.5) == 1  # flagged because it really is the one that disagrees
    assert np.median(miss(c["omega_shift_deg"] + 180)) > 1.0  # the reflected orbit is clearly wrong: the check can fail
    record("max_miss_mas", float(m[~suspect].max()))


@pytest.mark.validates("pipeline:binary-orbits", roots=["standards"], kind="guard")
@pytest.mark.parametrize("slug", SLUGS)
def test_split_is_consistent(slug):
    r = _rec(slug)
    files, epochs, sp = r["files"], r["epochs"], r["split"]
    ids = [f["dp_id"] for f in files]
    assert len(set(ids)) == len(ids) and all(f["sha256"] and f["bytes"] for f in files)  # all registered and checksummed
    assert len({f["sha256"] for f in files}) == len(files)  # no file twice
    roles = {f["role"] for f in files}
    assert roles <= {"paper", "paper_night", "heldout"}
    assert sp["paper_files"] == sum(f["role"] == "paper" for f in files)
    assert sp["heldout_files"] == sum(f["role"] == "heldout" for f in files)
    pub_nights = {int(np.floor(e["mjd"] - 0.5)) for e in epochs}
    for f in files:
        if f["role"] == "heldout":  # a held-out night never has a published epoch
            assert f["night_mjd"] not in pub_nights, f["dp_id"]
        else:
            assert f["night_mjd"] in pub_nights, f["dp_id"]
        if f["role"] == "paper":
            assert abs(f["t_min_mjd"] - epochs[f["epoch_index"]]["mjd"]) < 0.03
        assert f["after_paper_span"] == (f["role"] == "heldout" and f["t_min_mjd"] > sp["last_paper_mjd"])
    for i, e in enumerate(epochs):  # each epoch lists exactly the files that claim it
        assert e["dp_ids"] == [f["dp_id"] for f in files if f["role"] == "paper" and f["epoch_index"] == i]
    assert sp["heldout_nights"] == len({f["night_mjd"] for f in files if f["role"] == "heldout"})


@pytest.mark.validates("pipeline:binary-orbits", roots=["standards"], kind="guard")
def test_classify_roles_by_time_and_night(builder):
    pub = [58000.20, 58010.30]
    rows = [dict(dp_id=f"ADP.{i}", instrument_name="GRAVITY", t_min=str(t), t_max=str(t + 0.01), proposal_id="x",
                 obs_release_date="2020-01-01", access_estsize="9728", dataproduct_subtype="SINGLE_SCI_VIS", obs_id="o")
            for i, t in enumerate([58000.21, 58000.40, 58005.0, 58020.0])]
    got = builder.classify(rows, pub)
    assert [g["role"] for g in got] == ["paper", "paper_night", "heldout", "heldout"]
    assert [g["after_paper_span"] for g in got] == [False, False, False, True]
    assert got[0]["epoch_index"] == 0 and got[1]["epoch_index"] is None
    assert got[0]["night_mjd"] == builder.night(58000.21) == 57999  # floor(MJD - 0.5)
