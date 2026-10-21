"""scripts/gallery_orbits.py is a thin, blind caller of virgil's OrbitPipeline (no JAX, no data needed)."""

import ast
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

AWAITING_TRACKB = {"ak_for", "hd188088", "hd210763", "hd41255", "hd70937", "hd_224974", "hd_9312", "kap_vel",
                   "ll_aqr", "nn_del", "omi_leo", "v963_cen"}
EFFORT = {"n_candidates", "n_refine", "max_steps", "n_is", "num_warmup", "num_samples", "num_chains", "quicklook_execute"}
PUBLISHED_KEYS = {"orbit", "epochs", "conventions", "reference_orbit", "reference_positions"}


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("gallery_orbits", ROOT / "scripts" / "gallery_orbits.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")
def test_every_trackb_target_of_the_gallery_has_a_system(mod):
    assert AWAITING_TRACKB <= set(mod.slugs())


@pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")
def test_nights_split_at_ut_noon():
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import gallery_orbits as g

    nights = g.group_nights([("a", 59000.4), ("b", 59000.6), ("c", 59001.2), ("d", 59000.45)])
    assert [(n, sorted(f)) for n, f in nights] == [(58999, ["a", "d"]), (59000, ["b", "c"])]


@pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")
def test_the_driver_is_blind_to_the_published_solution(mod):
    """Solved problem: published orbit leaking into fits. The only settings the pipeline gets are effort-only keys
    (``QUICK``) and ``quicklook_execute``; ``OrbitPipeline`` is called with ``output`` and ``**settings`` alone, and
    the driver reads only the record's instrument."""
    assert set(mod.QUICK) <= EFFORT
    tree = ast.parse((ROOT / "scripts" / "gallery_orbits.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "OrbitPipeline"]
    assert len(calls) == 1
    assert [k.arg for k in calls[0].keywords] == ["output", None]  # None: **settings
    sets = [n for n in ast.walk(tree) if isinstance(n, ast.Subscript) and getattr(n.value, "id", "") == "settings"]
    assert all(isinstance(s.slice, ast.Constant) and s.slice.value in EFFORT for s in sets)
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not (strings & PUBLISHED_KEYS)


@pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")
@pytest.mark.parametrize("slug", ["gl229b_blind", "hr6819_blind"])
def test_gl229b_and_hr6819_are_job_targets_through_their_own_record(mod, slug):
    rec = mod.record_of(slug)
    assert slug in mod.slugs() and rec["instrument"] == "GRAVITY" and rec["schema"] == "gallery-target/1"


@pytest.mark.validates("virgil.oidata.OIData", roots=["self-consistency"], kind="guard")
def test_mircx_nights_carry_gains_and_offsets(mod, tmp_path):
    """iota Peg's viscal files, one night per UT date, are read with per-frame gains and closure offsets."""
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "tests"))
    import numpy as np
    from test_oidb_fits import UTS, _binary, _observe

    d = tmp_path / mod.MIRCX["iota_peg"]["collection"]
    d.mkdir()
    for i, night in enumerate(["2018Oct20", "2018Oct21", "2018Oct22"]):
        _observe(d / f"{night}_mircx0000{i}_oifits_viscal.fits", _binary(3.0 + i, -4.0, 0.3), stations=UTS, seed=i + 1)
    nights, prov = mod.load_mircx_nights("iota_peg", tmp_path)
    assert sorted(nights) == ["2018Oct20", "2018Oct21", "2018Oct22"]
    assert all(n.gains is not None for n in nights.values()) and np.asarray(nights["2018Oct20"].vis).size
    assert prov["data_path"] == "viscal+gains" and prov["gain_widths"]["telescope"] == mod.MIRCX_GAINS["telescope"]
    assert all(n.phase_offsets is not None for n in nights.values())
    assert all(len(v["files"]) == 1 for v in prov["nights"].values())
