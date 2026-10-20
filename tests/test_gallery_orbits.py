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
