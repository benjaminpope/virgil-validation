"""scripts/gallery_orbits.py is a thin, blind caller of virgil's OrbitPipeline (no JAX, no data needed)."""

import ast
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.validates("evidence", roots=["standards"], kind="guard")

AWAITING_TRACKB = {"ak_for", "hd188088", "hd210763", "hd41255", "hd70937", "hd_224974", "hd_9312", "kap_vel",
                   "ll_aqr", "nn_del", "omi_leo", "v963_cen"}
PUBLISHED_KEYS = {"orbit", "epochs", "conventions", "reference_orbit", "reference_positions"}


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("gallery_orbits", ROOT / "scripts" / "gallery_orbits.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_every_trackb_target_of_the_gallery_has_a_system(mod):
    assert AWAITING_TRACKB <= set(mod.slugs())


def test_nights_split_at_ut_noon():
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import gallery_orbits as g

    nights = g.group_nights([("a", 59000.4), ("b", 59000.6), ("c", 59001.2), ("d", 59000.45)])
    assert [(n, sorted(f)) for n, f in nights] == [(58999, ["a", "d"]), (59000, ["b", "c"])]


def test_the_driver_is_blind_to_the_published_solution():
    """Solved problem: published orbit leaking into fits. The driver reads only the record's instrument, and passes
    the pipeline no period, window, start or prior: every setting is virgil's default (``--quick`` only shrinks effort)."""
    src = (ROOT / "scripts" / "gallery_orbits.py").read_text()
    tree = ast.parse(src)
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not (strings & PUBLISHED_KEYS)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in ("period_range", "a_range", "ecc_max", "primary_diam", "init")
    keys = {k for k in __import__("re").findall(r"(\w+)=", src)}
    assert not keys & {"period_range", "a_range", "ecc_max", "primary_diam", "init", "p_range"}
