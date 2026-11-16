"""trust/roots.yml describes every root the trust engine knows, and no other."""

import importlib.util
import pathlib

import pytest
import yaml

from evidence import ROOTS

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("trust", ROOT / "scripts" / "trust.py")
trust = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trust)

REGISTRY = yaml.safe_load(open(ROOT / "trust" / "roots.yml"))
CLASSES = REGISTRY["classes"]
ENTRIES = {e["id"]: e for e in REGISTRY["roots"]}
LINKS = REGISTRY["links"]
TEXT_FIELDS = ("what", "why", "cannot_catch")

# What the engine knows: the pool in ROOTS, every name it can display, every
# root that counts as external, and the one root all golden sources share.
# (ROOT_ALIAS also maps the legacy record name "render", which is not a root.)
KNOWN = set(ROOTS) | set(trust.ROOT_NAMES) | set(trust.EXTERNAL) | {trust.GOLDEN}


def _mark(fn):
    return pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")(fn)


@_mark
def test_every_engine_root_has_an_entry():
    for root in sorted(KNOWN):
        assert trust.ROOT_ALIAS.get(root, root) in ENTRIES, f"{root} has no entry in trust/roots.yml"
        assert root in ENTRIES or root in trust.ROOT_ALIAS, f"{root} has no entry in trust/roots.yml"


@_mark
def test_every_entry_is_a_known_root():
    assert len(ENTRIES) == len(REGISTRY["roots"]), "duplicate root id"
    assert set(ENTRIES) <= KNOWN, f"not engine roots: {sorted(set(ENTRIES) - KNOWN)}"


@_mark
def test_classes_are_declared_and_have_text():
    for cid, c in CLASSES.items():
        for field in ("name",) + TEXT_FIELDS:
            assert isinstance(c.get(field), str) and c[field].strip(), f"class {cid}: empty {field}"
    for rid, e in ENTRIES.items():
        assert e["class"] in CLASSES, f"{rid}: undeclared class {e['class']}"
        assert not e["class"].startswith("links"), rid
    unused = set(CLASSES) - {e["class"] for e in ENTRIES.values()}
    assert not unused, f"classes with no root: {sorted(unused)}"


@_mark
def test_entries_have_text_and_review_flag():
    for rid, e in ENTRIES.items():
        for field in ("name",) + TEXT_FIELDS:
            assert isinstance(e.get(field), str) and e[field].strip(), f"{rid}: empty {field}"
        for field in ("citation", "url"):
            assert isinstance(e.get(field, ""), str), f"{rid}: {field} must be a string"
        assert isinstance(e.get("reviewed"), bool), f"{rid}: reviewed must be true or false"
        if e.get("url"):
            assert e["url"].startswith("https://"), rid


@_mark
def test_external_roots_are_other_peoples_code_and_published_results():
    """EXTERNAL (roots that may confirm a convention taken from virgil's docs)
    is the written-by-others classes: other people's code, and published results."""
    external_classes = {"other-peoples-code", "published-results"}
    from_registry = {rid for rid, e in ENTRIES.items() if e["class"] in external_classes}
    assert from_registry == set(trust.EXTERNAL)


@_mark
def test_aliases_point_at_their_entry():
    for alias, target in trust.ROOT_ALIAS.items():
        if alias in ENTRIES:
            assert ENTRIES[alias].get("counts_as") == target, f"{alias} must say counts_as: {target}"
            assert target in ENTRIES
    for rid, e in ENTRIES.items():
        if "counts_as" in e:
            assert trust.ROOT_ALIAS.get(rid) == e["counts_as"], rid


@_mark
def test_weak_and_golden_classes_match_the_engine():
    assert {rid for rid, e in ENTRIES.items() if e["class"] == "self-consistency"} == set(trust.WEAK)
    assert {rid for rid, e in ENTRIES.items() if e["class"] == "golden-values"} == {trust.GOLDEN}
    assert trust.GOLDEN not in ROOTS


@_mark
def test_links_are_not_roots():
    assert LINKS, "no links"
    for lid, link in LINKS.items():
        assert lid not in KNOWN and lid not in ENTRIES, f"{lid} is a root, not a link"
        for field in ("name",) + TEXT_FIELDS:
            assert isinstance(link.get(field), str) and link[field].strip(), f"link {lid}: empty {field}"
        assert isinstance(link.get("reviewed"), bool), f"link {lid}: reviewed must be true or false"


@_mark
def test_virgils_own_tests_are_named_under_self_consistency():
    """The engine's VIRGIL_CI (virgil's own test suite, weak) is described by the self-consistency root."""
    assert trust.VIRGIL_CI in ENTRIES["self-consistency"]["what"]
    assert trust.VIRGIL_CI in CLASSES["self-consistency"]["what"]
