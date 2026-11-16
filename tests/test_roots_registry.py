"""trust/roots.yml describes every root the trust engine knows, and no other."""

import collections
import importlib.util
import pathlib
import re

import pytest
import yaml

from evidence import ROOTS

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("trust", ROOT / "scripts" / "trust.py")
trust = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trust)



class _Strict(yaml.SafeLoader):
    """Rejects a key that appears twice in one mapping (YAML keeps the last silently)."""


def _no_duplicates(loader, node, deep=False):
    keys = [loader.construct_object(k, deep=deep) for k, _ in node.value]
    dupes = sorted(k for k, n in collections.Counter(keys).items() if n > 1)
    assert not dupes, f"duplicate keys in trust/roots.yml: {dupes}"
    return yaml.SafeLoader.construct_mapping(loader, node, deep)


_Strict.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates)
REGISTRY = yaml.load((ROOT / "trust" / "roots.yml").read_text(encoding="utf-8"), Loader=_Strict)
CLASSES = REGISTRY["classes"]
ENTRIES = {e["id"]: e for e in REGISTRY["roots"]}
LINKS = REGISTRY["links"]
TEXT_FIELDS = ("what", "why", "cannot_catch")

# What the engine knows: the pool in ROOTS, every name it can display, every
# root that counts as external, and the one root all golden sources share.
# (ROOT_ALIAS also maps the legacy record name "render", which is not a root.)
KNOWN = set(ROOTS) | set(trust.ROOT_NAMES) | set(trust.EXTERNAL) | {trust.GOLDEN}


# Module-level, so the AST marker scanners (scripts/marker_inventory.py, evidence.plugin) see it.
pytestmark = pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")


def test_every_engine_root_has_an_entry():
    for root in sorted(KNOWN):
        assert root in ENTRIES, f"{root} has no entry in trust/roots.yml"


def test_every_entry_is_a_known_root():
    assert len(ENTRIES) == len(REGISTRY["roots"]), "duplicate root id"
    assert set(ENTRIES) <= KNOWN, f"not engine roots: {sorted(set(ENTRIES) - KNOWN)}"


def test_classes_are_declared_and_have_text():
    for cid, c in CLASSES.items():
        for field in ("name",) + TEXT_FIELDS:
            assert isinstance(c.get(field), str) and c[field].strip(), f"class {cid}: empty {field}"
    for rid, e in ENTRIES.items():
        assert e["class"] in CLASSES, f"{rid}: undeclared class {e['class']}"
        assert not e["class"].startswith("links"), rid
    unused = set(CLASSES) - {e["class"] for e in ENTRIES.values()}
    assert not unused, f"classes with no root: {sorted(unused)}"


def test_schema_keys_are_exact():
    """A misspelt key (count_as, reviewd) must not pass silently."""
    assert set(REGISTRY) == {"classes", "links", "roots"}
    for cid, c in CLASSES.items():
        assert set(c) == {"name", *TEXT_FIELDS}, f"class {cid}: {sorted(c)}"
    for lid, link in LINKS.items():
        assert set(link) == {"name", *TEXT_FIELDS, "reviewed"}, f"link {lid}: {sorted(link)}"
    required = {"id", "class", *TEXT_FIELDS, "citation", "url", "reviewed"}
    for rid, e in ENTRIES.items():
        assert required <= set(e) <= required | {"counts_as"}, f"{rid}: {sorted(set(e) ^ required)}"


def test_entries_have_text_and_review_flag():
    for rid, e in ENTRIES.items():
        assert "name" not in e, f"{rid}: the label comes from ROOT_NAMES, not from this file"
        for field in TEXT_FIELDS:
            assert isinstance(e.get(field), str) and e[field].strip(), f"{rid}: empty {field}"
        for field in ("citation", "url"):
            assert isinstance(e[field], str), f"{rid}: {field} must be a string"
        assert isinstance(e.get("reviewed"), bool), f"{rid}: reviewed must be true or false"
        if e.get("url"):
            assert e["url"].startswith("https://"), rid


def test_external_roots_are_other_peoples_code_and_published_results():
    """EXTERNAL (roots that may confirm a convention taken from virgil's docs)
    is the written-by-others classes: other people's code, and published results."""
    external_classes = {"other-peoples-code", "published-results"}
    from_registry = {rid for rid, e in ENTRIES.items() if e["class"] in external_classes}
    assert from_registry == set(trust.EXTERNAL)


# `golden` is the one root with no ROOT_NAMES label: the engine prints its column as
# "golden" (roots_of maps every listed golden:<source> to GOLDEN). ROOT_NAMES is not ours
# to edit, so it is exempt here.
NO_ENGINE_LABEL = {"golden"}


def test_roots_are_labelled_by_the_engine():
    """The Trust page labels every root from ROOT_NAMES, so each root entry needs one."""
    for rid, e in ENTRIES.items():
        if "counts_as" not in e and rid not in NO_ENGINE_LABEL:
            assert rid in trust.ROOT_NAMES, f"{rid} has no ROOT_NAMES label"
    assert NO_ENGINE_LABEL.isdisjoint(trust.ROOT_NAMES), "golden now has a label: drop the exemption"


def test_aliases_point_at_their_entry():
    for alias, target in trust.ROOT_ALIAS.items():
        assert target in ENTRIES, f"{alias} counts as {target}, which has no entry"
        if alias in ENTRIES:
            assert ENTRIES[alias].get("counts_as") == target, f"{alias} must say counts_as: {target}"
    for rid, e in ENTRIES.items():
        if "counts_as" in e:
            assert trust.ROOT_ALIAS.get(rid) == e["counts_as"], rid


def test_weak_and_golden_classes_match_the_engine():
    assert {rid for rid, e in ENTRIES.items() if e["class"] == "self-consistency"} == set(trust.WEAK)
    assert {rid for rid, e in ENTRIES.items() if e["class"] == "golden-values"} == {trust.GOLDEN}
    assert trust.GOLDEN not in ROOTS


def test_links_are_not_roots():
    assert LINKS, "no links"
    for lid, link in LINKS.items():
        assert lid not in KNOWN and lid not in ENTRIES, f"{lid} is a root, not a link"
        for field in ("name",) + TEXT_FIELDS:
            assert isinstance(link.get(field), str) and link[field].strip(), f"link {lid}: empty {field}"
        assert isinstance(link.get("reviewed"), bool), f"link {lid}: reviewed must be true or false"


def test_virgils_own_tests_are_named_under_self_consistency():
    """The engine's VIRGIL_CI (virgil's own test suite, weak) is described by the self-consistency root."""
    assert trust.VIRGIL_CI in ENTRIES["self-consistency"]["what"]
    assert trust.VIRGIL_CI in CLASSES["self-consistency"]["what"]


def test_golden_entry_names_every_golden_source():
    for source in trust.load_golden():
        assert source in ENTRIES["golden"]["what"], f"trust/golden.yml source {source} is not in the golden entry"


def test_documented_upstream_bugs_are_cited_by_their_root():
    """Each ledger entry ruled `external:<root>` (P1, P2, ...) is cited by that root's entry, and
    the other-people's-code class states the total."""
    ledger = yaml.safe_load((ROOT / "trust" / "ledger.yml").read_text(encoding="utf-8"))
    found = [(e["id"], e["ruling"].split(":", 1)[1]) for e in ledger if str(e.get("ruling", "")).startswith("external:")]
    assert found, "no external rulings in the ledger"
    for pid, root in found:
        assert root in ENTRIES, f"{pid}: ruled external:{root}, which is not a root entry"
        assert re.search(rf"\b{pid}\b", ENTRIES[root]["cannot_catch"]), f"{pid} is not cited under {root}"
    n = len(found)
    assert f"{n} are documented here (P1–P{n})" in CLASSES["other-peoples-code"]["cannot_catch"]
