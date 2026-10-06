"""Every part on the Trust page links to the docs section that explains its
checks (scripts/trust.py resolve_docs): the page is in the site's navigation
and the section exists."""

import importlib.util
import pathlib

import pytest
import yaml

from evidence.plugin import record

ROOT = pathlib.Path(__file__).resolve().parents[1]


def trust():
    spec = importlib.util.spec_from_file_location("trust", ROOT / "scripts" / "trust.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_every_part_links_to_a_docs_section():
    t = trust()
    graph = yaml.safe_load(open(ROOT / "trust" / "graph.yml"))
    docs = t.resolve_docs(graph)
    nav = set(t.nav_pages())
    missing = [n for n in graph["nodes"] if n not in docs]
    assert not missing, f"parts with no docs page: {missing}"
    bad = []
    for node, doc in docs.items():
        page, _, anchor = doc.partition("#")
        if page not in nav or page in t.NOT_TOPIC_PAGES:
            bad.append((node, doc, "page not a topic page in the navigation"))
        elif anchor and anchor not in {a for a, _ in t.page_sections(page)}:
            bad.append((node, doc, "no such section"))
    record("parts", len(docs))
    assert not bad, bad


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_findings_page_is_the_ledger():
    """docs/method/findings.md is generated from trust/ledger.yml and lists
    every finding (regenerate with scripts/trust.py --findings)."""
    t = trust()
    ledger = yaml.safe_load(open(ROOT / "trust" / "ledger.yml"))
    page = (ROOT / "docs" / "method" / "findings.md").read_text()
    assert page == t.render_findings(ledger), "stale: rerun scripts/trust.py --findings docs/method/findings.md"
    missing = [x["id"] for x in ledger if f'id="{x["id"]}"' not in page]
    assert not missing, missing
