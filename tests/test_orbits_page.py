"""The one binary-orbit page (scripts/build_orbits_page.py): the gallery and the period aliases as sections."""

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import build_orbits_page as P  # noqa: E402

pytestmark = pytest.mark.validates("pipeline:binary-orbits", roots=["standards"], kind="guard")


def test_a_generated_page_becomes_a_section():
    text = "# Title\n\nSee [Period aliases](aliases.md).\n\n## Method\n\n```\n# not a heading\n```\n\n### Deeper\n"
    out = P.section(text)
    assert "# Title" not in out and "### Method" in out and "#### Deeper" in out
    assert "# not a heading\n" in out and "(#period-aliases)" in out and "aliases.md" not in out


def test_page_has_one_title_and_both_sections(tmp_path):
    gallery = tmp_path / "gallery.md"
    gallery.write_text("# Binary orbit gallery\n\n## How to read a plot\n\ntext, [aliases](aliases.md)\n")
    page = P.render(gallery)
    assert [line for line in page.splitlines() if line.startswith("# ")] == ["# Binary orbits: GRAVITY and PIONIER"]
    for heading in ("## Orbit gallery", "### How to read a plot", "## Period aliases", "### Gl 229 Ba-Bb",
                    "### Systems not included"):
        assert heading + "\n" in page
    assert "$" + "GALLERY" not in page and "(aliases.md)" not in page
    assert "Gallery not built" in P.render(tmp_path / "missing.md")
