"""The binary-orbit page (scripts/build_orbits_page.py): the gallery first, its notes after, an alias page per system."""

import json
import pathlib
import re
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import build_orbits_page as P  # noqa: E402
from star_names import greek  # noqa: E402

pytestmark = pytest.mark.validates("pipeline:binary-orbits", roots=["standards"], kind="guard")

GALLERY = """# Binary orbit gallery

<div class="og-intro" markdown>

Intro, see [aliases](aliases.md).

</div>

<div class="og"></div>

<script type="application/json" id="og-data">{"targets":[{"slug":"omi-leo","target":"o Leo"},{"slug":"hr6819","target":"HR 6819"}]}</script>

<div class="og-about" markdown>

## How to read a plot

text

</div>

<div class="og-static" markdown>

## The numbers without JavaScript

**kappa Vel**

</div>
"""


def test_a_generated_page_becomes_a_section():
    text = "# Title\n\nSee [Period aliases](aliases.md).\n\n## Method\n\n```\n# not a heading\n```\n\n### Deeper\n"
    out = P.section(text)
    assert "# Title" not in out and "### Method" in out and "#### Deeper" in out
    assert "# not a heading\n" in out and "(#period-aliases)" in out and "aliases.md" not in out


def test_tiles_come_first_and_the_notes_after(tmp_path):
    gallery = tmp_path / "gallery.md"
    gallery.write_text(GALLERY)
    page, systems = P.render(gallery)
    assert [line for line in page.splitlines() if line.startswith("# ")] == ["# Binary orbits: GRAVITY and PIONIER"]
    first = page.split("\n", 2)[2].lstrip()
    assert first.startswith('<div class="og"></div>')
    assert page.index("## What this page tests") < page.index("og-intro") < page.index("### How to read a plot")
    assert page.index("### How to read a plot") < page.index("## Period aliases") < page.index("## Related pages")
    assert page.index("## Related pages") < page.index("og-static") < page.index("## The numbers without JavaScript")
    assert "Release gate" not in page and "$" + "GALLERY" not in page and "(aliases.md)" not in page
    assert "Gallery not built" in P.render(tmp_path / "missing.md")[0]
    assert set(systems) >= {"gl229", "omi_leo"} and all(f"(aliases/{n}.md)" in page for n in systems)


def test_tiles_link_their_alias_page_and_names_are_greek(tmp_path):
    gallery = tmp_path / "gallery.md"
    gallery.write_text(GALLERY)
    page, _ = P.render(gallery)
    data = json.loads(re.search(r'id="og-data">(.*?)</script>', page).group(1))
    by = {t["slug"]: t for t in data["targets"]}
    assert by["omi-leo"]["aliases"] == "aliases/omi_leo/" and "aliases" not in by["hr6819"]
    assert by["omi-leo"]["target"] == "ο Leo" and "**κ Vel**" in page


def test_greek_names():
    assert greek("kappa Vel, o Leo, alf Equ, pi1 Gru, beta Pic b") == "κ Vel, ο Leo, α Equ, π¹ Gru, β Pic b"
    for text in ("omi_leo", "kap-vel", "beta and gamma", "to Leo", "eta"):  # slugs and words stay as they are
        assert greek(text) == text
