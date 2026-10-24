"""Generate docs/orbits/index.md, the binary-orbit page, and its period-alias pages docs/orbits/aliases/<system>.md.

The gallery is written by ``scripts/orbit_gallery.py render`` to build/orbit_gallery.md (CI and
``scripts/gallery_pull_back.py`` do this). This script puts its tiles first under the page title and its notes
after them (scripts/templates/orbits_page.md.tmpl), with each section's title dropped and its headings one level
down. The alias pages come from ``scripts/build_aliases_page.py`` and the committed results in
docs/assets/orbit_aliases/; each gallery tile with alias results links to its page. Bayer star names are written
in Greek (scripts/star_names.py) on every page.

    python scripts/build_orbits_page.py [--gallery FILE] [--out FILE]
"""

import argparse
import json
import pathlib
import re
import string
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_aliases_page as B  # noqa: E402
from star_names import greek  # noqa: E402

TEMPLATE = REPO / "scripts" / "templates" / "orbits_page.md.tmpl"
GALLERY = REPO / "build" / "orbit_gallery.md"
OUT = REPO / "docs" / "orbits" / "index.md"
# Links between the pages this one replaces, now anchors within it.
LINKS = {"(aliases.md)": "(#period-aliases)"}
NO_GALLERY = ('!!! note "Gallery not built"\n'
              "    Run `scripts/orbit_gallery.py render --out build/orbit_gallery.md --json docs/assets/orbits.json`, "
              "then this script again.\n")
DATA = re.compile(r'(<script type="application/json" id="og-data">)(.*?)(</script>)', re.S)
NOTES = re.compile(r'<div class="og-(intro|about)" markdown>.*?\n</div>\n', re.S)
STATIC = re.compile(r'<div class="og-static" markdown>.*?\n</div>\n', re.S)


def section(text):
    """A generated page as a section: its title dropped, headings one level down, links made local."""
    out, fence = [], False
    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            fence = not fence
        if not fence and re.match(r"^# ", line):
            continue
        if not fence and re.match(r"^#{2,5} ", line):
            line = "#" + line
        out.append(line)
    text = "\n".join(out).strip() + "\n"
    for old, new in LINKS.items():
        text = text.replace(old, new)
    return text


def alias_url(name):
    """The alias page of ``name`` relative to the orbit page, as the site serves it (directory URLs)."""
    return f"{B.PAGES}/{name}/"


def link_tiles(gallery, names):
    """The gallery with ``aliases`` (the system's alias page) on each tile that has one."""
    def sub(m):
        data = json.loads(m.group(2))
        by_slug = {s: n for n in names for s in B.tiles(n)}
        for t in data.get("targets", []):
            if t.get("slug") in by_slug:
                t["aliases"] = alias_url(by_slug[t["slug"]])
        return m.group(1) + json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/") + m.group(3)
    return DATA.sub(sub, gallery)


def split(gallery):
    """(tiles, notes, static): the gallery's tiles, its intro and reading notes, and its tables for readers without
    JavaScript (hidden when the tiles work, so they go last)."""
    notes = "\n".join(m.group(0) for m in NOTES.finditer(gallery))
    static = "\n".join(m.group(0) for m in STATIC.finditer(gallery))
    return STATIC.sub("", NOTES.sub("", gallery)).strip() + "\n", notes, static


def aliases_sentence(assets, names):
    pages = ", ".join(f"[{v['label']}]({B.PAGES}/{n}.md)" for n, v in names.items())
    return (f"Where the nights are few, the cycle count between them is ambiguous, and each undersampled system has a "
            f"page of its alias bands, their evidence and which one wins, linked from its tile, or from this list where it has no tile: {pages}. "
            f"{B.tally_sentence(assets, names)}\n\n"
            '??? note "Systems with no alias search"\n\n' + "\n".join("    " + line for line in B.skipped().splitlines()) + "\n")


def render(gallery=GALLERY, assets=B.ASSETS, template=TEMPLATE):
    """(the orbit page, {system: its alias page})."""
    gallery, assets = pathlib.Path(gallery), pathlib.Path(assets)
    names = B._registry()
    if gallery.exists():
        tiles, notes, static = split(link_tiles(gallery.read_text(), names))
        tiles = re.sub(r"\A# .*\n", "", tiles).strip() + "\n"  # the tiles open the page
        notes, static = section(notes), section(static).replace("\n### ", "\n## ")
    else:
        tiles, notes, static = NO_GALLERY, "", ""
    page = string.Template(template.read_text()).substitute(GALLERY=tiles, NOTES=notes, ALIASES=aliases_sentence(assets, names),
                                                            STATIC=static)
    return greek(page), {n: greek(text) for n, text in B.render(assets).items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gallery", default=str(GALLERY))
    ap.add_argument("--assets", default=str(B.ASSETS))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    page, systems = render(a.gallery, pathlib.Path(a.assets))
    out = pathlib.Path(a.out)
    out.write_text(page)
    pages = out.parent / B.PAGES
    pages.mkdir(exist_ok=True)
    for n, text in systems.items():
        (pages / f"{n}.md").write_text(text)
    print(f"wrote {out} and {len(systems)} alias pages in {pages}")


if __name__ == "__main__":
    main()
