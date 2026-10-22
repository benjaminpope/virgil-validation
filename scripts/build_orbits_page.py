"""Generate docs/orbits/index.md, the one binary-orbit page, from scripts/templates/orbits_page.md.tmpl.

The page has two generated sections. The orbit gallery is written by ``scripts/orbit_gallery.py render`` to
build/orbit_gallery.md (CI and ``scripts/gallery_pull_back.py`` do this); the period aliases come from
``scripts/build_aliases_page.py`` and the committed results in docs/assets/orbit_aliases/. Each section's
own title is dropped and its headings move one level down, so the page has a single title.

    python scripts/build_orbits_page.py [--gallery FILE] [--out FILE]
"""

import argparse
import pathlib
import re
import string
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_aliases_page as B  # noqa: E402

TEMPLATE = REPO / "scripts" / "templates" / "orbits_page.md.tmpl"
GALLERY = REPO / "build" / "orbit_gallery.md"
OUT = REPO / "docs" / "orbits" / "index.md"
# Links between the pages this one replaces, now anchors within it.
LINKS = {"(aliases.md)": "(#period-aliases)", "(binaries.md)": "(#orbit-gallery)"}
NO_GALLERY = ('!!! note "Gallery not built"\n'
              "    Run `scripts/orbit_gallery.py render --out build/orbit_gallery.md --json docs/assets/orbits.json`, "
              "then this script again.\n")


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


def render(gallery=GALLERY, assets=B.ASSETS, template=TEMPLATE):
    gallery = pathlib.Path(gallery)
    body = section(gallery.read_text()) if gallery.exists() else NO_GALLERY
    return string.Template(template.read_text()).substitute(GALLERY=body, ALIASES=section(B.render(assets)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gallery", default=str(GALLERY))
    ap.add_argument("--assets", default=str(B.ASSETS))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    pathlib.Path(a.out).write_text(render(a.gallery, pathlib.Path(a.assets)))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
