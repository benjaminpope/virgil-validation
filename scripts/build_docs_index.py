"""Write docs/overview.md (the site's Overview page) from README.md.

README links point into docs/ from the repository root; on the site they
are relative to docs/. Run before `mkdocs build`; docs/overview.md is
generated and git-ignored. The home page, docs/index.md, is the Trust
page (scripts/trust.py --index).
"""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
text = (ROOT / "README.md").read_text()
text = text.replace("](docs/", "](")
header = "<!-- Generated from README.md by scripts/build_docs_index.py. -->\n"
(ROOT / "docs" / "overview.md").write_text(header + text)
