"""Write docs/index.md (the site's home page) from README.md.

README links point into docs/ from the repository root; on the site they
are relative to docs/. Run before `mkdocs build`; docs/index.md is
generated and git-ignored.
"""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
text = (ROOT / "README.md").read_text()
text = text.replace("](docs/", "](")
header = "<!-- Generated from README.md by scripts/build_docs_index.py. -->\n"
(ROOT / "docs" / "index.md").write_text(header + text)
