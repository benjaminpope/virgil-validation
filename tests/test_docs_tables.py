"""Every Markdown table in docs/ renders as a table: no row stranded after a
blank line, and every row with as many cells as its header (an unescaped
`|` inside a cell splits it)."""

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def cells(row):
    """Cells of a table row, ignoring escaped pipes and pipes in code spans."""
    row = re.sub(r"`[^`]*`", "code", row.replace("\\|", ""))
    return row.strip().strip("|").count("|") + 1


def tables(text):
    """Blocks of consecutive lines starting with '|'."""
    block, out = [], []
    for i, line in enumerate(text.splitlines() + [""]):
        if line.lstrip().startswith("|"):
            block.append((i + 1, line))
        elif block:
            out.append(block)
            block = []
    return out


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_docs_tables_are_well_formed():
    bad = []
    for path in sorted((ROOT / "docs").rglob("*.md")):
        if path.name in ("index.md", "trust.md") and path.parent == ROOT / "docs":
            continue  # generated HTML, not Markdown tables
        for block in tables(path.read_text()):
            if len(block) < 2 or not re.match(r"^\|\s*:?-{3}", block[1][1].strip()):
                bad.append(f"{path.relative_to(ROOT)}:{block[0][0]}: table rows without a header (a blank line inside a table?)")
                continue
            n = cells(block[0][1])
            bad += [f"{path.relative_to(ROOT)}:{i}: {cells(line)} cells, header has {n}"
                    for i, line in block[2:] if cells(line) != n]
    assert not bad, "\n".join(bad)
