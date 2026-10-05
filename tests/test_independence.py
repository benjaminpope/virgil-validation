"""The simulation code must not lean on virgil (or its Bessel functions)."""

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).parents[1] / "src"
INDEPENDENT = [ROOT / "crosscheck", ROOT / "external_bridge", ROOT / "evidence"]


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_crosscheck_does_not_import_virgil():
    pattern = re.compile(r"^\s*(import|from)\s+(virgil|drpangloss|jaxbessel)\b", re.M)
    offenders = [
        str(p.relative_to(ROOT))
        for src in INDEPENDENT
        for p in src.glob("*.py")
        if pattern.search(p.read_text())
    ]
    assert not offenders, offenders
