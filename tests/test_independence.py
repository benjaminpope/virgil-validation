"""The simulation code must not lean on virgil (or its Bessel functions)."""

import pathlib
import re

SRC = pathlib.Path(__file__).parents[1] / "src" / "crosscheck"


def test_crosscheck_does_not_import_virgil():
    pattern = re.compile(r"^\s*(import|from)\s+(virgil|drpangloss|jaxbessel)\b", re.M)
    offenders = [p.name for p in SRC.glob("*.py") if pattern.search(p.read_text())]
    assert not offenders, offenders
