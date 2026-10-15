"""crosscheck.references: every reference orbit says whose omega it tabulates.

(tests/test_references.py checks our reference *code*; this file checks the reference *data* in oidb/references.)
"""

import numpy as np
import pytest

from crosscheck import orbits as O
from crosscheck import references as R

pytestmark = pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")


def test_there_are_reference_files_and_each_has_orbit_blocks_with_omega_of():
    docs = R.load_references()
    assert docs, "no reference files found"
    problems = {name: R.missing_omega_of(doc) for name, doc in docs.items()}
    assert not any(problems.values()), problems
    blocks = [b for doc in docs.values() for _, b in R.orbit_blocks(doc)]
    assert len(blocks) >= 8 and all(b["omega_of"] in R.OMEGA_OF for b in blocks)


def test_the_schema_check_fails_when_omega_of_is_missing_or_invalid():
    # control: the check above cannot pass by finding nothing
    doc = {"orbit": {"P_day": [1.0, 0.1], "omega_deg": [10.0, 1.0]}}
    assert R.missing_omega_of(doc) and "omega_of" in R.missing_omega_of(doc)[0]
    doc["orbit"]["omega_of"] = "companion"
    assert R.missing_omega_of(doc)
    doc["orbit"]["omega_of"] = "secondary"
    assert R.missing_omega_of(doc) == []
    # an omega_primary_deg key and omega_of = secondary contradict
    assert R.missing_omega_of({"o": {"omega_primary_deg": [1.0], "omega_of": "secondary"}})
    # rows of an all_solutions table may be lists, so the table's children carry the field
    sols = {"orbit": {"all_solutions": {"Table 3": {"col": [[1.0]]}}}}
    assert R.missing_omega_of(sols)
    sols["orbit"]["all_solutions"]["Table 3"]["omega_of"] = "primary"
    assert R.missing_omega_of(sols) == []


def test_the_known_conventions_are_recorded():
    docs = R.load_references()
    xuan = docs["782185b2-0727-42b0-a185-b2072732b047"]["orbit"]
    assert xuan["pmoired_frequentist"]["omega_of"] == "primary"  # Xuan+24 quote the primary Ba's omega
    klement = docs["696baf06-6c3c-424d-abaf-066c3c324d99"]["orbit"]
    assert klement["omega_of"] == "secondary"  # Klement+25: the companion's, omega_shift_deg = 0
    assert docs["696baf06-6c3c-424d-abaf-066c3c324d99"]["conventions"]["omega_shift_deg"] == 0.0


def test_the_primary_omega_is_the_secondarys_plus_180():
    assert R.secondary_omega(100.0, "secondary") == 100.0
    assert R.secondary_omega(100.0, "primary") == 280.0
    assert R.secondary_omega(300.0, "primary") == 120.0  # wrapped into [0, 360)
    for bad in (None, "", "Be", "primary "):
        with pytest.raises(ValueError):
            R.secondary_omega(100.0, bad)


def test_reading_a_primarys_omega_as_the_secondarys_is_the_point_reflection():
    # the reason omega_of is required: the wrong choice negates every relative position
    t = 60000.0 + np.linspace(0.0, 10.0, 7)
    args = (12.0, 60002.0, 0.3, 50.0)
    right = O.position(t, *args, R.secondary_omega(60.0, "primary"), 120.0, 10.0)
    wrong = O.position(t, *args, 60.0, 120.0, 10.0)
    assert np.allclose(right[0], -np.asarray(wrong[0]), atol=1e-9) and np.allclose(right[1], -np.asarray(wrong[1]), atol=1e-9)
    assert np.max(np.hypot(*right)) > 1.0


def _system(omega=102.9):
    return {
        "reference_orbit": dict(
            P_day=[98.8], T0=[57000.5, 0.1, "BJD-2400000"], e=[0.004], omega_deg=[omega], Omega_deg=[10.0],
            i_deg=[60.0], a_mas=[10.0],
        ),
        "reference": {"bibcode": "x"},
    }  # fmt: skip


def test_trackb_reference_needs_omega_of_and_applies_it():
    s = _system()
    with pytest.raises(ValueError):
        R.trackb_reference(s)  # neither given nor in the file
    assert R.trackb_reference(s, omega_of="secondary")["omega"] == pytest.approx(102.9)
    assert R.trackb_reference(s, omega_of="primary")["omega"] == pytest.approx(282.9)
    assert R.trackb_reference(s, omega_of="primary")["t_peri"] == pytest.approx(57000.0)  # BJD -> MJD
    s["reference_orbit"]["omega_of"] = "primary"  # the file's own field is used when the caller gives none
    assert R.trackb_reference(s)["omega"] == pytest.approx(282.9)
    assert R.trackb_reference({"reference_orbit": {}}) is None
