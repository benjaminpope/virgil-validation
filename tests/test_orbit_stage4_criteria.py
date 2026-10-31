"""The Stage 4 criteria file parses and every gated entry is machine-readable (metric, comparator, threshold, n)."""

import pathlib

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")


def load():
    return yaml.safe_load((ROOT / "design" / "orbit_stage4_criteria.yml").read_text())


def test_the_file_parses_and_the_holm_family_is_the_listed_p_values():
    spec = load()
    ids = {t["id"]: t for t in spec["tests"]}
    p_values = {k for k, t in ids.items() if t["kind"] == "p_value"}
    assert set(spec["family"]["holm_members"]) == p_values and len(p_values) == 8


def test_every_gated_entry_states_its_comparator_threshold_and_sample_size():
    for t in load()["tests"]:
        if t["kind"] == "gate":
            assert {"metric", "comparator", "roots"} <= set(t), t["id"]
            assert "threshold" in t or "tolerances" in t, t["id"]
        elif t["kind"] == "p_value":
            assert {"metric", "comparator", "n"} <= set(t), t["id"]
        else:
            assert t["kind"] == "reported" and "reported, not gated" in t["note"], t["id"]


def test_pruning_threshold_is_data_not_a_comment():
    pruning = load()["must_hold"]["pruning_measured"]
    assert pruning["delta"] == 1.0e-3 and pruning["n_se"] == 3
