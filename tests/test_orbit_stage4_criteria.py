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
    assert set(spec["family"]["holm_members"]) == p_values and len(p_values) == 13


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


def test_nothing_is_left_pending():
    assert "pending_definitions" not in load() and load()["needs_ben_signoff"]


def test_real_data_is_a_gate_that_blocks_adoption():
    spec = load()
    gate = spec["must_hold"]["real_data_gate"]
    assert "real_data_not_held_out" not in spec["must_hold"]
    assert {"a_N_pub_in_99_percent_set", "failures_never_removed", "b_alias_power", "c_sky_comparison", "published_without_errors"} <= set(gate)
    assert ">= 13 of the 14" in gate["a_N_pub_in_99_percent_set"]
    assert "real_data_gate" in spec["decision_rule"] and "blind_inputs" in spec["decision_rule"]
    c = gate["c_sky_comparison"]
    assert "Thiele-Innes" in c and "NOT inflated" in c and "identity" in c and "swap" in c


def test_blind_inputs_record_every_input_and_name_the_hard_systems():
    blind = load()["must_hold"]["blind_inputs"]
    assert {"systems", "record", "gate", "p_window_rule", "flag_rule", "ud_rule"} <= set(blind)
    for word in ("starts", "prior bounds", "P window", "flagged nights", "UD diameter"):
        assert word in blind["record"]
    assert "iota Peg" in blind["systems"] and "Gl 229" in blind["systems"]
    assert "regression only" not in load()["named_targets"]["gl229"].split("blind recovery")[0]


def test_alias_power_and_variance_and_suite_entries():
    ids = {t["id"]: t for t in load()["tests"]}
    power = ids["alias_power_simulated"]
    assert power["kind"] == "gate" and power["threshold"] == 0.5 and "not from a run" in power["derivation"]
    assert "alias_power_simulated" not in load()["family"]["holm_members"]
    var = ids["single_run_variance_calibrated"]
    assert var["threshold"] == 0.8 and var["upper_reported"] is True
    suite = ids["existing_suite_unchanged"]["tolerances"]
    assert "count does not decrease" in suite and "skip" in suite and "xfail" in suite


def test_change_log_row_for_2026_11_08_is_honest_about_seen_results():
    rows = [r for r in load()["change_log"] if str(r["date"]) == "2026-11-08"]
    assert len(rows) == 1 and str(rows[0]["changed_after_seeing_a_result"]).startswith("partly")
    assert "#630" in rows[0]["changed_after_seeing_a_result"]
