"""The registered real-data design (design/realdata_parity.yml, written before any run).

No network and no external package: these check that the registered file is the one that was
committed (its SHA-256 is in design/realdata_parity.sha256, so an edit after the fact shows), that
every OiDB collection it names is in the manifest, and that its counts add up.
"""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "design" / "realdata_parity.yml"
HASH_FILE = ROOT / "design" / "realdata_parity.sha256"


def _design():
    return yaml.safe_load(DESIGN.read_text())


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_registered_design_is_unchanged():
    registered = HASH_FILE.read_text().split()[0]
    assert hashlib.sha256(DESIGN.read_bytes()).hexdigest() == registered, (
        "design/realdata_parity.yml changed after it was registered: a change is a new version "
        "(a new file), not an edit"
    )


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_targets_are_in_the_manifest_and_counts_add_up():
    d = _design()["parity"]
    manifest = json.loads("".join(
        line for line in (ROOT / "oidb" / "manifest.yml").read_text().splitlines(True)
        if not line.startswith("#")))
    ids = {c["id"] for c in manifest["collections"]}
    targets = d["targets"]
    assert len(targets) == d["counts"]["targets"] >= 10
    for t in targets:
        source = t["source"]
        if isinstance(source, dict) and t["id"] != "naco_sam":
            collection = source["oidb"].split(" ")[0]
            assert collection in ids, t["id"]
    assert sum(t["fits"] for t in targets) == d["counts"]["planned_fits"]
    tier_s = sum(t["fits"] for t in targets if t["id"] == "naco_sam")
    assert tier_s == d["counts"]["tier_S_fits"]
    assert d["counts"]["planned_fits"] - tier_s == d["counts"]["scored_fits"]
    for t in targets:
        assert t["externals"] and t["epochs"] and t["model"] and t["literature"], t["id"]


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_every_registered_criterion_has_a_number():
    d = _design()
    tol = d["parity"]["tolerances"]
    assert tol["A1_best_fit"]["per_comparison"] == 0.25
    assert tol["A3_error_bars"]["band"] == [0.90, 1.10]
    assert d["null_test"]["population"]["minimum_size"]["S_clean"] == 30
    assert set(d["null_test"]["procedure"]["variants"]) >= {"raw", "rescaled"}
    assert {"N1_counts", "N2_pit"} <= set(d["null_test"]["criteria"])
    assert d["misspecification_later"]["campaigns"]
