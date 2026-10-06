"""False-alarm and contrast calibration of virgil.detection, from the
summary of the OzSTAR campaign, version 2 (scripts/detection_campaign.py;
ozstar_scripts job detection_mc_v2): 2x10^4 companion-free and 2x10^4
injected four-UT files from our own simulator, near the detection threshold,
and 10^5 of virgil's own null simulations.

The criteria are registered in design/detection_v2.yml (SHA-256 in the
summary): Chernoff's mixture at a fixed position; virgil's gaussian_null
against our simulator in the bulk (Anderson-Darling) and in the tail
(exceedances, Fisher); DetectionMC's bookkeeping; and delta_chi2 at an
injected companion's true position against max(0, sqrt(lambda) + Z)^2. One
Holm family at 0.01. This test reads the committed summary and skips until
the campaign has run, or when it ran on another virgil commit. Version 1's
summary, if any, is superseded and not read.
"""

import importlib.util
import json
import pathlib

import pytest

from evidence.meta import virgil_source
from evidence.plugin import record

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "trust" / "campaigns" / "detection_mc_v2.json"


def script():
    spec = importlib.util.spec_from_file_location("detection_campaign", ROOT / "scripts" / "detection_campaign.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not SUMMARY.exists(), reason="the detection campaign has not run (ozstar_scripts job detection_mc_v2)")
@pytest.mark.validates("virgil.detection.detection_statistics", "virgil.detection.injection_recovery",
                       "virgil.detection.DetectionMC", "virgil.detection.gaussian_null", roots=["statistics"], tier="C")
def test_detection_campaign_meets_its_criteria():
    summary = json.loads(SUMMARY.read_text())
    assert summary["criteria_hash"] == script().criteria_hash(), "criteria changed after the campaign ran"
    installed = virgil_source()["commit"]
    if summary["virgil_commit"] != installed:
        pytest.skip(f"the campaign ran on virgil {summary['virgil_commit'][:7]}, not the installed {str(installed)[:7]}")
    criteria = script().criteria()
    for name, p in summary["pvalues"].items():
        record(f"p_{name}", p)
    assert summary["counts"]["null"] >= criteria["counts"]["min_null"]
    assert summary["counts"]["inject"] >= criteria["counts"]["min_injected"]
    assert summary["counts"]["virgil"] >= criteria["counts"]["min_virgil"]
    assert summary["design_hash"] == script().design_hash(), "the experiment changed after the campaign ran"
    assert summary["must_hold"]["bookkeeping"]["pass"]
    assert not summary["rejected"], summary["rejected"]

@pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")
def test_detection_campaign_refuses_empty_and_mixed_runs(tmp_path):
    import subprocess
    import sys

    s = script()
    run = lambda *d: subprocess.run([sys.executable, str(ROOT / "scripts" / "detection_campaign.py"), "aggregate",  # noqa: E731
                                     *map(str, d)], capture_output=True, text=True)
    assert "no results found" in run(tmp_path).stderr
    head = {"part": "null", "draws": 1, "criteria_hash": s.criteria_hash(), "design_hash": s.design_hash(),
            "complete": True, "rows": [{"seed": 1, "grid": {}, "fixed": 0.0}]}
    (tmp_path / "null_0000.json").write_text(json.dumps({**head, "task": 0, "seed0": 1, "virgil_commit": "a" * 40}))
    (tmp_path / "null_0001.json").write_text(json.dumps({**head, "task": 1, "seed0": 2, "virgil_commit": "b" * 40,
                                                         "rows": [{"seed": 2, "grid": {}, "fixed": 0.0}]}))
    assert "different virgil commits" in run(tmp_path).stderr
    # the same experiments counted twice: one directory given twice, or a copied task file
    same = tmp_path / "same"
    same.mkdir()
    (same / "null_0000.json").write_text(json.dumps({**head, "task": 0, "seed0": 1, "virgil_commit": "a" * 40}))
    assert "given twice" in run(same, same).stderr
    (same / "null_0001.json").write_text(json.dumps({**head, "task": 1, "seed0": 1, "virgil_commit": "a" * 40}))
    assert "repeated seeds" in run(same).stderr
    # another experiment design is refused
    other = tmp_path / "other"
    other.mkdir()
    (other / "null_0000.json").write_text(json.dumps({**head, "task": 0, "seed0": 1, "virgil_commit": "a" * 40,
                                                      "design_hash": "0" * 16}))
    assert "experiment changed" in run(other).stderr
