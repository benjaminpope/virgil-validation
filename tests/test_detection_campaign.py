"""False-alarm and contrast calibration of virgil.detection, from the
summary of the OzSTAR campaign (scripts/detection_campaign.py; ozstar_scripts
job detection_mc): 10^4 companion-free and 10^4 injected four-UT files from
our own simulator, and 10^4 of virgil's own null simulations.

The registered criteria (CRITERIA in the script, hash in the summary):
Chernoff's ½δ₀ + ½χ²₁ for delta_chi2 at a fixed position under the null;
virgil's gaussian_null against our simulator (two-sample KS on the three
statistics); DetectionMC's false-alarm probabilities, intervals and
thresholds against our counts; and delta_chi2 at the true position of an
injected companion against noncentral χ²₁(λ), λ our own noiseless
chi-squared, by the probability integral transform. This test reads the
committed summary and skips until the campaign has run, or when it ran on
another virgil commit.
"""

import importlib.util
import json
import pathlib

import pytest

from evidence.meta import virgil_source
from evidence.plugin import record

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "trust" / "campaigns" / "detection_mc.json"


def script():
    spec = importlib.util.spec_from_file_location("detection_campaign", ROOT / "scripts" / "detection_campaign.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not SUMMARY.exists(), reason="the detection campaign has not run (ozstar_scripts job detection_mc)")
@pytest.mark.validates("virgil.detection.detection_statistics", "virgil.detection.injection_recovery",
                       "virgil.detection.DetectionMC", "virgil.detection.gaussian_null", roots=["statistics"], tier="C")
def test_detection_campaign_meets_its_criteria():
    summary = json.loads(SUMMARY.read_text())
    assert summary["criteria_hash"] == script().criteria_hash(), "criteria changed after the campaign ran"
    installed = virgil_source()["commit"]
    if summary["virgil_commit"] != installed:
        pytest.skip(f"the campaign ran on virgil {summary['virgil_commit'][:7]}, not the installed {str(installed)[:7]}")
    for name, check in summary["checks"].items():
        record(f"pass_{name}", float(check["pass"]))
    record("ks_p_noncentral", summary["checks"]["noncentral"]["ks_p_pit"])
    record("chernoff_zero_fraction", summary["checks"]["chernoff"]["zero_fraction"])
    assert summary["counts"]["null"] >= summary["criteria"]["min_null"]
    assert summary["counts"]["inject"] >= summary["criteria"]["min_injected"]
    assert summary["counts"]["virgil"] >= summary["criteria"]["min_virgil"]
    assert summary["design_hash"] == script().design_hash(), "the experiment changed after the campaign ran"
    failing = [k for k, c in summary["checks"].items() if not c["pass"]]
    assert not failing, failing


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
