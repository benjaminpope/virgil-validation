"""False-alarm and contrast calibration of virgil.detection, from the
summaries of the OzSTAR campaigns (scripts/detection_campaign.py).

Version 2 (job detection_mc_v2; criteria design/detection_v2.yml): 2x10^4
companion-free and 2x10^4 injected four-UT files from our own simulator, near
the detection threshold, and 10^5 of virgil's own null simulations; one Holm
family at 0.01. It passed, with a tail excess (288 against 240 exceedances at
the 1e-2 threshold) that is a sample fluctuation (ledger C3,
tests/test_tails.py), and is superseded by version 3: its summary is a
historical record whose criteria hash is still checked, but it no longer
counts as the campaign's evidence.

Version 3 (criteria design/detection_v3.yml, rationale design/detection_v3.md):
the tail is the primary test in its own family, an equivalence of the
exceedance-rate ratio at 1e-2 and 1e-3 (importance sampling at 1e-3); the
bulk and boundary tests are a secondary family; simulator and device are
separated. The test reads trust/campaigns/detection_mc_v3.json and skips until
the campaign has run, or when it ran on another virgil commit.
"""

import importlib.util
import json
import pathlib

import pytest

from evidence.meta import virgil_source
from evidence.plugin import record

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUMMARY_V2 = ROOT / "trust" / "campaigns" / "detection_mc_v2.json"
SUMMARY = ROOT / "trust" / "campaigns" / "detection_mc_v3.json"


def script():
    spec = importlib.util.spec_from_file_location("detection_campaign", ROOT / "scripts" / "detection_campaign.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not SUMMARY_V2.exists(), reason="the version 2 summary is not committed")
@pytest.mark.validates("virgil.detection.detection_statistics", roots=["statistics"], kind="regression", tier="C")
def test_version_2_summary_is_a_historical_record_of_its_own_criteria():
    """Version 2 is superseded (its tail test was absorbed by its Holm family): it is not evidence of the tail, so
    it is a regression record, not a check. Its criteria file is unchanged since it ran."""
    summary = json.loads(SUMMARY_V2.read_text())
    assert summary["criteria_hash"] == script().criteria_hash(script().CRITERIA_V2), "design/detection_v2.yml changed"
    assert summary["pass"] and summary["reported"]["exceedance_ratio_ci"]["delta_chi2@0.01"]["ours"] == 288


@pytest.mark.skipif(not SUMMARY.exists(), reason="the version 3 detection campaign has not run (see design/detection_v3.md)")
@pytest.mark.validates("virgil.detection.detection_statistics", "virgil.detection.injection_recovery",
                       "virgil.detection.DetectionMC", "virgil.detection.gaussian_null", roots=["statistics"], tier="C")
def test_detection_campaign_meets_its_criteria():
    summary = json.loads(SUMMARY.read_text())
    assert summary["version"] == 3
    assert summary["criteria_hash"] == script().criteria_hash(), "criteria changed after the campaign ran"
    installed = virgil_source()["commit"]
    if summary["virgil_commit"] != installed:
        pytest.skip(f"the campaign ran on virgil {summary['virgil_commit'][:7]}, not the installed {str(installed)[:7]}")
    for name, p in summary["pvalues"].items():
        record(f"p_{name}", p)
    for name, test in summary["primary"].items():
        record(f"p_equivalence_{name}", test["p"])
        record(f"ratio_{name}", test["ratio"])
    assert summary["design_hash"] == script().design_hash(), "the experiment changed after the campaign ran"
    assert summary["enough"], summary["counts"]
    assert not summary["rejected_primary"], summary["primary"]  # the tail: equivalence shown in every primary test
    assert not summary["rejected"], summary["rejected"]
    assert all(v["pass"] for v in summary["must_hold"].values()), summary["must_hold"]
    assert summary["pass"]


@pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")
def test_version_3_criteria_are_registered_and_the_aggregator_records_their_hash(tmp_path):
    """The aggregator refuses results made under other criteria, and its summary carries the registered file's
    SHA-256, on a small real run (2e3 proxy draws of each simulator; the counts are far short, so it does not pass)."""
    import hashlib
    import subprocess
    import sys

    s = script()
    assert s.CRITERIA_FILE.name == "detection_v3.yml"
    spec = s.criteria()
    assert set(spec["thresholds"]) == {"1e-2", "1e-3", "1e-4", "3sigma", "5sigma"}
    assert set(spec["primary"]["tests"]) == {"noise_tail_1e-2", "noise_tail_1e-3", "e2e_tail_1e-2", "e2e_tail_1e-3"}
    assert spec["primary"]["margin"] == 1.3 and spec["primary"]["family_alpha"] == 0.05
    assert "chernoff_mean" in spec["secondary"]["tests"] and not set(spec["secondary"]["tests"]) & set(spec["primary"]["tests"])
    env = {**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")}
    cmd = [sys.executable, str(ROOT / "scripts" / "detection_campaign.py")]
    for sim in ("ours", "virgil"):
        out = subprocess.run([*cmd, "run", "--part", f"proxy_{sim}", "--task", "0", "--draws", "2000", "--out", str(tmp_path)],
                             capture_output=True, text=True, env=env)
        assert out.returncode == 0, out.stderr[-2000:]
    out = subprocess.run([*cmd, "aggregate", str(tmp_path), "--summary", str(tmp_path / "summary.json")],
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["version"] == 3
    assert summary["criteria_hash"] == hashlib.sha256((ROOT / "design" / "detection_v3.yml").read_bytes()).hexdigest()
    assert summary["pass"] is False and summary["enough"] is False  # 2e3 draws are not 1e6
    assert summary["primary"]["noise_tail_1e-2"]["ours"] > 0 and summary["primary"]["noise_tail_1e-2"]["virgil"] > 0
    # results made under other criteria are refused
    result = next(tmp_path.glob("proxy_ours_*.json"))
    altered = json.loads(result.read_text())
    altered["criteria_hash"] = "0" * 64
    result.write_text(json.dumps(altered))
    refused = subprocess.run([*cmd, "aggregate", str(tmp_path)], capture_output=True, text=True, env=env)
    assert "criteria changed" in refused.stderr


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
