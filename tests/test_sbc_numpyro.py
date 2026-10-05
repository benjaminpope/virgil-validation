"""Simulation-based calibration of virgil.likelihood.numpyro_model, from the
summary of the OzSTAR campaign (scripts/sbc_numpyro.py; ozstar_scripts job
sbc_numpyro).

The campaign (tier C) draws truths from Jeffreys priors, simulates data with
our own crosscheck.simulate, samples numpyro_model's posterior with NUTS, and
records each truth's rank among 99 posterior draws. Its criteria are
registered in the script (CRITERIA): rank uniformity by chi-squared with a
Bonferroni family-wise 1%, and 68% and 95% coverage within the 99%
binomial band, over at least 400 replicates. This test reads the committed
summary, checks it was made under the same criteria, and reports it; it
skips until the campaign has run.
"""

import importlib.util
import json
import pathlib

import pytest

from evidence.plugin import record

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "trust" / "campaigns" / "sbc_numpyro_model.json"


def script():
    spec = importlib.util.spec_from_file_location("sbc_numpyro", ROOT / "scripts" / "sbc_numpyro.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not SUMMARY.exists(), reason="the SBC campaign has not run (ozstar_scripts job sbc_numpyro)")
@pytest.mark.validates("virgil.likelihood.numpyro_model", roots=["statistics"], tier="C")
def test_sbc_ranks_are_uniform():
    summary = json.loads(SUMMARY.read_text())
    assert summary["criteria_hash"] == script().criteria_hash(), "criteria changed after the campaign ran"
    for name, p in summary["parameters"].items():
        record(f"chi2_p_{name}", p["chi2_p"])
    record("replicates", summary["replicates"])
    assert summary["replicates"] >= summary["criteria"]["min_replicates"]
    failing = [k for k, p in summary["parameters"].items() if not p["pass"]]
    assert not failing, f"ranks not uniform or coverage off for {failing}"


@pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")
def test_sbc_ranks_and_criteria_are_well_formed():
    """The rank of a circular parameter is measured away from the wrap, and
    the registered criteria are what the docstrings say."""
    s = script()
    assert s.circular_rank([350.0, 355.0, 5.0, 10.0], 359.0) == 2
    assert s.circular_rank([170.0, 180.0, 190.0], 200.0) == 3
    assert s.CRITERIA["draws_per_replicate"] == 99 and s.CRITERIA["family_alpha"] == 0.01
