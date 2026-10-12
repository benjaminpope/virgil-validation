"""Precision targets (design/coldstart_pull_precision.yml, design/pull_campaign_precision.yml,
design/sbc_v2_precision.yml; rationale design/precision_targets.md). They check that the registered
files that these addenda amend are untouched, that every power and count stated in the addenda is
what scripts/precision_power.py computes (so the numbers are reproducible), that the helpers are
right, and that the rules (rerun on fresh seeds, order, the multimodality limit) are written where
they must be."""

import hashlib
import importlib.util
import json
import pathlib

import numpy as np
import pytest
import yaml
from scipy import stats

from crosscheck import precision as P

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("precision_power", ROOT / "scripts" / "precision_power.py")
pp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pp)


def load(name):
    return yaml.safe_load((ROOT / "design" / name).read_text())


COLD, PULL, SBC = (load(f"{n}.yml") for n in ("coldstart_pull_precision", "pull_campaign_precision", "sbc_v2_precision"))

# The registered files the addenda amend, as committed before the addenda (never edited).
REGISTERED = {
    "pull_campaign.yml": "75fa91a25fa8dc1d3faefa7a6e5fcf064b842ec7aeb5694384577082460a6439",  # = pull v1's criteria_hash
    "sbc_v2.yml": "e7537647d8d5d74747dc275d20eea3bab77f9614239da94e245565ae3907afc9",
    "sbc_v2_addendum.yml": "ea58d5eb96cfabcc2ca64b7fac69194dcf0d7253e548edee1700724fd2f8c91d",
    "coldstart_pull.yml": "96fb09a785f999f840c41d0c7cfc2d7373705f248d0526dfdc87c10b1d8ddba2",
}


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
@pytest.mark.parametrize("name", sorted(REGISTERED))
def test_registered_files_are_not_edited(name):
    assert hashlib.sha256((ROOT / "design" / name).read_bytes()).hexdigest() == REGISTERED[name]


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_pull_v1_summary_still_matches_its_criteria():
    s = json.loads((ROOT / "trust" / "campaigns" / "pull_campaign.json").read_text())
    assert s["criteria_hash"] == REGISTERED["pull_campaign.yml"]


@pytest.mark.validates("evidence", roots=["mathematics"], kind="reference")
def test_exact_interval_helpers():
    # k = 0: the one-sided upper limit is 1 - alpha^(1/n) (rule of three at alpha 0.05)
    assert P.cp_upper(0, 100, 0.05) == pytest.approx(1 - 0.05 ** (1 / 100), rel=1e-12)
    lo, hi = P.cp_interval(10, 100, 0.025)
    assert (lo, hi) == pytest.approx((0.0490, 0.1762), abs=2e-4)  # the textbook 95% interval of 10/100
    # coverage of an exact one-sided limit: P(limit >= true) >= 1 - alpha for every true rate
    n, alpha = 200, 0.05
    for p in (0.01, 0.05, 0.2):
        ks = np.arange(n + 1)
        ucl = stats.beta.ppf(1 - alpha, ks + 1, np.maximum(n - ks, 1))
        ucl[-1] = 1.0
        assert stats.binom.pmf(ks, n, p)[ucl >= p].sum() >= 1 - alpha - 1e-12
    # the power functions are probabilities that a Monte Carlo reproduces
    rng = np.random.default_rng(0)
    k = rng.binomial(1000, 0.02, 20000)
    assert np.mean(k <= P.max_failures_below(1000, 0.04, 0.01)) == pytest.approx(P.power_upper_below(1000, 0.04, 0.02, 0.01), abs=0.01)
    x = rng.standard_normal((4000, 400))
    mean_ci = stats.norm.isf(0.05) / np.sqrt(400)
    inside = np.abs(x.mean(1)) + mean_ci <= 0.1
    assert inside.mean() == pytest.approx(P.power_mean_equivalence(400, 0.1, 0.0, 0.05), abs=0.02)
    sd = x.std(1, ddof=1)
    half = stats.norm.isf(0.05) * sd * np.sqrt(2 / (4 * 400))
    assert ((sd - half >= 0.9) & (sd + half <= 1.1)).mean() == pytest.approx(P.power_sd_equivalence(400, 0.9, 1.1, 1.0, 0.05), abs=0.03)


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_coldstart_counts_give_the_stated_power():
    cells = COLD["precision"]["cells"]
    assert {c: (v["bound"], v["rule_out"]) for c, v in cells.items()} == pp.COLD
    assert {c: v["bound"] for c, v in cells.items()} == {c: v["failure_bound"] for c, v in load("coldstart_pull.yml")["cells"].items()}
    nmin, nplan = COLD["counts"]["min_draws_per_cell"], COLD["counts"]["draws_per_task"] * COLD["counts"]["tasks_per_cell"]
    assert (nmin, nplan, COLD["counts"]["tasks_per_cell"]) == (1510, 1625, 13)
    for n in range(nmin, nplan + 1):  # power is jagged in n: every n from the minimum to the plan
        assert pp.cold_joint(n) >= COLD["precision"]["design_power"]["min_power"]
        for cell, (bound, rule_out) in pp.COLD.items():
            assert P.power_upper_below(n, rule_out, bound, pp.COLD_ALPHA) >= 0.8
    t = pp.coldstart_table((1000, nplan))
    assert t["binary_bright"][nplan]["max_failures"] == 26 and t["binary_bright"][nplan]["power"] == pytest.approx(0.991, abs=5e-4)
    assert t["resolved_companion"][nplan]["max_failures"] == 46 and t["binary_faint"][nplan]["max_failures"] == 97
    assert pp.cold_joint(nplan) == pytest.approx(0.875, abs=5e-4) and pp.cold_joint(1000) == pytest.approx(0.34, abs=5e-3)
    assert pp.cold_joint(1000) < 0.8  # the registered N = 1000 was too few
    q = COLD["pull_equivalence"]
    assert pp.pull_power(q["min_draws"], q["mean_margin"])["mean"] == pytest.approx(0.88, abs=5e-3)
    assert min(pp.pull_power(q["min_draws"], q["mean_margin"]).values()) >= 0.8


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_pull_counts_give_the_stated_power():
    e = PULL["equivalence"]
    assert (e["sd_interval"], e["mean_margin"], e["alpha"], e["tests"]) == ([0.9, 1.1], 0.1, 0.05, 40)
    nmin, nplan = PULL["counts"]["min_draws_per_cell"], PULL["counts"]["draws_per_task"] * PULL["counts"]["tasks_per_cell"]
    assert (nmin, nplan) == (1850, 1875)
    for n in range(nmin, nplan + 1):
        assert pp.pull_joint(n) >= e["design_power"]
    assert pp.pull_joint(1250) == pytest.approx(0.29, abs=5e-3) and pp.pull_joint(1750) == pytest.approx(0.80, abs=5e-3)
    assert pp.pull_power(1250)["mean"] == pytest.approx(0.941, abs=5e-4) and pp.pull_power(1875)["mean"] == pytest.approx(0.993, abs=5e-4)
    v1 = json.loads((ROOT / "trust" / "campaigns" / "pull_campaign.json").read_text())
    assert sum(len(c["params"]) for c in v1["cells"].values()) * 2 == e["tests"]


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_pull_v1_achieved_precision():
    """The post-hoc statement in design/pull_campaign_precision.yml, recomputed from the committed summary."""
    rows, worst = pp.pull_achieved()
    assert len(rows) == 20 and all(r["n"] == 1250 for r in rows)
    assert all(r["sd_inside_0.9_1.1"] for r in rows)
    assert sum(r["mean_inside_0.1"] for r in rows) == 19
    odd = next(r for r in rows if not r["mean_inside_0.1"])
    assert (odd["cell"], odd["param"]) == ("star_rim_fit", "rim.inc")
    assert odd["mean_ci"][0] == pytest.approx(0.009, abs=5e-4) and odd["mean_ci"][1] == pytest.approx(0.104, abs=5e-4)
    assert worst["smallest_mean_margin_certified"] == pytest.approx(0.1036, abs=5e-4)
    assert worst["smallest_sd_margin_certified"] == pytest.approx(0.0896, abs=5e-4)
    widest = max(rows, key=lambda r: r["sd_ci"][1])
    assert (widest["param"], round(widest["sd_ci"][0], 3), round(widest["sd_ci"][1], 3)) == ("rim.flux", 1.019, 1.090)


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_sbc_counts_give_the_stated_power():
    p = SBC["precision"]
    assert (p["margins"], p["alpha"], p["tests"]) == ({"coverage_68": 0.06, "coverage_95": 0.03, "mean_rank": 0.04}, 0.05, 36)
    nmin = SBC["counts"]["min_replicates_per_regime"]
    for regime in SBC["counts"]["regimes"].values():
        assert regime["tasks"] * regime["replicates_per_task"] == 1280
    for n in range(nmin, 1281):
        assert pp.sbc_joint(n) >= p["design_power"]
    pw = pp.sbc_power(1280)
    assert pw["coverage_68"] == pytest.approx(0.997, abs=5e-4) and pw["coverage_95"] == pytest.approx(0.993, abs=5e-4)
    assert pp.sbc_joint(1280) == pytest.approx(0.87, abs=5e-3) and pp.sbc_joint(1024) == pytest.approx(0.56, abs=5e-3)
    assert pp.sbc_acceptance(68) == (68, 0.68) and pp.sbc_acceptance(95) == (94, 0.94)
    assert p["tests"] == 2 * len(load("sbc_v2.yml")["quantities"]) * 3
    f = pp.width_equivalents()
    assert f[0.9][68] == pytest.approx(0.629, abs=1e-3) and f[1.1][68] == pytest.approx(0.726, abs=1e-3)


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_rerun_on_fresh_seeds_is_a_must_hold_of_every_new_criteria_file():
    bases = {}
    for name, spec in (("coldstart", COLD), ("pull", PULL), ("sbc", SBC)):
        assert "rerun_on_fresh_seeds" in spec["must_hold"] and "fresh seeds" in spec["must_hold"]["rerun_on_fresh_seeds"]
        bases[name] = spec["rerun_seed_base"]
    # the largest seed the first run can use: base + cell index x 1e6 + task x draws (cold-start, pull);
    # base + 1e7 (the low regime's offset) + task x replicates (SBC v2)
    first_max = {"coldstart": 20261012 + 5 * 1_000_000 + 13 * 125, "pull": 20261008 + 5 * 1_000_000 + 16 * 125,
                 "sbc": 20261007 + 10_000_000 + 160 * 8}
    for name, base in bases.items():
        assert base > first_max[name]  # clear of every seed the first run can use
    assert COLD["rerun_seed_base"] == 30261012 and PULL["rerun_seed_base"] == 30261008 and SBC["rerun_seed_base"] == 40261007
    docs = " ".join((ROOT / "docs" / "method" / "index.md").read_text().split())
    assert "Rerun any campaign signal on fresh seeds before ruling on it" in docs


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_order_and_the_multimodality_limit_are_written_down():
    for name in ("coldstart_pull_precision.yml", "sbc_v2_precision.yml"):
        text = (ROOT / "design" / name).read_text()
        assert "runs BEFORE" in text and "cold-start" in text
    assert "coldstart_first" in SBC["must_hold"] and "multimodality_stated" in SBC["must_hold"]
    covered = " ".join((ROOT / "trust" / "not_covered.md").read_text().split())
    assert "Multimodal posteriors" in covered and "says nothing about posteriors with several modes" in covered
