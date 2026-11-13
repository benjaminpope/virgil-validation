"""Ring features (Lai et al. 2026 eqs 4–9; EHT Paper IV REX) and stage 1 of the M87* replication
(design/m87_replication.md).

Tier A: our extractor (crosscheck.ring_extraction) on analytic rings (criterion 1d), and virgil's
``metrics.ring_features`` against it on the same rings. Tier C: the stage 1 summary
(scripts/m87_stage1.py, trust/campaigns/m87_stage1.json) against Table 1 (1a), their own RingFit (1b)
and virgil (1c).
"""

import json
import pathlib

import numpy as np
import pytest

from crosscheck.preregistration import criteria_commit
from crosscheck.ring_extraction import ring_features, ring_image
from evidence.plugin import record, stamp_campaign
from virgil_bridge import m87 as vb

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "trust" / "campaigns" / "m87_stage1.json"
PS = 225.0 / 64

RINGS = [  # radius, sigma, a, pa, centre (East, North): M87*-like rings on the published 64-pixel grid
    (21.0, 4.0, 0.6, 200.0, (3.0, -2.0)),
    (20.0, 3.0, 0.3, 165.0, (0.0, 0.0)),
    (22.5, 5.0, 0.8, 30.0, (-5.0, 8.0)),
]


def dangle(a, b):
    return abs((a - b + 180) % 360 - 180)


@pytest.mark.parametrize("radius,sigma,a,pa,centre", RINGS)
@pytest.mark.validates("crosscheck.ring_extraction", roots=["mathematics"], kind="reference")
def test_extraction_recovers_an_analytic_ring(radius, sigma, a, pa, centre):
    """1d: diameter within 0.25 px, orientation within 1°, A = a/2 within 0.01, centre within 0.25 px."""
    f = ring_features(ring_image(64, PS, radius, sigma, a=a, pa=pa, centre=centre), PS)
    record("diameter_error_uas", f["diameter"] - 2 * radius)
    record("pa_error_deg", dangle(f["pa"], pa))
    assert abs(f["diameter"] - 2 * radius) <= 0.25 * PS
    assert dangle(f["pa"], pa) <= 1.0
    assert abs(f["asymmetry"] - a / 2) <= 0.01
    assert np.hypot(f["centre"][0] - centre[0], f["centre"][1] - centre[1]) <= 0.25 * PS


@pytest.mark.validates("crosscheck.ring_extraction", roots=["mathematics"], kind="reference")
def test_orientation_is_east_of_north():
    """A ring brightest to the East (left of the image, column 0) has η = 90°; to the North (row 0), 0°."""
    east = ring_image(64, PS, 21.0, 4.0, a=0.9, pa=90.0)
    assert east[:, :32].sum() > east[:, 32:].sum()
    assert dangle(ring_features(east, PS)["pa"], 90.0) <= 1.0
    north = ring_image(64, PS, 21.0, 4.0, a=0.9, pa=0.0)
    assert north[:32].sum() > north[32:].sum()
    assert dangle(ring_features(north, PS)["pa"], 0.0) <= 1.0


@pytest.mark.validates("crosscheck.ring_extraction", roots=["mathematics"], kind="reference")
def test_central_fraction_and_width_of_a_filled_ring():
    """A ring on a uniform interior floor: f_c is the floor over the ring brightness at r_pk, and the
    width is the profile's FWHM broadened by the pixel and bilinear sampling (within one pixel)."""
    im = ring_image(64, PS, 21.0, 3.0) + 0.1 * (np.hypot(*np.meshgrid(*(2 * [(np.arange(64) - 31.5) * PS]))) < 12)
    f = ring_features(im, PS)
    assert abs(f["central_fraction"] - 0.1) <= 0.02
    assert abs(f["width"] - 2.3548 * 3.0) <= PS


@pytest.mark.skipif(not vb.has_ring_features(), reason="virgil without metrics.ring_features (virgil#715)")
@pytest.mark.parametrize("radius,sigma,a,pa,centre", RINGS)
@pytest.mark.validates("virgil.metrics.ring_features", roots=["mathematics"])
def test_virgil_ring_features_match_the_crosscheck(radius, sigma, a, pa, centre):
    """1c on analytic rings: diameter within 0.5 px, orientation within 2°, asymmetry within 0.02 of ours
    and of the truth. Width and central fraction use different floors (EHT's 50/5 μas vs Lai et al.'s
    2 r_pk / 0.2 r_pk) and are not compared."""
    im = ring_image(64, PS, radius, sigma, a=a, pa=pa, centre=centre)
    v, x = vb.ring_features(im, PS), ring_features(im, PS)
    record("diameter_diff_uas", v["diameter"] - x["diameter"])
    record("pa_diff_deg", dangle(v["pa"], x["pa"]))
    assert abs(v["diameter"] - x["diameter"]) <= 0.5 * PS
    assert dangle(v["pa"], x["pa"]) <= 2.0
    assert abs(v["asymmetry"] - x["asymmetry"]) <= 0.02
    assert abs(v["diameter"] - 2 * radius) <= 0.5 * PS


def _summary():
    summary = json.loads(SUMMARY.read_text())
    assert summary["criteria_commit"] == criteria_commit(summary["criteria_file"]), \
        "design/m87_replication.md changed after stage 1 ran: rerun scripts/m87_stage1.py"
    return summary


@pytest.mark.skipif(not SUMMARY.exists(), reason="stage 1 has not run (scripts/m87_stage1.py)")
@pytest.mark.validates("pipeline:m87-closure-invariants", roots=["literature"], tier="C", kind="reference")
def test_published_images_reproduce_table1_ring_diameter_and_orientation():
    """1a: our extraction of their 225 μas median reconstructions against Table 1 (a three-field mean)."""
    summary = _summary()
    for day, d in summary["days"].items():
        record(f"{day}_diameter_uas", d["X"]["diameter"])
        record(f"{day}_pa_deg", d["X"]["pa"])
    failed = {day: d["1a"] for day, d in summary["days"].items() if not all(d["1a"].values())}
    assert not failed, failed


@pytest.mark.skipif(not SUMMARY.exists(), reason="stage 1 has not run (scripts/m87_stage1.py)")
@pytest.mark.validates("virgil.metrics.ring_features", roots=["literature"], tier="C")
def test_virgil_ring_features_on_the_published_images():
    """1c on their images: virgil against our extraction."""
    summary = _summary()
    if not any("1c" in d for d in summary["days"].values()):
        pytest.skip("stage 1 ran without virgil.metrics.ring_features")
    stamp_campaign(summary)
    for day, d in summary["days"].items():
        record(f"{day}_diameter_diff_uas", d["V"]["diameter"] - d["X"]["diameter"])
    failed = {day: d["1c"] for day, d in summary["days"].items() if not all(d["1c"].values())}
    assert not failed, failed


@pytest.mark.skipif(not SUMMARY.exists(), reason="stage 1 has not run (scripts/m87_stage1.py)")
@pytest.mark.validates("external_bridge.m87_worker", roots=["literature"], tier="C", kind="reference")
def test_our_extraction_matches_their_ringfit():
    """1b: their RingFit (GenDIReCT's data/ringFitting.py, in .venv-basis) on the same medians."""
    summary = _summary()
    if not any("1b" in d for d in summary["days"].values()):
        pytest.skip("stage 1 ran without their RingFit (scripts/m87_stage1.py --their; needs .venv-basis)")
    failed = {day: d["1b"] for day, d in summary["days"].items() if not all(d["1b"].values())}
    assert not failed, failed
