"""crosscheck.skyplane: one sky orbit under convention relabellings, and a different one."""

import numpy as np
import pytest

from crosscheck import skyplane as S

pytestmark = pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")

REF = dict(P=12.1, t_peri=60002.0, ecc=0.3, inc=50.0, omega=60.0, Omega=120.0, a_mas=10.0)
MJD = 60000.0 + np.array([0.0, 2.4, 5.1, 238.0, 240.6, 247.4])
META = dict(mjd=MJD.tolist(), reference=REF)


def _best(**kw):
    b = dict(period=REF["P"], dt_peri=REF["t_peri"] - 60000.0, ecc=REF["ecc"], inc=REF["inc"], omega=REF["omega"],
             Omega=REF["Omega"], a_mas=REF["a_mas"])
    return {**b, **kw}


def test_the_same_orbit_scores_zero():
    sc = S.score(META, _best(), t_ref=60000.0)
    assert sc["rms"] < 1e-6 and sc["verdict"] == "SAME-SKY-ORBIT" and sc["mirror"] == "none"


def test_the_label_swap_is_a_point_reflection():
    # omega + 180 negates every relative position: the flux-ratio flip, not a different orbit
    sc = S.score(META, _best(omega=REF["omega"] + 180.0), t_ref=60000.0, flipped=True)
    assert sc["rms"] < 1e-6 and sc["mirror"] == "flip" and sc["flux_flip_matches"]
    assert sc["verdict"] == "SAME-SKY-ORBIT"
    assert S.score(META, _best(omega=REF["omega"] + 180.0), t_ref=60000.0, sign_free=True)["verdict"] == "SAME-SKY-ORBIT"


def test_a_reflection_without_a_label_swap_fails():
    # control: -ref with flipped=False and sign_free=False is a sign error (e.g. a closure-phase sign), not agreement
    sc = S.score(META, _best(omega=REF["omega"] + 180.0), t_ref=60000.0, flipped=False)
    assert sc["rms"] < 1e-6 and sc["mirror"] == "flip" and sc["verdict"] == "UNEXPLAINED-FLIP"


def test_a_different_orbit_is_different_and_names_the_convention_that_would_fix_it():
    sc = S.score(META, _best(inc=100.0, a_mas=4.0), t_ref=60000.0)
    assert sc["verdict"] == "DIFFERENT" and sc["rms"] > 1.0


def test_a_wrong_Omega_is_diagnosed():
    # Omega + 180 negates every position, exactly as omega + 180 does (Thiele-Innes A, B, F, G all change sign), so
    # it is the point reflection: not DIFFERENT, but, with no label swap, UNEXPLAINED-FLIP, and a convention variant
    # reproduces it. A genuinely different Omega (+90) is DIFFERENT and no variant fixes it.
    sc = S.score(META, _best(Omega=REF["Omega"] + 180.0), t_ref=60000.0)
    assert sc["verdict"] == "UNEXPLAINED-FLIP"
    assert sc["best_variant_rms"] < 1e-6 and sc["best_variant"].split(" ")[0] in ("omega+180", "Omega+180")
    assert sc["variant_rms"]["Omega+180"] < 1e-6 and sc["variant_rms"]["as-is +flip"] < 1e-6
    sc = S.score(META, _best(Omega=REF["Omega"] + 90.0), t_ref=60000.0)
    assert sc["verdict"] == "DIFFERENT" and sc["best_variant_rms"] > 1.0


def test_a_reversed_sense_of_motion_is_different():
    sc = S.score(META, _best(inc=180.0 - REF["inc"]), t_ref=60000.0)
    assert sc["verdict"] == "DIFFERENT" and sc["best_variant"] == "inc->180-inc" and sc["best_variant_rms"] < 1e-6


def test_a_wide_posterior_does_not_excuse_a_30_percent_error():
    rng = np.random.default_rng(0)
    samples = {k: v + 0.0 * rng.standard_normal(50) for k, v in _best().items()}
    samples["a_mas"] = REF["a_mas"] + 3.0 * rng.standard_normal(50)
    sc = S.score(META, _best(a_mas=REF["a_mas"] * 1.3), t_ref=60000.0, samples=samples)
    assert sc["offset_over_spread"] < S.SAME_SIGMA  # the spread alone would have passed it
    assert sc["frac_sep"] > S.SAME_FRACTION and sc["verdict"] == "DIFFERENT"


def test_a_tight_posterior_on_the_reference_passes():
    rng = np.random.default_rng(0)
    samples = {k: v + 0.0 * rng.standard_normal(50) for k, v in _best().items()}
    samples["a_mas"] = REF["a_mas"] + 0.01 * rng.standard_normal(50)
    sc = S.score(META, _best(), t_ref=60000.0, samples=samples)
    assert sc["verdict"] == "SAME-SKY-ORBIT"
