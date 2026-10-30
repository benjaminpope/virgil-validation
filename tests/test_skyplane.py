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


def _tight(best, jitter=0.01, n=50):
    """A narrow posterior about ``best`` (a_mas jittered): the samples the spread gate needs to run."""
    rng = np.random.default_rng(0)
    s = {k: v + 0.0 * rng.standard_normal(n) for k, v in best.items()}
    s["a_mas"] = best["a_mas"] + jitter * rng.standard_normal(n)
    return s


def test_the_same_orbit_scores_zero():
    sc = S.score(META, _best(), t_ref=60000.0, samples=_tight(_best()))
    assert sc["rms"] < 1e-6 and sc["verdict"] == "SAME-SKY-ORBIT" and sc["mirror"] == "none"


def test_the_label_swap_is_a_point_reflection():
    # omega + 180 negates every relative position: the flux-ratio flip, not a different orbit
    refl = _best(omega=REF["omega"] + 180.0)
    sc = S.score(META, refl, t_ref=60000.0, samples=_tight(refl), flipped=True)
    assert sc["rms"] < 1e-6 and sc["mirror"] == "flip" and sc["flux_flip_matches"]
    assert sc["verdict"] == "SAME-SKY-ORBIT"
    assert S.score(META, refl, t_ref=60000.0, samples=_tight(refl), labels_free=True)["verdict"] == "SAME-SKY-ORBIT"


def test_a_reflection_without_a_label_swap_fails():
    # control: -ref with flipped=False and labels_free=False is a sign error (e.g. a closure-phase sign), not agreement
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


def test_point_estimates_without_samples_are_incomplete_not_a_match():
    """D-C: with no posterior samples the spread gate cannot run, so a separation-only match is INCOMPLETE. A failure
    by the separation fraction needs no spread and stays DIFFERENT; the match with samples is unchanged."""
    assert S.score(META, _best(), t_ref=60000.0)["verdict"] == "INCOMPLETE"
    assert S.score(META, _best(a_mas=REF["a_mas"] * 1.3), t_ref=60000.0)["verdict"] == "DIFFERENT"
    assert S.verdict_from_stats(0.001, None, False) == "INCOMPLETE"
    assert S.verdict_from_stats(0.001, float("nan"), False) == "INCOMPLETE"
    assert S.verdict_from_stats(0.001, 1.0, False) == "SAME-SKY-ORBIT"


def test_a_stored_row_is_rejudged_under_the_current_rule():
    """Aliases page: a run made under the older fraction-only rule stores SAME-SKY-ORBIT for an orbit that is
    over 3 posterior spreads off; recomputing from the stored statistics says DIFFERENT. A mirror without a
    label swap is UNEXPLAINED-FLIP, and is accepted with the swap or for a twin."""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    from build_aliases_page import current_verdict
    row = dict(frac_sep=0.0044, offset_over_spread=3.7, mirror="none", flipped=False, verdict="SAME-SKY-ORBIT")
    assert current_verdict(row) == "DIFFERENT"
    assert current_verdict(dict(row, offset_over_spread=2.0)) == "SAME-SKY-ORBIT"
    assert current_verdict(dict(row, offset_over_spread=float("nan"))) == "INCOMPLETE"  # no samples: the spread gate has not run
    assert current_verdict(dict(row, offset_over_spread=2.0, mirror="flip")) == "UNEXPLAINED-FLIP"
    assert current_verdict(dict(row, offset_over_spread=2.0, mirror="flip", flipped=True)) == "SAME-SKY-ORBIT"
    assert current_verdict(dict(row, offset_over_spread=2.0, mirror="flip", labels_free=True)) == "SAME-SKY-ORBIT"
    assert current_verdict(dict(row, frac_sep=0.03, offset_over_spread=2.0)) == "DIFFERENT"


def test_the_score_carries_the_verdict_rule_version():
    assert S.VERDICT_RULE_VERSION == 3
    assert S.score(META, _best(), t_ref=60000.0)["rule_version"] == S.VERDICT_RULE_VERSION


def test_a_good_fit_far_from_the_reference_is_a_reference_disagreement():
    far = _best(inc=100.0, a_mas=4.0)
    assert S.score(META, far, t_ref=60000.0)["verdict"] == "DIFFERENT"  # no chi2 supplied: nothing says the fit is good
    assert S.score(META, far, t_ref=60000.0, chi2_n=1.1, n_data=200)["verdict"] == "REFERENCE-DISAGREES"
    # control: a poor fit far from the reference is still just DIFFERENT; chi2/N without N never calls a fit good
    assert S.score(META, far, t_ref=60000.0, chi2_n=25.0, n_data=200)["verdict"] == "DIFFERENT"
    assert S.score(META, far, t_ref=60000.0, chi2_n=1.1)["verdict"] == "DIFFERENT"
    # a good fit on the reference is the same orbit (given samples)
    assert S.score(META, _best(), t_ref=60000.0, samples=_tight(_best()), chi2_n=1.1, n_data=200)["verdict"] == "SAME-SKY-ORBIT"


def test_the_goodness_of_fit_depends_on_n():
    """chi2/N = 1.5 is a gross misfit for 2000 points (p ~ 1e-27) and unremarkable for 10 (p ~ 0.13); the old rule
    (chi2/N <= 2, N-blind) called both good. The tail probability is checked against SciPy's chi-square survival
    function, and alpha is the module's stated constant."""
    from scipy.stats import chi2
    assert S.GOOD_FIT_ALPHA == 1e-3
    assert S.good_fit(1.5, 10) and not S.good_fit(1.5, 2000)
    assert S.good_fit(1.5, 10) == (chi2.sf(15.0, 10) >= S.GOOD_FIT_ALPHA)
    assert not S.good_fit(None, 10) and not S.good_fit(1.0, None) and not S.good_fit(float("nan"), 10)
    far = _best(inc=100.0, a_mas=4.0)
    assert S.score(META, far, t_ref=60000.0, chi2_n=1.5, n_data=10)["verdict"] == "REFERENCE-DISAGREES"
    assert S.score(META, far, t_ref=60000.0, chi2_n=1.5, n_data=2000)["verdict"] == "DIFFERENT"


def test_the_reflection_is_accepted_only_with_the_fits_own_labels_report():
    refl = _best(omega=REF["omega"] + 180.0)
    assert S.score(META, refl, t_ref=60000.0, samples=_tight(refl), labels_free=True)["verdict"] == "SAME-SKY-ORBIT"
    assert S.score(META, refl, t_ref=60000.0, labels_free=True)["verdict"] == "INCOMPLETE"  # no samples (D-C)
    assert S.score(META, refl, t_ref=60000.0)["verdict"] == "UNEXPLAINED-FLIP"
    with pytest.raises(TypeError):  # the hard-coded system flag is gone
        S.score(META, refl, t_ref=60000.0, sign_free=True)


def test_there_is_one_thiele_innes():
    from crosscheck import orbits as O

    assert S.thiele_innes is O.thiele_innes


def test_the_goodness_of_fit_uses_n_minus_the_fitted_parameters():
    """14 positions at chi2/N = 2.5 (chi2 = 35): against 14 degrees of freedom p = 1.5e-3 passes alpha, against the
    14 - 7 = 7 left after the orbital elements p = 1e-5 does not. Control: the same chi2/N over 2000 points is the usual case."""
    from scipy.stats import chi2
    assert chi2.sf(35.0, 14) >= S.GOOD_FIT_ALPHA > chi2.sf(35.0, 7)
    assert not S.good_fit(2.5, 14)  # default n_free = 7 orbital elements
    assert S.good_fit(2.5, 14, n_free=0)  # the N-blind reading the review objected to
    assert not S.good_fit(1.0, 7) and not S.good_fit(1.0, 5)  # N <= n_free: nothing to test against
    assert S.good_fit(1.05, 2000)
