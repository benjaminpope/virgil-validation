"""Sky-plane comparison of a fitted orbit with a reference orbit.

Orbits are compared where they are observed, as tracks of the companion on the sky at the epochs, not by their
elements: (P, e, i, Omega, omega) have conventions and aliases (Omega and omega each shifted by 180 deg, a mirror
for a bright companion) that give one sky orbit many element sets. The positions come from `crosscheck.orbits`
(our own Kepler solver). The reference's omega must be the secondary's (the companion's), as there: normalise it
first with `crosscheck.references.secondary_omega`, which needs the reference's ``omega_of`` field, rather than
guessing whose omega a paper quotes (a primary's omega is the secondary's + 180 deg, a point reflection).

Solved problems handled here, each commented where it is handled:

* The (Omega + 180, omega + 180) twin is one sky orbit, so comparing positions (or Thiele-Innes constants) makes it
  invisible; a single 180 deg shift is the point reflection.
* A label swap of the two stars negates every relative position, which is the flux-ratio flip of the fit. The score
  compares with the reference or its point reflection, whichever is nearer, and reports which. The reflection is
  accepted only if the fit swapped the labels (``flipped``) or the fit itself reported its labels free
  (``labels_free``, near-equal fluxes). The caller passes the fit's own report, never a list of system names.
  Otherwise a reflected match is a sign error, not agreement, and the verdict is UNEXPLAINED-FLIP.
* A good fit that sits far from the reference is not the fit's failure: with the fit's raw chi^2/N supplied the
  verdict is REFERENCE-DISAGREES (a disagreement to report, with the chi^2 against the reference configuration),
  not DIFFERENT.

``VERDICT_RULE_VERSION`` is stored with every score so a stored verdict can be recognised as stale when the rule
changes.
"""

import numpy as np
from scipy.stats import chi2 as _chi2

from . import orbits as O
from .orbits import thiele_innes  # the one Thiele-Innes implementation (the copy that was here is deleted)

# Variants of the reference tried to diagnose a convention problem: (d_Omega, d_omega, i -> 180 - i)
VARIANTS = {
    "as-is": (0, 0, False),
    "omega+180": (0, 180, False),
    "Omega+180": (180, 0, False),
    "omega+180,Omega+180": (180, 180, False),
    "inc->180-inc": (0, 0, True),
}
SAME_FRACTION = 0.02  # rms offset below this fraction of the mean separation, and
SAME_SIGMA = 3.0  # below this many posterior spreads, is the same sky orbit (published residuals are ~0.1 mas)
VERDICT_RULE_VERSION = 3  # 3: INCOMPLETE without samples, N-aware chi^2 p-value; 2: fraction AND sigma; 1: fraction alone. Bump on any rule change.
# A fit "describes its data" when the upper-tail probability of its raw chi^2 under chi^2_N is at least GOOD_FIT_ALPHA. Stated
# before looking at any system: 1e-3 only rejects a gross misfit (the white errors of these fits are approximate, so a stricter
# alpha would reject fits for noise-model slack, not for the orbit). It was not chosen to make any system pass or fail.
GOOD_FIT_ALPHA = 1e-3


def _pos(mjd, period, t_peri, ecc, inc, omega, Omega, a):
    """Companion (dRA, dDec) at ``mjd`` from `crosscheck.orbits.position`, stacked to shape (..., 2) (mas)."""
    x, y = O.position(mjd, period, t_peri, ecc, inc, omega, Omega, a)
    return np.stack([np.asarray(x), np.asarray(y)], -1)


def ref_pos(ref, mjd, d_omega=0.0, d_Omega=0.0, flip_inc=False):
    """Reference orbit's (dRA, dDec) at ``mjd``, optionally under a convention variant.

    Parameters
    ----------
    ref : dict
        P, t_peri, ecc, inc, omega, Omega, a_mas, with omega the secondary's (see `crosscheck.references`).
    mjd : array_like
        Times, MJD.
    d_omega, d_Omega : float
        Degrees added to omega and Omega (the convention variants of ``VARIANTS``).
    flip_inc : bool
        Replace inc by 180 - inc (reversed sense of motion).

    Returns
    -------
    numpy.ndarray
        Positions, shape (len(mjd), 2), mas.
    """
    inc = 180 - ref["inc"] if flip_inc else ref["inc"]
    return _pos(mjd, ref["P"], ref["t_peri"], ref["ecc"], inc, ref["omega"] + d_omega, ref["Omega"] + d_Omega, ref["a_mas"])


def fit_pos(best, mjd, t_ref):
    """Fitted orbit's (dRA, dDec) at ``mjd``.

    Parameters
    ----------
    best : dict
        period, dt_peri (days from ``t_ref``), ecc, inc, omega, Omega, a_mas.
    mjd : array_like
        Times, MJD.
    t_ref : float
        Reference time of ``dt_peri``, MJD.

    Returns
    -------
    numpy.ndarray
        Positions, shape (len(mjd), 2), mas.
    """
    return _pos(
        mjd, best["period"], t_ref + best["dt_peri"], best["ecc"], best["inc"], best["omega"], best["Omega"], best["a_mas"]
    )


def _rms(d):
    """Root-mean-square length of the 2-vectors ``d`` (shape (N, 2)), as a float."""
    return float(np.sqrt(np.mean(np.sum(d**2, -1))))


def good_fit(chi2_n, n_data):
    """Whether a raw chi^2/N of ``chi2_n`` over ``n_data`` points is consistent with the data: P(chi^2_N >= chi^2) >= ``GOOD_FIT_ALPHA``.
    False when either is missing (nothing then says the fit is good)."""
    if chi2_n is None or n_data is None or not (n_data > 0) or not np.isfinite(chi2_n):
        return False
    return bool(_chi2.sf(chi2_n * n_data, n_data) >= GOOD_FIT_ALPHA)


def verdict_from_stats(frac_sep, offset_over_spread, mirror, flipped=False, labels_free=False, chi2_n=None, n_data=None):
    """The verdict rule of :func:`score`, applied to stored statistics (so a stored run can be re-judged).

    The position match needs ``frac_sep < SAME_FRACTION`` AND, when there are posterior samples
    (``offset_over_spread`` finite), ``offset_over_spread < SAME_SIGMA``. A far orbit whose raw chi^2/N
    (``chi2_n`` over ``n_data`` points) passes `good_fit` is REFERENCE-DISAGREES; without chi^2/N and N it is DIFFERENT.
    A sky match by point estimates alone (``offset_over_spread`` missing: no posterior samples) is INCOMPLETE, since the
    spread gate has not been applied; a failure by the fraction, or an unexplained reflection, needs no spread. A match only as
    the reference's point reflection (``mirror``) needs the fit to have swapped labels or ``labels_free``; otherwise
    UNEXPLAINED-FLIP.
    """
    sky_ok = frac_sep < SAME_FRACTION
    no_spread = offset_over_spread is None or not np.isfinite(offset_over_spread)
    if not no_spread:
        sky_ok = sky_ok and offset_over_spread < SAME_SIGMA
    if not sky_ok:
        # A fit that describes its own data but is far from the reference is a disagreement with the reference,
        # not a failed fit (the caller supplies the raw chi^2/N; without it the verdict is DIFFERENT).
        return "REFERENCE-DISAGREES" if good_fit(chi2_n, n_data) else "DIFFERENT"
    if mirror and not (flipped or labels_free):
        return "UNEXPLAINED-FLIP"
    if no_spread:
        return "INCOMPLETE"
    return "SAME-SKY-ORBIT"


def score(meta, best, t_ref=None, samples=None, flipped=False, labels_free=False, chi2_n=None, n_data=None):
    """Compare a fitted orbit with ``meta["reference"]`` on the sky at ``meta["mjd"]``.

    Parameters
    ----------
    meta : dict
        ``mjd`` (epochs) and ``reference`` (P, t_peri, ecc, inc, omega, Omega, a_mas; omega the secondary's,
        normalised by `crosscheck.references`).
    best : dict
        period, dt_peri (days from ``t_ref``), ecc, inc, omega, Omega, a_mas; optionally ``t_ref``.
    t_ref : float, optional
        Reference time of ``dt_peri`` (default ``best["t_ref"]`` or 0).
    samples : dict of array_like, optional
        The same keys as ``best``, as arrays, for the posterior spread.
    flipped : bool
        The fit's label swap (band.flipped, or fitted flux > 1).
    labels_free : bool
        The fit's own report that its labels are free (near-equal fluxes, so the data cannot tell the stars
        apart). Never derived from a list of system names. Only with this or ``flipped`` is a point-reflected
        match accepted.
    chi2_n : float, optional
        The fit's raw chi^2 per data point. With ``n_data``, when `good_fit` holds and the orbit is far from the
        reference, the verdict is REFERENCE-DISAGREES rather than DIFFERENT.
    n_data : int, optional
        The number of data points behind ``chi2_n``; without it the fit is never called good.

    Returns
    -------
    dict
        rms and max offset (mas), the rms as a fraction of the mean separation (``frac_sep``), the rms over the
        posterior spread, whether the reference's point reflection matches (``flux_flip_matches``), the rms of each
        convention variant, the Thiele-Innes constants, ``rule_version`` (``VERDICT_RULE_VERSION``) and the
        ``verdict``: SAME-SKY-ORBIT, DIFFERENT, REFERENCE-DISAGREES, INCOMPLETE (a match with no ``samples``), or UNEXPLAINED-FLIP (matches only as a point
        reflection, though the fit did not swap labels and ``labels_free`` is false). The position match needs
        both frac_sep < SAME_FRACTION and, with samples, offset_over_spread < SAME_SIGMA.
    """
    mjd = np.asarray(meta["mjd"], float)
    ref = meta["reference"]
    t_ref = best.get("t_ref", 0.0) if t_ref is None else t_ref
    ours = fit_pos(best, mjd, t_ref)
    variants = {}
    for name, (d_Omega, d_omega, flip_inc) in VARIANTS.items():
        r = ref_pos(ref, mjd, d_omega, d_Omega, flip_inc)
        for sign, tag in ((1, ""), (-1, " +flip")):
            variants[name + tag] = _rms(ours - sign * r)
    # The near-equal reflection: for a near-equal pair the data cannot tell the stars apart, so a good fit may sit on
    # the reference's point reflection (every relative position negated). The nearer of the two is compared, and
    # which one is reported as ``mirror``; whether it counts as agreement is decided at the verdict below.
    sign = 1 if variants["as-is"] <= variants["as-is +flip"] else -1
    r0 = sign * ref_pos(ref, mjd)
    dist = np.hypot(*(ours - r0).T)
    sep = np.hypot(*r0.T)
    rms = float(np.sqrt(np.mean(dist**2)))
    out = dict(
        rule_version=VERDICT_RULE_VERSION,
        variant_rms=variants,
        mirror="flip" if sign < 0 else "none",
        flux_flip_matches=sign < 0,
        flipped=bool(flipped),
        labels_free=bool(labels_free),
        chi2_n=None if chi2_n is None else float(chi2_n),
        n_data=None if n_data is None else int(n_data),
        rms=rms,
        max=float(dist.max()),
        mean_sep=float(np.mean(sep)),
        frac_sep=rms / float(np.mean(sep)),
        best_variant=min(variants, key=variants.get),
        best_variant_rms=min(variants.values()),
        TI_ours_ABFG=np.array(thiele_innes(best["a_mas"], best["inc"], best["omega"], best["Omega"])).tolist(),
        TI_ref_ABFG=(sign * np.array(thiele_innes(ref["a_mas"], ref["inc"], ref["omega"], ref["Omega"]))).tolist(),
        spread=float("nan"),
        offset_over_spread=float("nan"),
    )
    if samples is not None:
        n = len(samples["period"])
        idx = np.linspace(0, n - 1, min(n, 400)).astype(int)
        tracks = np.stack(
            [
                _pos(
                    mjd,
                    samples["period"][k],
                    t_ref + samples["dt_peri"][k],
                    samples["ecc"][k],
                    samples["inc"][k],
                    samples["omega"][k],
                    samples["Omega"][k],
                    samples["a_mas"][k],
                )
                for k in idx
            ]
        )
        spread = float(np.sqrt(np.mean(np.sum(tracks.var(0), -1))))
        out["spread"], out["offset_over_spread"] = spread, rms / max(spread, 1e-12)
    out["verdict"] = verdict_from_stats(out["frac_sep"], out["offset_over_spread"], sign < 0, flipped,
                                        labels_free, chi2_n, n_data)
    return out
