"""Comparison of fitted binary positions with published ones (Track B).

Independent of virgil: NumPy and SciPy only, from the definitions in
design/trackb_criteria.md.

* Error ellipse to covariance: an ellipse with semi-axes sigma_maj, sigma_min
  (1 sigma) and major axis at position angle theta (East of North) has, in
  (East, North) coordinates, C = sigma_maj^2 u u^T + sigma_min^2 v v^T with
  u = (sin theta, cos theta) and v = (cos theta, -sin theta).
* A Gaussian's FWHM is 2 sqrt(2 ln 2) sigma.
* d^2 = delta^T (C_a + C_b)^{-1} delta is the squared Mahalanobis distance;
  for independent Gaussian errors it follows chi^2 with 2 degrees of freedom.
* Per system, the sum of N independent chi^2_2 values is chi^2_{2N}.
"""

import math
import re

import numpy as np
from scipy import stats

FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
D2_FLAG = float(stats.chi2.isf(1e-3, 2))  # 13.8155..., p = 0.001 for 2 dof
SYSTEM_ALPHA = 1e-3
OVERALL_ALPHA = 1e-2
SCALE_FAILURE = 3.0  # a fitted error scale above this marks an error-model failure
NEAR_EQUAL = ("al_dor", "zet_boo", "hd41255", "hd188088")


def sep_pa_to_offsets(sep, pa_deg):
    """(dra, ddec) East and North from separation and position angle East of North."""
    t = np.radians(pa_deg)
    return sep * np.sin(t), sep * np.cos(t)


def offsets_to_sep_pa(dra, ddec):
    return float(np.hypot(dra, ddec)), float(np.degrees(np.arctan2(dra, ddec)) % 360.0)


def ellipse_covariance(sigma_maj, sigma_min, theta_deg):
    """1-sigma error ellipse (major axis at PA theta, East of North) to a 2x2 (East, North) covariance."""
    t = np.radians(theta_deg)
    u = np.array([np.sin(t), np.cos(t)])
    v = np.array([np.cos(t), -np.sin(t)])
    return sigma_maj**2 * np.outer(u, u) + sigma_min**2 * np.outer(v, v)


def correlated_covariance(sigma_ra, sigma_dec, rho):
    return np.array([[sigma_ra**2, rho * sigma_ra * sigma_dec], [rho * sigma_ra * sigma_dec, sigma_dec**2]])


def mahalanobis2(delta, cov):
    delta = np.asarray(delta, float)
    return float(delta @ np.linalg.solve(np.asarray(cov, float), delta))


def d2_compare(fit, cov_fit, ref, cov_ref, fold180=False):
    """d^2 of fit minus ref with the summed covariance; with fold180, the smaller over fit -> +-fit.

    Returns (d2, sign, delta). The covariance of -fit equals that of fit."""
    fit, ref = np.asarray(fit, float), np.asarray(ref, float)
    cov = np.asarray(cov_fit, float) + np.asarray(cov_ref, float)
    signs = (1, -1) if fold180 else (1,)
    best = min(((mahalanobis2(s * fit - ref, cov), s) for s in signs), key=lambda x: x[0])
    return best[0], best[1], (best[1] * fit - ref).tolist()


def _num(x):
    return None if x is None else float(x)


def published_position(rec, paper):
    """Published (dra, ddec) and its 1-sigma covariance, per design/trackb_criteria.md.

    rec: one reference_positions record of a system.json; paper: the bibcode.
    Returns dict(pos, cov, convention, cov_alt, alt_note) where cov_alt is the uncounted variant
    (Halbwachs+2020 unrescaled, Rowan+2026 unscaled), or None."""
    dra, ddec = _num(rec.get("dra_mas")), _num(rec.get("ddec_mas"))
    if dra is None or ddec is None:
        dra, ddec = sep_pa_to_offsets(float(rec["sep_mas"]), float(rec["pa_deg"]))
    out = dict(pos=[float(dra), float(ddec)], cov_alt=None, alt_note=None)
    smaj, smin, th = (_num(rec.get(k)) for k in ("sigma_major_mas", "sigma_minor_mas", "sigma_pa_deg"))
    if paper.startswith("2017A&A...601A..34L"):  # Le Bouquin+2017: FWHM axes
        out.update(cov=ellipse_covariance(smaj / FWHM_PER_SIGMA, smin / FWHM_PER_SIGMA, th),
                   convention="Le Bouquin+2017 FWHM axes / 2.3548")
    elif paper.startswith("2020MNRAS.496.1355H"):  # Halbwachs+2020: rescaled, adopted as 1 sigma
        out.update(cov=ellipse_covariance(smaj, smin, th), convention="Halbwachs+2020 rescaled ellipse as published")
        k = 0.1626 * 1.0856
        out.update(cov_alt=ellipse_covariance(smaj / k, smin / k, th), alt_note="unrescaled (/0.17652)")
    elif paper.startswith("2026PASP..138b4203R"):  # Rowan+2026: sigmas and rho in the note, times s_ast
        sra, sdec, rho = rowan_sigmas(rec["note"])
        s = math.exp(0.8)
        out.update(cov=correlated_covariance(s * sra, s * sdec, rho), convention="Rowan+2026 x s_ast = e^0.8")
        out.update(cov_alt=correlated_covariance(sra, sdec, rho), alt_note="bootstrap errors unscaled")
    elif paper.startswith("2025OJAp....8E..63W"):  # Waisberg+2025: per system, see published_for_system
        out.update(cov=None, convention="Waisberg+2025 isotropic epoch scatter")
    elif smaj is not None:
        out.update(cov=ellipse_covariance(smaj, smin, th), convention="1-sigma ellipse as published")
    elif _num(rec.get("sigma_dra_mas")) is not None:  # Gallenne+2016: independent dRA, dDec errors
        out.update(cov=np.diag([rec["sigma_dra_mas"] ** 2, rec["sigma_ddec_mas"] ** 2]),
                   convention="independent 1-sigma dRA, dDec")
    else:
        out.update(cov=None, convention="no published errors")
    return out


WAISBERG_SIGMA = {"zet_boo": 0.040, "eta_oph": 0.020}


def published_for_system(system, rec, paper):
    """published_position, with the per-system Waisberg+2025 isotropic errors filled in."""
    out = published_position(rec, paper)
    if out["cov"] is None and system in WAISBERG_SIGMA:
        out["cov"] = np.eye(2) * WAISBERG_SIGMA[system] ** 2
        out["convention"] = f"Waisberg+2025 isotropic {1e3 * WAISBERG_SIGMA[system]:.0f} uas per axis"
    return out


def rowan_sigmas(note):
    """sigma_dRA, sigma_dDec (mas) and rho from a Rowan+2026 note string."""
    def g(pat):
        return float(re.search(pat, note).group(1))
    return (g(r"sigma_dRA=([-+0-9.]+)"), g(r"sigma_dDec=([-+0-9.]+)"), g(r"rho\(dRA,dDec\)=([-+0-9.]+)"))


def system_statistic(d2s):
    """Sum of N counted d^2 against chi^2_{2N}: dict(n, S, dof, p_upper, p_lower, flagged)."""
    d2s = np.asarray([d for d in d2s if d is not None and np.isfinite(d)], float)
    n = d2s.size
    if n == 0:
        return dict(n=0, S=None, dof=0, p_upper=None, p_lower=None, flagged=False)
    S, dof = float(d2s.sum()), 2 * n
    p_up, p_lo = float(stats.chi2.sf(S, dof)), float(stats.chi2.cdf(S, dof))
    return dict(n=n, S=S, dof=dof, p_upper=p_up, p_lower=p_lo, flagged=p_up < SYSTEM_ALPHA,
                conservative=p_lo < SYSTEM_ALPHA)


def overall_ks(d2s):
    """KS of all counted d^2 against chi^2_2: one-sided for d^2 too large (fails if p < 0.01), and two-sided."""
    d2s = np.asarray(d2s, float)
    if d2s.size == 0:
        return dict(n=0, p_larger=None, p_two_sided=None, fails=None)
    cdf = stats.chi2(2).cdf
    p_larger = float(stats.kstest(d2s, cdf, alternative="less").pvalue)
    p_two = float(stats.kstest(d2s, cdf).pvalue)
    return dict(n=int(d2s.size), p_larger=p_larger, p_two_sided=p_two, fails=p_larger < OVERALL_ALPHA,
                n_flagged=int((d2s > D2_FLAG).sum()), expected_flagged=1e-3 * d2s.size)
