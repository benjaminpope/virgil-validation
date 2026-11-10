---
applies_to:
  slug: nn_del
visibility: sealed
source: 2019A&A...632A..31G
generated_by: campaigns/build_registries.py
---
Published solution: `oidb/trackb/nn_del.json` (Gallenne, A.; Mérand, A.; Kervella, P.; et al. 2019, A&A 632, A31).

Kind (from the published orbit): short_arc.

Third light: None reported (not known to be triple).

Caveats recorded with the reference: Only 4 PIONIER epochs; orbit relies on dense STELLA RVs (final fit uses STELLA RVs only). Eccentric orbit (e=0.52). Fitted parallax 4.2 sigma larger than Gaia DR2 (offset-corrected), consistent with Hipparcos.

Conventions: Elements from Gallenne et al. 2019, Table 2 (joint fit of PIONIER astrometry and STELLA radial velocities). omega is the primary's (RV model V1 = gamma + K1[cos(omega + nu) + e cos omega]; the V2 sign of K2 is opposite although the typeset formula does not show it). T_peri is printed as 5235.164 d with the note '-2445000': read as JD - 2445000. Omega is the position angle of the ascending node, measured east of north, with the equinox not stated; the time scale of T_peri (UTC, HJD or BJD) is not stated. Positions are the secondary relative to the primary (east and north positive). The paper's four positions are reproduced to about 0.003 mas by the standard Campbell formulas with these omega and Omega. Compare orbits on the sky by their Thiele-Innes constants (orbit.thiele_innes_mas: with X = cos E - e, Y = sqrt(1 - e^2) sin E, dDec = A X + F Y and dRA = B X + G Y): they do not depend on which star omega refers to or on the (omega, Omega) -> (omega + 180, Omega + 180) degeneracy of astrometry without radial velocities. For kappa Vel they are taken from the paper's own Eqs. 6-7.
