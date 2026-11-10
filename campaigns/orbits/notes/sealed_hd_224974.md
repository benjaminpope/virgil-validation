---
applies_to:
  slug: hd_224974
visibility: sealed
source: 2023A&A...672A.119G
generated_by: campaigns/build_registries.py
---
Published solution: `oidb/trackb/hd_224974.json` (Gallenne, A.; Mérand, A.; Kervella, P.; et al. 2023, A&A 672, A119).

Kind (from the published orbit): sb2_subresolution.

Third light: none reported

Caveats recorded with the reference: Errors include systematic floors (8-50 uas), so the smallest ellipses are floors not statistical errors. Astrometry is SC-only. The paper's orbit used these positions, so orbit predictions at these epochs are consistency checks only. The Phase 3 products are uncalibrated (SINGLE_SCI_VIS): use closure phases only.

Conventions: Elements from Gallenne et al. 2023, Table 6 (joint radial-velocity and astrometry fit). omega is the primary's (the RV model is V1 = gamma + K1[cos(omega + nu) + e cos omega], V2 with -K2); T_peri is a full JD (here also as MJD = JD - 2400000.5). Omega is the position angle of the ascending node, measured east of north, with the equinox not stated; the time scale of T_peri (UTC, HJD or BJD) is not stated. Positions are the secondary relative to the primary (east and north positive). Checked against the paper's own tables: the tabulated omega and Omega, used as they stand in the standard Campbell formulas, reproduce the published positions (Table B.1); changing either alone by 180 degrees does not. (Equivalent to the companion's omega + 180 degrees with Omega + 180 degrees.) Where e is near zero (AK For, o Leo) omega and T_peri are poorly determined. Compare orbits on the sky by their Thiele-Innes constants (orbit.thiele_innes_mas: with X = cos E - e, Y = sqrt(1 - e^2) sin E, dDec = A X + F Y and dRA = B X + G Y): they do not depend on which star omega refers to or on the (omega, Omega) -> (omega + 180, Omega + 180) degeneracy of astrometry without radial velocities. For kappa Vel they are taken from the paper's own Eqs. 6-7.
