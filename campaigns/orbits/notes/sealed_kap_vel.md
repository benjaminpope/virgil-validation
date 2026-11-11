---
applies_to:
  slug: kap_vel
visibility: sealed
source: 2026PASP..138b4203R
generated_by: campaigns/build_registries.py
---
Published solution: `oidb/trackb/kap_vel.json` (Rowan, D. M.; Kraus, S.; Thompson, T. A. 2026, PASP 138, 024203).

Kind (from the published orbit): short_arc.

Third light: No tertiary reported. Photocenter bias of Hipparcos parallax up to ~22% noted. Original selection note: companion could in principle be an inner binary, but a luminous secondary was detected.

Caveats recorded with the reference: Single-field AT; sep 2.7-6.6 mas so well inside fibre. Astrometric residuals required scaling bootstrapped errors by s_ast ~ 2.2 (ln s_ast=0.8). Quoted sigma are bootstrap values and underestimate scatter. Separation never below ~2.7 mas vs resolution ~1 mas. Resolution lambda/2D=1.0 mas at 201.5 m. Calibrators differ per night (UD diameters in Table 1). Primary is B2IV photometric primary; the secondary is detected at first epoch.

Conventions: Elements from Rowan, Kraus and Thompson 2026, Table 4 (joint SB1 radial-velocity and GRAVITY astrometry fit). omega is the photometric primary's. T0 (73.1 d) is printed without an epoch: JD - 2416500 is the scale of the paper's RV table and the only offset that fits it, so T_peri here is derived, not stated. The paper's Omega and i are in a form (Thiele-Innes constants with x = -A(cosE - e) - F sqrt(1-e^2) sinE) that does not give the published positions in the standard east-of-north Campbell formulas; the equivalent standard elements, found here by comparing the paper's Eqs. 6-7 with the standard form and then tried against Table 3; their Thiele-Innes constants equal those of the paper's own Eqs. 6-7 to 1e-15, so the mapping is exact algebra, not a fit to Table 3), are Omega = 270 - 109.74 = 160.26 degrees and i = 180 - 74.03 = 105.97 degrees (retrograde, as the position angles run). Use 'standard' for any comparison; the published values stay in 'orbit'. Omega is the position angle of the ascending node, measured east of north, with the equinox not stated; the time scale of T_peri (UTC, HJD or BJD) is not stated. Positions are the secondary relative to the primary (east and north positive).  Compare orbits on the sky by their Thiele-Innes constants (orbit.thiele_innes_mas: with X = cos E - e, Y = sqrt(1 - e^2) sin E, dDec = A X + F Y and dRA = B X + G Y): they do not depend on which star omega refers to or on the (omega, Omega) -> (omega + 180, Omega + 180) degeneracy of astrometry without radial velocities. For kappa Vel they are taken from the paper's own Eqs. 6-7.
