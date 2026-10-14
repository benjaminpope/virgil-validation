# 9 Sgr and HD 168137: onboarding as blind-orbit targets

Status: references and gallery tiles are in; raw-data download and the GRAVITY reduction are a separate job
(`ozstar_scripts/scripts/eso_9sgr`, not yet run; PIONIER and AMBER reduction wait for the JMMC pipelines); no fit has been run. The blind solver is
designed elsewhere, and fits wait for the project lead's sign-off. Nothing here
starts a fit from a published orbit.

## Why these targets

9 Sgr (HD 164794; O3.5 V + O5.5 V, P about 3300 d, e about 0.65, a about 14.8 mas) has
a published astrometric orbit from three VLTI instruments over 2009 to 2017, and a
second, refitted orbit that resolves its mass tension. The astrometry is one binary
with a long period and a large, well-sampled separation: a better test of epoch
astrometry and the blind orbit solver than the 12-day or 40-day gallery targets. It is
also the target whose NACO aperture-masking fit sat at its flux bound
(`design/plan_eso_binaries.md`, provisional).

HD 168137 (O7 V + O8 V, P = 1836 d, e = 0.90) is in the same paper, and shares nights and
programmes. Its published orbit is preliminary and weak (a = 7.0 ± 3.9 mas), so
it can only give a broad posterior check, not a tight pass or fail.

## References

| File | What | Source |
|---|---|---|
| `oidb/references/eso-9sgr.json` | 23 epochs, Le Bouquin+17 orbit, Fabry+21 orbit | Le Bouquin et al. 2017 (A&A 601, A34; arXiv:1608.03525), VizieR J/A+A/601/A34, Fabry et al. 2021 (A&A 651, A119; arXiv:2105.09968) Table A.2 |
| `oidb/references/eso-hd168137.json` | 11 epochs, preliminary Le Bouquin+17 orbit | as above, Fig. A.9 |

Elements were read from the papers' own pages (the Fig. A.7 and A.9 element panels of the
arXiv PDF, not a numbered table: the task brief's "Table 4" is the Fig. A.7 panel), and the 18 + 11
positions from VizieR. The Fabry+21 five extra rows are from its Table A.2. A transcription
check without fitting: `crosscheck.orbits` evaluated at the published epochs reproduces
them to 0.19 mas rms (LB17 9 Sgr), 0.16 mas (Fabry), 0.30 mas (HD 168137), against error ellipses of 0.1 to 2 mas.

Things the references record that a reader should not have to rediscover:

- **ω.** Both papers use the secondary's ω (the relative orbit's), from the ascending node
  where the secondary recedes: virgil's ω, so `omega_shift_deg = 0` (checked numerically, not
  assumed). Fabry+21 Table 1 note (a): the ω of their RV curves is the primary's, 180° away.
- **Error ellipses.** LB17 calls the emax/emin of Table B.1 FWHM; Fabry+21 calls the same numbers
  1σ. The two cannot both be right (factor 2.355). Stored as published. The first blind fit will
  show which reading gives pulls near 1; do not convert beforehand.
- **Fabry+21 gives no a.** The 14.83 mas in the reference is derived from M, P and d (circular, as
  their d came from the same a); an earlier fit to their Table A.2 gave 14.656 mas. Marked as derived in the file.
- **Epoch 57558.303** is labelled PIONIER in Fabry+21 Table A.2, but the archive holds no PIONIER
  frame that night, only GRAVITY, and the next row (57558.305) is the GRAVITY refit. Probably the same night entered twice; to confirm when the night is reduced.
- **Flux ratio.** f_H = 0.62 ± 0.02 (secondary/primary, H band); LB17 assume it is constant over
  epochs. The NACO Ks flux of 0.95 at its bound of 1 (`plan_eso_binaries.md`) is at a different band.

## Which epochs have public raw data

Checked 2026-10-14 against `dbo.raw` and `ivoa.ObsCore` (ESO TAP), by target name and a 36 arcsec box.

9 Sgr, 23 rows of Fabry+21 Table A.2 (22 nights):

| Instrument | Epochs (MJD) | Programme(s) | Calibrators |
|---|---|---|---|
| AMBER (3) | 54995.306, 55644.304, 55648.389 | 083.D-0066(A), 386.D-0198(A), 086.D-0586(B) | HD 174774; HIP 82272 and HIP 88850; none found for 55648 |
| PIONIER (7, public) | 56549.143, 56714.359, 56751.389, 56783.350, 56787.340, 56818.282, 56903.045 | 091.D-0334, 092.D-0590, 093.D-0673(A, C), 60.A-9209, 093.D-0039 | HD 165015 |
| PIONIER (6, no raw) | 56154.134, 56189.027, 56191.087, 56376.318, 56383.324, 56438.244 | none: no PIONIER frame of any target between MJD 56100 and 56500 is public | not reducible |
| PIONIER (3, 2016 to 2017) | 57601.215, 57900.386, 57995.105 | 596.D-0495(D), (J) | HD 164386, HD 165015 |
| GRAVITY (4 rows, 3 nights; single field) | 57558.303 (?), 57558.305, 57560.312, 57647.081 | 60.A-9168 (commissioning) | HD 164031 (`CAL_HD164031_for_HD164794`, two exposures a night) |

The PIONIER archive has Phase 3 products (pndrs-reduced OIFITS, so a cross-check, not an input) for 2016-08-01, 2017-05-27
and 2017-08-30 only. Not in either paper: a GRAVITY frame on 2017-06-22 (099.D-0777(B), one object frame, no sky or calibrator),
and an AMBER night on 2010-06-09 (085.C-0389(B)).

HD 168137: 9 of 11 epochs have raw frames (4 AMBER, 5 PIONIER; two PIONIER epochs, MJD 55782.179 and 56383.375, have none) and four PIONIER Phase 3 epochs after the paper (2017).

## Pipelines

| Instrument | Pipeline | Where it comes from | Note |
|---|---|---|---|
| PIONIER | pndrs (Yorick), the one used by LB17 | JMMC download, login needed | cannot be fetched without credentials; Ben supplies the tarball |
| AMBER | amdlib 3 (Tatulli+07, Chelli+09), as LB17 | JMMC download, login needed | same |
| GRAVITY | ESO pipeline (esorex), as LB17 and Fabry+21 | existing Apptainer image `eso_pipelines_sif` (GRAVITY 1.12.0) | no JMMC |

The point is that virgil's input is the raw frames reduced by the same chain the authors used (the OIFITS
produced by pndrs and amdlib and the ESO pipeline), not the authors' positions. virgil never sees the
published epochs. The OiDB PIONIER L2 collection is private and unused.

Calibrator choices follow the observatory's own OB names (`CAL_<star>_for_HD_164794`), which pair each
science OB with its calibrator: HD 164386 and HD 165015 for PIONIER; for AMBER the calibrator frames carry
`OBJECT = OBJECT` in the archive, so they are found by OB name, not target name. Calibrator diameters come from the
JMMC SearchCal catalogue at reduction time (not stored here).

## Known gaps and next steps

1. Six PIONIER epochs (2012 to mid-2013) cannot be reduced, so the blind orbit sees at most 17 of the
   23 epochs. The June 2016 GRAVITY nights have no p2vm sequence in CalSelector's association; the job takes the
   afternoon sequence of programme 60.A-9256 that follows each night.
2. The 55648 AMBER night (2011-03-27) has calibrator cubes of other targets within an hour of the 9 Sgr OB, but none named for it: check the OB set before reducing.
3. When the raw reduction has run, point `oidb_fit`/the epoch solver at the OIFITS, and add
   `eso-9sgr` and `eso-hd168137` to `oidb_compare.COMPARE`; until then the tiles read "awaiting fit".
4. Fits are for the blind solver (see `design/orbit_comparison.md`), not for this note.
