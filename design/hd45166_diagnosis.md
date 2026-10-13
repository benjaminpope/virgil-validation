# HD 45166 (f4afc4cd): why the orbit gallery drops it

Collection `f4afc4cd-fd31-40d3-afc4-cdfd3150d340`, Deshmukh+2025 (A&A 695, L20, arXiv:2503.00981), one GRAVITY epoch
(2023-11-26). **Status: not reproduced, dropped from the gallery. PMOIRED agrees with virgil on the same raw
closure phases, so this is not a virgil disagreement** (no README findings row, ledger entry or xfail test). The
paper's position is not recoverable from the uncalibrated files we have.

## What the paper has and what we get

- Paper: a *measured* position (not an orbital prediction), dE 4.83, dN 9.75 mas, i.e. 10.9 mas at PA 26.3 deg
  (`oidb/references/f4afc4cd-….json`; PA = atan2(dE, dN), the same convention as our dra/ddec).
- Our scored fit: 3.54 mas at PA 302 deg, flux 0.33.

## Independent check: PMOIRED 26.10.1 (`scripts/pmoired_hd45166.py`)

Same raw files, T3PHI only, same continuum windows, one unresolved companion (PMOIRED's own chi2_red on the quoted
errors; virgil's scored CP chi2_red is 0.21-0.22 after a fitted error scale of 0.46):

| data | start / constraint | x (dE) | y (dN) | rho (mas) | PA (deg) | flux | chi2_red |
|---|---|---|---|---|---|---|---|
| science, raw CP | virgil minimum | -2.991 | 1.866 | 3.53 | 302.0 | 0.329 | 0.22 |
| science, raw CP | free, started at the paper position | 0.020 | 10.376 | 10.38 | 0.1 | 0.134 | 34.9 |
| science, raw CP | fixed at the paper position (4.83, 9.75) | 4.830 | 9.750 | 10.88 | 26.4 | 0.000 | 859 |
| calibrator, raw CP | virgil minimum | -2.891 | 1.718 | 3.36 | 300.7 | 0.326 | 2.15 |
| calibrator, raw CP | fixed at the paper position | 4.830 | 9.750 | 10.88 | 26.4 | 0.000 | 4850 |

- PMOIRED reproduces virgil's science fit to three decimals (virgil: dra -2.993, ddec 1.868, flux 0.329) and the
  calibrator's (virgil: -2.890, 1.717, 0.325). So the 3.4-3.5 mas, PA 301-302 deg solution is in the data and is not
  a virgil systematic (a baseline-orientation or CP-sign error would not be reproduced by a second code).
- The raw closure phases **reject** the paper's position decisively (chi2_red 859 for the science files, with the
  flux driven to 0), and the same signal is in the calibrator's raw closure phases. They are an instrument/
  calibration signature of that night, not HD 45166's companion. (A previous version of this note said the closure
  phases cannot separate the two positions, from a mis-indexed chi2; that was wrong.)
- Not checked: PMOIRED on calibrated data. The paper fitted the transfer function (four calibration strategies, priors
  from 32 single-star calibrators); we do not have those, and the pipeline-calibrated files carry the binary
  calibrator's signal (raw chi2/N 1234).
- We do not claim a mechanism for the pipeline-calibrated files. The calibrator is stated to be a binary (the paper,
  Sect. 2), but its own raw closure phases do not show a separate companion (the best fit is the common 3.4 mas
  solution), so its binarity is not detected in our fits.
- Ruled out: the grid (+-20 mas covers 10.9 mas), the PA/dRA sign convention between our fit and the paper's
  (dE, dN), and the data selection (continuum 2.02-2.40 um with the qWR lines removed).

## Evidence (all from `scripts/oidb_fit.py recipe_hd45166`, no NUTS)

| fit | position | PA | flux | loss | V2 / CP chi2_red |
|---|---|---|---|---|---|
| scored (uncalibrated V2 + CP, TF as marginalised gains), job 18496382 @ validation 7abfb33 | 3.54 mas | 302 | 0.33 | -10718 | 1731 / 0.22 |
| closure phases only, uncalibrated | 3.53 | 302 | 0.33 | -3729 | n/a / 0.21 |
| same TF-gains model started at the published position, job 18514503 @ e07de91 | 3.30 | 5 | 0.24 | -7775 | 13629 / 2123 |
| pipeline-calibrated V2 + CP (binary calibrator) | 11.22 | 22.4 | 0.87 | -1054 | 1160 / 1384 |
| calibrator TYC 732-806-1, raw CP | 3.36 | 301 | 0.33 | -3741 | n/a / 2.09 |
| `output_zpcal.fits` | 3.33 | 301 | 0.34 | -3078 | 411 / 3.1 |

- Scoring V2 as well did not move the fit: started at the published position the TF-gains model drifts away to
  3.3 mas and a loss worse by about 2940, with error scales near 46.
- The paper's position appears only with the pipeline-calibrated files, whose raw chi2/N is 1234 (scales about 35).
- The scale-marginalised grid start does not support data with gains (virgil raises `NotImplementedError`); not
  a finding, a stated limitation.
- Jobs: 18496382 (validation 7abfb33), 18514503 (validation e07de91), virgil 3420ea1. Those two validation commits
  only added a V2-scored recipe and published-start/grid-start variants; they were not merged.

## What would bring it back

The paper's single-calibrator transfer-function priors, or a pipeline-calibrated V2-only fit (closure phases excluded),
run through both virgil and PMOIRED. Either needs new data products or a recipe change.
