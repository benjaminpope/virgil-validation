# HD 45166 (f4afc4cd): why the orbit gallery drops it

Collection `f4afc4cd-fd31-40d3-afc4-cdfd3150d340`, Deshmukh+2025 (A&A 695, L20, arXiv:2503.00981), one GRAVITY epoch
(2023-11-26). **Status: not reproduced, dropped from the gallery. Cause unresolved: not yet shown independently of
virgil.** Every fit below is a virgil fit, so a virgil systematic is not excluded; no README findings row, ledger
entry or xfail test has been written because we have not shown a virgil error either. Until a CANDID or PMOIRED fit
on the same data is cited here, treat this as an open question, not a ruling.

## What the paper has and what we get

- Paper: a *measured* position (not an orbital prediction), dE 4.83, dN 9.75 mas, i.e. 10.9 mas at PA 26.3 deg
  (`oidb/references/f4afc4cd-….json`; PA = atan2(dE, dN), the same convention as our dra/ddec).
- Our scored fit: 3.54 mas at PA 302 deg, flux 0.33.

## What the evidence supports (and does not)

- The same solution (about 3.4 mas, PA 301 deg, flux 0.33) is found in HD 45166's raw closure phases, in the
  calibrator's raw closure phases and in `output_zpcal.fits`. That fits a signal common to the data of that night
  (instrumental or calibration), **and equally a virgil systematic tied to the shared uv geometry** (a baseline
  orientation or closure-phase sign problem would give the same PA on all three). Both predict the same result; this
  document cannot tell them apart.
- We do not claim a mechanism for the pipeline-calibrated files. The calibrator is stated to be a binary (the
  paper, Sect. 2), but its own raw closure phases do not show a separate companion (their best fit is the common
  3.4 mas solution), so its binarity is not detected in our fits.
- The 3.5 mas solution is not a demonstrated detection: the CP chi2_red is 0.21 (errors apparently overestimated by
  about 2x, fitted scale 0.46) and CP chi2 differs by only 3.4 over 654 points from the published position. We have
  not computed its significance against a single-star model with rescaled errors. The honest summary is that neither
  data product has been shown to constrain the companion.
- Ruled out: the grid (+-20 mas covers 10.9 mas), the PA/dRA sign convention between our fit and the paper's (dE, dN),
  and the data selection (continuum 2.02-2.40 um with the qWR lines removed).
- The paper fitted the transfer function with priors from 32 single-star calibrators, which we do not have.

## Evidence (all from `scripts/oidb_fit.py recipe_hd45166`, no NUTS)

| fit | position | PA | flux | loss | V2 / CP chi2_red |
|---|---|---|---|---|---|
| scored (uncalibrated V2 + CP, TF as marginalised gains), job 18496382 @ validation 7abfb33 | 3.54 mas | 302 | 0.33 | -10718 | 1731 / 0.22 |
| closure phases only, uncalibrated | 3.53 | 302 | 0.33 | -3729 | n/a / 0.21 |
| same TF-gains model started at the published position, job 18514503 @ e07de91 | 3.30 | 5 | 0.24 | -7775 | 13629 / 2123 |
| pipeline-calibrated V2 + CP (binary calibrator) | 11.22 | 22.4 | 0.87 | -1054 | 1160 / 1384 |
| calibrator TYC 732-806-1, raw CP | 3.36 | 301 | 0.33 | -3741 | n/a / 2.09 |
| `output_zpcal.fits` | 3.33 | 301 | 0.34 | -3078 | 411 / 3.1 |

- Closure phases alone cannot tell the two positions apart (CP chi2 differs by 3.4 over 654 points).
- Scoring V2 as well did not move the fit: started at the published position the TF-gains model drifts away to
  3.3 mas and a loss worse by about 2940, with error scales near 46.
- The paper's position appears only with the pipeline-calibrated files, whose raw chi2/N is 1234 (scales about 35).
- The scale-marginalised grid start does not support data with gains (virgil raises `NotImplementedError`); not
  a finding, a stated limitation.
- Jobs: 18496382 (validation 7abfb33), 18514503 (validation e07de91), virgil 3420ea1. Those two validation commits
  only added a V2-scored recipe and published-start/grid-start variants; they were not merged.

## What would settle it

1. A CANDID or PMOIRED fit (the independent roots `design/plan_oidb.md` lists for this target) on the calibrator's raw
   closure phases and on HD 45166's raw closure phases. If it finds about 3.4 mas at about 301 deg the data carry
   the signal; if not, it is a virgil finding (README findings row, `trust/ledger.yml`, a pinned test and a
   strict-xfail test, per AGENTS.md).
2. The significance of the 3.5 mas CP solution against a single-star model with rescaled errors.
3. Then, to bring HD 45166 back: the paper's single-calibrator transfer-function priors, or a pipeline-calibrated
   V2-only fit. Either needs a new run and a recipe change.
