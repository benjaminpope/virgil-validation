# HD 45166 (f4afc4cd): why the orbit gallery drops it

Collection `f4afc4cd-fd31-40d3-afc4-cdfd3150d340`, Deshmukh+2025 (A&A 695, L20, arXiv:2503.00981), one GRAVITY epoch
(2023-11-26). **Status: not reproduced, dropped from the gallery. Not a virgil disagreement** (no README findings
row, no `trust/ledger.yml` entry, no xfail test).

## What the paper has and what we get

- Paper: a *measured* position (not an orbital prediction), dE 4.83, dN 9.75 mas, i.e. 10.9 mas at PA 26.3 deg
  (`oidb/references/f4afc4cd-….json`; PA = atan2(dE, dN), the same convention as our dra/ddec).
- Our scored fit: 3.54 mas at PA 302 deg, flux 0.33.

## Root cause

The 3.5 mas, PA 302 deg "companion" is a common calibration/instrumental signal, not HD 45166's companion. The same
solution (about 3.4 mas, PA 301 deg, flux 0.33) appears in the calibrator's own raw closure phases and in
`output_zpcal.fits`. The paper's calibrator (TYC 732-806-1) is itself a binary, so the pipeline-calibrated files carry
its signal; the paper fitted the transfer function with priors from 32 single-star calibrators, which we do not have.
Ruled out: the grid (+-20 mas covers 10.9 mas), the PA/dRA sign convention, and the data selection (continuum
2.02-2.40 um with the qWR lines removed).

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

## What would bring it back

Pipeline-calibrated V2 only (closure phases excluded, since the binary calibrator mostly corrupts those), or the
paper's single-calibrator transfer-function priors. Either needs a new run and a recipe change.
