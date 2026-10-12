# Cold-start campaign (registered 2026-10-12)

Planning note, off the site. Criteria: `design/coldstart_pull.yml` (SHA-256 recorded in every
task output and the summary). Script: `scripts/coldstart_pull.py`. Test:
`tests/test_coldstart_pull.py`.

## Why

The adversarial review found that every Monte Carlo campaign starts at the truth.
`scripts/pull_campaign.py` fits with `start=scene.truth`; `scripts/sbc_numpyro.py` uses
`init_to_value(truth)`; `scripts/sbc_v2.py` starts from a grid search, but its call to
`likelihood_grid` predates the model-first argument order (see F21) and it has not run. None
tests finding the mode, which is what fails on real data: the Gl 229 free orbit search lands at
i ~ 110 deg against Xuan et al.'s 31.4 deg though the likelihood prefers theirs by a delta-loss
of 266; the NACO 9 Sgr position angle is 41 deg off; delta Vel 9 deg (`design/plan_eso_binaries.md`).

## What it tests

The pipeline a user runs, with no truth in it: simulate data from a truth drawn per draw
(`crosscheck.simulate`), `virgil.grid_fit.likelihood_grid` (keyword arguments) over a grid that
is the same for every draw of a cell, `best_grid_point`, then `virgil.fitting.fit` from that
point under priors that are the same for every draw of a cell, then `laplace_cov`.
The truth enters only the scoring: a reference fit of the same data started at the truth, and the pulls.

Six cells: `binary_bright` (3 to 20 mas, flux 0.03 to 0.3), `binary_faint` (flux 0.005 to 0.03),
`binary_wide` (15 to 35 mas, several fringe spacings: many modes), `binary_sparse` (one snapshot of
three telescopes: reflection and fringe aliases), `resolved_companion` (a uniform disk of 0.8 to 3 mas
and a companion, diameter in the grid), `envelope` (point star and elongated Gaussian envelope).
Separation, flux (contrast), position angle and noise scale (0.5 to 2 times the pull campaign's
sigma_V2 = 0.02, sigma_CP = 1 deg) are drawn per draw, so the failure rate is reported against S/N,
separation and contrast.

## Primary metric

A draw FAILS when the cold fit raised, was non-finite, or ended at chi2 more than 1 above the
reference fit's. Per cell the failure rate must not exceed a registered bound (1% bright, 2%
resolved and envelope, 5% faint, wide, sparse): exact one-sided binomial test of H0 rate <= bound,
Holm over the six cells at 1%. Power at N = 1000 is in the yml. Cold fits that beat the reference
fit (BETTER) are not failures; equal-likelihood solutions in a different place (ALIAS) are reported.
Secondary and not pass/fail: pulls of the SAME MODE draws (mean and sd tests), gross-error rate
(|pull| > 5), failure rate by S/N, separation and contrast, sensitivity to the 1.0 threshold.

## What it does not cover

- Orbit fits. The orbit-alias search (`virgil.aliases.fit_orbit_aliases`, Gl 229, Track B) is the
  real-data cold start; its cells need many epochs of OIFITS and run for hours each, so it is
  not a Monte Carlo cell here. A synthetic multi-epoch Kepler campaign (`crosscheck.orbits`
  truth, `fit_orbit_aliases` recovery, reference chi2 from the true orbit) is the natural
  follow-up and would reuse this script's definitions.
- The modulated rim (7 parameters, no natural grid), limb-darkened star, and aperture-masking scales.
- Real-data systematics (calibration errors, bandwidth smearing, correlated noise).
- Non-Gaussian errors: S/N is by Gaussian chi-squared gain.

## Provenance of the numbers

The failure bounds and sizing are set from what a user should expect and from the binomial
arithmetic, before any draw was run. The smoke run (`coldstart_pull.py smoke`, three draws per
cell, seeds outside the registered range) is for timing and for checking the pipeline; its
outputs are marked and refused by `aggregate`.

## Amendment 1 (2026-10-12, before any campaign draw)

The smoke run (25 draws per cell, seeds 7.1e9 to 7.1e9+5e6, outside the registered range;
virgil `0c1be69`) showed two flaws of the *grids*, so the grids of two cells were changed
and nothing else (failure definition, bounds, tests, sizes, noise and draw ranges are as
registered; the original file, SHA-256 `5c0229024d9d...`, is at commit `2ca9c5a`).

- `envelope`: the grid had ratio 1.0 and pa 0, the prior's bounds, where `fit` cannot move
  (it warns: "stopped after 0 steps"). 2/25 failed this way. The grid is now strictly inside the
  priors (ratio 0.3 to 0.9, pa 22.5 to 157.5). Result with the amended grid: 0/25.
- `resolved_companion`: a single 4-D grid with a 0.5 mas diameter step failed 4/25, all at high S/N
  (27 to 154): the diameter mismatch is absorbed by a fake companion at the centre. virgil's pipeline
  scans the diameter first (`docs/pipeline.md`, `fit` stage). The start is now two stages (a scan of the
  star's diameter alone on 60 log-spaced values, then the companion grid at that diameter). Result with
  the amended start: 1/25.

Smoke numbers for the unchanged cells (25 draws): `binary_bright` 0, `binary_faint` 1 (a noise
peak at S/N 7.6, dchi2 = 1.8), `binary_wide` 0, `binary_sparse` 2 (aliases: dchi2 = 2.7 and 3.4).
25 draws cannot tell a 5% rate from 10%; they say only that the pipeline runs and that failures occur
in the cells designed to be hard. The bounds were not changed in light of them.

## Cost

About 5 to 10 s per draw on one core (grid, two fits, two Laplace covariances, S/N): 6 cells
x 1000 draws = 6000 draws = 8 to 17 core-hours, as a 6 x 8 array of 125-draw tasks of 15 to 20
minutes (OzSTAR CPU, 4 GB each).
