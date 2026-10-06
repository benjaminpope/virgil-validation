# Sampling

`numpyro_model` is checked by simulation-based calibration (Talts et al. 2018).
Truths are drawn from the priors, data are simulated from them with our own
code, and NUTS samples virgil's posterior. If that posterior is right, the rank
of each truth among the posterior draws is uniform. Script:
[`scripts/sbc_numpyro.py`](https://github.com/benjaminpope/virgil-validation/blob/main/scripts/sbc_numpyro.py); test:
[`tests/test_sbc_numpyro.py`](https://github.com/benjaminpope/virgil-validation/blob/main/tests/test_sbc_numpyro.py).

**Status: passes its registered criteria; not yet counted as checked.** All
1000 registered replicates are in, and every test fixed in advance in
[`design/sbc_addendum.yml`](https://github.com/benjaminpope/virgil-validation/blob/main/design/sbc_addendum.yml)
passes. The campaign ran on virgil 4afc5b8, so it counts only once a CI job
evaluates it against that commit. The sampler was rarely healthy (below), and
a second campaign is planned to fix that.

## Design

Each replicate draws a binary from Jeffreys priors:
- log-uniform diameter (0.5–3 mas), flux ratio (0.02–0.1), separation
  (5–25 mas) and V² error scale (0.7–1.5);
- a uniform position angle, sampled as an `AngleVector`.

Three VLTI UTs observe it; virgil runs NUTS with 4 chains × 1000 draws; each
truth is ranked among 99 thinned draws.

The campaign has two known limits. Every chain starts at the truth, so it
cannot reveal failures to find the right mode. And every companion is bright,
so the posteriors are nearly Gaussian. A second campaign is planned without
either limit.

## Results (1000 replicates)

| Parameter | χ² of ranks (p) | Mean rank (p) | ECDF (p) | 68% coverage | 95% coverage |
| --- | --- | --- | --- | --- | --- |
| diameter | 0.036 | 0.87 | 0.68 | 0.681 | 0.936 |
| flux ratio | 0.55 | 0.15 | 0.24 | 0.668 | 0.944 |
| separation | 0.70 | 0.13 | 0.38 | 0.687 | 0.944 |
| position angle | 0.55 | 0.52 | 0.31 | 0.685 | 0.941 |
| V² error scale | 0.064 | 0.75 | 0.31 | 0.666 | 0.932 |

The expected coverages are 0.68 and 0.94: with 99 draws, the central 95%
window holds 94 of the 100 possible ranks. The pass rule is a Holm correction
over all 25 tests at a family-wise rate of 1%; the smallest p-value, 0.036, is
far from rejection.

- **Diameter near the prior bound.** The 500 replicates seen first suggested
  low 95% coverage for true diameters below 0.9 mas. The registered
  confirmatory test on the unseen second 500 alone (163 replicates) does not
  reproduce it (χ² p = 0.25, coverage p = 0.51).
- **Divergences.** 978 of the 1000 replicates had at least one divergent
  transition (median 29 in 4000 draws). They rise with the companion's flux
  (Spearman ρ = 0.38), not with nearness to the prior bounds. The likely cause
  is the narrow position-angle posterior on the `AngleVector` ring. 111
  replicates had R̂ above 1.01 (worst 3.2). Only 11 replicates were fully
  healthy, too few to test on their own.

The ranks are uniform despite the divergences, but a calibration resting on
unhealthy chains is weak. The second campaign changes the mass matrix and the
ring width to remove them, starts chains away from the truth, and adds faint
companions.
