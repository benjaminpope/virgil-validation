# Sampling

`numpyro_model` is checked by simulation-based calibration (Talts et al. 2018).
Truths are drawn from the priors, data are simulated from them with our own
code, and NUTS samples virgil's posterior. If that posterior is right, the rank
of each truth among the posterior draws is uniform. Script:
[`scripts/sbc_numpyro.py`](https://github.com/benjaminpope/virgil-validation/blob/main/scripts/sbc_numpyro.py); test:
[`tests/test_sbc_numpyro.py`](https://github.com/benjaminpope/virgil-validation/blob/main/tests/test_sbc_numpyro.py).

**Status: interim. `numpyro_model` is not yet counted as checked.** The
campaign was registered at 1000 replicates before it ran, and only 500 are in.
The analyses are fixed in advance in
[`design/sbc_addendum.yml`](https://github.com/benjaminpope/virgil-validation/blob/main/design/sbc_addendum.yml), registered before the
second 500 were seen.

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

## Interim results (500 of 1000 replicates)

| Parameter | χ² of ranks (p) | Mean rank (p) | 68% coverage | 95% coverage |
| --- | --- | --- | --- | --- |
| diameter | 0.035 | 0.20 | 0.680 | 0.920 |
| flux ratio | 0.50 | 0.80 | 0.656 | 0.942 |
| separation | 0.54 | 0.30 | 0.680 | 0.940 |
| position angle | 0.14 | 0.45 | 0.676 | 0.926 |
| V² error scale | 0.35 | 0.28 | 0.646 | 0.938 |

The expected coverages are 0.68 and 0.94: with 99 draws, the central 95%
window holds 94 of the 100 possible ranks.

No departure from uniform is detected at this size, but the test is not yet
sensitive. At 500 replicates it would detect a posterior 10% too narrow only
about 60% of the time, and a 0.2σ bias only about 40%.

Two observations are being followed up:
- **Diameter.** For true diameters below 0.9 mas, near the 0.5 mas prior
  bound, the 95% coverage was 0.88 (p = 0.003, not significant after about 30
  looks). It is tested on the unseen second 500 alone.
- **Divergences.** Of the 500 replicates, 489 had at least one divergent
  transition (median 28 in 4000 draws). They rise with the companion's flux
  (Spearman ρ = 0.40), not with nearness to the prior bounds. The likely cause
  is the narrow position-angle posterior on the `AngleVector` ring. 65
  replicates had R̂ above 1.01, and a few chains were stuck (worst R̂ 3.2).
