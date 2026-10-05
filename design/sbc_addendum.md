# SBC analysis addendum (registered 2026-10-06)

The simulation-based calibration of virgil's `numpyro_model` was registered in
`docs/method/index.md` as 1000 posteriors with a χ² test of uniform ranks.
`scripts/sbc_numpyro.py` added 68% and 95% coverage bands. An adversarial
review of the first 500 replicates (tasks 0–49) showed that these tests:

- detect a 10% posterior-width error only about 55% of the time at n = 500,
  and a 0.2σ bias 37% of the time;
- leave the coverage tests outside the multiple-testing correction, so the
  family-wise false-failure rate is about 8%, not 1%;
- hinted (p = 0.0025, not significant after about 30 looks) that the
  diameter is miscalibrated where the star is unresolved (truth below
  0.9 mas, near the 0.5 mas prior bound).

Before any result of tasks 50–99 was seen, we registered the analyses in
`sbc_addendum.yml`:

1. **Primary.** For each parameter, over all 1000 replicates, five tests:
   - the χ² of the rank histogram;
   - exact binomial tests of 68% and 95% coverage;
   - a z-test of the mean normalized rank, which is sensitive to bias;
   - an ECDF test (the largest gap between the ranks' empirical CDF and
     the uniform one, with its p-value from 10⁴ simulations of uniform
     ranks).

   That is 25 p-values, Holm-corrected at a family-wise 1%. The campaign
   passes only if none is rejected, and only if the original rule (χ²
   Bonferroni at 1% and the coverage bands) also passes.
2. **Confirmatory.** The diameter hint is tested on tasks 50–99 alone,
   which were unseen when it arose: χ² and 95% coverage for replicates with
   a true diameter below 0.9 mas, Holm at 1%.
3. **Descriptive** (reported, not pass/fail): the same tests without
   unhealthy replicates (any divergence, R̂ > 1.01 or bulk ESS < 400), and
   the correlation of divergences with each true parameter. The first 500
   showed divergences rising with the companion's flux (Spearman ρ = 0.40),
   not with the prior bounds.

Starting chains at the truth, and the strong-companion regime, are limits
of this campaign that the addendum does not fix. A second SBC (v2: data-based
starting points, a low-signal regime, a joint test quantity) is planned
separately.
