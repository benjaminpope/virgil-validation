# Detection campaign, version 3

## Why a third version

Version 2 passed, but one of its numbers did not look right. Above virgil's
empirical 1e-2 threshold our null draws exceeded it 288 times of 24000 (1.20%;
240 expected); the 95% interval on the ratio of exceedance rates, ours over
virgil's, was [1.05, 1.37] for `delta_chi2` and for `max_snr`. Fisher's exact test
gave p = 0.006. The version 2 criteria put that test in one Holm family of eleven
p-values at 0.01, most of them tests with no power in the tail (zero fraction,
bulk Anderson–Darling, mean), so 0.006 was not rejected.

## What the excess is (diagnosis, 2026-10-12)

The two simulators draw the same noise. Checked directly, on the campaign
template (four UTs, 5 hour angles, 4 wavelengths: 120 V², 80 closure phases),
without the grid search:

* **Covariance.** 3000 draws from each: V² has σ = 0.01 and no correlation;
  closure phases have unit standardized variance and correlation ±1/3 between
  triangles sharing a baseline, as `ClosureNoise` (R = T Tᵀ/3, Kammerer et al.
  2020) models; the 6400 covariance entries of the closure phases of each
  simulator have mean z² of 1.00 and 1.08 against that R. The chi-squared of
  the null model under virgil's whitening has mean 179.4 (ours) and 179.8
  (virgil's) against 180 degrees of freedom, and the two distributions are
  indistinguishable (KS p = 0.63). Both draw per-baseline phase noise σ/√3;
  `observe` writes σ on every triangle, as the likelihood expects; V² noise is
  additive and unclipped in both; neither draws gains or closure-phase
  offsets (`OIData.gains` and `phase_offsets` are `None` for the file).
* **The statistic is a linear score.** For the campaign template and grid,
  `delta_chi2` equals S = max_j max(0, s_j)², with s_j = g_jᵀ C⁺ r /
  √(g_jᵀ C⁺ g_j) the score of a companion at position j (g_j the response of
  the data to flux), to 0.015 on every one of 300 draws from each simulator.
  So S, which costs a millisecond, carries the whole of the simulators'
  difference through to the statistic.
* **Tails.** 2×10⁵ draws of S from each simulator: 2039 (ours) and 2000
  (virgil's) exceedances of virgil's 1e-2 quantile, ratio 1.02 (95% interval
  [0.96, 1.08]); Kolmogorov–Smirnov p = 0.92, Anderson–Darling p ≥ 0.25. By
  importance sampling (below), with 10⁵ draws each, the exceedance probabilities of
  the two simulators agree within their errors (1–3%) at thresholds from 1e-2
  down to 3×10⁻⁸ (5σ is 2.87×10⁻⁷).
* **Replay of version 2.** The 24000 version 2 null seeds, run through S,
  reproduce the record: 289 exceedances of 13.9 (virgil's empirical 1e-2
  threshold was about that; the recorded count is 288) and 25 of 19.03. Twelve
  other blocks of 24000 (other seeds) give 245 ± 16 of 13.82, the expectation
  (1.02%), against 300 for the version 2 block at the same threshold. The
  version 2 null sample was a fluctuation of about +3σ (p of order 0.003),
  not a property of the simulator. That it is chance, and not something in
  the file or statistic path that S does not see, is what version 3's
  end-to-end tests are for.

So neither simulator is wrong, the noise model they share is the likelihood's,
and virgil's empirical thresholds are right to 1%. What failed was the
campaign: a tail test with 200–300 exceedances, with no stated precision,
absorbed into a family of mostly powerless tests, with simulator and device
confounded (ours on CPU, virgil's null on GPU), so that a 3σ fluctuation could
neither be rejected nor repeated. Ledger entry C3.

## Version 3

**Primary family: the tail.** Its own Holm family at α = 0.05, four tests of
the exceedance-rate ratio ours / virgil's at fixed thresholds
`delta_chi2` = 13.93 (P = 1e-2) and 19.03 (P = 1e-3), each as an *equivalence*:
two one-sided tests that ln(ratio) lies within ±ln 1.3 (a FAP error of 30%, 0.07σ).
An interval too wide for the margin fails. The thresholds are fixed in the
criteria file from the Gaussian reference (the linear statistic S under the
likelihood's own covariance), not from either simulator's sample, so that
neither can set its own pass line.

| test | statistic | draws | counts | s.e. of ln ratio |
| --- | --- | --- | --- | --- |
| `noise_tail_1e-2` | S | 10⁶ per simulator | 10⁴ each | 0.014 |
| `noise_tail_1e-3` | S | 10⁶ per simulator | 10³ each | 0.045 |
| `e2e_tail_1e-2` | virgil's `delta_chi2` | 2.4×10⁴ ours, 1.2×10⁵ virgil's, both on CPU | 240, 1200 | 0.065 |
| `e2e_tail_1e-3` | virgil's `delta_chi2` | 5×10³ importance draws each | ESS ≥ 400 each | ≤ 0.07 |

**Precision target:** the standard error of ln(ratio) must be at most 0.09 in
each test (the aggregator checks it, as a must-hold), so that at a true ratio of 1 each test
passes with probability above 0.99 at the Holm level, and a ratio of 1.2 (as in
version 2) is not certified. The campaign states the false-alarm precision it
validates: 30% at P = 1e-2 and 1e-3 to the statistic's own resolution,
and no better.

**Simulator and device are separated.** All our draws and virgil's
comparison null run on CPU. virgil's null with the *same keys* is repeated on the
GPU, and the two must agree draw by draw (a must-hold), because
`jax.random` is device independent; any difference is numerics, found directly
rather than as a rate. A `check` part runs virgil's `detection_statistics` on
draws of `gaussian_null` and requires `delta_chi2` = S within 0.05 on each
device.

**1e-3 and beyond: importance sampling.** A plain estimate at 1e-3 needs
about 2×10⁵ draws of the statistic per simulator (0.13 s each) for 200
exceedances, and 5σ (2.9×10⁻⁷) is out of reach. The method is a defensive
mixture proposal in noise space (Hesterberg 1995; Rubinstein and Kroese, ch. 5):
draw a component, the null (probability 0.1) or a companion at one of the 288
search positions with the flux whose noiseless χ² improvement equals the target
threshold t (probability 0.9/288 each), simulate with the *simulator under
test* around that mean, and weight by the exact Gaussian density ratio
w = 1 / Σ_k π_k exp(−½ b_kᵀ W b_k + b_kᵀ W r) (W = C⁺, b_k the shift of
component k, r the residual of the draw from the null model). About half the
draws then exceed t. Unbiasedness is checked by the mean weight, which must be
1 within four standard errors, and the importance estimates were checked against
plain Monte Carlo at 1e-2 and 1e-3 (1.02 and 0.93 of plain, the plain counts
having errors of 2% and 7%).
Per threshold, 5000 draws give a relative error about 2–3% (ESS about 3000 of
20000, measured). A generalised-Pareto fit to the top 2% of the 10⁶ plain
draws of S (peaks over threshold; Balkema–de Haan, Pickands) is reported as an
independent cross-check at 1e-4. Importance sampling was chosen over the
tail fit because it uses the exact weights, has a standard error from the draws
themselves, and is unbiased at 5σ; the fit's tail index and threshold
choice would be a modelling assumption in exactly the region being tested.

Beyond P = 1e-3 (1e-4, 3σ, 5σ) the ratios are *reported* with their errors,
not a pass line; the statement is then that false-alarm claims below the
precision shown are not validated.

**Secondary family.** The v2 bulk and boundary tests (Chernoff; k-sample
Anderson–Darling, our simulator against virgil's CPU draws; boundary PIT and
mean), Holm at 0.01, as before. Failing it fails the campaign, but it can no longer hide the tail.

**Counts.** 24000 each of our null and injected files (about 1 CPU-hour each);
120000 virgil null draws on CPU (4.3 CPU-hours) and the same keys on the GPU;
at least 2000 check draws per device; 10⁶ proxy draws per simulator (13 CPU-minutes
for ours, seconds for virgil's); 5000 importance draws per simulator and
threshold (11 CPU-minutes each).
