# Detection statistics and injection limits

`tests/test_detection.py`, against our own χ² (`crosscheck.chi2`) on
three-telescope files with and without a companion.

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `limits.injection_limits` (3σ, the injection method of Gallenne et al. 2015) | Brent's method on nsigma(χ²_null(data + signal(f))/χ²_null(data)) = 3 with our χ² | Mathematics | 1e-4 (bisection precision) |
| `detection.detection_statistics`: `delta_chi2`, `log_bayes_factor`, `max_snr` on a 7 × 7 × 60 grid | each rebuilt from our χ²: the best improvement with flux ≥ 0; a log-sum-exp with trapezoid weights in each axis's index; best flux over its curvature error | Mathematics | 1e-5; 1e-8; 1e-3 |
| `detection.local_nsigma` | the one-sided tail of ½χ²₁, √Δχ² | Mathematics | 1e-9 |
| `linear_flux_grid` with `LogUniform(f_min, f_max)` (the Jeffreys prior on a flux ratio) | the evidence ratio and posterior mean and sd by adaptive quadrature in ln f (not virgil's fixed 256-node rule), two bounds, with and without a companion | Mathematics | 1e-6; 1e-5 |
| `detection.gaussian_null`, error scale 1 and 1.5 | simulated minus predicted, over the errors, is 𝒩(0, scale²) (mean, sd, KS) | Statistics | within 4σ |
| `limits.flux_to_contrast`, `flux_to_delta_mag` and inverses | their formulae | Mathematics | 1e-14 |

