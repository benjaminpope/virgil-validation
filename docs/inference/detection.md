# Detection statistics and injection limits

`tests/test_detection.py` and `tests/test_sole_checks_hardened.py`, against our
own χ² (`crosscheck.chi2`) and sky observables (`crosscheck.sky`) on three- and
four-telescope OIFITS files (including correlated four-station `gaussian_null`
closure-phase pulls around asymmetric binaries and `sim_null(key, scene=...)`).

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `limits.injection_limits` (3σ, the injection method of Gallenne et al. 2015) | Brent's method on nsigma(χ²_null(data + signal(f))/χ²_null(data)) = 3 with our χ² | Mathematics | 1e-4 (bisection precision) |
| `detection.detection_statistics`: `delta_chi2`, `log_bayes_factor`, `max_snr` on a 7 × 7 × 60 grid | each rebuilt from our χ²: the best improvement with flux ≥ 0; a log-sum-exp with trapezoid weights in each axis's index; best flux over its curvature error | Mathematics | 1e-5; 1e-8; 1e-3 |
| `detection.local_nsigma` | the one-sided tail of ½χ²₁, √Δχ² | Mathematics | 1e-9 |
| `linear_flux_grid` with `LogUniform(f_min, f_max)` (the Jeffreys prior on a flux ratio) | the evidence ratio and posterior mean and sd by adaptive quadrature in ln f (not virgil's fixed 256-node rule), two bounds, with and without a companion | Mathematics | 1e-6; 1e-5 |
| `detection.gaussian_null`, error scale 1 and 1.5 | simulated minus `crosscheck` predicted (`V²` and 3-/4-station correlated closure phases on asymmetric binaries and `sim_null(key, scene=...)`), over the errors, is 𝒩(0, scale²) with signed rank-3 `(T Tᵀ)/3` correlation and `χ²(df=3)` KS | Mathematics + Statistics | within 4σ |
| `limits.flux_to_contrast`, `flux_to_delta_mag` and inverses | their formulae | Mathematics | 1e-14 |

## Monte Carlo detection calibration

`tests/test_detection_tier_a.py` validates `detection.DetectionMC` and
`detection.injection_recovery` at Tier A on every pull request against closed-form
statistical expressions, SciPy distributions, and our independent
`crosscheck.simulate` + `crosscheck.chi2` companion grid search.

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `DetectionMC`: `threshold` | empirical $(1 - \alpha)$ quantile (`numpy.quantile`), bootstrap sd, `RuntimeWarning` when $\alpha n < 1$, and domain `ValueError` guards | Mathematics + Statistics | 1e-13 |
| `DetectionMC`: `false_alarm_probability` | Phipson & Smyth (2010) $(k + 1)/(n + 1)$ and exact Clopper–Pearson Beta bounds (`scipy.stats.beta.ppf`); control rejects uncalibrated $k/n = 0$ at $k = 0$ | Mathematics + Statistics | 1e-15; 1e-14 |
| `DetectionMC`: `completeness`, `contrast_curve`, `roc`, `auc` | Gaussian-shift ROC theory $1 - \Phi(\Phi^{-1}(1 - \alpha) - f/\sigma_s)$, contrast $\sigma_s[\Phi^{-1}(1 - \alpha) - \Phi^{-1}(1 - c)]$, AUC $\Phi(\mu/\sqrt{2})$, and Cartesian/angular `match_radius` filtering | Mathematics + Statistics | < 2 % in contrast; < 0.03 in AUC |
| `DetectionMC`: `concatenate`, `save`, `load` | lossless `.npz` array/metadata round-trip and `ValueError` guards on duplicate seeds, mismatched `match_radius`, or unknown format versions | Mathematics | exact |
| `detection.injection_recovery` | noiseless and noisy (`bootstrap_null` sign-flip) 3-telescope VLTI datasets from `crosscheck.simulate` searched independently via `crosscheck.chi2` + `crosscheck.sky` across all six outputs (`delta_chi2`, `log_bayes_factor`, `max_snr`, `best_dra`, `best_ddec`, `best_flux`) on both `null` and off/on-grid `injected` draws, 7-hole NRM batch/chunk invariance, and `match_radius` sidelobe-alias control | Mathematics | < 1e-5 |


