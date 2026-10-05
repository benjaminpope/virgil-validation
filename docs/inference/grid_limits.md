# Grid search and detection limits

`tests/test_grids_limits.py`: simulated 3-telescope VLTI data (one closure
phase per snapshot), with and without a companion. Our chi-squared
(`crosscheck.chi2`) reads the file with astropy and uses closed-form binary
visibilities; the significance and limits follow Absil et al. (2011) and
Ruffio et al. (2018), written with SciPy and mpmath.

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `likelihood_grid` (9 × 9 positions) | −½ χ² from our own reading of the file | Mathematics | 5e-12 in log-likelihood |
| `nsigma` | χ² upper tail as a two-sided Gaussian significance (SciPy, in log space) | Mathematics | 2e-13 σ |
| `optimized_flux_grid`, `laplace_flux_uncertainty_grid` | our best flux (Brent) and the curvature of our χ² | Mathematics | < 1e-3 σ and 0.1 % |
| `absil_limits` (3σ, companion-free data, 12 positions) | root of nsigma(χ²(f)/χ²(0)) = 3 with the number of data points as degrees of freedom | Mathematics | 1e-4 (bisection precision since virgil#191) |
| `absil_limits` from a single start far below, far above and at saturation (virgil#191) | the same root | Mathematics | 7.0e-5 (half a bisection step) |
| `ruffio_upperlimit` (means from 20σ above to 30σ below zero) | truncated-Gaussian quantile at 50 digits (mpmath) | Mathematics | 1.5e-12 relative |
| flux at the true position over 200 noisy realisations | N(0, 1) pulls | Statistics | mean −0.13, sd 1.09 |

## Against CANDID

The second root for these steps is CANDID, the code virgil's grid search
follows. It is run on the same files in its own environment
(`tests/test_candid.py`; conventions and problems in
[CANDID](../method/candid.md)). The files: the same three-telescope VLTI
setup, a 0.8 mas uniform-disk primary, with and without a 3 % companion at
(6, −4) mas, with V² and closure phases or V² alone. CANDID's closure-phase
residual is the plain difference and virgil's the chord (ledger D5), so
V²-only files test the shared definition exactly.

| virgil | CANDID | Tag | Agreement |
| --- | --- | --- | --- |
| χ² (`whitened_residuals`, `model_loglike`) at 21 random binaries, V² only | `_chi2Func` × number of data points | CANDID | 1.3e-7 (the file is float32) |
| the same with closure phases | CANDID equals our plain-residual χ² (8e-8); virgil equals the chord one, up to 6e-4 lower far from the data | CANDID + Mathematics | definition D5 |
| `nsigma` (eight cases, 1–27σ) | `_nSigmas` | CANDID | 1e-12 relative |
| `likelihood_grid` as χ²(binary)/χ²(star) over a 32 × 32 map, 2–12 mas | `chi2Map` at 3 % (its fitted diameter) | CANDID | 1.6e-7 (V²), 8e-4 (with closure phases, D5); same minimum, East = +x |
| `absil_limits` (3σ, six positions) | CANDID's Absil criterion solved exactly with its own χ² and nσ | CANDID | 6e-5, virgil's bisection precision since virgil#191 (2e-7 and 4e-6 before) |
| — | CANDID's public `detectionLimit` against that exact solution | Mathematics | 1.0 % low on average (problem P4) |
| `fit` and `laplace_cov` (diameter, position, flux) | `fitMap`, with its √χ²_r scaling of the errors undone | CANDID | best fits 4e-4 σ apart; errors within 1.6 % |
| `injection_limits` (3σ, six positions, V² and V² + CP) | CANDID's injection criterion (`_detectLimit`: its `_injectCompanionData`, then nσ(χ²_UD / χ²_BIN)) solved exactly, the diameter held as virgil holds it; also with no flux bounds (virgil#235) | CANDID | 1.2e-7 (V²), 5.8e-6 (with closure phases) |
| — | the same criterion with the diameter refitted to the injected data, as CANDID does | CANDID | limits 20–34 % higher (definition D8: a refitted disk absorbs part of the companion's V² signal) |

## Against fouriever

`tests/test_fouriever_limits.py`, in fouriever's own environment. Files: three
and four UTs (five snapshots, four channels) without a companion, and a
seven-hole aperture mask (35 closure phases per snapshot, 15 independent, as
for JWST NIRISS AMI) with a 1 % companion.

| virgil | fouriever | Tag | Agreement |
| --- | --- | --- | --- |
| `nsigma`, 15 cases | `util.nsigma` (Absil et al. 2011, eq. 1), with SciPy and mpmath | fouriever | 1e-12; fouriever's SciPy branch saturates near 8σ, virgil's log-space form does not |
| `absil_limits`, `injection_limits`, three UTs | its `lim_absil` and `lim_injection` criteria (with `inj_companion`) solved exactly | fouriever | 4.7e-5 and 2.3e-6 |
| the same, four UTs with correlated closure phases | the same, with fouriever's closure-phase covariance | fouriever | 6.5e-5 and 4e-10 with the same degrees of freedom; 3 % apart with fouriever's own count (definition D7) |
| `likelihood_grid` on the seven-hole mask, correlated closure phases | `chi2_bin` with its covariance, as Δχ² from the null over a 9 × 9 map | fouriever | 5e-5, same minimum |
| — | fouriever's public `detlim` | Mathematics | fails under NumPy 2 and SciPy 1.18 (problem P8, pinned) |

## False-alarm and contrast calibration (campaign)

`scripts/detection_campaign.py` (OzSTAR job `detection_mc`): 10⁴
companion-free and 10⁴ injected four-UT files from our own simulator, and
10⁴ of virgil's own null simulations. `tests/test_detection_campaign.py`
reads its summary, under criteria registered in the script:

- under the null, Δχ² at a fixed position is ½δ₀ + ½χ²₁ (Chernoff 1954), as
  `detection_statistics` documents;
- virgil's `gaussian_null` gives our simulator's distributions of `delta_chi2`,
  `log_bayes_factor` and `max_snr` (two-sample KS);
- `DetectionMC`'s false-alarm probabilities are the documented (k + 1)/(n + 1),
  with SciPy's exact binomial interval, and its thresholds are our quantiles;
- at the true position of an injected companion, Δχ² is noncentral χ²₁(λ), with
  λ our own noiseless χ² (probability integral transform).

Written and smoke-tested on the laptop (3 + 3 + 20 draws, all four checks
pass); not yet run at full size.

