# Grey-scaled spectra and differential phases

This page checks the flux-spectrum and differential-phase likelihoods against dense Gaussians built with SciPy.
`tests/test_flux_visphi.py` uses a four-telescope file with a Brγ-like line
(unequal phase errors), written by `oifits.write_oifits`. The references
use the values the file holds (EFF_WAVE is float32).

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `observables.FluxSpectrum` (`"flux"`), one scale per dataset or per station, polynomial orders 0–2, the stated prior `scale=(mean, sd)` of virgil#217 | SciPy's multivariate normal about μ t with covariance D + Σ c_j c_jᵀ (c₀ = s t, c_j = τ μ t x^j), t the total spectrum over its group mean | Mathematics | 1e-8 in log L |
| `FluxSpectrum` (`"nflux"`), default prior (1, 0.1) | the same, with t the total spectrum over its continuum mean per row | Mathematics | 1e-8 |
| `likelihood.flux_scale_posterior`, per dataset or per station, polynomial orders 0–1 | the conjugate Gaussian posterior of the weights (k and the polynomial coefficients) under the stated prior, solved with SciPy; the total spectrum's scale is k over the template's group mean | Mathematics | 1e-8 σ in the mean, 1e-8 in the covariance |
| `FluxSpectrum` without a stated prior | refuses rather than taking it from the data | Mathematics | — |
| `observables.DifferentialPhase`, the pipeline's projection, with windows | SciPy's multivariate normal of (Qᵀ ⊗ N_line) φ, with Q from our own incidence matrix and N = I − L our own continuum fit | Mathematics | 1e-7 in log L |
| `DifferentialPhase`: the projection is restricted maximum likelihood (its docstring's claim), with and without windows | differences between models of −½ rᵀPr, P the D-weighted projector off nuisances we build ourselves: closure directions in every channel and, per baseline, offsets and delays or every pattern the line rows of N miss | Mathematics, statistics | 1e-7 relative |
| `DifferentialPhase` with `prior_width` | SciPy's multivariate normal of the closure-free phases, with every baseline's offset and slope (centred wavenumber spanning 1) added to the covariance | Mathematics | 1e-7 |
| `DifferentialPhase`, very wide priors, no windows | tends to the projection (its flat limit) | Mathematics | 1e-6 relative at width 1e5 |

Rechecked after virgil#232, which caches the projection's Cholesky
factor: unchanged agreement.

## Correlated spectral-channel tempering

`tests/test_042_smc_modes_tempering.py` validates `virgil.tempering` against
`crosscheck.smc_tempering_ref` (`scipy`/`numpy` only). For standardized
residual chains along each baseline or triangle spectral window,
`estimate_factors` forms the finite-chain variance ratio `f_raw` (sum of
squared chain sums divided by total sum of squared residuals) and floors `f` at
unity so `block_dof` (`1 / max(f_b, 1)`) never sharpens the likelihood.
Internal `virgil` diagnostic thresholds (`flat_signature`, `out_of_range`, and
the `MIN_CHAINS = 10` pooling fallback) are pinned via `properties:` and
boundary controls under Ruling 2.

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `estimate_factors`, `ChannelFactors`, `block_dof`, `out_of_range` | single- and multi-dataset finite-chain variance ratio `f_raw`, per-block and all-blocks-short `MIN_CHAINS = 10` pooled fallbacks, single-wavelength zero-chain `f = 1.0`, unit floor on alternating anticorrelated chains (`f_raw = 0 < 0.5`), and single- vs two-argument `(0.5, 2.0)` range checks | Mathematics | 1e-14 |
| `estimate_factors` on stationary AR(1) chains (n = 6, ρ = 0.7) | exact finite-n Priestley Cesàro expectation (3.25); control: infinite-chain Bartlett ratio `(1 + ρ) / (1 − ρ) = 5.67` overpredicts by > 2.2 | Mathematics, statistics | < 0.12 (4 000 chains) |
| `estimate_factors`, `flat_signature` on common-mode offsets (n = 8, c = 0.55) | exact common-mode factor `1 + (n − 1) c = 4.85` and lag-2 excess `ρ₂ − ρ₁² > 0.15` | Mathematics, statistics | < 0.20 (3 000 chains) |
| `channel_chains`, `adjacent_channels` | 4-station × 3-snapshot × 5-channel OIFITS layout: identical baseline `(u, v)` / closure triangle legs, identical snapshot MJD, and strictly increasing channel wavelength along every `vis` and `phi` chain | Mathematics | 1e-6 m |


