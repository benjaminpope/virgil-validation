# Analytic marginalisation of linear parameters

`tests/test_linear_marginals.py`. virgil is adopting Luger, Foreman-Mackey
& Hogg (2017) across the code. Each use is checked against a dense Gaussian
built with NumPy from our own reading of the file, or against brute-force
integration.

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `virgil._linear` (`LinearMarginal`, `posterior`, `whiten_cholesky`, `whiten_rank_one`, `whiten_blocks`), the shared helper since virgil#217: random designs, diagonal and full priors, rank-deficient designs, both whitening methods | dense 𝒩(m + Aμ, D + AΛAᵀ) from SciPy: log density, Mahalanobis distance and log-determinant; the whitening matrix M with MᵀM = C⁻¹; blocks with spanning columns; the conditional posterior by direct conditioning; the narrow-prior and flat-prior limits; gradients against finite differences at equal columns and zero width | Mathematics | 1e-10 to 1e-12 |
| `OIData.with_gains`: telescope, baseline and chromatic groups, alone and together, four UTs, V² | Δ log-likelihood = log 𝒩(r; 0, D + UUᵀ) − log 𝒩(r; 0, D), with U = 2V²_model τ m and the modes as documented (chromatic shape (λ_ref/λ)², λ_ref the median wavelength) | Mathematics | 1e-12 |
| the same with supplied modes, one value per sample in file order | the same | Mathematics | 1e-11 |
| the same on amplitudes (`OI_VIS` `VISAMP`) | U = \|V\|_model τ m | Mathematics | 1e-12 |
| `linear_flux_grid`: closed-form flux and its error | a Gauss–Newton step on our own whitened residual vector (central differences, step 1e-7) | Mathematics | 1e-8 σ |
| `linear_flux_grid` with a Gaussian prior on f: log Bayes factor (Gaussian-prior evidence, not the default; virgil's defaults are Jeffreys priors) | quadrature over f of the linearised likelihood times the prior | Mathematics | 1e-7 |
| the same, no companion | quadrature of the true likelihood | Mathematics | 2e-3 (with a strong companion the closed form differs by ~0.4 in log B ~ 100–400, the nonlinearity its docs warn of) |
| `linear_flux_grid(n_iter=5)` with a bright companion (0.3) | a direct optimiser of our own χ² | Mathematics | 1e-6 |

| `OIData.with_closure_offsets`: per-baseline (T e), per-triangle and supplied offsets common to a frame's channels, four UTs | the sines over σ, projected on an orthonormal basis of each channel's triangle column space, against 𝒩(y; 0, QᵀRQ + VVᵀ) | Mathematics | 1e-13 |
| `RVData.marginal_loglike` and `zero_point_posterior` (three instruments, with and without jitter), given virgil's Keplerian model | 𝒩(m + Aμ, C + AΛAᵀ); the Gaussian conditional of the zero points | Mathematics | 1e-11; 1e-13 |

To come: the `OI_FLUX` grey scale (for a fixed prior: virgil's default
prior on k is centred on the data's own mean level, which uses the data
twice and is expected to change), the `VISPHI` continuum operator, and the
shared `LinearMarginal` helper.

