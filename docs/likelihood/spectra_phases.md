# Grey-scaled spectra and differential phases

`tests/test_flux_visphi.py`, on a four-telescope file with a Brγ-like line
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

