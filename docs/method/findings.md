# Findings

| # | virgil | Finding | Status |
| --- | --- | --- | --- |
| 1 | `ModulatedGaussianRim` | The modulation azimuth of an inclined rim is the in-plane one, not on-sky position angle as the docstring formula read. | fixed, [virgil#134](https://github.com/benjaminpope/virgil/pull/134) |
| 2 | `GaussianArc` | Arc-length weight cut at ±3.5 σ (~1e-3 visibility error); wrapping undocumented for long arcs. | fixed, [virgil#134](https://github.com/benjaminpope/virgil/pull/134) |
| 3 | `OIData.uv_grid` | Set only for AMIGO DISCO records, contrary to its docstring. | fixed, [virgil#134](https://github.com/benjaminpope/virgil/pull/134) |
| 4 | `Image.from_model` | The default brightness floor adds ~1e-5 flux on large fields (documented). | note |
| 5 | `inference.laplace_cov` | Fails when any parameter path is array-valued. | fixed, [virgil#135](https://github.com/benjaminpope/virgil/pull/135) |
| 6 | `fitting.fit` | A fit started at an exact zero-residual optimum reports non-convergence. | fixed, [virgil#144](https://github.com/benjaminpope/virgil/pull/144) |
| 7 | `fitting.fit` | `.expand()`ed or `.to_event()` priors silently switch the default optimiser from LM to L-BFGS. | fixed, [virgil#142](https://github.com/benjaminpope/virgil/pull/142) |
| 8 | `oidata.OIData` | With every closure phase flagged it crashed inside the closure-phase whitening (`ValueError: zero-size array`). | fixed, [virgil#155](https://github.com/benjaminpope/virgil/pull/155); now visibility-only data, [virgil#158](https://github.com/benjaminpope/virgil/pull/158) |
| 9 | `spectra.reference_flux` | For `Tabulated` it returns every node, not the reference flux the class documents (the node mean, which `spectrum(None)` returns). Harmless inside virgil, which uses it to check that fluxes are non-negative. | fixed, [virgil#163](https://github.com/benjaminpope/virgil/pull/163) |

## Definition changes in virgil

| virgil PR | Change | Our response |
| --- | --- | --- |
| [#139](https://github.com/benjaminpope/virgil/pull/139) (T. De Prins) | `ModulatedGaussianRim` blurs the rim with a Gaussian isotropic in the rim's own plane, not on the sky; `fwhm` is now the in-plane FWHM | references follow the documented definition (`sky.in_plane_blur_factor`); the old definition is kept as a negative control; confirmed independently by PMOIRED's blurred-ring profile (1e-9) |

