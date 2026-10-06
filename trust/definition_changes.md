## Definition changes in virgil

| virgil PR | Change | Our response |
| --- | --- | --- |
| [#139](https://github.com/benjaminpope/virgil/pull/139) (T. De Prins) | `ModulatedGaussianRim` blurs the rim with a Gaussian isotropic in the rim's own plane, not on the sky; `fwhm` is now the in-plane FWHM | references follow the documented definition (`sky.in_plane_blur_factor`); the old definition is kept as a negative control; confirmed independently by PMOIRED's blurred-ring profile (1e-9) |

