# Cones, fields and invariants

The truncated cone is compared with a direct Fourier sum over its documented
3-D geometry, and the Gaussian-process field with a dense covariance matrix
built from the reflecting-boundary Laplacian. A last set of checks holds every
analytic component to the identities any visibility must satisfy. Test:
[`test_cone_field.py`](https://github.com/benjaminpope/virgil-validation/blob/main/tests/test_cone_field.py).

## Truncated cone

| virgil | Reference | Agreement |
| --- | --- | --- |
| `TruncatedCone`: side-on; wide, with negative tilt and a stretched cross-section; seen down its axis; in the sky plane; offset | a Fourier sum over a 3-D cloud on the cone's walls (Gauss–Legendre in slant distance, trapezoid in azimuth), projected onto the sky and blurred by the shell's Gaussian | 2e-6 at 1024 rings |
| the sign of the tilt | ±tilt give the same visibilities (the cone is optically thin) | 1e-14 |

Doubling the rings from 64 to 128 cuts the error by 4.0, the second-order
convergence the docstring states.

## Gaussian-process fields

| virgil | Reference | Agreement |
| --- | --- | --- |
| `GaussianField`, `field_spectrum`: orders 1–3, isotropic and anisotropic | the field's covariance against our dense Π(κ² + L)^(−order)Π, with L the reflecting-boundary Laplacian built from second differences, scaled to a mean pixel variance σ² | 2e-14 |
| `GaussianField`, order 1 | ½\|z\|² is proportional to \|η\|²/ℓ² plus the squared neighbour differences over h² | 1e-10 |
| `GaussianField` with a template, inside an `Image` | SciPy's orthonormal inverse DCT-II plus log(μ/max μ + ε); the image is the softmax of the field | 1e-12 |

## Invariants of every model

| virgil | Reference | Agreement |
| --- | --- | --- |
| `SourceModel` and `System`: six analytic components | V(0) = 1, V(−u, −v) = V(u, v)*, the shift theorem, and `System`'s flux-weighted mixing | 1e-12 |
