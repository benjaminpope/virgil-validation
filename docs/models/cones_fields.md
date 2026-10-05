# Cones, Gaussian fields and model invariants

`tests/test_cone_field.py`.

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `models.TruncatedCone`, four geometries (side-on, wide with negative tilt and a stretched cross-section, seen down its axis, in the sky plane), with an offset | a direct Fourier sum over a 3-D cloud on the cone's walls, built from the documented geometry and projected onto the sky (Gauss–Legendre in slant distance, trapezoid in azimuth), blurred by the shell's Gaussian | Mathematics | 2e-6 in V at 1024 rings; doubling 64 → 128 rings cuts the error by 4.0, the documented second order |
| `TruncatedCone` tilt sign | ±tilt give the same visibilities (optically thin) | Mathematics | 1e-14 |
| `fields.GaussianField`, `field_spectrum`, orders 1–3, isotropic and anisotropic | the field's covariance (from its response to each latent) against our dense Π(κ² + L)^(−order)Π, with L the reflecting-boundary Laplacian built from second differences, scaled to a pixel-averaged variance σ² | Mathematics | 2e-14 |
| `GaussianField`, order 1 | ½\|z\|² is proportional to \|η\|²/ℓ² plus the squared neighbour differences over h² | Mathematics | 1e-10 |
| `GaussianField` with a template, inside an `Image` | SciPy's orthonormal inverse DCT-II plus log(μ/max μ + ε); the image is the softmax of the field | Mathematics | 1e-12 |
| `SourceModel` invariants and `System` mixing, six analytic components | V(0) = 1, V(−u, −v) = V(u, v)*, the shift theorem, and `System` mixing | Mathematics | 1e-12 |

