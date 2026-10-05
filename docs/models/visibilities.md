# Geometric visibilities

## Model visibilities

500 random baselines up to 130 m at 1.5–2.4 µm (`tests/test_visibilities.py`).

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| our disk quadrature vs Airy (self-check) | `2 J1(x)/x` | Analytic | 3e-15 |
| our Gaussian quadrature vs closed form (self-check) | Gaussian transform | Analytic | 5e-16 |
| `PointSource` at an offset | shift theorem, OIFITS sign | Analytic | 4e-15 |
| `GaussianDisk` (σ = 0.3, 1.2, 6 mas) | Gaussian transform | Analytic | 3e-16 |
| `EllipticalGaussian` (PA 0, 35, 123, 300°) | anisotropic Gaussian transform | Analytic | 5e-16 |
| `UniformDisk` (0.5, 3, 12 mas; through the 4th null) | `2 J1(x)/x` with SciPy | Analytic | 4e-16 |
| `BinaryModelAngular`, `BinaryModelCartesian` | sum of shifted points | Analytic | 4e-16 |
| `ModulatedGaussianRim` (inc 0, 50, 80°; m = 0, 1, 2) | thin inclined ring by trapezoid quadrature × the Fourier transform of a Gaussian isotropic in the rim plane (an elliptical Gaussian of axis ratio cos inc on the sky) | Quadrature + Analytic | 1e-12, with the in-plane azimuth (finding 1) and the in-plane blur of virgil#139; the earlier blur, round on the sky, now differs (negative control) |
| `ModulatedGaussianRim`, on-sky reading of the azimuth | same | Quadrature | differs by 4e-2: confirms the convention (finding 1) |
| `GaussianArc` (R, L) = (5, 4), (15, 20), (5, 40) mas | full-circle arc-length quadrature × Gaussian blur | Quadrature + Analytic | 8e-4 before the fix for finding 2; 1e-5 required after |
| `Image` (random 7 × 10, off-centre) | direct transform of pixel centres, row 0 North, column 0 East | Analytic | 3e-16 |
| `Rotated` (0, 70, 200°) | rotated point cloud | Quadrature | 1e-15 |
| `Resolved` flux in a `System` | normalisation | Analytic | 1e-15 |
| 5 random constellations: 24 points, Gaussians, elliptical Gaussians and disks at random positions and fluxes, plus a nested, shifted, weighted `System` | sums of closed forms | Analytic | 1e-12 |
| the same in float32 (virgil's default precision) | same | Analytic | 3e-5 |

## Derivatives

`tests/test_gradients.py`, float64. Every fit, Laplace covariance and
posterior depends on JAX's derivatives of virgil's models and likelihood;
virgil's own tests only check that they are finite.

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| visibilities of `PointSource`, `GaussianDisk`, `EllipticalGaussian`, `UniformDisk`, `ModulatedGaussianRim` (array `az_amps`, `az_pas`), `GaussianArc`, both binaries, a `System` with `Resolved`, `Rotated`, `GravityDarkenedStar`, the three flared disks, with respect to every geometric parameter (`HarmonixModel` not: its derivatives are harmonix's) | central finite differences (`jax.test_util.check_grads`, forward and reverse mode) | Mathematics | within check_grads' default tolerance (1e-5) |
| `Image` visibilities with respect to every log-brightness pixel (softmax-coupled) | same | Mathematics | same |
| `whitened_residuals` and `model_loglike` on noisy V² and closure phases (correlated whitening included) | same | Mathematics | same |
| Hessian of `model_loglike` (used by Laplace covariances and Fisher matrices) | Richardson-extrapolated central difference of the gradient | Mathematics | < 1e-7 relative to the diagonal |

