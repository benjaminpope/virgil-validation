# Geometric visibilities

Every analytic model is compared with the Fourier transform of its documented
brightness, evaluated by our own code on 500 random baselines up to 130 m at
1.5–2.4 µm. Where a closed form exists we use it; otherwise we sum a point
cloud drawn from the brightness by quadrature. Tests:
[`test_visibilities.py`](https://github.com/benjaminpope/virgil-validation/blob/main/tests/test_visibilities.py) and
[`test_gradients.py`](https://github.com/benjaminpope/virgil-validation/blob/main/tests/test_gradients.py).

## Points, Gaussians and disks

| virgil | Reference | Agreement |
| --- | --- | --- |
| `PointSource`, offset | shift theorem, OIFITS sign | 4e-15 |
| `GaussianDisk`, σ = 0.3, 1.2 and 6 mas | Gaussian transform | 3e-16 |
| `EllipticalGaussian`, PA 0°, 35°, 123° and 300° | anisotropic Gaussian transform | 5e-16 |
| `UniformDisk`, 0.5, 3 and 12 mas, through the fourth null | 2 J₁(x)/x (SciPy) | 4e-16 |

## Rings and arcs

| virgil | Reference | Agreement |
| --- | --- | --- |
| `ModulatedGaussianRim`, inclinations 0°, 50° and 80°, m = 0, 1 and 2 | thin inclined ring by trapezoid quadrature, times a Gaussian isotropic in the rim's plane | 1e-12 |
| `GaussianArc`, (R, L) = (5, 4), (15, 20) and (5, 40) mas | arc-length quadrature round the full circle, times the Gaussian blur | 1e-5 |

The rim's modulation azimuth is measured in the rim's plane, not on the sky
([F1](../index.md#F1)): reading it on the sky gives visibilities 4e-2 away. Its blur
has been isotropic in the rim's plane since
[virgil#139](https://github.com/benjaminpope/virgil/pull/139); the earlier blur,
round on the sky, is kept as a negative control. The arc agreed only to
8e-4 before the fix for [F2](../index.md#F2).

## Binaries and composite scenes

| virgil | Reference | Agreement |
| --- | --- | --- |
| `BinaryModelAngular`, `BinaryModelCartesian` | sum of shifted points | 4e-16 |
| `Rotated`, 0°, 70° and 200° | rotated point cloud | 1e-15 |
| `Resolved` flux in a `System` | normalization | 1e-15 |
| `System`: five random scenes of 24 points, Gaussians, elliptical Gaussians and disks, plus a nested, shifted, weighted `System` | sum of closed forms | 1e-12 (float64), 3e-5 (float32, virgil's default) |

## Pixel images

| virgil | Reference | Agreement |
| --- | --- | --- |
| `Image`, random 7 × 10 pixels, off-centre | direct transform of the pixel centres (row 0 North, column 0 East) | 3e-16 |

## Derivatives

Fits, Laplace covariances and posteriors all use JAX's derivatives of these
models and of the likelihood; virgil's own tests check only that they are
finite. Here they are compared with central finite differences
(`jax.test_util.check_grads`, forward and reverse mode, float64).

| virgil | Agreement |
| --- | --- |
| visibilities of every geometric model with respect to every geometric parameter: the models above, `GravityDarkenedStar` and the three flared disks | within 1e-5 |
| `Image` visibilities with respect to every log-brightness pixel | within 1e-5 |
| `whitened_residuals` and `model_loglike` on noisy V² and correlated closure phases | within 1e-5 |
| Hessian of `model_loglike`, against a Richardson-extrapolated difference of the gradient | 1e-7 of the diagonal |

`HarmonixModel` is not included: its derivatives are harmonix's.

## Our references

Before they judge virgil, our quadrature routes are checked against closed
forms: the disk cloud against 2 J₁(x)/x to 3e-15, the Gaussian cloud against
its transform to 5e-16.
