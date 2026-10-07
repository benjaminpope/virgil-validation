# Limb-darkened disks

virgil's limb-darkened disks are compared with point clouds of each law's
brightness, with Hanbury Brown et al.'s (1974) closed form for the linear law,
and with Kipping's (2013) parametrization. The baselines are 300 random VLTI
samples and a dense radial cut to 130 m at 1.5 µm, which passes every null of
a 12 mas disk. Test: [`test_limb_darkening.py`](https://github.com/benjaminpope/virgil-validation/blob/main/tests/test_limb_darkening.py);
issue [#12](https://github.com/benjaminpope/virgil-validation/issues/12).

## Polynomial laws

| virgil | Reference | Agreement |
| --- | --- | --- |
| `LimbDarkenedDisk`: uniform, linear, quadratic, cubic and order 22; 0.5, 3 and 12 mas; offset | point cloud | 2e-14; 7e-12 at order 22 |
| `cvis_limb_darkened_disk`, powers (0, 1, ½), (0, ½, 1.5, 3.7) and (¼, 22) | cloud of Σ aᵥ μᵛ | 3e-14 |
| the same in float32 (quadratic, square-root) | point cloud | 2e-7 |

At order 22 the error grows because expanding (1 − μ)²² into powers of μ
cancels binomial coefficients as large as 7e5. Powers outside (−2, 22] are
refused, as documented.

## Elliptical disks

| virgil | Reference | Agreement |
| --- | --- | --- |
| `EllipticalLimbDarkenedDisk`, uniform, linear and quadratic laws, axis ratio 0.7, PA 35°, offset | our circular point cloud squashed across the major axis in the image plane (no similarity theorem on our side); the major axis at PA + 90° is far off (control) | 1e-14 |
| its major axis | oriented as an `EllipticalGaussian`'s: baselines along it see lower visibilities than across it, for both | same ordering |

## Kipping's parametrization

| virgil | Reference | Agreement |
| --- | --- | --- |
| `QuadraticLimbDarkenedDisk`, `SquareRootLimbDarkenedDisk`: four (q₁, q₂) each, 12 mas, offset | cloud of the law given by Kipping's maps | 2e-14 |
| `u1`, `u2`, `c`, `d`, `from_u` and `from_cd` on a 9 × 9 grid of (q₁, q₂) | Kipping's eqs. 15–18 and 23–24 | 1e-14 |
| inside the unit square, the coefficients meet Kipping's eqs. 8 and 20–22 and `is_physical` is true; at (1.2, 0.5), (0.5, −0.05), (0.5, 1.05) and (1.02, 0.02) both fail | Kipping's constraints | exact |
| `LimbDarkenedDisk.is_physical`: false where the profile goes negative, limb brightening allowed | the profile at 10⁴ values of μ | exact for u = 0.6, −0.3, (0.35, 0.25), 2, 3 and (0.5, 0.6) |

## Images and data

| virgil | Reference | Agreement |
| --- | --- | --- |
| `render` of an off-centre quadratic disk | the profile at the pixel centres, East left, North up | 3e-19 |
| off-centre star and companion, as an 8×-oversampled pixel image written to OIFITS | virgil's analytic model on that file | 7e-5 in V², 4e-3 rad in closure phase (pixelization) |
| the same scene through our OIFITS file, `OIData` and `data.model` | our visibilities | 1e-12 |
| noise-free recovery of the diameter, q₁, q₂ and the companion | the truth | 7e-10 |

There is no noisy-pulls test: at its noise level q₁ and q₂ are so weakly
constrained that their Laplace posterior is not Gaussian, and pulls would
test that approximation rather than virgil.

## Our references

`crosscheck/limb.py` samples the disk by Gauss–Legendre quadrature in
t = √μ, in which every power of μ, √μ included, is a polynomial, and by the
trapezoid rule in azimuth. Its point clouds agree with an adaptive Hankel
transform to 1e-14 for every law above, and with Hanbury Brown et al.'s
closed form and the Airy pattern to 2e-14.
