# Geometric visibilities

This page checks every analytic model against the Fourier transform of its
documented brightness, and then checks virgil's derivatives. Our own code
evaluates the transform on 500 random baselines up to 130 m at 1.5–2.4 µm,
using a closed form where one exists and otherwise a point cloud drawn from
the brightness by quadrature. Tests:
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
| `GaussianArc`, `pa = 37°` (`tests/test_arc_pmoired.py`) | `PMOIRED` native `ring` with von Mises Fourier–Bessel harmonics (`az ampN = 2 I_N(κ)/I_0(κ)`, `az projangN = pa`, `spatial kernel = width`); `pa + 90°`, `pa + 180°`, `−pa` controls fail by > 10° | < 1e-5 (V²), < 1e-3° (CP) |

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

## Closed-form helpers

[`test_models_spectra_extras.py`](https://github.com/benjaminpope/virgil-validation/blob/main/tests/test_models_spectra_extras.py) and
[`test_042_data_gains_angles.py`](https://github.com/benjaminpope/virgil-validation/blob/main/tests/test_042_data_gains_angles.py),
on random baselines given in wavelengths. For azimuthally modulated rings and
disks, the [Jacobi–Anger expansion](https://en.wikipedia.org/wiki/Jacobi%E2%80%93Anger_expansion)
guarantees that each azimuthal Fourier harmonic of order $m$ on a thin ring of
radius $r$ transforms into $(-i)^m J_m(2\pi q r)$ along the deprojected uv
azimuth.

| virgil | Reference | Agreement |
| --- | --- | --- |
| `cvis_uniform_disk`, offset | 2 J₁(x)/x, x = πθ\|u\|, times the shift phase; zero at the first root of J₁ | 4e-16 |
| `cvis_binary` | (1 + f e^(−2πi(u·dra + v·ddec))) / (1 + f), and the sum of two shifted points | 9e-16 |
| `multipole_disk_cvis` (inclined, blurred, conical ring `height`, negative `stretch`) | independent SciPy Bessel series with uv deprojection, conical height phase shift, and disk-plane Gaussian envelope | `< 1e-13` |
| `cvis_radial_dirac_delta_modulated` | SciPy Bessel series and 4096-node azimuthal ring quadrature (North = 0°, East = 90°); $+i^m$ conjugation control fails by `> 0.2` | `< 1e-14` (Bessel), `< 1e-12` (cloud) |

## Bandwidth smearing (`smeared_model`)

Across a finite spectral channel of width $\Delta\lambda$ centred on $\lambda_0$,
off-axis fringes precess with wavenumber $\sigma = 1 / \lambda$ and wash out at
separations beyond the coherence envelope $\lambda_0^2 / (B\,\Delta\lambda)$.
`virgil.models.smeared_model` (and `OIData(..., bandwidth=True)` / `with_bandwidth`)
accumulates sky offsets down `System` and `Rotated` trees, evaluates each leaf's
intrinsic shape at $\lambda_0$, and multiplies its offset phasor by the exact
integral over a uniform boxcar of width $\Delta\sigma = \Delta\lambda / \lambda_0^2$
in wavenumber (`tests/test_042_spiral_and_smearing.py`):

$$
V_{\text{smeared}}(u, v, \lambda_0, \Delta\lambda) = \sum_k w_k\,V_k^{\text{centred}}(u, v, \lambda_0)\,e^{-2\pi i \tau_k / \lambda_0}\,\operatorname{sinc}\!\left(\frac{\Delta\lambda\,\tau_k}{\lambda_0^2}\right), \qquad \tau_k = (u\,\mathrm{dra}_k + v\,\mathrm{ddec}_k)\,\text{mas}
$$

where $\operatorname{sinc}(x) = \sin(\pi x) / (\pi x)$. Because $\sigma = 1/\lambda$
is nonlinear in $\lambda$, a uniform wavenumber boxcar agrees with a uniform
wavelength boxcar to $O((\Delta\lambda / \lambda_0)^2)$ (`16.0×` error reduction
per `4×` bandwidth narrowing). External interferometry packages adopt two
distinct conventions, recorded as ledger entry [D23](../method/findings.md#D23):
PMOIRED's `'wl kernel'` convolves `OI_VIS2['V2']` *incoherently* ($\langle|V|^2\rangle$,
retaining an $f^2 / (1 + f)^2$ floor when fringes wash out), while CANDID's
default `Nsmear = 3` averages complex $V$ coherently with equal weights
$(1/3, 1/3, 1/3)$ at $\lambda_0 + (-1/2, 0, +1/2)\,\Delta\lambda$, which doubles
the spectral variance from $\Delta\lambda^2 / 12$ to $\Delta\lambda^2 / 6$.

| virgil | Reference | Agreement |
| --- | --- | --- |
| `smeared_model` on `BinaryModelCartesian`, `BinaryModelAngular`, nested `System` (with `UniformDisk` and `Resolved`), and `Rotated` | 64-node Gauss–Legendre quadrature over uniform wavenumber $\sigma = 1/\lambda$ (`crosscheck.spiral_smearing`) | `< 1e-14` |
| `smeared_model` vs uniform wavelength quadrature | 64-node Gauss–Legendre quadrature over $[\lambda_0 - \Delta\lambda/2, \lambda_0 + \Delta\lambda/2]$ at $\Delta\lambda/\lambda_0 = 0.008$ vs $0.002$ | $O((\Delta\lambda/\lambda_0)^2)$ ratio `16.0` |
| `OIData(..., bandwidth=True)`, `with_bandwidth`, `OrbitalBinary` via `at(mjd)`, and `TypeError` on `Image` / direct `OrbitalBinary` | `EFF_BAND` round-trip and per-sample `smeared_model` evaluation | `< 1e-7` (float32 `EFF_BAND`) |
| `smeared_model` vs PMOIRED (`roots=["pmoired"]`) | explicit `setupFit({'wl kernel': K}, auto=False)` with variance-matched FWHM $K = \sqrt{2\ln 2 / 3}\,(\Delta\lambda / \delta\lambda)$ at $f = 0.02$ | V² `< 8e-5`, CP `< 5e-3°` |
| `smeared_model` vs CANDID (`roots=["candid", "mathematics"]`) | CANDID `Nsmear = 3` with variance-matched `dwavel = dlambda / sqrt(2)` at $f = 0.15$ | $\chi^2_r = 0.04$ (unsmeared $\chi^2_r > 15$) |
| definition controls (`kind="control"`, [D23](../method/findings.md#D23)) | incoherent $\langle\|V\|^2\rangle$ floor at $f = 0.45$, and unscaled CANDID `Nsmear = 3` (`dwavel = dlambda`, $2\times$ variance) | both fail (`> 0.077` in V²; $\chi^2_r > 15$) |

## Pixel images

| virgil | Reference | Agreement |
| --- | --- | --- |
| `Image`, random 7 × 10 pixels, off-centre | direct transform of the pixel centres (row 0 North, column 0 East) | 3e-16 |
| `LinearImage`, 7 × 9 (`ft="dft"`) and 64 × 64 on 600 baselines (`ft="nufft"` and `ft="auto"`) in `"linear"` and `"sqrt"` (F = w²) parameterizations, masked, rotated, off-centre, and flux-weighted in a `System`, with `check_grads` pixel derivatives ([`test_standalone_nodes.py`](https://github.com/benjaminpope/virgil-validation/blob/main/tests/test_standalone_nodes.py)) | direct transform of the masked pixel centres, total-flux `System` weighting, NUFFT vs DFT agreement, and central finite differences; controls: linear vs squared weights and unit-weight `System` fail | 2e-15 (`dft`), < 1e-11 (`nufft`/`auto`), 1e-5 (grads) |

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

Before they judge virgil, we check our quadrature routes against closed
forms: the disk cloud against 2 J₁(x)/x to 3e-15, the Gaussian cloud against
its transform to 5e-16.
