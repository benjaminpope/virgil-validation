# Spectra, flared disks and the harmonix wrapper

`tests/test_remaining_models.py`. The flared-disk reference is
`crosscheck.disks`, written from the brightness distribution in virgil's
documentation (Blakely et al. 2024, eqs. 2–9): the scattering surface,
the three phase functions and the skewed Gaussian ring, evaluated at the
documented pixel centres and summed directly in Fourier space.

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `spectra.PowerLaw` | (λ/λ₀)^index | Mathematics | 1e-12 relative |
| `spectra.BlackBody` | Planck ratio with SciPy's CODATA constants; Rayleigh–Jeans limit | Mathematics | 9e-9 relative (virgil carries hc/k to ~10 digits) |
| `spectra.Tabulated` | linear interpolation, constant beyond the end nodes | Mathematics | 1e-12; reference flux after finding 9 |
| chromatic `System` (black-body companion, power-law envelope) | flux-weighted mean of closed-form visibilities, channel by channel | Mathematics | 5e-11 |
| `FlaredDiskHG`, `FlaredDiskGaussian`, `FlaredDiskPowerLaw` (face-on and flat, inclined, inclined at PA 200°, offset) | direct Fourier sum of our brightness | Mathematics | 1e-15 |
| near side of the flared disks | the disk rotated by 180° (near side at PA + 270) | Mathematics (control) | differs by > 1e-2, as it should |
| default flared-disk sampling | the same disk at twice the pixel density, on baselines that resolve the ring | Mathematics | 1e-9 |
| `HarmonixModel` wrapper | stand-in sources with closed-form visibilities, in cycles per radian and in metres; weight 1 inside a `System` | Mathematics | 1e-12 |

The harmonix maps themselves are harmonix's to validate (against starry or
a direct surface quadrature); here only virgil's wrapper is checked.

