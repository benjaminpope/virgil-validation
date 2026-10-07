# Flared disks and harmonix

The flared disks are compared with their documented brightness (Blakely et
al. 2024, eqs. 2–9: the scattering surface, the three phase functions and
the skewed Gaussian ring), evaluated by our own code at the documented pixel
centres and summed directly in Fourier space. Test:
[`test_remaining_models.py`](https://github.com/benjaminpope/virgil-validation/blob/main/tests/test_remaining_models.py).

## Flared disks

| virgil | Reference | Agreement |
| --- | --- | --- |
| `FlaredDiskHG`, `FlaredDiskGaussian`, `FlaredDiskPowerLaw`: face-on and flat, inclined, inclined at PA 200°, offset | direct Fourier sum of our brightness | 1e-15 |
| the default pixel sampling | the same disk at twice the density, on baselines that resolve the ring | 1e-9 |

The shared base class `FlaredDisk` gets its evidence through these three; on
its own it has no phase function, and evaluating it raises
`NotImplementedError` (a guard).

As a negative control, the same disk turned through 180° (near side at
PA + 270°) differs by more than 1e-2, as it should.

## The harmonix wrapper

| virgil | Reference | Agreement |
| --- | --- | --- |
| `HarmonixModel` | stand-in sources with closed-form visibilities, in cycles per radian and in metres; weight 1 in a `System` | 1e-12 |

Only the wrapper is checked here. harmonix's maps are harmonix's to validate,
against starry or a direct surface quadrature.
