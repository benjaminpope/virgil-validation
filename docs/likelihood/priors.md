# Priors, angles and the MAP

`tests/test_priors_angles.py`. virgil's default priors are the Jeffreys
(invariant) measures, and `fit` optimises each in the coordinate where it
is flat ("Priors and the MAP" in virgil's conventions).

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `IsotropicInclination` (density ∝ sin i, degrees), full, half, narrow and polar ranges | closed forms written without cancellation (cos a − cos b = 2 sin((a+b)/2) sin((b−a)/2)); SciPy quadrature of the density and mean; the quantile function inverts the CDF; Kolmogorov–Smirnov on 20 000 samples | Mathematics + Statistics | 1e-9 (density), 1e-8 (CDF, norm, mean) |
| `IsotropicLatitude` (∝ cos lat, radians) | the same | Mathematics + Statistics | 1e-8 |
| their flat coordinates | the CDF is affine in cos i and in sin lat | Mathematics | 1e-12 |
| `AngleVector`: uniform, von Mises and axial, two ring widths | normalised over the plane; the angle's marginal is the stated density (SciPy's von Mises); half the squared residuals is −log p up to a constant; samples' angles and radii (KS) | Mathematics + Statistics | 1e-6 (quadrature), 1e-10 |
| `fit` with `LogUniform` priors on a diameter and a flux | the likelihood maximum by direct (Nelder–Mead) search on the same residuals | Mathematics | MAP within 1e-3 σ of it |
| control: a tight `Normal` prior | — | Mathematics | moves the MAP, as it should (no flat coordinate) |

