# Limb-darkened disks

Issue [#12](https://github.com/benjaminpope/virgil-validation/issues/12)
(`tests/test_limb_darkening.py`). Our references are in `crosscheck/limb.py`:
a point cloud by Gauss–Legendre quadrature in t = √μ (r dr = 2R²t³ dt, so
every power of μ, √μ included, is a polynomial in t) with the trapezoid rule
in azimuth; the Hankel transform by adaptive quadrature; Hanbury Brown et
al.'s (1974) closed form for the linear law; and Kipping's (2013) maps and
constraints. 300 random VLTI baselines plus a dense radial cut to 130 m at
1.5 µm, which samples every null of a 12 mas disk.

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| our cloud vs our Hankel transform (self-check; uniform, linear, quadratic, cubic, order 22, square root; 3 and 12 mas) | adaptive quadrature | Quadrature | 1e-14 |
| our cloud vs Hanbury Brown et al. (1974), and the uniform limit vs Airy (self-check) | closed forms with SciPy | Analytic + Literature | 2e-14 |
| `LimbDarkenedDisk` (uniform, linear, quadratic, cubic, order 22; 0.5, 3, 12 mas; offset) | point cloud | Quadrature | 2e-14; order 22: 7e-12 (expanding (1 − μ)²² into powers of μ cancels binomial coefficients up to 7e5) |
| `QuadraticLimbDarkenedDisk`, `SquareRootLimbDarkenedDisk` (four (q₁, q₂) each, 12 mas, offset) | cloud of the law given by Kipping's maps | Quadrature + Literature | 2e-14 |
| `cvis_limb_darkened_disk` with powers (0, 1, ½), (0, ½, 1.5, 3.7), (¼, 22) | cloud of Σ aᵥ μᵛ | Quadrature | 3e-14; powers outside (−2, 22] are refused, as documented |
| the same in float32 (quadratic, square root) | same | Quadrature | 2e-7 |
| `u1`, `u2`, `c`, `d` and `from_u`, `from_cd` over a 9 × 9 grid of (q₁, q₂) | Kipping eqs. 15–18, 23–24 | Literature | 1e-14 |
| unit square ↔ physical: inside, the coefficients meet Kipping's eqs. 8 and 20–22 and `is_physical` is true; at (1.2, 0.5), (0.5, −0.05), (0.5, 1.05), (1.02, 0.02) both fail | Kipping's constraints | Literature | exact |
| `LimbDarkenedDisk.is_physical` (documented: false where the profile goes negative; limb brightening allowed) | the profile on 10⁴ values of μ | Analytic | exact for u = 0.6, −0.3, (0.35, 0.25), 2, 3, (0.5, 0.6) |
| `render` of an off-centre quadratic disk | the profile at pixel centres, East left, North up | Standards | 3e-19 |
| off-centre star + companion as an 8×-oversampled pixel image, through our OIFITS writer | virgil's analytic model on the file | Quadrature | 7e-5 in V², 4e-3 rad in closure phase (pixelisation) |
| off-centre star + companion, OIFITS end to end | our file → `OIData` → `data.model` | Standards | 1e-12 |
| noise-free recovery of diameter, q₁, q₂ and the companion | the truth | Mathematics | 7e-10 relative |

Not in the noisy-pulls test: q₁ and q₂ are weakly constrained at its noise
level, so their Laplace posterior is not Gaussian and the pulls would test
the approximation rather than virgil.

