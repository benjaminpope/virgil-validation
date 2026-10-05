# Imaging and grid machinery

`tests/test_grids.py`.

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `imaging.dirty_image` of an offset point, uv disk uniformly filled | Airy beam `2 J1(2πqr)/(2πqr)` centred on the source, East left, North up | Analytic | 3e-3 (sampling of the uv disk) |
| `imaging.beam`, filled uv disk | FWHM = 2√(2 ln 2)/(π q) | Analytic | 1e-4 |
| `imaging.beam`, filled uv ellipse (axis ratio 0.4, PA 0, 30, 100°) | FWHMs in inverse ratio; beam PA = coverage PA + 90° | Analytic | 1e-3; PA to 0.05° |
| `imaging.nyquist_pixel_scale` | λ / 2 B_max | Analytic | 1e-9 |
| `imaging.field_of_view` | min(λ / B_min, `largest_mas`) | Analytic | 1e-9 |
| `imaging.convolve_beam` of a delta (PA 0, 35, 120°) | unit sum; second moments of the beam Gaussian | Analytic | 2e-3 mas² |
| `SourceModel.render` of an offset point | lands on the expected pixel, East left, North up | Analytic | exact |
| `SourceModel.render` of a Gaussian, transformed back | Gaussian transform | Analytic | 1e-9 |
| `Image.from_model` → `Image.model` round trip | Gaussian transform | Analytic | 1e-9 (with the brightness floor off; finding 4) |
| `find_uv_grid` on rotated lattices (0, 17, −38°) | the lattice we built | Analytic | rotation to 1e-6° |
| `Image.model_on_grid` (two-sided matrix Fourier transform) and `Image.model` on those lattices | direct sum over rotated pixel centres | Quadrature | 1e-12 |

## Regularisers

`tests/test_imaging_regularisers.py`. eht-imaging 1.3.2 (`imager_utils`,
normalisation off) and MPoL 0.3.1 (`losses`) run in their own
environments. The test images are random, on a circular support, so their
outer ring of pixels is zero. With that, the three codes' different edge
conventions give the same sums; the mapping (signs, flips, the softening's
square, the cells each code counts) is in the test's docstring. Gradients
are virgil's, with respect to the log-brightness, against eht-imaging's
hand-written and MPoL's autograd pixel gradients through the softmax
chain rule.

| virgil | Reference | Tag | Agreement (value and gradient) |
| --- | --- | --- | --- |
| `TSV` | `stv2` (eht-imaging), `TSV` (MPoL) | eht-imaging, MPoL | 1e-15 |
| `TV` | `stv` and `TV_image` averaged over the four flips, plus the softening of the extra edge cells | eht-imaging, MPoL | 1e-15 |
| `MaxEntropy`, flat and random default image | `ssimple` (eht-imaging); `entropy` (MPoL, no support) | eht-imaging, MPoL | 1e-15 |
| `Laplacian` | SciPy's five-point Laplacian of the zero-padded image | Mathematics | 1e-15 |
| `StarletL1` and `starlet` (1, 3, 4 scales) | à-trous B3-spline transform written with SciPy (Starck, Murtagh & Fadili 2010) | Mathematics | 1e-15; details + coarse = image |
| `LogSum` | its formula and pixel gradient | Mathematics | 1e-15 |

## CLEAN

`imaging.clean`, virgil's gradient CLEAN, must keep its χ² finite. Test:
[`test_clean_finite.py`](https://github.com/benjaminpope/virgil-validation/blob/main/tests/test_clean_finite.py), on four-UT files
(correlated closure phases) and three-UT controls, at 40, 41 and 60 pixels
of 0.4 mas.

| virgil | Reference | Agreement |
| --- | --- | --- |
| `imaging.clean` without a base scene, three and four UTs | χ² finite at every iteration | finite |

Without a base scene, `clean` seeds the central pixel, where the Jacobian
column |J e_p| is zero. On even grids that pixel sits half a pixel off the
origin, so rounding left |J e_p|² near 1e-24; its score won and its step was
infinite, giving a NaN χ² on every even size from 34 to 68 pixels
([F12](../index.md#F12)). virgil#190 treats such pixels as dead.

