# Image metrics and ensembles

`virgil.metrics` scores a reconstruction against a known truth with the
figures of merit of the interferometric imaging contests;
`virgil.ensemble` draws, fits and averages randomized reconstructions after
Drevon et al. (2025). Tests:
[`test_metrics.py`](https://github.com/benjaminpope/virgil-validation/blob/main/tests/test_metrics.py) and
[`test_ensemble_draws.py`](https://github.com/benjaminpope/virgil-validation/blob/main/tests/test_ensemble_draws.py).

## Metrics

Our references are in `crosscheck.image_metrics`, written in NumPy from the
published definitions (not from virgil's code): Pearson's r for the NCC; the
2024 contest's L1 score minimized over the flux scale by brute force over
every breakpoint of its piecewise-linear objective; Lawson et al. (2004),
Eq. 2; the unit-sum rms difference of the 2008–2012 contests; exact
separable overlap rebinning. They are checked first against SciPy (bounded
minimization, `convolve2d`) and against identities.

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `ncc`, a noisy, rescaled copy of a lopsided scene | Pearson's r of the pixels; 1 for a positive affine copy | Mathematics | 1e-14 |
| `l1_score`, with and without negative pixels | 1 − min_a Σ\|a e − r\| / Σ r, negative pixels set to zero; 1 for a rescaled copy, 0 for an empty image | Mathematics | 1e-16 |
| `lawson_sigma_over_peak` | σ² = Σ r (e − r)² / Σ r over the unit-sum images, over the peak (Lawson et al. 2004, Eq. 2) | Literature + Mathematics | 3e-15 relative |
| `rms_convolved`, no beam and `relative=True` | rms of the unit-sum difference, and over the peak of the truth | Mathematics | 2e-15 relative |
| `rms_convolved` with a `Beam` (FWHM 3 × 2 pixels, PA 30°) | both images convolved with the pixel-sampled Gaussian, zeros outside the field; the beam turned by 90° is far off (control) | Mathematics | 4e-15 relative |
| `resample` to coarser, finer and cropped grids | exact overlap rebinning, pixels uniform; flux conserved when the new field covers the old | Mathematics | 4e-16 of the total flux |
| `align`, a scene moved 1 pixel South and 2 West | the shift back, (dra, ddec) = (+2, +1) pixels, East and North positive | Mathematics | 2e-14 mas |
| `score` | the same numbers; with `v2_only` the 180° turned scene is kept (NCC 1), without it is not | Mathematics | 1e-12 |

## Ensembles

Only the random settings are checked so far: `combine` (the MYTHRA
selection and the running mean), `run_group`, `ensemble`,
`reference_starts` and the `Group`, `Member` and `Ensemble` records need
fits and are not yet covered. On a simulated 7-hole masking dataset (21 V²
and 35 closure phases, 15 of them independent), 300 groups:

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `draw_groups`, `Draw`, `EnsembleSpec` | each group's family, start, pixel (`nyquist_pixel_scale` / oversample) and field (`field_of_view` × factor) from the spec; `n_weights` weights, largest first; families drawn uniformly (χ² test) | Mathematics | fields within 0.61 pixels |
| the weights per independent datum | max(w) / high ≤ N ≤ min(w) / low over all draws pins N = 36 (V² plus independent closure phases); log(w / N) uniform on the range (Kolmogorov–Smirnov) | Statistics | N bounds 36 to 36; smallest KS p = 0.03 |
| reproducibility and ordering | the same key gives the same draws; groups sharing a compilation (family, npix, pixel) are adjacent | Mathematics | exact |
