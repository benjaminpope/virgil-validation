# Image metrics and ensembles

`virgil.metrics` scores a reconstruction against a known truth with the
figures of merit of the interferometric imaging contests;
`virgil.ensemble` draws, fits and averages randomized reconstructions after
Drevon et al. (2026). Tests:
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
| `rms_convolved` with a `Beam` (FWHM 3 × 2 pixels, PA 30°) | both images convolved with the pixel-sampled Gaussian, zeros outside the field; the mirrored beam (PA −30°) misses the 1e-10 tolerance by orders of magnitude (control) | Mathematics | 4e-15 relative |
| `resample` to coarser, finer and cropped grids | exact overlap rebinning, pixels uniform; flux conserved when the new field covers the old | Mathematics | 4e-16 of the total flux |
| `align`, a scene moved 1 pixel South and 2 West | the shift back, (dra, ddec) = (+2, +1) pixels, East and North positive | Mathematics | 2e-14 mas |
| `score` | the same numbers; with `v2_only` the 180° turned scene is kept (NCC 1), without it is not | Mathematics | 1e-12 |

## Ensembles

[`test_ensemble_draws.py`](https://github.com/benjaminpope/virgil-validation/blob/main/tests/test_ensemble_draws.py)
checks the randomized hyperparameter draws on a simulated 7-hole masking dataset
(21 V² and 35 closure phases, 15 of them independent, 300 groups), and
[`test_042_lcurve_ensemble_radio.py`](https://github.com/benjaminpope/virgil-validation/blob/main/tests/test_042_lcurve_ensemble_radio.py)
validates `reference_starts`, `run_group`, `GPFit`, `EvidenceGrid`,
`evidence_grid`, `gp_members`, `combine` (MYTHRA windowing, MAD outlier cut,
sub-beam alignment, and non-compounding running-mean χ² gate), `ensemble`, and
the `Group`, `Member`, and `Ensemble` containers on synthetic VLTI datasets:

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `draw_groups`, `Draw`, `EnsembleSpec` | each group's family, start, pixel (`nyquist_pixel_scale` / oversample) and field (`field_of_view` × factor) from the spec; `n_weights` weights, largest first; families and starts each drawn uniformly (χ² tests) | Mathematics | fields within 0.61 pixels |
| the weights per independent datum | max(w) / high ≤ N ≤ min(w) / low over all draws pins N = 36 (V² plus independent closure phases); log(w / N) uniform on the range (Kolmogorov–Smirnov) | Statistics | N bounds 36 to 36; smallest KS p = 0.03 |
| reproducibility and ordering | the same key gives the same draws; groups sharing a compilation (family, npix, pixel) are adjacent | Mathematics | exact |
| `Group`, `Member`, `Ensemble` | `Ensemble.kept` filtering, whole-sky normalized `mean.flux` and `mean.brightness`, pixelwise `ddof=0` standard deviation `Ensemble.std` across kept members, and `summary()` | Mathematics | 1e-13 |
| `reference_starts`, `run_group`, `combine`, `ensemble` | unit-sum `moments`, `flat`, and `dirty` starts (closure-phase datasets raise `ValueError` on `starts=("dirty",)`), direct `run_group` L-curve sweep, `star=False` sub-beam alignment (`aligned_std < 0.05 × unaligned_std`), full deterministic MYTHRA cascade (`"window"`, `"diverged"`, `"chi2"`, `"outlier"`), end-to-end multi-family ensemble with every `trace` step bounded by `trace[0] × (1 + √(2/N_ind))`, and 180°-reflected V²-only twin rejected with `reason="mean"` (control) | Mathematics | NCC > 0.80; cascade & gate exact |
| `GPFit`, `EvidenceGrid`, `evidence_grid`, `gp_members` | adaptive edge growth (`gp_growths=1`, `2×2 → 3×3`) and automatic `"aniso"` grid when fitted Gaussian `axis_ratio < gp_axis_ratio`; every `"iso"` and `"aniso"` `GPFit.log_evidence` matches independent joint `(z, ln flux)` Cholesky Laplace integration (`crosscheck.starlet_kkt_ref.mackay_laplace_gp_ref`), and `gp_members` retains only fits within `gp_window_nats` of the peak | Mathematics | 1e-9 nats |
