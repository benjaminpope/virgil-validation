# Imaging and grid machinery

This page checks the imaging grids, beams and transforms against analytic results, then the regularisers against eht-imaging and MPoL, CLEAN against simulated scenes, and the L-curve against our own recomputation.

`tests/test_grids.py` covers the grids and transforms.

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
outer ring of pixels is zero, so the three codes' different edge
conventions give the same sums. The mapping (signs, flips, the softening's
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
| `StarletL1` and `starlet` (1, 3, 4 scales) | à-trous B3-spline transform written with SciPy (`mode="constant", cval=0.0`, Starck, Murtagh & Fadili 2010): exact `details.sum(0) + coarse = image`, linearity, and dyadic scale separation (`test_042_scenes_starlet_kkt.py`) | Mathematics | 1e-14; details + coarse = image |
| `LogSum` | its formula and pixel gradient | Mathematics | 1e-15 |

## Synthetic scenes and field sizing

`tests/test_042_scenes_starlet_kkt.py` and `tests/test_042_lcurve_ensemble_radio.py` check the synthetic test scenes (`virgil.scenes`), sky-coordinate centroids and priors, and short-baseline field-sizing rules against `crosscheck.starlet_kkt_ref`. Notably, `scenes.spiral` parameterizes arm cross-sectional width by Gaussian standard deviation `width_mas` (not FWHM), winds counter-clockwise from North through East (`pa_deg`), terminates each arm end in a circular Euclidean cap, and applies an optional radial exponential envelope `exp(-r / fade_mas)`.

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `scenes.gaussian_blob` | unit-sum circular Gaussian at `(dra, ddec)` (East left, North up); wrong East-sign control fails | Mathematics | 1e-14 |
| `scenes.ring` | inclined deprojected Gaussian ring (`cos i` foreshortening, major axis at `pa_deg`) times `1 + asymmetry cos(PA - asymmetry_pa_deg)`; wrong modulation-PA control fails | Mathematics | 1e-14 |
| `scenes.spiral` | face-on Archimedean locus `r = step_mas θ / (2π)` with radial cross-section `σ = width_mas`, rounded caps, and `exp(-r / fade_mas)`; FWHM-instead-of-σ and CW-winding controls fail | Mathematics | 1e-14; fade ratio 1e-13 |
| `imaging.Centroid` on `Image` and `LinearImage` (`rotation_deg` = 0°, +32°, `"grid"`, −22°) | flux-weighted sky `(dra, ddec)` and Gaussian penalty `½((c_x/σ)² + (c_y/σ)²)`; flipped rotation-sign control fails | Mathematics | 1e-13 |
| `imaging.image_priors` | `Normal(0, 1)` on `GaussianField.latent`, improper flat prior (`log_prob = 0`) on plain `Image.log_brightness` and `LinearImage.pixels` | Mathematics | 1e-13 |
| `imaging.second_moment_fwhm` | weighted least squares of `−ln V² = π² θ² ρ² / (2 ln 2)` with weights `V² / σ_{V²}` above `v2_min`; 0 for unresolved, `None` for `< 2` samples or DISCO data | Mathematics | 1e-11 mas |
| `imaging.image_field` and `imaging.Beam` | `field_fwhms × FWHM` clipped to `[inner_elements × λ/(2 B_max), min(λ/B_min, largest_mas, aperture_mas)]` (ceiling wins on crossover); `convolve_beam` recovers `Beam` second moments | Mathematics | 1e-9 mas |
| `imaging.envelope_fwhm` and `imaging.starting_image` | joint point-plus-Gaussian fit recovers undiluted envelope FWHM where raw `second_moment_fwhm` is biased low by stellar dilution (control); `moments` and `dirty` starts, central `hole_mas` mask, and `pad_to_mas` | Mathematics | FWHM within 8 % |

## CLEAN

`imaging.clean`, virgil's gradient CLEAN, must recover the scenes it is given
and keep its χ² finite.

Recovery: [`test_clean_scenes.py`](https://github.com/benjaminpope/virgil-validation/blob/main/tests/test_clean_scenes.py)
runs on every pull request. Two scenes from our simulator: four UTs, seven
snapshots, two channels, little noise. One is a point at the phase centre;
the other adds a companion at 30 % of its flux, on a pixel centre. CLEAN
runs without a base scene on a 31 × 31 grid of 0.4 mas.

The residual dirty image is ours, not virgil's. We take the complex
visibilities of the truth minus those of the components (a direct sum over
pixel centres) at every sampled (u, v, λ). A NumPy DFT turns them back into
an image, divided by the number of samples, so the dirty beam peaks at 1.
[`test_clean_recovery.py`](https://github.com/benjaminpope/virgil-validation/blob/main/tests/test_clean_recovery.py)
(slow) adds a companion next to an analytic star used as the base.

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `imaging.clean`, point and binary, no base | flux within one pixel of each true position | Mathematics | 0.02 of the total (found 8e-5); centre within 0.5 pixel (found 0) |
| `imaging.clean`, point and binary, no base | peak of our residual dirty image | Mathematics | < 0.01 of the total (found 1e-4; the companion alone is 0.23) |
| control | the binary's components rotated by 180°, no components, or the point in a corner | Mathematics | residual peak > 0.1 |
| `imaging.clean`, companion beside a uniform-disk base (slow) | the companion's flux and position; nothing at its mirror image | Mathematics | 5 %; 0.5 pixel |
| `imaging.clean` without a base scene, three and four UTs | χ² finite at every iteration | Regression | finite |
| `CleanResult.restored` and `rotation_deg="grid"` (`Image`, `dirty_image`, `clean`, `residual_image`) | matches explicit `rotation_deg = data.uv_grid.rotation_deg` (falls back to 0 without a lattice); `restored(beam)` rotates beam PA into the pixel frame by `pa_deg - rotation_deg` (`test_042_scenes_starlet_kkt.py`) | Mathematics | 1e-13 |

The finiteness tests,
[`test_clean_finite.py`](https://github.com/benjaminpope/virgil-validation/blob/main/tests/test_clean_finite.py),
use four-UT files (correlated closure phases) and three-UT controls at 40,
41 and 60 pixels of 0.4 mas.

Without a base scene, `clean` seeds the central pixel, where the Jacobian
column |J e_p| is zero. On even grids that pixel sits half a pixel off the
origin, so rounding left |J e_p|² near 1e-24; its score won and its step was
infinite, giving a NaN χ² on every even size from 34 to 68 pixels
([F12](../index.md#F12)). virgil#190 treats such pixels as dead.

## L-curve

`imaging.l_curve` fits an Image for each weight of one regularizer and
reports χ², χ² per point and the unweighted penalty per weight.
[`test_l_curve.py`](https://github.com/benjaminpope/virgil-validation/blob/main/tests/test_l_curve.py)
checks this on an 8 × 8 Image with TSV at five weights, fitted to a
three-UT file of a Gaussian blob with a point beside it. Everything is
recomputed with our own code: χ² from our direct-sum visibilities of the
fitted pixels, and TSV from its formula.

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `LCurve.chi2`, `chi2_red`, `penalty` | our χ², χ²/N and Σ (Δx b)² + (Δy b)² of each fitted image | Mathematics | 1e-8 (found 3e-15) |
| each fitted image | stationary point of χ² + κ w TSV in the log-brightness, our finite-difference gradients | Mathematics | κ = 2 to 1 % (found 1.4e-3); gradient left < 1e-5 of the starting gradient (found 1e-7) |
| control | κ = 1, χ² + w TSV | Mathematics | gradient left > 1e-4 |
| the sweep | χ² does not fall and TSV does not rise as w grows (exact minimizers) | Mathematics | 1e-6 |
| `LCurve.discrepancy` | linear interpolation of χ²/N in log w | Mathematics | 1e-8 |
| `LCurve.corner` and `LCurve.classic_maxent` | maximum log–log curvature and Gull (1989) `2 w S(b) = γ(w)` root from the brightness-weighted Gauss–Newton spectrum (`test_042_lcurve_ensemble_radio.py`) | Mathematics | 1e-5 relative |
| `LCurve.dof_discrepancy` and `DofDiscrepancy` | Morozov discrepancy `m(w) = χ²/s² − (n − G)` over positive free `LinearImage` pixels (`parametrization="linear"` with bounded L-BFGS-B for `TSV`, `parametrization="sqrt"` for `MaxEntropy`) against exact dense `G = tr[K(K + H_R)⁻¹]` and exact Rademacher probe variance (Hutchinson 1989; Bekas et al. 2007), MaxEntropy `h(w) = 2(w/s²) KL − G`, custom `curvature=`, and over-regularized no-crossing control (`weight=None`, `accepted=False`) | Mathematics + Statistics | within Hutchinson SE |

`fit` documents regularizers as "penalties added to the loss" without saying
whether the loss is χ² or ½χ². The optimality test settles it: the loss is
½χ² + w TSV, the negative log likelihood plus the penalty.

## KKT optimality certificates

A flat loss trace does not by itself certify a constrained or reparameterized minimum. `virgil.fitting.kkt_report` (`tests/test_042_scenes_starlet_kkt.py`, checked against `crosscheck.starlet_kkt_ref` following Nocedal & Wright 2006, ch. 12 and §16.7) evaluates stationarity and complementary slackness directly in pixel-flux space:

- **Simplex (`Image`)**: on unit-sum brightness $b$, the Lagrange multiplier is $\mu = \sum_i b_i g_i$ with $g_i = \partial J/\partial b_i$; lit pixels ($b_i > 10^{-3}\max b$) must have $g_i - \mu = 0$ (`kkt_w` in units of $w/F$ below `KKT_W_TOL`), while dark support pixels ($b_i < 10^{-6}\max b$) with $g_i - \mu < 0$ flag `want_grow`.
- **Square-root (`LinearImage("sqrt")`)**: unconstrained in $x$ with pixel flux $f_i = x_i^2$, so $\mu = 0$ and `kkt_w` measures $\operatorname{rms}(g_{\text{lit}})/(w/F_0)$ where $F_0 = \sum_i f_i$; an exact $x_i = 0$ pixel is a stationary point of $x_i^2$ and is not flagged by `want_grow`.
- **Box-constrained (`LinearImage("linear")`)**: projected gradient $pg_i = \min(g_i, 0)$ at $f_i = 0$ and $g_i$ for $f_i > 0$; active bounds with $g_i \ge 0$ satisfy dual feasibility (`zero_violation == 0`), whereas clamping a positive peak pixel to zero yields $g_i < 0$ and fails `converged` (control).
- **Nuisance `start=` gate**: non-image parameters report `grad_rel` relative to the initial gradient only when `start=` is supplied; perturbing a fitted component flux while re-optimizing image pixels passes without `start=` and fails `grad_rel < NUISANCE_TOL` when `start=` is given.

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `fitting.kkt_report` on `LinearImage("sqrt")`, `LinearImage("linear")`, and softmax `Image` | independent flux-space gradients, simplex multiplier `μ = Σ b g`, projected box gradients, exact `x = 0` stationarity, and `MaxEntropy` `band_limit` (`crosscheck.starlet_kkt_ref`) | Mathematics | 1e-6 |
| `fitting.kkt_report` nuisance `start=` gate and non-optimal controls | 2-step truncated fit (`kkt_rel > KKT_TOL`), clamped-peak `zero_violation > 0`, and perturbed nuisance flux (`converged=False` iff `start=` passed) | Mathematics | exact gate |

## Laplace GP evidence and imaging diagnostics

For `Image(GaussianField(...))` models with standard-normal whitened DCT latents $z \sim \mathcal{N}(0, I)$, `virgil.imaging` provides MacKay (1992) Laplace evidence, posterior sampling, empirical-Bayes error scaling, and post-fit pitfall checks (`tests/test_042_lcurve_ensemble_radio.py`).

$$
\ln Z \approx \ln L(z_{\star}) - \tfrac{1}{2}\|z_{\star}\|_2^2 - \tfrac{1}{2}\ln\det(I + J^{\mathsf{T}}J), \qquad \gamma = \sum_k \frac{s_k^2}{1 + s_k^2},
$$

where $s_k$ are the singular values of the whitened-residual Jacobian $J = \partial r/\partial z\big|_{z_{\star}}$ and $s = \sqrt{\chi^2 / (N - \gamma)}$ estimates the global or per-observable noise scale factor.

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `imaging.log_evidence`, `laplace_samples`, `error_scale` | Cholesky MacKay (1992) `ln Z`, posterior covariance `(I + JᵀJ)⁻¹`, and `√(χ² / (N − γ))` (`crosscheck.starlet_kkt_ref`); 3× over-quoted noise bars yield `error_scale < 0.55` (control) | Mathematics + Statistics | `ln Z` to 1e-4; sample cov Frobenius < 0.15 |
| `imaging.chi2_by_observable` and `short_baseline_residual` | independent V² and rank-projected closure-phase counts, exact `γ`, shortest-quantile mean residual and standard error `1/√n_short` (missing extended envelope gives > 4σ positive bias) | Mathematics | `γ` to 1e-10; SE to 1e-12 |
| `imaging.diagnose` and `Diagnosis` | flags unanchored single-`Image` translations (`anchored=False`) and outer-2-pixel border flux (`edge_flux`), cleared when anchored by a star or `Centroid` prior | Mathematics | exact flags |
