# ALMA IM Lup: a virgil-made CLEAN image for comparison with DSHARP's

Written 2026-10-22. Ben's request: "check that our CLEAN implementation produces a nice image to compare to DSHARP."
Code: `scripts/alma_imlup_clean.py` (virgil-facing: imaging, exact major cycles, FITS, comparison), `src/crosscheck/alma_clean.py`
(NumPy: multi-scale minor cycles, beam fit, statistics; imports nothing from virgil), `tests/test_alma_clean.py`. OzSTAR job:
`alma_imlup_clean` in ozstar_scripts. This note is the design and the proposal for a virgil API; **promotion to virgil is a later PR**.

## 1. Can `virgil.imaging.clean` take the IM Lup visibilities? No

**Type.** `clean(data, ...)` (virgil main, `src/virgil/imaging.py`) takes `OIData`: it reads `data.n_independent` and differentiates the
likelihood of closure, kernel-phase and V² observables. `ComplexVisData` has neither:

```
clean(ComplexVisData, 64, 5.0)  ->  AttributeError: 'ComplexVisData' object has no attribute 'n_independent'
```

**Cost, if the type were fixed.** Gradient CLEAN does a Jacobian-vector product per pixel for the atom norms ``|J e_p|`` (once, unless
`refresh_norms`) and one value-and-gradient of the whole likelihood per iteration. Measured here (laptop, 2 CPU threads, the 42,424-point
1/1000 subsample, virgil main, nufftax, f64): model evaluation 13 ms (32²), 35–39 ms (64²–256²); value+grad 63–74 ms, flat in the grid
because the points dominate. Both scale linearly in N. At N = 42,421,131 the A100 numbers of `alma_radio_bench` (job 18862592, f64,
eps 1e-6, 1028²) are **0.44–0.53 s per value+grad**; a forward model is about half of that.

| | per pixel / iteration | count | total at 4.24e7 points |
|---|---|---|---|
| atom norms, 1028² (MPoL's grid) | ≈ 0.25 s (one JVP) | 1.06e6 pixels | **≈ 2.6e5 s ≈ 3 days**, once |
| atom norms, DSHARP's 3 mas grid, 4000² (16e6 pixels) | ≈ 0.25 s | 1.6e7 | **≈ 4e6 s ≈ 46 days** |
| same with six scales (`scales_mas`) | × 6 | | 8 months |
| iterations (one gradient each) | ≈ 0.5 s | `tclean` needs ~10⁴–5·10⁴ components | **≈ 1.5–7 h** |

On the laptop subsample the same arithmetic is: norms for a 64² grid = 4096 × 35 ms ≈ 2.4 min for 1/1000 of the data, i.e. ≈ 40 h at full N on the same CPU.
The norms are the blocker, and for linear data they are known in closed form (every point source has the same ``Σ w`` response; a Gaussian of
scale s has ``Σ w |G_s|²``), so the generic machinery buys nothing here. **Verdict: not feasible at 4.24e7 × 1028² (3 days of norms) or at DSHARP's
grid (weeks), and it takes the wrong data type.** The data are linear in the image, so the right tool is the classic Cotton–Schwab loop, whose minor cycles
are in the image plane (ms per iteration) and whose major cycles need one model transform and one adjoint transform of the full data (0.5 s each at 1028²).
`virgil.clean` stays the tool for non-linear data (closure phases, V², DISCOs).

## 2. What DSHARP did (IM Lup)

Sources: Andrews et al. 2018, ApJL 869, L41 (DSHARP I), Sects. 4.2–4.3 and Table 4 (arXiv:1812.04040); the DSHARP release script
`IMLup_continuum.py` and its `reduction_utils.py` (`tclean_wrapper`), https://almascience.eso.org/almadata/lp/DSHARP/scripts/ (CASA 5.1.1–5);
the fiducial image https://almascience.eso.org/almadata/lp/DSHARP/images/IMLup_continuum.fits (a 4000 × 4000 float32 image: 64,010,880 bytes = 4000²·4 + header).

The fiducial image is the **last `tclean_wrapper` call of `IMLup_continuum.py`**, on the self-calibrated, shifted and flux-rescaled combined SB+LB measurement set
(`IMLup_continuum.ms`: the CORRECTED_DATA of the file we read; `design/alma_imlup_p3.md` D1):

| tclean parameter | value | source |
|---|---|---|
| `specmode`, `deconvolver`, `nterms` | `mfs`, `multiscale`, 1 (flat spectrum) | `tclean_wrapper`; paper 4.3 |
| `scales` | 0, 10, 50, 150, 300, 450 **pixels** = 0, 30, 150, 450, 900, 1350 mas | script, final call; paper: point-like, ~θb and 2–3 θb, then ×2–3 up to the mask radius (Gaussian) |
| `smallscalebias` | 0.6 (CASA's default) | `tclean_wrapper` |
| `weighting`, `robust` | `briggs`, 0.5 | `tclean_wrapper`; Table 4 col. 7 |
| `uvtaper` | 0.033″ × 0.026″, PA 137.53° | final call; Table 4 col. 8: 33 × 26 mas, PA 138° |
| `imsize`, `cell` | 4000 × 0.003″ (12″); the paper says 9″-wide images: the script wins (file size above) | script, final call |
| `gain`, `niter`, `cycleniter`, `cyclefactor` | 0.1, 50000, 300, 1 | final call and `tclean_wrapper` defaults |
| `threshold` | 0.05 mJy | final call (self-cal rounds used 0.06 mJy; paper: 2–3× the noise) |
| `mask` | ellipse centred 15h56m09.18880s −37°56′06.527374″, semi-axes 2.75″ × 1.75″, PA 145° | script: `common_mask` (it spells `mask_PA` where only `mask_pa = 145` is defined; 145 is the value used) |
| restoring beam | CASA default: Gaussian fitted to the PSF's main lobe (`psfcutoff` 0.35) | Table 4: 44 × 43 mas, PA 115° |
| image | Table 4 (IM Lup): rms 14 µJy/beam, peak 7.11 mJy/beam, 253 mJy inside the mask | paper |

Noise: `annulus[[mask centre], [1.1 × 2.75″, 5.75″]]` (script, `estimate_SNR`). Self-calibration (paper 4.2–4.3): phase solutions on {900, 360, 180, 60, 30} s with
the CLEAN model of each round (`savemodel='modelcolumn'`), then one amplitude round. **The visibilities therefore already carry DSHARP's own CLEAN model**: our image
and theirs are two CLEANs of the same self-calibrated data, and agreement is expected to be much closer than between independent reductions (it would also hide a
shared bias, which is what the MaxEnt comparison of `design/alma_imlup_p3.md` is for).

The taper explains the beam: the untapered combined beam is 33 × 26 mas (PA 47.5°) and the taper's Gaussian has the *same* FWHMs rotated by 90°,
so the convolution gives √(33² + 26²) = 42 mas in both directions: Table 4's 44 × 43.

## 3. What we did

`scripts/alma_imlup_clean.py run` repeats that recipe with virgil's public API:

* **Weights and PSF.** `ComplexVisData.dirty_image(weighting="briggs", robust=0.5, uv_cell=1/(4000·3 mas))`; CASA grids the Briggs density on
  1/(imsize·cell), so `uv_cell` is DSHARP's, not ours (virgil's default would use our own field). The taper is applied to the *visibilities*, `(V − M)·T`:
  with weights `w f` this is `w f T`, with the Briggs factor `f` computed without the taper, the order `tclean` uses. The PSF is the dirty image of `T`
  itself (tapered unit visibilities), and both are divided by its central value so that the PSF peaks at 1 and the images are in Jy/beam of the tapered PSF.
  The grid has an **odd** side (default 3001 × 3 mas = 9″): the phase centre is a pixel and the PSF peaks on it (virgil's even grids put the centre between four pixels).
* **Major cycles (Cotton & Schwab 1984; Schwab 1984, AJ 89, 1076).** Residual visibilities `V − M` where `M` comes from `ComplexVisData.model` evaluated in chunks of 10⁶ points
  (nufftax, exact to the NUFFT tolerance, no gridding of the data: **the model is never degridded**, unlike `tclean`'s gridded major cycle). virgil's `Image` is non-negative
  (`softmax`), and CLEAN models are signed: the positive and negative parts are two `Image`s (with `support`) subtracted. Checked against a direct DFT: 6e-14 on a random 21² signed image.
* **Minor cycles** (`crosscheck.alma_clean.multiscale_minor`, NumPy): multi-scale CLEAN of Cornwell (2008, IEEE JSTSP 2, 793), Gaussian scales, with the loop gain of Högbom (1974),
  CASA's small-scale bias `1 − 0.6 s/s_max` on the chi²-reduction score `D_s/√H_ss`, a window of the PSF (±256 px, widened to 2.5σ for broad scales) subtracted from all smoothed
  residuals at each iteration, tile maxima for the peak search, restricted to the mask's bounding box and the mask. The cycle ends at `max(threshold, cyclefactor·sidelobe·peak)` (CASA's
  cycle threshold), after `cycleniter` iterations, or at `niter` in total.
* **Restoration.** The model (Jy per pixel, scales rendered as peak-1 Gaussians) is convolved with the peak-1 Gaussian fitted to the PSF's main lobe above 0.35
  (`fit_beam`, least squares with a free centre; tested on analytic PSFs to 0.2%), plus the exact residual dirty image. FITS: DSHARP's WCS convention (SIN, RA decreasing with x, row 0 South in the file;
  virgil's row 0 is North, so rows are flipped on writing), `BUNIT=Jy/beam`, `BMAJ/BMIN/BPA`.
* **Comparison** (`compare`, also run after `run` when `--dsharp` is given): our image bilinearly resampled onto DSHARP's grid (valid where our field covers it); flux inside DSHARP's mask
  (sum / beam area, each image with its own beam); peak; fitted beam against the header's; rms in DSHARP's noise annulus; difference image (±10% of peak); deprojected radial profile
  (i = 47.5°, PA = 144.5°, Huang et al. 2018, ApJL 869, L42); centroid offset; orientation check = correlation of DSHARP's image with ours and with its three reflections about the mask
  centre (the identity must win). Output `summary.md`, `compare.json`, `compare.png`.

### Differences from `tclean`, so that none is a surprise

| | `tclean` (CASA 5.1.1) | here | expected effect |
|---|---|---|---|
| scale kernels | tapered truncated parabolas | Gaussians (the paper calls the scales Gaussian) | small, in how broad flux is shared between scales |
| scale selection | CASA's normalisation of the smoothed residual | `D_s/√H_ss` (chi² reduction) with CASA's bias | different scale mix, same image to the noise |
| major cycle | degrid/regrid, w-projection off | exact nufftax model and adjoint | ours is more accurate; differences ≪ noise |
| PSF in minor cycle | a fraction of the PSF image | a ±256 px window | residual error is removed at each major cycle |
| field | 4000² × 3 mas | 3001² × 3 mas (9″; `--npix`) | none inside the mask (2.75″ semi-major); the noise annulus is clipped to our field |
| `cycleniter`/`niter` | 300/50000 | same; `--time-limit` stops after a major cycle | a run cut by the time limit is not at the threshold: `clean.json` says `stop` |
| imaging weights | the MS weights, CASA's density | the npz `weight` as read from the MS, virgil's density (same Briggs formula, same cell) | none expected; neither applies the model-free ŝ error rescaling |
| data | MS (CORRECTED_DATA) | npz of `alma_imlup_read` 15fecaa-cd (agrees with MPoL's export to 5.6e-8 Jy) | none |

The local `~/data/alma_imlup/subsample_1000` is **from the DATA column** (written 2026-10-19, before D1), so it only serves timing and tests; the image comparison needs the OzSTAR
15fecaa-cd read.

### Tests (`tests/test_alma_clean.py`, 5 s, CPU)
Beam fit on analytic Gaussian PSFs (even and odd grids, four position angles, with and without sidelobes); the minor cycle recovers a point source and a broad Gaussian from
`dirty = model * PSF` exactly; `restore` puts a point at its pixel with unit peak and the right flux; the taper is the transform of the image-plane Gaussian; and, with
`virgil.radio`, the full Cotton–Schwab loop on synthetic `ComplexVisData` at 65² recovers two point sources (flux per pixel to 8%, restored peak to 8%) and a Gaussian through the uv taper
(peak and beam).

## 4. Proposed virgil API

Everything above uses public API except two things worth adding. Proposal (virgil PR later; this note is the spec):

```python
class ComplexVisData:
    def clean(self, npix, pixel_scale_mas, *, weighting="briggs", robust=0.5, taper=None,        # taper = (bmaj_mas, bmin_mas, pa_deg) or None
              scales_mas=(0.0,), gain=0.1, small_scale_bias=0.6, threshold=None, niter=50_000,
              cycle_niter=300, cycle_factor=1.0, mask=None, uv_cell=None, chunk_size=1_000_000,
              beam=None, time_limit=None) -> RadioCleanResult
```
`RadioCleanResult`: `model` (Jy/pixel, signed), `components` (per scale), `residual` (Jy/beam), `psf`, `beam` (`Beam(major_mas, minor_mas, pa_deg)`, fitted unless given),
`restored` (property), `history` (per major cycle: iterations, peak residual, flux, wall time), `stop` ("threshold" | "niter" | "time_limit" | "no_progress"), and `to_fits(path)` in the sky convention of `Image`.
Naming: a method on `ComplexVisData`, not a second `virgil.imaging.clean` overload, because the two algorithms share a name and nothing else (gradient matching pursuit on any likelihood against image-plane
CLEAN on linear data); `virgil.imaging.clean`'s docs should say which to use. Alternatively `virgil.radio.clean(data, ...)`; the method keeps the `dirty_image`/`dirty_beam` family together.

Prerequisites, each small and independently useful:

1. **`taper=(bmaj, bmin, pa)` on `dirty_image` and `dirty_beam`** (elliptical, applied after the Briggs density): our workaround, tapering the visibilities and normalising by the PSF's centre, works but is
   three lines every user would rewrite, and `CalibrationNuisance.taper_fwhm_mas` is circular and a model-side taper (a different thing).
2. **Imaging weights computed once.** `_imaging_weights` sorts all N points (`np.unique` on the density cell: ~10 s at 4.24e7) at *every* `dirty_image` call, i.e. every major cycle. Expose `data.imaging_weights(weighting, robust, uv_cell)` returning an array that `dirty_image`/`dirty_beam` accept (`weights=`), or cache them on the call.
3. **A public chunked residual**: `data.residual_visibilities(scene, values, nuisance, chunk_size)` (exists privately as `_model_chunked`); `dirty_image(scene=...)` already uses it.
4. **Signed images**: the positive/negative split costs two model transforms per major cycle when there are negative components; a `signed=True` `Image` (no softmax, explicit flux per pixel) would halve that and give CLEAN a native model type. Not required.
5. `fit_beam(psf, pixel_scale_mas, cutoff=0.35)`, `sidelobe_level(psf)` and `restore(model, residual, beam)` as functions in `virgil.radio` (or `virgil.imaging`), next to `convolve_beam`/`beam` which exist for `OIData`.

Where the minor cycles live: `virgil/_clean.py`, NumPy (they are a sequential greedy search with a data-dependent stopping rule; JAX buys nothing and the tile-max bookkeeping is awkward under jit).
Run the major cycles under `eqx.filter_jit` with the term passed as an argument (virgil#438).

Validation to carry over: point sources and a Gaussian on synthetic `ComplexVisData` (the tests here); the beam fit; and against CASA's `tclean` on a small simulated MS (not done: CASA is only in `alma-casa` on OzSTAR).

## 5. Built / results

*Built:* `scripts/alma_imlup_clean.py`, `src/crosscheck/alma_clean.py`, tests, this note; ozstar_scripts job `alma_imlup_clean`.
*Not yet run on the full data*: the first OzSTAR run (job id, wall time, the comparison table of `summary.md`) goes here when Ben has run the job; the numbers to compare are DSHARP's:
beam 44 × 43 mas PA 115°, peak 7.11 mJy/beam, 253 mJy in the mask, rms 14 µJy/beam (Table 4), and MPoL's analysis of the same image (flux 0.2548 Jy, peak 3.27 Jy/arcsec², beam area 0.00217 arcsec²).
