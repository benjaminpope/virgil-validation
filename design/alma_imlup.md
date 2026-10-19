# Plan: ALMA IM Lup, MaxEnt against the multipole GP

## Revision 2026-10-19

The likelihood is now an exact NUFFT on the visibilities, using nufftax
(virgil#426), instead of FFT-cell gridding. What changed, and why:

| Change | Why |
|---|---|
| FFT-cell gridding (`GriddedVisData`) dropped from the likelihood | Its cell-centre phase error is about 1.5 rad at the disk edge on the 1024² grid. Check (b) would almost certainly have failed. |
| Default likelihood: nufftax on the raw visibilities, f32, chunked | About 0.1 s per value + gradient for all 42.4 M points on an A100. That is cheaper than the old Hankel-on-cells plan, and exact to eps 1e-5. |
| Optional: fine bins at their weighted centroids | The error is second order in the bin width, so it is negligible at 650 λ. Compression is modest (an estimated ≲3×), so this is a laptop and Hankel option, not the default. |
| New: Toeplitz (dirty-beam) form for frozen offsets | It is exact for a pixel image and costs one 2N² FFT per evaluation, independent of N_vis. It makes the cross-validation and evidence sweeps cheap. |
| `MultipoleDisk` is rendered to pixels, then goes through nufftax or Toeplitz | Hankel on 4×10⁷ points is about 10¹¹ Bessel evaluations per gradient. The Hankel path stays only as a check on a subsample. |
| σ-factor formula corrected: s² = Σw\|r\|²/(2N), not 4N | With variance s²/w on each of Re and Im, E[w\|r\|²] = 2s². |
| Field size raised to at least 6″ | The NUFFT model does not wrap. The rotated CLEAN mask's RA × Dec extent (≈ 4.3″ × 4.9″) fits in 5.12″ only if centred on the phase centre; 6″ adds padding for a mask–phase-centre offset, at 1.56× the pixels. |
| New phase P2b: nufftax scale benchmark | The cost figures below are extrapolated from 10⁵ points. |
| Dependency: virgil#426 must merge first | The `ft="nufft"` path, the even-N phase fix and the eps defaults live there. |

**Sources:**

- A100 timings: ozstar_scripts#35, job 18762629 (`results/finufft_plans/out/18762629/combined.md`, f32 jitted value + grad of Σ|V|², in ms).
- A100 threshold scan: job 18819744 (jax 0.10.2, nufftax 0.6.1, value + grad, ms). At 1024² in f32: 0.517 (10² points), 0.523 (10³), 0.543 (10⁴), 0.571 (3×10⁴), 0.699 (10⁵). That confirms ≈ 0.55 ms fixed + ≈ 1.7 ns/point.
  In f64 at eps 1e-12: 1.08 ms floor, 1.85 ms at 10⁵, ≈ 7.7 ns/point, so an f64 check at 4×10⁷ points costs ≈ 0.3 s.
  virgil#426's final `ft="auto"` rules (nufftax when M ≥ 3000 and side ≥ 128 on GPU; M ≥ max(500, 2N) and side ≥ 64 on CPU) are far below ALMA sizes, so `"auto"` always picks nufftax here.
- M1 Max CPU timings: the scans of 2026-10-19 in the #426 session. The fixed cost varied 2–4× between runs.
- MPoL code: MPoL `main` at 8850b483, and MPoL-dev/examples `04-IMLup-multi-EB`.

## Why

So far every virgil validation case has been optical/IR interferometry: V²,
closure phases, 10²–10⁴ data. Nothing in the physics stops virgil working at
sub-mm wavelengths, but the engineering is different:

- **Complex visibilities with absolute phase**, and a thermal-noise weight
  on every visibility.
- **10⁶–10⁷ visibilities.** Here virgil's DFT (cost ∝ M·N²) and its dense
  Gauss–Newton curvature stop at about 10⁴. nufftax (virgil#426) removes the
  first limit; the second is avoided with Hessian–vector products.
- **Calibration nuisances for each execution block (EB)**: amplitude scale,
  weight scale, phase-centre offset.
- **A primary beam.** It is negligible here: IM Lup's continuum is about 2–2.5″
  across (outer radius to be fixed from DSHARP II), against a Band 6 primary beam of about 25″.

The MPoL example
[`04-IMLup-multi-EB`](https://github.com/MPoL-dev/examples/tree/main/04-IMLup-multi-EB)
(Czekala and collaborators) is a ready-made target. It images the public
DSHARP IM Lup continuum data with a regularised pixel image and fits the
calibration nuisances of each EB jointly with the image. IM Lup is also a
hard and interesting disk: rings in the outer disk and a complex m=2
trailing spiral inside about 1″ (DSHARP III). That makes it a test of
whether the `MultipoleDisk` polar GP can compete with a pixel image when the
structure is not purely axisymmetric.

## What the MPoL example does

Surveyed on 2026-10-17; code read again on 2026-10-19. Example files are
under `04-IMLup-multi-EB/`; library files are under `src/mpol/` in MPoL.

**Export** (`casa_analysis/src/export_asdf.py`):

- `visread.process.get_processed_visibilities(ms, ddid)` is run for each
  of the 26 spectral windows (ddids).
- It averages the two polarisations, does no channel averaging, and writes
  `imlup.asdf` nested as obsid → ddid.
- The ddid → observation map is hard-coded: obs0 = 0–4, obs1 = 5–10,
  obs2 = 11–12, obs3 = 13–14, obs4 = 15–17, obs5 = 18–21, obs6 = 22–25.

**Weight check** (`weight_scatter.py`): a Gaussian fit to the histogram of
(data − CLEAN model)/σ for each ddid.

**Model** (`mpol_analysis/src/sgd.py`):

- PyTorch BaseCube → GaussConvFourier → ImageCube → `fourier.NuFFT`
  (`sgd.py:58–62`, `forward` at `sgd.py:139–158`), at 1028² pixels and
  5 mas (`sgd.py:441`).
- The NuFFT is torchkbnufft's `KbNufft` on the **loose** visibilities of each
  batch (`fourier.py:111–141`). It is not `NuFFTCached`: that class
  precomputes sparse interpolation matrices for one fixed uv set
  (`fourier.py:436–499`), which does not suit shuffled mini-batches.
- Nuisances fitted jointly with the image:
  - `log10_sigma_factor` for each ddid (26), applied as w/s²
    (`sgd.py:64–70`);
  - `log10_amp_factor` for each observation (6 free, 1 fixed). It scales
    the model by g **and** the weights by 1/g² (`sgd.py:72–98`);
  - (δRA, δDec) for each observation, as a phase ramp
    exp(−2πi(δRA·u + δDec·v)) on the model (`sgd.py:100–137`).
- Loss: `losses.neg_log_likelihood_avg` (the full Gaussian ln L, including
  Σ ln w, divided by 2N; `losses.py:129–262`) plus an entropy term (prior
  1e-4, total flux 0.253 Jy) (`sgd.py:298–302`).
- Optimiser: AdamW on shuffled 40k-visibility batches, optionally grouped by
  ddid (`DdidBatchSampler`, `sgd.py:164–199`, 425–436, 444).
- Paper-run settings (commented in the Snakefile): λ_ent = 1e-9, FWHM 30 mas,
  AdamW at lr 1e-2, 10 epochs, batches of 40k visibilities, run twice from a
  checkpoint.

Elsewhere in MPoL, the standard path is gridded:

- `gridding.DataAverager` (`gridding.py:558`) averages onto FFT cells,
  "assuming that the visibility function is constant across the cell"
  (`gridding.py:428–429`).
- `datasets.GriddedDataset` (`datasets.py:14`) holds the non-empty cells, and
  `fourier.FourierCube` (`fourier.py:14`) evaluates the model there by FFT.
- `datasets.Dartboard` (`datasets.py:147`) provides the polar cross-validation
  cells; it keeps Hermitian pairs in the same cell (`datasets.py:286`).

**What it lacks:** cross-validation (the validation split is commented out
in `attic/`), written-up results, and any statement of the visibility count.
It is a prototype. We are replicating its *method*, not a published number.

## The data

Public DSHARP release, <https://almascience.eso.org/almadata/lp/DSHARP/>
(page last modified 2020-02-05). Reducer J. Huang, CASA 5.1.1.

| File | URL path | Size |
|---|---|---|
| Continuum MS | `MSfiles/IMLup_continuum.ms.tgz` | 1,908,078,667 B (1.9 GB, compressed) |
| CLEAN image | `images/IMLup_continuum.fits` | 64,010,880 B (64 MB) |
| Reduction script | `scripts/IMLup_continuum.py` | 35 KB |
| Radial profile | `profiles/IMLup.profile.txt` | small |

**Execution blocks.** Six datasets make up seven EBs:

| Label | Programme | Date | EBs |
|---|---|---|---|
| SB1 | 2013.1.00226.S | 2014-07-06 | 1 |
| SB2 | 2013.1.00226.S | 2014-07-17 | 1 |
| SB3 | 2013.1.00694.S | 2015-01-29 | 1 |
| SB4 | 2013.1.00694.S | 2015-05-13 | 1 |
| SB5 | 2013.1.00798.S | 2015-06-09 | 1 |
| LB1 | 2016.1.00484.L | 2017-09-25, 2017-10-24 | 2 |

**What the reduction did:**

- CO channels were flagged (−5 to 15 km/s for the SBs, −15 to 25 km/s for
  LB1) before averaging to pseudo-continuum.
- SB2 antennas were flagged (PM03, DV14, DA47, DV15, DA64), and
  atmospheric lines were flagged in SB3 and SB4.
- The EBs were self-calibrated and aligned by DSHARP. The nuisances we fit
  should therefore come out close to 1 and 0. That makes them a test of
  the machinery, not a correction to DSHARP.
- The CLEAN mask is an ellipse at PA 145°, with semi-axes 2.75″ × 1.75″,
  centred on 15h56m09.20s −37°56′06.42″.

### Visibility count

Measured on 2026-10-17 with `casatools` 6.7.0 (pip install, Python 3.11) on
the MS downloaded to `~/data/alma_imlup/`. The extracted MS is 2.5 GB. The
data are never committed.

**How the counts were made:**

- A visibility is one (row, channel) of a cross-correlation, kept only if
  both polarisations are unflagged. That follows visread's convention.
  Autocorrelations are excluded.
- "Cells" counts the distinct occupied cells of the 1024² × 5 mas FFT grid.
  The cell size is Δu = 40,286 λ, and the grid is Hermitian-folded with
  v ≥ 0.

| ddid | obs | chans | rows | N_vis | cells | max uv (Mλ) | ν (GHz) |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 1 | 92,103 | 92,103 | 165 | 0.49 | 231.39 |
| 1 | 0 | 1 | 92,103 | 91,476 | 165 | 0.49 | 231.30 |
| 2 | 0 | 1 | 92,103 | 92,103 | 166 | 0.49 | 231.20 |
| 3 | 0 | 1 | 92,103 | 91,563 | 164 | 0.49 | 230.52 |
| 4 | 0 | 4 | 92,103 | 368,412 | 169 | 0.50 | 234.92 |
| 5 | 1 | 1 | 94,938 | 66,426 | 162 | 0.53 | 245.51 |
| 6 | 1 | 1 | 94,938 | 66,426 | 163 | 0.53 | 245.54 |
| 7 | 1 | 1 | 94,938 | 66,426 | 150 | 0.52 | 241.54 |
| 8 | 1 | 1 | 94,938 | 66,251 | 151 | 0.52 | 241.68 |
| 9 | 1 | 1 | 94,938 | 66,276 | 151 | 0.52 | 241.75 |
| 10 | 1 | 1 | 94,938 | 66,126 | 151 | 0.52 | 241.77 |
| 11 | 2 | 16 | 97,567 | 1,365,938 | 50 | 0.27 | 242.02 |
| 12 | 2 | 16 | 97,567 | 1,365,938 | 49 | 0.27 | 243.75 |
| 13 | 3 | 16 | 129,935 | 1,815,674 | 91 | 0.38 | 242.00 |
| 14 | 3 | 16 | 129,935 | 1,685,983 | 92 | 0.38 | 243.73 |
| 15 | 4 | 1 | 221,344 | 219,248 | 202 | 0.60 | 231.22 |
| 16 | 4 | 1 | 221,344 | 209,358 | 201 | 0.60 | 230.53 |
| 17 | 4 | 16 | 221,276 | 3,088,092 | 207 | 0.61 | 232.70 |
| 18 | 5 | 16 | 282,509 | 3,955,126 | 10,598 | 9.46 | 232.58 |
| 19 | 5 | 16 | 282,509 | 3,955,126 | 11,223 | 9.96 | 244.97 |
| 20 | 5 | 16 | 282,509 | 3,955,126 | 11,304 | 10.03 | 246.89 |
| 21 | 5 | 8 | 282,509 | 2,260,072 | 8,940 | 9.36 | 230.52 |
| 22 | 6 | 16 | 341,408 | 4,779,712 | 17,104 | 10.82 | 232.59 |
| 23 | 6 | 16 | 341,408 | 4,779,712 | 18,141 | 11.39 | 244.98 |
| 24 | 6 | 16 | 356,929 | 4,997,006 | 18,941 | 11.48 | 246.89 |
| 25 | 6 | 8 | 356,929 | 2,855,432 | 14,724 | 10.70 | 230.53 |

**Totals by observation:**

| obs | Label | N_vis | Cells |
|---|---|---|---|
| 0 | SB1 | 0.74 M | 829 |
| 1 | SB2 | 0.40 M | 928 |
| 2 | SB3 | 2.73 M | 99 |
| 3 | SB4 | 3.50 M | 183 |
| 4 | SB5 | 3.52 M | 610 |
| 5 | LB1 EB 1 | 14.1 M | 42,065 |
| 6 | LB1 EB 2 | 17.4 M | 68,910 |
| **all** | | **42.4 M** | **113,624** |

**What this means for the design:**

- The FFT-cell count (1.1 × 10⁵) shows how coarse the 40 kλ cells are, not
  how far the data can be compressed safely (see "Likelihood options").
- The SBs (10.9 M visibilities) have short uv tracks and many channels, so
  they compress well into fine bins. LB1 (31.5 M, 74%) barely compresses.
- The CLEAN mask (semi-axes 2.75″ × 1.75″ at PA 145°) spans
  2√(a²sin²θ + b²cos²θ) × 2√(a²cos²θ + b²sin²θ) ≈ 4.3″ (RA) × 4.9″ (Dec).
  That fits in 5.12″ only if the mask is centred on the phase centre. A
  NUFFT model does not wrap, but emission outside the field goes
  unmodelled, so for padding and a possible mask–phase-centre offset the
  field is at least 6″ (1.56× the pixels, and the Toeplitz kernel memory): 1280² at 5 mas (6.4″, MPoL's pixel), or
  1024² at 6 mas (6.1″). Both are finer than the 9 mas Nyquist pixel for
  11.5 Mλ. Fix the outer radius from DSHARP II/Table 1 before P2.
- The DSHARP MS keeps up to 16 channels per window.

The counting script is a scratch prototype. P1 should turn it into
`alma_worker.py`, and add the occupied-bin count at 650 λ and 2 kλ.

## Design

### Ingest stays outside virgil

`src/external_bridge/alma_worker.py` runs in its own venv
(`scripts/setup_external.sh`) with `casatools`, `casadata` and `visread`.

- It writes one `.npz` per ddid: u, v in wavelengths (float64), Re, Im,
  weight, frequency, obsid, ddid. It also records the channel width and
  integration time, which set the smearing already in the data.
- Its conventions go in `docs/alma_conventions.md` first: conjugation, the
  sign of the u axis, polarisation averaging, and weight units.

Independent checks in `src/crosscheck/`:

- a DFT dirty image, compared with CASA `tclean` at `niter=0` on the same
  pixels;
- the weight scatter for each ddid, recomputed from the residual
  histogram. The residuals are against the DSHARP CLEAN model, so the check
  is independent of virgil.

The crosscheck code never uses nufftax: virgil does, so that would break
the independence rule. finufft is not fully independent either: nufftax
ports the same ES-kernel algorithm and mirrors its conventions (`isign`,
mode ordering, the N/2 centre on even N). So every sign and centre
convention in the crosscheck is pinned against the float64 DFT only. finufft
(eps 1e-12) may be used for scale, after it matches that DFT on a small
grid with an asymmetric source.

### Likelihood options

The model is a pixel image I on an N² grid (or a model rendered to one).
Its visibilities are V(u) = Σ_p I_p exp(−2πi u·x_p). The error columns give
the phase or amplitude error for emission at x = 2.5″ (1.2 × 10⁻⁵ rad).

| Option | Model evaluated at | Error at x = 2.5″ | A100 value + grad | M1 Max CPU | Memory |
|---|---|---|---|---|---|
| (a) **raw nufftax** (default) | each visibility | eps 1e-5 × V(0) ≈ 2.5 µJy | ≈ 0.1 s (0.55 ms fixed + ≈ 1.7 ns/point) | ≈ 3–8 s (0.06 ms per 10³ points, ×2–3 for the gradient) | M·ns² ≈ 42.4 M × 36 ≈ 1.5 × 10⁹ weights ≈ 6–8 GB in f32; chunked, peak ≈ largest chunk |
| (b) fine bins + nufftax | weighted centroid of each bin | second order: ½(2πx·Δu_b/√12)², ≈ 1 × 10⁻⁴ at Δu_b = 650 λ, ≈ 1 × 10⁻³ at 2 kλ | (a) × compression | (a) × compression | (a) × compression |
| (c) FFT cells (old plan, MPoL `DataAverager`) | cell centre | first order: π·Δu·x ≈ 1.5 rad at 1024² × 5 mas; even at the centroid, a ≈ 40% amplitude loss | one FFT | one FFT | small |
| (d) **Toeplitz form** (frozen offsets) | exact for a pixel image | eps of the precompute (f64, 1e-12) | ≈ 1–3 ms (one 2N² FFT) | ≈ 50–200 ms | one real 2N² kernel per ddid: 26 × 2560² × 8 B ≈ 1.4 GB |

**Notes on the table:**

- **(a), extrapolated.** It is extrapolated from 10⁵ points: 0.714 ms at
  1024²/10⁵ and 0.562 ms at 10⁴, so ≈ 1.7 ns per extra point. P2b measures
  it.
- **(a), memory.** ns ≈ 6 for eps 1e-5 in 2D. Chunk by memory, not by ddid:
  each call pays the grid-FFT fixed cost, so 26 calls add ≈ 14 ms on the
  A100 but seconds on a CPU.
- **(b), bins.** Take the weighted mean of (u, v, V) in each bin, after
  folding to v ≥ 0; a folded point has (u, v, V) → (−u, −v, V*), or
  Hermitian partners partly cancel. A bin never crosses a ddid, an EB or
  a dartboard cell, so the per-point nuisances stay exact and the CV mask
  applies per bin. None of these rules is caught by the χ² check at the
  fitted model, so P2 tests each directly. Evaluating the model at the weighted centroid cancels
  the first-order term, so the error is ½|∇²V|·⟨δu²⟩. The bin count is
  unknown until P1.
- **(b), compression estimate.** In 6 s an LB track at 3 Mλ moves ≈ 1.3 kλ.
  Adjacent channels are ≈ 1.5 kλ apart, if 16 channels span ≈ 1.9 GHz. So
  LB compresses ≲ 1.5× at 650 λ, while the SBs compress ≳ 5×. The expected
  total gain is ≲ 3×.
- **(c), dropped.** MPoL's own IM Lup example avoids it, by using `NuFFT`
  on the loose visibilities.
- **(d), derivation.** For a pixel image,
  χ² = Q − 2 Σ_p I_p D(x_p) + Σ_pq I_p I_q B(x_p − x_q), with:
  - D the (unnormalised) dirty image, Re Σ_k w_k V_k* e^(−2πi u_k·x);
  - B the dirty beam on the (2N−1)² lags;
  - Q = Σ w|V|².
- **(d), precompute.** D and B are each one type-1 NUFFT per ddid, done
  once (chunked, f64).
- **(d), what it allows.** Per-EB amplitude g scales the terms as g and g²,
  so they stay exact. A fixed spectral index is absorbed exactly into w and
  V. A free offset needs D at shifted positions (a new type-1 per change),
  so freeze the offsets first. This is the Toeplitz trick from MRI
  (Fessler et al. 2005).
- **(d), precision.** Q ≫ χ² at high SNR, because Σw|V|² on the SB data is
  ~10⁹. The precompute and the evaluation must therefore be in f64.

**Smearing already in the data.** The bandwidth term is δu ≈ |u|·Δν/ν. With
Δν ≈ 117 MHz at 11 Mλ, that is ≈ 5.5 kλ, a ≈ 0.4 rad phase spread at 2.5″.
The time term is δu ≈ |u|·ω_E·t_int, ≈ 4.8 kλ for 6 s at 11 Mλ. These
are estimates; P1 reads Δν and t_int from the MS. The smearing is the same
for every method here (MPoL does not model it either), and it is far larger
than the 650 λ bin error. Modelling it is out of scope; the comparison
metrics stay valid because both methods share it.

**Precision.** nufftax's f32 error at eps 1e-5 is relative to Σ|I| ≈ V(0)
≈ 0.25 Jy, so it is ≈ 2.5 µJy.

- An ALMA visibility in one channel and one integration has σ of order
  10 mJy per component (to be measured in P1), so the transform error is
  ≈ 2.5 × 10⁻⁴ σ. Over 4.24 × 10⁷ points, Σw|ε|² ≈ 2.7, and the cross
  term 2Σw Re(r*ε) has a standard deviation of ≈ 3.3. ε changes with the
  model, so this is χ² noise of a few units that f64 reduction does not
  remove.
- The transform is therefore f32 for the joint fit, where O(3) in χ² ≈
  8.5 × 10⁷ is irrelevant. Any cross-validation or evidence evaluation
  on option (a) uses f64 nufftax (≈ 0.3 s at 4 × 10⁷ points, job
  18819744); the Toeplitz form is f64 already. But the χ² reduction must be f64 (cast r
  before summing): 2N ≈ 8.5 × 10⁷ in f32 has a resolution of 8, which is
  too coarse for cross-validation and evidence differences.
- uv in f32 is fine: 11.5 Mλ lies in [2²³, 2²⁴), so the spacing is 1 λ and the rounding error ≤ 0.5 λ.
- f64 nufftax at eps 1e-12 roughly doubles ns, so it costs about 4× the
  memory and ≈ 4.5× the per-point time (job 18819744). Keep it for checks.

**Recommended defaults:**

| | Laptop (M1 Max) | OzSTAR A100 |
|---|---|---|
| Development and tests | SB1+SB2 raw (1.1 M), or any ddid subsample, option (a) | — |
| Joint image + all nuisances | not run (see `local-compute-limits`) | (a) raw, f32, f64 reduction, chunks of ≤ 10⁷ points |
| λ and cross-validation sweeps, evidence | (d) Toeplitz with the offsets frozen, after an OzSTAR joint fit | (d) Toeplitz, or (a) in f64 if offsets stay free |
| `MultipoleDisk` | rendered → (d) | rendered → (a) or (d) |
| (b) bins | only if P1 shows ≥ 3× compression | not needed |

### Following and departing from MPoL

| Topic | MPoL | Here | Follow or depart, and why |
|---|---|---|---|
| Forward model on the loose uv | `NuFFT`, torchkbnufft, per batch | nufftax per chunk, all data per step | **Follow**: exact at the raw uv. |
| Precomputed interpolation | `NuFFTCached` exists but is unused for IM Lup | not used | **Follow**: a precomputed-points prototype for #426 was slower on CPU. |
| Gridded data | `GriddedDataset` and `FourierCube` elsewhere | dropped from the likelihood | **Depart**: the cell-centre approximation fails at this resolution. |
| Optimiser | AdamW, 40k batches, 10 epochs | full-batch L-BFGS | **Depart**: a deterministic optimum, needed for CV differences, Laplace evidence and reproducible trust nodes. A full batch costs ≈ 0.1 s on the A100. |
| Amplitude | ×g on the model, w/g² | ×g on the model only | **Depart**: per-ddid s absorbs the weight scaling (s_ours = g·s_MPoL), which removes a near-degeneracy. Compare via that map. |
| Offsets | phase ramp on the model | phase ramp on the model | **Follow**: the shift theorem makes it exact. |
| σ factors | log10 s per ddid, by SGD | per ddid, profiled in closed form, or log s with a log-uniform prior | **Depart in method only**: same model, faster and deterministic. |
| Regulariser | entropy, λ 1e-9, plus a 30 mas Gaussian base layer | `MaxEntropy` + small `TSV`; optional Gaussian convolution | **Follow**, with λ set by cross-validation. |
| Cross-validation | `Dartboard`; not used in example 04 | dartboard folds on all groups at once | **Follow** the library, and do what the example left out. |
| Spectral index | ignored (231–247 GHz) | open decision: a fixed α absorbed into w and V, or ignored as in MPoL | **Flag**: each LB EB spans 230.5–246.9 GHz (ratio 1.071), so α ≈ 2–3 gives ≈ 15–23%, which per-EB g cannot absorb. |

### Nuisances

On raw or binned points, each nuisance is a per-point factor on the model.
The uv points never move.

- **Amplitude:** M_k = g_e(k) · V_k. One EB's g is fixed to 1. Parametrise
  as ln g (Jeffreys).
- **Offset:** M_k = g · exp(−2πi(δα u_k + δδ v_k)) · V_k. ∂M/∂δ =
  −2πi u_k M_k is analytic. Shifting the image is exactly this ramp, so
  no ∂V/∂u term and no nufftax JVP with respect to the points is needed.
  One EB's offset is fixed, to break the degeneracy with the model centre.
- **σ factor per ddid:** with variance s²/w on each of Re and Im,
  −ln L_d = 2N_d ln s_d + Σ w|r|²/(2s_d²) + const, which is minimised at
  s_d² = Σ w|r|²/(2N_d). N_d = Σ_k m_k is the **unmasked** raw count in
  the current fold (m_k the 0/1 CV mask); using the full count would bias
  s² low by the held-out fraction (20% for 5 folds) and with it every
  training χ². Profile s out, or sample ln s with the same N_d. With bins,
  Σ_k w|r|² = (Q_b − W_b|V̄_b|²) + W_b|V̄_b − M_b|² exactly, so store
  W_b, V̄_b, N_b per bin and Q per ddid, and N_d per fold.
- **Cross-validation:** the folds are `Dartboard`-style polar cells
  (q, φ, Hermitian-paired), and a held-out cell is held out in every ddid
  and EB together. On points this is a 0/1 weight mask, so the shapes and
  the jit stay fixed across folds. In the Toeplitz form, each fold needs its
  own D and B: k folds × 26 ddids, precomputed once.

### nufftax limits (0.6.1)

| Limit | Consequence here |
|---|---|
| The `_make_batcher` batching rule handles only one batch dimension, at axis 0. `jax.hessian` and nested `vmap` fail. | Laplace and evidence use hvp (`jvp` of `grad`), or a dense Hessian built from **one** `vmap` of hvps over basis chunks. Never `jax.hessian`. Do not monkeypatch; the fix goes upstream with Ben's OK. |
| f32 eps 1e-5 is relative to Σ\|I\| | Fine for the likelihood (see Precision). Reductions in f64. |
| Memory ∝ M·ns² | Chunk on the GPU. CPU runs at the full 42.4 M are not for the laptop. |
| The OzSTAR conda env pins jax 0.10.2 | Per `latest-jax`, P2b and P3 use a fresh pip env on the latest JAX with nufftax and virgil from the #426 merge. Never pin back. |

**Exploiting linearity for the evidence.** If the `MultipoleDisk`
brightness is linear in its latents (no positivity transform), then at
fixed geometry and nuisances the likelihood Hessian is A = RᵀFᵀWFR, where R
is the render matrix and F the Fourier operator. A does not depend on the
latents or the hyperparameters, so compute it once per geometry (6,800
hvps). The whitened Hessian for any hyperparameters is then I + LᵀAL, and
the Laplace evidence is exact. In the Toeplitz form each hvp is one 2N² FFT
(≈ 2 ms on the A100), so the whole 6,800² matrix takes ≈ 15 s.

New in virgil (P2, after #426):

- `ComplexVisData`: complex visibilities with a weight per point and a
  group index (ddid, EB). Its likelihood calls `image_visibilities` with
  `ft="nufft"` (or `"auto"`) in chunks, with f64 reductions. There is an
  optional bin-averaging constructor (option b).
- `CalibrationNuisance`: g and the offset per EB, and s per ddid
  (profiled or free).
- `ToeplitzVisData` (option d): it is built from `ComplexVisData` once the
  offsets are frozen, and its χ² matches `ComplexVisData` to 1e-8 relative.
- `GriddedVisData` is **not** needed. A gridded dirty image stays as a
  diagnostic in the crosscheck, not as a likelihood.

### MaxEnt

- `Image` at 1280² and 5 mas (or 1024² at 6 mas), with optional Gaussian
  convolution matching MPoL's 30 mas, plus `MaxEntropy` with a flat prior
  and a small `TSV`. Fit with full-batch L-BFGS on option (a), jointly with
  `CalibrationNuisance`.
- λ is chosen by k-fold dartboard cross-validation, as in MPoL and
  Zawadzki et al. 2023 (folds as under Nuisances). Run the sweep in the
  Toeplitz form with offsets frozen at the joint-fit values. Refit the
  offsets on (a) at the chosen λ, as a check that they did not move. The
  GP uses the identical folds.
- `l_curve` discrepancy and classic Gull–Skilling are reported as secondary
  choices.

### Multipole GP

`MultipoleDisk` (virgil `main`, `models.py`):

- The m=0 term is a `RadialGP` with n_r ≈ 256–400 nodes over 0–1.5″.
- Harmonics m = 1…M, with M ≈ 6–8. Each order has its own Matérn
  lengthscale, because spiral arms make a_m(r) and b_m(r) oscillate in r.
- Geometry (inclination, PA, centre) is fitted jointly with the EB offsets.
  One EB's offset is fixed to remove the degeneracy with the centre.

**Data path.** The Hankel quadrature is not an FFT. On raw data it costs
42.4 M × 300 nodes × 9 orders ≈ 10¹¹ Bessel evaluations per gradient, and
even on the 113k FFT cells (which are invalid anyway; see option c) it was
3 × 10⁸. Options:

| Path | Cost per gradient | Error | Use |
|---|---|---|---|
| Hankel on raw points | ≈ 10¹¹ Bessel | exact | check only, on a ≤ 10⁵ subsample |
| Hankel on fine bins | 10⁸–10¹¹, depends on P1 | second order, as (b) | only if P1 shows strong compression |
| **Render to pixels → nufftax or Toeplitz** | 1280² × (2M+1) interpolation + (a) or (d) | pixel sampling: 5 mas samples to 20.6 Mλ, so model power at 29.7–41 Mλ aliases into the data. Matérn spectra fall as power laws (Matérn-3/2 at ℓ = 10 mas: amplitude ≈ 10⁻² at 30 Mλ), and node interpolation and a non-zero edge add f⁻² tails, so this is **not** negligible at eps 1e-5. Render by pixel integration or ≥ 2× supersampling, and bound the residual from the Hankel model at 29.7–41 Mλ. | **default** |

Rendering needs a new pixel-image method on `MultipoleDisk` (P4). Compute
the aliasing bound before running the check (from the Hankel model at
29.7–41 Mλ, for the rendering scheme chosen), pin it, and then check the
rendering against the Hankel model on a raw subsample to within eps plus
that bound.

**Fitting.** Fit the geometry on the SB data first (10.9 M raw points; far
fewer in the Toeplitz form), then freeze it for the full latent and
hyperparameter fit. Hyperparameters come from `log_evidence`, a Laplace
approximation in (2M+1)·n_r whitened latents: about 3,300 (M=6, n_r=256) to
6,800 (M=8, n_r=400). Build the Hessian from hvps, never with
`jax.hessian`. Use the linearity shortcut above where it applies. The model
is also scored on the same dartboard cross-validation as MaxEnt.

frank (Jennings et al. 2020) is an external check on the m=0 profile.

## Comparison metrics

The same metrics for every method:

1. Held-out dartboard cross-validation χ².
2. Residual dirty images for each EB.
3. Radial profile, against DSHARP II, `profiles/IMLup.profile.txt` and
   frank.
4. Spiral pitch angle and arm phase, against DSHARP III, compared in sky
   coordinates.
5. Total flux and the fitted nuisances, against MPoL's
   `analyze_scale_factors.py` (mapping s_ours = g·s_MPoL).
6. The image convolved to the CLEAN beam, against the DSHARP FITS.

Expectation:

- MaxEnt wins wherever the structure is not polar-separable.
- The GP wins on cross-validation in the ringed outer disk, with calibrated
  uncertainties on ring and spiral parameters.

## Phases

| Phase | Where | Work |
|---|---|---|
| P0 | this note | References, data inventory, design, N_vis (done 2026-10-17); nufftax revision (2026-10-19). |
| P1 | virgil-validation PR | ALMA worker, download, export, dirty-image and weight-scatter checks. Record N_vis, Δν, t_int, typical σ per visibility, and occupied-bin counts at 650 λ and 2 kλ. |
| P2 | virgil PR, **after virgil#426 merges** | `ComplexVisData` on nufftax (chunked, f64 reductions, optional bins), `CalibrationNuisance`, `ToeplitzVisData`; tests against a float64 DFT. No `GriddedVisData`. |
| P2b | OzSTAR A100 + M1 Max | Scale benchmark: `bench_jax_nufft.py --npts 1e6 4e6 1.6e7 4.2e7` at 1024² and 1280², f32, value + grad, with peak memory. Repeat the M1 Max runs ≥ 3 times, because its fixed cost varied 2–4× between runs. Also time Toeplitz evaluation at 2560². Fill the cost table above. |
| P3 | OzSTAR GPU | MPoL reference run (`sgd.py` and `analyze_scale_factors.py` in `.venv-mpol` through `mpol_worker.py`, conventions in `docs/mpol_conventions.md`). Then the MaxEnt joint fit on (a), and the cross-validation sweep in the Toeplitz form (sbatch array over λ and folds). Use a fresh pip env on the latest JAX. |
| P4 | OzSTAR GPU | `MultipoleDisk` pixel rendering, with a Hankel check. Geometry on SB, then the full fit and evidence over M and lengthscales. |
| P5 | virgil-validation PR | `docs/imaging/alma_imlup.md`, trust graph and ledger, findings. |

## Verification

- **Transform.** nufftax (f32, eps 1e-5) against the float64 DFT in
  `src/crosscheck/` (root `mathematics`, not virgil's own DFT path) on a
  random 10⁴-point subsample of every ddid, at the fitted image. eps is a
  target for the relative l2 error, not a max-norm bound, so the
  criterion is ‖ΔV‖₂/‖V‖₂ ≤ 3·eps, with c = 3 fixed now. The max-norm
  error is measured in P2b and reported, not used as the pass bound.
  Gradients are checked the same way. The f64 path at eps 1e-12 meets the
  same l2 criterion at its eps.
- **Reduction.** Raw −ln L per ddid from the chunked f64 reduction,
  including the s-dependent constant 2N_d ln s_d with the per-fold
  unmasked N_d, equals an f64 numpy sum to 1e-10 relative.
- **Bins** (if used). χ²_raw − χ²_binned at the fitted model, per ddid,
  from the independent DFT in `src/crosscheck/`, is below about 1 per ddid,
  with SB and LB reported separately. The subsample is whole bins, not
  random points; a random draw puts about one visibility in each bin and
  hides the within-bin term.
- **Toeplitz.** Per-ddid χ² from `ToeplitzVisData` equals the
  `src/crosscheck/` float64 DFT χ² on the raw points to 1e-8 relative,
  per fold, at the fitted image and at a random image (to catch a missing
  lag or a wrong sign). Both virgil forms share the nufftax path, so a
  shared convention error (even-N centre, `isign`) would pass a
  virgil-vs-virgil comparison; that comparison (against `ComplexVisData`
  in f64) is kept only as a plain guard (`self-consistency`).
- **Hankel against rendering.** The rendered `MultipoleDisk` visibilities
  match the Hankel visibilities on a raw subsample to within eps plus the
  pixel-aliasing bound.
- **Dirty image.** It matches CASA `tclean niter=0` (with
  `weighting="natural"` and the N/2 phase-centre pixel convention pinned)
  to a tolerance set by the tclean gridding kernel, not by the noise: both
  images are linear in the same data and share its noise.
- **A control that must fail.** Inject an offset point source (or an
  asymmetric pair) with `simobserve`/`ft` into a copy of the MS. Check its
  position from both tclean and the crosscheck DFT, in RA and Dec signs.
  The conjugated mapping is registered as `kind="control"`. IM Lup's rings
  and m=2 spiral are invariant under a 180° rotation, so a conjugation bug
  would barely change its dirty image; only the injected source exposes it.
- **σ factors.** Fitted σ factors agree with the scatter estimate to about
  5%, where the residuals are taken against the DSHARP CLEAN model (the
  `.model` from `tclean`, or the released FITS degridded by the crosscheck
  DFT), never against a virgil fit.
- **Shared ingest.** Both MPoL's ingest and ours go through visread, so a
  visread convention error would agree on both sides. The tclean check with
  the injected source is the only root that reads the MS independently.
- **Injection-recovery.** A synthetic ring-plus-spiral disk on the real uv
  coverage and weights is recovered by both methods.

## References

ADS search run on 2026-10-17.

| Bibcode | Citation | arXiv |
|---|---|---|
| [2018ApJ...869L..41A](https://ui.adsabs.harvard.edu/abs/2018ApJ...869L..41A) | Andrews et al. 2018, DSHARP I: motivation, sample, calibration, overview | 1812.04040 |
| [2018ApJ...869L..42H](https://ui.adsabs.harvard.edu/abs/2018ApJ...869L..42H) | Huang et al. 2018, DSHARP II: annular substructures | 1812.04041 |
| [2018ApJ...869L..43H](https://ui.adsabs.harvard.edu/abs/2018ApJ...869L..43H) | Huang et al. 2018, DSHARP III: spirals in Elias 27, IM Lup, WaOph 6 | 1812.04193 |
| [2018ApJ...869L..44K](https://ui.adsabs.harvard.edu/abs/2018ApJ...869L..44K) | Kurtovic et al. 2018, DSHARP IV | 1812.04536 |
| [2023PASP..135f4503Z](https://ui.adsabs.harvard.edu/abs/2023PASP..135f4503Z) | Zawadzki et al. 2023, RML image synthesis and validation for ALMA disks | 2209.11813 |
| [2025ApJ...984L..14Z](https://ui.adsabs.harvard.edu/abs/2025ApJ...984L..14Z) | Zawadzki et al. 2025, exoALMA IX: RML imaging of non-Keplerian features | 2504.19111 |
| [2025arXiv250200100C](https://ui.adsabs.harvard.edu/abs/2025arXiv250200100C) | Czekala et al. 2025, MPoL: a PyTorch library for radio interferometric imaging | 2502.00100 |
| [2023zndo...4432501](https://ui.adsabs.harvard.edu/abs/2023zndo...4432501) | Czekala, visread (software) | — |
| [2020MNRAS.495.3209J](https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.3209J) | Jennings et al. 2020, frank | 2005.07709 |
| [2021ApJS..257....1O](https://ui.adsabs.harvard.edu/abs/2021ApJS..257....1O) | Öberg et al. 2021, MAPS I | 2109.06268 |
| [2021ApJS..257...14S](https://ui.adsabs.harvard.edu/abs/2021ApJS..257...14S) | Sierra et al. 2021, MAPS XIV: multiwavelength continuum substructures | 2109.06433 |
| — | Fessler et al. 2005, IEEE Trans. Signal Process. 53, 3393: Toeplitz-based iterative image reconstruction | — |
| — | nufftax (GragasLab/nufftax), pure-JAX NUFFT, 0.6.1 | — |

Still to do:

- Take IM Lup's distance, inclination and PA from DSHARP I Table 1, and its
  pitch angles from DSHARP III. Only the abstracts have been read so far.
- Find a refereed source for the multi-EB weight-scatter method. ADS found
  none by Czekala et al. for 2023–2026. It may be in the MPoL paper or in
  exoALMA II (Loomis et al. 2025, arXiv:2504.19870); not yet checked.
- Check the Fessler et al. 2005 citation on ADS or Crossref, and IM Lup's
  mm spectral index (for the α decision).
