# Plan: ALMA IM Lup, MaxEnt against the multipole GP

## Why

So far every virgil validation case has been optical/IR interferometry: V²,
closure phases, 10²–10⁴ data. Nothing in the physics stops virgil working at
sub-mm wavelengths, but the engineering is different:

- **Complex visibilities with absolute phase**, and a thermal-noise weight
  on every visibility.
- **10⁶–10⁷ visibilities**, where virgil's dense likelihood and Gauss–Newton
  curvature stop at about 10⁴.
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

Surveyed on 2026-10-17. All files are under `04-IMLup-multi-EB/`.

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

- PyTorch BaseCube → GaussConvFourier → ImageCube → NuFFT, at 1028² pixels
  and 5 mas.
- Nuisances fitted jointly with the image:
  - `log10_sigma_factor` for each ddid (26);
  - `log10_amp_factor` for each observation (6 free, 1 fixed);
  - (δRA, δDec) for each observation, as a phase ramp.
- Loss: the negative log-likelihood plus an entropy term (prior
  1e-4, total flux 0.253 Jy).
- Paper-run settings (commented in the Snakefile): λ_ent = 1e-9, FWHM 30 mas,
  AdamW at lr 1e-2, 10 epochs, batches of 40k visibilities, run twice from a
  checkpoint.

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

- Gridding compresses the data 373× on this grid. Even kept separate for
  each of the 26 ddids, there are only 1.1 × 10⁵ cells.
- The SBs have far more visibilities than cells: their uv tracks are short
  compared with the 40 kλ cell. That is exactly where per-ddid gridding pays
  off.
- The cost estimate for `MultipoleDisk` drops. It is about 1.1 × 10⁵ cells
  × 300 nodes × 9 orders, which is 3 × 10⁸ Bessel evaluations per
  gradient. That is laptop-scale for a single evaluation, though full fits
  and evidence sweeps still go to OzSTAR.
- The 5.12″ field gives only 2× padding of a disk of roughly 2–2.5″, and
  the CLEAN mask (5.5″ × 3.5″) is wider than the field. The FFT model is
  periodic with period FOV, so the outer radius must be taken from
  DSHARP II/Table 1 and the grid sized from it before P2. The cell-centre
  test in P2 should also try a 2048² grid.
- The DSHARP MS keeps up to 16 channels per window. LB1 is 74% of the
  visibilities and 98% of the cells.

The counting script is a scratch prototype. P1 should turn it into
`alma_worker.py`.

## Design

### Ingest stays outside virgil

`src/external_bridge/alma_worker.py` runs in its own venv
(`scripts/setup_external.sh`) with `casatools`, `casadata` and `visread`.

- It writes one `.npz` per ddid: u, v in wavelengths, Re, Im, weight,
  frequency, obsid, ddid.
- Its conventions go in `docs/alma_conventions.md` first: conjugation, the
  sign of the u axis, polarisation averaging, and weight units.

Independent checks in `src/crosscheck/`:

- a DFT dirty image, compared with CASA `tclean` at `niter=0` on the same
  pixels;
- the weight scatter for each ddid, recomputed from the residual
  histogram. The residuals are against the DSHARP CLEAN model, so the check
  is independent of virgil.

### Grid per calibration group, then fit all the data at once

Within each ddid, average the visibilities with their weights onto the
uv-cell lattice of the image (Hermitian-folded). For Gaussian noise, the
cell average and summed weight are sufficient statistics for any model
evaluated at cell centres and with s fixed. That is an approximation, not
an identity: a visibility up to Δu/2 from its cell centre sees a phase error
of about π·x/FOV for emission at offset x. For x ≈ 1.7″ and FOV = 5.12″ that
is about 1 rad, and about 0.5 rad on a 2048² grid; padding does not bound
it. The acceptance criterion is check (b) under Verification: ungridded
minus gridded χ² below about 1 per degree of freedom for each ddid, from a
DFT at the raw uv. MPoL's own example 04 uses `NuFFT` on the loose
visibilities for this reason. If (b) fails, fall back to a NUFFT likelihood
on the SB ddids, or to a finer grid, before any science run.

The gridder must store, per cell, V̄ and Σw, and per group the raw N_vis and
Σw|V|². The s term needs the within-cell scatter and the raw count, which
carry almost all the information on s for the SB ddids (about 10⁴
visibilities per cell). The within-cell scatter also contains source
structure across the cell, so s is biased high at high SNR; compare it with
the CLEAN-model estimate (see Verification), not with itself.

What this buys over MPoL's batched SGD:

- 1.1 × 10⁵ cells in all instead of 4.2 × 10⁷ raw visibilities.
  The model visibilities cost one FFT per group.
- Full-batch L-BFGS with a deterministic optimum.
- Cheap nuisances:
  - the amplitude of each EB scales that group's model grid;
  - the offset of each EB is a phase ramp on the cell centres;
  - the σ factor of each ddid enters the likelihood in closed form. With
    variance σ² s² on each of Re and Im, −ln L = 2 N_vis ln s +
    Σ w|r|² / (2 s²) + const, which is minimised at s² = Σ w|r|² /
    (4 N_vis). N_vis is the raw visibility count, and the sum runs over raw
    visibilities.

NUFFT (removed in virgil issue #75, code in PR #71's history) returns only
as a CPU check of the ungridded residuals, not for optimisation.

New in virgil (one PR):

- a complex-visibility observable with a σ on every sample;
- `GriddedVisData`, a grid of averaged visibilities and weights for each
  group, with its likelihood;
- `CalibrationNuisance`, the amplitude, σ factor and offset of each group.

### MaxEnt

- `Image` at 1024² pixels and 5 mas, with optional Gaussian convolution
  matching MPoL's 30 mas, plus `MaxEntropy` with a flat prior and a small
  `TSV`. Fit with L-BFGS.
- λ is chosen by k-fold dartboard cross-validation, as in MPoL and
  Zawadzki et al. 2023. Folds are drawn on the union cell index (or on
  dartboard sectors), and every group's copy of a held-out cell is held out
  together; otherwise the same uv cell leaks into training through the
  neighbouring ddids and EBs, and CV favours the method that overfits. The
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

Cost on the gridded cells is 1.1 × 10⁵ cells (measured, see "Visibility
count") × 300 nodes × 9 orders, about 3 × 10⁸ Bessel evaluations per
gradient. A single evaluation fits on the laptop, but the fits and evidence
sweeps go to OzSTAR. The geometry can be fitted on the SB data first (fewer
than 3000 cells), then frozen for the full latent and hyperparameter fit.

Hyperparameters come from `log_evidence`, a Laplace approximation in
(2M+1)·n_r whitened latents: about 3,300 (M=6, n_r=256) to 6,800 (M=8,
n_r=400). A dense Hessian at 6,800² is fine, but budget the OzSTAR
evidence sweeps on the upper figure. The model is also scored on the same dartboard
cross-validation as MaxEnt.

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
   `analyze_scale_factors.py`.
6. The image convolved to the CLEAN beam, against the DSHARP FITS.

Expectation:

- MaxEnt wins wherever the structure is not polar-separable.
- The GP wins on cross-validation in the ringed outer disk, with calibrated
  uncertainties on ring and spiral parameters.

## Phases

| Phase | Where | Work |
|---|---|---|
| P0 | this note | References, data inventory, design, N_vis (done 2026-10-17). |
| P1 | virgil-validation PR | ALMA worker, download, export, dirty-image and weight-scatter checks; record N_vis. |
| P2 | virgil PR | Complex observable, `GriddedVisData`, `CalibrationNuisance`, tests against the DFT. |
| P3 | OzSTAR GPU | MPoL reference run (`sgd.py` and `analyze_scale_factors.py` in `.venv-mpol` through `mpol_worker.py`, conventions in `docs/mpol_conventions.md`), then MaxEnt plus a cross-validation sweep (sbatch array over λ and folds). |
| P4 | OzSTAR GPU | `MultipoleDisk`: geometry on SB, then the full fit and evidence over M and lengthscales. |
| P5 | virgil-validation PR | `docs/imaging/alma_imlup.md`, trust graph and ledger, findings. |

## Verification

- Two separate gridding checks:
  - (a) Algebra: the gridded likelihood equals a float64 DFT likelihood
    evaluated at the cell centres to 1e-10 relative, including the
    s-dependent constant, and the gradients agree. This tests the
    bookkeeping only.
  - (b) Approximation: the ungridded χ² minus the gridded χ² at the fitted
    model, from an independent DFT at the raw uv (in `src/crosscheck/`), is
    below about 1 per degree of freedom for each ddid, with SB and LB
    reported separately. The subsample is drawn so that cells hold several
    visibilities each; a random 10⁴ draw from 4.2 × 10⁷ puts about one
    visibility in most LB cells and hides the within-cell term.
- The dirty image matches CASA `tclean niter=0` (with `weighting="natural"`
  and the N/2 phase-centre pixel convention pinned) to a tolerance set by
  the tclean gridding kernel, not by the noise: both images are linear in
  the same data and share its noise.
- A control that must fail: inject an offset point source (or an asymmetric
  pair) with `simobserve`/`ft` into a copy of the MS. Check its position
  from both tclean and the crosscheck DFT, in RA and Dec signs. The
  conjugated mapping is registered as `kind="control"`. IM Lup's rings and
  m=2 spiral are invariant under a 180° rotation, so a conjugation bug
  would barely change its dirty image; only the injected source exposes it.
- Fitted σ factors agree with the scatter estimate to about 5%, where the
  residuals are taken against the DSHARP CLEAN model (the `.model` from
  `tclean`, or the released FITS degridded by the crosscheck DFT), never
  against a virgil fit.
- Both MPoL's ingest and ours go through visread, so a visread convention
  error would agree on both sides. The tclean check with the injected
  source is the only root that reads the MS independently.
- Injection-recovery: a synthetic ring-plus-spiral disk on the real uv
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

Still to do:

- Take IM Lup's distance, inclination and PA from DSHARP I Table 1, and its
  pitch angles from DSHARP III. Only the abstracts have been read so far.
- Find a refereed source for the multi-EB weight-scatter method. ADS found
  none by Czekala et al. for 2023–2026. It may be in the MPoL paper or in
  exoALMA II (Loomis et al. 2025, arXiv:2504.19870); not yet checked.
