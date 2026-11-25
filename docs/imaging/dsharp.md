# DSHARP discs and HL Tau against the public ALMA images

virgil's ALMA reconstruction (the MPoL objective at our optimiser, with per-observation gains, offsets and a per-spectral-window error scale)
is run on five discs of the DSHARP sample and on HL Tau in two bands: IM Lup, HL Tau (Band 6 and Band 7), HD 163296, RU Lup and AS 209.
The comparison is with the **public ALMA images**: the DSHARP data-release fiducial images (Andrews et al. 2018) and the ALMA HL Tau release images.
MPoL is not used here.

## Data

- **DSHARP discs.** The released continuum measurement sets already carry the self-calibration. The prep stage reads `CORRECTED_DATA` where the set has it (IM Lup's) and otherwise `DATA`, and records the column it used in `datacolumn.txt`.
- **HL Tau.** The released calibrated measurement sets are combined with the released self-calibration tables (`pcal2`, `apcal`) and the flags of the
  release's imaging scripts, then split, so the fit reads the self-calibrated data. Band 6 and Band 7 are both run, and `dsharp_compare.py preview`
  puts our image next to the ALMA image for each band.
- **Provenance.** Sizes are checked on download and the SHA-256 of each tarball is recorded on first fetch and verified after (`data/dsharp/checksums.json`).
  See `DATA_ATTRIBUTION.md`.

## The production driver

`scripts/dsharp_fit.py` is the corrected driver of the convergence plan (`reports/dsharp-convergence-plan.md` of virgil-paper, revision 2). The first run, whose
method is described under "The first run" below, did not converge: pixels stuck at the floor under its 1″ Gaussian start, one runaway offset (RU Lup at 50 mas),
`chi2="auto"` switching engines silently, and HL Tau's weights about 2000× off. The production model is:

- `virgil.LinearImage` in √I pixels, the pixels owning the flux; classic MEM (`MaxEntropy` on the unblurred pixels, flat default); a **flat start**; the kernel only in the ICF arm.
- Fitted per-observation gains and offsets (spread, ±20 mas box); error scales fixed at 1 on weights rescaled by the model-free pair scatter ŝ² of each window
  (per window × baseline bin where ŝ² varies by more than 20% across bins); spectral index fixed at the model-free value when it differs from 0 by more than 2σ, else 0.
- `chi2="toeplitz"` named explicitly and the engine read back from the term, never `"auto"`.

`dsharp_fit.py probe` (stage S1) fits no image. It writes `probe.json`: ŝ² per window and bin, the model-free spectral index, each observation's model-free shift
(the phase slope of the cross-power with the reference observation) against a short joint image-and-offsets fit (not a held flat image, which has no offset gradient), the pre-alignment verdict (agreement within 5 mas: pre-align by the
model-free shift and fit the residual in ±20 mas; otherwise no pre-alignment; a disc whose offsets still leave the box is dropped and reported, never run direct), the
loss and gradient of the Toeplitz and direct engines on the production model at a perturbed point, and the timings. Gate G1 is printed from them. The run modes `s3`
(certification pilot), `s4` (diagnosis: `--start flat|gaussian`, `--arm mem|icf`) and `s5` (production, with the image and nuisance restarts) read `probe.json`,
fit the λ ladder (w = 10⁶ to 3×10³, strongest first, each rung warm-started), the tuning folds (fold 0 is the test fold and tunes nothing) and pick λ by the one-standard-error
rule. A rung that stops on a cap is a failed rung. Every round checkpoints and `result.json` records both commits. The optimiser sits behind `OPTIMISERS` (default virgil's L-BFGS
until stage S2 picks), and certification is PR-B's `alma_certify` (`--certify module:function`); without it `certified` is `null`. The comparison page (`dsharp_report.py`)
still reads the first run's `result.json`; it is rebuilt for these results by the comparison PR.

## The first run

The first run's `dsharp_fit.py` (virgil `09b2685`, vv `0542235`; git history of this file) built the same problem for every target: a pixel image with a 30 mas taper and the MPoL entropy, with the reference observation
(largest summed weight) fixing the gain and offset gauge, and the start flux estimated from the shortest baselines. The start is a 1″ Gaussian raised to MPoL's entropy
floor (its outer pixels would otherwise be too dark for the optimiser to revive), and MPoL's F_tot is that start flux rather than IM Lup's 0.253 Jy, so λ_M means the same
on every disc and band (with IM Lup's value the entropy weight would grow with the flux, about 8× on HL Tau Band 7). One all-data stage is run to the C8 convergence
criterion, checkpointed each round, and scored at eps 1e-12. `result.json` records the dead-pixel and at-bound reports and the virgil and virgil-validation commits.

The model's visibilities are those of the image convolved with the 30 mas Gaussian, so `image.fits` is that **tapered** image, the sky the fit sees (`image_base.npy` is
the image before the kernel). The 30 mas taper is wider than Band 7's 19 mas minor beam axis, so the Band 7 image is smoother than CLEAN's along that axis; the radial profile
panel of the comparison shows it.

`scripts/dsharp_compare.py compare` convolves our image (Jy arcsec⁻²) with the released clean beam, multiplies by the beam area to get Jy beam⁻¹, regrids it
onto the released image grid, and reports the difference statistics inside the central 2″, using only pixels inside our image's footprint and where both images are finite.
virgil fits the apparent sky, so the primary-beam-corrected Band 7 reference is multiplied by the ALMA 12 m primary beam (a Gaussian of FWHM 1.13 λ/D, about 17″ at 344 GHz) first; `compare.json` records `pbcor_applied`. The panels share one arcsinh stretch with the ALMA image.
`crosscheck.dsharp` tests the convolution against an analytic Gaussian and a shifted-image control.

## Results

All six fits ran on OzSTAR (virgil `09b2685`, vv `0542235`; 20 000 steps each, 5 mas pixels, 30 mas taper). **None converged**: every run stopped at its step limit without meeting the C8 criterion, so the numbers below are the optimiser state at that point.

Central 2 arcsec, virgil convolved with the released beam against the public image (Band 7 reference primary-beam corrected, multiplied by the primary beam first; Band 6 and the DSHARP images compared as released):

| Target | correlation | flux ratio | peak ratio | rms(diff) / peak |
|---|---|---|---|---|
| IM Lup | 0.994 | 0.945 | 0.983 | 0.0026 |
| HL Tau B6 | 0.995 | 0.991 | 0.960 | 0.0031 |
| HL Tau B7 | 0.990 | 0.983 | 0.915 | 0.0046 |
| HD 163296 | 0.996 | 0.977 | 0.954 | 0.0070 |
| RU Lup | 0.947 | 1.025 | 0.699 | 0.0073 |
| AS 209 | 0.980 | 0.997 | 1.155 | 0.0115 |

RU Lup and AS 209 differ most, in the central pixels (RU Lup's hidden image also has dead pixels, so treat it as an optimiser artefact). The released HL Tau weights are about 2000 times too large in variance (fitted error scale about 47), so the HL Tau raw chi² per 2N (about 2300) is not a fit quality; with the free scale it is 1.00. The data term was requested as direct but virgil's `chi2="auto"` may use the dirty-beam (Toeplitz) form where it is exact, as on the single-observation HL Tau sets; `result.json` now records `auto` rather than `direct`, and the earlier HL Tau fits ran with that automatic choice. **None of the six converged.** The side-by-side page (`scripts/dsharp_report.py`) shows images, profiles, metadata and these caveats.
