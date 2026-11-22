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

## Method

`scripts/dsharp_fit.py` builds the same problem for every target: a pixel image with a 30 mas taper and the MPoL entropy, with the reference observation
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

Not yet run on the cluster. The job is `dsharp` in ozstar_scripts; the plots and statistics will be added here once the fits are back.
