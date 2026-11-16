# DSHARP discs and HL Tau against the public ALMA images

virgil's ALMA reconstruction (the MPoL objective at our optimiser, with per-observation gains, offsets and a per-spectral-window error scale)
is run on five discs of the DSHARP sample and on HL Tau in two bands: IM Lup, HL Tau (Band 6 and Band 7), HD 163296, RU Lup and AS 209.
The comparison is with the **public ALMA images**: the DSHARP data-release fiducial images (Andrews et al. 2018) and the ALMA HL Tau release images.
MPoL is not used here.

## Data

- **DSHARP discs.** The released continuum measurement sets already carry the self-calibration (`CORRECTED_DATA`); the fit reads that column.
- **HL Tau.** The released calibrated measurement sets are combined with the released self-calibration tables (`pcal2`, `apcal`) and the flags of the
  release's imaging scripts, then split, so the fit reads the self-calibrated data. Band 6 and Band 7 are both run, and `dsharp_compare.py preview`
  puts our image next to the ALMA image for each band.
- **Provenance.** Sizes are checked on download and the SHA-256 of each tarball is recorded on first fetch and verified after (`data/dsharp/checksums.json`).
  See `DATA_ATTRIBUTION.md`.

## Method

`scripts/dsharp_fit.py` builds the same problem for every target: a pixel image with a 30 mas taper and the MPoL entropy, with the reference observation
(largest summed weight) fixing the gain and offset gauge, and the start flux estimated from the shortest baselines. One all-data stage is run to the C8 convergence
criterion and scored at eps 1e-12.

`scripts/dsharp_compare.py compare` convolves our image (Jy arcsec⁻²) with the released clean beam, multiplies by the beam area to get Jy beam⁻¹, regrids it
onto the released image grid, and reports the difference statistics inside the central 2″. The panels share one arcsinh stretch with the ALMA image.
`crosscheck.dsharp` tests the convolution against an analytic Gaussian and a shifted-image control.

## Results

Not yet run on the cluster. The job is `dsharp` in ozstar_scripts; the plots and statistics will be added here once the fits are back.
