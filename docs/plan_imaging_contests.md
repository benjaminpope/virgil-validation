# Plan: the interferometric imaging contests

## Why

Every two years since 2004 the optical/IR interferometry community has held
a blind image-reconstruction contest at SPIE Astronomical Telescopes +
Instrumentation. It started as the "Interferometry Imaging Beauty Contest"
and is now the "Optical Interferometry Imaging Contest" (I2C). The
organisers simulate data from a truth image that the contestants do not
see. Teams reconstruct images with their own codes (BSMEM, MiRA, WISARD,
MACIM, SQUEEZE, IRBis and others), and the organisers score the results
against the truth.

For virgil these contests are the natural external root for **image
reconstruction**, which the trust graph lists as unchecked:
`virgil.imaging.TSV`, `TV`, `MaxEntropy`, `l_curve` and the `Image` fits
behind them. Three things make them a good root:

- **Independent data.** The OIFITS files were written by other people's
  simulators: OYSTER (Hummel), ASPRO (JMMC), Cotton's CHARA simulator, and
  Aspro2 for the recent editions. They carry real-world conventions and
  quirks.
- **A published score to compare with.** Each paper reports the winning
  reconstructions, and often a full score table, so virgil can be placed
  among the established codes.
- **A truth, sometimes.** Where the truth image or model is available,
  virgil's image can be scored with the contest's own metric.

## The contests

The ADS search was run on 2026-10-04. Details for each contest, including
URLs, checksums, organisers and winners, are in `contests/manifest.yml`.
`scripts/fetch_contests.py` downloads the data, about 19 MB, into
`~/data/imaging_contests`. The data are never committed.

| Year | Paper | Data | Instrument | Truth | Winner |
|---|---|---|---|---|---|
| 2004 | [2004SPIE.5491..886L](https://ui.adsabs.harvard.edu/abs/2004SPIE.5491..886L) | ✅ JMMC | NPOI 6T at 550 nm, simulated | data2 is analytic (paper); data1 from Tuthill | BSMEM |
| 2006 | [2006SPIE.6268E..1UL](https://ui.adsabs.harvard.edu/abs/2006SPIE.6268E..1UL) | ✅ JMMC (needs a virgil reader fix, see C0) | AMBER on the UTs, J/H/K, 48 channels, grey | Chesneau's disk model, not posted | BSMEM |
| 2008 | [2008SPIE.7013E..1NC](https://ui.adsabs.harvard.edu/abs/2008SPIE.7013E..1NC) | ✅ JMMC | CHARA 6T, J/H/K, model differs per band | not posted | MiRA |
| 2010 | [2010SPIE.7734E..2NM](https://ui.adsabs.harvard.edu/abs/2010SPIE.7734E..2NM) | ✅ JMMC | AMBER 3T on ATs; Low HK (20 channels) and Med H (512 channels, 94k V²) | Chiavassa's CO5BOLD model, not posted | BSMEM |
| 2012 | [2012SPIE.8445E..1EB](https://ui.adsabs.harvard.edu/abs/2012SPIE.8445E..1EB) | ❌ lost with OLBIN | CHARA/MIRC-6T, H band | was posted, lost | MACIM |
| 2014 | [2014SPIE.9146E..1QM](https://ui.adsabs.harvard.edu/abs/2014SPIE.9146E..1QM) | ❌ raw frames only (60.A-9237(A)) | PIONIER, **real** data on VY CMa and R Car | none (real data) | BSMEM (Sanchez-Bermudez) |
| 2016 | [2016SPIE.9907E..1DS](https://ui.adsabs.harvard.edu/abs/2016SPIE.9907E..1DS) | ❌ not archived | GRAVITY and MATISSE, chromatic | not posted | IRBis? (Hofmann) |
| 2018 | [2018SPIE10701E..1UM](https://ui.adsabs.harvard.edu/abs/2018SPIE10701E..1UM) | ❌ not hosted | CHARA and PIONIER, grey | analytic, parameters in the paper | SQUEEZE then BSMEM |
| 2022 | [2022SPIE12183E..1GS](https://ui.adsabs.harvard.edu/abs/2022SPIE12183E..1GS) | ✅ OiDB | GRAVITY K (11 channels) and JWST/NIRISS AMI | not posted | BSMEM (Young) |
| 2024 | [2024SPIE13095E..14M](https://ui.adsabs.harvard.edu/abs/2024SPIE13095E..14M) | ✅ OiDB | PIONIER, GRAVITY, MATISSE L and N; chromatic, with calibration biases | not public (ImageMetrics) | MiRA (Drevon) |
| 2026 | Proc. SPIE 14148, 1414811 | not yet | ? | ? | ? |

Ben has written to Fabien Baron (2012), Joel Sanchez-Bermudez (2016, and
the 2014 and 2022 truth) and Antoine Mérand (2018) to ask for the missing
data. The 2024 truth images are with Florentin Millour and Ferréol Soulez.

## What the stages test

The stages run in order of what they need from virgil: reading the files,
then grey imaging, then chromatic imaging, then scoring against truths.

### C0. Reading the files (done)

Every contest file is read by `virgil.oidata.OIData`. 24 of 29 data files
read as they are.

- **The 2006 files fail, and this is a virgil bug.** Their closure phases
  sit under a different `INSNAME` (`AMBER-LR_TR01_OB01`) from their V²
  (`AMBER-LR_OB01`). The two `OI_WAVELENGTH` tables are identical, which
  the OIFITS standard allows. virgil pairs triangles with baselines only
  under the same `INSNAME`, and its error message says the wavelengths
  differ, which is wrong. [virgil#167](https://github.com/benjaminpope/virgil/pull/167) fixes it: it matches by wavelength table, and also supports reversed T3 legs (the 2006 V² store (2,0) where the triangle needs (0,2)), which virgil also did not support. Once it merges, the
  bug will be recorded as a ledger entry ruled `virgil`.

### C1. The organisers' test binaries (done)

Before each contest the organisers released a binary with published
parameters. These tests fit three of them (`tests/test_contest_binaries.py`,
root `literature`). Each fit starts both from the published position and
from its mirror image. The published side must win by more than 5σ, so a
sign error in East, PA or closure phase would fail the test.

| Contest | Separation (mas) | PA (°) | faint/bright | Δχ², mirror − published | reduced χ² |
|---|---|---|---|---|---|
| 2004 (OYSTER, NPOI) | 21.23 vs 21.2 | 341.5 vs 341.6 | 0.173 vs 0.174 | 649 | 3.9 |
| 2008 (CHARA, J/H/K) | 4.995 vs 5.0 | 30.00 vs 30 | 0.107 vs 0.112 | 4.0×10⁵ | 3.9 |
| 2010 (ASPRO, AMBER) | 18.000 vs 18 | 128.00 vs 128 | 0.1000 vs 0.1 | 2.0×10⁷ | 0.45 |

Two open questions remain. Reduced χ² near 4 in 2004 and 2008 may come
from limb-darkened components or from the simulators' noise models; the
2008 binary is also chromatic between sub-bands. LM did not converge in
1000 steps for 2008 and 2010, although the answers are right. The 2006
binary (`2006-double.fits`) has no published parameters; it will be fitted
once C0 is fixed and compared with the image.

### C2. Grey reconstructions (in progress)

This stage reconstructs a grey image from each contest dataset that is
grey or nearly so:

- 2004 data1 and data2
- 2006 (once C0 is fixed)
- 2008, each band separately
- 2010 Low HK, each band treated as grey
- 2022 object 1 (GRAVITY) and object 2 (AMI)

virgil's standard recipe is `starting_image` followed by `fit` with `TSV`
or `MaxEntropy`, choosing the weight by `l_curve` or the discrepancy
principle; the GP prior (`fields.GaussianField`) is the alternative. Where
a star is clearly unresolved, the analytic-star composite (imaging part 4)
is used.

The outputs are:

1. images and residual maps;
2. reduced χ² for V² and for closure phases separately;
3. a side-by-side comparison with the published winning images.

Without truths the comparison is qualitative: are the structures the
papers describe present? Examples are the LkHα 101 shell, Chesneau's disk,
the AGB star and the AGN, and the companion in 2010. That gives evidence
of kind `check`, root `literature`, tier `C`. Where a truth is analytic,
scoring is quantitative: 2004 data2 is rebuilt from the paper, then
convolved and aligned as in the paper's metric.

Compute: these are 64² to 128² images on 10² to 10⁴ points. They run on
OzSTAR (`ozstar_scripts`, CPU nodes), one job per dataset, not on the
laptop. Outputs go back to `results/` and then into the evidence records.

`scripts/contest_images.py` does this, with one task per dataset (`--list`
shows the 16). The OzSTAR job is `contest_imaging` in `ozstar_scripts`.
Its results come back to `~/data/imaging_contests/results/<virgil commit>/`.
A smoke run on 2004 data1 (three weights, 50 steps each) already shows an
asymmetric shell 6 mas across, at χ²/N = 3.6.

### C3. Chromatic data

The 2010 Med H data (512 channels, with differential phases), all of 2024,
and the 2016 data (if they arrive) have structure that changes with
wavelength. This is where virgil has a known gap.

- `Image` is grey: its shape cannot change with wavelength. Spectra only
  weight components (design note `chromatic_sources.md`).
- The SPARCO-style composite already works: a parametric star with a
  `PowerLaw` spectrum, plus a grey environment image with its own spectral
  index. That suits 2024 Obj2 (a disk around a Herbig star) and the 2010
  supergiant with its companion.
- Truly chromatic morphology, such as the 2024 Wolf-Rayet spiral across
  H to N bands, needs one image per channel or per band, regularised
  together across wavelength (as MiRA-3D and the 2016 entries do). The
  first step is independent images per band, then joint fits. If joint
  fits are needed, that becomes a virgil design issue, not a workaround
  here.
- 2010 Med H has 94k V² against an image DFT, which is the case the
  removed NUFFT backend was for (virgil issue #75). Bin it in wavelength
  first; whether the full set is ever needed is a decision for later.

### C4. Scoring against truths

When truth images arrive, this stage scores virgil's images with each
contest's own metric:

- the σ/peak sums (2004);
- RMS pixel differences after convolution (2008, 2010, 2012);
- the 2024 L1 distance minimised over shift and flux scale per wavelength,
  from [JMMC ImageMetrics](https://github.com/JMMC-OpenDev/ImageMetrics),
  used as an external package in `src/external_bridge`, never vendored.

Each metric is also written independently in `src/crosscheck` from the
paper's definition and checked against the published score tables, so the
metric itself is validated first. Then virgil's scores are placed in each
year's table. This is root `literature` (the published scores), and also
`golden:ImageMetrics`.

### C5. 2014: real data

If Joel or John Monnier have the reduced OIFITS, virgil's images of VY CMa
and R Car can be compared with the published crowd-sourced median images.
Otherwise the raw frames would need reducing with pndrs. That is a larger
job, and it would also test calibration, so it is out of scope until the
other stages are done.

## Trust graph

A new pipeline, `contest-imaging` (published blind data → `Image` fit with
regularisers → image scored against the published entries), is added as
`planned`. Once C2 and C4 produce evidence, it validates the imaging nodes
against `literature`, the second root they need alongside eht-imaging or
MPoL (`rml-imaging`).

## Open decisions for Ben

1. **External comparison codes.** Running MiRA (Julia) or SQUEEZE ourselves
   on the same files would give a same-data comparison stronger than the
   papers' figures. That would add packages to the root pool. Should it
   wait for C2's results?
2. **2024 truths.** Should we ask Millour or Soulez for the 2024 truth
   images? With ImageMetrics, that is the quickest route to a quantitative
   score.
3. **Chromatic imaging in virgil.** If C3 shows that per-band images are
   not enough, should a cross-wavelength regulariser go into virgil (a
   design note first)?
