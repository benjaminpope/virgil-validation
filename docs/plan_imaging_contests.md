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
| 2018 | [2018SPIE10701E..1UM](https://ui.adsabs.harvard.edu/abs/2018SPIE10701E..1UM) | ✅ organiser's page (eso.org/~amerand) | CHARA and PIONIER, grey | analytic, parameters in the paper | SQUEEZE then BSMEM |
| 2022 | [2022SPIE12183E..1GS](https://ui.adsabs.harvard.edu/abs/2022SPIE12183E..1GS) | ✅ OiDB | GRAVITY K (11 channels) and JWST/NIRISS AMI | not posted | BSMEM (Young) |
| 2024 | [2024SPIE13095E..14M](https://ui.adsabs.harvard.edu/abs/2024SPIE13095E..14M) | ✅ OiDB | PIONIER, GRAVITY, MATISSE L and N; chromatic, with calibration biases | not public (ImageMetrics) | MiRA (Drevon) |
| 2026 | Proc. SPIE 14148, 1414811 | not yet | ? | ? | ? |

Ben has written to Fabien Baron (2012), Joel Sanchez-Bermudez (2016, and
the 2014 and 2022 truth) and Antoine Mérand (2018) to ask for the missing
data. The 2018 data have since been found, still served on Mérand's ESO
page, and are in the manifest; only the 2018 truth model is needed from him. The 2024 truth images are with Florentin Millour and Ferréol Soulez.

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
  differ, which is wrong. [virgil#167](https://github.com/benjaminpope/virgil/pull/167) fixes it: it matches by wavelength table, and also supports reversed T3 legs (the 2006 V² store (2,0) where the triangle needs (0,2)), which virgil also did not support. It is finding F10 in the ledger and
  README, fixed by virgil#167 (merged 2026-10-04) and now checked by
  `test_2006_files_with_separate_t3_insname_are_read`. CI fetches the 2006 files so the test runs.

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

First results (2026-10-04, OzSTAR jobs 17998336 and 17998363, 2006):

- **2006 failed in the baseline run**, identically on virgil `dd06e9f` and
  `ec4cf51`. The fit never moved: χ² was the same at all 13 weights, χ²/N
  was 22–134 per night, and the image was stripes.
- **The cause is the data, not the fitter.** The target is almost entirely
  resolved on the UT baselines: V² is at most 0.03 and typically 1e-3, with
  errors of 1e-4. The truth model is 105 mas across, while λ/B_min is at
  most 18 mas. A unit-flux image inside 18 mas cannot produce V² ≈ 1e-3.
- **The fix is a resolved component.** `--halo` adds a fully resolved
  `Resolved` component with a free flux. In a smoke run its flux came out
  about 2 (relative to the image), and χ² then fell with weight. Most
  datasets have short-baseline V² below 1 (0.16 for 2024 Obj1, 0.34 for
  2010, 0.5–0.6 for 2004 and 2008), so the halo variant will be run for
  every task after the baseline run, with labels suffixed `_halo`.
- **2022 AMI has extreme signal-to-noise**: closure-phase errors of about
  0.001–0.002° and V² errors of about 1e-4. χ²/N ≈ 1 may be out of reach
  for a grey image.

Baseline run (2026-10-04, job 17998393, virgil `ec4cf51`, all 15 tasks
except 2022 AMI, which is still running):

| Dataset | χ²/N at the chosen weight | Result |
|---|---|---|
| 2004 data1 (LkHα 101 model) | 1.83 | an asymmetric shell with a bright north-west rim |
| 2004 data2 (spotted star and companion) | 179 | **failed**: virgil's default field (±6.9 mas, from λ/B_min) excludes the companion 10 mas East |
| 2006 | 22–134 | **failed**: flux over-resolved (see above) |
| 2008 AGB, J/H/K | 1.2–1.4 at the corner | ring- or shell-like; the discrepancy weight over-fits in J and H |
| 2008 AGN, J/H/K | 1.3–2.9 | an elongated core with extended features; K is worst |
| 2010 Low HK | 13 353 | **failed**: the image never moved; V² errors are all 1e-4, and the source is chromatic across H and K |
| 2022 GRAVITY, 2024 Obj2 GRAVITY | NaN | **script bug**: the star-flux prior was Uniform(0, 1), but the start was 5.8 and 1.1 |
| 2024 Obj1 PIONIER / GRAVITY | 1.23 / 1.15 | a compact bar with arc fragments (Obj1 GRAVITY has most of its closure phases flagged) |
| 2024 Obj2 PIONIER | 2.0 | — |

Many L-BFGS fits stop on "line search ran out of float64 precision"
without converging, and χ² sometimes rises from one weight to the next.
Both are worth a look once the images are compared with the papers.

Fixes, smoke-tested locally on virgil `ec4cf51`:

- 2004 data2 starts from a flat image over the published 24 mas field
  (`FORCE_FIELD`); χ²/N fell from 4693 to 447 in 50 steps.
- The star-flux prior now reaches max(100, 10× the start).
- The L-curve plot falls back to linear axes, and the text summary is
  written before plotting.
- 2010 with `--halo` reaches χ²/N ≈ 5700 in a smoke run, against 13 400
  without; it really needs C3 (per-band images, with the organisers' SEDs).

Second run (2026-10-04/05, jobs 17999358 for the three fixed tasks and
17999361 for all 16 with `--halo`, virgil `1057928`, cluster JAX 0.11.2):

| Dataset | χ²/N | Result |
|---|---|---|
| 2004 data2 (field forced to 24 mas) | 29 (was 179) | **structure right**: an elliptical spotted star and a compact companion about 10 mas East of it, as in the published model; χ² is still high |
| 2022 AMI (+halo) | 1.45 at the discrepancy weight | **works**: a bright arc about 150 mas from the star, curving round the north and east |
| 2006 (+halo) | 11.6 | **still failing**: halo flux about 1.4 (V² ≈ 1e-3 needs ~30); edge flux 15%; mostly empty image |
| 2022 GRAVITY, 2024 Obj2 GRAVITY (star) | 462 and 296 at every weight | **stuck**: no NaN now, but χ² is the same at all 13 weights, and the strongest-weight image is a scatter of single pixels, which is not a MaxEnt solution |
| 2010 (+halo) | 5750 at every weight | stuck; needs C3 |
| others with `--halo` | unchanged | the halo flux fits to ~0 everywhere except 2006 (1.4) and 2008 AGN K (0.02) |

**The stalls.** On the cluster, 6–13 of the 13 L-BFGS fits per task stop
with "line search ran out of float64 precision". Wherever χ² is flat
across all weights, the first fit stalled and the warm start carried that
on to every weight. The cluster's `virgil` env has JAX 0.11.2, against
0.9.1 locally and in CI, and another session found gradient failures under
0.11.2. None of the logs shows its error signatures
(`RuntimeProgramInputMismatch`, "Expected cotangent"), but the stalls
could still be numerical.

The laptop's virgil-validation environment also has JAX 0.11.2 (the latest
on PyPI), so cluster and laptop agree. The aim is correct results on the
latest JAX, so the stalls are being debugged there, not by going back to an
older version. Only virgil's own lockfile pins 0.9.1, which means virgil's
CI never sees the JAX its users install.

`scripts/diagnose_stall.py` takes each stalled fit at its strongest weight.
Its case numbers change as fits are fixed. In the first diagnosis below
(job 18019122) they were 0 = 2022 GRAVITY, 1 = 2024 Obj2 GRAVITY,
2 = 2010 and 3 = 2006 with halo. They are now 0 = 2008 AGB J (task 3),
1 = 2008 AGN K (task 8), 2 = 2022 GRAVITY (task 10) and 3 = 2024 Obj2
GRAVITY (task 15), the fits still stuck on virgil 98eaf86, after #174.
It records:

- snapshots after 10 to 20 000 steps (χ², effective and dead pixels);
- JAX against finite-difference gradients at the stall, and a loss scan
  along −grad;
- remedies: a step cap of 0.2 and 0.05, Adam, and the image flux held fixed.

A 10-step local smoke run on 2022 GRAVITY already shows two things.
Gradients agree with finite differences to 1e-10 under JAX 0.11.2 (early
in the fit). Holding `env.flux` fixed reaches χ²/N 1623 in 10 steps,
against 5091 with it free, which points at the flux parameter's coupling
to the pixels. The full runs go to OzSTAR (`--diagnose`).

**First diagnosis (job 18019122, virgil `760c720`, JAX 0.11.2; the old case numbering): finding F11.**
The two worst stalls sit on a discontinuity in virgil's likelihood, not a
JAX problem.

- In 2022 GRAVITY (case 0) and 2010 (case 2), JAX gradients and central
  finite differences disagree by 10²–10⁴ at the stall. The
  finite-difference value scales as 1/ε, so the loss jumps by a fixed
  amount: Δχ² ≈ 2 × 426 for GRAVITY, ≈ 2 × 6000 for 2010.
- The loss rises along −grad at every step size, so the line search has
  nowhere to go.
- The cause is virgil's correlated closure-phase likelihood (4+ telescopes):
  wrapped residuals → chords 2 sin(Δ/2) → whitened together. A chord flips
  sign across ±π, and the cross terms make χ² jump. In a smooth sweep on our
  own 4T file the largest step is 860× the median, against 48× with three
  telescopes (`tests/test_closure_continuity.py`, strict xfail).
- A virgil PR making that likelihood continuous is being written (Sonnet
  agent).
- 2024 Obj2 GRAVITY (case 1) is different: gradients agree with finite
  differences to 1e-7, and descent continues. It is converging slowly
  (χ²/N 310 after 20 000 L-BFGS steps; Adam reaches 254). That is a
  question of initialisation and conditioning, not a bug.
- Case 3 (2006 with halo) crashed on a bug in the diagnostic (multi-night
  data), now fixed.

**Initialisation.** Every task starts from `starting_image(start="moments")`,
a Gaussian sized by a quick parametric fit. A dirty-image start is
impossible without absolute phases. A residual near ±π, which is what
triggers F11, needs a poor start. Next: use everything the contestants had
before submitting (clue images, readmes, rules, suggested fields and pixel
scales, SEDs, test binaries) to initialise and set priors. A research agent
is compiling that per contest.

### Initialisation: only what contestants had

Each task's settings in `scripts/contest_images.py` (`TASKS`) come only from
what the contests published before the deadline (manifest `presubmission`
entries; pre-submission files are in `~/data/imaging_contests/<year>/pre_submission/`).
Truths, model parameters and target identities revealed in the papers are
not used.

An earlier version broke this rule. It took the 2004 fields (12 and 24 mas)
from the papers; those values have been removed.

| Contest | Given before the deadline | Used as |
|---|---|---|
| 2004 | nothing (data released blind; OI_TARGET names are decoys) | field chosen from the data |
| 2006 | clue image (the model at 10 mas resolution, 106 mas field); the rules allow the field of view only | field 106 mas, flat start; no morphology prior |
| 2008 | "an AGB star" / "an AGN"; "tapered with a 15 mas FWHM Gaussian" | field 30 mas; that Gaussian as the start and the MaxEnt default image |
| 2010 | a bright source; SEDs; grey category judged as Low HK channels 1–10 and 11–20 | separate H and K images (the other channels flagged); SEDs kept for C3 |
| 2018 | "a young star's disk, with a planet" | analytic star plus image; data recovered from the organiser's live page |
| 2022 | nothing found; the GRAVITY OI_TARGET names the real star, probably by accident | treated as blind; the header is ignored. An analytic star is inferred from the data (an unresolved source dominates), which a contestant could also have seen |
| 2024 | "a hot star with an environment" / "a young star", suspected companion; uncalibrated OI_FLUX; cubes required | analytic star in both; grey images per instrument as a first look; cubes in C3 |

**Central stars** are common in these targets, so every dataset is also
imaged both with and without an analytic central star (Ben, 2026-10-05:
choosing between them from the data is not misusing information).
`--star on|off` overrides a task's default, and labels get
`_star`/`_nostar`. virgil's `log_evidence` applies only to Gaussian-field
images, so for these MaxEnt images the choice is made on the L-curves: the
χ² reached at equal entropy, and whether the star-free image builds a
compact central peak to imitate a star.

**Data-chosen fields** start from virgil's `starting_image`, whose field
is limited by λ/B_min. They grow ×1.5, at most three times, while that
lowers χ² by more than 10% in a 3000-step probe fit at w = 100.

Flux at the image edge is not a usable trigger: the support and the
centroid prior keep it off the edge even when the field is too small. On
2004 data2 the χ² rule picks 20.6 mas (χ²/N 228 → 38; growing again
gives 363), wide enough for the companion, without using the paper.

**Test binaries.** The 2006 test binary is now checked too (published
10 mas, PA 30°, Δm = 1, UD 3 and 1 mas). virgil recovers 10.000 mas,
29.99°, ratio 0.399 (10^-0.4 = 0.398), χ²/N 0.94. That file needs
virgil#167 (different INSNAMEs, reversed legs), so this also checks the
fix against the literature.

### C2b. CLEAN starts and Gaussian-process priors

Two further options in `scripts/contest_images.py`, combinable with `--star`
and `--halo`:

- **`--init clean`.** virgil's gradient CLEAN (`imaging.clean`, which works
  on V² and closure phases) runs on a grid of at most 65 pixels over the
  same field, relative to the analytic star if there is one. Its components,
  convolved with the beam, become the starting image, resampled to the fine
  grid. A start with the right layout keeps closure-phase residuals away
  from ±π (F11) and places far components, such as the 2004 companion or
  the 2010 one at about 84 mas. On 2004 data2, CLEAN puts 6% of the flux
  more than 5 mas East and 2% more than 5 mas West, matching the published
  layout. It fits much better than its inversion (χ² 35 395 vs 57 960).
- **`--prior gp`.** `GaussianField` log-brightness about the template (the
  moments or CLEAN image), fitted on a grid of σ ∈ {1, 2, 4} and
  ℓ ∈ {0.5, 1, 2} × beam minor axis. The fits are compared by
  `log_evidence`:
  - *Isotropic*: one correlation length.
  - *Anisotropic*: lengths (ℓ/√r, ℓ√r) along and across a position angle,
    via `Image(rotation_deg=PA)`. The axis ratio r and PA come from an
    elliptical-Gaussian fit to the data. The template is resampled onto the
    rotated grid, with the rotation's sign checked numerically against
    virgil's rendering.
  - Unlike the MaxEnt L-curves, the evidence also compares runs with and
    without a star on the same data.
  - Smoke runs: 2022 AMI prefers isotropic (log Z −198 vs −208); 2004
    data2 with a CLEAN start prefers anisotropic (−27 444 vs −27 451).
- **Grid cap.** Every grid is capped at 255 pixels, including
  `starting_image`'s own: 2018 asked for 386, and CLEAN's per-pixel setup
  cost scales as npix⁴ × N.

**Finding F12 (fixed in virgil#190).** Without a base scene, `clean` seeds
the central pixel, where |J e_p| = 0. On even grids that pixel is half a
pixel off the origin, so rounding left |J e_p|² ~ 1e-24 instead of
exactly 0. Its score g²/|J e|² won, and its step (−g over a zero
curvature) was infinite, so the next χ² was NaN. virgil#174 only changed
the rounding, which is why the bisection pointed at it. virgil#190 treats
pixels with |J e_p|² below (100 ε)² of the maximum as dead. The
contest script's workaround seed is removed.

Bisection on simulated 4T data found no NaN grids on #174's parent
(53dd4b5) and 18/100 on its merge (2b2df8c). The root cause above is the
seed pixel's near-zero |J e_p|; #174 changed only the rounding that
decided whether that pixel won.

### C2c. Methods from the winning entries (methods only, not results)

From the contest papers and the codes' method papers (BSMEM, MiRA, MACIM,
SQUEEZE, IRBis, SPARCO, PYRA/MYTHRA). Results are excluded. The papers
also describe their truth models, so **blindness is enforced by rules in
`contest_images.py`, not by what the author of a run happens to know.**
Every setting must be justified from the pre-submission information or
from the data.

Methods that recur among winners:

- **Iterate the default model** (BSMEM, every year it won). Start flat or
  from a Gaussian fitted to V², reconstruct, then smooth and threshold the
  image into a new MaxEnt default and reconstruct again. A fit that stalls
  above χ² = N signals a default that is too narrow.
- **Grey first, then per channel** (2010 and 2022 BSMEM; the 2024
  organisers). Make a grey image from binned channels, use it as the
  default for each channel, and combine the channels with a median or a
  thresholded mean, since artefacts appear in one channel only.
- **Ensembles with an L-curve window** (2012 random starts; 2022 and 2024
  Millour/Drevon, PYRA/MYTHRA):
  - randomise μ, pixel, field (1–2× the interferometric one), start and
    regulariser parameters;
  - keep the window just before the L-curve turnover and drop χ² outliers;
  - re-solve on a common grid, and add images only while χ² on V² *and* CP
    does not worsen;
  - output the mean and a per-pixel σ map.
- **Locate features first, then refine** (2018 winner: 10 SQUEEZE chains,
  mask compact sources on beam sidelobes, refine each with BSMEM,
  average). This is what our CLEAN start already does.
- **Parametric star plus image** (SPARCO; MACIM's unregularised point;
  2018 and 2022 entries). Choose the flux ratio by total cost or evidence
  on a grid, then free it. Use a `PowerLaw` spectral weight for chromatic
  data.
- **Model selection on residual diagnostics, not total χ² alone** (MACIM,
  MiRA 2012):
  - χ² on V² and CP separately, each near 1;
  - mean residual on short baselines near 0, so that long baselines are not
    overfitted at the expense of short ones.
- **Match the regulariser to the morphology** (2012 organisers; Renard et
  al. 2011 found TV best overall and compactness second). Compare TV,
  MaxEnt/TSV and StarletL1; features that change between them are suspect.
- **Systematics** (2014, 2018, 2024): cut low-SNR points, inflate errors,
  fit an over-resolved flux. Do not trust automatic hyperparameters
  (including Laplace evidence) when the data are biased, and cross-check
  with the L-curve.

Planned options for `contest_images.py`, in order:

1. Residual diagnostics in every summary: χ²/N for V² and CP separately,
   and the mean whitened residual on the shortest 20% of baselines.
2. `--default iterate`: one or two BSMEM-style default-model iterations.
3. `--regulariser tv|tsv|starlet` beside MaxEnt, for artefact comparison.
4. `--ensemble N`: randomised settings, L-curve-window selection, mean and
   σ map (PYRA-style); one OzSTAR array task per member, plus a combiner
   job.
5. C3: grey-then-per-channel with a median combine; SPARCO-style star with
   `PowerLaw`.

### C2d. The convergence diagnosis and the ensemble campaign

`docs/contest_convergence_diagnosis.md` separates the causes of the
non-converging fits: optimiser and conditioning, likelihood, regularisation
and hyperparameters, and initialisation. Initialisation and the image
parameterisation dominate. The campaign follows the winners' methods taken
together:
- CLEAN-started GP fits in an ensemble of 8 randomised starts per dataset
  (`--member`);
- star on and off alternating;
- evidence-chosen σ, ℓ and isotropy;
- an `error_scale` refit when the errors are off by more than 1.3x;
- `scripts/combine_ensemble.py`: χ² filter, mean and σ maps, and the star
  evidence.

It covers 22 datasets, including the 2024 MATISSE L and N bands.

### C2e. After the first campaign: what is left, and the next round

The campaign (virgil `8562a39`, MATISSE rerun on `3018a9f` after
virgil#237) gives χ²/N ≈ 0.7–1.1 for 2004 data1, 2006, all six 2008 sets,
2018 and both 2024 Obj1 sets (PIONIER and GRAVITY). Ensemble table:
`~/data/imaging_contests/results/ensemble_8562a39_3018a9f/ensemble.md`.
Five problems remain. Each plan below uses only what contestants had
before submitting.

**2004 data2 (χ²/N ≈ 21, error scale 4.7).**
- *What we see.*
  - V² dominate: 31 per point, against 5.8 for the closure phases.
  - The short-baseline V² residuals have a mean of −8σ.
  - The evidence's best σ (4) and ℓ (0.75 beam) both sit on the edges of
    the hyperparameter grid.
  - Members with the analytic point star are far worse (χ²/N ≈ 1300).
- *What the contestants knew.* The contest page listed the possible
  morphologies: a limb-darkened star with spots, a compact source with an
  envelope, or something more exotic.
- *Plan.*
  1. Widen the hyperparameter grid whenever the evidence's best point lies
     on an edge (σ up to 16, ℓ down to 0.25 beam), for every dataset.
  2. Add *resolved-star* members. The base becomes a `LimbDarkenedDisk`
     whose diameter is fitted (log-uniform from 0.1 beam to a quarter of
     the field) by `clean(base_priors=...)` and again in the GP fit,
     instead of the point star.
  3. Scan the field (1×, 2× and 4×), with the evidence choosing.

**2010 Low HK (χ²/N ≈ 2500–4000, error scale 52–67).**
- *What we see.* Every V² error is 1e-4, so any model mismatch dominates
  χ².
- *What the contestants knew.* The grey category was judged as three
  images: Med H, Low HK channels 1–10, and Low HK channels 11–20. Our two
  Low HK tasks (split at 1.9 µm) roughly match the last two, so colour within a half-band is
  probably not the main cause.
- *Plan.*
  1. Scan the field (1×, 2× and 4×, star on and off) to test for flux
     outside the 69 mas field.
  2. Then try a SPARCO weighting using the SED the organisers gave out.
  3. Image per channel only if those leave structured residuals (C3).

**2024 Obj2 GRAVITY (χ²/N ≈ 185–250, error scale 14–16).**
- *What we see.*
  - V² dominate: 290–370 per point, against 9–35 for the closure phases.
  - The short-baseline V² residuals have a mean of −4σ.
  - The evidence strongly prefers the star (ΔlogZ ≈ +1e4).
- *What the contestants knew.* A young star with a suspected companion,
  and uncalibrated `OI_FLUX` spectra.
- *Diagnosis.* A grey environment with a fixed star-to-environment ratio
  cannot follow a star and a disk with different spectra across 2.0–2.5 µm
  (the problem SPARCO solves). Over-resolved flux would also lower the
  short-baseline V².
- *Plan.* Add SPARCO members: the star with a `PowerLaw` spectrum, and the
  environment with its own fitted index. Add a halo option. The evidence
  chooses between them. 2024 Obj2 PIONIER (χ²/N 1.7) gets the same
  members.

**2022 AMI (χ²/N 0.02–0.13).**
- *Diagnosis.* This is not over-fitting in the error-bar sense. There are 56
  data points (21 V² and 15 independent closure phases, from one
  snapshot). MacKay's γ is about 53, so nearly every point is used to fix
  an image parameter, and χ² ≈ N − γ is small by construction. The error
  scale is 0.63–1.58, consistent with 1. The problem is under-determined:
  the image has more freedom than the data constrain.
- *Plan.*
  1. Report N − γ alongside χ²/N.
  2. Compare the images' evidence with parametric models (a point star;
     a star plus a companion from `grid_fit`, since the organisers called
     it a high-contrast test).
  3. Treat the ensemble's σ map as the honest image uncertainty.

**MATISSE (error scale 0.34–0.53).**
- *What we see.* In the N band, χ² per point is 0.005 for V² and 0.49 for
  the closure phases. The two blocks are mis-scaled by very different
  factors, so one global scale cannot fix both.
- *Plan.*
  1. Use a MacKay error scale per observable block (V², closure phase),
     with γ split between the blocks by the diagonal of the hat matrix.
     This comes from the same SVD the evidence already uses. It is a
     virgil feature (`error_scale(..., blocks=...)`).
  2. Adjacent MATISSE channels are correlated, so the effective N is
     smaller than the count of points. That is noted, not yet modelled.
  3. The 19 members that hit the 2-hour limit are being rerun with
     8 hours.

**Scoring against the winners (C4).**
- *Available truths.* Only 2004 data2 and 2018 have truths we can use, as
  analytic models in the papers. They were revealed after the contests, so
  they are used for scoring only. Every other year needs the organisers'
  truth images.
- *Which first.* 2018 already fits (χ²/N 1.11), so it is the first case to
  score.
- *Sources.* The metric definitions and published score tables are
  collected in `docs/contest_scoring_sources.md`.

**Order of work.**
1. Per-block error scales (virgil PR).
2. The widened hyperparameter grid, resolved-star members and field scans
   (`contest_images.py`), then a second campaign on 2004 data2, 2010 and
   2024 Obj2.
3. SPARCO members for 2024 Obj2.
4. Scoring 2018, then 2004 data2.
5. Parametric comparisons for AMI.

### C2f. A synthetic benchmark before more campaigns

χ²/N says the data are fitted, not that the image is right, and only
2004 data2 has a truth we can score against numerically. Campaign 1 scored
σ/peak 0.250 there, third of four (BSMEM 0.116, WISARD 0.163, MIRA 0.532,
VLBMEM 0.798). With the truth's unpublished details fitted to the data it
scores 0.260, and even that truth fits only at V² χ²/N ≈ 50: the 2004
data carry calibration errors. So, before more contest campaigns, every
method choice is tested on **known phantoms observed with the real contest
uv coverages** (Ben, 2026-10-06).

- **Phantoms** (`src/crosscheck/phantoms.py`, NumPy only). These are
  generic families matching each contest's *pre-submission* description:
  - a spotted elliptical limb-darkened star with a companion (2004);
  - a thin flared disk (2006);
  - a core in a clumpy envelope (2008 AGB);
  - a star, ring disk with gap, and a planet (2018);
  - a binary with a dust spiral (2024 Obj1).

  Sizes are in beams and parameters are random. They are never the
  published truths, so methods are not tuned to the answers.
- **Simulation** (`scripts/contest_bench.py simulate`). V² and closure
  phases come from a direct DFT (`crosscheck.sky`) on each contest file's uv
  points, with noise from its quoted errors. Closure noise is drawn from
  shared baseline phases per (MJD, TIME) frame and topped up to each
  triangle's quoted variance. In tests, virgil fits noiseless simulations
  exactly, fits noisy ones at χ²/N within 30% of 1, and fits the inverted
  image far worse.
- **Metrics** (`src/crosscheck/image_metrics.py`):
  - the 2004 σ/peak;
  - the RMS after convolution ×10⁶;
  - the 2024 L1 with the optimal flux scale;
  - NCC (normalised cross-correlation).

  Alignment uses a padded, zero-filled whole-pixel shift; inversion is
  allowed only for V²-only data.
- **Arms** (`contest_bench.CONFIGS`). Each changes one factor against
  `baseline`, the campaign-1 CLEAN-started GP with no star:
  - `point_star`, `disk_star`, `halo`;
  - fields of 1× and 4×;
  - `half_beam_mean`: the GP starts from CLEAN components restored with
    half a beam;
  - `mem`: MaxEnt with the CLEAN image as its default model, the winners'
    own method.

  Arms from the virgil PRs follow once they merge:
  - a parametric elliptical limb-darkened start (#250);
  - multi-scale CLEAN (#252);
  - converged-only evidence and Laplace σ maps (#251);
  - per-observable error scales (#247).
- **Adoption rule.** A change is adopted only if it raises the median score
  of its target family without lowering the others beyond the scatter
  between noise draws.
- **Status.** The smoke run of the first six arms (job 18097494) ran clean
  in 1–7 minutes per task. Next: smoke the two new arms, then the full
  benchmark of 60 datasets × 8 arms on OzSTAR (`bench.sbatch`).

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

**Inversion symmetry in scoring.** Visibility amplitudes are unchanged
when the image is inverted through the origin, I(x, y) → I(−x, −y), by
Hermitian symmetry. In two dimensions that inversion *is* a 180°
rotation, for any image. Mirror reflections (x → −x) are not a symmetry:
V² changes by ~20% for an asymmetric test image. The two coincide only
for a binary, which is symmetric about its own axis. Fourier phases change
sign under inversion, so closure phases break the degeneracy, by an
amount that depends on their S/N. The scoring rules are:

- **V²-only data:** score both I and its inversion against the truth, and
  keep the better, reporting both.
- **Data with closure phases** (every contest set we have): do not flip,
  but report the flip Δχ² (`diagnose`'s `flip_dchi2`). A small value means
  the orientation is barely constrained, and the score should say so.
- **Inversion centre:** invert about the true origin. On a pixel grid
  that is (N − 1)/2, which `[::-1, ::-1]` gives exactly for odd N, so
  scoring grids are kept odd.

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

## Decisions

- **No other codes run here** (Ben, 2026-10-04). virgil's images are
  compared only with the truth images, where we have them, and with the
  quantitative comparisons of the other pipelines published in the contest
  papers (score tables, metrics, figures). MiRA, SQUEEZE and the rest are
  not run on the data.

## Open decisions for Ben

1. **2024 truths** (decided by Ben, 2026-10-06). He will ask Millour or
   Soulez later, not yet. Until then, 2024 is assessed on the data alone
   (χ²/N, residuals, agreement within the ensemble) and is left out of the
   comparison with the winners. It is recorded as pending in
   `docs/contest_scoring_sources.md`.
2. **Chromatic imaging in virgil.** If C3 shows that per-band images are
   not enough, should a cross-wavelength regulariser go into virgil (a
   design note first)?
