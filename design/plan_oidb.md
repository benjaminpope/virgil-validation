# Plan: published datasets from the JMMC OiDB

## Why

The ESO-binaries plan (`design/plan_eso_binaries.md`) gives virgil real
anchors, but every one of them needs us to reduce raw frames first, so a
mismatch can come from our reduction as easily as from virgil. The JMMC
Optical interferometry DataBase ([OiDB](https://oidb.jmmc.fr/collections.html))
holds the *authors' own calibrated OIFITS* behind published papers. Fitting
those files with virgil, and comparing with what the paper reports, removes
the reduction from the chain: same files, same errors, different code.

This plan uses OiDB for three goals:

- **(a) Binaries.** Every binary whose published data we can get becomes a
  `published-binary` record: separation, PA and flux ratio per epoch, and an
  orbit where the paper fits one.
- **(b) Image reconstruction.** The CHARA Imaging Workshop 2023 data, and the
  data behind the published images of R Dor, π¹ Gru, R Car and Polaris,
  reconstructed with virgil's imaging code and with the PYRA/MYTHRA-style
  ensembles in `virgil.ensemble`.
- **(c) Rapid rotators.** Planned separately, because the results are for a
  paper and stay out of this repository (see "Rapid rotators" below).

Terms (read from <https://oidb.jmmc.fr/doc.html>, sections 2, 3 and 9, on
2026-10-07; Ben has accepted them; decision 1 below): the portal and its contents are under
CC BY-NC-SA 4.0, **and** the terms of use add obligations that depend on the
data's calibration level. Use is for public astronomical research only.
For **L2** (unpublished) data the user must contact the dataPI *before*
presenting any work using them and agree the citation, acknowledgement or
collaboration policy. For **L3** (published) data the user must at least
thank the dataPI (the paper's first author) and cite the original paper.
Any publication should also carry OiDB's acknowledgement sentence. Data
files are never committed here (see "Data handling").

## Access: what was checked

Read-only, on 2026-10-07, for one small collection (Gl 229 B, 22 files):

- A collection page (`collection.html?id=<id>`, paginated with `&page=N`,
  25 rows a page) lists each granule with a link
  `https://oidb.jmmc.fr/get-data.html?id=<granule>&name=/<file>`. WebFetch
  gets 403; curl with a browser User-Agent works.
- `curl -I` on `get-data.html` returns **400**; a GET returns **303 See
  Other** to
  `/exist/apps/oidb-data/oifits/staging/<uuid>/<file>`. `curl -I` on that URL
  returns **200**, `application/octet-stream`, with `Content-Length` (469 440
  bytes for `GRAVI.2023-12-26T04_46_32.726_dualscivis.fits`).
- **No login, no cookie and no terms click-through** were needed for a
  public collection (embargo `P0M`). The VizieR-imported collections
  (`J/A+A/...`) use the same `get-data.html` links; their redirect target
  was not checked.
- Collections list "granules" (one row per target × instrument × night) and
  "OIFITS files"; the file count is the one that matters for downloads.
- Search is rendered on the server (corrected in O0, 2026-10-07; an earlier
  draft said JavaScript-only): `search.html?collection=~<id>&perpage=100&page=P`
  lists a collection's granules, each row carrying `data-access_url`,
  `data-calib_level`, `data-bib_reference` and the dataPI. A cone search is
  `search.html?conesearch=<name or hh:mm:ss +dd:mm:ss>,J2000,<radius>,<arcmin|arcsec|deg>`
  (names resolved by Sesame; combinable with `collection=`; the form's
  `cs_position` fields are ignored over GET), so whether a star is inside a
  big collection (PIONIER L2, 10 982 files) can be checked without
  downloading. L0 rows are observation logs, not data. No TAP service was
  found.

So the download mechanism is: scrape the collection pages, follow each
`get-data.html` redirect with a GET, and stream the staging URL. Nothing was
downloaded for this plan.

## Data handling

- **Locations.** OzSTAR: `/fred/oz440/bpope/oidb/<collection-id>/` (fetched
  on a trevor node, the only ones with internet). Laptop:
  `~/data/oidb/<collection-id>/`, only for the small collections a stage
  needs to inspect, never in git. `<collection-id>` is OiDB's id with `/`
  replaced by `_` for the VizieR ones.
- **Manifest.** `oidb/manifest.yml`, in the style of `contests/manifest.yml`:
  per collection the id, title, bibcode, instrument, **calibration level
  (L2 or L3), dataPI, the terms that follow from them** (L3: thank the
  dataPI and cite the paper; L2: dataPI contacted, date, and the agreed
  policy) and the attribution line, and per file the name, granule id, staging URL, bytes
  and sha256. The sha256 is recorded on the first fetch and checked on every
  later one; a changed file is a failure, not an update.
- **Fetcher.** `scripts/fetch_oidb.py`, stdlib only (it runs on trevor):
  `--collection <id> --dest <dir>`, one request at a time, a `.part` file and
  an atomic rename, resumable, and `--verify` to re-hash without the network.
  It is driven from an OzSTAR job `scripts/oidb_dl` in `~/code/ozstar_scripts`
  (copied from `scripts/_template`, submitted with `bin/oz submit-trevor`),
  pinned to the full virgil-validation commit hash of the manifest.
- **Fits.** Every virgil fit runs on OzSTAR (no heavy JAX on the laptop,
  diagnostics included), as jobs pinned with `--ref=<validation sha>
  --ref-virgil=<VIRGIL_REF>`; results come back with `bin/oz pull` and only
  the small numbers (positions, elements, χ², image metrics) are committed,
  as evidence records.
- **Sizes.** Most collections here are 1–30 MB. The largest planned ones are
  R Dor AMBER (75 files), WR 104 (121), κ Tuc A (136) and Betelgeuse MATISSE (361);
  O0 sizes each from `Content-Length` before any fetch.

## Independence and ground truth

- Same files, different code: virgil reads the authors' OIFITS and fits
  them. The published number is the `literature` root ("published numbers
  reproduced from the authors' data"), not `golden:`, which is reserved for
  reference values from someone's code registered in `trust/golden.yml`.
  Since both analyses use the same data, they should agree to well within
  the published σ, and a large difference is a finding.
- **In-sample vs out-of-sample, per epoch.** O0 records, for every epoch in
  `oidb/references/<collection>.json`, whether the paper's orbit used it.
  An epoch the paper used is a same-data `literature` check with the tight
  tolerance below, not a prediction. "Out-of-sample" is kept for epochs
  that are really new: Gl 229's 2024-12 and 2025-02 nights, and any κ Tuc A,
  σ Ori or Polaris Ab epoch O0 shows the paper did not use. The ι Peg
  2018-10-22 night (workshop and Anugu's collection), the σ Ori 2011-09-29
  night (very probably in Schaefer+2016) and the Polaris epochs Evans+2024
  used are in-sample until O0 shows otherwise.
- **Roots are claimed only when exercised.** A published CANDID or PMOIRED
  number is `literature`; the `candid` or `pmoired` root needs that code
  re-run by us on the same files through `src/external_bridge/`.
- Reference positions from published elements come from the NumPy Kepler
  evaluator in `src/crosscheck/orbits.py`; diameters from the SciPy
  uniform and limb-darkened disk visibilities in `src/crosscheck/limb.py`.
  Neither imports virgil.
- External codes run through `src/external_bridge/` workers, never imported
  next to virgil: **CANDID** (binary grids and detection limits; the tool
  several of these papers used), **PMOIRED** (parametric fits with pinned
  conventions), **fouriever** (second grid root, once its P5 issue is
  settled) and **eht-imaging** (RML images on the same data).
- Following the contest decision, we do not run MiRA, SQUEEZE or other image
  reconstructors; images are compared with the published images and with
  eht-imaging.
- **Statistics.** Every fit quotes the raw χ²/N on the file's errors first.
  A rescaled χ² ≈ 1 is not evidence; an error scale s ≫ 1 means a failed fit
  or bad errors and is reported as such. Priors are Jeffreys priors under
  the relevant group: log-uniform separations, flux ratios, diameters,
  periods, semi-major axes and error scales; uniform PAs, positions, Ω, ω
  and time of periastron; uniform cos i; eccentricity uniform on [0, 1)
  (a stated choice, no group acts on it). All bounds are stated. These are
  the priors of virgil's own headline fits. A comparison with a *published
  posterior* instead uses the paper's priors and the paper's data set, in
  the style of `orbitize-posterior` (RVs included where the paper used
  them, or the comparison stated as astrometry-only with ω taken mod 180°),
  so that a prior or data mismatch cannot pass or fail the check.

### Pass criteria (defaults; a dataset may tighten them)

| Check | Pass |
|---|---|
| Per-epoch binary on the authors' files (in-sample, `literature`) | Δρ, ΔPA, Δ(flux ratio) each ≤ 0.25σ_pub, against the statistical-only part of σ_pub where the paper states a systematic term (wavelength scale, error rescaling, bootstrap over nights); the flux-ratio definition (secondary/primary, primary/total, band) recorded in `oidb/references/*.json` and matched; raw χ²/N reported, flagged if > 3 |
| Same files, virgil vs CANDID or PMOIRED re-run through the bridge (`candid`, `pmoired`) | best fits within 0.25σ (as `pmoired-identical-fits`), once conventions are pinned |
| Orbit vs a published posterior | paper's priors and data set (above); percentiles of each element within 0.15 posterior sd (as `orbitize-posterior`) |
| Orbit, virgil's headline fit (Jeffreys priors) | reported, not a pass/fail against the paper; out-of-sample positions within 2σ_pred |
| Detection limits (`literature`) | virgil's 5σ contrast curve within 0.1 mag of the published one (CANDID's Absil limits run ~1% low, P4) |
| Injection controls on the same files | a companion injected 0.5 mag above the published limit is detected; one 0.5 mag below is not |
| Diameter / parametric | within 2σ_pub; raw χ²/N reported |
| Image | the baseline is the best-fit `LimbDarkenedDisk` (step 1). The reconstruction must beat it in raw χ²/N on the given errors by a stated margin (fixed per target before the run), and the *residual* images (image − best-fit disk, both convolved to one beam) must have normalised cross-correlation ≥ 0.5 with the published residual; named features (below) recovered. Raw χ²/N also ≤ the published reconstruction's where stated |
| Image vs eht-imaging, same data and regulariser | the same minimum (as `rml-imaging`) |
| Ensemble (MYTHRA mean) | raw χ²/N ≤ the best single member's × 1.1; **spread bounded**: median spread over the stellar disk ≤ 0.3 × the contrast of the features it is used to judge; **spread calibrated**: on the workshop ι Peg/σ Ori data (known answer) and on a simulated twin of each real target's uv coverage, the 1σ spread map covers the truth at 68% ± 15% of pixels inside the disk, not much more |

Thresholds are fixed and hashed before each OzSTAR run (`CRITERIA_HASH`).

## Candidate datasets

Abbreviations: **L** = laptop-light (file inspection, NumPy cross-checks
only); **OC** = OzSTAR CPU; **OG** = OzSTAR GPU. "Files" are OIFITS files as
OiDB lists them on 2026-10-07. Published numbers marked "(O0)" are to be
read from the paper's tables in Stage O0, not from memory or abstracts.

### Binaries (goal a)

| Collection id | Target, instrument | Files | Reference | Published result to match | virgil model | Cross-check | Compute |
|---|---|---|---|---|---|---|---|
| 782185b2-0727-42b0-a185-b2072732b047 | Gl 229 Ba–Bb, GRAVITY (dual-field, SC on B) | 22 | Xuan+2024 Nature 634, 1070 | orbit P = 12.134 d, e ≈ 0.23, flux ratio 0.47±0.03; CPs only, 2.05–2.18 µm | per night: a linearly moving binary (the `LinearBinary` SourceModel in `fit_nights.py`, see O1), then `KeplerOrbit` + `Attached` on visibilities | Xuan's per-night positions (`literature`); `crosscheck.orbits`; PMOIRED was the authors' tool, so `literature` unless re-run through the bridge (`pmoired`); our esorex reduction (`self-consistency` only) | OC |
| 647a22a9-5047-4220-ba22-a95047022072 | CHARA Imaging Workshop 2023: ι Peg (MIRC-X, 7 files, 2018-10-22) and σ Ori (MIRC, 1 file, 2011-09-29) | 8 | Workshop; ι Peg orbit (Anugu+2020, arXiv:2007.12320); σ Ori Aa–Ab (Schaefer+2016 AJ 152, 213) | position predicted by each orbit at the epoch; flux ratio (O0) | `BinaryModelAngular` (+ `UniformDisk` primary); also imaging, below | `crosscheck.orbits`; CANDID via the bridge; in-sample unless O0 shows otherwise | OC |
| fac164e1-d9d0-4500-8164-e1d9d0450099 | ι Peg, MIRC-X GRISM-190 | 24 | Anugu+2020 (MIRC-X instrument paper) | 5-epoch positions and SB2 orbit, P ≈ 10.2 d (O0) | per epoch, then `KeplerOrbit` | CANDID; workshop files of 2018-10-22 | OC |
| bda75673-61c6-49f0-a756-7361c699f0c4 | A-star companions, MIRC-X | 28 | De Furio+2022 ApJ 941, 118 | detections (ρ, PA, contrast) and limits | `linear_flux_grid`, `fit`, `absil_limits` | published CANDID numbers (`literature`) and CANDID re-run through the bridge (`candid`) | OC |
| f4afc4cd-fd31-40d3-afc4-cdfd3150d340 | HD 45166, GRAVITY, one night | 7 | Deshmukh+2025 A&A 695, L20 | ρ, PA, flux ratio (O0) | `BinaryModelAngular` | CANDID; PMOIRED | OC |
| 855397ef-dca8-4125-9397-efdca851259d | κ Tuc A, MATISSE (2019–24) + GRAVITY (2024) | 136 | Stuber+2026 AJ 171, 1 | per-epoch positions and orbit of the low-mass companion; hot-dust disk (O0) | chromatic binary (`spectra.PowerLaw` flux ratio) + `GaussianDisk` dust; `KeplerOrbit` | CANDID per epoch; published | OC |
| 1e9bab59-54c9-416d-9bab-5954c9416db7 | GG Tau Ab1–Ab2, PIONIER | 3 | Duchêne+2024 (arXiv:2404.02469) | relative positions feeding the full orbit (O0) | `BinaryModelAngular` | `crosscheck.orbits`; CANDID | OC |
| be66c71c-d400-469c-a6c7-1cd400469cd0 | A-type Gaia–Hipparcos accelerators, GRAVITY | 44 | Waisberg+ (bibcode O0) | companion detections and limits | grid + `fit` + `absil_limits` | CANDID | OC |
| c3f46f3d-dbb6-425f-b46f-3ddbb6325f58 | 39 Galactic WR stars, GRAVITY | 88 | Deshmukh+2024 A&A 692, A109 | resolved companions (ρ, PA, flux ratio) and non-detection limits | grid + `fit` + `absil_limits` | CANDID | OC |
| 0ab199d0-29c4-40d7-b199-d029c420d795 | transition disks, SPHERE IRDIS/IFS SAM | 9 | Stoker+2024 A&A 682, A101 | candidate companions and contrast limits | grid + `absil_limits` (calibrated OIFITS: no AMICAL needed) | CANDID | OC |
| 371c145f-7889-47d5-9c14-5f7889c7d510 | θ¹ Ori C, AMBER | 2 | Kraus+2009 (Messenger 136, 44; A&A 497, 195) | 2007 position on the ~11 yr orbit (O0) | `BinaryModelAngular` | `crosscheck.orbits`; CANDID | OC |
| J/A+A/586/A35 | TZ For, PIONIER | 11 | Gallenne+2016 A&A 586, A35 | astrometric + SB2 orbit (O0) | per epoch, `KeplerOrbit` | `crosscheck.orbits` | OC |
| J/A+A/536/A55 | SS Lep, AMBER + PIONIER | 8 | Blind+2011 A&A 536, A55 | 8-epoch positions, P ≈ 260 d, Roche-filling giant (O0) | `UniformDisk` + point; also images | published | OC |
| 534fbba6-b715-4b41-9278-b6bd0de9f674 | γ² Vel, AMBER | 43 | Lamberts+2017 MNRAS 468, 2655 | binary positions on the known orbit (North+2007); wind-collision zone | binary + `GaussianDisk` | `crosscheck.orbits`; published | OC |
| 45840351-f65e-446c-8403-51f65e546c16 | HD 174881 (HR 7112), CHARA | 2 | Torres+2025 ApJ (bibcode O0) | positions on the published orbit | `BinaryModelAngular` | `crosscheck.orbits` | OC |
| 696baf06-6c3c-424d-abaf-066c3c324d99 | HR 6819 (Be + stripped star), GRAVITY HR | 12 | Klement+2025 A&A 694, A208 | orbit, dynamical masses | binary + disk; `KeplerOrbit` | published | OC |
| f9cf0622-5fb3-410c-8f06-225fb3e10c5d | T CrA B, MATISSE | 32 | Varga+2025 | first companion detection | chromatic binary + disk | published | OC |
| 89aec164-3796-4f59-aec1-6437969f59c4 | Gaia BH3, GRAVITY | 13 | Kervella+2025 A&A 695, L1 | NIR upper limit | `absil_limits`: a `check` of the limit *depth* against the paper (`literature`), plus injection controls on these files 0.5 mag above (must detect) and below (must not) the published limit; "no detection" alone tests nothing | CANDID via the bridge | OC |
| 1b17307c-9691-4fea-9730-7c96917feaa9 | M17 young O stars, GRAVITY | 8 | Bordier+2021 | multiplicity, companion positions | grid + `fit` | CANDID | OC |
| 544cd41c-6a2b-4374-8cd4-1c6a2bf374aa | massive YSOs, GRAVITY | 16 | Koumpia+2021 A&A 654, A109 | binarity at au scales | grid + `fit` | CANDID | OC |
| 52853bac-9dbe-44d4-853b-ac9dbee4d480 | R136 central stars, GRAVITY+ | 3 | Millour (unpublished?) | (O0: is there a paper?) | grid | CANDID | OC |
| 337cfbf7-5327-4875-bcfb-f753270875d0 | Polaris Aa–Ab, MIRC/MIRC-X | 28 | Evans+2024 ApJ 971, 190 | Ab positions and orbit (P ≈ 29.6 yr) | binary with `LimbDarkenedDisk` primary | `crosscheck.orbits`; published | OC |
| 92afa975-2a24-4ba0-afa9-752a246ba061 | WR 137, JWST/NIRISS AMI | 6 | Lau+2024 ApJ 963, 127 | dust structure (extended; not a resolved pair) | `GaussianDisk`/`Image` | published | OC |
| 73adfa26-04d5-48d7-adfa-2604d538d763 | symbiotic giants, MIRC-X | 4 | (bibcode O0) | giant diameters; companion if resolved | `UniformDisk` (+ point) | `crosscheck.limb` | OC |
| J/A+A/597/A137 | α Cen A, B and HD 123999, PIONIER | 13 | Kervella+2017 A&A 597, A137 | LD diameters; HD 123999 orbit (O0) | `LimbDarkenedDisk`; binary | `crosscheck.limb` | OC |

The JWST AMI files are already calibrated OIFITS, so the masking rule
(AMICAL only for ground-based masking, never on JWST) is not touched.

Three Be-star collections that the OiDB spreadsheet tags as rapid rotators
are planned privately with the rapid rotators (see "Rapid rotators" below).

**Clean binaries are the validation targets.** A system with significant
circumstellar or circumbinary material can fail a binary fit for reasons
that have nothing to do with virgil, so it is never a binary pass/fail
check. Rows above flagged **not clean** are fitted as binary + disk/dust
models at low priority and reported, not scored: κ Tuc A (hot exozodiacal
dust), the SPHERE SAM transition disks, SS Lep (circumbinary dust),
γ² Vel (colliding winds), HR 6819 (Be decretion disk; see O1), T CrA
(Herbig disk), the massive YSOs, the symbiotics (accretion, nebula) and
WR 137 (dust). The WR survey is scored only on companions the paper
reports as point-like.

Further binary from the OiDB spreadsheet (`~/data/oidb/oidb_collections.csv`,
2026-10-07), Stage O2 (published result and model read in O0):

- **not clean** `512015b1-fb46-4cde-a015-b1fb46ccdec6`: ρ Oph A, GRAVITY, 8 files (Klement+2025, bibcode O0): binary position and magnetosphere offset; binary + `GaussianDisk`; OC.

Excluded (2026-10-07): the post-AGB binaries (`3722c1a7-…`, `557172ab-…`,
`bf01675e-…`) and 89 Her (`J/A+A/559/A111`), whose circumbinary disks make
them unclean; ε Aur (`76557b92-…`, `J/A+A/544/A91`), the system eclipsed by
a disk. WR 104 moves to O3 imaging.

Also tagged by the spreadsheet and already covered elsewhere: the imaging
contest collections (`7f7fb9ed-…`, `3e71bedc-…`, `6df579f8-…`), which belong
to `plan_imaging_contests.md`.

### Imaging (goal b)

| Collection id | Target, instrument | Files | Reference | Published result | Features to recover | Compute |
|---|---|---|---|---|---|---|
| 647a22a9-5047-4220-ba22-a95047022072 | CHARA Workshop 2023 (ι Peg, σ Ori) | 8 | workshop; orbits above | two point sources with known ρ, PA, flux ratio | peak positions within 0.1 beam of the orbit; flux ratio within 10% | OG |
| 19f7e2cf-2a03-4bb2-b7e2-cf2a03bbb245 | π¹ Gru, PIONIER H | 1 | Paladini+2018 Nature 553, 310 | H-band image with a few large granulation cells; diameter | diameter within 2σ; number and size of cells in the residual image (after the best-fit disk) match the paper, with the bounded, calibrated spread | OG |
| 7e5740b8-745c-40bd-9740-b8745cd0bd52 | R Car (Mira), GRAVITY | 15 | Rosales-Guzmán+2024 A&A | K-band images (continuum and CO) | diameter; brightness asymmetry; chromatic change | OG |
| 6cfa202a-e35c-458c-837e-c512e73c3e45 | R Dor (AGB), AMBER HR | 75 | Ohnaka, Weigelt & Hofmann 2019 ApJ 883, 89 | MiRA images in continuum and CO lines | continuum diameter; line-vs-continuum size; asymmetry | OG |
| 337cfbf7-5327-4875-bcfb-f753270875d0 | Polaris, MIRC/MIRC-X | 28 | Evans+2024 ApJ 971, 190 | spotted surface images; LD diameter | diameter; spot positions in the residual image match the paper, with the bounded, calibrated spread | OG |

For each: (1) a parametric fit (diameter, limb darkening) against the paper
and `crosscheck.limb`; (2) single TSV/MaxEntropy reconstructions with an
`l_curve`; (3) a `virgil.ensemble.ensemble` run, one group per array task
(`draw_groups` → `run_group` → `combine`), with the published image compared
to the MYTHRA mean and spread map; (4) eht-imaging on the same data and
regulariser. Published images are asked for as FITS from the authors where
they are not in the papers' supplementary data (question 5).

Further imaging targets from the spreadsheet, all Stage O3, one line each
(the four imaging steps above apply; compute OG):

- `093ff33b-dca8-4f5a-bff3-3bdca81f5a16`: SU Aur, MIRC-X + PIONIER, 29 files (Labdon+2023 A&A 678, A6): warped disk wind image and disk inclination/PA.
- `cb5b2fd1-e23a-4c6c-9b2f-d1e23aec6c95`: HD 163296, MIRC-X + PIONIER, 13 files (Setterholm+, status O0): inner-disk images and rim geometry.
- `f89ea077-14c6-49db-9ea0-7714c6d9db0f`: HD 190073, MIRC-X + PIONIER, 2 files (Ibrahim+2023 ApJ 947, 68): inner-au disk image.
- `3d64620d-a152-4e75-9bc4-c68735d293b1`: CL Lac, MIRC-X, 36 files (Chiavassa+, bibcode O0): convective surface structure; LD diameter baseline.
- `f87fdde1-c4b6-4de7-bfdd-e1c4b69de708`: Arcturus, IOTA/IONIC, 9 files (Lacour+2008 A&A 485, 561): limb-darkened disk image; a near-featureless control for the residual-image test.
- `f9418307-0127-49cf-8183-070127f9cf08`: Betelgeuse, MATISSE 2018–20, 361 files (Drevon+2024 MNRAS 527, L88): surface images; Drevon is the PYRA author, so the most direct check of `virgil.ensemble` if his images used PYRA (O0).
- `c5a5d133-a178-4949-a5d1-33a1781949e5`: WR 104, MATISSE, 121 files (status O0): colliding-wind WR binary with a pinwheel dust nebula; image deconvolution and the PYRA/ensemble comparison, not a binary fit.
- `12803c68-6124-40ad-803c-68612470ad3d`: R Scl, MATISSE, 236 files (Drevon+2022 A&A 665, A32): a PYRA comparison target (Drevon's own data).
- Drevon's Betelgeuse MATISSE runs, PYRA comparison targets with `f9418307-…` above: `5866421b-f263-46c0-a642-1bf263a6c0bd` (2018-12 commissioning, 111 files), `6c5e6803-f7af-4f0e-9e68-03f7afcf0e18` (2020-02, 18 files), `07f92587-f7ef-4f04-b925-87f7ef3f04c1` (2020-12, 54 files); publication status O0.
- `706d923a-657a-44bb-ad92-3a657a34bb1f`: Betelgeuse, SPHERE-IRDIS SAM during the Great Dimming, 2 files (Montargès, paper O0): asymmetric photosphere; calibrated OIFITS, no AMICAL needed.

### Rapid rotators (goal c): in elr-pavo-paper, not here

The rotator *results* are for Ben's paper and live privately in
`~/code/elr-pavo-paper/plans/oidb_rotators.md`. This repository keeps only
the model-level checks it already has (`virgil.models.GravityDarkenedStar`
against `src/crosscheck/elr.py`), which name no science target.

## Stages

### Stage O0: access, manifest, references (about 3 h; L)

Write `scripts/fetch_oidb.py` and `oidb/manifest.yml` for the Stage O1
collections; fetch Gl 229 B (≈10 MB) on trevor and record sha256s; read the
first-stage papers' tables into `oidb/references/<collection>.json` (the
"(O0)" numbers above), with page and table numbers. Check the redirect of
one VizieR collection. **Confirm each collection's publication status**
(journal version of arXiv-only rows; the spreadsheet's status is not
trusted: e.g. ι Peg's bibcode is an arXiv one, the R Car, GG Tau,
HD 163296 and symbiotics papers are marked preprint or unverified, and the
A-type and workshop collections unknown). Record each collection's calibration level, dataPI
and resulting terms in the manifest, and list the L2 collections whose
dataPIs must be contacted before results are presented. Tests: manifest
schema (level, dataPI and terms required); every reference value carries a
source.

### Stage O1: first targets (about 12 h)

In order:

1. **Gl 229 Ba–Bb, authors' files.** The per-night recipe is not in this
   repository yet: `fit_nights.py` (with its `LinearBinary` SourceModel, a
   binary moving linearly within a night) and `fit_orbit.py` live in the
   E1 analysis folder `~/data/eso_binaries/gravity/scripts/`, outside git.
   O1 first ports them into `scripts/` here (or rewrites them from the E1
   description), then runs them on Xuan's 22 files. The virgil checks are
   per-night positions vs Xuan's published ones (`literature`, in-sample)
   and vs `crosscheck.orbits`, and the orbit vs Xuan's posterior with
   Xuan's priors and data (CPs + CRIRES+ RVs; astrometry-only with ω mod
   180° if the RVs are not used). Positions from Xuan's files vs from our
   own esorex reduction of the same exposures is a reduction check with
   virgil on both sides, recorded as `self-consistency` only. The 2024-12
   and 2025-02 nights stay out-of-sample.
2. **HR 6819, GRAVITY dynamical masses** (`696baf06-6c3c-424d-abaf-066c3c324d99`,
   12 files). Per-epoch positions of the stripped star relative to the Be
   star and the orbit vs Klement+2025 (A&A 694, A208), with the paper's
   priors and data for the posterior comparison. The Be star's decretion
   disk is resolved, so the model is binary + disk (`EllipticalGaussian`
   or `GaussianDisk` around the Be star), and the check is labelled
   **not clean**: positions are scored only against the paper's own
   binary + disk solution, never as a clean-binary pass.
3. **CHARA Imaging Workshop 2023.** ι Peg and σ Ori as binaries (positions
   vs orbits) and as images (goal b). Eight files, two goals.
4. **ι Peg, Anugu's collection.** Five epochs and an orbit; the
   2018-10-22 night is also in the workshop set, so the two files of that
   night should give the same position (O0 checks whether they are the same
   reduction).
5. **A-star companions (MIRC-X).** CANDID was the paper's tool: compare
   with the published numbers (`literature`) and with CANDID re-run on the
   same files through the bridge (`candid`).
6. **HD 45166.** One night, seven files.
7. **π¹ Gru.** One file: the cheapest imaging target, and a first ensemble
   run on real data.

**As built (parametric part).** `scripts/oidb_fit.py` holds one recipe per
collection (model choices in its docstring) and runs as the OzSTAR array job
`oidb_o1_fits`; `scripts/oidb_compare.py` writes the table, with its rules
frozen in `CRITERIA` (hash in the table): per-epoch ρ, PA and flux ratio
within 0.25 σ_pub (published statistical errors; error ellipses projected
on ρ and PA, σ Ori's divided by its 2.24 inflation); diameters, resolved
flux and orbit flux ratios within 2 σ_pub; HD 45166 within 1 σ of the
adopted mean over its four calibrations; orbital elements reported, not
scored; HR 6819 compared, never scored; L2 rows withheld; a scored value
virgil does not deliver (or a missing fit file) is MISSING and counts as a
failure. The A-star flux ratios and resolved fluxes are scored under the
2 σ parametric rule, not the per-epoch one: CANDID's bandwidth smearing
averages V² and the bispectrum over three points, ours the complex
visibility over seven (`docs/method/candid.md`), so the two definitions
can differ by more than 0.25 σ_pub; the A-star positions stay per-epoch. Two departures
from the list above. Gl 229 has no per-night `LinearBinary`: the orbit is
refitted on every file as a snapshot at its own time, which carries the
motion within a night, and static per-night fits serve only as starts.
HR 6819 is fitted on the K continuum with the Brγ and He I windows left
out instead of binary + disk: Klement+2025's disk contributes only in the
line, so the continuum model is theirs without the disk.

### Stage O2: the remaining binaries (about 2 h per small collection, 6 h per survey; OC)

Orbit systems first, clean ones scored: GG Tau Ab, TZ For, θ¹ Ori C,
HD 174881, Polaris Ab, HD 123999; then the not-clean orbit systems,
reported but not scored: κ Tuc A, γ² Vel, SS Lep. **Decide after the
orbits:** the detection-limit surveys (WR survey, A-type accelerators,
SPHERE SAM, M17, massive YSOs) and the Gaia BH3 limit-depth check with its
injection controls. Last, low priority and not clean: ρ Oph A, T CrA, R136
and the symbiotics.

### Stage O3: imaging (about 4 h per target plus OzSTAR GPU time; OG)

R Car, then Polaris, then R Dor (chromatic; needs C3-style chromatic
imaging from the contest plan), then the further targets listed above
(Arcturus first as the featureless control, Betelgeuse MATISSE once O0 has
checked which runs Drevon's papers used; WR 104, R Scl and Drevon's
Betelgeuse sets as PYRA comparisons). Images are compared with the
published figures and quoted values only: we do not ask authors for FITS
images, and we do not run MiRA or SQUEEZE. Each with the four steps above.

### Stage O4: reporting (2 h)

Trust-graph edits, made before the first test names them
(`tests/test_trust_graph.py`):

- a new `oidb-binary` pipeline (dataset: OiDB authors' calibrated OIFITS;
  reference: the papers; url: this plan) with the steps actually exercised:
  `read_oifits`, `OIData`, `BinaryModelAngular`, `BinaryModelCartesian`,
  `LimbDarkenedDisk`, `likelihood_grid`/`linear_flux_grid`, `fit`,
  `laplace_cov`, `absil_limits`, and for orbits `KeplerOrbit`, `Attached`
  and `PositionData` (adding any missing nodes). `published-binary` stays
  the ESO-archive pipeline of `plan_eso_binaries.md`;
- `virgil.ensemble.draw_groups`, `run_group`, `combine` and `ensemble` as
  nodes, then a new `published-image` pipeline using them with `Image`,
  the regularisers and `l_curve`;

both in state `running` until O1 passes. Ledger entries for every
mismatch; a results page per goal on the docs site, carrying for each collection
the terms recorded in the manifest (thanks to the dataPI, the paper, the
OiDB acknowledgement and the CC BY-NC-SA notice). Results from an L2
collection are not published on the site until its dataPI has been
contacted and has agreed the policy.

## Decisions (2026-10-07)

Ben's answers to the open questions on PR #69:

1. **Licence and attribution.** Derived numbers and figures go on the public
   site with attribution. `DATA_ATTRIBUTION.md` records OiDB's CC BY-NC-SA
   4.0 licence, the L3 obligation (thank the dataPI, cite the paper), the L2
   obligation (contact the dataPI before presenting) and OiDB's
   acknowledgement sentence; its per-collection entries are filled in as O0
   runs.
2. **Scope.** Systems with orbits first: O1, then the O2 orbit systems. The
   detection-limit surveys (WR survey, A-type accelerators, SPHERE SAM, M17,
   massive YSOs, Gaia BH3) stay in the plan, marked "decide after the
   orbits".
3. **User-Agent.** The fetcher keeps a browser-like User-Agent string; JMMC
   is not asked.
4. **PIONIER L2 search.** Deferred until O1 works.
5. **Published images.** Authors are not emailed for FITS images; image
   checks compare with the published figures and quoted values, and Ben
   compares the images himself.
6. **Other imagers.** MiRA, SQUEEZE and the like are not run here either.
7. **Drevon's data.** R Scl and Drevon's Betelgeuse MATISSE sets are O3
   PYRA comparison targets.
8. **Rapid rotators.** Kept out of this repository (see "Rapid rotators").
9. **Clean binaries only** for pass/fail: post-AGB binaries and 89 Her
   dropped, ε Aur excluded, WR 104 moved to O3 imaging, and other systems
   with circumstellar material flagged "not clean".
10. **GRAVITY dynamical masses** (HR 6819, `696baf06-…`) moves into O1,
    straight after Gl 229.

## Order of work

O0 → O1 (items 1–7 in order) now, on virgil main. O2 and O3 can run in
parallel once O1 has passed. Downloads only from trevor via `scripts/oidb_dl`;
fits only on OzSTAR. About 40–50 agent hours for everything, of which O0–O1
is about 15.
