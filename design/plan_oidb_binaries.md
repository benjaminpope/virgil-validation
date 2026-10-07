# Plan: every single-epoch binary astrometry dataset in the JMMC OiDB

Stage O5 of `design/plan_oidb.md`. Companion to Track B of
`design/plan_eso_binaries.md`, whose per-epoch comparison it reuses. Criteria are
registered in `design/oidb_binaries_criteria.md`, commit
`e34f1351a21b75b0008cc6976debea8b203836ab`, SHA-256
`2420cdb1574da18c12e401b25a07d4594230086c5f7c4624640684a6d34b225d`. They were
committed before this plan and before any fit.

## Why

Track B compares virgil with published per-epoch positions on ESO Phase 3
products, which were reduced by a pipeline the authors did not use. The OiDB
holds the authors' own calibrated OIFITS, so here the files are the same and
only the fitting code differs. The OiDB also holds binaries from instruments the
ESO archive does not (CHARA MIRC-X and MYSTIC, NPOI, VEGA, SPICA). A published
position per epoch, with its uncertainty, is the simplest comparison there is,
so this plan asks for all of them, not only the orbit systems of `plan_oidb.md`
O1 and O2.

## (a) The census, public-safe

Built on 2026-10-08 from metadata only: the OiDB TAP service
(`tap.jmmc.fr/vollt/tap`, table `oidb`), a SIMBAD position cross-match, and ADS
bibcode records. Nothing was downloaded; sizes are `access_estsize`. The census
files are `~/data/oidb_census/census.{json,md}`, outside this repository.

How candidates were found. The OiDB has 24 758 rows at calibration level 1 or
higher (level 0 rows are observation logs), in 18 257 files and 97 collections.
SIMBAD flags 1054 of 2736 target names as binaries or multiples, but that screen
mostly catches calibrators and diameter targets. The candidates are the
collections whose paper reports a companion position, a detection or an orbit,
chosen from titles and abstracts. The ADS check is a full-text term screen: it
shows that a paper discusses separations, position angles and uncertainties, not
which table holds them. **Only Gl 229 and HD 45166 have had their positions and
uncertainties read from the paper (in O1). For every other row, a "probable" σ
means that the paper reports positions with errors, and OB0 must read the table
before anything is fitted.**

Totals:

- **Public (L3, released) collections reviewed: 24.** They hold 15 individual
  systems with 105 epoch-nights between them, 7 survey collections (145 target
  names, 79 SIMBAD-flagged, 120 seen on one night only), and 2 collections with
  no astrometry (Gaia BH3, an upper limit; symbiotic giants, diameters).
- **Published σ per epoch.** Confirmed for 2 systems (Gl 229, HD 45166). The
  paper reports positions with errors, table unread, for 8 systems and 4
  surveys. Unconfirmed for 4 systems and 3 surveys.
- **Instruments.** GRAVITY, MATISSE, PIONIER, AMBER, MIRC-X, MYSTIC, MIRC,
  CHARA H_PRISM and SPHERE SAM have candidates. NPOI (diameter surveys at L3,
  contest data at L2), VEGA (ε Aur and a Cepheid), MIDI (V² only, disks), IOTA
  (L2), SPICA (L2 only), JWST AMI, CLASSIC and CLIMB were reviewed and have
  nothing public.
- **Private (L2).** 19 collections, 12 574 files, 2089 target names, 713
  (collection, target) pairs flagged by SIMBAD, 402 of them on two or more
  nights. Counts only: names and ids are kept outside the repository. One L1
  collection (178 files) holds calibrators only. L2 stays out of the public
  record until its dataPI has been contacted (`plan_oidb.md`, Decision 1).
- **Excluded at L3.** 54 collections (disks, YSOs, AGN, diameter surveys,
  imaging, post-AGB binaries, ε Aur, 89 Her), with 563 target names of which 271
  are SIMBAD-flagged.
- **Size.** The 22 public candidate collections with astrometry total 1.32 GB in
  651 files: 216 MB for the individual systems and 1.10 GB for the surveys, of
  which the Wolf-Rayet survey is 0.78 GB.

Individual systems. Category A: an orbit or per-epoch system with few caveats.
Category B: not clean (disk, wind, hierarchy), reported and not scored. Epochs are
distinct nights among OiDB granules, which is not the file count.

| system | collection | instruments | files | epochs | σ | orbit | cat |
|---|---|---|---|---|---|---|---|
| Gl 229 Ba-Bb | 782185b2 | GRAVITY | 22 | 5 | yes | yes | A |
| HD 45166 | f4afc4cd | GRAVITY | 4 | 1 | yes | spectroscopic 22.5 yr; no interferometric orbit | A |
| iota Peg | fac164e1 | MIRCX | 24 | 5 | probable | yes | A |
| kappa Tuc A (Aa-Ab) | 855397ef | GRAVITY,MATISSE | 136 | 16 | probable | yes | B |
| GG Tau Ab1-Ab2 | 1e9bab59 | PIONIER | 3 | 2 | probable | yes | B |
| theta1 Ori C | 371c145f | AMBER | 2 | 2 | probable | yes | A |
| TZ For | J/A+A/586/A35 | PIONIER | 11 | 11 | probable | yes | A |
| SS Lep | J/A+A/536/A55 | AMBER,PIONIER | 8 | 14 | unverified | yes | B |
| gamma2 Vel | 534fbba6 | AMBER | 43 | 10 | unverified | yes | B |
| HD 174881 (HR 7112) | 45840351 | MIRCX-HPRISM | 2 | 2 | probable | yes | A |
| HR 6819 (QV Tel) | 696baf06 | GRAVITY | 12 | 12 | probable | yes | B |
| Polaris Aa-Ab | 337cfbf7 | H_PRISM,MIRCX | 28 | 8 | probable | yes | A |
| kappa Dra | 222e5f53 | MIRC,MIRCX,MYSTIC | 7 | 6 | probable | yes | B |
| rho Oph A | 512015b1 | GRAVITY | 8 | 7 | unverified | unknown | B |
| HD 123999 (d Boo) | J/A+A/597/A137 | PIONIER | 4 | 4 | unverified | unverified | A |

Surveys (category S; one table row per detected companion):

| survey | collection | instruments | files | nights | σ | orbit | cat |
|---|---|---|---|---|---|---|---|
| A-star companions (MIRC-X survey) | bda75673 | MIRCX | 28 | 3 | probable | no | S |
| A-type accelerators (GRAVITY) | be66c71c | GRAVITY,GRAVITY_FT | 44 | 10 | unverified | no | S |
| Wolf-Rayet survey (GRAVITY) | c3f46f3d | GRAVITY | 88 | 30 | probable | no | S |
| M17 O stars (GRAVITY) | 1b17307c | GRAVITY | 8 | 6 | unverified | no | S |
| Newborn Be + stripped systems (GRAVITY/MIRC-X) | 648ec766 | GRAVITY,MIRCX | 26 | 12 | probable | some | S |
| Be-star multiplicity (CHARA) | ddf733ce | MIRCX,MYSTIC | 134 | 30 | probable | some SB1 | S |
| SPHERE SAM transition disks | 0ab199d0 | SPHERE SAM | 9 | 1 | unverified | no | S |

Reviewed, no astrometry (category X):

| system | collection | instruments | files | nights | σ | orbit | cat |
|---|---|---|---|---|---|---|---|
| Gaia BH3 | 89aec164 | GRAVITY | 13 | 3 | no | yes | X |
| Symbiotic giants (MIRC-X) | 73adfa26 | MIRCX | 4 | 3 | no | some | X |

## (b) Preregistered criteria

Registered in `design/oidb_binaries_criteria.md` (hash above), committed before
this plan was written and before any fit. In short:

- **Reused from Track B.** d² = Δᵀ(C_v + C_p)⁻¹Δ with the larger of the scaled
  and unscaled virgil covariance; flags at d² > 13.816; a per-system test of Σd²
  against χ² on 2N degrees of freedom at p < 0.001; an overall one-sided KS test
  against χ²₂ at p < 0.01; the 180° rule for near-equal pairs; the convention
  refits (u → −u, λ × 1.01, λ × 0.99, mirrored position) and 2Δloss at the
  published position, to separate a finding from a definition difference.
- **Where Track B's criteria do not fit.** Seven epoch classes decide what is
  counted: `scored`, `no_sigma`, `smeared`, `third_body`, `resolved`,
  `not_clean` and `limit`.
  - *No published σ.* The epoch is reported, never given an assumed σ, and does
    not enter the tests.
  - *Calibration and observables.* The files are the authors', so calibration is
    shared. The observables and wavelength window are the paper's; when the paper
    is silent, the worse of the closure-phase-only and all-observable fits
    counts.
  - *Bandwidth smearing.* Epochs with s·B/(λR) > 0.1 count only if the model
    smears at the file's own resolving power.
  - *Triples and resolved components.* They are fitted as the paper fitted them.
    A paper that does not tabulate the pair's position gives no counted epoch.
  - *Not clean systems* (disk, wind, dust) are reported against the paper's own
    binary + disk solution and never counted.
- **Two registrations.** The criteria file is hashed now. OB0 then commits one
  reference record per system (table, column, uncertainty convention, model,
  epoch MJDs and class), and the hash of that set is entered here before the
  first fit. The number of counted epochs, and hence the expected chance flags,
  is fixed by it.

## (c) Loader readiness

From census metadata only (no files read). Each item is classed as a **virgil**
issue (the package has to change) or a **script** fix (in the campaign script
`scripts/oidb_binaries_fit.py`, built on `scripts/oidb_fit.py`'s `load`).

| blocker | census evidence | class | fix |
|---|---|---|---|
| MIRC-X and MYSTIC closure-phase rows whose baselines have no V² row | 20 candidate files: 13 of the 28 A-star survey files and 7 of the Be survey; 2 MIRC-X rows in the OiDB have T3 and no V² at all. Track B met the same in an AL Dor file (a triangle naming baseline 3–4) | virgil | compute the T3 uv points from the table's own U1/V1/U2/V2 (the third baseline is their sum), not from the V² baselines. Until then the script drops the unmatched triangles, counts them, and flags the epoch |
| Several targets in one file | 27 candidate files (Gl 229 22, Gaia BH3 3, newborn Be 2). OiDB target names have 2 to 5 spellings for κ Tuc, γ² Vel, HR 6819 and SS Lep | script | `target=` selection by regex or alias list in the reference record, as `oidb_fit.pick_target` does for Gl 229. The coordinates are a check |
| Several wavelength tables in one file | 220 of 680 candidate files (for example GRAVITY SC and FT) | script | `insname_prefix`, as Track B and `oidb_fit.load` do. GRAVITY_FT-only rows (4 to 5 files of the accelerator survey) are used only where there is no SC table, and flagged |
| Several nights in one file | 9 candidate files (SS Lep 5, Polaris 2, symbiotic giants 2); 367 of 18 257 files in all of OiDB | script | split by MJD window with `OIData.select`, matching published epochs within 0.5 d (criteria rule 7) |
| VizieR collections link to the CDS FTP, not OiDB staging | TZ For, SS Lep, HD 123999 (3 collections) | script | `fetch_oidb.py` accepts the `cdsarc` host, with the size from the metadata |
| Instrument-name variants | MIRCX, MIRCX-HPRISM, H_PRISM, MYSTIC; SPHERE-IFS_1 to _40 | script | normalize to a family for the per-family statistics; take the resolving power from each file's own `EFF_BAND` and `EFF_WAVE` |
| SAM files (SPHERE) have V² and T3 counts that differ from the VLTI pattern | all 16 SPHERE and SPHERE_1 rows have fewer V² than the triangles imply | script, to check | OB1 reads one file's table headers and confirms before any fit |
| No bibcode in the OiDB | ρ Oph A, symbiotic giants, the ι Peg workshop copy | script (reference record) | OB0 finds the paper |
| A wavelength-dependent flux ratio | κ Tuc A (MATISSE), γ² Vel | virgil (model), not a loader fault | `spectra.PowerLaw` exists; the reference record states the paper's choice |

Only the first item needs a virgil change. If virgil does not take the T3
table's own uv points by OB1, the 20 affected files are fitted with the script's
workaround and reported as such.

## (d) Dedupe

- **Track B (20 systems).** One public overlap: TZ For. Its 11 PIONIER nights
  here are the 2014–15 epochs that Track B cannot compare for lack of Phase 3,
  so this is the same system on different data, not a repeat. Twelve of the 20
  Track B systems also have private L2 files in the OiDB (PIONIER 10, ISSP 2,
  and IOTA for α Equ again). They hold the authors' own calibration of systems
  that Track B fits from Phase 3, but they are private and enter this plan only
  as counts (decision D2: not pursued for now).
- **ESO binaries set.** Gl 229 only, public L3 (O1 already fits it) and
  PIONIER L2. The Nowak et al. 21 systems and the other orbit benchmarks
  (HD 136164, δ Vel) are not in the OiDB. 9 Sgr is, in PIONIER L2.
- **O1 of `plan_oidb.md`.** Gl 229, HR 6819, ι Peg, HD 45166 and the five A-star
  detections are O1 fits with their own thresholds (0.25σ_pub). They are refitted
  here with the Track B statistic on the same files, so nothing is downloaded
  twice and the O1 results are not replaced.
- **Within the OiDB.** The workshop L2 ι Peg night (2018-10-22) duplicates one
  night of `fac164e1`; one fit serves both and is reported only for the public
  copy. κ Tuc is split across four target spellings and two instruments, and
  HR 6819 across two. The census counts systems, not names.

## (e) Compute plan (OzSTAR; Ben runs it, agents do not touch the cluster)

Downloads first: the 22 public candidate collections are 1.32 GB in 651 files,
fetched by the existing `oidb_dl` job on trevor, one task per collection, with
sizes from the census; none is above 0.8 GB. Nothing is committed to git.

The fits are a CPU array. Each epoch is one grid, five peak fits and one Laplace
covariance, with no NUTS. The O1 task that timed out at 12 h was a Keplerian
orbit under NUTS, which this campaign does not run. The caps keep each task
short:

- at most 24 spectral channels and at most 1.5 × 10⁶ position cells per epoch,
  windowed beyond that (criteria, search section);
- at most 40 min per epoch, after which the epoch is `not fitted: budget`;
- each epoch writes its own JSON, so a task is resumable, and a task stops
  itself at 5 h of its 6 h limit;
- 8 CPUs and 16 GB per task.

| group | tasks | epochs | estimate per task | total |
|---|---|---|---|---|
| A systems (HD 45166, ι Peg, θ¹ Ori C, TZ For, HD 174881, Polaris, HD 123999) | 7 | 33 | 0.3 to 2 h | 6 h |
| B systems (κ Tuc A in 2 tasks, GG Tau, SS Lep, γ² Vel, κ Dra, ρ Oph A, HR 6819) | 8 | 67 | 1 to 4 h | 20 h |
| A-star survey | 3 | 27 targets | 1 h | 3 h |
| A-type accelerators | 2 | 11 targets | 1 h | 2 h |
| Wolf-Rayet survey | 5 | 42 targets | 1.5 h | 8 h |
| M17, SPHERE SAM | 2 | 13 targets | 1 h | 2 h |
| Newborn Be and Be multiplicity | 7 | 52 targets | 1.5 h | 10 h |
| **total** | **34** | | | **about 50 task-hours, 6 h wall** |

These estimates are first-principles upper bounds, not measurements. The Track B
job (18210048) is the same kind of per-epoch fit on GRAVITY and PIONIER, so its
`sacct` elapsed times per epoch replace these numbers once Ben has them. The
array is launched in two waves: the A and B systems first, the surveys after
they pass the loader checks.

## (f) Ownership: resolved

Agreed with the orbit-fitting session on 2026-10-08.

- **Per-epoch positions are ours.** They come from `likelihood_grid`, `fit` and
  `laplace_cov`, as in Track B, and not from the orbit session's epoch-position
  scorer (D1, virgil #298, still a draft). That keeps this campaign independent
  of the orbit machinery.
- **Orbits are theirs.** The joint orbit fits, and predictions for later nights,
  belong to the orbit session. Track B's per-epoch job (18210048) and its
  criteria remain ours.
- **Staging.** The census lists every public system with a known orbit. Those
  with at least two epochs in the archive are staged for the orbit session in
  `~/data/oidb_census/orbits/<system>/system.json`, in the layout of
  `~/data/orbit_benchmarks`: granule lists (file, MJD, instrument, wavelength
  range, size; no downloads), epochs, instrument and band, the reference orbit
  with its source, and caveats. Ten are staged: ι Peg, θ¹ Ori C, TZ For,
  HD 174881, HR 6819, κ Tuc A, Polaris, κ Dra, γ² Vel and SS Lep. Gl 229 is
  already theirs. The staged files say that orbital elements are not yet
  transcribed: OB0 reads them from the papers, with the position tables.

## Stages

- **OB0 (about 15 h, laptop).** Read the position table of every candidate
  paper. Write the reference records and epoch classes, transcribe the orbit
  elements for the staged systems, hash the set and enter the hash here. This
  replaces the "probable" and "unconfirmed" entries with read ones and drops
  surveys whose paper reports no companion. Gate: no fit before this commit.
- **OB1 (about 6 h).** `scripts/oidb_binaries_fit.py`, tests for the ellipse
  conversion and epoch matching, the loader fixes of section (c), one-file header
  checks for SAM and GRAVITY_FT, and the virgil issue for the T3 uv points.
- **OB2.** The download and the fit array of section (e), in two waves.
- **OB3 (about 4 h).** Summary, flags, finding or definition for each flag,
  ledger entries (`trust/ledger.yml`), and the trust-graph edit that extends the
  `oidb-binary` pipeline with the per-epoch chain, made before the first test
  names it.
- **OB4.** A results page with each collection's terms (thanks to the dataPI,
  the paper, OiDB's acknowledgement, CC BY-NC-SA 4.0). No L2 results.

## Decisions

Decided by Ben, 2026-10-08.

- **D1. O1 systems are refitted with the Track B statistic: yes.** ι Peg,
  HD 45166, HR 6819 and the A-star detections are refitted here on the same
  files, as planned. The O1 comparisons stay as they are and are not replaced.
- **D2. dataPIs of private files are not contacted yet: deferred.** The 12
  Track B systems with private L2 files (PIONIER, ISSP, IOTA) stay out of this
  work, and no second private check is made. The current focus is public ESO
  archive and JMMC data only. Revisit under `plan_oidb.md` Decision 1 once the
  public results are in.

## Open questions

1. Orbits: may the staged systems also become orbit tests (the orbit session
   decides), and should they wait for OB0's transcribed elements?
2. Which survey targets count? A target counts only when its paper reports a
   companion with an uncertainty. The census cannot tell detections from
   non-detections among the survey target names until OB0 reads the tables.
3. The SIMBAD screen is coarse. After OB0, is a second pass over the 54 excluded
   L3 collections worth about 2 h, to catch a binary table in a disk or diameter
   paper?
