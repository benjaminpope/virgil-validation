# Real-data parity and a real-data null test

Criteria: `design/realdata_parity.yml`, registered on 2026-10-12 before any run. Its SHA-256 is in
`design/realdata_parity.sha256` and checked by `tests/test_realdata_design.py`. This file is the
reasoning. Nothing here has been run: no data were downloaded and no external package was run
(Part A and B need data and installs; see "What was checked").

## Why

The review found that real data are essentially unvalidated:

- Only 4 evidence records use the `literature` root (`trust/evidence/latest.jsonl`): the
  imaging-contest test binaries of 2004 to 2010, which are simulated.
- The ESO verdicts (`plan_eso_binaries.md`, Stage E4 and the Gl 229 grid fits) are provisional,
  pending a rerun after the `likelihood_grid` argument-order change.
- Real GRAVITY fits show raw closure-phase chi2/N of 3 to 606, absorbed by a free `phi_scale`
  (`plan_eso_binaries.md`, Stage E3; HD 45166's calibrated files gave 1234 in `oidb_fit.py`).
  A fit that rescales its errors to chi2/N = 1 says nothing about whether the model is right.
- Our simulated noise is the likelihood's own model at high S/N, so pulls are close to
  tautological. Four things are untested: low-S/N non-Gaussian closure phases,
  wavelength-correlated errors, calibration errors and a wrong sigma.

So this design does what a simulation cannot: it fits the same real file with several codes
(A), measures how often virgil's detection fires on real files with no companion (B), and lists
the simulator extensions that the first two will show to be needed (C).

## Part A: same-file parity

### What parity does and does not show

The papers' positions came from other codes on the authors' own reductions, which Track B and
the OiDB plan already compare against. Agreement with a published number mixes the code, the
data reduction and the model. Fitting the **identical file with the identical model** removes
the reduction and the model, so any difference is the code: conventions, the likelihood, the
optimiser, or the error bar. It is still **not** a test against truth, and three codes sharing a
misconception (for example, that a real closure phase's noise is Gaussian with the quoted sigma)
would agree. Part B and Part C address truth; Part A addresses the machinery. That is stated in
the design so a PASS is not read as more than it is.

### The conventions that must be mapped

All pinned in `docs/method/{pmoired,candid,fouriever}.md` and their tests. In the order they
matter for real files:

1. **Same data in.** The parity file is cut once by one script (target, INSNAME, spectral
   window, unmatched closure triangles) and hashed. Each code reads it with its own reader and
   may not select or flag data. The sample counts per code must be equal (A2), because a code that
   silently drops a flagged channel is the commonest cause of a parity difference on real files.
2. **Error bars.** PMOIRED multiplies its uncertainties by sqrt(chi2_red) by default; CANDID
   by sqrt(chi2_r) with chi2_r = chi2/(N - n_fit + 1); virgil's Laplace covariance is the plain
   curvature. The bridges already undo the first two. On real data the question is sharper: virgil
   as a user runs it fits `vis_scale` and `phi_scale`, which inflate the Laplace errors by the
   scales. So there are two virgil arms. **V0** has quoted errors, no scales and independent
   closure-phase residuals, and is compared with the externals' un-normalised errors; this is the
   strict, scored comparison. **V1** has the free scales, is compared with the externals as shipped
   (which are normalised by sqrt(chi2_red), the same thing in spirit), and has one scored limit:
   V1/V0 errors must scale with the fitted scales (A5). V1 is where a scale of 2 or 600 shows
   as a flagged epoch, with the raw chi2/N alongside.
3. **Closure-phase correlations.** With 4 or more telescopes virgil whitens the closure phases of a
   snapshot as a correlated group; PMOIRED and CANDID treat them as independent, which makes their
   errors 4 to 6 percent smaller (`docs/method/pmoired.md`). V0 therefore sets `cp_noise` to
   `None` for the comparison with them (to be verified on a simulated file in the first run), and
   V0c keeps virgil's default and is compared with fouriever, which models the same correlation
   (its pseudo-inverse and virgil's whitened inverse differ by a few percent with unequal errors:
   ledger D6). Wider bands are registered for 4T+ files.
4. **Closure-phase residual.** Chord (virgil), plain difference (PMOIRED, fouriever), wrapped
   plain difference (CANDID): ledger D5, 1e-6 relative near a good fit, up to 1e-3 far from one.
   Real fits with chi2/N of 600 are far from a good fit, so the same-parameter chi2 tolerance
   (A2) is 5e-3 there, not 1e-3.
5. **Flux and position.** virgil's `flux` is companion/primary; PMOIRED's `f` with f1 = 1; CANDID's
   `f` is a percent. East and North, and the T3 sign, are pinned. De Furio et al. and Anugu et al.
   define flux ratios differently from each other (percent of primary; F_pri/sec): the reference
   records already carry that, and `oidb_compare` handles it.
6. **Models.** CANDID has only a UD primary and a point companion (plus resolved flux `fres`). The
   common model M1 is that. The papers' models (fixed UDs, two points) are M2, scored against
   the literature only, and PMOIRED covers them too. A single star (pi1 Gru) is M3.
7. **Smearing.** Off in every code for the scored comparison. CANDID's smearing averages V^2 and
   the bispectrum over 3 points, ours the complex visibility over 7: a definition difference, so
   the smeared variant is reported and compared with the paper at 2 sigma, as the OiDB plan does.
8. **PMOIRED defaults.** `auto=False` (problem P1: default setup gives NaNs on few-channel files
   with a small `EFF_BAND`); error floors set to zero. Both to be confirmed in the dry run.

### Targets

12 targets, 36 planned fits, from what is already in the repository (`oidb/manifest.yml`, the
reference records and the plans). The epochs are chosen by rule (first, middle, last, or all) so
they cannot be picked after seeing a fit.

| target | instrument | epochs | model scored | literature | externals |
|---|---|---|---|---|---|
| Gl 229 Ba-Bb | GRAVITY | 5 nights, one file each | M1 point primary; M2 two points | Xuan+2024 orbit (reported) | PMOIRED, CANDID, fouriever |
| HD 45166 | GRAVITY | 2023-11-26, uncalibrated science CP | M1, M2 | Deshmukh+2025, 10.886 +/- 0.082 mas (adopted, 1 sigma) | PMOIRED, CANDID, fouriever |
| HR 6819 (not clean) | GRAVITY HR | first and last | M1, M2 | Klement+2025 (reported) | PMOIRED, CANDID, fouriever |
| pi1 Gru | PIONIER | the one file | M3 | Paladini+2018 UD 18.37 +/- 0.18 mas (2 sigma) | PMOIRED, CANDID (UD scan) |
| TZ For | PIONIER | first, middle, last of 11 | M1; M2 with UDs 0.414, 0.197 | record pending | PMOIRED, CANDID |
| d Boo (HD 123999) | PIONIER | 4 nights | M1 | record pending | PMOIRED, CANDID |
| iota Peg | MIRC-X | 5 nights (one at 1.13 mas) | M1; M2 with UDs 1.05, 0.6 | Anugu+2020 Table 2 and CANDID columns (0.25 sigma) | PMOIRED, CANDID, fouriever |
| A-star companions | MIRC-X | HD 5448, 11636, 28910, 29388, 48097 | M1 with `fres` | De Furio+2022, CANDID on the same files (2 sigma) | PMOIRED, CANDID |
| HD 174881 | MIRC (H_PRISM) | 2 nights | M1 | record pending | PMOIRED, CANDID |
| Polaris Aa-Ab | MIRC / MIRC-X | 3 of 8 nights | M1 | record pending | PMOIRED, CANDID |
| theta1 Ori C | AMBER | 2 nights | M1 | record pending | PMOIRED, CANDID |
| NACO SAM (tier S) | NACO (AMICAL) | 9 Sgr, delta Vel, HD 136164 | M1 | orbit predictions (reported) | PMOIRED, CANDID |

Instruments: GRAVITY, PIONIER, MIRC, MIRC-X, AMBER, NACO SAM. The OiDB L2 and private
collections stay out (`DATA_ATTRIBUTION.md`). The NACO SAM rows are tier S, not in the verdict: their
AMICAL reductions are provisional and AMICAL 1.6.0's calibrator weighting biases V^2
(SAIL-Labs/AMICAL#332), so closure phases only are scored there.

The A-star detections are the sharpest test: the authors' published values *are* CANDID run on
these files, with separation errors of 0.002 to 0.007 mas. If virgil, PMOIRED and CANDID disagree
at 0.25 of that error, a convention or a data-selection difference has been found; the 2 sigma
literature rule only reflects the smearing definition.

### Criteria, in words

(Numbers in the YAML.)

- **A1 best fit**: per parameter, |virgil - X| / sigma_X at most 0.25 for 90 percent of
  comparisons, none above 1 sigma unless the mode rule explains it. A comparison above 1 sigma is
  not simply a failure: virgil is restarted at X's best fit and X's loss evaluated at virgil's, so a
  search miss, two local minima and a convention difference are told apart (the convention-refit
  procedure of the OiDB plan), and the code with the deeper minimum on the common chi2 wins.
- **A2 chi2**: the same total chi2 at the same parameters in all four codes (1e-3 relative, 5e-3 when
  chi2/N > 10), at each code's own best fit within 1 percent, and the same N exactly.
- **A3 error ratio**: sigma_virgil(V0)/sigma_X in [0.90, 1.10] for 90 percent of comparisons, none outside
  [0.75, 1.33]; wider for 4T+ files.
- **A4 literature**: `scripts/oidb_compare.CRITERIA` unchanged; 0.25 sigma per epoch, 2 for parametric, 1 for
  an adopted value; reference records pending are reported only; NACO orbit predictions reported.
- **A5 scales**: V1 errors within 25 percent of V0 times the larger fitted scale; any fitted scale above 3
  is listed as a model or error failure.
- **Verdict**: PASS needs A1 to A3 and A5 for both PMOIRED and CANDID, plus A2 for fouriever where it
  applies; INCONCLUSIVE if under 80 percent of fits complete; each failure is a finding unless ruled a
  definition difference or an external's own problem.

The registered tolerance of 0.25 sigma is the OiDB plan's per-epoch tolerance, and the simulated
Stage 2 result (under 0.25 sigma, typically under 0.1, `docs/method/pmoired.md`). Real data are
less kind: non-Gaussian low-S/N closure phases, a chi2/N far above 1 and multimodal likelihoods
all make two correct codes disagree more. That is why 10 percent of comparisons may exceed it and
why the mode rule exists. If the real-data spread among the externals themselves (PMOIRED vs
CANDID on the same file) already exceeds 0.25 sigma, the run reports that spread and the
tolerance is judged against it in the report; the registered tolerance is not changed.

### Gaps in the external bridge

What `external_bridge` already supports: PMOIRED `fit` (maps virgil's parameters, undoes the
sqrt(chi2_red) normalisation); CANDID `fitmap` (best fit, both uncertainties, nsigma), `chi2`,
`chi2map`, `nsigma`, limits and exact-limit tasks; fouriever `chi2`, `chi2_grid`, `detlim`,
`limits_exact`; and `scripts/oidb_fit.py` / `scripts/oidb_compare.py` for the virgil side and the
published-value comparison (reused unchanged). What is missing, in the order it is needed:

1. A harness script (`scripts/realdata_parity.py`) that cuts and hashes the parity file, runs the
   arms and codes, and writes one JSON per epoch.
2. A PMOIRED chi2-at-given-parameters helper (`doFit` with every parameter fixed).
3. CANDID `ud_fit`: minimise `chi2` over `diam*` with f = 0 (for pi1 Gru).
4. A fouriever `fit` task; until then it is only a chi2 evaluator at the other codes' best fits.
5. The reference records still pending (TZ For, d Boo, HD 174881, Polaris, theta1 Ori C): OB0 of
   `plan_oidb_binaries.md`. They make those targets parity-only until written.

## Part B: a real-data null test

### Question and population

On real calibrated data with no companion, how often does virgil's detection statistic exceed
the thresholds that virgil's own Monte Carlo sets for a given false-alarm probability (FAP), with
and without `rescale_errors`? This is the empirical calibration of the pipeline's FAP on real noise,
which the simulated campaigns (`detection_v2`) cannot give because their noise is the model.

**Selection rule**, fixed and committed before any file is fitted (a second registration of the
list of observation ids with hashes): released (L3) OiDB data or ESO Phase 3 calibrated OIFITS, at least 3
telescopes, at least 60 closure-phase samples; exclude SIMBAD binaries, spectroscopic or eclipsing
binaries, Be and emission-line stars, WR, YSO, post-AGB and Mira types, anything in WDS, ORB6 or
SB9, Gaia DR3 RUWE above 1.4, anything with a published companion, and anything whose name carries a
component suffix. L2 stays out.

Three strata: **S_clean** (calibrators and single stars) is the primary set; **S_survey** (science
stars with a published non-detection) may hide a faint companion, so an excess confined to it is
"possible real companions", not charged to virgil; **S_stress** (Be, WR) shows the false-alarm rate
when the null model is wrong, reported only.

**Candidates found in the repository** (to be pruned by the rule):

- The MIRC-X A-star collection (De Furio+2022, `bda75673`): 28 files in 27 OI_TARGET names. Without
  the 5 published detections, omega Aur and the SIMBAD-flagged, about 21 remain. HD 99787, HD 21912, HD 120047,
  HD 32301 and HD 74198 are the paper's own calibrators (Table 2), so S_clean; the rest S_survey.
  The paper's 5-sigma detection limits (Table 5) give an external check on the same files.
- The GRAVITY calibrators of Track B (`gravity_calibrators.json`, 2.8 GB, not downloaded).
- pi1 Gru's PIONIER calibrators HD 209688 and HD 215104, and the OiDB's L1 calibrators-only
  collection (178 files; id in the private notes; dataPI terms permitting).
- NACO calibrators from `plan_eso_binaries.md` E4 (HIP 74479, HIP 74865, HD 152249, delta Phe),
  calibrated against another calibrator so the closure phase is that of a point source. Achernar
  is excluded: a rapid rotator with a known companion.
- TYC 732-806-1, the HD 45166 calibrator, which is a 3.3 mas binary: not a null but the positive control.

The collections give MIRC-X (H), GRAVITY (K), PIONIER (H) and SAM (L', Ks) as the four families,
so the minimum of 4 instrument families in the design is attainable. The minimum size is 60 nulls,
30 of them in S_clean; below it the result is INCONCLUSIVE and reported. The registered selection
query is the first thing the run does; if the pool is smaller than that, the design says so rather
than relaxing the exclusions.

### Procedure

Per file: load with the parity windows; fit a UD null (the diameter free); search a coarse grid
(H = 30 mas, at most about 4e4 cells, flux geomspace(1e-4, 0.3, 40)) with
`detection.detection_statistics`; build the null distribution from `injection_recovery` with
`gaussian_null` and 2000 draws; convert the observed `delta_chi2` to its FAP (look-elsewhere
included by construction). The coarse grid is the registered price of the Monte Carlo: a finer grid
for the observed value alone would not match the thresholds. Variants:

- **raw**: quoted errors;
- **rescaled**: `rescale_errors(null, data)` (each block's null chi2/n = 1);
- **bootstrap** (reported): `bootstrap_null` sign-flip thresholds, which remove a pure scale error and
  keep heteroscedasticity;
- **CANDID nsigma** (reported): a second code's view of the same file.

Rescaling does not make the test tautological. It removes the mean error scale; what is left to
fail is the shape of the residuals, their correlation across channels and baselines, outliers,
and the closure-phase wrapping. That is what the real-noise test is about.

### Criteria

N1: one-sided exact binomial count of files with FAP below alpha (0.05 and 0.01). N2: one-sided
Kolmogorov-Smirnov of the FAPs against uniform (too many small ones). Both at p < 0.01, one Holm family of six p-values
(4 counts: 2 levels x raw and rescaled; 2 PIT tests); the verdict uses the rescaled variant on S_clean. The raw variant is judged by the same tests and decides one documentation
statement (whether `rescale_errors` is needed on real data), not the verdict. A FAP deficit
(conservative) is reported, not a failure. **A positive control guards against vacuous passes**: at
least 5 of the 6 known companions (the five De Furio detections and TYC 732-806-1) must exceed the
FAP 1e-3 threshold, or the null test says nothing.

**Power.** With 60 independent nulls the count test at 0.05 fails from k = 8 (expected 3.0) and has
power 0.25, 0.70, 0.99 against an inflation of the FAP by 2, 3, 5. At 0.01 it fails from k = 4
and has power 0.03, 0.11, 0.35: nearly blind, reported for its interval. The PIT test carries the
weight for smaller inflation. The report states the inflation that is excluded and nothing finer,
as the detection campaign states its precision.

### Cost

Measured on a laptop (single process, eager) 2026-10-12: 0.16 s per null draw for a 4-UT file with 120
V^2 and 80 closure phases on a 17 x 17 grid, 0.32 s on 41 x 41. A MIRC-X file (15 baselines, 8 channels)
and a 4e4-cell grid is 5 to 20 times heavier, so 1 to 5 s per eager draw, and the compiled
`injection_recovery` kernel should be several times faster. 60 files x 2000 draws x 2 variants is 70 to
330 CPU-hours eager, 15 to 70 compiled: an OzSTAR array of 60 tasks of 1 to 5 hours (or GPU). Not
a laptop job. Downloads: the A-star files are already in Part A (25.5 MB); the rest by the selection
query, estimated 100 to 300 MB.

## Part C: misspecification, later campaigns

Not registered, only scoped. Each needs its own registered file before it runs, and its magnitude is
set by the real data rather than guessed: Part A gives fitted error scales and raw chi2/N, Part B the
FAP excess, and a campaign injects the misspecification at the size that reproduces those numbers.

- **C1 non-Gaussian low-S/N closure phases**: the argument of a bispectrum average (wrapped, heavy
  tailed, uniform below S/N of about 1 per bispectrum) instead of a Gaussian with sigma. Simulator
  extension in `crosscheck/simulate.py`. Measure the pull standard deviation, coverage and the `delta_chi2`
  tail, and what the chord and periodic-penalty residual do about it.
- **C2 wavelength-correlated errors**: correlation across channels with a correlation length of 0.5 to 8
  channels, shared by V^2 and closure phase through the transfer function. Measure the ratio of true
  to quoted error against the effective number of independent channels, and what the free
  `phi_scale` does to it.
- **C3 calibration errors**: per-baseline per-night multiplicative V^2 errors (transfer-function drift,
  calibrator diameter error) and closure-phase offsets, with virgil's gains and phase offsets
  marginalised or not. Measure bias in position and flux ratio, coverage, the file's own chi2/N and the FAP.
- **C4 wrong sigma**: true noise k times quoted, k in {0.5, 1, 2, 4}, uniform and varying across baselines
  and channels (`gaussian_null(error_scale=k)` already exists for the detection part). Measure coverage with
  free scales, with `rescale_errors` and with neither, and the FAP.
- **C5 unmodelled structure**: a faint companion or a disk in the "null", the realistic null of a survey,
  because S_survey and S_stress cannot separate it from the above.

Cost: one to two sessions of simulator and test work each, and OzSTAR CPU time of the order of the
detection campaign (1e4 to 1e5 draws).

## Cost of the whole design

- **Setup**: PMOIRED (pip), CANDID (`scripts/setup_candid.sh`) and fouriever (`scripts/setup_external.sh`), about 30
  minutes once. None is installed in this checkout.
- **Development**: the parity harness and the four bridge additions, 1 to 1.5 days; the null harness,
  about 1 day; the reference records for the five pending targets, half a day; analysis and the
  report, half a day.
- **Compute, Part A**: 36 fits. Estimated virgil 5 min (grid then fit) each, PMOIRED 2 min, CANDID
  `fitmap` 5 to 20 min on a 60 mas search, fouriever 1 min: about 8 to 15 CPU-hours, as an OzSTAR array
  of 36 tasks of 30 minutes at most, or about 15 hours of a laptop (tasks under 10 minutes each).
  These are estimates, not measurements, because no external package could be run here.
- **Compute, Part B**: 15 to 330 CPU-hours (above); OzSTAR.
- **Downloads**: about 150 MB for Part A, 100 to 300 MB more for Part B. Ben runs the fetch
  (`scripts/fetch_oidb.py`); nothing was downloaded for this design.

## What was checked while writing this

- virgil 0.3.0 and `virgil.detection` (`gaussian_null`, `rescale_errors`, `bootstrap_null`, `injection_recovery`,
  `DetectionMC.threshold`, `false_alarm_probability`) import and run; timing above.
- The external bridge reports PMOIRED, CANDID and fouriever as **not installed** here
  (`external_bridge._subprocess.available`), so "the tooling works" is verified for virgil and the
  bridge's code paths by reading, not for the externals. The first step of the run is to install
  and re-run `tests/test_candid.py`, `tests/test_fouriever.py` and the PMOIRED tests.
- The design reuses `scripts/oidb_fit.py` (`load`, `fit_epoch`, `raw_chi2`, the recipes) and
  `scripts/oidb_compare.py` (`CRITERIA`) unchanged, so the virgil side of Part A is the code
  that already ran on OzSTAR.

## Open questions for Ben

1. Is NACO SAM to be in the first verdict? It is tier S because the AMICAL fits are provisional.
2. May the L1 calibrators-only OiDB collection be used for S_clean (dataPI terms)?
3. Should there be a ledger ID reserved for these findings? None is reserved for this item; a
   finding would take the next free one.
