# Handover: why the informed contest fits miss the truth

Written 2026-11-12 by the cloud session that built `tools/mandate` and `campaigns/`, for a local session that
can read the truth images. The cloud session could not read them.

## Task

Ben's request: go through the truth images and information, compare them with what mandate and the code
produce, and diagnose why the informed contest fits do not recover the truth. The cloud session answered from
the truth *parameters* the papers publish. The comparison with the truth *images* is still to do, and is the
first job here.

## State of the repositories

- `virgil-validation`, branch `claude/quirky-hopper-v32dq0`, draft PR #246: replaces
  `campaigns/contests/first_run/` with `campaigns/contests/informed_run/` (17 dataset reports and a table),
  plus this note. Green before this note was added. Not merged.
- `virgil`, branch `claude/quirky-hopper-v32dq0`, PR #706: adds `ArchimedeanSpiral`. Green, review threads
  resolved. Not merged. The 2024 pinwheel family still uses three untied arcs.
- On `main` already: `tools/mandate/` (PR #243), `campaigns/orbits/`, `campaigns/contests/`,
  `scripts/contest_informed.py`, `scripts/contest_informed_quicklook.py`.
- Nothing from the diagnosis below is committed as code. The test scripts were in the cloud scratchpad and
  are gone; each test is described well enough here to rewrite in a few lines.

## Where things are

| What | Where |
|---|---|
| Truth images | `~/data/imaging_contests/bench/<case>/truth.npz`, `~/data/imaging_contests/2026/truth/` (local only) |
| Published truth parameters, metrics, score tables | `design/contest_scoring_sources.md` |
| Earlier convergence diagnosis (blind images) | `design/contest_convergence_diagnosis.md` |
| Informed model families and the fit ladder | `scripts/contest_informed.py` (`FAMILIES`, `prior_of`, `fit_rung`) |
| What each family was told | `campaigns/contests/informed_models.yml`, `notes/informed_<year>.md` |
| Generator of registries and notes | `campaigns/build_registries.py` (reads `contests/manifest.yml`) |
| Fit results | `campaigns/contests/informed_run/*.json` and `README.md` |
| Ben's feedback dashboard | local session "Contest fit-advice dashboard (local prototype)", worktree `gracious-lovelace-ac14c5`, `scripts/contest_advice.py` (not on `main`; session identity not confirmed) |
| 2004 data2 truth rebuild | `scripts/score_2004.py` (read its docstring) |

## What was tested, and the results

Each test built the paper's model in virgil and evaluated it on the contest data with
`virgil.likelihood.whitened_residuals`, or fitted it with `virgil.fitting.fit` from stated starts. Data were
loaded as `contest_informed.run` loads them, without the flux spectra, and models were grey. All on a 4-core
CPU.

| Dataset | Our fit | Test | Result | Cause |
|---|---|---|---|---|
| 2022 GRAVITY | star + halo + companion, chi2/N 4392 | `PointSource` + `ModulatedGaussianRim` (one harmonic) + `PointSource`, 12 starts at inc 50, PA 50, companion at +/-(2, 10) mas, ring diameter 6, 12, 20 mas | chi2/N 3.18 in 2 starts (18.1 in 2 more); inc 50.0, PA 50.0, companion (2.005, 10.03), ring diameter 17.4 mas, fluxes 18.4 / 68.6 / 13.0 % | wrong family |
| 2022 AMI | star + halo + companion at the field edge | `PointSource` + `GaussianArc`, 16 random starts | 9 of 16 starts agree: arc flux 16 % (paper 15 %), radius 115 mas, PA 42; arc width at its lower bound (10 mas) | wrong family |
| 2004 data1 | chi2/N 259; shell flux at its cap of 10x the star, modulation amplitude at 1 | same family, flux cap raised to 300x | chi2/N 22; ring flux 31x the star, inc 8, modulation 0.69 | prior excludes the truth |
| 2010 low H, K | companion at (-9, -67) and (24, 8) mas | limb-darkened disk + point started at the paper's (-83.3, -12.7) mas | stays there; flux 0.27 % (H), 0.42 % (K); star diameter 18.1-18.5 mas; chi2/N falls only from 6070 to 6048 (H), 7832 to 7674 (K) | prior stops at 68 mas |
| 2004 data2 | chi2/N 28; limb darkening 0, diameter 4.99 (paper 5.5) | published geometry with free fluxes; then a fit started at the truth | 45-80 at the published geometry; the fit returns to our optimum | data convention (below) |
| 2024 Obj2 | "companion" 0.6 mas from the star, flux at its cap of 1 | likelihood map for a point beside the three-ring fit | peaks at (20.2, 0.0) [PIONIER] and (-6.4, -7.7) [GRAVITY]; paper's planets at (20.5, -0.5), (-6.5, -7.5); each changes chi2/N by about 0.03 | component misused |
| 2006 | disk at PA 27, inc 27, chi2/N 272 | `EllipticalGaussian` + `GaussianDisk`, 6 random starts, size cap 18 and 150 mas | best 118-123 at PA 93-95, axis ratio 0.38-0.43, FWHM 18-20 mas; six starts gave four different optima | optimisation |

Not tested: 2008 (component parameters not public), 2024 Obj1 (needs `ArchimedeanSpiral` from virgil #706),
2018 (data not downloadable in the cloud), the MATISSE datasets.

One caution on the numbers: `whitened_residuals` gave chi2/N 20.3 for the 2004 data2 values that the fit
report gives as 28.4. The two count the data differently; this was not resolved. Compare like with like.

### Facts about the data that matter

- Short-baseline V2 is far below 1 for most datasets (2010 H 0.27, 2024 Obj1 0.29-0.33, 2008 AGB J 0.46,
  2022 GRAVITY 0.54, 2004 0.59-0.67). Much of the flux is on scales at or beyond lambda/B_min.
- 2006: V2 never exceeds 0.03. The object is almost wholly over-resolved; the clue image spans 106 mas and
  the informed fit assumed 18 mas.
- 2004 data2: the visibilities were normalised without the spot's flux, so V2(0) is about 1.145
  (`scripts/score_2004.py`). A unit-sum model cannot fit them.
- 2010: V2 errors are 1e-4 on every point; 2022 AMI closure-phase errors are 0.0012 deg. chi2 on quoted
  errors ranks nothing sensibly there.

## Diagnosis

1. The informed tier holds one-line labels from `contests/manifest.yml` ("a high-contrast test"), not what
   the papers say. The fuller descriptions were already in `design/contest_scoring_sources.md`. Nobody signed
   the notes off.
2. The prior scales in `prior_of` were invented by the cloud session and exclude the truth: flux at most 10x
   the star, positions within 1.5 lambda/B_min, sizes within lambda/B_min. The blind track's field settings
   (`field=106` for 2006, `field=30` for 2008 in `contest_images.TASKS`) are not passed to the informed fit.
3. Knowledge found elsewhere in the repository does not reach the fits: the 2004 normalisation, and the
   finding that CLEAN-started GP images reach chi2/N near 1 on these data. Parametric fits are not seeded
   from those images.
4. The landscapes are multimodal and 4 starts is too few. Started near the truth, 2 of 12 starts reached it
   for 2022 GRAVITY.
5. Nothing flags a failed fit. Parameters sat at bounds (2004 data1, 2008 AGN, 2024 Obj1 and Obj2). A richer
   2006 rung fitted worse than the simpler one it contains. Models were chosen by BIC on error bars known to
   be wrong. Mandate recorded all of it as `converged`.
6. Only 2004 data2 has a scoring stage. The other truths are not in mandate's sealed store.

## What to do next, in order

1. **Compare with the truth images.** For each `truth.npz`: render our chosen model
   (`contest_informed.model_of`, as `contest_informed_quicklook.py` does) on the truth grid, check
   orientation and scale against the notes in `contest_scoring_sources.md` (2024 truths are rotated by
   -73 deg and +23 deg and the spiral's pixel scale is divided by 10), and show truth, model and difference
   side by side. Confirm or correct each row of the table above. Check in particular: 2004 data1 (is it a
   ring with a faint star?), 2006 (PA near 94 deg? size?), 2008 AGB and AGN (what are the components?),
   2010 (spot scale on the surface, so what harmonic degree is needed).
2. **Rewrite `informed_models.yml` per dataset** from `contest_scoring_sources.md` and the truth images, and
   get Ben's sign-off on each entry before any run. New families needed: star + modulated ring + companion
   (2022 GRAVITY), star + arc (2022 AMI), `ArchimedeanSpiral` (2024 Obj1).
3. **Fix the priors**: no cap on flux relative to the star (or fit flux fractions); sizes and positions from
   the task's field where one is given; an over-resolved component where short-baseline V2 is below 1.
4. **Add failure checks to `contest_informed.py`**: any parameter at a bound, any nested rung worse than the
   rung below, fewer than two starts at the best. Report them and make the mandate stage `requires` them.
5. **Carry data conventions**: a visibility scale for 2004 data2; a note level in mandate that every stage
   sees for conventions of this kind.
6. **Seed parametric fits from the blind ensemble image** (a `needs` link from `informed_fit` to `ensemble`).
7. **Scoring stages for every year with a truth**, with the truth images under mandate's `sealed_root`.

## Rules to keep

- Independence and anti-reward-hacking rules in `AGENTS.md` apply. Informed fits take structure from papers,
  never values. Using published values as starts, as some tests above did, is a diagnostic, not a result;
  label it so. Truth images are sealed: they score and diagnose, they do not tune.
- The truths are not public. Do not push them to GitHub.
- Do not skip or weaken tests to get CI green. Agents do not touch the cluster.
- Ben wants direct, concise reporting, and models that match what the papers say the objects are.

## Open items outside this task

- NN Del orbit through mandate fails here with JIT out-of-memory; it has to run on OzSTAR
  (`mandate -C campaigns/orbits run --stage data`, then `run --stage orbit --slurm`).
- Reflection on mandate, awaiting Ben's decision: freeze features until one campaign runs end to end on
  OzSTAR; put the code version in order identity; make quicklook a stage with human sign-off; move invented
  thresholds into `design/` criteria files.
- virgil-validation issue #245: independent check of the spiral's Fourier transform.
