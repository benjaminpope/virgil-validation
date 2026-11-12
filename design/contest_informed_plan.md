# Informed contest fits: fields, priors, starts and checks

Written 2026-11-12 after comparing the second informed run with the papers' truth figures
(`design/handover_contest_truth_diagnosis.md` has the earlier diagnosis). Model descriptions signed off by Ben the
same day are in `campaigns/contests/informed_models.draft.yml`.

This note holds no published value and must not gain one: every scale, start and bound below comes from the data
or from a file entrants were given. Published values are compared with the fit afterwards, in a sealed stage.

Rulings by Ben, 2026-11-12: structure seen only in a truth figure is not used for now; the 2010 surface is tried
as an image on a disk first.

## 1. What failed, in one paragraph

Seven of eleven objects had a family that could not draw the object. Beyond that, one length, lambda / B_min, was
used as the size limit, the search radius and the image field; fluxes were capped relative to the star; starts were
four random draws; chi2 on quoted errors chose the model; and nothing flagged a fit that sat on its bounds.
A fit could pass with chi2/N near 1 and the wrong object (2024 Obj1, 2008 AGB).

## 2. Three scales, not one

`scales()` returns one field. Replace it with three, each recorded in the report with its source.

| Scale | Meaning | Rule |
|---|---|---|
| extent `E` | how far the resolved light reaches; upper limit of size priors, field of every image and render | the contest's field where it gave one (`TASKS[...]["field"]`: 2006 from the clue image, 2008 from the taper); otherwise 4 x the major FWHM of the visibility-moment fit of section 4.1, at least 8 resolution elements |
| reach `S` | how far a compact companion is searched for | 3 x lambda / B_min (the blind track's largest field growth), or half the given field when the contest tapered the field (2008) |
| pixel | for any image (image on top, image on a disk, GP disk) | lambda / 4 B_max; field = `E`; if that needs more than 128 pixels a side, warn in the report, do not coarsen silently |

When the shortest baselines already see almost nothing (2006: V2 below 0.03 everywhere) the data bound `E` only from
below. Then a given field decides; without one, use `S` and set a check `extent_unconstrained`.

Sizes: `LogUniform(resolution / 5, E)`. Positions of parts of the object ("near"): within `E / 2`. Inclination:
0 to 89 degrees (edge-on is allowed).

virgil already documents helpers for part of this: `imaging.second_moment_fwhm`, `imaging.envelope_fwhm`,
`imaging.field_of_view`, `imaging.image_field`, `imaging.nyquist_pixel_scale`. Use them where they do the job.

## 3. Fluxes, colours and errors

- **Fluxes.** No cap relative to the star. Every component after the reference has a ratio in
  `LogUniform(1e-4, 1e4)`; the reference is the first component of the family's simplest rung. A new component can
  then start at negligible flux, so a richer rung always contains the rung below (section 5, `nested`). In the last
  run a star forced to at least 1/11 of the flux made "star + disk" fit worse than "disk" for 2006.
- **Over-resolved light.** Every family gets an optional `Resolved` component (virgil: visibility zero on every
  baseline), in a rung of its own.
- **Planets and faint companions.** Where the paper says planet or faint companion: flux at most 0.1 of the star,
  and at least two resolution elements from it. A stellar companion (2022) has the wide prior.
- **Colours.** Fit grey first; then free the spectral indices from zero, prior `Uniform(-4, 4)`. Indices at +-8
  were bound hits.
- **Errors.** Several contests quote errors far below any model residual (2010: 1e-4 on every V2; 2022 AMI:
  0.0012 degrees). Fit with virgil's noise terms (`vis_error_rel`, `phi_error`, or `vis_scale` / `phi_scale`; see
  virgil's docs/common_arguments.md) so the likelihood carries its normalisation, and compare rungs by BIC on that
  likelihood. Report chi2 on the quoted errors beside it, never choose by it.
- **Tempering.** Start every rung with errors inflated by `sqrt(chi2/N)` of the visibility-moment fit
  (`OIData.with_error_scale`), then refit at the fitted noise. A smooth first landscape is what the blind track
  found it needed.
- **Visibility scale.** 2004 data2 was normalised without the spot's flux, so V2 at zero baseline exceeds 1
  (`scripts/score_2004.py`). virgil has no visibility-scale parameter; use one shared gain (`OIData.with_gains`) or
  a fitted constant, for this dataset only, named in the report.

## 4. Starts

Each start records where it came from (`moments`, `image`, `clean`, `map`, `grid`, `below`, `random`), and the
report counts starts at the best chi2 by origin.

### 4.1 Visibility moments (new; every dataset; seconds)

Fit V2 with a point, an elliptical Gaussian and over-resolved flux, from a small grid of sizes and angles, robust
loss. It gives the compact fraction, the extended light's major FWHM, axis ratio and position angle, and the
over-resolved fraction. Run on the data alone on 2026-11-12 (numpy, the OIFITS files only):

| dataset | compact | over-resolved | FWHM (mas) | ratio | PA (deg) |
|---|---|---|---|---|---|
| 2004_data1 | 0.04 | 0.00 | 4.9 | 0.79 | 148 |
| 2006_disk | 0.03 | 0.62 | 14.4 | 0.39 | 93 |
| 2008_agb J, H, K | 0.05-0.08 | 0.00-0.11 | 2.4-3.1 | 0.80-0.88 | (round) |
| 2008_agn J, H, K | 0.06-0.12 | 0.05 | 2.2-2.8 | 0.74-0.82 | 4-49 |
| 2010_lowH, K | 0.03 | 0.00 | 11.6 | 0.96-0.99 | (round) |
| 2018_disk | 0.59 | 0.03 | 3.3 | 0.79 | 52 |
| 2022_gravity | 0.23 | 0.00 | 18.8 | 0.65 | 41 |
| 2022_ami | 0.82 | 0.00 | 288 | 0.54 | 124 |
| 2024_obj1 PIONIER, GRAVITY | 0.21-0.23 | 0.08-0.28 | 16-29 | 0.56-0.90 | 54-58 |
| 2024_obj2 PIONIER, GRAVITY | 0.35-0.48 | 0.05-0.15 | 1.2-13 | 0.40-0.89 | 22 (GRAVITY) |

Use: `E` (section 2); the star's or core's flux fraction; a disk's or ring's position angle and inclination
(`arccos(ratio)`); the size scale of a ring, halo, envelope or arc. The compact fraction is also the median
visibility amplitude on the longest tenth of the baselines; use the two as a cross-check.

Public virgil has no moment estimate with an angle. The branch `contest-round3-doctor` (worktree
`~/code/virgil-validation-doctor`) already has `crosscheck.starts`, `moments_of`, `jittered_starts`,
`multistart_fit`, `fit_primary` (moments, then a diameter scan, then jittered starts) and `companion_search` in
its `scripts/contest_images.py`. Port those; do not write them again.

### 4.2 Stratified geometry starts (replaces most random draws)

The landscapes are multimodal in a few geometric parameters. Scan them, do not draw them:

- size (ring diameter, stellar diameter, arc radius, spiral pitch): 0.5 to 2 x the moment scale, 8 steps;
- inclination x position angle: the moment values, then a 4 x 6 grid;
- thin, long parts (a jet, an arc): position angle in 12 steps; arc length in three.

Other parameters start at their moment values or the middle of their priors.

### 4.3 Likelihood maps for compact parts (existing; fix the grid)

`seeds_from_map` and `map_seeds` cap the grid at 121 points a side, so beyond about 60 resolution elements the
step exceeds the resolution and peaks are missed. Tile the map (`likelihood_grid(..., batch_size=)`) at half a
resolution element over the whole reach `S`. Run it beside the fit below, on tempered errors: a companion carrying
well under a hundredth of the flux is invisible until the surface is modelled.

### 4.4 Seeds from the blind image and from CLEAN

The blind ensembles exist: `~/data/imaging_contests/results/ensemble_8562a39_3018a9f/<label>_ensemble.npz`
(`mean`, `sigma`, `fov`). They come from the data alone. In mandate this is a `needs` link from `informed_fit` to
`ensemble`. From the mean image, anchored on its brightest compact peak:

- second moments: size, axis ratio, position angle (a check on 4.1);
- the radial profile in the deprojected frame: ring radii and widths; its first azimuthal harmonic: a ring's
  modulation;
- compact peaks above five times `sigma`: companions and spots, with fluxes;
- the principal axis of what is left after the compact parts: a jet's or an arc's direction.

Where no ensemble exists, one `virgil.imaging.clean` run (gradient CLEAN; it fits V2 and closure phases directly;
`CleanResult.components`) gives the same seeds: cluster its components for the compact parts, take moments of
`restored(beam)` for the rest. `starting_image(start="moments")` is virgil's own start image.

An image seed is a start, never a prior or a bound.

### 4.5 The rung below, and random draws

Each rung starts once from the rung below with the new component at negligible flux, and keeps 8 random draws as a
control. If only seeded starts reach the best, the report says so.

### 4.6 Starts by dataset

| dataset | family | starts |
|---|---|---|
| 2004_data1 | star, modulated ring, halo | star fraction from 4.1; ring diameter scan; inclination x angle grid; modulation from zero |
| 2004_data2 | spotted star, companion | visibility scale; companion and spot from maps |
| 2006_disk | optically thin disk, over-resolved flux | `E` = the clue's field; angle and ratio from 4.1; inclination to 89; rim radius scan; image seed 4.4 |
| 2008_agb | star, envelope, companion, image on top | sizes from 4.1 (a few mas: not the 30 mas taper); companion map within `S` after star + envelope |
| 2008_agn | disk, jet, counter-jet | disk from 4.1; jet angle in 12 steps, axis ratio `LogUniform(0.02, 0.3)`, length to the taper; image seed for the jet's axis |
| 2010_low | limb-darkened disk, image on the disk, companion | diameter scan; image on `circular_support` of the fitted disk, started flat and from CLEAN; companion map last, out to `S` |
| 2018_disk | as now | 4.1 and maps; not yet run |
| 2022_gravity | star, modulated ring, companion | compact fraction, angle, inclination, size from 4.1; companion map beside star + ring |
| 2022_ami | star, arc | star fraction from 4.1; arc radius scan, angle in 12 steps, three lengths; width from a third of the resolution to the radius |
| 2024_obj1 | star, spiral | waits for virgil PR 706; pitch scan, phase grid, both handednesses |
| 2024_obj2 | star, rings, planets | angle and inclination from 4.1; ring radii from the peaks of a radial-GP disk fitted first (`gp_disk` exists: run it before the ladder); planets by map outside two resolution elements |

## 5. Checks (a failed fit must say so)

`result.json` gets these, and the mandate stage `requires` them. The thresholds are proposals; they move to a
criteria file under `design/` with a change log once Ben agrees them.

| Check | Fails when |
|---|---|
| `no_bounds` | any fitted parameter lies within 1% of a prior bound (log scale for log priors) |
| `nested` | a rung fits worse than the rung below it |
| `starts_agree` | fewer than two starts within 1% of the best chi2 |
| `family_complete` | the image fitted on top carries more than 0.2 of the model's flux |
| `no_collapse` | a component has shrunk to something else: an arc shorter than the resolution, a ring narrower than a fifth of it, a companion within two resolution elements of the star |
| `extent_known` | `E` came from neither a given field nor a constrained moment fit |

Also fix `scripts/contest_informed_quicklook.py` line 98: the extra panels use `origin="lower"`, but virgil's
`render` puts north in row 0, so they are shown flipped north-south. Use `virgil.plotting` or `origin="upper"`.

## 6. Order of work

1. **Infrastructure** in `scripts/contest_informed.py`: sections 2, 3, 4.1-4.3, 4.5, 5, and the quicklook fix.
   Proved on the families that already exist (2004 data1, 2004 data2, 2024 Obj2) by smoke runs.
2. **New families** (2006, 2008 AGB, 2008 AGN, 2010, 2022 GRAVITY, 2022 AMI): the signed entries replace
   `informed_models.yml`; notes regenerated; `mandate.yaml` options; tests.
3. **Image and CLEAN seeds** (4.4) with the `needs` link, and a sealed stage that compares each fit with the
   published values in `design/contest_scoring_sources.md`.
4. **Spiral** (2024 Obj1) once virgil PR 706 is merged.

Full runs go through mandate on OzSTAR (32 starts a rung on a GPU); Ben submits. On the laptop: smoke runs only,
one at a time.
