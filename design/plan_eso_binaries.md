# Plan: binaries in the ESO archive with published orbits

## Why

Everything so far checks virgil against mathematics and other people's
codes on simulated data. Real binaries with published orbits add a root of
trust of a different kind: a position predicted by an orbit that other people
fitted to other data, at an epoch where the ESO archive holds data. Virgil
must recover that position from the archive data one epoch at a time, and,
once Stage 6a.1 lands (orbits on jaxoplanet; virgil's
`design/orbit_scene_joint_fitting.md`), recover the orbit from all epochs at
once. This also answers the "real anchor binaries" reminder in §7 of that
note, which asks for a binary with a well-known orbit observed with GRAVITY
and one observed with NACO or SPHERE masking.

## The catalogue

A Claude chat session using ADS, SIMBAD and the ESO archive (TAP, queried
2026-10-04) built a catalogue of 138 published binaries with VLTI or VLT
masking data. It lives outside this repository, in `~/Downloads/eso_binaries/`:

- `eso_vlti_sam_binaries.xlsx`: targets (93 clean, 45 other), 1409
  observation sets (target × instrument × programme × role), 98 papers;
- `orbit_posteriors_summary.json`: medians and 16/84 % of the orbital
  elements, from the [whereistheplanet](https://github.com/semaphoreP/whereistheplanet)
  posteriors, for **21 systems**;
- `few_epoch_binary_positions.json`: published positions for systems with
  one or two epochs.

Its own caveats apply: element values come from abstracts, ORB6 and
whereistheplanet, not from paper tables, and "Public" is often `unknown`.
**Nothing from it is evidence until Stage E0 has checked it.**

## The 21 systems with orbit posteriors

`sci + cal` counts GRAVITY nights with the system as a science target and as
another programme's calibrator. `after mid-2024` counts nights in programmes
starting from June 2024, which are candidates for epochs the posteriors did
not use (to be confirmed per system in E0). Phase 3 is the number of
ESO-reduced GRAVITY OIFITS products.

| System | Orbit | a (mas) | P | e | GRAVITY nights, sci + cal | after mid-2024 | years | Phase 3 | other |
|---|---|---|---|---|---|---|---|---|---|
| Gl 229 Ba–Bb | Thompson et al. 2025 | 7 | 12.1 d | 0.23 | 7 + 0 | 0 | 2023–2025 | — | — |
| HD 72946 B | Balmer et al. 2023 | 252 | 16 yr | 0.50 | 8 + 0 | 0 | 2021–2023 | 17 | MATISSE |
| HD 136164 Ab | Balmer et al. 2024 | 184 | 75 yr | 0.43 | 4 + 0 | 0 | 2022–2023 | 28 | NACO SAM |
| HD 206893 B | Kral et al. 2026 | 260 | 30 yr | 0.05 | 21 + 0 | 1 | 2019–2025 | 234 | SPHERE |
| HD 984 B | GRAVITY, unpublished | 257 | 37 yr | 0.39 | 3 + 0 | 1 | 2021–2025 | 20 | MATISSE |
| HD 1663 | Nowak et al. 2024 | 667 | 644 yr | 0.51 | 1 + 1 | 1 | 2021–2025 | 9 | — |
| λ¹ Scl | Nowak et al. 2024 | 563 | 342 yr | 0.42 | 7 + 14 | 16 | 2021–2026 | 30 | — |
| HD 25535 | Nowak et al. 2024 | 1453 | 359 yr | 0.41 | 10 + 15 | 9 | 2019–2026 | 55 | — |
| HD 32642 | Nowak et al. 2024 | 797 | 735 yr | 0.41 | 3 + 0 | 0 | 2021 | 12 | — |
| HD 36584 | Nowak et al. 2024 | 956 | 459 yr | 0.79 | 3 + 10 | 10 | 2020–2025 | 25 | — |
| HD 46716 | Nowak et al. 2024 | 786 | 405 yr | 0.39 | 2 + 1 | 1 | 2021–2026 | 17 | — |
| HD 46780 | Nowak et al. 2024 | 899 | 115 yr | 0.75 | 1 + 0 | 0 | 2020 | 4 | — |
| HD 73900 | Nowak et al. 2024 | 651 | 64 yr | 0.85 | 5 + 11 | 11 | 2021–2026 | 36 | — |
| HD 90444 | Nowak et al. 2024 | 440 | 137 yr | 0.95 | 2 + 1 | 1 | 2022–2025 | 8 | — |
| HD 91881 | Nowak et al. 2024 | 870 | 156 yr | 0.75 | 6 + 13 | 8 | 2020–2026 | 47 | — |
| HD 123227 | Nowak et al. 2024 | 1433 | 349 yr | 0.12 | 6 + 24 | 12 | 2020–2026 | 45 | — |
| HD 174536 | Nowak et al. 2024 | 1591 | 34 000 yr | 0.50 | 7 + 15 | 15 | 2017–2026 | 21 | — |
| HD 196885 | Nowak et al. 2024 | 613 | 81 yr | 0.41 | 11 + 3 | 3 | 2019–2026 | 40 | — |
| HD 30003 | GRAVITY binary | 4377 | 1100 yr | 0.78 | 17 + 2 | 0 | 2017–2023 | 6 | — |
| HR 5362 | GRAVITY binary | 4270 | 7100 yr | 0.79 | 9 + 3 | 0 | 2018–2024 | 1 | — |
| κ¹ Scl | GRAVITY binary | 1255 | 600 yr | 0.72 | 1 + 0 | 0 | 2019 | 9 | — |

## What the data can and cannot test

Every GRAVITY observation of these systems is in dual-field mode, and that
splits them into two groups.

**One system is a binary inside one fibre.** Gl 229 Ba–Bb is a 7 mas pair of
brown dwarfs with a 12.1-day orbit. Its observations are also dual-field,
presumably with Gl 229 A (about 5″ away) as the fringe-tracking star (E0
checks this), so the science fibre sees both Ba and Bb. The binary is
in the V² and closure phases, which virgil's existing binary model fits.
Seven nights spread over 14 months cover about 35 orbits, so the joint orbit
is well constrained, and its 12-day period makes the
nightly position change large enough to test per-datum times (R3 in the
orbit note). The two components have nearly equal fluxes, which makes the
180° ambiguity (§5.3 of the orbit note) a real issue, not just a
hypothetical one.

**Twenty systems put the companion in its own fibre.** At 180–4400 mas, the
separation is many times the field of view of a GRAVITY fibre (about 60 mas
on the UTs and 250 mas on the ATs). Each fibre sees one star, so its V² and
closure phases are those of a single star and carry no binary signal. The
relative position is only in the *phase-referenced* visibility phase: the
science fibre's phase, referenced to the fringe tracker through the
metrology, is the phase of a point source offset from where the fibre was
pointed. Recovering it needs the metrology zero point, which comes from swap
observations or an on-axis reference (the exoGRAVITY method). Nowak et al.
2024 published these binaries' orbits as calibrators for that astrometry. Virgil does not model
phase-referenced astrometry today, and the virgil-vlti boundary puts that
calibration outside core.

So "per-epoch recovery" means different things for the two groups:

| Group | Per-epoch observable | Per-epoch reference | Joint fit |
|---|---|---|---|
| Single fibre (Gl 229 Ba–Bb) | V², closure phase → virgil binary fit | Published per-epoch positions (Xuan et al. 2024, to be checked in E0) | virgil `KeplerOrbit` on V² and CP directly (6a.1, R3) |
| Dual field (the other 20) | Phase-referenced phase → (Δα, Δδ, covariance) | Published exoGRAVITY positions in the papers and in whereistheplanet's data | virgil `PositionData` → orbit (6a.1); short arcs (R4) |

## Independence and ground truth

- **The posteriors are not independent of the GRAVITY data.** Whereistheplanet
  fits all the astrometry it has, GRAVITY included, so a predicted position at
  a GRAVITY epoch used in the fit agrees partly by construction. Use three
  references, weakest first:
  1. the posterior's prediction at an epoch it used (a consistency check only);
  2. the *published per-epoch position* at that epoch (a check of our
     reduction against the authors');
  3. the posterior's prediction at an epoch it did *not* use: an
     out-of-sample test. The 2024–2026 calibrator nights in the table are
     candidates, once E0 has confirmed which epochs each posterior used and
     that the data are public.
- **The orbit evaluator must be independent.** Reference positions come from
  a NumPy/SciPy Kepler evaluator in `src/crosscheck/orbits.py`, written from
  the textbook relations, not from virgil's orbit code or jaxoplanet. Its
  conventions for whereistheplanet's elements (whose ω, which node Ω refers to,
  τ relative to `tau_ref_epoch` = MJD 58849) are pinned against
  whereistheplanet's own predicted separations and PAs, as was done for
  PMOIRED in `docs/pmoired_conventions.md`. Whereistheplanet runs on orbitize!;
  using its posteriors here as data does not break virgil's no-orbitize rule,
  which is about virgil's own dependencies.
- **Roots of trust:** `mathematics` for the evaluator, and `golden:<source>`
  for published positions and posteriors (for example
  `golden:whereistheplanet`, `golden:Nowak2024`).

## Stages

Each stage is a separate PR with its own tests and evidence records, as in
`plan_external.md`. Heavy fits and Monte Carlo campaigns run on OzSTAR
(tier B or C), not on the laptop.

### Stage E0: verify the catalogue and pin conventions (3–4 h; no virgil changes needed)

1. Copy into `data/eso_binaries/` the parts the tests need: the 21-system
   element summary and an observation-set table for those systems, with a
   provenance note. Leave the 300 kB spreadsheet out. Leave out HD 984 B's
   unpublished posterior unless Ben says it may be in a public repository.
2. Check each of the 21 rows against the ESO archive (TAP, `dbo.raw` and
   `ivoa.ObsCore`): night list, dual-field or single-field mode, telescopes,
   public or not, and Phase 3 products. Check the orbit references against ADS.
3. For each posterior, record the astrometric epochs it used, so that the
   out-of-sample nights are known.
4. `crosscheck/orbits.py`: Kepler solve and Thiele–Innes, with `reference`
   tests (energy/period, symmetries). Then pin the whereistheplanet
   conventions: predict separation and PA on a grid of epochs from posterior
   samples and match `whereistheplanet`'s own predictions to 0.1 mas and
   0.01°. Two deliberately wrong mappings, ω + 180° alone (which reflects
   the companion through the primary) and Ω + 180° alone, are `control`s that
   must fail. Shifting both together is the usual visual-orbit degeneracy,
   with the same sky positions, and is a `check` that they agree.

### Stage E1: Gl 229 Ba–Bb, one epoch at a time (3–4 h; virgil main)

1. Download the reduced or pipeline-reducible GRAVITY data for the seven nights
   (`~/data/eso_binaries/`; to OzSTAR if the raw data are needed).
2. Fit the virgil binary model per night, with Stage 6.0 closure whitening
   and per-night errors. Report separation, PA and flux ratio.
3. Compare with the published per-epoch positions (reference 2) and with
   Thompson et al.'s posterior predictions (reference 1). Record the
   pulls as evidence; a disagreement beyond the stated errors becomes a
   finding (a strict xfail), not a tuned tolerance.
4. Check the 180° choice against the orbit: virgil's PA must agree with the
   orbit's, not the flipped one. This is test 5.3.2's real GRAVITY anchor.

#### Status (2026-10-07)

Steps 1–2 are done for seven nights with a script outside this repository
(`fit_nights.py`): closure phases only, 2.05–2.18 µm, a linear-motion binary
per night, a grid over ±12 mas refined from the five best peaks, and a free
`phi_scale` per night. The reference is Xuan et al. 2024 (Nature 634, 1070),
whose elements we evaluate independently (Ω = 213°, ω_secondary = ω_primary +
180°). No evidence records yet: the runs are not CI-produced.

- **Four nights have a best peak within 0.1–0.9 mas of the prediction; this is
  not yet a test in σ.** Positions (dRA, dDec, mas), with the offset from the
  prediction (our NumPy evaluation of Xuan et al.'s PMOIRED elements in
  `crosscheck/orbits.py`):

  | Night | Xuan et al. prediction | Per-night best peak | Offset (mas, PA) |
  |---|---|---|---|
  | 2023-12-25 | (+0.2, −5.6) | (+0.1, −5.6) | 0.10, 1.0° |
  | 2023-12-29 | (−2.7, +6.5) | (−2.4, +5.6) | 0.93, −1.0° |
  | 2024-04-29 | (−1.7, +7.3) | (−1.6, +7.1) | 0.15, 0.2° |
  | 2025-02-11 | (−4.5, −3.3) | (−4.7, −2.9) | 0.38, 3.8° |

  A tolerance of about 1 mas is far looser than GRAVITY's per-epoch precision
  (tens of µas) at separations of 5–8 mas: it would pass a plate-scale or
  wavelength error of about 10% or a PA error of about 8°. The comparison must
  be made in σ before it is evidence (E5): the offset divided by the quadrature
  sum of the per-night fit error and Xuan et al.'s per-epoch (or
  orbit-propagated) uncertainty. Neither is in hand yet. The night fits store
  no covariance, and `xuan2024_table1.json` has only the orbital elements and
  their errors. Until both are added, these four rows are consistent with the
  prediction, with no pull quoted.

- **Orbit comparisons without RVs use the projected degrees of freedom.** The
  GRAVITY-only fits have no RVs, so $(\omega,\Omega)\to(\omega+180^\circ,
  \Omega+180^\circ)$ leaves the sky positions unchanged and raw $\omega$ and
  $\Omega$ pulls are not scored (the first fits gave $-90\sigma$ and
  $-28\sigma$ for a mirror mode that fits equally well). Compare the sky track
  (statistic A) and the projected elements (statistic B), and show folded
  corner plots, as defined in `design/orbit_comparison.md`. Gl 229 is already
  fitted, so this is a post hoc comparison, labelled as not preregistered.
  Systems not yet fitted need a pre-fit amendment to their criteria file
  (`design/trackb_criteria.md` is hashed and is not edited here).
- **Two nights are ambiguous.** On 2024-02-27 and 2024-03-28 the top two peaks
  differ by Δloss 0.6 and 0.5, and no top-five peak lies near the prediction
  (the nearest are 7.9 and 13.3 mas away, the best peaks). Xuan et al. flag
  both as bad. In the orbit comparison these are flagged single-night alias
  peaks (`design/orbit_comparison.md`, section 5): not used as constraints, but
  still given a track comparison (statistic A).
- **2024-12-18 is a disagreement to be ruled on, not an ambiguous night.** It
  is not one of Xuan et al.'s epochs, so their orbit predicts it out of
  sample, and they do not flag it. Our best peak, (−3.6, +3.3), is 10.0 mas
  from the prediction (+5.6, +7.1), at PA −86° from it, with flux 0.95 (at the
  bound). The second and third peaks (Δloss 0.8) are 11.5 mas away, and the
  only peak near the prediction is the fifth, (+5.9, +7.3), 0.37 mas from it
  with Δloss 15.1 behind the best. It is likewise a flagged alias peak in the
  orbit comparison, 8–32 nats worse at the orbit position than at its own
  best peak. This corrects an earlier statement that no
  top-five peak lies near the prediction. Resolve it with the shared-flux
  scorer or with the same-data check of virgil-validation#69 (see Open)
  before any of the seven nights becomes evidence.
- **Flux is free on every night**, so a wrong peak can look decisive (2024-02-27
  at flux 0.21, 2024-12-18 at 0.95). These are per-night positions with free
  flux, not orbit constraints. The shared-flux scorer of the orbit session
  (virgil PR D1) supersedes them.
- **Noise excess.** The virgil-vlti pair test on the science frames (Gl 229 has
  no calibrator frames) gives a closure-phase white-noise excess of 1.85 on
  2023-12-25 and 1.90 on 2025-02-11, matching the fitted `phi_scale` of
  1.8–2.6: real noise beyond the pipeline errors. The phase offsets (11.9° and
  17.8°) are upper bounds, since no target model was subtracted. V² is
  uncalibrated, so its baseline-gain term is expected.
- **Open.** 2024-12-18, as above: the loss at the predicted position on the
  grid (the fifth peak's value is the nearest we hold), and a ruling. Xuan et al.'s own 22 GRAVITY files (OiDB, virgil-validation#69) give
  a same-data check of our reduction. The 180° check (step 4) waits for the
  orbit fit.

Stage E3 now lives in the orbit-fitting session (virgil#268, virgil#279). Its
first results: the free search converges on i ≈ 110° against Xuan et al.'s
31.4°, with raw closure-phase χ²/N of 3–25 on the training nights and 606 and
355 on the held-out nights. Seeding at Xuan et al.'s orbit recovers their
elements (P 12.13 d, e 0.23, i 29–30°, a 7.3 mas) and the held-out 2025-02-11
to 0.52 mas, with raw χ²/N 10.1. On the same five training nights, with the
same free per-night `phi_scale`, the Xuan-seeded refit reaches loss −5961.1
and the free search −5695.2: the likelihood prefers Xuan et al.'s orbit by
Δloss 265.9, so the free search failed to find the better optimum (`fit_orbit`
outputs at `~/data/eso_binaries/gravity/fits/9820bc3/fit_orbit/`). That
supports a search failure, not a likelihood or convention problem, for this
data set. The nights' grid fits (step 2) are a separate matter, see the
provisional note above the E4 table.

### Stage E2: dual-field positions (decision needed; see Questions)

- **Route 1, positions as data (3 h).** Take the published per-epoch
  exoGRAVITY positions and covariances (from the papers' tables, or
  whereistheplanet's data files if they carry the GRAVITY points) as
  `PositionData`. This needs nothing new in virgil beyond 6a.1. It tests
  the orbit fitting, not virgil's visibility modelling.
- **Route 2, positions from the archive (about 15–25 h, mostly in
  virgil-vlti).** Reduce the dual-field files to phase-referenced
  visibilities, calibrate the metrology zero point (swap or on-axis), and fit
  an offset point source per epoch. Compare with the published positions
  (reference 2), and then with the out-of-sample predictions (reference 3).
  This is the dual-field astrometry that Nowak et al.'s calibrators exist for.
  The instrument-specific part belongs in virgil-vlti, and a generic
  phase-referenced likelihood in core only if it proves general.

**Decided (Ben, 2026-10-04): Route 2 is out.** Phase referencing against
the other fibre is not worth the effort just to validate virgil. Route 1
stays as an optional orbit-fitting test.

### Stage E3: joint orbits with Stage 6a.1 (after virgil 6a.1; 6–8 h)

1. **Gl 229 Ba–Bb, from visibilities.** One `KeplerOrbit` fitted to all seven
   nights' V² and closure phases at once, with per-datum times. Compare
   with (a) the two-step route (E1 positions → orbit) and (b) Thompson et
   al.'s posterior: P, e, i, ω, Ω and a within their quoted 16/84 % ranges,
   the bias measured in posterior σ.
2. **Wide calibrators, from positions (E2 Route 1).** Their GRAVITY arcs are
   a few years of orbits lasting 64–34 000 yr, which is the short-arc case of R4.
   Fit `StateVectorOrbit` (position and rate at `t_ref`, with μ, dz, vz under
   priors). Compare its posterior with the state vector the whereistheplanet
   posterior implies at the same `t_ref`. That posterior uses decades of
   speckle and visual astrometry, so the comparison is genuinely external.
   Then refit with the posterior's elements as priors and predict the
   out-of-sample nights (reference 3).
3. **The substellar companions** (HD 72946 B, HD 136164 Ab, HD 206893 B and
   HD 984 B, 16–75 yr) are the middle case: `KeplerOrbit` and
   `ThieleInnesOrbit` both apply. HD 206893 B (21 nights, e ≈ 0.05) tests the
   near-circular regularisation; HD 90444 (e = 0.95) and HD 73900 (e = 0.85),
   if Route 1 data allow, test high eccentricity.
4. **Coverage (tier C, OzSTAR).** For Gl 229 Ba–Bb and two wide calibrators:
   draw orbits from the published posterior, simulate data on the real
   time and uv sampling (`simulate`, R8), and refit. The credible
   intervals must cover the truth at their nominal rates. Without RVs, coverage
   is assessed on the projected quantities (statistic A positions, statistic B
   elements; `design/orbit_comparison.md`), not on raw $\omega$ and $\Omega$. This is a Monte
   Carlo campaign in the sense of `design.md`.

### Stage E4: masking anchors with NACO SAM (AMICAL; about 3 h, then 2 h per target)

NACO SAM data on binaries with published orbits. These are from ESO `dbo.raw`
(template `NACO_sam_obs_GenericOffset`), matched **by target name**, and were
checked against the frame headers on 2026-10-04. Each target has one epoch:

| System | Epoch | Mask, filter | Calibrators | Orbit | Separation then | Why |
|---|---|---|---|---|---|---|
| HD 136164 Ab (HIP 75056) | 2012-08-01 | L′ | HIP 74479, HIP 74865 | Balmer et al. 2024 (posterior) | ~100–260 mas | well resolved; the posterior is in the summary JSON |
| 9 Sgr | 2011-03-10 | 7 holes, Ks | HD 152249 | Fabry et al. 2021 (P = 9.1 yr) | ~15 mas | below λ/2B: tests the separation–contrast degeneracy, which an orbit prior breaks |
| δ Vel Aa–Ab | 2009-01-07 | 18 holes, NB 1.64 and IB 2.12 | δ Phe, Achernar | Mérand et al. 2011; Kervella et al. 2013 (P = 45 d) | ~16 mas | a precise orbit; very bright (K = 1.7), so check for saturation |

**The catalogue's NACO rows were matched by coordinates and are unreliable.**
NACO header and `dbo.raw` coordinates scatter by up to degrees between cubes of
one star, so a coordinate box picks up other stars. The earlier version of this
table therefore listed 9 Sgr nights in 2012 and 2013 and a 2011 epoch for
HD 136164, all of which were other stars. HD 150136 (HIP 81702) has no SAM data
under any name. GG Tau's programme observed no calibrator star, so it is
dropped. The download and survey jobs (OzSTAR `eso_binaries_dl`,
`eso_binaries_naco`) now select by name.

Start with HD 136164 Ab, the cleanest test. Reduce with AMICAL, fit with virgil,
and compare separation and PA with each orbit's prediction at the epoch. Compute
the predictions with `crosscheck/orbits.py`, from elements copied from the papers'
tables. This covers the masking anchor of the orbit note's §5.3.2. With one epoch
per target, these are per-epoch tests only. Each check is of position (and of
flux ratio where it is published), not of a joint orbit.

#### Status (2026-10-07)

All three targets are reduced with AMICAL 1.6.0 (OzSTAR job `eso_binaries_naco`)
and fitted with virgil 032cf70 (job 18193592: `likelihood_grid`, then `fit` from
five grid peaks; `dra`, `ddec` uniform, flux LogUniform(1e-4, 1), `vis_scale` and
`phi_scale` LogUniform(0.1, 10)). Outputs are in
`~/data/eso_binaries/naco/fits/032cf70/fit_naco/`. Raw χ²/N is about s² of the
fitted scales; the stored `chi2_red` ≈ 1 is after rescaling.

**All "virgil result" and "Verdict" entries below are provisional (pre-fix,
to rerun).** The fits predate the `likelihood_grid` argument-order change. If
it swapped axes or arguments, the grid peaks that seed `fit` may be wrong, and
the 9 Sgr PA, the δ Vel PA and the HD 136164 non-detection could be artefacts
of the call. No verdict is settled before step 4 below is rerun.

| Target | Prediction | virgil result (provisional) | Verdict (provisional) |
|---|---|---|---|
| 9 Sgr (Ks, 7 holes, 2011-03-10) | 11.89 ± 0.3 mas, PA 72.3 ± 1.5° (Fabry et al. 2021, a = 14.656 mas fitted to their Table A.2) | 16.95 mas, PA 31.1°, flux 0.95 (at the bound), `vis_scale` 3.0 (raw V² χ²/N ≈ 9); pinned at the prediction, 2Δloss = 78 worse | Unresolved. The fit has the flux at its bound of 1, where an equal-flux binary's closure phases vanish and the PA is unconstrained, so the flux bound is a candidate cause of the degeneracy. The 41° offset (31.1° against 72.3°) is not explained by 180° ambiguity. At 11.89 mas, about 0.4 λ/2B, closure phases still constrain the PA at modest contrast, so a resolution limit is not shown either. A disagreement, a degeneracy and a call artefact are all still possible. |
| HD 136164 Ab (L′, 2012-08-01) | 166.1 ± 3.4 mas, PA 99.7 ± 1.4° (Balmer et al. 2024, via whereistheplanet) | not detected: best peak 279 mas, flux 0.0037 (noise); at the prediction the flux goes to the 1e-4 bound | A detection-limit case (expected flux ratio about 1e-3, ΔL′ ≈ 7). virgil limits at that position are still to do. |
| δ Vel Aa–Ab (18 holes, NB 1.64 and IB 2.12, 2009-01-07) | 14.26 ± 0.14 mas, PA 155.3 ± 0.6° (Mérand et al. 2011, T0 refitted to their AMBER vectors) | 15.23 mas, PA 344.3° (164.3° modulo 180°), flux 0.45, `vis_scale` 1.21, `phi_scale` 1.81 (raw χ²/N ≈ 1.5 in V², 3.3 in CP) | Off by 1.0 mas and 9°. Near periastron the PA moves about 10° per day, so the comparison depends on the per-file MJDs and T0. δ Vel B (~0.6″) is in the stamp, and the narrow-band data are photon-starved. |

The star finder had to be fixed for HD 136164: destripe, mask the border and
refine in two passes (the old version locked onto edge column x = 511).

**Calibrator weighting.** AMICAL 1.6.0 averages calibrators with error weights,
which biases the V² normalization (fixed in SAIL-Labs/AMICAL#332, unreleased).
Treat V² scales with care until the data are re-reduced with the fork.

**Next steps.**

1. δ Vel: per-file MJDs from the OIFITS, the prediction propagated to each, and
   the T0 uncertainty carried through, before calling the 9° a disagreement.
2. HD 136164 Ab: virgil contrast limits at 166 mas, PA 100° and over the whole
   field, against the expected ΔL′ ≈ 7.
3. 9 Sgr: keep it unresolved. Show the degeneracy directly with a
   contrast-separation profile, or refit with a flux prior bounded away from 1
   that does not come from the reference (fixing the flux at Fabry et al.'s
   ratio would feed the reference into the fit, so the PA agreement would be
   partly self-fulfilling). Rerun after step 4.
4. Rerun the fits with the corrected `likelihood_grid` call (all three
   targets), then re-read every verdict above. Optionally re-reduce with the
   AMICAL fork first.
5. Evidence records and the independent NumPy orbit evaluator (the 9 Sgr
   prediction logic) in `crosscheck/`, once the numbers are final.

### Stage E5: reporting (1–2 h)

Evidence table rows per system and stage; a results page with per-epoch pulls
and joint-fit comparisons (sky track and projected elements, folded corner
plots, `design/orbit_comparison.md`; no raw $\omega$ and $\Omega$ pulls without
RVs); findings in the README.

## Beyond the 21: single-fibre binaries with published orbits

Only one of the 21 tests virgil's core binary model on visibilities. The
catalogue has many more **single-fibre** binaries with published
interferometric orbits and Phase 3 OIFITS. These are better at exercising
virgil's visibility modelling, both per epoch and jointly:

- the Araucaria GRAVITY SB2s and eclipsing binaries (Gallenne et al.):
  HD 9312, HD 41255, HD 70937, HD 210763, HD 224974, ο Leo, LL Aqr, AK For,
  HD 188088 and V963 Cen, with 11–29 nights each, all with Phase 3 products.
  Radial velocities make Ω absolute (§5.1.2 of the orbit note);
- Araucaria PIONIER eclipsing binaries: AL Dor (30 nights), V4090 Sgr and
  ψ Cen.

Their elements are in paper tables, not in the summary JSON, so E0's
reading of the tables would have to extend to them. Proposed as Track B after
E1, starting with two or three of the Araucaria GRAVITY SB2s.

**Status (2026-10-07):** proposed, not started, awaiting Ben's decision.
Gallenne et al. 2023 (A&A 672, A119) fit per-epoch positions of the ten GRAVITY
SB2s with CANDID (diameters fixed, bootstrap errors); their Table B.1 gives the
positions with error ellipses, and they report a +0.02% wavelength correction
and 8–50 µas systematics. Phase 3 visibility OIFITS exist for all of them and
are small (a few MB per epoch, likely under 1 GB in all). The comparison would
be virgil per-epoch fits against Table B.1, then joint orbits against their
elements, with their radial velocities as `RVData`.

## Questions for Ben

1. **Dual field:** Route 1 (published positions as data) only, or also
   Route 2 (phase-referenced astrometry from the archive, in virgil-vlti)?
2. **Track B:** add the Araucaria single-fibre binaries? Recommended, since
   they test what the 20 dual-field systems cannot.
3. **HD 984 B:** its posterior is labelled unpublished GRAVITY work. Keep it
   out of this public repository?
4. **Data:** `~/data/eso_binaries/` for reduced products, and OzSTAR for any
   raw data and pipeline runs?

## Order of work

E0 → E1 now, on virgil main (neither needs 6a.1). E2 Route 1 and E4 can run
in parallel. Data for E1 and E4 are downloaded to OzSTAR
(`/fred/oz440/bpope/eso_binaries/data`). E3 waits for virgil's Stage 6a.1. In total about 20–25 agent
hours without Route 2 or Track B.
