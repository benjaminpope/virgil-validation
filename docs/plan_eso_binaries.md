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
   0.01°. A deliberately wrong mapping (ω + 180°, Ω + 180°) is a `control`.

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

Route 1 first. Route 2 only if dual-field astrometry is wanted as a virgil
capability anyway (exoplanets and brown dwarfs), and not merely to validate
virgil.

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
   intervals must cover the truth at their nominal rates. This is a Monte
   Carlo campaign in the sense of `design.md`.

### Stage E4: a masking anchor (3–4 h; AMICAL)

HD 136164 Ab was observed with NACO SAM in L′ (2011 and 2012, Ireland's
programmes). Its separation is about 100–260 mas, which is inside the range
where masking is sensitive. Reduce with AMICAL, fit with virgil, and compare
the PA and separation with Balmer et al. 2024's predictions at those epochs.
This is the masking anchor of the orbit note's §5.3.2. First check that the
companion's L′ contrast is detectable with SAM.

### Stage E5: reporting (1–2 h)

Evidence table rows per system and stage; a results page with per-epoch pulls
and joint-fit element comparisons; findings in the README.

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
in parallel. E3 waits for virgil's Stage 6a.1. In total about 20–25 agent
hours without Route 2 or Track B.
