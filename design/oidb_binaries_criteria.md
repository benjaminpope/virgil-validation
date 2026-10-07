# OiDB binaries: per-epoch positions, registered criteria

Registered on 2026-10-08, before any fit was run. `scripts/oidb_binaries_fit.py`
(not yet written) implements this file. Each result records this file's commit
and SHA-256 (`criteria`, as in Track B). A change to anything here after the
first run is a new version of the campaign: the new version gets a new file,
and the old results stay as they were. The plan that motivates these criteria
is `design/plan_oidb_binaries.md`; it may be edited freely, this file may not.

This file reuses the Track B criteria (`design/trackb_criteria.md`) wherever
they apply. Track B's text is the reference for anything not restated here:
the ellipse-to-covariance conversion, the 180 degree rule, the d² metric
(the larger of the scaled and unscaled covariance versions), the per-epoch,
per-system and overall tests, and the finding-or-definition procedure.

## What is tested

For each system in the census (`design/plan_oidb_binaries.md`, section a), and
each epoch whose position the paper tabulates, virgil fits the authors' own
calibrated OIFITS of that night. Its position is compared with the published
one. The authors fitted the same files, so, as in Track B, this tests
virgil's observables-to-position chain against theirs on the same photons. It
is not a test against independent truth. A published orbit used these same
epochs, so it adds nothing independent here.

Per-epoch positions come from `likelihood_grid`, `fit` and `laplace_cov`, as in
Track B. They do not come from the orbit session's epoch-position scorer, so
that the two campaigns stay independent.

## Before any fit: the reference records

For each system, Stage OB0 reads the paper's tables and commits a reference
record (`oidb/references/<collection>.json`) holding: the table and column of
each position, its uncertainty convention and its conversion to a 1σ
covariance, the observables and wavelength window the authors used, the model
(components, fixed diameters, flux-ratio definition), the epoch MJDs, and the
class of each epoch (below). The set of reference records is hashed
(SHA-256 of their sorted concatenation) and the hash is entered in the plan
before the first fit. A fit that records a different reference hash is
invalid. Nothing in a reference record may be chosen after seeing a virgil
result.

## Epoch classes

Every published epoch is given exactly one class in OB0.

| class | meaning | counted in the tests |
|---|---|---|
| `scored` | published σ (ellipse, or σ per axis), two or more resolved point-like components, model as the paper's, no flag below | yes |
| `no_sigma` | position published without an uncertainty | no: reported with Δ in units of virgil's own σ, and as the rms of Δ over the system's epochs |
| `smeared` | separation s with s·B_max/(λ·R) > 0.1 (R the file's resolving power), the point where bandwidth smearing changes the flux ratio by more than 10 per cent | yes if virgil's model includes smearing at the file's R, as CANDID's does; otherwise reported only |
| `third_body` | the paper's model has three or more components, or the system is a known triple | yes if the paper tabulates the pair's position and virgil fits the paper's model; otherwise reported only |
| `resolved` | a component has a published diameter above 0.5 λ/(2B_max) | yes, with the diameter fixed as in the paper (as Track B); a second fit with it free is a sensitivity check, not counted |
| `not_clean` | disk, wind, circumbinary dust, or magnetosphere in the paper's model | no: reported against the paper's own binary + disk solution, as `plan_oidb.md` says |
| `limit` | a non-detection or upper limit | no: belongs to the detection campaign |

A published position with no uncertainty is never given an assumed one, because
a σ chosen here would decide the verdict.

## Rules for the cases where Track B's criteria do not fit

1. **No published σ.** Class `no_sigma` above. If a system has fewer than 3
   `scored` epochs it gets no system test.
2. **Different calibration or observables.** The files are the authors', so
   the calibration is the same. The observables may not be: a paper that used
   closure phases only is fitted with closure phases only, the same wavelength
   window, and the same emission-line exclusions. If the paper does not say,
   both closure-phase-only and all-observable fits are run and the counted one
   is the one with the larger d², fixed here to avoid a choice after the fact.
3. **Bandwidth smearing.** Class `smeared`. The smearing term uses the
   resolving power read from each file's `EFF_BAND` and `EFF_WAVE`, not a
   nominal one.
4. **Triples and hierarchical systems.** Class `third_body`. The third
   component is fitted only if the paper fits it.
5. **Resolved components.** Class `resolved`.
6. **Flux-ratio definition.** The paper's definition (secondary/primary,
   or secondary/total; band) is converted to virgil's before comparing. The
   flux ratio is compared in the same way as position, as a one-dimensional
   z² = Δ²/(σ_v² + σ_p²), against the χ²₁ quantile 10.828 (p = 0.001), and is
   never part of the overall test.
7. **Epoch assignment.** A published epoch is matched to the granules whose
   `t_min` lies within 0.5 d of the published MJD (or the night, when only a
   date is given). A file that spans several nights is split by MJD window. If
   no granule matches, the epoch is `not fitted` and listed.
8. **Unreadable files.** Treated as in Track B: left out of their epoch, and
   the file and reason listed. If no file of an epoch can be read the epoch is
   `not fitted`.
9. **Detections in surveys.** A survey target counts only if the paper reports
   a companion with (ρ, PA) and an uncertainty at an epoch. Targets with no
   reported companion are `limit`. A survey detection is also reported as
   recovered or not: recovered means a grid peak within 3σ of the published
   position with Δχ² above the survey's own threshold, stated in its
   reference record. This is reported, not a pass criterion.

## Search, priors, model

As in Track B: Jeffreys priors (uniform positions; log-uniform flux and error
scales), a grid step of λ_min/(4B_max) over (−H, H) with
H = max(3·sep_published, 30) mas, five best peaks refitted, and the flux
maximized at each cell. The grid is capped (see plan, section e): at most 24
spectral channels and at most 1.5 × 10⁶ position cells per epoch. If the cap
would coarsen the step beyond λ_min/(2B_max), the grid is run on a window of
±2·sep_published centred on the published position and the epoch is marked
`windowed`. Windowed epochs are counted, but the window is a prior on position
and is reported.

## Pass criteria

The same tests as Track B, on the epochs of class `scored`, `smeared` (when
modelled), `third_body` (when modelled) and `resolved`:

1. **Per epoch.** Flag d² > 13.816.
2. **Per system.** S = Σ d² against χ² on 2N degrees of freedom, flagged if
   the upper-tail p < 0.001. A lower-tail p < 0.001 is "errors conservative",
   not a failure. Systems with N < 3 get no system test.
3. **Overall.** One-sample KS test of all counted d² against χ²₂. The campaign
   fails if the one-sided test (d² stochastically larger) gives p < 0.01. A
   rejection only because d² is too small means conservative errors, not a
   failure.
4. **Per category.** The same three tests are also run separately for each
   instrument family (GRAVITY, PIONIER, MIRC-family, AMBER, MATISSE), reported
   and not part of the verdict, to show whether a failure is one instrument's
   loader or errors.

The number of counted epochs is fixed by the reference records, before any fit;
the expected number of chance flags is 0.001 times that number.

## Finding or definition

As in Track B, with these additions to the list of named choices that can make
a flag a definition difference: the paper's choice of observables or
wavelength window (rule 2), the flux-ratio definition (rule 6), the paper's
treatment of a third body (rule 4), and bandwidth smearing (rule 3). Each flag
is also tested for a convention error by refitting with u → −u, λ × 1.01,
λ × 0.99 and the published position mirrored, as in Track B. A flag that
survives all of these with 2Δloss between 0 and 13.816 is undecided.

## Summary schema

As Track B's, with `criteria` also recording `references_hash`, and each epoch
carrying its `class`, `windowed`, `observables`, and the matched granules.
