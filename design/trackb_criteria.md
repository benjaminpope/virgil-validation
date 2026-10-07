# Track B: per-epoch binary positions, registered criteria

Registered on 2026-10-07, before any fit was run. `scripts/trackb_fit.py`
implements this file. Each result records this file's commit and SHA-256
(`criteria` in the summary schema below). A change to anything here after the
first run is a new version of the campaign. The new version gets a new file,
and the old results stay as they were.

## What is tested

Track B has 20 single-fibre binaries in `~/data/eso_binaries/trackb/` (its
README, `manifest.json` and `<system>/system.json`). Each has per-epoch
positions published from GRAVITY or PIONIER data, and ESO Phase 3 OIFITS for
the same nights. For each epoch, virgil fits the Phase 3 products of that
night, and its position is compared with the published one.

The published positions were fitted to the same nights by other codes:
CANDID, PMOIRED, LITpro-like codes and the authors' own codes. Most of them
also used other reductions. So agreement tests virgil's reduction-to-position
chain against theirs on the same photons. It is **not** a test against
independent truth. A published orbit used these same epochs, so it adds
nothing independent either. The comparison is against the per-epoch
positions only.

## Data per instrument

The instrument comes from each file's `INSTRUME` header. δ Cir and TZ For
mix both instruments.

- **GRAVITY** (`SINGLE_SCI_VIS`, not calibrated). Only the closure phases are
  used, of `INSNAME = GRAVITY_SC`, between 2.05 and 2.38 µm, with the Brγ
  window 2.155–2.175 µm left out (it holds the hot stars' line, where the
  flux ratio is not the continuum's). The fringe tracker is not used.
  - **V² is not used.** It carries the transfer function, a multiplicative
    bias per baseline and wavelength. A free `vis_scale` per baseline or per
    epoch only multiplies the *errors*, so it cannot remove that bias, and
    the bias would bias the positions. Calibrating with the same-programme
    calibrators (`gravity_calibrators.json`, 2.8 GB, not downloaded) is a
    separate task.
- **PIONIER** (`TARGET_OIDATA_CALIBRATED`). All V² and closure phases, in
  every spectral channel, read with `virgil.OIData(files)`.

An epoch is one row of `reference_positions`. Its data are the files named
by `phase3_dp_ids` (with `:` written as `_`), read together. Rows with no
Phase 3 product are not fitted. Like the papers, the fits neglect orbital
motion within a night.

A file that virgil cannot read is left out of its epoch, and the file and
the reason are listed. The dry run found one: in AL Dor 2017-11-24
(`ADP.2018-09-28T21:20:56.836`), a closure triangle names baseline 3–4,
which has no V² row. The reader's handling of such files is noted as a
possible virgil limitation, not as a finding. If no file of an epoch can be
read, the epoch counts as "not fitted".

## Model

The model is a binary of two components, with the secondary at
(dra, ddec) mas (East, North) relative to the primary, and flux ratio
`flux` (secondary/primary).

- If `diameters_mas` gives a value for a component, that component is a
  `UniformDisk` fixed at that diameter. Otherwise it is a point. If both
  are points, the model is `BinaryModelCartesian`. For these compact
  photospheres (≤ 0.5 mas, apart from the stars below), the papers' LD
  diameters are used as UD diameters. At these sizes the difference is far
  below the errors.
- **Resolved primaries.**
  - ο Leo's primary (1.3 mas) was fitted per epoch by Gallenne+2023. Closure
    phases alone constrain it poorly, so the registered fit fixes it at
    their mean UD diameter, 1.285 mas, with the secondary at 0.49 mas. A
    second fit with the primary's diameter free, LogUniform(0.1, 10) mas, is
    reported as a sensitivity check and not counted.
  - η Oph (0.82/0.64 mas), ζ Boo (0.47/0.43), ψ Cen (0.424/0.211) and
    TZ For (0.414/0.197) are fixed at the published values.
- The flux ratio is one number per epoch, constant with wavelength.
  Bandwidth smearing within a channel is not modelled. That matters only
  for PIONIER's six broad channels at the widest separations (9 Sgr at
  21.7 mas, KQ Vel at 18.7 mas). It changes contrast more than position,
  and it is a *definition* difference (below).

## Priors (Jeffreys rule)

| parameter | prior |
|---|---|
| dra, ddec | Uniform(−H, H), with H the grid half-width |
| flux | LogUniform(1e-3, 1) |
| flux, near-equal pairs (AL Dor, ζ Boo, HD 41255, HD 188088) | LogUniform(1e-3, 2), since their flux ratios reach 1.03 |
| GRAVITY `phi_scale` | LogUniform(0.1, 10) |
| PIONIER `vis_scale`, `phi_scale` | LogUniform(0.1, 10) |
| ο Leo primary diameter (sensitivity fit only) | LogUniform(0.1, 10) mas |

For two points, (pos, f) and (−pos, 1/f) give the same normalized
visibilities. The bound f ≤ 1 picks the brighter star as the primary.

## Search

1. **Grid.** `likelihood_grid(model, data, grid)` of `BinaryModelCartesian`
   over dra and ddec in [−H, H], with H = max(3·sep_published, 30) mas.
   - The step is λ_min / (4 B_max) of the epoch's data, and flux takes 13
     log-spaced values between the prior's bounds.
   - The grid is only a search, so for GRAVITY it uses every k-th spectral
     channel, with k chosen to leave at most 24 channels.
   - The grid uses the errors as given.
2. **Fit.** `fit` of the full model is run from each of the 5 best grid
   peaks, with the flux maximized at each (dra, ddec) cell.
   - Peaks are at least λ_min / B_max apart.
   - Starts are clipped strictly inside the prior.
   - The fit uses all the selected channels and the error scales above.
   - The epoch's answer is the fit with the lowest loss. All 5 are kept in
     the output.
3. **Covariance.** `laplace_cov` on the data with the errors multiplied by
   the fitted scales, at the best fit, over (dra, ddec, flux). The position
   covariance C_v is its (dra, ddec) block, with flux marginalized. The same
   matrix on the unscaled errors is also reported.
4. **At-reference refit (diagnostic).** dra and ddec are fixed at the
   published position (for the near-equal pairs, at whichever sign is
   nearer), and flux and the error scales are refitted.
   - It reports 2Δloss = 2(loss_ref − loss_best), where loss is the full
     Gaussian negative log likelihood that `fit` returns with fitted error
     scales (including Σ log σ). The priors on dra, ddec, flux and the scales
     are Uniform or LogUniform, which are flat in the coordinate `fit`
     optimises and so add nothing to the loss (documented in `fit`); the
     script asserts that all priors are of these two kinds, so 2Δloss has no
     prior offset.
   - It is not a pass/fail quantity. It only separates optimizer misses from
     everything else (below): both losses are virgil's own likelihood, so
     2Δloss ≫ 0 is also what a convention error inside virgil produces.

## Reported quantities, per epoch

These are listed in this order:

1. **Raw χ²/N on the quoted errors.** χ² is Σ `whitened_residuals(model,
   data)²` at the best fit, with the uncertainties as delivered, and
   N = `data.n_independent`.
   - The rescaled χ² ≈ 1 is tautological, so it is not reported as a
     quality measure.
   - A fitted scale above 3 marks an error-model failure. The epoch is still
     counted, but it is listed, and its flag (if any) is diagnosed as a data
     problem first.
2. The fitted error scales.
3. dra, ddec, sep, PA, flux, and C_v.
4. The published position and its 1σ covariance C_p (conversion below), and
   Δ = virgil − published.
5. d², the 180° choice, and whether the epoch is counted.
6. The 5 peak fits, and 2Δloss at the reference.

## Published errors as 1σ covariances

The papers state each ellipse as (σ_maj, σ_min, θ). θ is the position angle
of the major axis, East of North. With û = (sin θ, cos θ) and
v̂ = (cos θ, −sin θ) in (East, North),
C_p = σ_maj² ûûᵀ + σ_min² v̂v̂ᵀ. The positions given as sep/PA become
dra = sep sin PA and ddec = sep cos PA.

| paper | systems | conversion to 1σ |
|---|---|---|
| Gallenne+2023 (CANDID) | HD 41255, HD 188088, ο Leo, HD 70937, HD 210763 | ellipse as published; it includes their systematic floors (8–50 µas and 0.02% of sep), which stay in |
| Gallenne+2019 (CANDID) | ψ Cen, NN Del, AL Dor | ellipse as published |
| Le Bouquin+2017 | 9 Sgr, HD 152314, HD 168137 | axes are FWHM: σ = FWHM / (2√(2 ln 2)) = FWHM / 2.3548 |
| Halbwachs+2020 | α Equ | ellipse as published (their errors after rescaling by 0.1626 × 1.0856, which they adopt as 1σ); d² with the unrescaled ellipse (÷ 0.17652) is also reported, not counted |
| Rowan+2026 (PMOIRED) | κ Vel | σ_RA, σ_Dec, ρ from the table, both σ multiplied by their s_ast = e^0.8 = 2.2255; d² with the unscaled bootstrap errors is also reported, not counted |
| Waisberg+2025 | ζ Boo, η Oph | no ellipses; isotropic σ per axis from their epoch scatter: 40 µas (ζ Boo) and 20 µas (η Oph) |
| Švrčková+2026 (PMOIRED) | δ Cir | ellipse as published |
| Schöller+2020 (CANDID) | KQ Vel | ellipse 0.07 × 0.03 mas at 6° as 1σ (stated to follow the CLEAN beam; convention uncertain) |
| Deshmukh+2026 (CANDID in PMOIRED) | CPD−71 172, TYC 1703-394-1 | ellipse as published |
| Gallenne+2016 | TZ For | its 2014–15 epochs predate PIONIER Phase 3: not compared |

**180° rule.** In the near-equal pairs (AL Dor, ζ Boo, HD 41255,
HD 188088), only the RV orbit sets which star is the primary. virgil's fit
uses interferometry alone, so it cannot know the sign, and these pairs are
compared modulo 180°: d² = min over s ∈ {+1, −1} of the distance of s·virgil
from the published position, and s is reported. When the two images are
many σ apart, taking the minimum does not change the distribution.
Other systems are compared with their sign as fitted.

## Agreement metric

d² = Δᵀ (C_v + C_p)⁻¹ Δ, with Δ = virgil − published in (dra, ddec) mas.

The counted d² is the larger of two values: d² with C_v from the errors
multiplied by the fitted scales, and d² with C_v on the errors as delivered
(`cov_virgil_unscaled`). A worse fit has larger fitted scales and hence a
larger C_v, so the scaled d² alone would reward it; taking the larger keeps a
bad fit from certifying itself. Both values are reported (`d2_scaled`,
`d2_unscaled`).

Both fits use the same photons, so their errors are correlated. Summing the
covariances then overstates the variance of Δ, and χ²₂ is a *conservative*
reference: d² values smaller than χ²₂ predicts are expected, not a failure.
The floors in Gallenne+2023 push the same way.

## Pass criteria (stated in advance)

**Counted epochs.** Every fitted epoch is counted, except:

- epochs with `below_lambda_2B` (separation below λ/2B), reported
  separately;
- η Oph's 2023-06-16 epoch, beyond the AT fibre field, which its paper also
  excludes. This is the only exclusion by flag: the script tests for exactly
  this system and date, and a `flag` in `system.json` on any other epoch
  is reported but does not exclude it;
- the ο Leo sensitivity fit;
- TZ For's nights (no published epoch with Phase 3). These are fitted and
  reported only. So are any other files that no published epoch names,
  grouped by night and instrument.

The run has 121 counted epochs in 19 systems, and 9 epochs reported
separately.

The criteria are:

1. **Per epoch.** Flag d² > 13.816, the χ²₂ quantile at p = 0.001. With 121
   epochs, about 0.12 flags are expected by chance.
2. **Per system.** S = Σ d² over the counted epochs is compared with χ² on
   2N degrees of freedom. A system is flagged if the upper-tail p is below
   0.001. A lower-tail p below 0.001 is reported as "errors conservative",
   not as a failure.
3. **Overall.** A one-sample KS test of all counted d² against χ²₂.
   - The campaign **fails** if the one-sided test for d² stochastically
     larger than χ²₂ gives p < 0.01. That is SciPy's `kstest(...,
     alternative="less")`: the empirical CDF lies below χ²₂'s.
   - The two-sided p is reported.
   - A rejection only because d² is too small means conservative errors.
     It is not a failure.

## Finding or definition

Every flagged epoch and system is examined, and each outcome is entered in
`trust/ledger.yml`.

- **Finding (`virgil`).**
  - A flag is a finding when 2Δloss < 0 at the reference: virgil's search or
    optimizer missed a better solution on its own likelihood.
  - It is also a finding when a convention error reproduces the offset: an
    East/North or uv sign, a PA origin, a wavelength or unit scale inside
    virgil, or a wrong diameter or flux handling in the model.
  - Every flag is tested for this, as a required step, by refitting the
    epoch with u → −u, with λ × 1.01 and λ × 0.99, and with the published
    position mirrored (dra → −dra, ddec → −ddec). A variant that brings
    the offset to d² < 5.991 (the χ²₂ 95%
    quantile) names the convention error, and the flag is a finding.
  - These are pinned as strict-xfail tests.
- **Definition.** A flag is a definition difference when our data prefer
  our position (2Δloss > 13.816, the same χ²₂ quantile as the per-epoch
  flag), none of the convention variants above reproduces the offset, and
  the offset follows from a choice that differs between the two analyses.
  A flag with 0 ≤ 2Δloss ≤ 13.816 is reported as undecided and is not
  ruled definition without such a named choice:
  - the reduction (ESO Phase 3 against the authors' pndrs or own
    calibration);
  - the observables (closure phases only against all of them);
  - fixed or free diameters and flux ratios;
  - bandwidth smearing;
  - the wavelength scale, only for an offset along the separation vector
    that is common to a system's epochs and is at most the instrument's
    calibration uncertainty as a fraction of the separation: 1% for PIONIER
    (its calibration is 0.35–1%) and 0.1% for GRAVITY. No wavelength scale
    is fitted. A larger common radial offset is the wavelength or unit
    scale case of a finding above, and the λ × 1.01 and λ × 0.99
    refits are what tell the two apart;
  - the reference point (δ Cir's companion is measured from the inner
    pair's centre of mass);
  - the error convention.
- **External.** An error in a published table (transcription, sign or
  epoch) that the authors' own orbit also contradicts is ruled
  `external:<paper>`.
- **Crosscheck.** A mistake in our comparison code is ruled `crosscheck`.
  `src/crosscheck/astrometry.py`, which does not import virgil, holds the
  ellipse conversion, the 180° folding, d², and the per-system and overall
  statistics. They are tested against SciPy.

## Summary schema

Each `<system>.json` written by `scripts/trackb_fit.py` holds:

- `criteria`: `{file, commit, sha256}`;
- `virgil`: version and commit;
- `system`, and `epochs[]`. Each epoch has:
  - date, mjd, files, instrument, n_obs;
  - `chi2_raw`, `n_independent`, `chi2_raw_per_n`, scales;
  - dra, ddec, sep, pa, flux, cov_virgil, cov_virgil_unscaled;
  - published, cov_published, `convention`;
  - delta, d2 (the larger of d2_scaled and d2_unscaled), d2_alt (the
    uncounted variant), sign, counted, reason, note;
  - peaks[], at_reference;
- `system_stat`: `{n, S, dof, p_upper, p_lower}`.

`scripts/trackb_summary.py` gathers all the systems:

- counted d²;
- per-epoch flags;
- per-system flags;
- the KS p values (one-sided and two-sided);
- the epochs with a fitted scale above 3;
- the verdict;
- the `criteria` block.
