# A test of GRAVITY's field-dependent aberrations with ζ Boo and η Oph

Status: **proposal, with a laptop smoke test** (2026-10-14). Nothing here is a fit result: no orbit has been fitted, and no
published orbit was used as a start. The blind orbit solver is designed elsewhere, and orbit fits on these two systems wait for
Ben's sign-off.

## 1. Why these two systems

GRAVITY Collaboration 2021 (A&A 647, A59, arXiv:2101.12098) shows that the optical aberrations of the VLTI (unequal and
field-dependent phase and amplitude maps over each telescope pupil, plus the fibre's own misplacement, tilt and defocus) bias
single-field astrometry. The bias grows with the offset between the fibre and the source, and the paper says it matters for
single-fibre pointings at large offsets and not for dual-field work, where the same aberration acts on both fibres. For the UTs
they report phase errors of 4–5° per telescope at 20 mas and 14–20° at 50 mas, and shifts of 0.1–0.5 mas on S2.

Waisberg+2025 (2025OJAp....8E..63W) observed two wide binaries in single-field mode with the ATs, with the fibre on the primary:

| System | Separations | Public epochs | Published scatter per axis |
|---|---|---|---|
| ζ Boo | 45–88 mas | 4 | 40 µas |
| η Oph | 10–265 mas | 6 | 20 µas |

Those separations are at or beyond the range over which the aberration model was calibrated (the paper's UT tests reach 50 mas).
The scatter of the epochs about a fitted orbit is therefore a direct measurement of the systematic, if the orbit is well
constrained and the fits are unbiased. That is what these two targets can test.

The repository now carries both as gallery tiles with positions-only references (`oidb/references/waisberg2025-zet-boo.json`,
`waisberg2025-eta-oph.json`), provenance for the ten Phase 3 products (`oidb/eso_phase3.json`), and a planned trust-graph
pipeline (`gravity-aberration-wide`).

## 2. Does virgil model the fibre-offset aberration?

No. A read-only search of `~/code/virgil` (source, design notes, docs) finds no aberration, phase-map or field-dependent
pupil term. What exists, and what is planned:

- **Built:** the intrinsic total spectrum, OI_FLUX and NFLUX observables, a grey flux scale, closure and differential phases.
  The model's visibility is the ideal one, Σ fᵢ(λ) exp(−2πi αᵢ·b/λ) over components, with a uniform-disk or other extended
  factor per component.
- **Planned, not built** (`design/spectro_interferometry_workflow.md`, §2.2 and §2.8, "6a PR C"): the *coupled flux* of a
  primary beam, F_k = k Σ fᵢ ηₖ(xᵢ, λ), the spectral-resolution kernel, and bandwidth smearing. Without them η = 1, which holds
  only when the scene is much smaller than the fibre beam. The GRAVITY-specific fibre injection (§2.7) belongs to `virgil-vlti`.
- **Not planned anywhere:** a per-telescope phase and amplitude map (Zernike pupil and focal-plane terms) as in the 2021 paper.

So a fit of these wide pairs with virgil today treats the instrument as ideal apart from the fitted `phi_scale`, and any
aberration bias lands in the fitted position.

## 3. What a minimal implementation needs

The 2021 model (their §3) writes the observed single-field complex visibility of a source at sky position α as

V_ij(α, λ) ∝ Σ_s f_s A_i A_j O  exp[−2πi α_s·b_ij/λ + i(φ_i − φ_j)],

with the per-telescope amplitude A_i and phase φ_i evaluated at the source's field position, each a Zernike expansion
(pupil terms A_n^m, focal-plane terms B_n^m) multiplied by the fibre mode. A minimal, *static* version for virgil would have:

1. **A per-telescope, per-component complex gain** g_{k,s}(λ) = A_k(α_s, λ) exp[iφ_k(α_s, λ)], supplied by the caller as a
   callable or a table over (telescope, offset, wavelength). Virgil's visibility is then Σ_s f_s g_{i,s} g*_{j,s} exp(−2πi α_s·b_ij/λ).
   This is the same hook as the planned primary-beam coupling η_k(x_s, λ) (the modulus of g), so it should reuse that interface rather
   than add a second one.
2. **The fibre offset per telescope**, from the file headers: `ESO QC MET SOBJ DRA1..4` and `DDEC1..4` (mas, the metrology
   fibre position relative to the science object, per telescope). `scripts/aberration_smoke.py` already reads these.
3. **The phase-map coefficients.** Widmann's `GRAVITY-Phasemaps` code (https://github.com/widmannf/GRAVITY-Phasemaps) generates the
   UT maps; the AT maps are the UT maps rescaled, as the 2021 paper does, and carry a larger uncertainty. We would check
   its licence and provenance before any use, and this note does not assume either.
4. **A prior, not a fit,** for the map coefficients at first: fix them at the published values and compare the residuals with and
   without (§5). Free Zernike coefficients are degenerate with the position for a single epoch, so they cannot be fitted per night.
5. **A test of the hook itself:** with the gains set to one the code reproduces the existing visibilities to round-off, and a
   unit-modulus gain with a pure tilt φ_k = κ_k·α shifts the fitted position by the analytic amount.

This is a modest change to the observable once the primary-beam hook exists, and it belongs in `virgil-vlti` with the GRAVITY
specifics, not in the core. Bandwidth smearing and the spectral-resolution kernel are separate and should be done first, because at these
separations they are at least as large (§6).

## 4. HIP 41426 has no archive data

The 2021 paper's own test binary is HIP 41426. The ESO archive holds no GRAVITY data on it that we know of (the task
brief's own check; this note did not repeat the query), so the paper's own validation cannot be repeated here. ζ Boo and η Oph
are the wide single-field AT pairs with public Phase 3 products that Waisberg+2025 used; we have not searched for others.

## 5. How per-epoch scatter would be compared with the published floors

Two reference numbers:

- **Waisberg+2025:** scatter-based errors of 40 µas (ζ Boo) and 20 µas (η Oph) per axis, from the residuals of their own orbit
  fit (ORB6 + GRAVITY).
- **Gallenne+2023** (Table B.1): systematic floors added in quadrature by fractional flux of the companion, 50, 20, 12 and
  8 µas for f ≈ 10, 30, 50 and 80 %, derived from tests at separations of at most 50 mas.

The comparison, for each system:

1. Fit each night blindly, with the registered Track B recipe, and record the fitted position and its formal error.
2. After a blind orbit is accepted (Ben's sign-off), compute each night's residual from the orbit's prediction, in the radial and
   tangential directions, and the corresponding χ² with the formal error, with the floor added in quadrature, and with the
   Waisberg scatter added instead.
3. Test whether the *radial* residual grows with separation across the 10–265 mas range (η Oph) and whether the residual
   direction follows the fibre offset vector from the headers. A systematic with a fixed direction in the pupil would rotate
   with parallactic angle and not with the sky PA, which separates it from an orbit error.
4. Repeat with the static aberration model on, and report whether the scatter falls to the formal level.

A residual several times the Waisberg scatter, nights apart, would be the signature; a residual at the floor says the paper's
numbers are fair. Four and six epochs are few, and the orbit fit absorbs three or more of the degrees of freedom, so this is a
consistency check and not a measurement of the aberration.

## 6. Local smoke test (single-night binary fits)

`scripts/aberration_smoke.py` runs the registered per-epoch recipe (`scripts/trackb_fit.py`) on single nights, with no orbit:
GRAVITY_SC closure phases only, 2.05–2.38 µm without Brγ, a likelihood grid, then `fit` from the five best peaks, fixed
uniform-disk diameters from the paper, a free `phi_scale`, and a flux ratio prior up to 2 for the near-equal pairs. The data are
**uncalibrated** and the likelihood is **closure-phase only**; this is a laptop check of the pipeline, not a campaign. The
residuals are the fitted position minus Waisberg's published position (Tables 3 and 4), at the sign nearer the paper's.

| System | Night | Separation (published) | Residual (µas) in dRA, dDec | Radial, tangential (µas) | In published σ | Flux ratio fitted |
|---|---|---|---|---|---|---|
| ζ Boo | 2023-06-14 | ≈ 64 mas | −17, −10 | −20, −1 | 0.5 (σ = 40 µas) | 1.09 |
| η Oph | 2024-06-17 | ≈ 23 mas | +15, +3 | −7, −14 | 0.8 (σ = 20 µas) | 0.66 |
| η Oph | 2024-07-26 | ≈ 28 mas | −49, −168 | −131, +115 | 8.7 (σ = 20 µas) | 0.68 |
| ζ Boo | 2024-05-22 | 88 mas | wrong peak: fitted 43 mas | not meaningful | not meaningful | 0.10 |

Reading it, with the caveats first:

- Two nights land within one published σ of the paper. This shows the registered recipe reaches 20–40 µas agreement on a
  closure-phase-only, uncalibrated fit, which is the same order as the paper's own scatter.
- η Oph 2024-07-26 is 170 µas off, 8.7σ of the published scatter, mostly radial. It is the only night with an obvious
  disagreement, with a short maximum baseline (34 m, an AT compact configuration) that makes closure phases a weak constraint on
  the separation. The note does **not** attribute this to the aberration: there is no calibration, the fit is closure-phase
  only, and the night has not been fitted with the visibilities.
- ζ Boo 2024-05-22 went to a wrong peak (43 mas against 88 mas, flux ratio 0.098). Its maximum baseline is 25 m, so the
  companion at 88 mas is only marginally resolved by the longer closure triangles. This is unexplained and is a failure of the
  closure-phase-only grid on a compact configuration, not a measurement of anything.
- The residual 2Δloss at the published position is 325–1450 even for residuals of 17–49 µas, which says the formal errors of
  the closure-phase fit are tiny (a few µas) and cannot be compared with the paper's scatter.

Confounders virgil does not model, and which would have to be removed before the aberration can be isolated:

- **Bandwidth smearing** at R ≈ 4000 (coherence length Rλ ≈ 8.8 mm at 2.2 µm): a companion at ≳ 19 mas on a 94 m baseline
  loses visibility amplitude across a spectral channel. This depends on the same separation and baseline as the aberration
  effect, so it must be modelled first. (§2.8 of the workflow note.)
- **Fibre throughput** (the primary beam) at 57–88 mas, which changes the effective flux ratio per telescope and enters the
  closure phases.
- **Closure-phase-only, uncalibrated data**, as above.

## 7. What this note does not claim

No orbit is fitted. No published orbit was used to start a fit; the published orbits are used only to draw the gallery curves
and in `tests/test_waisberg_references.py`, which checks that they reproduce the paper's own positions. The smoke test is a
pipeline check on four nights, and says nothing about the aberration model. The decision to fit full orbits, with or without a
model of the aberrations, is Ben's.

## 8. Open questions (defaults in force)

1. Should the aberration hook be built in `virgil-vlti` after the primary beam, or specified now as an interface in the core?
   Default: wait for the primary beam (6a PR C), and reuse its interface.
2. Is `GRAVITY-Phasemaps` usable (licence, AT scaling)? Default: assume not, until checked.
3. Does Ben want the visibility-based single-night fits (calibrated) run on OzSTAR for the two compact-configuration nights
   before the orbit fits? Default: no; wait for the blind solver.
