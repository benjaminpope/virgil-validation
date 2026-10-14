# Orbit solver v2: a blind, deterministic solver for undersampled binary orbits

Design note, 2026-10-14. Status: **draft for Ben's sign-off; nothing here is implemented.** It replaces
the search strategy of `virgil.aliases.fit_orbit_aliases` (the alias track, `scripts/reanalyse_orbits.py`)
and of the orbit recipes in `scripts/oidb_fit.py` (the OiDB gallery track). The prototypes behind the
numbers in §9 are scratch code, not part of this branch; §9 says what each one did.

Contents: §1 what goes wrong now; §2 fixed decisions; §3 the pipeline; §4 the likelihood (tempering and
nuisance terms); §5 deterministic initialisation; §6 the optimiser; §7 sampling and evidences; §8
execution; §9 local prototypes and measurements; §10 validation and pass/fail criteria; §11 the
implementation plan; §12 review and responses; §13 open questions for Ben.

## 1. What goes wrong now

From the Opus review of 2026-10-14 (profiling in the coordinating session's scratchpad, `orbit_review/`)
and from the prototypes of §9:

1. **Speed.** `OIData._cvis_in_time` evaluates time-dependent models with a serial `lax.map` over
   samples, which makes every likelihood call about 21× slower than a vectorised one (Gl 229 band 34:
   1497 s against 83 s). A virgil session is fixing this; this design assumes it lands.
2. **Overconfident evidences.** Every spectral channel counts as an independent datum while the raw
   χ²/N is 15 to 3000, so band probabilities come out 1.000 every time. `aliases._loglike` hard-codes
   `dof=1.0`.
3. **Hidden use of the published orbit.** The alias track warm-starts the reference band at the published
   orbit and takes a period window of P_ref/1.25 to 1.25 P_ref (Track B) or 10–14 d (Gl 229). The gallery
   recipes take windows of 11.8–12.5 d (Gl 229), 39.5–41.5 d (HR 6819) and 10.0–10.4 d (ι Peg),
   eccentricity caps of 0.55, 0.35 and 0.3, and a `published=` start.
4. **Bands that fail score p = 0**, so a failed fit looks like a rejected alias.
5. **Unconverged optimisation.** Serial scipy L-BFGS-B restarts end with non-positive-definite or
   unconverged Hessians in many bands.
6. **Single-mode sampling.** NUTS chains start in one mode, so a clean R-hat hides a wrong alias.
7. **Single-peak initialisation.** Each night contributes its single best peak. §9.1 shows that the
   true Gl 229 position is not the best peak on 4 of 7 nights. On 3 of them (2024-02-27, 03-28, 12-18)
   it is not within 1 mas of any of the top 25 peaks. The gallery's Gl 229 orbit (i = 105°, P = 12.059 d)
   and ι Peg orbit (P = 10.33 d) are built on such wrong peaks (§9.5).
8. **Nuisance terms missing where they matter.** The per-night position search has no V² amplitude.
   The real V² amplitudes are 0.07–0.8, so a search without them is fitting the calibration. A first,
   contaminated injection suggested that this moves the best peak off the companion on most nights;
   §9.6 explains why that run cannot be trusted.

## 2. Fixed decisions (Ben)

1. **No published-orbit information anywhere in the fit**: no starts, no period windows or priors
   centred on the published values. Published orbits are read only by the comparison step, after
   fitting. §5.1 sets the period window from the data and from physics stated before fitting.
2. **Tempered likelihood.** Weight each block by its effective number of independent channels,
   estimated from the residuals' correlation along wavelength. Calibrate model and band probabilities by
   injection and recovery at the real (u, v, t) sampling.
3. **Both Gl 229 data selections are reported**: all 7 nights, and the §5 selection of
   `orbit_comparison.md` (2024-02-27 and 2024-03-28 flagged; 2024-12-18 excluded pending its ruling),
   which leaves 4 nights. The §5 selection is **reference-informed**: 2024-12-18 is held out because it
   disagrees with the published orbit. It is reported, labelled as such, and never used to tune
   anything.

## 3. The pipeline

Every stage is deterministic: fixed grids, fixed orderings, ties broken by index, no random numbers
before the sampler. The same data, settings and software versions give the same candidate list, and
its hash is written to the output.

| stage | what | output | measured (Gl 229, laptop) |
|---|---|---|---|
| S0 | Load the data. Register the period window, selections and settings before fitting | `settings.json` | – |
| S1 | Per-night peak catalogues: every local maximum of each night's static-binary surface, with the V² nuisance fitted, a covariance per peak, a tempered height and a "none of these" bound | `peaks.json` | 21 s (7 nights, 25 peaks) |
| S2 | Combinatorial Thiele–Innes screen over (P, e, τ), with every per-night mode choice | candidate orbits | 300 s (10–14 d window); §9.4 for wide windows |
| S3 | Direct-to-data scoring of every candidate on every night's observables, over a flux scan | ranked candidates | 10 s for 4000 rows |
| S4 | Vmapped Levenberg–Marquardt from every candidate; merge duplicates | the mode set | 10 s (32 starts × 40 iterations, compiled) |
| S5 | Per-mode curvature (Gauss–Newton and full Hessian), tempered Laplace evidence, importance-sampling check | mode table | 0.1 s per mode after compile |
| S6 | NUTS per mode with the inverse Gauss–Newton metric; between-mode weights from tempered evidences | posterior mixture | not prototyped (§7) |
| S7 | Injection and recovery at the real (u, v, t): calibrate the temperature and check coverage | calibration report | §9.6 |
| S8 | Compare with the published orbit (`crosscheck.orbits`, statistics A and B) | comparison table | existing |

S1 to S5 run in minutes on one node and are the same code for every system. S6 is the only expensive
stage. S7 repeats S1 to S5 many times.

## 4. The likelihood

### 4.1 Blocks, scales and the tempering factor

Each night n has observable blocks b (V², closure phases, others). Each block has an unknown error scale
s_nb under a Jeffreys prior, integrated out as now (`epochs.marginal_loglike`). This gives

  ℓ(θ) = Σ_n Σ_b −(f_nb ν_nb / 2) ln χ²_nb(θ),

where ν_nb is the block's number of independent observables and f_nb ∈ (0, 1] is the **tempering
factor**: the fraction of ν_nb that is effectively independent. f = 1 is the present likelihood. A
fractional power of a likelihood is a standard device for a misspecified model (Bissiri, Holmes & Walker
2016, JRSS B 78, 1103; Holmes & Walker 2017, Biometrika 104, 497). Here f is not a free knob: it is
measured from the residuals (§4.2) and then checked by injection and recovery (§10.2). The tempered
posterior has covariance (f-weighted curvature)⁻¹, so error bars widen by about 1/√f.

`aliases._loglike` passes `dof=1.0`. v2 passes the per-block f array instead, so every stage (S1 to S6)
uses the same ℓ.

### 4.2 Estimating f from the residuals

f is estimated **once, before S2, and then frozen** (review objection 7). It is pooled per (instrument,
observable), not set per night. Per-night values reward a bad night: on Gl 229, 2024-03-28 (flagged for a
failing fringe tracker) got f = 0.23 against 0.04–0.09 for the others, because a wrong peak that fits
its data well leaves white residuals. A per-mode f would favour the worse-fitting mode. Every result is
also reported at f × 0.5 and f × 2, and the injection multiplier c (§10.2) absorbs the overall scale.

At each night's best static peak, with the §4.3 nuisance model fitted, take the whitened residuals of
each block and group them into **spectra**: one exposure frame × one baseline (V²) or
triangle (closure phases), ordered by wavelength. Two estimators:

- **binning**: R(b) = b · mean(r̄_b²) / mean(r²), where r̄_b is the mean over b adjacent channels. For
  independent residuals R = 1 for every b; for residuals correlated over τ channels R rises to τ.
  f = 1/R on the plateau (below).
- **autocorrelation**: f = 1/τ_int, with τ_int = 1 + 2 Σ_k ρ_k summed until ρ_k first drops below 0.

Use the binning estimate, read from the plateau of R(b) over b ≤ L/4 (L channels per spectrum, so
there are at least four bins per spectrum), not from max_b R(b). A maximum of noisy ratios is biased
high, and at b ≈ L there is one bin per spectrum. Report both estimates, and floor f at
1/(channels per spectrum): a whole spectrum is never worth less than one datum. Measured on Gl 229
(§9.3): the raw residuals give f = 0.03–0.07 for V² and 0.04–0.15 for closure phases. That is about one
independent number per 59-channel spectrum, because the residual is mostly a per-spectrum offset. Once
the offsets are modelled (§4.3), the remaining f is 0.2–0.8. So the nuisance model and f have to be
estimated together: first model the offsets, then temper what is left.

### 4.3 Hierarchical nuisance terms

Driven by the variance split of §9.3:

1. **V² gains: per (night, baseline), and per spectrum only as a tested variant.** The model is
   V²_obs = g_s · V²_model + noise. On Gl 229 the per-spectrum offsets are 75% of the V² χ² (16 of 22 on
   2023-12-25), split 40/60 between a fixed per-baseline offset for the night and a frame-to-frame part.
   A free gain for every spectrum would absorb most of that, but it would also absorb the V² *signal* of
   a close companion (review objection 3). Across 2.05–2.18 µm, a companion at ρ mas on a 134 m baseline
   moves only 0.019 ρ fringe cycles: 0.14 cycles at 7.3 mas, 0.02 at 1 mas. Within one spectrum the V²
   signal is therefore an offset plus a slope, and a per-spectrum gain removes the offset. For the
   sub-resolution systems of §10.1, that offset is nearly all the information V² carries. So:
   - the default gain is one per (night, baseline), shared by every frame. Its prior width σ_g is
     hierarchical per night, so a frame-to-frame scatter in the gain is not modelled and falls to the
     error scale and f;
   - the per-spectrum gain is a model variant. It is adopted for an instrument only if injections of
     sub-resolution companions (§10.2) recover them as well with it as without;
   - both are linear in g, and they **marginalise analytically**. A conjugate prior is used,
     g ~ N(ḡ_n, s² τ²), whose width scales with the block's unknown error scale s. Then the marginal
     covariance is s²(W⁻¹ + τ² m mᵀ), the log-determinant splits into a term in s and ln(1 + τ² mᵀWm),
     and the Jeffreys integral over s still has a closed form. Per gain group:
     ℓ = −(ν/2) ln Q − ½ ln(1 + τ² mᵀWm), with
     Q = yᵀWy − (mᵀWy + ḡ/τ²)² / (mᵀWm + 1/τ²) + ḡ²/τ²,
     where m is the model, y the data, W the inverse error variances, and ν counts the group's data.
     This corrects the draft's rᵀWr (it is yᵀWy) and its log-normal statement (the prior is on g, with
     mean ḡ_n, not on log g). The gains add no parameters to the optimiser. ḡ_n and τ_n are optimised in
     S4 and sampled in S6. The case τ → 0 with ḡ_n free is the present per-night `vis_amp`.
   - **How this plugs into the GN layer** (agreed with the GN-layer session). Q is the squared norm of
     the variable-projection residual at the profiled gain ĝ: [W^½(y − mĝ), (ĝ − ḡ)/τ]. That residual
     is the block's "χ²" in the score, so the layer's IRLS weight is fν/Q. −½ ln(1 + τ² mᵀWm) depends on
     the model, so it goes into `loss(z)`, where its gradient is exact. Its curvature is either one extra
     row per block or dropped as small. virgil's existing `GainModes` (zero-mean, fixed width, not scaled
     by s) are a different prior, which is why `marginal_loglike` refuses them. This conjugate gain is
     P1's own nuisance term.
2. **Per-night, per-block error scales** (Jeffreys, as now).
3. **Closure-phase offsets per (night, triangle)**: available (`OIData.with_closure_offsets`) but off by
   default. On Gl 229 they carry under 10% of the CP χ² (§9.3); the CP excess is structure *within*
   spectra, which f handles. Turn them on per instrument when the variance split says so.
4. **Flux ratio**: one per band, shared by all nights by default. A hierarchical per-night flux
   (log f_n ~ N(log f̄, σ_f²)) is a model variant compared by evidence. The single-night fluxes of
   ι Peg (0.35–0.51 against a published 0.22) show why the shared flux must not be imposed in S1.
5. **Primary diameter, bandwidth smearing**: per instrument recipe, as now (Track B `PRIMARY_DIAM`; the
   CANDID smearing model for MIRC-X).
6. **Jitter**: the error scales already absorb a multiplicative jitter. An additive floor is a variant,
   used only if the per-block scales correlate with the error bars.

The same nuisance model is used in S1 (the per-night catalogues) as in S3 to S6. A V² search without
the amplitude is fitting a calibration error of a factor of 1.3 to 14, so the prototype's S1 for
injections used closure phases only. With the clean injector (§9.6), the closure-phase-only catalogue
puts the injected companion at rank 0 on 4 of 7 nights for both injections. The catalogue in S1 therefore uses either the full
nuisance model or closure phases only. It never uses V² without its gain. Every block's χ² is floored at
10⁻³ ν before the logarithm: a gain-marginalised block with few spectra can otherwise be fitted exactly,
which would send Φ to −∞.

## 5. Deterministic initialisation

### 5.1 The period window, without the paper

The window is set in S0 from data and physics, stated before fitting, and recorded in the settings:

- **field**: the S1 search radius ρ_max is set by the instrument, not by the system: the largest
  separation at which the model is valid. That is the bandwidth-smearing or fibre field-of-view limit
  (stated per instrument recipe, from the spectral resolution and the fibre field; for MIRC-X, the
  CANDID smearing limit).
  The prototype's ±18 mas was not chosen this way, which is review objection 10.
- **lower bound**: a night is a snapshot only if the companion moves less than the position resolution
  during it. At separation ρ ≤ ρ_max it moves up to 2πρ Δt/P in a night of length Δt, so
  P_min = 2π ρ_max Δt / (λ/4B). (The draft's 10 × Δt allowed 0.63a of motion in a night: 4.6 mas for
  Gl 229.) Systems with shorter periods (AK For at P = 3.98 d) need the within-night model: S1 then
  fits short-period orbits night by night, not static positions. That is a separate recipe, listed in
  §13.
- **upper bound**: P_max = 4 × the span of the data. Longer periods are a single short arc, the N = 0
  family, fitted the same way on a coarse grid. Candidate orbits whose track leaves the S1 field on a
  night score that night's bound (§5.2).
- **eccentricity**: 0 to 0.95 everywhere, never capped from a paper. (The prototype grid stopped at
  0.9.)

For Gl 229 (Δt ≈ 0.05 d, the within-night motion of ≤ 0.18 mas at 7.3 mas; λ/4B ≈ 0.8 mas), a field of
ρ_max = 30 mas would give P_min ≈ 12 d, which is the true period. So the static snapshot is not safe at
the edge of the field. The fix is to apply the rule per separation: at ρ ≈ 8 mas it gives P_min ≈ 3 d. So S1 records the separation of each peak, and the
screen uses only peaks with 2πρΔt/P < λ/4B at each grid period, with the rest bounded. Then the window
is 1 d to 1660 d, against the present 10–14 d. Its cost is set out in §9.4 and §8.

### 5.2 S1: per-night peak catalogues

For each night, a static binary (with the §4.3 nuisance model) on a position grid at λ_min/(4B_max),
over a flux grid. Keep the K highest distinct local maxima (K = 25; the cost is linear in K and is 21 s
for all of Gl 229). Refine each, and record:

- position x_nk and flux; the 3 × 3 covariance C_nk from the Hessian of −ℓ_n at the peak (position and
  flux). `EpochPeaks` has no per-peak covariance today; S1 adds it;
- the tempered height h_nk = f_n (m_nk − m_n0), relative to the night's best peak;
- the **truncation bound** b_n = h_n,K−1. If S1 found every local maximum of the continuous surface,
  every point outside the listed basins would lie in an unlisted basin, no higher than the lowest
  listed peak. It does not (review objection 5b): S1 searches a grid of finite pitch and a finite field,
  and it can miss a narrow peak. So b_n is a heuristic, not a bound. That is one reason S2's final score
  (below) is read from the gridded surface, not from the catalogue;
- the night's **whole gridded surface**, maximised over flux (47 × 47 floats for Gl 229), kept for S2's
  scoring.

**When a night cannot be trusted.** The ι Peg gallery job (OzSTAR 18576666, MIRC-X, 2026-10-14) shows
four ways S1 fails. None of them may crash the solver or silently bias it:

- **Lost closure-phase legs.** On every ι Peg file, 10 of 20 OI_T3 rows (16 of 20 on one) were removed.
  They were removed by `oidb_fit.py:160-185`, a pre-filter written before virgil 48f8727 learnt to
  place an unmatched leg from the OI_T3 coordinates; virgil itself would have kept them. v2 reads data
  only through `read_oifits` and records, per night, how many rows of each table it kept. A night that
  has lost more than half of a table is flagged in S0, before any fitting.
- **Calibration far off.** The raw χ²/N reached 1.1 × 10⁴ to 3.1 × 10⁴. The Jeffreys scale absorbs a
  uniform inflation, but not one that varies per baseline. S1 therefore fits the §4.3 gains before it
  ranks peaks, and reports χ²/N after the nuisance fit. A night whose χ²/N is still above 100 after
  the nuisances is marked **untrusted**: its catalogue is kept but carries no seeds (§5.3), and it
  enters S2 only through its interpolated surface.
- **Best peak at the resolution floor.** Two ι Peg nights put their best peak at 0.50 mas (below the
  0.82 mas floor), with flux 0.92–1.0. That is the near-equal-flux, unresolved solution that a badly
  calibrated V² prefers, while the companion is at 10 mas. S1 keeps K peaks and never acts on the best
  one alone, so such a night still contributes its other peaks. Peaks inside λ/4B are listed but not
  used as seeds.
- **Every night barred.** `oidb_fit.py:579` dropped the barred epochs and built an empty `Epochs`,
  which raised a ValueError. v2 never drops a night from the data. A night is downgraded (untrusted,
  no seeds), and S2 runs while at least two nights can seed. Otherwise the solver writes a result with
  status **"unsolvable: fewer than 2 seeding nights"**, listing each night's reason, and exits normally.
  That is a reported outcome, not an exception.

### 5.3 S2: the combinatorial Thiele–Innes screen

At fixed (P, e, τ) the predicted positions are linear in the Thiele–Innes constants β = (A, B, F, G):
μ_n = M_n(P, e, τ) β. Score an orbit against the catalogues by

  S(P, e, τ, β) = Σ_n max( max_k [h_nk − ½ (μ_n − x_nk)ᵀ W_nk (μ_n − x_nk)],  b_n ),

with W_nk = (C_nk/f_n + σ_g² I)⁻¹. Here σ_g is a **grid floor**: the largest position error that the
(P, e, τ) grid makes near the truth. Without it, a grid point 0.3 mas off a peak with σ = 0.02 mas would
score as if it were wrong. The draft fixed σ_g at 0.5 mas. The review measured the actual miss at
a = 7.3 mas, with β re-solved at the nearest grid point: 0.35–1.27 mas for e = 0.23–0.85, growing
linearly with a, to about 3.5 mas at a = 20 mas. A fixed floor therefore cuts the truth for wide or
eccentric orbits. In v2 the floor is computed per seed: σ_g = ½ Σ_θ |∂μ/∂θ| Δθ over θ = (P, e, τ),
evaluated at the seed's β, where Δθ is the grid cell. It is never less than 0.2 mas. Each night either
picks one of its peaks or takes the bound: this is the per-night mode choice.

At each grid point, maximise S over β and over the per-night choices:

1. **Seeds from well-conditioned minimal subsets.** Two nights give four equations for the four
   unknowns β. For a pair of nights (i, j) and a pair of their top K_p peaks (K_p = 5), solve the 4 × 4
   linear system exactly. The exact solve amplifies the peak and grid errors by cond(M_ij) (objection
   5a), so pairs are ranked by cond(M_ij) at each grid point. Only the best-conditioned N_pair pairs are
   used: N_pair = min(C(N, 2), 12), so 12 × 25 = 300 seeds per point. Above a threshold of
   cond(M_ij) > 10³, a pair is dropped and its two nights are combined with a third into a 6-equation
   least-squares seed. This also caps the cost for systems with many nights: the cost no longer grows
   as N², which matters for LL Aqr's 39 epochs (objection 11).
2. **Hard EM from each seed.** Assign every night to its best component (a peak, or the bound), re-solve
   the weighted least squares on the assigned nights, and repeat 3 times. This is a local ascent
   (objection 5d), which is why it is restarted from many seeds.
3. **Score on the surfaces, not the catalogue.** At the seed's final β, each night's contribution is its
   gridded S1 surface, interpolated bilinearly at μ_n and tempered: f (m_n(μ_n) − m_n0). Outside the
   field, the night contributes its bound. This removes the Gaussian-basin approximation from the
   ranking (objection 5c). The catalogue is used only to make seeds. Interpolating 2 × N floats per
   seed costs less than the 4 × 4 solve.
4. Keep the best seed's (S, β, assignment) for the grid point.

**What the seeding guarantees, and what it does not.** Suppose that, at the true orbit, at least two
nights have their true position within the basin of one of their top K_p peaks, and that their pair is
well conditioned at the grid point nearest the truth. Then that pair's seed is within the amplified grid
error of β_true, and hard EM starts in the truth's basin. Nothing guarantees that it stays there, and the
review is right that the draft's "completeness argument" was not a proof. Three things are not covered:
- orbits identifiable only jointly, from three or more low-SNR nights, none of which shows the truth
  among its top peaks (objection 5f);
- a pair whose two nights both have their truth near a missed (narrow) peak;
- a wrong orbit seeded by two wrong peaks. It looks seeded, so "unseeded" cannot be detected on real
  data (objection 5e), and the draft's claim that it could was wrong.

Completeness is therefore measured, not argued: per system, by injection–recovery at the real (u, v, t)
(§10.2), including low-SNR injections where at most one night shows the truth. It is reported as the
completeness of each stage, S1 to S4. The brute-force test (§10.3) checks the seeding against full
enumeration on small problems.

**Grid.** P from `period_grid` (9 points per cycle of drift over the span) × e from 0 to 0.95 in steps
of 0.05 × τ in n_τ phases. n_τ grows with e, since the phase of periastron passage is sharper at high
e: n_τ = 36 / (1 − e)^{1/2}, rounded up. With the per-seed floor, a coarser grid costs completeness
only gradually, not at a cliff.

**Candidate selection: no global cap.** The screen does not rank reliably: on Gl 229 the truth-like
candidate is 21.6 tempered nats below the screen's best (§9.2). So S2 is only a **filter**. In the
prototype, both selections hit the 400-candidate cap before reaching Δ_S = 25 nats (objection 2). The
last kept candidate was at −23.3 (all nights), and the truth at −21.6 was 1.7 nats from being cut, in a
window with 34 times fewer periods than the blind one. A global cap tuned on the answer is the most
fragile choice in the draft. v2 instead keeps, **per (band N, e-bin of width 0.1)**, the best m = 8
distinct candidates (sky tracks more than 0.5 mas apart at some epoch). This is a stratified selection
whose size is set by the grid, not by the score. Δ_S applies only within a cell, in **untempered**
units, so it does not stretch when f changes. For Gl 229's blind window that is about 3700 × 10 × 8 / 9
≈ 3 × 10⁴ candidates, and S3 scores them in about 14 min at 2.5 ms per row with 10 fluxes (§8.4). The
de-duplication is vectorised (the prototype's O(n²) Python loop was the slowest step at large n).

The constants for the near-equal-flux systems (al_dor, hd41255, hd188088) are already handled by
`starting_orbits(sign_free=True)`. In S2 each peak simply appears twice, at ±x, with the flux inverted.

### 5.4 S3: direct-to-data scoring

Each S2 candidate × a 10-point flux scan (0.05–1, log-spaced) is scored by the full tempered ℓ (§4) on
every night, batched (`jax.vmap` in chunks of 256). S4 then starts from the best m_4 = 2 per (band,
e-bin) cell and from the top 256 overall: every band keeps a start, so no band is lost to a ranking
error. This is the first stage that sees the observables,
and it is the stage that ranks. In §9.2 the truth-like candidates move from screen rank ~291 to S3 ranks
0–4 (all nights) and 2, 8, 20 and 31 (§5 selection). The best flux per candidate starts S4.

## 6. The optimiser (S4)

### 6.1 Coordinates

S4 works in the **Thiele–Innes coordinates**

  x = (log P, τ, u, A, B, F, G, log flux, nuisance hyperparameters),  e = 0.95 σ(u),  t_peri = t_ref + τP.

- There are no angles. A, B, F, G are Cartesian (mas). There is no i ∈ [0°, 180°] bound, and the sense
  of motion is the sign of AG − BF, so it changes continuously. The (ω + 180°, Ω + 180°) twin of Campbell
  elements is a single point, which halves the number of modes to find and merge.
- τ (the periastron phase at t_ref, in cycles) is unbounded. It enters only through the mean anomaly,
  so it needs no wrapping during the fit. It is reduced modulo 1 at the end.
- log P is unbounded. The fit may cross from one alias band to the next, which is fine: bands are labels
  assigned after convergence, N = round(T/P).
- e uses a logistic map onto [0, 0.95). At e → 0, τ and a rotation of β become degenerate. Levenberg
  damping handles that flat direction, and the mode table reports τ as undefined when e < 0.02.
- t_ref is the mean epoch of the data, so the P–τ correlation is small.

Positions are linear in β, so Gauss–Newton is exact for the geometric part and the optimiser converges
in a few steps (§9.2: 5 iterations from the S3 starts).

### 6.2 Levenberg–Marquardt with marginalised scales

The objective is Φ(x) = −ℓ(x) = Σ_nb (f_nb ν_nb / 2) ln χ²_nb(x), not a sum of squares. Its gradient is
exactly that of a weighted least-squares problem with weights w_nb = f_nb ν_nb / χ²_nb(x) on the block
residuals r_nb, re-evaluated at every step (iteratively reweighted least squares):

  ∇Φ = Σ_nb w_nb J_nbᵀ r_nb,  H_GN = Σ_nb w_nb J_nbᵀ J_nb.

The step solves (H_GN + λ diag H_GN) δ = −∇Φ. It is accepted if Φ falls; λ is divided by 3 on
acceptance and multiplied by 4 on rejection, as in `zodiax`/`optimistix` style LM. Linear nuisances (the
V² gains, §4.3) are marginalised analytically inside r, which is variable projection (Golub & Pereyra
1973): Gauss–Newton on the projected residual is valid.

**Vmapped multistart.** All S3 starts run together: `jax.vmap` over starts of a `lax.scan` over a fixed
number of iterations (40), with per-start λ. No Python loop, no scipy and no per-start compilation. The
Jacobian is `jax.jacfwd` (8–20 parameters, so forward mode costs a few model evaluations). Measured:
7.5 ms per start-iteration for the 7 Gl 229 nights (§9.2); 32 starts × 40 iterations in 10 s. Converged
modes (gradient norm in the H_GN⁻¹ metric < 10⁻³, and Φ changed by less than 10⁻⁶ over 5 iterations)
are merged by sky track (0.5 mas at every epoch). Unconverged starts are reported, never dropped.

### 6.3 Where Gauss–Newton stops being enough

H_GN drops two terms of the true Hessian of Φ:

- the residual-curvature term Σ w r ∇²r, of relative size about √(χ²/ν) / √ν per block. This is small
  once the nuisances are fitted, but not when χ²/ν ≫ 1 remains;
- the rank-one term −Σ_nb (2 w_nb / χ²_nb)(J_nbᵀ r_nb)(J_nbᵀ r_nb)ᵀ from the ln χ² (the scale
  marginalisation). It is negative semidefinite and vanishes only where every block's own gradient
  vanishes, which is not the case at a joint mode.

So H_GN is used for the steps and for the NUTS metric (§7), where positive definiteness is what matters.
**The evidence (S5) uses the full Hessian** from `jax.hessian`, computed once per mode. It costs 8 s to
compile and 0.1 s per mode after that. At the Gl 229 mode the two differ by 1.1 in ln det (all nights)
and 0.23 (§5 selection), so a Laplace evidence from H_GN would be off by about 0.5 nats. If the full
Hessian is not positive definite at a converged mode (a ridge or saddle), the mode is flagged. Its
evidence then comes from importance sampling with a Student-t (ν = 4) proposal scaled by H_GN⁻¹, and it
needs an effective sample size of at least 200 and agreement with Laplace within 1 nat to be accepted.

### 6.4 Building on virgil's Gauss–Newton tools, not beside them

The zodiax-style plan is already half built in virgil, but the orbit code bypasses it. The plan is to use
the curvature at the mode both to precondition the optimiser and as the NUTS metric
(https://louisdesdoigts.github.io/zodiax/optimisation_tools/). Strictly, zodiax's "Fisher" is the
Hessian of the loss, used for natural-gradient steps and eigen-projection; virgil uses the Gauss–Newton
JᵀJ in its place. A GN audit of virgil (coordinating session, 2026-10-14, `gn_audit/` e1–e9) found the
following, and its findings are folded in here and in §7:

- `fitting.fit(..., method="lm")` is a Levenberg–Marquardt (optimistix) on `_Problem.residuals`.
  `fitting.gauss_newton_mass` (fitting.py:881) gives the NUTS `inverse_mass_matrix` from JᵀJ, including
  the prior residuals. `_gauss_newton_covariance` is jitted and uses Woodbury. `angles.AngleVector`
  is recognised by both, and `svi.py` starts its guides from it.
- The orbit paths do not use any of it. `aliases._optimize` (aliases.py:503) and `epochs.py:1311` use
  scipy L-BFGS-B, and aliases.py:494 notes that L-BFGS-B stops early on the curved ridge.
  `pipeline/binary.py:813` runs NUTS with `dense_mass=False`, and `oidb_fit.run_nuts` passes no mass
  matrix.

Why the orbit paths never reached these tools: `_check_least_squares` (fitting.py:339-362) rejects a
fitted `noise=`, gains and `log_norm`, and every orbit path uses scale marginalisation, gains, or both.
visibility_orbits.md V19 (2026-10-06) planned `gauss_newton_mass(noise=…)`, but it was never built.
The audit measured what the tools would buy:
- on the scale-marginalised alias objective, damped Newton from the start reached the same optima as
  L-BFGS-B with restarts and polish in 82 against 230 evaluations (near starts), and 37 against 341
  (mid starts);
- LM beat L-BFGS-B from near orbit starts: 88% against 75% success, in 9 against 126 steps;
- from far starts every method failed. That is multimodality, which is what S1–S3 are for.

v2 moves the orbit solver onto these tools. Three things stop them working as they stand:

1. **No least-squares form for the orbit likelihood.** `_Problem._check_least_squares`
   (fitting.py:339) refuses data with gains or marginalised flux scales, fitted error terms, and terms
   with a `log_norm`, and sends them to L-BFGS. Two terms of the orbit likelihood are not sums of
   squares: the Jeffreys-marginalised block scales (Φ = Σ (fν/2) ln χ²) and the gain marginal's
   ln(1 + σ² mᵀWm). The extension is a second objective form in `_Problem`: **block-weighted
   residuals**. `residual_blocks(z)` returns (r_b, fν_b) per block. The loss is Φ. The gradient and
   H_GN use the IRLS weights w_b = fν_b/χ²_b of §6.2, and the gain log-determinant joins the gradient
   exactly. Its curvature is either dropped (it is smooth and small) or added through its own Jacobian.
   optimistix's `least_squares` cannot carry this: with frozen weights, its own loss ½ Σ w_b χ²_b equals
   Σ fν_b/2, a constant, so its acceptance test means nothing. The LM loop is therefore virgil's own,
   about 40 lines with an explicit Φ acceptance test, as in the prototype `exp3_lm.py`. It shares
   `_Problem`'s bijections, priors and residual builder. `fit(method="lm")` keeps optimistix for true
   least-squares problems and dispatches to the IRLS loop when the problem has block weights.
2. **One start at a time.** `fit` runs one start, in a Python loop over optimistix chunks. v2 adds
   `fit_multistart(problem, Z0, n_iter)`: `jax.vmap` over starts of a `lax.scan` of the IRLS-LM step,
   with damping per start, a fixed iteration count, and per-start convergence flags. A test checks that
   it agrees with `fit` from a single start, on a least-squares problem where both apply.
3. **`gauss_newton_mass` is not robust for orbits as it stands.**
   - It forms JᵀJ of unit-weighted whitened residuals. Under scale marginalisation the right weights
     are w_b = fν_b/χ²_b. On Gl 229 (χ²/ν = 3–37, f = 0.04–0.6) the unit-weighted matrix is stiffer by
     a factor of 5–900, differently per block, so NUTS would adapt its step size around a wrongly
     shaped metric.
   - At e → 0 and face-on, JᵀJ is singular. The GN layer always adds the prior and transform
     curvature as a diagonal, D = max(0, ∂²_z[−log p − log|dx/dz|]) per coordinate (the NUTS
     potential's own curvature). There is no threshold rule, so nothing changes discontinuously, and the
     result is identical to today wherever the transform is the identity. This replaces both the draft's
     diagonal fallback, which discarded the P–τ correlation, and its condition-number trigger (review
     objection 12).
   - The fix: it takes the same block weights (from `residual_blocks`) and reports its condition
     number. The orbit model must be built through the callable-model path, so that the sampling
     coordinates of §7.1, with `AngleVector` for Ω and the √e vector, are `_Problem` sites.

**Who builds what.** Ben accepted the GN audit's six recommendations, and a separate session plans the
library layer in virgil `design/gauss_newton_plan.md` (branch `gauss-newton-plan`). The orbit solver
builds on that layer and does not duplicate it. The proposed split, sent to that session on
2026-10-14, was **confirmed by that session the same day**, against its draft at virgil commit 43b0202
(branch `claude/gauss-newton-plan-8cc372`). Their PRs are numbered GN1–GN5.
- **The GN layer owns:**
  - the kernel `virgil/_gauss_newton.py` (GN1). It is dense damped GN with pure `init`, `step`,
    `solve` and `curvature` functions, box bounds, acceptance on Φ itself, stop codes, and is safe under
    `vmap` and `scan`. `_Surface` moves to `virgil/_marginal.py`, with `weights(chi2, dof, data)` giving
    the IRLS weights (also GN1);
  - the private `_Objective(marginal=f)`, with a per-dataset, per-block tempering factor f defaulting to
    1, IRLS-weighted `curvature_rows`, and `flat_problem(scale)` returning a kernel problem (GN3);
  - V19 `gauss_newton_mass(noise=…)` with gains, the PSD check and the size guard (GN2), and the guarded
    `mass="gauss_newton"` in the pipelines (GN5);
  - **the `aliases._optimize` rewrite** (damped Newton/IRLS-LM from the start, L-BFGS-B only as a
    fallback), and the same change to `epochs._refine_marginal` (GN4).
- **The orbit solver owns:**
  - `fit_multistart` (item 2): a `vmap` of `_gauss_newton.solve` (or a `scan` of `step`) over starts, on
    an `_Objective`;
  - the Thiele–Innes coordinates, catalogues, screen, mode merging, evidences and per-mode NUTS seeding.

Time-dependent models already work through `fit`'s callable-model path (`OrbitalBinary` evaluated at
each sample's time), once the vectorised `_cvis_in_time` (P0) lands. Without it, every Jacobian costs
21 times more. S1's per-night static search, `epochs.py:1311`, moves to the same IRLS-LM (static binary,
per-night block weights), which retires its last scipy call. zodiax's `eigen_projection` (preconditioning
in the Fisher eigenbasis) is not needed for 8–20 parameters, where a dense solve is cheap. It becomes
useful only if the hierarchical per-night terms (§4.3) push the parameter count past a few hundred.

## 7. Sampling and evidences (S5, S6)

- **Modes kept**: every distinct converged mode within Δ_keep = 20 tempered nats of the best (about
  e⁻²⁰ in posterior odds before the volume factor). The rest are reported in the table and not sampled.
- **Per-mode evidence**: Laplace with the full Hessian, plus the prior volume (§7.1), checked by
  importance sampling (S5). The between-mode weights are p_k ∝ Z_k. The band probability is the sum over
  the band's modes. A band with no converged mode is reported as **"no mode found"**, with its starts and
  failures, and never as p = 0.
- **NUTS per mode**: numpyro NUTS, dense inverse-mass metric = H_GN⁻¹ at the mode, in the sampling
  coordinates of §7.1 (`gauss_newton_mass` already computes it). 4 chains per mode, started at the mode
  plus draws from N(0, H_GN⁻¹). Chains are vectorised within a task. A chain that leaves its mode's basin
  (its sky track moves beyond the merge distance from the mode at more than 10% of draws) is reported as
  a **mode merge**. That is a finding, so the mode set is redone; such a chain is never silently pooled.
- **The GN metric is fixed, never re-adapted, with a guarded fallback.** The GN audit measured, on an
  8-parameter orbit dominated by its data:
  - a fixed GN mass took 6.3 leapfrog steps per draw, against 87 for an adapted diagonal mass, for
    about 50 times more ESS per gradient;
  - re-adapting the mass after starting from GN lost the gain (78 steps per draw). So the step size
    adapts, and the mass matrix does not;
  - with χ²/N large, the shape of the metric still held: eig((JᵀJ)⁻¹H) was 0.86–1.09.
  The sampled block scales (when they are sampled rather than marginalised) take the V19 form
  blockdiag(JᵀJ, 2N_b): their cross terms with the orbit vanish at the MAP.
- **The fallback is triggered by divergences.** The same audit found 35–59 divergences per run at
  e = 0.03, from the curved, bounded ridge near e = 0 (an `AngleVector` alone did not help), as well as
  failures for non-identifiable faint companions. A mode's chains are rerun with an adapted dense mass
  if they show more than 1% divergences or a tree depth at its limit on more than 5% of draws. The mode
  table records which metric was used. The (√e cos ω, √e sin ω) coordinates of §7.1 remove the e = 0
  boundary, and whether they remove those divergences is a P7 test at e = 0.03, the AK For case.
- **Mixture posterior**: draws from mode k weighted p_k / n_k. R-hat and ESS are reported per mode, and
  the mixture is never summarised by one R-hat.
- **Evidence check from the samples**: bridge sampling between each mode's NUTS draws and its Laplace
  Gaussian gives a second ln Z_k. A disagreement above 1 nat is flagged.
- **Is the truth outside the mode set?** The p_k are normalised over the modes that were found, so a
  wrong orbit can come out with p ≈ 1 (review objection 6). Three blind diagnostics go with every result:
  1. **leave-one-night-out prediction.** Refit the top mode on N − 1 nights, from its own parameters (S4
     only). Predict the held-out night's position, and report the held-out night's static surface at
     that position, relative to that night's best peak and to its peak nearest the prediction. A good
     orbit predicts each night to within its peak's error, or to within a known bad-night offset. A
     wrong alias usually fails on at least one night.
  2. **the orbit against free positions.** The tempered evidence of the top mode against a model with
     an independent static position and flux on every night (the sum of S1's best marginals, with
     their Laplace volumes). If free positions win by a lot, the orbit is not explaining the data.
  3. **the orbit against a single star.** The tempered evidence of the top mode against no companion,
     with the same nuisance model. The null injection of §9.6 shows why this is needed: with the real
     companion removed, the solver still returned a faint (flux 0.08), wide (21 mas) "orbit" fitted to
     residual systematics.
  None of these is a probability. All three are calibrated by injection (§10.2), including null
  injections, so "a held-out residual of x is rare for a correct orbit" and "an evidence ratio of y
  is rare without a companion" have numbers attached.

### 7.1 Sampling coordinates and priors

NUTS samples Campbell-type coordinates with vectors for the angles (virgil's `AngleVector` and the
position-angle parametrisation of `orbits.position_angle_prior`): log P, (√e cos ω, √e sin ω), cos i, Ω
as a vector, θ (the position angle at t_ref, well defined at e = 0), log a, log flux. The priors are the
present ones (log-uniform P over the §5.1 window, uniform e, isotropic orientation, log-uniform a and
flux). The S4 modes are converted with `ThieleInnesOrbit.to_kepler()`, and H_GN is recomputed in these
coordinates. Where the Laplace evidence is computed in Thiele–Innes coordinates, the isotropic prior
needs the Jacobian

  |∂(A, B, F, G) / ∂(a, cos i, ω, Ω)| = a³ sin² i,

checked numerically to 10⁻¹⁵ during prototyping, and checked again by the review. Two corrections from
the review (objection 8):
- the map from Campbell elements to Thiele–Innes constants is 2:1, since (ω + π, Ω + π) gives the same
  β. A Laplace evidence in Thiele–Innes coordinates therefore takes the prior density times 2, or the
  modes evaluated in the two coordinate systems are ln 2 apart;
- at e → 0, τ is flat, and in the logistic u the prior density σ′(u) → 0 along a half-infinite ridge. A
  Laplace approximation is meaningless there. **Laplace evidences are therefore computed in the
  regular coordinates** (log P, √e cos ω, √e sin ω, mean longitude at t_ref, cos i, Ω vector, log a,
  log flux), which are smooth through e = 0. Thiele–Innes coordinates are used only for the optimiser.
  Within 10° of face-on, where these are singular too, the evidence comes from importance sampling.

**Tempering and the Occam factor.** Under a power likelihood, the prior-volume and ln det terms are not
tempered, but the likelihood is. Their weight relative to the likelihood grows by about 1/f ≈ 5–20, so
band probabilities depend more on the period prior than they would at f = 1. This is a property of
tempering, not a bug, and it is handled two ways: band probabilities are also reported under a second
period prior (uniform in frequency against log-uniform), and the injection calibration (§10.2), which
uses the same prior as the fit, says whether the reported p are right.

## 8. Execution

### 8.1 Shape of the jobs

- **S1–S5, one node per (system, data selection, model variant).** Everything is vmapped over a leading
  axis: grid points (S2), candidates (S3) and starts (S4). On a multi-core node, shard that axis over CPU
  devices with `jax.sharding` / `shard_map` (`XLA_FLAGS=--xla_force_host_platform_device_count=<cores>`,
  one device per core, a `Mesh` over the batch axis). `pmap` is legacy and is not used. The laptop used
  about 3.4 of 10 cores, through XLA's own threading. Explicit sharding of an embarrassingly parallel
  axis should scale close to linearly, and that is to be measured on the first OzSTAR run (§11, PR V4).
- **S6, a Slurm array with one task per mode** (4 vectorised chains each), then a **combine job** that
  reads every mode's ln Z and draws and forms the mixture with log-sum-exp. Per-mode tasks are
  independent and of similar length, so an array is the natural shape.
- **S7, a Slurm array over injections**, each a full S1–S5 run on one node (or a slice of one).

### 8.2 How to split S2

v1 split the work by alias band: each task compiled its own kernels (about 20 s each, for several
kernels) and paid Python start-up, for a few minutes of work. In v2 the period axis is one axis of the
S2 grid, and S3 and S4 need the pooled candidates, so splitting by band duplicates compilation for
nothing. But a blind-window S2 is hours of work per system (§8.4). The review argued, and this design
accepts, that the **default split for S2 is a Slurm array over contiguous frequency ranges**. Each task
writes its per-(band, e-bin) candidates, and a gather job runs S3–S5 on one node. It is simpler and more
robust than one long `shard_map` job: a failed task reruns alone, and the wall-time limit applies per
task. Within each task, `shard_map` over the node's CPU devices parallelises the grid axis. A single
`shard_map` job with no array is kept for narrow windows and for the injections, whose S2 takes minutes.

### 8.3 CPU, float64

All of it is float64 on CPU. The consumer RTX Blackwell cards run float64 at a small fraction of their
float32 rate, the batches here are modest, and the v1 GPU jobs bought nothing. GPUs are not requested.

### 8.4 Expected cost

Laptop measurements (Apple M4, 10 cores, `fastcvis` patch for the vectorised visibilities):

| stage | Gl 229, 7 nights, P 10–14 d | Gl 229, §5 selection (4 nights) |
|---|---|---|
| S1 (25 peaks per night) | 21 s | (same catalogues) |
| S2 screen | 300 s (73 872 grid points × 525 seeds) | 88 s (× 150 seeds) |
| S3 scoring | 10 s (4000 rows) | 12 s |
| S4 LM (32 starts × 40 iterations) | 10 s compiled, 15 s with compile | 12 s / 20 s |
| S5 full Hessian per mode | 8 s compile, 0.1 s after | same |

Against v1's 7–24 h per OzSTAR task. S6 is not measured.

**Blind windows (extrapolated, not measured).** The S2 cost is N_P × (points per period) × (seeds per
point) × (cost per seed). The inputs are:
- N_P = 9T (1/P_min − 1/P_max) ≈ 9T/P_min;
- 20 eccentricities × on average 59 phases (n_τ = 36/√(1 − e)), i.e. 1180 points per period;
- 300 seeds per point (§5.3);
- the cost per seed scales roughly with the number of nights N. The laptop managed 1.3 × 10⁵ seeds/s
  at N = 7, on about 3.4 effective cores.

| system | N nights | span T (d) | N_P (P ≥ 1 d) | S2, laptop | S2, one 32-core node (assumed ≈ 10× the laptop) |
|---|---|---|---|---|---|
| Gl 229 | 7 | 414 | 3700 | ≈ 2.8 h | ≈ 17 min |
| HD 224974 | 13 | 288 | 2600 | ≈ 3.8 h | ≈ 25 min |
| LL Aqr | 29 | 706 | 6400 | ≈ 19 h | ≈ 2 h |

The 10× node speed-up is an assumption for V4 to measure. S3 and S4 add about 1 h per system on the
laptop at the §5.3 candidate counts.

**S7 dominates the budget.** Each injection is a full S1–S5. 300 + 100 + 8 × 30 = 640 runs per system
and selection, at about 20 min of node time each for Gl 229, is about 200 node-hours. Across the 20 or
so systems of §10.1 that is several thousand node-hours. §13 asks Ben how to phase it. At about 9 ms per gradient and 30–60
leapfrog steps per draw with a dense GN metric, 2000 draws per chain take 10–20 min on one core. §9.4
gives the cost of the wide blind window.

## 9. Local prototypes and measurements

Scratch code (not committed): the coordinating session's notes folder has a copy and the commands
(`orbit_solver_v2_prototypes/`). All runs used `~/code/virgil/.venv/bin/python` (virgil at `da76726`),
float64 on the laptop CPU, the Gl 229 GRAVITY data in `~/data/eso_binaries/gravity/oifits/` (7 nights,
210 channels in 2.05–2.18 µm, V² and closure phases), and the `fastcvis.py` vectorised-visibility patch
from the review folder. **The reference orbit (Xuan et al. 2024) was read only after each blind step**,
to report where it landed.

### 9.1 S1 catalogues on Gl 229

`epoch_positions` (V² + CP, no amplitude term), grid 47 × 47 × 13 (0.79 mas step, ±18 mas): 17 s for
8 peaks per night, 21 s for 25. The rank of the peak nearest the Xuan et al. track at each night's
mean time:

| night | K = 8: rank, offset | K = 25: rank, offset | Δm to the best peak (untempered) |
|---|---|---|---|
| 2023-12-25 | 0, 0.32 mas | 0, 0.32 mas | 0 |
| 2023-12-29 | 1, 1.25 mas | 1, 1.25 mas | 171 |
| 2024-02-27 | none within 5 mas | 7, 2.54 mas | 103 |
| 2024-03-28 | none within 6 mas | 12, 1.29 mas | 92 |
| 2024-04-29 | 0, 0.18 mas | 0, 0.18 mas | 0 |
| 2024-12-18 | none within 7 mas | 5, 1.07 mas | 131 |
| 2025-02-11 | 0, 0.72 mas | 0, 0.72 mas | 0 |

The per-peak position errors are 0.01–0.06 mas, so offsets of 0.2–0.7 mas on the good nights reflect the
published orbit's own uncertainty propagated to these epochs (σ_P = 0.003 d over 28 cycles is about
0.3 mas), not a bad peak. On the three bad nights the true position is **not a peak at all** in the
static surface, even at K = 25. That is why S2 needs the truncation bound, and why a night's best peak
must never be imposed.

### 9.2 S2–S5 on Gl 229 (blind start, reference-informed 10–14 d window)

**What this tests.** These runs test the search (S2 to S4), not the final likelihood. The prototype
profiles one V² amplitude per night (clipped to 0.01–1.5), applies one f per night to both blocks, and
uses S1 catalogues with V² + CP and no amplitude term. §4 changes all three (review objection 3), so P1
must rerun this table. The 10–14 d window is reference-informed. §9.4 is the blind-window run.

Tempering f per night from §9.3: 0.04–0.23. Screen: K_p = 5, σ_g = 0.5 mas, window 10–14 d (see §9.4 for
the blind window), 400 distinct candidates within 25 nats of the best. Then S3 (10 fluxes per candidate)
and S4 (LM from the top 32 by S3 score).

| selection | screen: truth-like candidate | S3 rank of truth-like starts | S4 best mode | runner-up mode |
|---|---|---|---|---|
| all 7 nights | rank ~291, ΔS = −21.6 | 0, 1, 2, 3, 4 | P 12.1359 d, e 0.233, i 28.2°, a 7.30 mas, flux 0.471; track rms to Xuan 0.19 mas | N = 41, P 10.085 d: Δℓ = −133.6 |
| §5 (4 nights) | rank ~105, ΔS = −10.6 | 2, 8, 20, 31 | P 12.1395 d, e 0.220, i 30.7°, a 7.71 mas, flux 0.488; track rms 0.20 mas | N = 41, P 10.072 d: Δℓ = −28.7 |

Xuan et al. 2024: P 12.134 ± 0.003 d, e 0.234, i 31.4°, a 7.36 mas, flux 0.47–0.53. Both selections
find the published orbit as the top mode by a wide tempered margin, within that window (§9.4: the
blind window is still to run). Every start that reached it
converged to the same ℓ to 10⁻², in 5 LM iterations from the S3 starts. The gallery's present Gl 229 orbit
(i = 104.8°, P = 12.059 d, a = 12.5 mas) is not among the S4 modes within 75 nats.

Points that matter for the design:

- the screen alone would have picked a wrong alias (N = 41, at the window's edge), so S3 is essential;
- in the §5 selection the truth-like starts sit at S3 ranks up to 31 of 32, so **S4 must start from every
  S3 candidate, not a top few** (400 starts × 40 iterations is about 2 min on the laptop);
- the full Hessian is positive definite at both modes. H_GN overestimates ln det by 1.1 and 0.23.

### 9.3 Effective channels and the shape of the residuals (Gl 229)

At each night's best static peak, whitened residuals grouped into 59-channel spectra (72 V² and 48 CP
spectra on 2023-12-25, fewer on other nights):

| night | V² χ²/N | V² f (binning / ACF) | V² f, spectrum means removed | CP χ²/N | CP f | CP f, means removed |
|---|---|---|---|---|---|---|
| 2023-12-25 | 27.8 | 0.039 / 0.068 | 0.44 / 0.35 | 5.1 | 0.16 / 0.15 | 0.82 / 0.74 |
| 2023-12-29 | 17.4 | 0.035 / 0.027 | 0.41 / 0.46 | 5.7 | 0.061 / 0.062 | 0.20 / 0.19 |
| 2024-02-27 | 7.2 | 0.040 / 0.046 | 0.21 / 0.44 | 18.2 | 0.070 / 0.068 | 0.80 / 0.77 |
| 2024-03-28 | 6.1 | 0.037 / 0.028 | 0.15 / 0.14 | 2.3 | 0.69 / 0.60 | 0.70 / 0.63 |
| 2024-04-29 | 3.3 | 0.049 / 0.036 | 0.63 / 0.63 | 5.8 | 0.092 / 0.080 | 0.21 / 0.36 |
| 2024-12-18 | 36.9 | 0.036 / 0.026 | 0.35 / 0.33 | 3.6 | 0.068 / 0.055 | 0.60 / 0.53 |
| 2025-02-11 | 22.1 | 0.035 / 0.047 | 0.30 / 0.25 | 8.9 | 0.055 / 0.043 | 0.38 / 0.40 |

The binning ratio R(b) of the raw V² residuals rises linearly with b up to b = 32: a whole spectrum moves
together. At the blind 7-night orbit, the V² χ²/N splits into within-spectrum and spectrum-mean parts
of 5.4 + 16.1 (2023-12-25), 2.4 + 11.6, 4.3 + 5.0, 1.1 + 2.6, 1.4 + 2.0, 1.0 + 4.8 and 3.4 + 17.3. About 40% of
the spectrum-mean part is a fixed per-baseline offset for the night and 60% varies frame to frame. The
CP χ²/N splits 3.8 + 0.5 on 2023-12-25 and similarly elsewhere, so closure-phase offsets are a minor
term. This sets §4.3: per-spectrum V² gains, then f on what is left.

The fitted per-night V² amplitudes are 0.07–0.76: Gl 229 B's V² is very poorly calibrated, and any
stage that fits V² without its gain is fitting the calibration.

### 9.4 The blind period window (not run: the first OzSTAR job)

Every Gl 229 result above used the reference-informed 10–14 d window. A blind-window run was started
locally: P from 3 d (the §5.1 rule at ρ ≈ 8 mas) to 1660 d, e in steps of 0.1, 24 phases, K_p = 3, the
best 8 candidates per (band, e-bin), and LM starts in every band. That is 1241 periods × 10 × 24 =
297 840 grid points × 189 seeds = 5.6 × 10⁷ seeds. It was **stopped after 8 minutes**, because the shared
laptop was overloaded (load about 55 on 10 cores). Ben's rule is that runs of this size go to OzSTAR.

So the claim that S1–S4 recover Gl 229 blind **is not yet shown for the blind window**. It is the
acceptance test of the first OzSTAR job (PR V4). The job runs `exp2_screen.py` and `exp3_lm.py` from the
prototypes folder with `PLO=3 PHI=1660 ESTEP=0.1 NTAU=24 PERCELL=8 DSCREEN=1e9 PERBAND=2`. It passes if
the S4 top mode is the §9.2 mode (track within 0.5 mas). The expected cost is about 30 min on one node,
from the measured 3–6 × 10⁴ seeds/s. The full-resolution window (20 e × 59 phases × 3700 periods)
follows (§8.4).

### 9.5 ι Peg: what the gallery's positions show

The ι Peg OIFITS are on OzSTAR only and were not fetched, so S1 and S3–S4 could not run on them. The
gallery run (`~/data/oidb_o1_fits/f1185a8+3420ea1/fit_fac164e1….json`) saved each night's best position.
Against the Konacki et al. (2010) orbit (T in MJD):

| night | gallery best peak (mas), flux | Konacki position (mas) |
|---|---|---|
| 2018-10-22 | (−0.96, 10.00), 0.51 | (−0.91, 9.97) |
| 2018-10-23 | (−1.28, 6.41), 0.40 | (−1.24, 6.44) |
| 2018-11-21 | (−0.50, −0.00), **0.92** | (−0.46, 10.13) |
| 2019-07-31 | (1.30, −4.81), 0.35 | (1.24, −4.83) |
| 2019-08-06 | (−0.40, −0.30), **1.00**, covariance at its floor | (−1.03, −0.41) |

So three nights are right to 0.05 mas and two have the wrong peak: a near-equal-flux solution at 0.5 mas
on a night where the companion is at 10 mas (Anugu et al. measure 10.18 mas, PA 357°). The wrong P of
10.33 d follows from the wrong peaks. Running S2 on these five single peaks with a 3–60 d window (not blind: chosen knowing P ≈ 10 d)
returns a flat screen: every band within 2.4 nats, and nothing within 4 mas of Konacki's track. Four
positions (8 numbers for 7 orbital parameters) cannot rank aliases. S1 with K = 25 and the V² gain, and
then S3, are what fix this, and that run needs the data (§13).

### 9.6 Injection and recovery at the real Gl 229 (u, v, t)

The injection keeps the real noise: per night, data_new = data − model_night + model_injected, with the
night's fitted V² amplitude, and closure phases wrapped. This "residual transplant" keeps the real
per-spectrum offsets and chromatic structure. What model_night should be took three attempts:

1. **The blind 7-night orbit.** It misses each night's real position by a few tenths of a mas, so on
   high-SNR nights a ghost binary stayed in the data.
2. **Each night's best static peak.** This is the wrong peak on 4 of 7 nights (§9.1), so the real Gl 229 B
   stayed in every injection as structured noise. The adversarial review caught it (objection 1). The
   runs made with this injector, including an early finding that the V²+CP catalogue misses the
   companion on 6 of 7 nights, are withdrawn.
3. **A static binary refined locally from the blind top mode's predicted position** (L-BFGS on the
   amplitude-profiled, scale-marginalised likelihood). This is what v2 specifies (§10.2). The refinement
   moved 0.08–0.31 mas from the prediction on 4 nights, and 0.95, 1.99 and 0.95 mas on 2023-12-29,
   2024-02-27 and 2024-03-28. Those three shifts are the residual ghost that no static subtraction
   can avoid on nights the orbit does not fit.

Results with injector 3. The window is the reference-informed 10–14 d of §9.2, which contains both
injected periods (§9.4 is the blind window). S1 used closure phases only (K = 25); S2 used K_p = 5,
σ_g = 0.5 mas and a cap of 3000; S4 had 256 starts × 40 iterations. The laptop was shared with other
jobs, so the timings are about twice those of §9.2.

| | injection A | injection B | null (no companion) |
|---|---|---|---|
| injected | P 11.37 d, e 0.45, i 62°, a 6.5 mas, flux 0.45 | P 13.2 d, e 0.10, i 140°, a 4.0 mas, flux 0.30 | nothing |
| S1: rank of the truth per night | 0, 2, 9, 0, 21, 0, 0 | 0, 2, 10, 0, 3, 0, 0 | real Gl 229 B: nearest peak 2.1–8.4 mas away on every night (§9.6 note) |
| S2: rank of the truth-like candidate | 0 (of 3000), band N = 36 | 0 (of 1513), N = 31 | flat: every band within 2.0 nats of the best; best N = 41, flux-free screen (no flux in S2) |
| S4 top mode | P 11.3701, e 0.450, i 63.9°, a 6.51, flux 0.412; track rms 0.21 mas | P 13.2023, e 0.082, i 136.2°, a 4.11, flux 0.283; track rms 0.25 mas | P 12.025 d, e 0.14, a 21.4 mas, **flux 0.076**; next modes at Δℓ = −2.9 (same band), −19.6 (N = 40); all with flux 0.04–0.08 and a = 13–40 mas; none within 9 mas of the real orbit |
| starts reaching the top mode | all of the top 12 (and 31+ in total), from S3 ranks up to 247 | all of the top 12 | – |
| full Hessian at the mode | positive definite; ln det 76.23 (GN 76.74) | positive definite; ln det 69.10 (GN 68.97) | – |
| wall time S1 / S2 / S3 / S4 | 49 s / 642 s / 91 s / 201 s | 65 s / 1060 s / 58 s / 290 s | – |

Both injections are recovered blind in their band, with the track within 0.25 mas, every orbital element
inside the scatter expected from 7 nights, and every converged start agreeing. The null injection
checks that the subtraction removed the real companion: in its S1 catalogue, the real Gl 229 B's
positions are not among the peaks within 2 mas on any night (best: rank 10 at 2.1 mas on 2024-02-27).
In S2–S4 the null finds no trace of the real orbit: no S4 mode's track is within 1.5 mas of it, and
the nearest is 9.4 mas off. But **the solver still returns a "best orbit"**: a faint (flux 0.04–0.08),
wide (13–40 mas) companion fitted to the residual systematics, with neighbouring modes within 3 nats.
The screen being flat across all 12 bands is the warning sign. Without one, the result would be reported
as a detection. So v2 adds a third blind diagnostic (§7): the tempered evidence of the top mode against
a **single star** (no companion), with the same nuisance model. Its distribution under null injections
(§10.2) sets the detection threshold, and a system below it is reported as "no significant orbit".

Two injections are a demonstration, not a calibration: §10.2's 300 are what decide.

## 10. Validation

### 10.1 Blind recovery of published orbits

Run S0–S6 with the §5.1 window on Gl 229 (both selections), ι Peg, HR 6819 and the Track B systems of
`docs/orbits/aliases.md`. The published orbit is read only by S8. A system **passes** when:

1. the published orbit's band N is the top band, with p ≥ 0.9 after calibration (§10.2), and
2. by statistic A (`orbit_comparison.md` §2), the top mode's track agrees with the published one:
   joint d² p > 0.01 where the reference has samples or a covariance; otherwise (unscored), the maximum
   track separation over one period is < 5% of a_ref, and
3. the run is reproducible: two runs give the same candidate-list hash and modes equal to 10⁻⁸ in x.

**The validation set, by failure mode.** These are drawn from Ben's ESO-archive binaries sheet: 93 clean
binaries, 69 with visual orbits. Nights and span come from its observation-set table. P, e and a come from
the local orbit files (`~/data/eso_binaries/trackb/*/system.json` and `catalogue/*.json`). It is a
proposal; Ben has not chosen. The full triage is in `design/eso_binaries_triage.md` (onboarding branch).

| failure mode | systems (P d, e, a mas; nights, instrument; span/P) | data on disk |
|---|---|---|
| sub-resolution (separation < λ/2B) | AK For (3.98, 0.000, 1.7; 22 GRAVITY; 181), LL Aqr (20.2, 0.32, 1.4; 29 GRAVITY; 35), HD 224974 (10.7, 0.34, 2.1; 13 GRAVITY; 27), V963 Cen (15.3, 0.42, 1.4; 16 GRAVITY + 7 PIONIER; 71), AL Dor (14.9, 0.20, 2.3; 30 PIONIER; 50); HR 6819 (≈40, 0.03, ≈1.3; reported only, Be disk) | yes, except HR 6819 (OzSTAR) |
| single-night aliases, annual gaps, many cycles between epochs | Gl 229 Ba–Bb (12.1, 0.23, 7.3; 7 GRAVITY; 34), HD 224974 (above), AK For (span/P 181), NN Del (99.3, 0.52, 3.5; 6 PIONIER; 4.1), ψ Cen (38.8, 0.55, 5.1; 8 PIONIER; 8.6) | yes |
| short arcs (span < 1 P) | κ Vel (116.8, 0.18, 7.3; 9 GRAVITY in 61 d; 0.52), ζ Boo (125 yr, 0.98, 809; 4 GRAVITY; 0.009), η Oph (87.8 yr, 0.93, 1204; 6 GRAVITY; 0.013) | yes |
| high eccentricity, periastron sparsely sampled | HD 210763 (42.4, 0.62, 3.8; 12 GRAVITY; 8.6), HD 188088 (46.8, 0.69, 21.6; 23 GRAVITY; 15), ζ Boo and η Oph (e 0.98, 0.93); HD 168137 (e 0.90) and 9 Sgr (e 0.65) only once their data are staged | yes; HD 168137 and 9 Sgr have 1–2 epochs on disk |
| resolved primaries | ο Leo (14.5, 0.001, 4.5; primary 1.30 mas; 11 GRAVITY; 26), η Oph (both components 0.6–0.8 mas, fitted), TZ For (75.7, 0, 3.0; giants 0.41/0.20 mas; 16 PIONIER) | yes |
| triples | δ Cir (1603, 0.53, 12.4; inner eclipsing binary unresolved; 12 PIONIER + 3 GRAVITY; 2.7). HD 150136 when staged | δ Cir yes; HD 150136 no |
| near-equal flux (180° ambiguity) | AL Dor (flux ratio ≈ 1.0), V963 Cen (0.96), HD 188088 (0.90), HD 224974 (0.89), ζ Boo (0.89) | yes |
| e ≈ 0 | AK For, ο Leo, TZ For, HR 6819 | yes |
| genuine prediction (answer unseen) | HD 188088, HD 70937, HD 210763, HD 41255, ο Leo (1–6 later public GRAVITY epochs not in their papers, e.g. 110.23QY.001); LL Aqr (about 22 later epochs, 109.22XY.002). Fit blind on the paper's epochs, freeze the mode set, then predict the new epochs | LL Aqr and AK For extras on disk; others to stage |

Gaps:
- Resolved primaries above 1 mas: ο Leo is the only good case. The giants and Cepheids in the sheet's
  "other" tab have 1–2 epochs and no orbit.
- Triples: δ Cir is the only one on disk.
- ι Peg (CHARA/MIRC-X) is not in the ESO pool. It stays in the set as the CHARA case, run on OzSTAR.

A system where the published orbit is in the mode set but not on top, or whose winning mode disagrees,
is a **finding** for the ledger: either the data favour another orbit, or the solver is wrong, and the
injection results decide which. HR 6819 is reported, not scored (Be disk). Its a ≈ 1.3 mas is below the
resolution λ/B ≈ 3 mas, so S1's catalogues are unimodal and degenerate in flux and separation; it tests
the sub-resolution regime.

### 10.2 Injection and recovery

The injector is the residual transplant of §9.6. Each night subtracts a static binary refined locally
from the blind top mode's predicted position, not that night's best peak. The best peak is the wrong
peak on 4 of 7 Gl 229 nights, and subtracting it leaves the real companion in every injection (review
objection 1). Three controls go with every system:
- a **null injection** (no companion): the solver must not recover the system's real orbit from it, and
  the size of the ghost it leaves is reported;
- injected orbits are drawn away from the real track. Recovery is reported as a function of the
  injected track's distance from it;
- the subtraction's per-night shift from the mode's prediction is reported. On Gl 229 it is 1–2 mas on
  2023-12-29, 2024-02-27 and 2024-03-28: the residual ghost that no subtraction can avoid.

Draws, per system and selection:
- **300 regime-matched injections**: P log-uniform over the §5.1 window, e uniform on [0, 0.95],
  isotropic orientation. a and flux are drawn so that the per-night detection SNR spans the range of
  the real system's S1 peaks: from half the weakest night's to twice the strongest's. Draws over the
  whole window are mostly unlike the system being solved (objection 9);
- **100 with S6**, for coverage.

Pass/fail (objection 9 corrected the draft's statistics):
- **completeness**: the injected orbit's band is in the S4 mode set in every one of the 300. Zero misses
  in 300 bounds the miss rate below 1.0% at 95% confidence; 200 could bound it only below 1.5%. Every
  miss is listed by stage: S1 (not a peak), S2 (cut), S3 (ranked out) or S4 (did not converge). Low-SNR
  injections where at most one night shows the truth among its top peaks are reported separately, as a
  measured completeness rather than a pass/fail;
- **calibration of mode probabilities: a reliability diagram over all modes.** Pool every reported mode
  of every injection, bin by its p, and compare each bin's mean p with the fraction of its modes that
  contain the truth. Each bin with ≥ 30 modes must lie within its binomial 95% interval. The draft
  binned only the truth's mode, which is not a calibration test. The 95% credible mode set must contain
  the truth in ≥ 92% of the 300 (95% − 2σ, with σ = 1.26%);
- **temperature**: if the calibration fails, fit one global multiplier c on f (f_eff = c f) to the
  injections by maximising the log score of the truth's mode probability, re-check, and report c. c is
  a property of the instrument and pipeline, and is fixed per instrument before the real-data runs.
  The f × 0.5 and f × 2 runs (§4.2) show how much rides on it;
- **blind diagnostics**: the distributions of the leave-one-night-out residual, the orbit-against-
  free-positions evidence and the orbit-against-single-star evidence (§7), for correct top modes, for
  wrong ones, and for **30 null injections** (no companion). These turn the diagnostics into tail
  probabilities and set the detection threshold;
- **parameter coverage within the true mode** (the 100 with S6): SBC rank statistics of P, e, i, a and
  flux, uniform at KS p > 0.01. At 100 draws this detects a coverage error of about 15% in width;
- **stress cases**, 30 injections each, each with zero misses:
  - an annual gap (the real Gl 229 spacing already has one);
  - single-night aliases: an injection in which one night is replaced by that night's real data, i.e. a
    bad night;
  - e < 0.02;
  - i within 10° of face-on;
  - near-equal flux (sign-free);
  - a < λ/2B;
  - a = 20–40 mas with e = 0.8–0.95 (the grid floor);
  - a period within 5% of a window edge.
  A stress case where the truth falls outside the 95% mode set is reported by name.

### 10.3 Software tests

Unit tests in virgil for every new piece (§11), including a brute-force check of S2: on small problems
(4 nights, 3 peaks each) enumerate all 4⁴ assignments × the grid, and compare with the pair-seeded
result.

## 11. Implementation plan

PR-sized pieces, in order. Each is one Sonnet worker session (`pr-worker`), with tests in the same PR.
virgil PRs land first, then virgil-validation pins the merged commit.

| # | repo | files | what | tests |
|---|---|---|---|---|
| P0 | virgil | `oidata.py` | Vectorised `_cvis_in_time` (in progress elsewhere) | existing, plus the review's `check_fast.py` agreement to 10⁻¹⁴ |
| P1 | virgil | `epochs.py` (`_Surface`, `marginal_loglike`), new `nuisance.py` | V² gains per (night, baseline), with per spectrum as an option; conjugate prior so the gain and Jeffreys-scale marginals compose (§4.3); χ² floor; per-block `dof` arrays everywhere; remove `dof=1.0` from `aliases._loglike` | the closed form against 2-D quadrature over (g, s); reduces to `vis_amp` as τ → 0; gradients against finite differences; a sub-resolution injection keeps its V² information with the per-baseline gain |
| P2 | virgil | new `effective_dof.py` | Spectrum grouping (frame × baseline/triangle, wavelength order); binning (plateau over b ≤ L/4) and ACF estimators; floor; pooled per (instrument, observable), frozen before S2 | synthetic AR(1) noise of known τ recovered to 10%; white noise gives f within 5% of 1; the floor; a wrong-peak night does not raise the pooled f |
| P3 | virgil | `epochs.py` (`epoch_positions`, `EpochPeaks`) | Catalogues with the nuisance model, per-peak 3 × 3 covariances, tempered heights, the heuristic bound, peak separations, the flux-maximised gridded surface; field from the instrument; L-BFGS-B refinement replaced by the P5 LM | synthetic binary with known fringe aliases: truth in the catalogue; covariances against finite differences; a byte-identical rerun |
| P4 | virgil | new `orbit_screen.py` (reuses `orbits._unit_orbit`, `period_grid`) | The S2 screen: cond-ranked pair seeds (plus triplets), hard EM, scoring on the interpolated surfaces, per-seed grid floor, per-(band, e-bin) selection, vectorised de-duplication, sign-free peaks, a frequency-range split for Slurm arrays, `shard_map` within a node | brute-force enumeration on small problems; synthetic positions with wrong and missing nights; a = 20–40 mas, e = 0.8–0.95; determinism hash; split and unsplit runs give the same candidates |
| P5 | virgil | `fitting.py` (`fit_multistart` on top of the GN layer's IRLS-LM step), new `orbit_lm.py` (Thiele–Innes coordinates, mode merging, GN and full Hessian per mode) | §6.4: vmapped multistart LM, convergence flags, mode merging. **Depends on the GN layer** (`gauss_newton_plan.md`), which delivers `residual_blocks`, the IRLS-LM step, V19 mass and the `aliases._optimize` rewrite | `fit_multistart` equals the layer's single-start LM from each start; a non-converging start is reported; modes merge by sky track |
| P6 | virgil | new `orbit_solver.py` | Pipeline API `solve_orbit(data, window, ...)` for S0–S5 with checkpoints; the mode table; Laplace and IS evidences; "no mode found" bands | end-to-end on a synthetic 5-night dataset with a known alias; checkpoint and resume |
| P7 | virgil | `orbit_solver.py`, `likelihood.py`, `pipeline/binary.py` (dense GN mass instead of `dense_mass=False`) | S6: NUTS per mode (`gauss_newton_mass` metric, seeded chains), basin check, bridge-sampling ln Z, mixture | two-mode synthetic posterior: weights against the analytic answer within 0.1; a mode merge is detected; a fixed GN metric is not re-adapted; the divergence-triggered fallback to an adapted dense mass fires on a synthetic e = 0.03 orbit, or the √e coordinates avoid it |
| V1 | virgil-validation | new `scripts/solve_orbits.py`; `design/orbit_solver_v2.yml` (criteria, preregistered) | Runs P6/P7 on Gl 229 (both selections), Track B, ι Peg, HR 6819 with §5.1 windows; no `init=`, no `published=` | `--dry-run` prints the windows; a test asserts that no reference value reaches the solver (the loaders hand the reference only to S8) |
| V2 | virgil-validation | new `scripts/inject_orbits.py`, `src/crosscheck/` (truth tracks) | The residual-transplant injector, anchored on the blind top mode (§10.2), the null injection, regime-matched draws, the reliability diagram, the §10.2 statistics, the blind diagnostics' distributions | an injection of zero companion at a synthetic dataset leaves it unchanged to machine precision; on a synthetic dataset with a known companion, the anchored subtraction removes it to < 1% of its V² signal |
| V3 | virgil-validation | `scripts/oidb_compare.py`, `scripts/build_aliases_page.py`, docs pages | Reports: mode tables, both Gl 229 selections, calibration plots, findings | the page builder runs on fixture outputs |
| V4 | ozstar_scripts | `scripts/orbit_solver/` | Node job for S1–S5 (`shard_map`, CPU), Slurm array for S6 and S7, combine job; the first run measures scaling | dry-run on the laptop with a 2-device host mesh; **first job: the §9.4 blind-window Gl 229 run**, acceptance = the §9.2 mode on top |

`reanalyse_orbits.py` and the orbit recipes in `oidb_fit.py` are kept until V1 reproduces their systems,
and are then retired. Their single-epoch (non-orbit) recipes stay.

## 12. Review and responses

An adversarial review (a fresh Opus agent, 2026-10-14) read the draft, the prototypes and virgil. It
raised 12 objections, two of them blockers. It ran only one short check of its own: the period-grid
size, the Thiele–Innes Jacobian (confirmed) and the grid floor. Each objection, and what changed:

| # | severity | objection | response |
|---|---|---|---|
| 1 | blocker | The injector subtracted each night's best peak, which is the wrong peak on 4 of 7 Gl 229 nights. So every injection still contained the real Gl 229 B as structured noise. | **Accepted, and the prototype was rerun.** The subtraction is now anchored on the blind top mode: a local static refinement from its predicted position on each night. A null injection is added, and recovery is reported against distance from the real track (§9.6, §10.2). |
| 2 | blocker | The candidate cap (400) was the cut that decided, and was tuned with the answer known. The truth survived by 1.7 nats, in a window with 34× fewer periods than the blind one. Δ_S was in tempered nats, which stretch when f changes. | **Accepted.** No global cap: the best m = 8 per (band, e-bin), Δ_S within cells in untempered units, and S4 starts in every cell (§5.3, §5.4). The cost is minutes for S3 (§8.4). |
| 3 | major | §9.2 does not test the §4 likelihood. The per-spectrum gain absorbs the V² signal of close companions. The gain formula had rᵀWr for yᵀWy, a log/linear prior mismatch, and does not compose with the Jeffreys scale. | **Accepted.** The default gain is per (night, baseline); per spectrum is a variant that must pass a sub-resolution injection test. The prior is conjugate, so both marginals have a closed form together, and the formula is corrected (§4.3). §9.2 is labelled as a test of the search, not of the final likelihood, and P1 must rerun it. |
| 4 | major | σ_g = 0.5 mas is unjustified: the grid miss is 0.35–1.27 mas at a = 7.3 mas and scales with a. | **Accepted.** σ_g is now computed per seed from the grid cell and β; n_τ grows with e; a stress case covers a = 20–40 mas with e = 0.8–0.95 (§5.3, §10.2). |
| 5 | major | The completeness argument and the truncation bound are not true as stated: error amplification by cond(M), a gridded and finite S1, the Gaussian basins, local hard EM, and "unseeded" being undetectable. | **Accepted.** The argument is restated as what seeding does and does not guarantee. S2's score is read from the interpolated S1 surfaces. Pairs are ranked by cond(M), with triplets for ill-conditioned pairs. The bound is called a heuristic. Completeness is measured by injection, stage by stage, including low-SNR cases (§5.2, §5.3, §10.2). |
| 6 | major | A wrong orbit can come out with p ≈ 1, because p is normalised over the modes found. | **Accepted.** Two blind diagnostics are added: a leave-one-night-out prediction, and the orbit against free per-night positions. A third, the orbit against a single star, comes from the null injection (§9.6). All three are calibrated by injection, including null injections (§7, §10.2). |
| 7 | major | f per night puts the most weight on a bad night (2024-03-28 got 0.23). max_b R(b) is biased. Re-estimating f per mode biases evidences. | **Accepted.** f is pooled per (instrument, observable), read from the plateau of R(b), frozen before S2, and run again at × 0.5 and × 2 (§4.2). |
| 8 | major | The Laplace evidence fails at e → 0 (flat τ, a logistic ridge). The Campbell → TI map is 2:1. Tempering inflates the untempered Occam terms. | **Accepted.** Laplace is computed in coordinates regular at e = 0, with the factor 2, and importance sampling near face-on. Band probabilities are also reported under a second period prior (§7.1). |
| 9 | major | The calibration test was ill-posed (it binned only the truth's mode); 200 injections cannot show 99% completeness; the 2σ arithmetic was wrong; 20 SBC runs have no power; the draws were unlike the systems. | **Accepted.** §10.2 now uses a reliability diagram over all modes, 300 regime-matched injections with zero misses, 100 SBC runs, a 92% mode-set threshold, and 30 per stress case. |
| 10 | major | The §5.1 window was inconsistent with S1: P_max implies separations outside the ±18 mas field, and 10 × the night's length allows 4.6 mas of motion within a night. | **Accepted.** The field comes from the instrument, P_min from the motion allowed within a night, applied per separation, and tracks that leave the field score the bound (§5.1). Short-period systems such as AK For need a within-night recipe (§13). |
| 11 | major | Cost: the blind window is about 3 h on the laptop for Gl 229 and tens of hours for systems with many nights; S7 costs thousands of node-hours; the de-duplication is O(n²). | **Partly accepted.** At most 300 cond-ranked seeds per point bounds the N² growth. S2 is split by Slurm array over frequency ranges by default. The de-duplication is vectorised. A budget table is in §8.4. The S7 budget remains large, and its phasing is a question for Ben (§13). |
| 12 | minor | The diagonal GN fallback discards correlations; Φ → −∞ for exactly fittable blocks. K = 25 (chosen after K = 8 failed), the §5 selection and the 3–60 d ι Peg window are reference-informed. The e grid stops at 0.9. The `inject.py` docstring is stale. | **Accepted.** The GN layer always adds the prior and transform curvature as a diagonal, with no fallback rule, and χ² is floored (§4.3, §6.4). K is set by injection. The §5 selection is labelled reference-informed (§2) and never used for tuning. The ι Peg window caveat is noted in §9.5. The e grid goes to 0.95. The docstring is fixed. |

The review found the §6.4 design sound. The IRLS gradient is exact, H_GN drops only a negative
semidefinite rank-one term (so its steps are conservative), and the block-weighted GN matrix is the
right tempered Fisher matrix for the NUTS metric.

## 13. Open questions for Ben

1. **The S7 budget.** A full calibration (640 injections per system and selection) is about 200
   node-hours for Gl 229, and several thousand across §10.1. Proposal: run the full set on Gl 229 and on
   three sub-resolution systems (HD 224974, LL Aqr, V963 Cen) to fix c per instrument, then 100
   injections plus the stress cases for the rest. Is that acceptable, or should calibration be per
   system throughout?
2. **The V² gain model.** The default is one gain per (night, baseline), which keeps the V² signal of
   close companions. Per-spectrum gains fit Gl 229's frame-to-frame offsets better but would blind the
   sub-resolution systems. Is a per-instrument choice by injection test (§4.3) the right rule, or should
   V² be dropped from S1 entirely, with closure phases only, for sub-resolution systems?
3. **Short-period systems.** For AK For (P = 3.98 d), the static per-night snapshot is not valid, so S1
   needs a within-night orbit recipe. Is AK For in scope for v2, or deferred to a v2.1?
4. **What counts as "blind" for the period window.** §5.1 derives P_min from the motion allowed within a
   night and P_max from 4 × span. Is a physical prior on mass (hence a(P)) acceptable, given the
   distance? It would cut the window considerably, but it uses a parallax and a mass guess, which some
   would call outside information.
5. **The §5 selection.** It is reference-informed (2024-12-18 is held out because it disagrees). It is
   reported and labelled as such. Should it also be dropped from the pass/fail criteria, leaving only
   the 7-night run to be scored?
6. **Genuine predictions.** Fitting on the paper epochs and predicting the later public epochs (HD 188088,
   HD 70937, HD 210763, HD 41255, ο Leo, LL Aqr) is the only test whose answer nobody has seen. Should
   those predictions be registered (frozen and committed) before the new epochs are reduced?
7. **ι Peg and HR 6819 data.** Their OIFITS are only on OzSTAR. Fetching them to the laptop would let V1's
   prototype run locally; that is a download needing your approval.

### Known risks in the comparison step (S8)

These affect how a blind result is judged against a paper, not the fit itself:

- **Gl 229.** The node Ω (the 180° ambiguity of Ω from astrometry alone) and the 2024-12-18 night are
  open (`orbit_comparison.md` §5). The §5 selection holds 2024-12-18 out until it is ruled on.
- **ι Peg.** It is unclear which component ω refers to, whether T_peri is MJD or RJD, and whether Ω is
  epoch-of-date or J2000 (`scripts/orbit_gallery.py:97-100`). A blind fit can be right and still look
  half a period or a degree out of phase.
- **HR 6819.** The circular-orbit T₀ in the paper's Table 3 looks half a period away from Table 2's
  (`orbit_gallery.py:89`).

S8 must report the convention choices it made, and each of these is a candidate "finding" rather than a
solver failure.
