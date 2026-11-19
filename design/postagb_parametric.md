# Post-AGB parametric disc models: pre-registration

Status: committed before any real-data fit of this campaign (WP4). Sections 1 to 7 and 9 below are copied verbatim from the
final executable plan (18 Nov 2026, revised after review; `reports/postagb-parametric-plan.md` of the virgil-paper working
copy, not in git). The numbering is the plan's. Conventions, models, priors, noise arms, evidence rules, degeneracy checks,
robustness checks and decision rules are fixed here; any deviation in the production run is listed under "deviations" in
the results report with its reason (risk R9, stop rule S5). Implementation notes that are not part of the pre-registration
are in the last section.

## 1. Conventions

* **Scene**: `P.sparco_scene(cfg | dict(bg_index_fixed=0.0), disc, f_env=…)`: star (UD and index fixed, flux 1), grey point
  secondary where the paper has one (IW Car, HR 4049, AI Sco, EN TrA, HD 95767), `env` = the parametric disc with
  `PowerLaw(f_env, d_env, 1.65 µm)`, `bg` = `Resolved` with index fixed at 0. All fluxes are ratios to the star at 1.65 µm.
  This is the gp-polar scene except that gp-polar frees the bg index (it sat on ±6 in 5 of 6 gp-polar fits).
* **Disc**: `MultipoleDisk(GaussianRing(radius, sigma), harmonics=Harmonics(re=…, im=…, form="positive"), inc, pa, dra, ddec,
  n_r=64, flux=PowerLaw(…))`. Radial profile exp[−(r − r₀)²/2σ²]; azimuthal factor non-negative for every w (fact table).
  Report the equivalent linear (c_m, s_m) from `virgil.harmonics`, and check them once against a numerical Fourier
  transform of the rendered azimuthal factor on 720 azimuths. c₁ is along the major axis, s₁ along the minor axis (towards
  PA + 90°). First-order amplitude a₁ = √(c₁² + s₁²) is the number to put beside De Prins's a_mod and Lazareff's c₁, s₁.
* **Literature forms (context only; nothing is selected with them)**: Lazareff et al. (2017) fit a thin ring convolved with a
  kernel, with azimuthal terms c_j cos jφ + s_j sin jφ (j ≤ 2) and a halo; Kluska et al. (2019) applied that family to
  PIONIER post-AGB snapshots; De Prins et al. (2026) fit a PMOIRED rim with one harmonic (a_mod, ψ_mod). These descriptions
  are from memory apart from the bibcodes (review table, row 14); the Lazareff-form rung was cut, so no number in this
  campaign depends on them. Table B.1 of the first draft is `postagb_paper.PMOIRED`, whose docstring calls it Table 3.
* **Angles**: PA of the major axis, prior `Uniform(pa₀ − 90, pa₀ + 90)` with pa₀ the start, reported mod 180. PA + 180° with
  the odd orders negated is the same model, so one 180° window holds each solution once and the prior density (1/180 per
  degree) does not depend on pa₀. A fit ending within 1.8° of a window edge is refitted once with the window re-centred on
  it. Report the on-sky PA (0–360°) of the brightest point of the ring, measured on the rendered model.
* **Three χ² per fit, always**: `raw` = `P.chi2r(model, data_raw)` (quoted errors, independent, no gains; N = its row count,
  the same count as gp-polar's `n_points`); `raw_v2`, `raw_cp` from `P.residual_stats`; `own` = χ²/pt under the arm's own
  likelihood (`result.info["chi2_red"]`). Nothing is rescaled or tuned towards the published images.

## 2. Models

k = free parameters with a secondary (3 fewer without).

| id | disc | free | k | tier |
|---|---|---|---|---|
| M0 | centred elliptical Gaussian: `GaussianRing` with `radius` fixed at 0.01 mas, no harmonics | σ, inc, PA, f_env, d_env, f_bg, sec (dra, ddec, f) | 9 | P1 |
| M1 | Gaussian ring | + r₀ | 10 | P1 |
| M2 | + m = 1 (`degree=1`) | + Re w₁, Im w₁ | 12 | P1 |
| M2o | + ring-centre offset | + env.dra, env.ddec | 14 | P1 |
| M3 | M2 or M2o (whichever is selected) + m = 2 (`degree=2`) | + Re w₂, Im w₂ | +2 | P2 |
| M5 | selected + outer ring, `RadialSum`, shared inc/PA/harmonics | r_out, σ_out, w_out | +3 | P3, IW Car and HR 4049 only |

Arms at the selected model (P2): `nosec` (no secondary) and `secfree` (secondary position prior Normal(PMOIRED, 1 mas)) for
the five targets with a secondary; `bgfree` (bg index Uniform(−6, 6), as gp-polar) for all.

## 3. Priors (fixed now; not changed after any fit)

| path | prior |
|---|---|
| `env.inc` | `virgil.priors.IsotropicInclination(0, 85)` |
| `env.pa` | `Uniform(pa₀ − 90, pa₀ + 90)` |
| `env.profile.radius` | `LogUniform(0.6, 25)` mas, all targets (0.6 = half the λ/2B beam; 25 = HR 4049's rim radius × 3) |
| `env.profile.sigma` | `LogUniform(0.1, 10)` mas |
| `env.harmonics.re`, `.im` | `Normal(0, 1)` each order |
| `env.dra`, `env.ddec` (M2o) | `Normal(0, 1)` mas |
| `env.flux.ratio`, `env.flux.index`, `bg.flux.ratio`, `sec.*` | exactly `P.scene_priors(cfg, scene)` |
| M5 | r_out `LogUniform(1.5 × r₀ of the selected fit, 25)`, σ_out `LogUniform(0.3, 10)`, w_out `LogUniform(1e-3, 10)`, ring weight 1 |

**Starts**, for every (model, arm): inc ∈ {20, 45, 65}° × PA ∈ {0, 60, 120}° × r₀ ∈ {0.7, 1.4} × (image θ_rim / 2) = 18,
σ = 0.3 r₀, w = 0.1 + 0.1i, f_env = (1 − f_prim − f_sec)/f_prim and d_env = SIND0 from the header, f_bg = 0.05; plus three
named starts (De Prins image; PMOIRED where it exists; the stored `scale.json` geometry) = 21. IW Car has no PMOIRED
entry (20 starts). M0 uses the nine (inc, PA) pairs and the named geometries. M2o and M3 also start from the best two modes
of their parent. Keep every start's result. Two results are the same mode when |Δinc| < 5°, |ΔPA mod 180| < 10° and
|Δ ln r₀| < 0.1.

**Optimiser**: `fit(scene, priors, data, max_steps=2000, time_limit=60)`. Not converged → once more with
`method="lbfgsb"`. Still not converged → recorded, not eligible. No refit from an optimum that has `at_bound` entries.
Build all starts of one (model, arm) with the same model structure so the compiled objective is reused.

## 4. Noise arms (fixed; no noise is fitted in this campaign)

| arm | data object | purpose |
|---|---|---|
| N0 | as read | raw fit; ΔBIC beside De Prins's −100 threshold |
| N1 | `data.with_error_scale(s)`, s = `error_scale` of the stored gp-polar `image.json` (fact table) | same data object as gp-polar `--noise current` |
| N2 (**selection arm**) | N1 plus one gain mode per (night, baseline) on log\|V\| of width 0.03 and one offset per (night, triangle) of 2° | same as gp-polar `--noise nightw`; the likelihood recommended for the paper. Known to count the night scatter twice (s was derived without night terms), so it under-selects |

The data are loaded with `virgil_vlti.to_oidata` and the night modes come from `virgil_vlti.with_night_terms`, through
`scripts/postagb_noise.py`, which the gp-polar script uses too (one definition of the night: it starts at 12 UT, which gives the same night
as the earlier MJD 0.65 split for every INSPIRING file). N2 keeps the pre-registered global widths (`NIGHTW`); the per-target night widths
of the gp-polar `night` arm exist as `noise_arm(..., night=night_widths(cfg))` and are not used in the pre-registered ladder. A test
(`tests/test_postagb_vlti.py`) checks the shared construction against the earlier local one on AI Sco. ln Z is compared only within one arm.

## 5. Evidence, checks and selection rules (fixed now)

Per fit, with θ the k free parameters at the optimum, r the whitened residuals and J their Jacobian
(`virgil.imaging._residual_jacobian(model, data, paths, on_device=True)`), L = `_log_norm(model, data)`:

* `lnZ_lap` = −r·r/2 − L + Σ ln p(θ_e) + (k/2) ln 2π − ½ ln det(JᵀJ + Λ), Λ = diag(−∂² ln p_e/∂θ_e²) by
  `jax.grad(jax.grad(prior.log_prob))` (Normal and TruncatedNormal 1/σ²; Uniform 0; LogUniform −1/θ²; sin i its own).
* `lnZ_scene_rule`: the same with Λ = 1/σ² for `Normal` only (the rule of `scene_evidence`). Used only against gp-polar.
* `lnZ_is`: importance sampling in the same coordinates. Proposal: multivariate Student-t, 5 degrees of freedom, centred
  on θ, scale matrix 1.5² (JᵀJ + Λ)⁻¹, 4000 draws, seed fixed. Target: `log_density` of `numpyro_model(model, priors, data)`
  (jit + vmap, chunks of 100) minus its constant, the constant being `log_density(θ)` − (−r·r/2 − L + Σ ln p) at the
  optimum; non-finite values count as −∞. Record `lnZ_is`, the effective sample size `ess` and the fraction of draws outside
  the prior support. Run it for the best eligible mode of every model under N1 and N2 (not N0, not every start).
* `lnZ_use` = `lnZ_is` if `ess` ≥ 200, else `lnZ_lap` with flag `evidence_unverified`. Flag `laplace_poor` if
  `ess` ≥ 200 and |`lnZ_is` − `lnZ_lap`| > 1.
* BIC = N × own χ²/pt + 2 L + k ln N (N = `chi2r` row count).
* The two private imports live in one function with a clear error message. No other virgil private name is used.

**Eligibility of a fit**: E1 converged; E2 `info["at_bound"]` empty and no parameter within 1 % of a prior bound (PA
excepted, see §1; r₀ on its floor means "this is M0"); E3 det(JᵀJ + Λ) > 0, all Laplace sds finite, `min_intensity() ≥ 0`.
Further flags, not exclusions: `full_modulation` (a₁ > 0.95: the Normal prior on w, not the data, sets |w|);
`pa_unconstrained` (Laplace sd of PA > 30°); `multimodal` (a distinct eligible mode within 5 of the best `lnZ_lap`; both are
listed, and `lnZ_modes` = log-sum-exp over distinct eligible modes is recorded but not used for selection).

**A rung is accepted over its parent** under N2 when Δ`lnZ_use` ≥ 5 **and** ΔBIC ≤ −10, both models eligible. If either has
`evidence_unverified`, the same thresholds apply to `lnZ_lap` and the pair is flagged. Order: M1 over M0, M2 over M1, M2o
over M2, then (P2) M3 over the selected, (P3) M5 over the selected. **Selected** = the highest accepted rung of the chain;
a rung whose parent was not accepted is still tested against the last accepted one. The same table is printed under N1
and N0 (N0 with BIC only and the −100 line). `arm_dependent` is flagged when the N1 selection differs from N2's or their
geometries differ by more than 15° in inc or 25° in PA.

**"Is it a ring"**: M1 or higher selected, and hole contrast 1 − I(0)/I(r₀) ≥ 0.5 at the selected model (r₀ > 1.18 σ).

**Parametric against gp-polar (Q3)**, six targets, per arm N1 and N2: Δ = stored `log_evidence_scene` −
`lnZ_scene_rule`(selected parametric, same arm), quoted only if the stored `error_scale` equals s to 1e-3 and `n_points`
equals N; otherwise print the mismatch. Beside it: own χ²/pt of both (stored `chi2_per_point` is gp-polar's own) and the
parametric raw χ²/pt. The gp-polar number is the maximum over a hyperparameter grid at fixed geometry with a free bg
index, so it is biased high by a few units; |Δ| < 20 is reported as "not decisive" (the next-steps rule). PS Gem: against
`…_unbounded_wide_image.json`, marked provisional (its field is being changed). HR 4049, HD 108015: χ² only.

Nothing is selected by NCC or by eye.

## 6. Degeneracies and the check for each

| degeneracy | check |
|---|---|
| r₀ – σ – f_bg (ring collapses to a blob plus halo; seen on AI Sco under N1, and on EN TrA in the stored fit) | M0 is a rung; E2; multi-start over r₀ in every arm; the ln Z gap between the blob and the best ring mode is reported |
| star flux – compact disc flux – f_bg | star index, UD and bg index fixed; Laplace correlation matrix stored; `bgfree` arm |
| star and disc level – night gains (N2 raw χ²/pt 8–17 on AI Sco at own χ²/pt 1.0–1.1) | raw and own χ² side by side; f_prim under N0, N1, N2 side by side; jackknife |
| PA 180° – sign of odd harmonics | 180° window; brightest-point PA from the rendered model |
| m = 1 – centre offset – secondary | M2, M2o, `nosec`, `secfree` all reported; AI Sco's M2o centre beside De Prins's (0.51, −1.4) mas |
| m = 1 skew – inclination (s₁ along the minor axis can be geometry) | (c₁, s₁) reported separately; not interpreted |
| inc → 0 leaves PA free | `pa_unconstrained` flag; PA tests waived below 20° |
| HD 95767's 25 % companion at 0.7 beam – ring radius | `secfree`; target reported as data-limited whatever the result |
| PS Gem: moving primary – disc size | no azimuthal statement for this target |

## 7. Robustness

* **Leave-one-night-out jackknife** (P1, in the core): at the N2-selected model, refit with each night removed, starting from
  the optimum. Jackknife sd of inc, PA, r₀, σ, c₁, s₁, f_env, f_bg. This is the error bar for geometry.
* **NUTS** (P2; AI Sco and IRAS 15469 only): `numpyro_model(result.model, priors, data_N2)` with
  `NUTS(…, init_strategy=init_to_value(values=result.values), target_accept_prob=0.9, **gauss_newton_mass(…))`, 2 chains in
  sequence, 500 + 1000. One attempt, 30 min box per target. Keep if r̂ < 1.01 and bulk ESS > 400 for every parameter;
  otherwise store the draws with `not_converged`. If `multimodal`, sample the second mode too and report both separately.
* **Injection–recovery** (P3; IRAS 15469, 3 seeds): simulate from the selected model on its own (u, v) with N2 noise
  including drawn night terms; rerun M0–M2o under N2. Pass: selected rung recovered in ≥ 2 of 3, inc within 10°, PA within
  15°, no higher rung accepted.

## 9. Decisions the results feed (all rules fixed now; nothing waits for Ben)

* **`usable_for_gp`** is true when, under N2: the selected model is a ring (§5); it is eligible; `lnZ` flags do not
  include `evidence_unverified` for the M1-over-M0 pair; jackknife sd ≤ 10° in inc and ≤ 15° in PA (PA waived below
  inc 20°); no `multimodal` competitor differing by > 15° in inc or > 30° in PA. Otherwise the geometry JSON carries the
  De Prins image geometry (`postagb_paper.IMAGE`) with `source: "De Prins image"` and the reason.
* **AI Sco (Q1)**. The geometry written to the JSON is:
  1. ours, if `usable_for_gp` and within (15° inc, 25° PA) of the De Prins image ellipse (46°, 135°);
  2. the De Prins image geometry otherwise, with our modes, their `lnZ_use`, raw and own χ², the `arm_dependent` flag and
     the M2o centre beside their (0.51, −1.4) mas listed in the report. Case "ours is usable but elsewhere" and case "no ring
     or unstable" are both this branch and are named as such.
* **Ring as GP mean (Q2, stage F5 input)**: `<T>_param_image.npz` and the geometry JSON are the inputs of a later
  `--mean ring` / `--geometry-file` option of `postagb_imaging.py`. That option is not written here; the report gives its
  three-line spec.
* **Q3**: the table of §5; a list per target of whether gp-polar, ORGANIC and the ring agree on θ_rim, i, PA and the side of
  the brightness maximum.


## Implementation notes (not part of the pre-registration)

Written with `scripts/postagb_parametric.py`; none of this changes a rule above.

* **The rows of `chi2r`.** `P.chi2r` divides by virgil's `n_residuals`. For 4-telescope data the closure phases are correlated (three independent of
  four triangles), so virgil whitens them to independent combinations and adds one periodic penalty row per closure phase, which is about zero at a
  good fit. On AI Sco, 2184 = 1008 V² + 504 whitened independent closure-phase combinations + 672 penalty rows (`data.n_independent` = 1512;
  the plan's "672 closure phases + 504 further rows" is the same count labelled the other way round). N below is the 2184 of gp-polar's `n_points`;
  χ² per row under any likelihood is therefore about n_independent / N = 0.69 of χ² per independent observable at a good fit. Both counts are stored
  (`n_rows`, `n_independent`). virgil's `info["chi2_red"]` divides by `n_independent`; the scripts use χ² / N throughout so that they match gp-polar.
* **Schema keys the report stage reads.** Selection chain rows: `pair` ("M1/M0"), `dlnZ`, `dbic`, `verdict`, `flags` (plus `d_lnZ`, `d_bic`, `basis`).
  `gp_polar.<arm>`: `delta`, `mismatch`. Jackknife: sd of `inc, pa, r0, sigma, c1, s1, f_env, f_bg`. `<T>_param_residuals.npz` (written by `select`).
* **Selected model and the jackknife.** Run `select`, then `jackknife` (it reads the N2 selection), then `select` again so that `selected.json` carries
  the jackknife sds.
* **M3 / M3o, M5.** `--models M3` means the selected model plus m = 2 (`M3o` if M2o was selected). M5 (outer ring) is not in WP1.
* **Selection when M0 has no eligible fit** (for example f_bg on its 0 bound, as in the IRAS 15469 smoke): the first rung with an eligible fit is
  accepted by default and the row says so.
