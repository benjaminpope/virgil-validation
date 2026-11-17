# gp-polar hyperparameters: why four discs break into dots, and how to choose them

Status: the azimuthal cap, cold logged grid cells and a bounded evidence selector (§8) are in place (`scripts/postagb_imaging.py`). The
calibration that chose the bounds is §4 and §7. This note replaces the floors of PR #264.

## 1. The failure

AI Sco, EN TrA, HD 95767 and PS Gem all pick the same grid corner: σ = 4 (the upper edge after one extension), ℓ_θ = 0.0625 rad
(the lower edge after one extension) and ℓ_r = 0.5 beam (the first value). The polar grid has 32 azimuthal nodes 0.196 rad apart,
so ℓ_θ = 0.0625 rad is white noise from node to node. Softmax positivity with σ = 4 turns each node into its own blob, and bilinear
interpolation draws a ring of dots. The two clean controls pick σ = 2 with ℓ_θ = 0.25 rad (IW Car) or 2 rad (IRAS 15469).
Numbers: `virgil-paper/reports/postagb-chains-diagnosis.md`.

## 2. What the evidence is doing

`virgil.imaging.log_evidence` is the Laplace evidence in whitened latents z:
ln Z = −½χ² − L − ½|z|² − ½ ln det(I + JᵀJ). The Occam term is there, so the corner pick does not come from a missing penalty.
Four things could still make it misleading here. The calibration run is designed to tell them apart.

1. **The error model.** χ²/N = 0.56–0.74 at the pick is not a contradiction in itself. At an evidence optimum MacKay's
   self-consistency gives χ² ≈ N − γ, with γ = Σ s²/(1 + s²) the number of well-determined directions. So χ²/N ≈ 0.6 means the
   evidence thinks the data pin down γ ≈ 0.4 N, roughly 900 directions for AI Sco's 2184 points. That only holds if the points
   are independent. PIONIER's six channels of one (frame, baseline) share a calibration gain. AI Sco and EN TrA show it clearly:
   their adjacent-channel residual correlation is 0.77 and 0.65, their common-mode ratio 4.0 and 3.6, and their error scales
   1.34 and 1.28. Under errors treated as independent, structure that fits those correlated residuals is credited as signal.
   Filling uv-plane wiggles needs power at large radius, which is where the dots are.
2. **The Laplace approximation under the exp link.** Pixels the field has made dark have J ≈ 0, so their latents add nothing to
   ln det(I + JᵀJ). A sparse image (a few bright nodes) pays the Occam cost only for its bright nodes. Whether this biases the pick
   toward large σ on data that does not need it is exactly what a smooth truth with white noise tests.
3. **The comparison is not like for like.** Each cell was warm-started from the previous one in itertools order, with LM's
   1000-step limit and no saved loss history. So the evidences of neighbouring cells came from different starting points.
4. **Model mismatch.** The pixel-image fits leave out bandwidth smearing, which virgil does not support for pixel Images. The
   parametric disc fits of the `scale` stage reach χ²/pt only 1.3–4.8. Real discs do have structure beyond a ring with a first
   harmonic, so some of the evidence's appetite may be honest.

Earlier evidence: `scripts/postagb_noise_check.py synth` (design/postagb_noise_model.md) found that the gp-iso evidence picks the
smooth, long-length cell on a smooth IW Car truth, under white noise and under gain noise alike. That check covers gp-iso, the long
end of the length grid and one clean target. It says nothing about gp-polar's short end or the four failing targets.

## 3. The azimuthal cap (done)

Ben's direction: none of these discs should go to high azimuthal order. A floor on ℓ_θ cannot enforce that, because the periodic
Matérn (ν = 3/2) has a long tail in m and σ multiplies it inside an exponential:

| ℓ_θ (rad) | variance in orders m > 3 | rms log-brightness in m > 3 at σ = 2 | at σ = 4 | amplitude at m = 6 / m = 0 |
|---|---|---|---|---|
| 0.0625 | 74 % | 1.72 | 3.44 | 0.96 |
| 0.25 | 43 % | 1.31 | 2.62 | 0.57 |
| 0.5 (PR #264's floor) | 17 % | 0.83 | 1.66 | 0.25 |
| 1 | 3.8 % | 0.39 | 0.78 | 0.08 |
| 2 | 0.5 % | 0.15 | 0.29 | 0.02 |

At PR #264's floor (ℓ_θ = 0.5 rad, σ = 2) the high orders still modulate the brightness by a factor of e^0.83 ≈ 2.3. Even ℓ_θ = 1 rad
at σ = 4 leaves a factor of 2.2. Only a hard cap removes them. The options were:

* **(a) n_θ = 8 nodes, so m ≤ 4.** Rejected: bilinear interpolation between nodes 45° apart draws a polygon, which is a new artefact
  at the scale of the arcs we want to see.
* **(c) ℓ_θ ≥ 1 rad alone.** Rejected because of the table above.
* **(b) An explicit order cap (chosen).** `LowOrderField`, a subclass of `MultipoleField` in the script, low-passes the polar field
  along the azimuth to orders m ≤ `POLAR_M_MAX` = 3 (per target: `polar_m_max`). It keeps n_θ = 32, so an m ≤ 3 field is sampled at
  10 or more nodes per period and interpolates smoothly. It rescales the kept orders, using virgil's documented spectrum, so that
  σ is still the rms contrast per node. The removed latents have zero Jacobian, stay at zero, and add nothing to the evidence, so
  cells of one cap compare fairly. It needs no virgil change; tests in `tests/test_postagb_polar_cap.py`, including one that the rendered image has no
  power above m = 3.
  A documented `m_max=` argument on virgil's `MultipoleField` would be the cleaner long-term home: a short, separate virgil PR,
  proposed and not started.

Why m ≤ 3: m = 1 carries the inclination and illumination asymmetry, and m = 2 and 3 allow a two-sided or lopsided arc. Ben asked
for 2–4. At the hole (r ≈ 1.2 mas) the circumference is about 8 beams, so m ≲ 4 is all the data could resolve there anyway. At
4 mas the data could resolve m ≈ 10, so the cap is a prior choice, not a resolution limit. Under the cap, ℓ_θ = 0.25 rad already
gives orders 0–3 nearly equal variance (S(3)/S(0) = 0.71), so the ℓ_θ grid is now {0.25, 0.5, 1, 2} rad with no downward
extension (upward to 4 rad, nearly axisymmetric).

Also done:

* Every grid cell now starts cold from the same point: the scene's own parameters and zero latents. The evidence no longer
  depends on the order the grid is visited in.
* The LM step limit is raised to 3000.
* Each cell's loss history (one point per LM chunk), steps, stopping reason and time go into the image `.npz` (`mp_cells`). The
  summary records `gp_steps`, `gp_stop` and `gp_cells_unconverged`.

## 4. Calibration (stage C, to run)

`postagb_imaging.py calibrate --arm {white,corr,smear,geo} --m-max {-1,3} --seed N` runs, for each of AI Sco, EN TrA, HD 95767, PS Gem and the
controls IW Car and IRAS 15469:

* **Truth.** The target's own parametric scene (stars, fluxes and indices from `geometry_fit` on the real data). Its Gaussian ring
  is moved to where the image can hold it: width between half a beam and 0.15 of the polar span, radius at least 2.5 widths from
  the hole and from r_max. This matters because the real fits for AI Sco and EN TrA put the ring at the 0.5 mas prior floor, with
  flux inside the blanked hole, and fitting that would test mismatch, not selection. Nothing in the truth is above m = 1.
* **Noise arms:**
  * `white`: the quoted errors × the target's MacKay factor, fitted with those (correct) errors.
  * `corr`: the target's virgil-vlti error model (per-frame telescope and baseline gains, closure offsets, pair-test widths),
    fitted as the real data are: independent errors with a pilot's MacKay factor.
  * `smear`: white noise, but the truth seen through the spectral channels (bandwidth smearing), fitted without smearing.
  * `geo`: white noise, but the polar GP deprojects with an inclination and a position angle each 8° off the truth's
    (`CAL_GEO_ERROR_DEG`, a typical error of the real ring fits). A wrong geometry shows up in the polar frame as m = 2 and
    higher structure, which the evidence could buy with large σ and short ℓ_θ (item 4). The other arms give the GP the
    truth's own geometry, so they cannot test this.
* **Grid.** A fixed grid with no extension: ℓ_r ∈ {0.5, 1, 2} beams (those at or above one pixel), ℓ_θ ∈ {0.0625, 0.25, 1} rad
  and σ ∈ {0.5, 1, 2, 4}, every cell cold. That is 36 cells, or 24 where half a beam is below a pixel. It runs with three noise
  seeds capped (m ≤ 3) and one seed uncapped (`-1`), as a diagnostic. Each cell is also fitted on the even-numbered files and
  scored by χ²/pt on the odd ones (cross-validation over exposures, which respects per-frame correlation).
* **Scoring.** Each cell's error against the truth: relative rms of the unit-sum images, raw and convolved with the comparison
  beam, plus NCC. For three selectors (evidence, CV and the truth's own best), the run reports the picked cell and its rank
  against the truth. Figure: truth, the three picks, and ln Z against image error.

**Decision rule.** Fixed here before stage C runs. It is judged on the capped runs, three seeds per target and arm. A selector
passes an arm if, on all six targets, both hold:
* the median over seeds of its pick's rank fraction (rank by beam-convolved error ÷ number of cells; `rank_fraction` in the json)
  is ≤ 0.14, which is the best 5 of 36;
* it picks the failure corner (the smallest ℓ_r in the grid with σ = 4; `at_corner`) in at most one of the three seeds.

The same rule is applied to the evidence and to CV.

| white | corr | smear | geo | conclusion | change |
|---|---|---|---|---|---|
| pass | pass | pass | pass | the cap was the fix; the real-data picks reflect the data | keep the evidence; report edge picks |
| pass | fail | – | – | correlated channel errors drive the pick (item 1) | select by CV over exposures (implemented in `calibrate`; move into `fit_polar`) |
| pass | – | fail | – | missing smearing creates structure (item 4) | needs smearing for pixel Images in virgil (separate issue); meanwhile CV |
| pass | – | – | fail | geometry error is bought as structure (item 4) | fit inc / PA with the field, or select by CV; whichever passes `geo` |
| fail | – | – | – | the Laplace selector is biased under the exp link (item 2) | CV, or a fixed prior (σ = 1, ℓ_r = 1 beam, ℓ_θ = 1 rad) if CV also fails; whichever recovers the truth best |

Where several rows apply, the change has to pass every failing arm. A change chosen from these results is rerun as stage C before
it becomes the default.

The uncapped arm shows whether the cap alone changes the evidence's preferences. Comparing it with the real uncapped picks shows
whether the old warm-started chain contributed.

## 5. Runs

Submit blocks are in the PR. They chain stage A (`--array=0,4,5,6,7,8`, which rebuilds the scale files in this commit's
`out/<pins>/`) into stage C (`--array=0-95`) and into the B1 rerun of gp-polar with the cap (`--array=66,80,94,122,10,108`). If
the calibration changes the selector, B1 is run again from that commit.

## 6. PR #264

Superseded. Its ℓ_θ ≥ 0.5 rad floor leaves 17 % of the azimuthal variance above m = 3 (table in §3). Its ℓ_r ≥ 1 beam and σ ≤ 2 bounds
are untested guesses until stage C reports; they come back only if the calibration selects them.

## 7. Stage C result (jobs 21054315 → 21054318, vv 0e4377c, virgil 6fa8afe)

All 96 tasks finished, every cell converged (36 cells per run). Median rank fraction of the pick over the three capped seeds,
with the number of seeds at the failure corner in brackets (pass: ≤ 0.14 and ≤ 1):

| target | evidence: white | corr | smear | geo | CV: white | corr | smear | geo |
|---|---|---|---|---|---|---|---|---|
| AI Sco | 0.06 (0) | 0.08 (0) | 0.06 (0) | **0.25** (0) | 0.11 (0) | **0.36** (0) | 0.08 (0) | **0.61** (1) |
| EN TrA | 0.08 (0) | 0.14 (1) | 0.08 (0) | **0.28** (0) | 0.14 (0) | 0.11 (0) | 0.14 (0) | **0.53** (1) |
| HD 95767 | 0.03 (0) | 0.03 (0) | 0.03 (0) | 0.03 (0) | **0.25** (0) | 0.03 (0) | **0.22** (0) | 0.08 (0) |
| PS Gem | **0.19** (0) | **0.33** (0) | **0.19** (0) | **0.28** (0) | **0.42** (0) | 0.14 (0) | **0.42** (0) | **0.67** (1) |
| IW Car | 0.06 (0) | 0.11 (0) | 0.06 (0) | 0.06 (0) | 0.06 (0) | 0.08 (0) | 0.06 (0) | 0.06 (0) |
| IRAS 15469 | 0.03 (0) | 0.11 (0) | 0.03 (0) | 0.06 (0) | 0.11 (0) | 0.14 (0) | 0.11 (0) | **0.17** (1) |

The fixed prior (σ = 1, ℓ_r = 1 beam, ℓ_θ = 1 rad) scores 0.06–0.39, failing IW Car on every arm and AI Sco / EN TrA under `geo`.

**By the rule as written, no selector passes any arm.** The evidence fails `white` only through PS Gem: it always picks ℓ_r = 0.5
beam, where the truth's best is 2 beams (beam-convolved error 1.26 × the best cell). It fails `geo` on AI Sco, EN TrA and PS Gem
(error ≤ 1.10 × best on the first two). CV does worse than the evidence everywhere except PS Gem `corr`. The fixed prior is
worse than the evidence on 19 of 24 target-arms.

What the run does settle:
* **On a smooth truth the evidence does not run to the corner.** It picked σ = 4 / smallest ℓ_r in 1 of 72 capped runs and none
  of the 24 uncapped ones, under white, correlated, smeared and mis-deprojected noise alike. Items 1–3 of §2 do not by themselves
  produce the real-data failure.
* **On the real data it still does.** The capped B1 rerun (job 21054320) picks σ = 4, the upper edge, on AI Sco, EN TrA,
  HD 95767, PS Gem and now IW Car, with ℓ_r = 0.25 beam (the downward extension) on four of them. Only IRAS 15469 sits inside
  (σ = 2, ℓ_r = 2 beams, ℓ_θ = 1 rad). The cap turns the chains of dots into short arcs and blobs, but the blobs remain.

So the real data contain something none of the four arms simulate: real compact structure, or a systematic outside the error
models tested (the parametric fits reach χ²/pt only 1.3–4.8). Choosing between "the evidence is right about the data" and "cap σ
or ℓ_r by prior" is a science call for Ben, not something this calibration can make.

## 8. The selector: the evidence within prior bounds (Ben: "bound by prior")

Fixed here from the stage C runs alone, before the bounds were applied to any real target:

* **ℓ_r ≥ 1 beam** (`POLAR_LR_MIN_BEAMS`). The data do not constrain radial structure below the beam. The truth's best cell was
  2 beams in 65 of the 72 capped runs. The evidence's one `white` failure (PS Gem) came from picking 0.5 beam.
* **σ ≤ 2** (`POLAR_SIGMA_MAX`), an rms log-contrast per node of e² ≈ 7×. The truth needed σ = 4 in only 5 of the 72 runs.
* **ℓ_θ unchanged.** The m ≤ 3 cap already bounds the azimuthal structure, and the grid stays {0.25, 0.5, 1, 2} rad.

Within these bounds the evidence still chooses. The grid is never extended past a bound, and a pick on one is reported in the
summary (`gp_at_bound`) along with the bounds themselves (`gp_lr_min_mas`, `gp_sigma_max`).

Applied to the stage C runs (simulations only), the bounded evidence passes `white`, `corr` and `smear` on all six targets
(median rank fraction ≤ 0.14, beam-convolved error ≤ 1.13 × the best cell). `geo` still fails on AI Sco and EN TrA (0.25 and
0.28, error ≤ 1.10 × best). An 8° geometry error remains a known limitation of the deprojected field.

## 9. Night-level calibration errors: the decisive test (pre-registered)

Diagnosis (virgil-paper `reports/postagb-corner-diagnosis.md`): on all four failing targets the V² channel-mean residual of the
corner fit is coherent per (night, baseline). The ANOVA share is 0.50–0.73, against 0.19–0.42 after shuffling. On the controls the
fit absorbs it. Stage C simulated only per-frame gains, so it could not reproduce this.

**Model** (`--noise night`, `night_noise`): the vlti terms, plus a gain on log|V| per (night, baseline) and a closure-phase
offset per (night, triangle). Each spans every frame of its night, and the likelihood marginalises them (virgil's spanning
modes). A night runs from MJD .65 to the next.

**Widths, fixed before any night fit or pick was looked at** (`scripts/postagb_night_widths.py`, written to
`scripts/postagb_night_widths.json`):
* Residuals are taken against the *published* SPARCO + ORGANIC model (`pub_v2`, `pub_t3` of the compare stage, vv 0e4377c), never
  against a virgil image.
* A two-level method-of-moments ANOVA follows: channel means per (frame, baseline) or (frame, triangle), then the between-cell
  component over (night, baseline) or (night, triangle). Frames are the replicates, so per-frame gains and white noise are not
  counted twice.
* Published-model misfit shared by a night also lands in τ, so these are upper estimates.
* A target missing from the table gets the median of the others.

| target | nights | gain per (night, baseline), log\|V\| | closure offset per (night, triangle) |
|---|---|---|---|
| AI Sco | 10 | 0.025 | 1.33° |
| EN TrA | 7 | 0.000 | 0.91° |
| HD 95767 | 7 | 0.012 | 1.03° |
| PS Gem | 5 | 0.0035 | 0.09° |
| IW Car | 8 | 0.012 | 0.75° |
| IRAS 15469 | 5 | 0.023 | 1.27° |

EN TrA's night-to-night V² offsets do not exceed what its frame-to-frame scatter predicts with 2–3 frames per cell, and PS Gem's
are small. On those two the night arm is close to `corr`.

**Runs:**
* `calibrate --arm night` (capped, the stage C grid, three seeds) on AI Sco, EN TrA, HD 95767 and PS Gem. It simulates the
  night terms and fits as the real data are fitted: independent errors with a pilot MacKay factor.
* B1 gp-polar `--noise night --unbounded` on all six targets. It is unbounded because, under the §8 bounds, the picks are inside by
  construction and could not move.

**Decision rule.** The night errors are confirmed as the cause if both hold:
1. The night-arm simulations reproduce the corner: `at_corner` in ≥ 2 of 3 seeds on ≥ 3 of the 4 targets.
2. Under `--noise night --unbounded` the real picks move inside the bounds (ℓ_r ≥ 1 beam and σ ≤ 2) on ≥ 3 of the 4 failing
   targets.

If both hold, the night model replaces the §8 bounds as the fix, and B1 is rerun with `--noise night` (bounded and unbounded should
then agree). If either fails, the cause is not established, the §8 bounds stay, and the chromatic test of the diagnosis (item 3)
is next.

**Also noted:** `LowOrderField` caps the log-brightness field at m ≤ 3. The brightness exp(field) is not capped. At σ = 4 the
rendered images carry 44–80 % of their non-axisymmetric power above m = 3 (diagnosis §c), and `test_the_cap_reaches_the_rendered_image`
checks the log-field at σ = 1 only. The §8 ceiling σ ≤ 2 limits this; the cap is a statement about the log-field.

## 10. §9 result (jobs 21074145 / 21074149 / 21074152, vv c14cb93, virgil 6fa8afe): not confirmed

**Criterion 1 fails (1 of 4).** Evidence pick in the night-arm simulations, per seed (`at_corner`, rank fraction, ℓ_r in beams, σ):

| target | seed 0 | seed 1 | seed 2 | corner |
|---|---|---|---|---|
| AI Sco | yes, 0.97, 0.5, 4 | yes, 1.00, 0.5, 4 | yes, 1.00, 0.5, 4 | 3/3 |
| EN TrA | no, 0.06, 2, 2 | yes, 0.83, 0.5, 4 | no, 0.39, 2, 2 | 1/3 |
| HD 95767 | no, 0.67, 0.5, 2 | yes, 0.92, 0.5, 4 | no, 0.67, 0.5, 2 | 1/3 |
| PS Gem | no, 0.33, 0.5, 1 | no, 0.22, 0.5, 1 | no, 0.19, 0.5, 1 | 0/3 |

All cells converged. Night-level errors of the measured widths drive the evidence to the corner only on AI Sco.

**Criterion 2 fails (0 of 4).** Under `--noise night --unbounded`, the real picks go further from the bounds, not inside them:

| target | ℓ_r (beams) | σ | χ² | bounded B1 (§8): ℓ_r, σ, χ² |
|---|---|---|---|---|
| AI Sco | 0.5 | 4 | 1.89 | 1, 2, 1.20 |
| EN TrA | 0.25 | 4 | 1.49 | 1, 2, 1.04 |
| HD 95767 | 0.25 | 2 | 1.09 | 1, 2, 0.99 |
| PS Gem | 0.25 | 4 | 3.38 | 1, 2, 1.12 |
| IW Car | 0.25 | 4 | 1.53 | 1, 2, 1.31 |
| IRAS 15469 | 1 | 2 | 1.03 | 2, 2, 1.04 |

ℓ_r = 0.25 beam is one pixel. With the night terms marginalised, the evidence prefers even rougher fields, and the χ² worsens.

**Consequence (as pre-registered).** The night errors are not the cause. The §8 bounds stay and the chromatic test is next.
Under the bounds, all six bounded B1 picks sit on a bound: ℓ_r = 1 beam and σ = 2 on five targets, and σ = 2 on IRAS 15469. The
images are set by the prior, not by the data. They are not yet figure quality: AI Sco still breaks into blobs, and PS Gem has an
arc at the field edge with its peak 1.3 mas from the star.
