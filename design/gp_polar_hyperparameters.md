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
(median rank fraction ≤ 0.14, beam-convolved error ≤ 1.13 × the best cell). The rank fraction is the bounded pick's image-error rank among all
36 cells of the capped grid, divided by 36: the same denominator as the unbounded selector, so the bounded pick competes with
cells it cannot choose (≤ 0.14 is the top 5 of 36). Every stage C cell is fitted cold and independently, so re-selecting within
the bounds on the existing cells gives exactly the runs a bounded stage C would; that is the rerun §4 requires. `geo` still fails on AI Sco and EN TrA (0.25 and
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

## 10. §9 result (jobs 21074145 / 21074149 / 21074152, vv c14cb93, virgil 6fa8afe): confounded, not a rejection

**Correction (2026-11-18), after an independent review (virgil-paper `reports/postagb-five-targets-review.md` §1, §3).** This test
changed two things at once, so its failure does not reject the night-level cause:
* `--noise night` builds on the `vlti` model. That multiplies the quoted V² errors by 0.31–0.61 and the closure errors by
  0.64–0.89, where the B1 run multiplies by 0.95–1.34. With white errors 2–4 times smaller, every spectral wiggle within a
  baseline becomes significant, and χ² at the pick is 1.1–3.4.
* The night gain width for EN TrA was 0 (`postagb_night_widths.py`: with 2–3 frames per cell, frame scatter swamps the
  between-night term), so EN TrA had no gain term. The method-of-moments widths are probably too small elsewhere too.
* The sim arm did reproduce the corner on AI Sco in 3 of 3 seeds.

The review refitted three cells per target with the white errors left at the MacKay scale and night terms of 3 % (gain) and
2° (closure) marginalised. The corner preference collapses on AI Sco, EN TrA and HD 95767, and ln Z rises by 240–300; on
EN TrA the smoothest cell wins. Its other two findings: PS Gem's 5 mas field cuts off most of the environment (with a 15 mas
field the smoothest cell wins), and IW Car goes to the corner only under the m ≤ 3 cap. So the night cause is open, not
rejected. A clean rerun changes only the night terms (MacKay white errors kept, widths fixed in advance) and runs the full grid.
The original text follows, unchanged.


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

**Consequence (as pre-registered; superseded by the correction at the head of this section).** As run, the night errors were not the cause. The §8 bounds stay and the chromatic test is next.
Under the bounds, all six bounded B1 picks sit on a bound: ℓ_r = 1 beam and σ = 2 on five targets, and σ = 2 on IRAS 15469. The
images are set by the prior, not by the data. They are not yet figure quality: AI Sco still breaks into blobs, and PS Gem has an
arc at the field edge with its peak 1.3 mas from the star.

## 11. Chromatic mismatch: the second test (pre-registered, Ben: "Run chromatic test")

**Hypothesis.** The SPARCO scene has a grey environment: one image times one power law. If the real disc's spectrum changes with
radius (a hot inner rim, cooler dust further out), or the star's index is not the fixed ORGANIC SIND1, a grey model cannot fit
the colour-dependent visibilities. The evidence might then pay for small-scale structure that absorbs this (the σ = 4, short-ℓ_r
corner). Stage C cannot test this: its truth is grey SPARCO, so the model is right by construction (diagnosis §a).

**Prior expectation: rejection.** The diagnosis projected the real residuals onto Legendre orders across the six channels. Most of
the excess over the published model is in the channel mean (c0² 2–12), not in the slope (c1² 0.6–2.6). Only AI Sco removes as
much slope as mean, and the largest slope excess is in the control IRAS 15469, which does not pick the corner. A chromatic error
mostly shows up in the slope, so this test should fail unless the slope is doing more than the diagnosis found.

**Sim arm `calibrate --arm chrom`** (capped, seeds 0–2, AI Sco, EN TrA, HD 95767, PS Gem; 12 tasks):
* Truth: the stage C representable truth, with two changes.
  * The ring's spectral index falls with radius: α(r) = d + s (r − r_t), with s = −Δα / (2w) and Δα = 2. The index is 2 higher
    one ring width inside the centre than one width outside. Here d is the fitted environment index, r_t the ring radius, w its
    width. This is virgil's chromatic `MultipoleDisk` (`spectral_index=RadialLinear`), with the same F_λ ∝ λ^α convention as
    `PowerLaw`.
  * The star's index is SIND1 − 1 (towards Rayleigh–Jeans, −4).
* Noise: white, at the MacKay factor, as in the white arm, so only the spectrum differs.
* Fit: unchanged, the grey SPARCO scene with the star at SIND1.

**Why these sizes** (from blackbody physics and the published model, not from virgil picks):
* ORGANIC's environment indices (SIND0 0.4–0.9) correspond to H-band dust at about 1500–1700 K. The local F_λ index at 1.65 µm is
  0.5 at 1600 K and 2.1 at 1230 K.
* With T ∝ r^−1/2, a ring from r − w to r + w (w/r about 0.25) spans a temperature ratio of about 1.3, so Δα ≈ 1.6. Δα = 2 is the
  strong end of physical, chosen so that a null result means something.
* A star offset of 1 is about the gap between the paper's SIND1 (−2.1 to −3.2) and a pure photosphere (−4). That is the error
  if SIND1 has absorbed hot dust emission.

**Real counterpart: B1 gp-polar `--chrom --unbounded`** (6 targets). The scene gains:
* a chromatic ring `rim`: a `MultipoleDisk(GaussianRing)` with no harmonics, inc and PA fixed at the scale-stage geometry.
  * radius LogUniform(0.5, field radius), width LogUniform(0.1, 12) mas, both started at the scale geometry;
  * flux LogUniform(10⁻³, 5);
  * index α(r) = value + slope (r − r_0), with r_0 the starting radius; value Uniform(−6, 6), as the environment index;
  * slope Uniform(±4 / (2 w_0)): up to twice the sim's Δα across ±w_0, where w_0 = max(starting width, half a beam);
* a free star index, Uniform(SIND1 − 2, SIND1 + 2).

The grey GP image keeps the rest. The grid, its axes and the evidence are as in B1. Output files carry `_chrom`.

**Decision rule.** Chromatic mismatch is confirmed as the cause only if all three hold:
1. The chrom sims reproduce the corner: `at_corner` in ≥ 2 of 3 seeds on ≥ 3 of the 4 targets.
2. Under `--chrom --unbounded`, the real picks move inside the bounds (ℓ_r ≥ 1 beam and σ ≤ 2) on ≥ 3 of the 4 failing targets.
3. On those targets the fit actually uses the chromatic terms: the ring carries ≥ 1 % of the flux with |Δα| = |2 slope w| ≥ 0.5,
   or the star index moves ≥ 0.5 from SIND1. Otherwise the picks moved for some other reason (for example the extra flux
   component), not because of colour.

**Outcomes.**
* All three hold: the chromatic scene replaces the §8 bounds. B1 is rerun bounded with `--chrom` for the paper figure, and the two
  picks should agree.
* 1 holds and 2 fails: colour errors can make the corner, but they are not what these data have. The bounds stay.
* 1 fails: a strong but physical colour error does not make the corner. Rejected.
* (Superseded by the §10 correction: §10 did not reject the night cause, so this outcome no longer follows.) If the cause is rejected here as well as in §10, no third cause test is planned. The §8 bounds are the final selector, and the
  paper says plainly that the picks sit on the bounds: the resolution of these images is set by the prior.

## 12. The binary: does freeing it fix the corner? (pre-registered, Ben: "the priority here is ... solving the binarity")

*Revised 18 Nov 2026, before anything was run, after the Fable plan review
(virgil-paper `reports/postagb-binary-plan-review.md`). The first version (88c923d) compared modes with an evidence that held the scene
parameters at their MAP. On simulated static binaries that version passes criterion 1 (`epoch` − `paper` +11 to +19), so its rule
could not test the claim. The model is unchanged; the evidence, the rule, PS Gem's field and the night arms are new.*

**De Prins's setup** (paper §4.1–4.3.2, Zenodo headers). For each component, f_j(λ) = f_j0 (λ/1.65 µm)^d_j in F_λ, with
absolute indices.
* Primary: a UD at the origin, d fixed at SIND1 from the SED (−2.1 to −3.2); flux and diameter free.
* Secondary: a grey point source, d = 0 (about a 1750 K blackbody at 1.65 µm, dust-like, not stellar); position fixed from
  the geometric fit, flux free. "Chromatic companions did not improve the BIC".
* Disc: one grey image with a fitted index.

Our B1 scene matches theirs, except that the secondary's position has a PMOIRED prior. The window is 2–9 % of the period. The
review's per-night positions shift by 0.1–0.2 mas, but against a fixed image; with the field free the per-night position sd is
0.08–0.18 mas, so those shifts are not detections.

**Hypothesis.** A secondary fixed and static, with an imposed colour, is wrong in position, in time (orbital motion within the
window) or in spectrum. The evidence then buys sub-beam structure near the stars to absorb the misfit. The disc is circumbinary,
so in a frame centred on the primary the primary also moves against the disc (reflex motion). That matters for PS Gem, whose
companion is dark (a₁ sin i 1.51 AU).

**Prior expectation: the binary model is better, and the corner stays.** In the review's local simulations, a secondary moving
at the orbital rate costs 11–31 in ln Z under `paper`, and `motion` recovers it. But it changes the corner-minus-bounded evidence
by only 2–9, where the real data have +14 to +63. So the likely outcome is rule 1 without rule 2. That is why the night arms run
in the same job.

**Models compared** (`--binary`, gp-polar, unbounded, every cell cold):

| mode | secondary | primary | extra parameters |
|---|---|---|---|
| `paper` | as B1: PMOIRED position prior, index 0, static | static at origin, index SIND1 | none |
| `free` | position Normal(sec0, 1 mas), flux LogUniform(10⁻³, 1), index Uniform(−5, 2) | as `paper` | 1 (index) |
| `motion` | `free` plus linear motion (v_E, v_N) | linear motion against the disc | 4 velocities |
| `epoch` | `free` plus a position offset per night | an offset per night | 2 per night per star |

The primary's index stays at SIND1 and the disc keeps its own power law throughout.

**Priors, from the orbits, not from picks.**
* Velocities Normal(0, 0.02 mas/day) per axis. Twice the fastest companion expected, 2π·sep/P = 0.011 mas/day for AI Sco
  (sep 1.75 mas, P 977 d); EN TrA 0.008, HD 95767 0.003.
* Per-night offsets Normal(0, 0.2 mas): about the largest orbital displacement from the mid-time (AI Sco ±0.18 mas).
* Time and night are as in `night_labels` (night = floor(MJD − 0.65)); velocities refer to the mid-time of the data.
* Targets without a secondary (PS Gem, IRAS 15469) get only the primary's terms. Their `free` equals `paper`, so it is skipped.

**Arms** (stage BN, one job, pins in the hand-over):
* Binary: the four modes on IW Car, AI Sco, EN TrA, HD 95767, IRAS 15469 and PS Gem, under `--noise current` (stage A's MacKay
  factor, the same in every mode; `binary_scene` does not touch the data, so the §10 confound is held fixed).
* PS Gem wide: `paper`, `motion` and `epoch` with `--wide-field` (15 mas half-widths in place of 5; the review found the 5 mas
  field to be PS Gem's cause). Stage A's error scale and geometry are reused.
* Night (Ben, 18 Nov): `paper` and `motion` on the six targets under `--noise nightw`: the MacKay white errors plus a gain of
  3 % per (night, baseline) on log|V| and an offset of 2° per (night, triangle), marginalised in the likelihood (the review's
  setting for the §10 correction).

**Evidence between modes.** The grid's `log_evidence` holds the scene parameters at their MAP, so it picks the cell within a
mode and nothing else. Between modes and arms the rule uses `log_evidence_scene` at each run's own pick: the Laplace evidence
with the scene parameters integrated too (`scene_evidence`; Λ = 1/σ² for Normal priors, 0 for flat ones). The night terms are
marginalised in the likelihood, so the night arms' evidences compare with the binary arms' on the same data. `paper` must
reproduce the unbounded picks of job 21054320 to within 1 in grid ln Z (AI Sco −965.7, EN TrA −653.7, HD 95767 −1071.3,
PS Gem −388.3, IW Car −944.5, all at σ = 4; IRAS 15469 −685.1 at σ = 2, ℓ_r = 2 beams). If it does not, the baseline has
moved and the test stops there.

**Ladder, per target.** Δ_free = `free` − `paper` (a static error in position, flux or colour). Δ_motion = `motion` − `free`
(linear motion; `motion` − `paper` where there is no secondary). Δ_epoch = `epoch` − `motion` (night-to-night change beyond
linear motion). A step counts at Δ ≥ 5.

**Decision rule.** On one target the binary is confirmed as the cause of the corner if all three hold:
1. Δ_free ≥ 5 or Δ_motion ≥ 5;
2. under the better of `free` and `motion` the pick is inside the §8 bounds (ℓ_r ≥ 1 beam and σ ≤ 2);
3. if Δ_motion is the step that counts, the motion is orbital in size: the speed of the secondary relative to the primary
   (`bin_rel_v_mas_per_day`) is 0.3–2 times 2π·sep/P (AI Sco 0.0113, EN TrA 0.0076, HD 95767 0.0028 mas/day), and no binary
   parameter sits on a prior bound (`bound_warnings`). The velocity of either star against the disc is not used: it is
   measured no better than the reflex speed expected (0.0015–0.0038 mas/day).

`epoch` is a diagnostic and cannot confirm the binary. Δ_epoch ≥ 5 says the nights disagree beyond linear motion. Per-night
offsets fit calibration errors as well as motion, so that outcome is recorded as "night-level, origin open", and the night
arms decide it.

The claim "binarity explains the corner" needs confirmation on at least 2 of the 3 companion binaries, AI Sco, EN TrA and
HD 95767 (Ben, 18 Nov). PS Gem is judged on its own, on the wide field: rule 1 with Δ_motion (`motion` − `paper`, both wide),
rule 2, and a primary speed of 0.3–3 times 0.0038 mas/day, since sin i is unknown (simulated sd 0.0006–0.0017 mas/day). The
stock-field PS Gem runs are reported beside them, to separate the field from the reflex motion. IW Car is reported, not counted.
A null on EN TrA says little: its expected motion reaches only +3.1 in the simulations (a 6 % companion over 28 days).

**Night arms** (reported per target beside the binary verdict, so that "night-level, origin open" gets an answer):
* The night terms explain the corner on a target if `paper` under `nightw` beats `paper` under `current` by ≥ 5 in
  `log_evidence_scene` and its pick is inside the §8 bounds.
* The binary signal survives the night terms if `motion` − `paper` under `nightw` is ≥ 5 and passes rule 3. If Δ_motion counts
  under `current` but not under `nightw`, the motion was absorbing calibration errors.

**Controls.** IRAS 15469: Δ_motion < 5 and Δ_epoch < 5. Simulated static truths on the (u,v) of AI Sco, EN TrA, HD 95767,
PS Gem and IRAS 15469 give Δ_free −0.5 to −4, Δ_motion −5 to −10 and Δ_epoch −17 to +3 (one seed, one cell; review of
18 Nov), so a static binary does not pass. Ben accepted these local simulations in place of a simulation arm on the cluster.

**Spectral reading (Ben's question; reported, not part of the rule).** Under `free`, a fitted secondary index between −3.8 and
−2.0 reads as photospheric light (a stellar companion at 1.65 µm, −2.5 to −3.3 ± 0.5). This window does not depend on the
primary's SIND1. Within 1 of 0 or of the disc's index, it reads as circumcompanion or accretion dust. An index with a posterior
sd above 0.5 is "unconstrained" and is not read. On HD 95767, whose companion is 0.7 beam from the primary, the index correlates
with the disc's (0.4–0.8) and carries a systematic error of about 0.5. A dust-like companion and the disc share a colour, and
the grey disc image can trade flux with the companion.

**Verdicts.** Each target gets one line: Δ_free, Δ_motion, Δ_epoch with the pick of each mode; rule 1–3; the relative velocity
against 2π·sep/P; the colour; the two night-arm results. Then the overall outcome:
* Confirmed on ≥ 2 of 3: the binary scene replaces the §8 bounds on those targets, and B1 is rerun bounded with it for the figure.
* Rule 1 holds, rule 2 fails: the binary model is better but is not the cause of the corner. Keep it in the scene; the night
  arms say whether calibration is.
* Only Δ_epoch counts, and the `epoch` pick is inside the bounds: night-level inconsistency is the cause; the night arms say
  whether it is motion or calibration.
* Nothing counts: the binary as modelled is rejected as the cause. The §8 bounds stay, and the night arms are the answer.

## 13. §11 result (jobs 21112767 / 21112786, vv f41305f, virgil 6fa8afe): chromatic mismatch rejected

**Rule 1 fails (0 of 4 targets).** None of the 12 chromatic simulations picks the corner. The evidence picks ℓ_r = 2 beams and
σ = 2 on AI Sco, EN TrA and HD 95767 (one HD 95767 seed: 1 beam, σ = 1), next to the truth's best cell (2 beams), with image-error
rank fractions 0.03–0.14 on AI Sco and EN TrA and 0.36–0.47 on HD 95767. PS Gem picks 0.5 beam and σ = 1 in all three seeds
(rank 0.14–0.31). That is its known small-field failure (§8, §10 correction), not the corner. A radial colour gradient of
Δα = 2 across the ring, together with a stellar index wrong by 1, does not make the evidence buy sub-beam structure. The
model's grey disc absorbs it at the cost of a worse image (HD 95767), not of a rougher field.

**Rule 2 fails too (0 of 4).** Under `--chrom --unbounded` every real pick stays at or past the corner: σ = 4 on all six
targets, ℓ_r = 0.5 beam on AI Sco and 0.25 beam (one pixel) on EN TrA, HD 95767, PS Gem and IW Car. IRAS 15469 moves to 3.8 beams,
still at σ = 4.

| target | ℓ_r (beams) | σ | χ² | ring flux | ring Δα | star index − SIND1 | terms at a prior bound |
|---|---|---|---|---|---|---|---|
| AI Sco | 0.5 | 4 | 0.75 | 1.0 % | −22 | −0.50 | ring radius (floor), bg flux |
| EN TrA | 0.25 | 4 | 0.66 | 1.7 % | +4.0 | +0.40 | ring radius (floor), ring index and slope, bg flux |
| HD 95767 | 0.25 | 4 | 0.59 | 0.1 % | −26 | +2.00 | ring width, star index (+2 bound), bg index |
| PS Gem | 0.25 | 4 | 0.64 | 7.8 % | −4.8 | +1.90 | ring radius (field edge), ring slope, bg |
| IW Car | 0.25 | 4 | 0.66 | 1.0 % | −0.1 | +1.43 | ring index |
| IRAS 15469 | 3.8 | 4 | 0.71 | 9.8 % | −3.3 | −0.02 | none |

Rule 3 is moot. The chromatic terms are not physical: Δα of −22 and −26 come from rings 10–12 mas wide with slopes at their
bounds, the star index runs to its +2 bound on HD 95767, and two rings sit at the 0.5 mas floor inside the star's hole. The fits
use the extra freedom to fit noise (χ² 0.59–0.75, below the bounded B1 values of 0.99–1.31 in §10), not to model a colour gradient.

**The grid evidence rises (+7 to +57 over the unbounded grey picks of job 21054320), but that is not evidence for colour.**
`log_evidence` holds the scene parameters at their MAP (the binary plan review of 18 Nov; §12 and virgil#734), so the ring's
five parameters and the free star index lower χ² and pay nothing. §11's rule did not use this comparison, and it is recorded
here only so that it is not misread.

**Caveats.**
* The ring's radius prior has a 0.5 mas floor, inside the 1.22 mas star hole, and on AI Sco and EN TrA the ring starts there
  (Carbon review of #265). A ring there can trade flux and colour with the star. This affects only the real arm (rule 2),
  which fails in any case. The simulations (rule 1) do not fit a ring.
* The simulated noise is white, at the MacKay factor, as in stage C's `white` arm. The §10 confound (vlti white scaling) does
  not apply.
* Even and odd simulation sets are drawn with independent keys (Carbon review). That matters for CV, not for the evidence
  picks used here.

**Consequence (as pre-registered).** Rule 1 fails: a strong but physical colour error does not make the corner, so chromatic
mismatch is rejected as its cause. The §8 bounds stay. §11's last outcome no longer follows, since §10 did not reject the night
cause. The open causes are the binary (§12, stage BN, job 21123993) and the night terms with MacKay white errors (§12 night
arms, same job).

## 14. §12 result (jobs 21123990 / 21123993, vv 15b154a, virgil 6fa8afe): the binary does not explain the corner

All Δ below are `log_evidence_scene` (§12). The paper arm reproduces job 21054320's grid ln Z to 0.1 on all six targets.

**The ladder (white errors at the MacKay factor).**

| target | Δ_free | Δ_motion | Δ_epoch | better pick (ℓ_r beams, σ) | rel. speed / 2π·sep/P | secondary colour |
|---|---|---|---|---|---|---|
| AI Sco | +26.2 | +141.8 | +135.8 | motion: 0.25, 4 | 0.75 | dust-like (+0.54 ± 0.32) |
| EN TrA | −8.2 | +122.9 | +129.7 | motion: 0.25, 4 | 0.37 | dust-like |
| HD 95767 | +36.1 | +177.7 | +21.7 | motion: 0.25, 2 | 1.68 | neither (−4.36 ± 0.43) |
| IW Car (reported) | −4.4 | +88.9 | +49.6 | motion: 0.25, 2 | — | photospheric (−2.18 ± 0.44) |
| IRAS 15469 (control) | — | +5.7 | −6.8 | motion: 1, 2 | — | — |

* **Rule 1 holds on all three companion binaries** (Δ_motion ≥ 5; Δ_free also on AI Sco and HD 95767), and rule 3's speed
  window holds on all three.
* **Rule 2 fails on all three:** the better of free and motion still picks a sub-beam ℓ_r (0.25 beam, one pixel), with σ = 4 on
  AI Sco and EN TrA. **Confirmed on 0 of 3; the pass needs 2. The binary model fits better, but it does not cause the corner.**
* The motion gain is not orbital. The primary drifts against the disc at 0.025 mas/day on AI Sco, 0.021 on EN TrA and 0.008 on
  HD 95767, which is 7–30 times its reflex speed. The control also gains under motion (+5.7, failing its < 5 rule) with no
  companion. The epoch arm, a diagnostic only, gains a further +22 to +136 by giving each night its own star offset. Both say
  that the nights disagree with one static scene, which is the night-level error of §9, not an orbit.
* **PS Gem (15 mas field):** `paper` already picks inside the bounds (2 beams, σ = 0.5). Motion gains +25.6, but its pick returns
  to the corner (0.25 beam, σ = 1) and its primary speed is 5.6 times the window. It fails rule 2 and the speed rule. PS Gem's
  corner was the 5 mas field (§8), not the binary.

**Night arms (`nightw`: MacKay white errors plus 3 % per baseline per night and 2° per triangle per night).** The cost of the
bound is the grid's Δ ln Z between the unbounded pick and the best cell with ℓ_r ≥ 1 beam and σ ≤ 2. The ≤ 5 criterion and
the HD 95767 night + motion test (cost ≤ 5, no concentric rings, speed 0.3–2 times 0.0028 mas/day, the two stars moving in
opposite senses) were fixed in virgil-paper `reports/postagb-next-steps.md` after the interim scores and before AI Sco's night
arms and HD 95767's night + motion were read. They refine §12's "inside" rule and are labelled as such. χ²/pt is the final fit's
value on the scaled errors and, after it, on the quoted errors (× s²; the night terms are still whitened).

| run | nightw − current paper | pick (ℓ_r, ℓ_θ, σ) | cost of the bound | motion − paper | χ²/pt scaled → quoted | image |
|---|---|---|---|---|---|---|
| AI Sco, paper | +300.4 | 0.25 b, 1, 4 | 9.6 | — | 0.43 → 0.77 | one lobe 2 mas E of the star plus blobs; no disc |
| AI Sco, motion | — | 0.25 b, 0.25, 4 | 13.4 | +27.0 | 0.38 → 0.68 | one compact clump plus blobs |
| EN TrA, paper | +246.2 | 0.5 b, 2, 2 | 0.1 | — | 0.33 → 0.54 | arc |
| EN TrA, motion | — | 2 b, 2, 2 | 0 | −0.6 | 0.32 → 0.52 | arc |
| HD 95767, paper | +187.7 | 0.25 b, 0.5, 2 | 19.6 | — | 0.44 → 0.40 | thin concentric rings (ringing) |
| HD 95767, motion | — | 0.25 b, 1, 2 | 5.3 | +24.2 | 0.43 → 0.39 | ring with bright NE rim, one faint outer arc |
| IW Car, paper | +175.5 | 1 b, 0.25, 2 | 0 | — | 0.47 → 0.47 | ring |
| IW Car, motion | — | 2 b, 1, 2 | 0 | +22.2 | 0.47 → 0.47 | ring |
| IRAS 15469, paper | +27.6 | 4 b, 2, 4 | 0.7 | — | 0.59 → 0.39 | crescent (as every arm) |
| IRAS 15469, motion | — | 1 b, 2, 2 | 0 | −3.2 | 0.58 → 0.38 | crescent |
| PS Gem 5 mas, paper | +108.3 | 8 b, 0.25, 0.25 | 0 | −0.5 | 0.39 | **degenerate** (flat saturated disc filling the field): not counted |

* **AI Sco:** the night terms cut the cost of the bound from 58.1 to 9.6, but not to ≤ 5 (the expectation was 10–30). Under night
  terms motion gains +27.0, but at a relative speed of 0.21 times the orbital rate (below the 0.3 floor; sd 0.0055 mas/day, so
  consistent with zero), while the primary still drifts at 0.023 mas/day. The binary does not survive the night terms, and AI Sco
  is left for step 3 of the next-steps plan.
* **HD 95767:** night + motion costs 5.3, a marginal fail. The speed is in the window (1.75 times), and the rings of the night
  paper image are mostly gone, but the two stars do not move in opposite senses: the primary moves at (−0.0037, +0.0008) and the
  secondary at (−0.0001, −0.0025) mas/day, at about 100° to each other, with the primary's speed only 1.5 sd from zero. It fails
  the pre-fixed test, as expected, and goes to "data-limited".
* **EN TrA and IW Car:** the night terms bring the cost to ≤ 0.1 (EN TrA's sub-beam pick is a tie). IW Car's motion gain (+22.2,
  pick inside, speed 1.5 times orbital) is the only binary signal that survives the night terms. It is doubtful: the primary drifts
  at 0.020 mas/day against the disc (about 5 times any reflex), while the "secondary" stays almost fixed in the disc frame. That
  reads as a disc clump plus a frame drift (next-steps report). IW Car is reported, not counted.
* **The night arms count the noise twice.** `nightw` keeps the MacKay factor from stage A, which was fitted without night terms
  and so already absorbs night-level scatter. χ²/pt at the picks falls to 0.32–0.59 on the scaled errors, so part of the
  smoothing, and of the +28 to +300, may come from errors that are now too large. The noise terms have not been fitted together.

**Post-hoc amendment (after looking at the images; not in §12).** A pick inside the bounds counts only if its image is not
degenerate, meaning it does not fill or saturate the field. This removes PS Gem's stock-field night arm, whose +108.3 and pick
inside the bounds come from a flat disc filling the 5 mas field. The PS Gem 15 mas `paper` image is a smooth resolved disc with
arms, extending to about 8 mas. (Corrected in §15: an earlier version said ORGANIC's emission was compact, within about 2 mas.
That was read off a peak-scaled plot. ORGANIC's own image has 58 % of its disc flux beyond 5 mas, and the 15 mas image matches
its radial profile.)

**Consequence.** §12 fails: the binary, freed in position, flux, colour and motion, does not move the pick out of the corner on
any of the three companion binaries, and its motion gain is a night-level error, not an orbit. Under night terms the bound costs
≤ 5 on EN TrA, IW Car and the control, 5.3 on HD 95767, 9.6 on AI Sco and 19.6 on HD 95767 without motion. The night-level
inconsistency is the leading cause, with the doubled-noise caveat open. The §8 bounds stay the selector, with the cost of the
bound quoted. Fitting the noise terms together is the next test (next-steps step 1); it is not pre-registered here.

**Evidence across error scales (added 18 Nov, after the Fable review of the parametric-disc plan).** virgil's `_log_norm` is
½ ln det of the marginalised gain and offset factors only. It leaves out −Σ ln σᵢ, so ln Z compared across different white
error scales favours the larger scale. Nothing in this section is affected: the grid picks, the ladder, nightw − current and the
cost of the bound all hold the white errors fixed (the MacKay factor of stage A), and the night terms enter through the gain
factors that `_log_norm` does carry. §10's "ln Z rises by 240–300" is affected only if it was measured against the vlti-scaled
§10 runs (white errors 2–4 times smaller), and that has not been checked. A scan of the white scale by evidence (next-steps
stage F1) needs the term added first.

## 15. PS Gem's support (18 Nov, Ben: "there's something wrong here. Think about it carefully and fix it.")

**The fault.** PS Gem was imaged in a hand-set 5 mas support (`fov_override=11`, half, comparison and mask radii of 5 mas), set
in October from the belief that its emission lies within about 5 mas. Every other target uses the paper's field (14–25 mas). The
data see further than 5 mas: a ring at radius r has visibility J0(2π B r / λ) on a baseline B. On PS Gem's shortest baseline
(9.9 m), that is above 0.5 out to r = 8.9 mas, and virgil's own `field_of_view(data)` is 36.6 mas across. Emission outside the
support can only go to the over-resolved background, whose visibility is 0 on every baseline. The numbers:

| run (job 21123993 unless stated) | support | disc flux | background | flux in the band at the field's edge | flat fraction |
|---|---|---|---|---|---|
| current noise, `paper` | 5 mas | 4.0 % | 7.6 % | 0.59 | 0.00 |
| current noise, `motion` | 5 mas | 5.5 % | 6.1 % | 0.54 | 0.02 |
| night terms, `paper` | 5 mas | 1.6 % | 9.8 % | 0.18 | **0.97** |
| night terms, `motion` | 5 mas | 2.4 % | 9.1 % | 0.48 | 0.01 |
| current noise, `paper`, `--wide-field` | 15 mas | 11.0 % | 0.7 % | 0.03 | 0.04 |
| current noise, `paper` (local, after the fix) | 18.9 mas | 11.8 % | 0.1 % | 0.01 | 0.00 |
| night terms, `paper` (local, after the fix) | 18.9 mas | 8.8 % | 2.7 % | 0.10 | 0.03 |

(Band: the flux within 0.15 of the field's half-width of its boundary, 22–27 % of its area; the gp-polar field ends on an
inclined ellipse. The flat fraction is the share of the field above half the image's peak. The other 35 gp-polar BN images have
band ≤ 0.20 and flat ≤ 0.04.)

* The 5 mas support truncated the disc in every arm, not only under night terms. With room the disc takes 11–12 % of the flux and
  the background falls to its lower bound. On the same data and errors (s = 1.059), the 15 mas field beats the 5 mas field by
  117 in grid ln Z (−271.6 against −388.3).
* ORGANIC agrees with the wide field. Its image has 86 % of the disc flux beyond 2 mas, 58 % beyond 5 mas and 31 % beyond 8 mas.
  The 18.9 mas current-noise image has 93 %, 63 % and 30 %, while the 5 mas one had 0 % beyond 5 mas by construction.
* Under night terms the truncated disc fell to 1.6 % with its index at the +6 bound. The night gains (3 % per baseline and night)
  are as large as such a disc's whole V² signature, so the evidence chose the smoothest cell (σ = 0.25, ℓ_r 8 beams) and the image
  was the prior's flat shape. It is not a plotting artefact: the flatness is measured on the brightness array. The peak-scaled
  colour map only made a 1.6 % plateau look saturated.
* The background is not the cause. Without a background component (local test, same errors) the 18.9 mas fits are unchanged:
  current noise 11.9 % disc, the same radial profile, ln Z_scene −397.3 against −399.0 with it; night terms 11.6 % disc, −389.2
  against −387.1.
* MacKay factor: the pilot GP on the 18.9 mas field gives s = 0.861, against 1.059 on the 5 mas field. The truncated pilot left
  structure in the residuals, which the factor absorbed. So ln Z of the new PS Gem runs must not be compared with the old ones.

**The fix (postagb_imaging).**
* PS Gem takes the paper's field like the other targets: support radius 18.9 mas; the hole stays at the UD (0.37 mas).
* `check_support` refuses a support radius below `support_floor_mas`: the radius at which a ring is at visibility 0.5 on the
  shortest baseline at the longest wavelength. Floors: 8.3–11.0 mas; every other target's support (14–25 mas) passes.
* `image_degeneracy` writes `img_flat_frac`, `img_edge_frac` and `image_degenerate` (the reasons) into every image JSON. An image
  is degenerate if more than 0.5 of its field lies above half its peak (flat), or if more than 0.25 of its flux lies in the band
  at the field's boundary (edge-piled). A degenerate run is printed as such, never reused as a warm start or a converged result,
  and is not scored. On the 40 BN gp-polar images it flags exactly the five 5 mas PS Gem runs.

**After the fix** (local, virgil d0f7086, MacKay factor refitted; figure `design/postagb_noise/ps_gem_support_before_after.png`):

| run | pick (ℓ_r, ℓ_θ, σ) | cost of the bound | χ²/pt | image |
|---|---|---|---|---|
| current noise | 0.25 beam, 0.25, 2 | 4.5 | 0.64 | arcs and spiral-like rings to about 12 mas; not degenerate |
| night terms | 0.5 beam, 0.25, 1 | 0.5 | 0.51 | the same arcs on a faint plateau that fills the field (31 % of the flux beyond 12 mas, ORGANIC 16 %); not degenerate |

So on the right field PS Gem is no longer a corner case: the cost of the bound is 4.5 under current noise and 0.5 under night
terms, both ≤ 5. The night-terms image is more extended than ORGANIC's, and the per-night gains may be standing in for a faint
halo. That belongs to the noise fit (stage F1), not to the field.

**What this changes.** Every earlier PS Gem gp-polar result used the truncated field: §7's calibration notes, §10, §13 (rule 2's
PS Gem row), §12's 5 mas arms and §14's PS Gem rows. §12's verdict for PS Gem already rested on the 15 mas field and stands.
The BN outputs on disk keep their 5 mas images; a rerun with the fix is a cluster job, not done here.

**A second check: the polar field's own radius (AI Sco).** gp-polar's field ends at `half_mas` in the disc plane (the figure
extent), not at the support. For AI Sco that is 8.47 mas at inclination 63.5°, only 3.8 mas on the sky along the minor axis, below
the data's 10.7 mas floor (HD 95767: 9.31 mas, floor 10.2). AI Sco also has the highest edge band outside PS Gem (0.10–0.20). A
local test widened AI Sco's field to cover the whole 14 mas support (r_out = 31.3 mas in the disc plane), with stage BN's MacKay
factor (1.338), so its ln Z compares directly with job 21123993:

| AI Sco | ln Z_scene, 8.47 mas field (BN) | ln Z_scene, field covering the support | pick | cost of the bound | edge band |
|---|---|---|---|---|---|
| current noise | −989.4 | −931.1 (+58.3) | 0.25 beam, σ = 4 | 58.1 → 83.0 | 0.10 → 0.03 |
| night terms | −689.0 | −689.1 (0.0) | 0.25 beam, σ = 4 | 9.6 → 10.6 | 0.14 → 0.00 |

The wider field fits the white-error data better by 58 and removes the edge flux, but the pick stays in the corner and the cost
of the bound does not fall. Under night terms nothing changes. So AI Sco's field was short, but that is not why it picks the
corner. Main has since made gp-polar's field cover the support in every target (8dd4095, `fit_radius_mas` = mask_mas / cos i),
which is what this test did; this PR takes that change in its merge of main.
