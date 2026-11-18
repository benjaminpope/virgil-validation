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

**Consequence (as pre-registered).** The night errors are not the cause. The §8 bounds stay and the chromatic test is next.
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
