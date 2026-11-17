# gp-polar hyperparameters: why four discs break into dots, and how to choose them

Status: the azimuthal cap and cold, logged grid cells are in place (`scripts/postagb_imaging.py`). How to pick σ and ℓ_r waits on the
calibration run (stage C of `postagb_noise`, §4). This note replaces the floors of PR #264.

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
  cells of one cap compare fairly. It needs no virgil change; tests in `tests/test_postagb_imaging.py`.
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

`postagb_imaging.py calibrate --arm {white,corr,smear} --m-max {-1,3}` runs, for each of AI Sco, EN TrA, HD 95767, PS Gem and the
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
* **Grid.** A fixed grid with no extension: ℓ_r ∈ {0.5, 1, 2} beams, ℓ_θ ∈ {0.0625, 0.25, 1} rad and σ ∈ {0.5, 1, 2, 4}, every
  cell cold. It is run uncapped (`-1`) and capped (m ≤ 3). Each cell is also fitted on the even-numbered files and scored by
  χ²/pt on the odd ones (cross-validation over exposures, which respects per-frame correlation).
* **Scoring.** Each cell's error against the truth: relative rms of the unit-sum images, raw and convolved with the comparison
  beam, plus NCC. For three selectors (evidence, CV and the truth's own best), the run reports the picked cell and its rank
  against the truth. Figure: truth, the three picks, and ln Z against image error.

**Decision rule** (capped arm). The evidence passes on an arm if, on all six targets, its pick ranks among the best five of the
36 cells by beam-convolved error and is not at the σ = 4 / ℓ_r = 0.5 beam corner.

| white | corr | smear | conclusion | change |
|---|---|---|---|---|
| pass | pass | pass | the cap was the fix; the real-data picks reflect the data | keep the evidence; report edge picks |
| pass | fail | – | correlated channel errors drive the pick (item 1) | select by CV over exposures (implemented in `calibrate`; move into `fit_polar`) |
| pass | – | fail | missing smearing creates structure (item 4) | needs smearing for pixel Images in virgil (separate issue); meanwhile CV |
| fail | – | – | the Laplace selector is biased under the exp link (item 2) | CV, or a fixed prior (σ = 1, ℓ_r = 1 beam, ℓ_θ = 1 rad) if CV also fails; whichever recovers the truth best |

The uncapped arm shows whether the cap alone changes the evidence's preferences. Comparing it with the real uncapped picks shows
whether the old warm-started chain contributed.

## 5. Runs

Submit blocks are in the PR. They chain stage A (`--array=0,4,5,6,7,8`, which rebuilds the scale files in this commit's
`out/<pins>/`) into stage C (`--array=0-35`) and into the B1 rerun of gp-polar with the cap (`--array=66,80,94,122,10,108`). If
the calibration changes the selector, B1 is run again from that commit.

## 6. PR #264

Superseded. Its ℓ_θ ≥ 0.5 rad floor leaves 17 % of the azimuthal variance above m = 3 (table in §3). Its ℓ_r ≥ 1 beam and σ ≤ 2 bounds
are untested guesses until stage C reports; they come back only if the calibration selects them.
