# Why the contest reconstructions do not converge

A diagnosis of the stalled and poor image fits in the imaging-contest work
(`docs/plan_imaging_contests.md`, Stage C2). The evidence comes from our own
OzSTAR runs on virgil `b73ff6a`, `98eaf86` and `02aad5e` (results in
`~/data/imaging_contests/results/<commit>/`). The methods come from the
contest papers and the winning codes' own method papers. Only **methods**
from those papers are used here, never their results.

## Summary

- **Initialisation and parameterisation are the main cause.** The same
  data that stall from a Gaussian or flat start in the per-pixel
  maximum-entropy (MEM) basis converge to χ²/N ≈ 0.7–1.1 from a CLEAN start
  in a whitened Gaussian-process (GP) basis. That covers 2008 AGB J and
  AGN J, H and K, 2022 GRAVITY (without a star) and 2024 Obj1.
- **Two likelihood bugs added to it, both now fixed.** F11 (correlated
  closure-phase χ² jumped at ±π) and F13 (OIFITS v1 snapshots merged into
  one frame). A third discontinuity is not yet explained (below).
- **Hyperparameters are not the bottleneck.** Evidence picks sensible GP
  hyperparameters, except where the error bars are wrong.
- **Two datasets need what the contestants needed:**
  - 2010 needs per-channel imaging with a grey default (its V² errors are
    1e-4, and the source is chromatic);
  - 2024 needs per-band images.

## 1. Optimiser and conditioning

- **MEM in pixel log-brightness is badly conditioned on high
  signal-to-noise data.** L-BFGS stops with "line search ran out of float64
  precision" in nearly every fit (6–15 per task), good fits included. On
  the stuck datasets the first, strongest-weight fit never leaves its
  start, and warm starts carry that to every weight. χ² is then flat
  across the L-curve:
  - 2008 AGB J 6.99, AGN J 6.24, AGN K 14.1;
  - 2022 GRAVITY 457;
  - 2024 Obj2 GRAVITY 182.
- **Adam escapes where L-BFGS cannot.** On 2022 GRAVITY at w = 1e4, Adam
  reaches χ²/N 2.55 in 20 000 steps; L-BFGS stays at 467 (job 18061164,
  case 2). The landscape has descent that L-BFGS's line search cannot
  follow.
- **The GP basis is a preconditioner.** Standard-normal latents with a
  smooth spectrum whiten the image prior. In that basis the same data
  converge: 2008 AGB J goes from MEM 6.99 to GP 0.70, and 2022 GRAVITY
  without a star from 457 to 0.93.
- **This is Stage 7's known MEM problem.** virgil#226 (LM in each prior's
  flat coordinate) now gives LM for the parametric parts. MEM in pixels is
  still not a least-squares problem.

## 2. Objective and likelihood

- **F11** (fixed, virgil#174). Correlated closure-phase chords flipped
  sign at Δ = ±π, so χ² jumped and fits stalled there.
- **F13** (fixed, virgil#203). OIFITS v1 snapshots told apart by `TIME`
  were merged into one frame. The 2004 files whitened 130 closure phases
  to 10, and #174's per-phase penalty then fought the under-constrained
  whitened part:
  - before #203: 2004 data1 χ²/N 24 and data2 121;
  - after #203: data1 0.87 MEM, 0.90 GP.
- **Open: a remaining discontinuity in the MEM objective** on 6- and
  4-telescope data. At the MEM stall on virgil `02aad5e` (job 18061164),
  JAX gradients disagree with finite differences, the finite difference
  scales as 1/ε, and the loss rises along −∇ for every step (2008 AGB J
  by +10, AGN K by +8700, 2022 GRAVITY by +845). That is the signature of
  a jump in the loss. It is **not** the closure-phase chord (#174 removed
  that), and it does not show in the GP basis at the same data's solution.
  Candidates still to separate: the MEM regulariser at the image support's
  edge, and the analytic star's flux bounds.
  - **Partial check: Obj2 GRAVITY has no jump.** 2024 Obj2 GRAVITY, also
    4T, shows AD = FD to 1e-8 and descent along −∇. It is slow, not
    trapped.
- **Error bars.** χ²/N ≪ 1 for 2022 AMI GP (0.02–0.09) means the quoted
  closure-phase errors (0.001–0.002°) or the GP grid's smallest lengths
  over-fit. Conversely, 2010's V² errors of exactly 1e-4 on every point,
  against a chromatic source, make any grey image χ²/N ~ 10⁴. The contest
  papers report the same issues: the 2014 and 2018 entrants inflated
  errors 1.3–1.4× and cut low-SNR points, and the 2024 organisers warned
  "Noisy data for Obj1? That's expected!".

## 3. Regularisation and hyperparameters

- **GP evidence works.** It picks σ ∈ {1, 2, 4} and ℓ ∈ {0.5, 1, 2} beams
  sensibly. It prefers anisotropic fields for elongated targets: 2006,
  2008 AGB J and H, 2022 GRAVITY, 2024 Obj1 GRAVITY.
- **GP evidence answers the star question from the data:**
  - star clearly preferred: 2024 Obj1 (+48 to +50), 2008 AGN H (+57),
    2022 AMI (+24);
  - no star clearly preferred: 2006 (−75), 2022 GRAVITY (−2.6e5).
- **Evidence assumes correct error bars**, which several contest datasets
  violate on purpose (calibration biases in 2018 and 2024). The winners
  set hyperparameters by an L-curve when the data had systematics (2018
  BSMEM), and treated automatic α as unreliable then. virgil's
  `error_scale` (MacKay's fixed point) re-estimates the error scale from a
  GP fit, and is the principled check.

## 4. Initialisation and multimodality

- **Start matters more than anything else.** It does in our runs, and in
  the papers:
  - MiRA's method paper warns its result depends on the starting image.
  - The 2012 entrants ran 300 random starts and averaged the good ones.
  - The 2018 winner ran 10 SQUEEZE chains, refined each and averaged.
  - The 2022–24 Millour/Drevon entries used thousands of randomised MiRA
    runs (PYRA/MYTHRA): an L-curve window, χ² filtering, then mean and σ
    maps.
- **Our runs agree.** A single CLEAN start fixes most stalls, but 2004
  data2 is still χ²/N 20.7 (GP) / 172 (MEM). It is a spotted elliptical
  star plus a compact component 10 mas away: multimodal and badly
  conditioned. One start is not enough.
- **Inversion.** V² cannot tell an image from its inversion through the
  origin. Closure phases can, but only to the extent their S/N allows, so
  multimodality is intrinsic. `flip_dchi2` is reported for every image.

## What follows: the campaign

The winners, taken together, did five things:
1. located features first (CLEAN, SQUEEZE or a model fit), then refined with a smooth regulariser;
2. ran many starts and averaged the images that fit;
3. set hyperparameters by evidence, or by an L-curve when systematics were present;
4. treated unresolved stars and companions parametrically;
5. imaged multi-channel data per band, starting from a grey default.

The OzSTAR campaign in `scripts/contest_images.py` (`--member k`, one array task per member) does the same:
- an **ensemble of CLEAN-started GP fits** per dataset, randomising the
  start (CLEAN gain, field 1–2× and pixel scale) and the star or no-star
  choice;
- hyperparameters by evidence within each member;
- an **error-scale check** (`error_scale`) with a refit when it is far
  from 1;
- a **combiner** that keeps the members within a χ² and evidence window
  and writes mean and σ images (PYRA-style).

It runs for every dataset with pre-submission information only, including
the 2024 MATISSE L and N bands, which no earlier run used.
