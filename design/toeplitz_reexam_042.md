# Toeplitz χ² re-examination (virgil 0.4.2 gate G2): pre-registered criteria

Machine-readable criteria: `design/toeplitz_reexam_042.yml`. `scripts/toeplitz_reexam.py` hashes the file
and refuses to run unless the hash equals its `REGISTERED_SHA256`. Every output records that hash and the
virgil commit. This note and the criteria were written **before any run under them**, but **after** both old
A100 runs had been read. Section 8 says what that means.

## 1. Why a re-examination

Virgil's Toeplitz (dirty-beam) χ² (`ComplexVisData.toeplitz`, the term `chi2="auto"` picks when it is exact)
was checked on OzSTAR against the direct NUFFT term at IM Lup's R_mpol solution
(`scripts/alma_imlup_toeplitz_check.py`). The two runs, read from their own outputs:

| Run | vv commit | Printed verdict | Failing rows |
|---|---|---|---|
| 19063376 (gina17) | 157a941 | NOT ACCEPTED | gradient, worst site, relative to its max: 2.07e-6 against ≤ 1e-8 (`cal.sigma`); GPU peak, worst of the two paths: 21.92 GB against < 20 GB |
| 19078265 (gina18) | 5ece518 | NOT ACCEPTED | speed-up 16× against ≥ 30× |

Between the runs, two criteria changed (5ece518): the gradient moved to a perturbed point, and the memory limit
applied to the Toeplitz path only. After the second run, a49802a made the speed-up report-only on any GPU other
than an A100. Neither run recorded its GPU model in its outputs. The old job's README attributes 34× to an A100 and 16× to an
RTX PRO 6000 Blackwell, which has weak FP64. No run passed every criterion: each one passed on some run,
but never all on the same run. Because criteria changed after results were seen, Ben made the acceptance a 0.4.2
gate (G2) on 2026-10-28, to be re-examined from first principles.

## 2. What is tested

The test compares virgil's Toeplitz term with a reference. It covers:

- every **χ²_d** (per ddid, unit error scales);
- the **loss** in each error-scale mode;
- the **gradient** at every fitted site: log-brightness η, flux, gains, offsets, and σ in the fitted mode.

The loss conventions match virgil:

- fixed: ½Σχ²_d;
- fitted: Σ[χ²_d/(2s_d²) + 2N_d ln s_d];
- profiled: Σ N_d(1 + ln(χ²_d / 2N_d)).

There are two tiers.

- **Tier I (independent).** Virgil against `crosscheck.radio_exact`, a numpy-only direct sum that imports nothing
  from virgil:
  - each χ²_d is an exactly rounded `math.fsum`;
  - the gradients are analytic;
  - the result is checked against central differences in the tests.

  It runs on four held-out synthetic scenes (section 5), in all three modes and at all points (section 4).
- **Tier II (self-consistency at scale).** At IM Lup R_mpol (42,421,131 points, 1028² × 5 mas, free offsets),
  virgil's Toeplitz term is compared with virgil's direct NUFFT term at eps 1e-12, in the fitted mode, at the
  three points. Tier II shares virgil's model code, so it cannot catch an error common to both paths. Tier I is
  the part that can.

  Tier II also gates the following on an A100:
  - Nyquist fraction < margin;
  - speed-up ≥ 30× per value+grad, against the direct term at eps 1e-6 (the fit's eps);
  - GPU peak of the Toeplitz path < 20 GB.

## 3. Tolerances, derived

Every tolerance is an **absolute worst-case bound** on |virgil − reference|. It is derived from float64
round-off (u = 2⁻⁵³) and the documented tolerances of the two methods. Nothing is a chosen number or a safety
factor. Code: `radio_exact.tolerances`.

The pieces of the bound:

- **Sums.** A sum of n terms in an unknown order has relative error at most γ_n = nu/(1 − nu) of Σ|terms|
  (Higham 2002, *Accuracy and Stability of Numerical Algorithms*, Lemma 3.1, eq. 4.4). The order is unknown
  because GPU scatter-adds and XLA reductions do not fix it. The three reductions of the Toeplitz path run one after another (the data sum over N_d, the image
  contraction with the dirty image, and with the convolved beam), and sequential sums add, γ_a + γ_b ≤ γ_{a+b}
  (Higham 2002, Lemma 3.3). Take n = N_d + 2P. The FFT is counted in ρ below, not in n. The bound applies to
  both sides, hence 2γ_n.
- **FFT convolution.** A radix-2 FFT of n points has relative l2 error log₂n·η/(1 − log₂n·η), with η = u + γ₄(√2 + u)
  (Higham 2002, Thm 24.2). The beam convolution uses three FFTs. Its error, contracted with a unit-sum image, is
  at most ρ = 3·fft_factor(n_fft)·√n_fft relative to Σw.
- **NUFFT kernels.** The implementation's documented tolerance eps_impl is relative to the l1 norm of the strengths
  (Barnett, Magland & af Klinteberg 2019, SIAM J. Sci. Comput. 41, C479). For the Toeplitz term this is the kernel
  eps 1e-14, plus the ψ̂ quadrature error 4e-14 with spread offsets. Both figures are virgil's documented values
  (`design/toeplitz_likelihood.md` §3).
- **Reference.** One evaluated phase term exp(−iφ) has relative error ≤ (3φ_max + 2)u. The image sum adds γ_P.
  Together they give eps_ref (`radio_exact.phase_error`). In Tier II the reference is the direct NUFFT, so
  eps_ref = 1e-12.

Per ddid, with g_d = flux × gain of the ddid's observation, S_d = 2g_dΣw|V| + g_d²Σw and C_d = Σw|V|²:

    τ_d = (eps_impl + 2 eps_ref + 2γ_n + ρ) S_d + 2γ_n C_d

How τ_d bounds each quantity:

- **χ²_d:** τ_d directly.
- **Loss:** Σ|∂L/∂χ²_d|·τ_d + u|L|. In the profiled mode the factor is evaluated to first order at χ²_d ± τ_d.
- **Gradients:** the same weighted sum, times the derivative's factor, as an l2 norm per site:
  - 2/F for flux;
  - 2/g_e for gains;
  - 2·2π·MAS·q_max for offsets;
  - 4‖b‖₂ for η, with b = softmax(η). Each pixel's error in ∂L/∂b is at most 2Σwt (l∞). The softmax Jacobian
    J = diag(b) − bbᵀ gives ‖Jδ‖₂ ≤ ‖b∘δ‖₂ + ‖b‖₂|bᵀδ| ≤ 2‖b‖₂‖δ‖∞, because Σb = 1 (‖x∘y‖₂ ≤ ‖x‖₂‖y‖∞, and
    Cauchy–Schwarz; Horn & Johnson 2013, *Matrix Analysis*, §5.6). A unit-sum blur has l2 norm ≤ 1 and does not raise it.
- **σ (fitted mode):** τ_d/s³ + u(χ²_d/s³ + 2N_d/s).

Gating on worst-case accumulation is a judgement call, made after the old runs were seen; section 8 flags it.

Not gated, only reported:

- the probabilistic bound λ√n·u, with λ = √(2 ln(2n/δ)) and δ = 1e-6 (Higham & Mary 2019, SIAM J. Sci. Comput. 41,
  A2815), which holds for random-like rounding errors;
- each quantity's relative error against the old 1e-10.

## 4. Points, and why the point no longer matters for the threshold

The original gradient criterion was relative: |Δ∇|/max|∇| ≤ 1e-8. At the optimum ∂L/∂σ ≈ 0 by construction, so
that ratio divides round-off by a near-zero number. The 2.07e-6 failure was this effect. The absolute error in
`cal.sigma` was ≈ 4.4e-4 at the optimum and ≈ 3.2e-4 at the moved point, about the same; only the denominator
changed. 5ece518 moved the point so that the denominator grew. That changed the ratio, not the error.

With absolute, derived tolerances (section 3), the threshold does not depend on the gradient's size, so it no
longer matters where the gradient is evaluated. All three points are gated:

- **optimum:** the truth for synthetic scenes, the saved solution for R_mpol, with fitted σ at
  ŝ_d² = χ²_d/2N_d. This is where a fit stops, so it matters most, and it is the cancellation case the original
  criterion stumbled on.
- **moved:** η + 0.3z, flux × 1.15, gains × (1 + 0.05z), offsets uniform in ±½ bound, σ × 1.1. These are generic
  values, with no cancellation.
- **edge:** spread offsets only. The moved point with every offset component at ±0.9 bound, the largest phase
  ramps the spread margin allows.

These are the Tier I definitions (`toeplitz_scenes.points`). For R_mpol, the moved and edge points are those of
`alma_imlup_toeplitz_check.moved_point` (552da46): σ × 1.1, gains × 1.05, offsets ±2 mas; edge at ±0.9 of
`max_offset_mas`.

## 5. Held-out scenes and seeds

None of the four Tier I scenes was used in developing virgil's Toeplitz term or its tests. Those used log-normal
images on uniform uv discs, plus R_mpol.

| Scene | Image | Grid | uv | Offsets | What it stresses |
|---|---|---|---|---|---|
| S1_ringed_disc | inclined disc, 3 rings, 2 gaps | 128² × 4 mas | 40 ant × 64 times, 3 obs, 4 ddid, reach 0.6 Nyquist | spread, ±15 mas | sharp radial structure, spread offsets |
| S2_sources_on_halo | 5 point sources over 1e4 in brightness, one in a corner pixel, on a 1e-4 halo | 97² (odd) × 3 mas | 36 × 60, 2 obs, 4 ddid, reach 1.4 (beyond Nyquist) | none | dynamic range, odd grid, aliasing, longest lags |
| S3_edge_filled | log-normal field filling every pixel | 256² × 2 mas | 50 × 120, 3 obs, 6 ddid, reach 0.9 | spread, ±10 mas | every lag of the beam in use |
| S3_edge_filled_no_offsets | as S3 | as S3 | as S3 | none | the no-offset (integer-lag) Toeplitz path at scale, built with `data.toeplitz`; that `chi2="auto"` selects it is not tested here |

The uv tracks are Earth-rotation tracks over 8 h at dec −30°, from log-uniform antenna radii. The weights are
10^N(3, 0.3). The noise variance is s_d²/w, with s_d log-uniform in 10^±0.15.

Every random draw uses the seed sha256("toeplitz-reexam-042/<scene>/<role>")[:8]. The role is truth, moved or
edge. No seed is chosen by trial.

## 6. Verdict

- **ACCEPTED:** every gated row passes on **one run of one virgil commit**.
- **NOT ACCEPTED:** any gated row fails. NaN counts as a failure.
- **INCOMPLETE:** some gated row is NOT EVALUATED and none fails.

Rows from different runs or commits are never combined. A device whose JAX `device_kind` lacks the token `A100`
(matched as a whole word, so an A1000 does not count) leaves the speed-up and memory rows NOT EVALUATED. The
verdict is then INCOMPLETE, not ACCEPTED.

The speed-up limit is tied to the A100 because the 30× figure was set for that GPU
(`design/toeplitz_likelihood.md` §7). A GPU with weak FP64 (e.g. a consumer or Blackwell part) changes the
ratio for reasons unrelated to the algorithm. That is a caveat to report, not a pass.

The memory limit applies to the Toeplitz path because the 20 GB budget is the fit's budget. The direct term is a
reference that a Toeplitz fit never runs, and its peak is reported. This is the one memory change from the
original wording; the change log marks it as made after a result.

## 7. What the old evidence supports

Measured against the original registered criteria (`design/toeplitz_likelihood.md` §7, as the check script
encoded them in 99d5c7c), **the first run failed**:

- the gradient row failed at the optimum (relative criterion);
- the memory row failed, worst of the two paths;
- the speed-up row passed, at 34×.

**The second run also failed**, on the speed-up: 16× on a GPU the outputs do not name. **No run ever met every criterion at
once.**

Measured against these criteria, the old evidence is **INCOMPLETE**, never ACCEPTED:

- Tier I, the independent reference, and the edge point were never run.
- The GPU model was never recorded, so the A100-only rows cannot be evaluated.
- Whether the old optimum σ-gradient error (≈ 4.4e-4) lies inside the worst-case budget cannot be decided from the
  saved outputs, which lack per-ddid S_d and C_d. A rough estimate (2γ_n·C_d with n ≈ 4.2e6, s ≈ 1) puts it
  probably inside.
- Under the probabilistic bound it would fail by about 100×. That suggests worst-case-like accumulation, e.g.
  GPU atomic adds.

The new run must stand on its own.

## 8. Change log

| Date (AEDT) | Commit / file | Change | Changed after seeing a result? |
|---|---|---|---|
| before 2026-10-23 | virgil `design/toeplitz_likelihood.md` §7 | Original acceptance: χ² rel ≤ 1e-9; loss Δ ≲ 1e-2 abs; gradient ≤ 1e-8 rel; ≥ 30× speed-up; memory < 20 GB; a cold-start fit to the same loss | no |
| 2026-10-23 11:55 | vv 99d5c7c | Check script; χ² tightened to ≤ 1e-10 rel; perturbed-image loss Δ ≤ 1e-6 rel / 1e-2 abs; memory = worst of the two paths; the cold-start fit not encoded (it stayed with the IM Lup fits) | no |
| 2026-10-23 12:02 | vv a359725 | Rows from each path's own outputs, NaN fails, direct timed at eps 1e-6 | no |
| 2026-10-23 13:43 | vv 157a941 | Records virgil's import path; no criteria change | no |
| 2026-10-23 14:51 | run 19063376 | NOT ACCEPTED (gradient 2.07e-6; memory 21.92 GB) | — |
| 2026-10-23 14:59 | vv 5ece518 | Gradient judged at a moved point; memory limit on the Toeplitz path only | **yes** |
| 2026-10-23 22:03 | run 19078265 | NOT ACCEPTED (speed-up 16×) | — |
| 2026-10-23 22:10 | vv a49802a | Speed-up judged on an A100 only, otherwise report-only | **yes** |
| 2026-10-23 22:13 | vv 010b2ab | A100 matched as a whole token | yes (follows a49802a; tightens it) |
| 2026-10-23 23:39 | vv 552da46 | Adds an edge-offset moved point | yes (adds a check, loosens nothing) |
| — | vv `tests/test_alma_toeplitz_check.py` | Inherits the relative 1e-8 at the moved point | yes (inherited from 5ece518) |
| 2026-10-28 | this note + `toeplitz_reexam_042.yml` | Absolute worst-case tolerances; all three points gated; Tier I independent reference on held-out scenes; off an A100 the verdict is INCOMPLETE instead of the speed-up becoming report-only | **yes**: written after reading both old runs, before any run under these criteria |
| 2026-10-30 | vv PR #202 review | (a) η-gradient budget 2√P → 4‖b‖₂ (tighter); (b) sequential reductions: n = max(N_d, P, n_fft) → N_d + 2P (the FFT stays in ρ; larger at R_mpol, where the old n understated the bound); (c) a gradient site missing from either side is a FAIL row, a truncated output FAILs; (d) every output records a run id and `summary` refuses several; (e) the S3_edge_filled_no_offsets claim narrowed to what the driver runs | no: before any run under these criteria. The yml is unchanged, so its hash is unchanged; the budget's code changed, and no run used either version |

Later changes go here first, with the new hash, and make every earlier result under the old hash void for G2.

## 9. Judgement calls for Ben

1. **Worst-case vs probabilistic accumulation.** The gate uses γ_n, a rigorous worst case. At IM Lup's size it
   allows about 1e-9 of C_d. The probabilistic bound is 100–1000× tighter, but GPU reductions need not behave
   randomly. I chose the rigorous bound after seeing that the old σ-gradient error is probably inside it and
   outside the probabilistic one. The probabilistic ratio is reported on every row so the choice is visible.
   Changing it is a criteria change (section 8).
2. **Memory on the Toeplitz path** (section 6): kept from 5ece518, and flagged as made after a result.
3. **The cold-start fit** in the original §7 is not part of G2. It belongs to the IM Lup fit gate.
