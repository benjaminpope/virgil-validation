# ALMA IM Lup, phase P3: MPoL reference, joint MaxEnt fit, cross-validation

Spec written 2026-10-20 by the coordinating session. It extends
`design/alma_imlup.md` (vv main 0278333bf89b3995462909d9c27412a1a6caab15),
whose decisions B1–B10 still hold. Workers commit this file as
`design/alma_imlup_p3.md` (W9) and keep its "Built" sections current.

## Inputs (all measured, 2026-10-20)

| Input | Where | Notes |
|---|---|---|
| Per-ddid arrays, all 26 ddids, stride 1 | OzSTAR `/fred/oz440/bpope/alma_imlup/vis/55eb2e9/ddid_{00..25}.npz` | vv format (`u, v, vis, weight, freq, ddid, obsid`), from job 18862565. N = 42,421,131. Both readers agreed (u, v 2.6e-16; vis 2.5e-7 rms; w 6e-8). |
| MS | `/fred/oz440/bpope/alma_imlup/ms/IMLup_continuum.ms` | For MPoL's own export only. |
| Model-free ŝ per ddid | `~/data/alma_imlup/read/out/55eb2e9/ddid_NN.json`, `summary.md` | SB 1.05–1.15; LB EB1 1.66–1.75; LB EB2 1.99–2.08. Upper bounds. |
| Envs | OzSTAR conda `alma-casa` (casatools 6.7.0.31), `alma-virgil` (latest JAX, virgil dcf1db8, casatools pinned 6.7.0.31), `alma-radio-bench` | P3 builds new envs only where named below. |
| A100 cost | job 18862592, `~/data/virgil_validation/alma_radio_bench/out/18862592/combined.md` | f64 eps 1e-6 value+grad at 4.24e7: 0.44–0.53 s; L-BFGS ≈ 0.55 s/step; 18–24 GB peak. **No Toeplitz** (B2 closed). `chunk_size=10**6` is faster than the default at full N. |
| jit hygiene | virgil#438 | `fit` captures no data constants and does not recompile across fold masks or λ. Never close over a term in `jax.jit`; use `fit` or `eqx.filter_jit(fn)(term, values)`. |
| MPoL code | MPoL-dev/examples 5a7c3f0b373f62341ca4baf5869f95e916bbabb3, `04-IMLup-multi-EB`; mpol 0.3.1 | vv#146 ran its `Net` unmodified against virgil (map in `docs/alma_conventions.md`, Part 3). |

## Decisions for P3

| # | Decision | Why |
|---|---|---|
| C1 | Two virgil configurations. **R** ("replication"): MPoL's objective exactly, our optimiser. **F** ("full"): our model, λ by cross-validation. | R separates "different optimiser" from "different model"; F is the result we stand behind. |
| C2 | R: 1028² at 5 mas (MPoL's grid, half-pixel map D11), 30 mas Gaussian taper, α = 0, no primary beam, g per observation (obs 0 fixed), offsets per observation (obs 0 fixed), s per ddid, MPoL's entropy (below) at λ_R. | Matches `sgd.py:441`, its `Net` defaults (`fixed_amp_index=0`, `fixed_position_index=0`) and the paper-run flags. |
| C3 | R's penalty is MPoL's entropy, not virgil's `MaxEntropy`: S_M = (1/F_tot) Σ I log(I/p), on the **tapered** image I (Jy/arcsec²), p = 1e-4, F_tot = 0.253 (constants, `sgd.py:298–302`, `mpol/losses.py` `entropy`). virgil's `MaxEntropy` normalises the image to unit sum, so it differs from S_M by a flux-dependent term, and it acts on the untapered image. Implement S_M in vv as a fit term whose `log_norm(values)` is the penalty (it needs both the image and `cal.flux`), or as a custom regulariser if fit gives regularisers the nuisance values. | Without this R is not MPoL's objective. |
| C4 | λ_R = 2N·λ_M = 2 × 42,421,131 × 1e-9 = 0.08484, in units where the data term is −ln L. MPoL's batch loss is −ln L_b/(2 n_b) + λ_M S_M; its stationary point over an epoch is that of −ln L + 2N λ_M S_M. | Exact map; checked numerically by test R1. |
| C5 | F: 1536² at 5 mas (7.68″ field), `Image` log-brightness offset by `primary_beam(1536, 5, 239e9, 12, 1.13)`, no taper, α free (Uniform(0, 5), ν₀ = 239 GHz), virgil `MaxEntropy` (flat prior over the support) with λ from CV, TSV = 0. | DSHARP's CLEAN mask has semi-axes 2.75″ × 1.75″; MPoL's 5.14″ field clips it, so F must be wider. Bench cost is set by N, not npix. |
| C6 | Priors (Jeffreys, `jeffreys-priors`): ln g, ln s log-uniform; offsets uniform within ±1/(2 q_min) (CalibrationNuisance default); α uniform; flux log-uniform. | House rule. |
| C7 | Numerics: float64 throughout, `set_ft_defaults(eps_f64=1e-6)` for fits, 1e-12 for every reported −ln L, χ² and CV score; `chunk_size=10**6`. | B3 and the bench. |
| C8 | Optimiser: `fit(method="lbfgs")`, `progress=` records loss each step, explicit `max_steps` and `time_limit`. **Converged** = loss change < 0.1 (in −ln L units, i.e. Δχ² < 0.2) over the last 100 steps; otherwise report "not converged" with the trajectory. | B4: fit's gtol is normalised by N. |
| C9 | Staging for both configs: (1) SB only (obs 0–4, 10.9 M points), image + all nuisances, from a Gaussian of FWHM 1″ and flux 0.25 Jy, s_d initialised at ŝ_d; (2) all data, warm-started from (1), LB s_d at ŝ_d. | Cheap geometry and SB calibration first; avoids the LB data pulling a bad start. |
| C10 | Report raw χ²/N per ddid with the quoted weights (s = 1) **first**, then with fitted s. | `report-raw-chi2`. |
| C11 | CV: `dartboard_folds(data, k=5, n_q=12, n_phi=16, seed=0)` on all data, saved once (per-ddid int8 arrays + parameters + a sha256 of the assignment). Grid λ_F ∈ λ₀ × 10^{−2, −1.5, …, +2} (9 values), λ₀ = the F-equivalent of λ_R at the SB-stage flux (state the conversion). 45 tasks, each a training fit (mask = fold ≠ j) warm-started from the full-data F fit at λ₀, `max_steps=1500`, `time_limit=50 min`. Score: held-out −ln L at eps 1e-12 using that task's fitted s, g, offsets, α; also raw χ² (s = 1); both per ddid and per uv annulus (the 12 dartboard q-annuli). | Plan "MaxEnt" section and B10 of the second review; the GP (P4) reuses these exact folds. |
| C12 | λ* = argmin of the summed held-out −ln L; also report the one-standard-error λ. If λ* is at a grid edge, extend the grid by two half-decades and rerun those tasks before the final fit. Final F fit at λ* on all data is the released model. | Standard CV. |
| C13 | Acceptance on s: SB ddids (0–17) |s_d/ŝ_d − 1| < 3 %; LB ddids (18–25) s_d ≤ 1.01 ŝ_d. Failing these is a finding, not a tuning knob. | ŝ is model-free and tight on SB (B5). |
| C14 | The MPoL reference uses MPoL's own export (visread → asdf), not our arrays, and runs `sgd.py` unmodified with the paper flags. Only `--log-interval` changes (200, from 10), which affects TensorBoard plotting, not the optimisation. | A faithful reference; and a third reader (visread) on the full data for free. |

## Workers and order

| Worker | Repo | Depends on | Delivers |
|---|---|---|---|
| W8 | vv PR | — | MPoL bridge tasks for the reference run (below). |
| W9 | vv PR | — | `scripts/alma_imlup_maxent.py` + helpers in `src/virgil_bridge/alma.py`; laptop tests; commits this spec. |
| W10 | ozstar_scripts PR | W8 and W9 merged | Jobs `alma_mpol_ref`, `alma_imlup_maxent`, `alma_imlup_cv` pinned to vv main and virgil main. Ben submits. |

W8 and W9 run in parallel; both may add to `trust/graph.yml` (merge, keep both sides). Only W8 edits `src/external_bridge/mpol_worker.py`; W9 calls its existing tasks (vv#146) and W8's new ones only after W8 merges.

## W8: MPoL bridge for the reference run (vv)

In `src/external_bridge/mpol_worker.py` (runs in an MPoL env, never imports virgil), add tasks:

- `export_checkpoint(checkpoint, examples_dir)`: rebuild `Net` (1028², 5 mas, the example's `obsid_dict`, FWHM 0.03) from the clone, load the state dict, write an npz: the sky-oriented tapered image (`icube`, Jy/arcsec², with its pixel grid), the base image, the 7 amplitude factors (fixed one included), 26 σ factors, 7 δRA and δDec (arcsec, fixed included); and a JSON with the same scalars plus `analyze_scale_factors.py`'s per-EB averages computed the same way.
- `objective_full(checkpoint, asdf, batch=10**6, sparse_matrices=True)`: float64, the example's `forward` with MPoL's accurate NuFFT, summed over all data in batches: −ln L (MPoL's `log_likelihood`, adjusted weights), S_M, and the per-ddid χ² with s = 1 and with the fitted s. GPU if present.
- `asdf_vs_npz(asdf, npz_dir)`: apply `loaddata.py`'s flag handling and unit conversion, then compare with our per-ddid npz: N, u, v (kλ → λ), vis, weight per ddid; JSON of max discrepancies. A conjugation mismatch here is a finding about visread or us; report it, do not "fix" it.

Tests (laptop, `.venv-mpol`, skip if absent): `export_checkpoint` round-trips a randomly initialised `Net` saved with `torch.save`; `objective_full` on the stride-1000 subsample written as a small asdf equals vv#146's per-batch result; `asdf_vs_npz` agrees on a stride-1000 export made with W2's synthetic MS or the subsample.

Also: `scripts/setup_external.sh` learns to build the MPoL env for OzSTAR use (torch with CUDA when available, mpol 0.3.1, visread 0.0.5, asdf, tensorboard, matplotlib); record versions.

## W9: virgil analysis script (vv)

`scripts/alma_imlup_maxent.py`, subcommands, each writing JSON + npy into `--out`:

- `fit --config {R,F} --lam X [--sb-only] [--init DIR] [--mask FOLD_FILE --fold j]`: loads the npz arrays via `virgil_bridge/alma.py` (`ComplexVisData.from_arrays`), builds the model and `CalibrationNuisance` per C2/C5, runs C9's stages unless `--init` is given, and writes: image (npy, plus a FITS with a SIN WCS centred on the MS phase centre), nuisances (g, offsets in mas, s, α, flux), the loss trajectory, `info["stop"]`, convergence per C8, raw χ²/N per ddid (C10) at eps 1e-12, and residual dirty images per observation (`crosscheck.alma.gridded_dirty_image` on the residuals, 1024² at 3 mas).
- `cv-task --lam X --fold j --init DIR`: a training fit then the C11 scores.
- `cv-aggregate DIR`: the CV curve (per-fold, sum, mean ± sd), per-annulus curves, λ*, the one-SE λ, an edge flag, and plots.
- `compare --virgil DIR --mpol DIR`: R against MPoL: s_ours vs g·s_MPoL per ddid, g, offsets, flux; image difference on the common 1028² grid (map D11) raw, at 30 mas and at the DSHARP CLEAN beam (from the FITS header); the R objective (C3/C4) at MPoL's solution vs at ours, both with virgil at eps 1e-12, and MPoL's own `objective_full` beside it. Ours must be ≤ MPoL's; if not, our optimiser failed (a finding). F against DSHARP: the F image at the CLEAN beam vs `images/IMLup_continuum.fits`. Writes `compare.md`.

Laptop tests (subsample_1000, ≤ 128², one process; skip without `virgil.radio`):

- **R1 (tier A, root `mpol`):** at random image and nuisance values on the subsample, the R objective (data term + C3 penalty at λ = 2N_sub·λ_M) equals 2N_sub × MPoL's per-sample loss `neg_log_likelihood_avg + λ_M·entropy` from the example's `Net` (vv#146 tasks), to 1e-6 relative with MPoL's accurate NuFFT, after removing constants. A control with virgil's `MaxEntropy` in place of S_M must fail.
- **R2:** `fit --config R` runs 20 steps on the subsample at 128² and the loss falls; trajectory and outputs exist.
- **R3:** `cv-task` + `cv-aggregate` on 2 λ × 2 folds at 64² produce the files; held-out and training masks are disjoint and cover all points; Hermitian partners share a fold.
- **R4:** `compare` runs on a synthetic MPoL export (from W8's format; mock it if W8 has not merged) and on a self-comparison, which must give zero differences.

Nothing here runs the full data locally.

## W10: OzSTAR jobs (ozstar_scripts)

All GPU (A100), envs per job's `setup_env.sh`, vv and virgil pinned by `--ref`/variables, outputs small enough to download (JSON, md, PNG, ≤ 50 MB npy/FITS).

- `alma_mpol_ref`: (1) CPU: run `casa_analysis/src/export_asdf.py` (visread, in an env with casatools 6.7.0.31) on the MS → `/fred/.../alma_imlup/mpol/imlup.asdf`; download `images/IMLup_continuum.fits` → `dsharp.fits`. (2) GPU: `sgd.py ... --lr 1e-2 --FWHM 0.03 --lam-ent 1e-9 --epochs 10 --log-interval 200` → chk0; (3) the same with `--load-checkpoint chk0` → chk1; (4) `analyze_scale_factors.py chk1`, `compare_CLEAN_mpol.py chk1 dsharp.fits ... --fitsfile mpol_0.03.fits`, W8's `export_checkpoint`, `objective_full`, `asdf_vs_npz`. Request 4 h per training stage; record wall time per epoch.
- `alma_imlup_maxent`: array of 3 tasks: R full (stages per C9), F SB stage then F full at λ₀. 3 h each.
- `alma_imlup_cv`: (1) folds; (2) the 45-task array (%8 throttle) warm-started from the F full fit; (3) aggregate; (4) the final F fit at λ*; (5) `compare` against `alma_mpol_ref` (`--after` both). Each submit block in the PR, chained with `--after` job ids; `--foreground` where a later command needs the result.

## Budget

| Stage | Estimate (A100) |
|---|---|
| MPoL reference, 2 × 10 epochs of ≈ 1,060 batches | 1–4 h, unmeasured |
| R and F fits, ≤ 3,000 steps × 0.55 s | ≈ 30 min each, plus SB stages |
| CV, 45 × ≤ 1,500 steps | ≈ 10 GPU-h expected, ≤ 38 h cap |
| Final F fit + compare | < 1 h |

## Reported in P5

The comparison metrics 1, 2, 5 and 6 of `alma_imlup.md` come out of `compare` and `cv-aggregate`. The radial profile (3) and spiral pitch (4) wait for P5, against DSHARP II/III.

## Things to tell Ian (collect, do not send)

- `log_likelihood` docstring Σ w_i vs code Σ ln w_i (vv#146, P11).
- Default NuFFT 6e-4 from exact at 256² (P12).
- AdamW's default weight decay (0.01) also decays the image, σ, amplitude and offset parameters, so `sgd.py` does not optimise exactly the stated objective; R shows how far its solution is from the optimum of that objective.
- The 1028² × 5 mas field (5.14″) is narrower than DSHARP's CLEAN mask (5.5″ × 3.5″).

## Built: W9 (virgil analysis script)

Code: `scripts/alma_imlup_maxent.py`; helpers in `src/virgil_bridge/alma.py` (P3 section); numpy side
`src/crosscheck/alma_maxent.py`; tests `tests/test_alma_maxent.py` (R1–R4 and numpy checks); ledger `D12`.
Every path is an argument; `--data DIR` (a directory of `ddid_NN.npz`, `vis` complex or `re`/`im`) has no default.

**C3 (entropy of the tapered image).** `virgil.imaging.MaxEntropy` is a regulariser: it sees the model, not the
fitted flux or the taper, and its entropy is of the unit-sum untapered image. R therefore adds a *fit likelihood
term*, `MPoLEntropy`, with empty residuals and `log_norm(values) = λ S_M`; a term sees every fitted value
(`flux`, `log_brightness`), and `fit` adds `log_norm` to the loss. `S_M = (1/F_tot) Σ I ln(I/p)` on
`I = taper ⋆ (flux · softmax(log_brightness) / dx²)` (FFT with the Gaussian multiplier, floor 1e-30), `p = 1e-4`,
`F_tot = 0.253`. F keeps `MaxEntropy(λ_F, prior = exp(primary beam))`.

**λ conversions (C4, C11).** `λ_R = 2N λ_M` (N = 42,421,131 gives 0.084842262). `λ_F = λ_R · F / (F_tot dx²)`,
`dx` in arcsec; at F = F_tot, `λ₀ = λ_R / dx² = 3393.69` at 5 mas. The conversion ignores the taper and the beam
(the CV grid decides); `λ₀` is computed at the nominal flux 0.253 for stage 1 and for the CV grid; `--flux-lam0`
records it at the fitted flux. `lambda0 --n N [--flux F]` prints it; `--lam-r-equiv` on `fit` applies it.

**Deviations.** (1) `--folds FILE --fold J` replaces the `--mask` of the spec: `folds` writes the
fold assignment (per-ddid `fold_NN`, `annulus_NN`, a JSON with a sha256) once, and every job reads it. (2) `fit`
reports in 200-step chunks, so the C8 rule "loss change < 0.1 over ≥ 100 steps" compares the last record with
the latest one at least 100 steps earlier (the first if shorter). (3) R1's data term agrees to 1.1e-6 of |−ln L|
(MPoL's accurate NuFFT, `P12`: 1.6e-6 on visibilities), the entropy to 3.5e-8; the test bound is 1e-5.

**CLI** (`$D` the vis directory, `$S` the ŝ directory, `$O` an output root; `python scripts/alma_imlup_maxent.py`):

    fit --config R --data $D --shat $S --out $O/R                       # λ_R = 2N·1e-9, two stages
    fit --config F --data $D --shat $S --sb-only --lam-r-equiv 0.084842262 --out $O/F_sb
    fit --config F --data $D --shat $S --lam-r-equiv 0.084842262 --out $O/F
    folds --data $D --k 5 --seed 0 --out $O/folds.npz
    cv-task --data $D --folds $O/folds.npz --fold J --init $O/F_sb --lam L --out $O/cv/lam_L/fold_J
    cv-aggregate $O/cv                                                  # cv_summary.json, cv_curve.png, cv_annulus.png, cv.md
    fit --config F --data $D --shat $S --init $O/F_sb --lam L_STAR --out $O/F_final
    compare --virgil $O/R --mpol $MPOL_EXPORT --data $D --f-dir $O/F_final --dsharp DSHARP.fits --out $O/compare

`compare --mpol DIR` reads `export.npz`, `export.json` and `objective.json` from W8's export.

**Outputs of `fit`:** `values.npz` (all fitted values), `image.npy` (float32 sky, Jy/arcsec²; N² × 4 B),
`image.fits`, `image_tapered.npy` (R only), `resid_dirty_obs{0..6}.npy` (1024² float32, 4 MB each, 28 MB in all;
`--no-dirty` omits), `result.json` (stages with trajectories, convergence, stop reason, nuisances, −ln L,
χ² per ddid at s = 1 and fitted, C13 acceptance, `objective_R`, versions). `cv-task`: `cv.json` (a few kB).
`compare`: `compare.json`, `compare.md`.

**Review changes (PR #150).**
- `cv-task --lam L`: `L` is the full-data strength (the grid key in `cv.json["lam"]`); the fit uses `L · n_train / N` (`lam_train`), so each CV fit keeps the data-to-penalty balance of the final fit on all N points.
- Config F's `image.npy`/`image.fits` is the **sky**: `flux · exp(η) / Σ exp(η + ln PB) / dx²`, whose product with the primary beam sums to `cal.flux` (the fitted pixels are sky × beam). R has no beam. Whether DSHARP's image is beam-corrected is for W10/P5 to state when reading `F_vs_dsharp`.
- `folds` records the sha256 of each ddid's (u, v) in the file; `cv-task` refuses (ValueError) data whose points, count or order differ for any ddid it loads, and `--ddids` subsets are checked too.
- `compare` requires the reference observation's amplitude `amp[0] == 1` in MPoL's export (`values_from_mpol_export`), and `resample` no longer zeroes edge pixels on identical grids (found by running `F_vs_dsharp` end to end).
- R1's bound is 1e-5 of the *model-dependent* part (½Σw|r|² + λ S_M), not of |−ln L|, in which the constants cancel; near-miss controls (prior 1e-3, F_tot halved, untapered) fail by more than 10× the bound. A new test shows `fit`'s own loss contains `+ λ S_M`, with a control that drops the term.

## Built: W9, first-run fixes (job 18912039)

The first A100 run fitted F_sb, R and F but lost R and F: the per-point report called `sub.model` on a whole observation
(1.74e7 points, a 22.1 GB buffer, about 1.27 kB per point) and ran out of memory before anything was saved.

- **Save first.** Each stage writes `values.npz`, `image.npy`, `image.fits` and a provisional `result.json` (`complete: false`,
  stages, loss history, steps, stop, `stopped_because`) after every restart round and at the stage end, atomically. A new
  `report --data D --out DIR` recomputes the full result (and dirty images unless `--no-dirty`) from the saved `values.npz`
  without refitting.
- **Chunked evaluation.** `virgil_bridge.alma.model_chunks` evaluates the model on index ranges of at most `--chunk-size`
  points (default 1e6, via `data.subset`); `per_point`, `score`, `r_objective` and `residual_dirty_images` use it, so
  chunked equals unchunked to rounding. Memory at 1e6 points is about 1.3 GB per such buffer, so the 4.24e7-point set peaks
  at a few GB rather than the about 54 GB the unchunked report needs.
- **Stopping is C8.** `info["stop"] is None` meant only that `fit`'s `gtol=1e-4` gradient test passed. `run_fit_c8` restarts
  the fit warm (each restart's test is 1/1000 of the current gradient, so it tightens) until C8 holds (|Δloss| < 0.1 over
  at least 100 steps) or the budget is spent, and records `stopped_because` in {c8, max_steps, time_limit, stalled,
  non_finite, max_rounds}. Defaults: `--max-steps 20000`, `--time-limit 18000` (5 h) per stage; `--gtol` sets the first
  round's tolerance. The cv-task defaults are unchanged.
- C13 (SB |s/ŝ−1| up to 0.123 at ddid 5–6) is reported as a finding, not tuned. Science settings are unchanged.
