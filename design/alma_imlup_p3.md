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

## Decision: F fixes alpha = 0 (2026-10-21)

Configuration F no longer fits the spectral index by default: `CalibrationNuisance(..., alpha=False)`, so alpha = 0 as in MPoL
(one monochromatic image for all EBs and spws). Ben: "let's fix it to what MPoL does for now, because we care about the
comparison". Evidence: the data barely constrain alpha (a few % fractional bandwidth) and it trades off against the per-obs
gains; in MaxEnt run 03531b4 free alpha pegged at 5.0, the upper bound of virgil's default (0, 5), and in bd3f748 it sat near 0.
Primary beam, entropy form, lambda and `nu0_hz` are unchanged. Free alpha stays as an opt-in for a later scan
(alpha in {0, 2.5, 5}): `FULL["alpha_free"]` / `build_problem(..., alpha_free=True)` / `--alpha-free` on `alma_imlup_maxent.py`
(fit, cv-task; `report` reads it from the fit's saved args). `result.json` carries `alpha` (float, or null when fixed) and
`alpha_label` ("fixed 0 (MPoL)" or the fitted value). alma_imlup_cv passes no alpha flag, so it follows the new default.

## Decision: the data column is CORRECTED_DATA (D1, 2026-10-22)

Status: the vv side (this section, `alma_worker` `datacolumn`, `asdf_vs_npz` PASS/FAIL) is in vv#166; the OzSTAR side (`alma_imlup_read --datacolumn`,
`datacolumn.txt`, the `-cd` tag, `--check-only`, `--data` for maxent and cv) is ozstar_scripts#58. **Until #58 has run, the IM Lup npz are still DATA.**
Ledger: D13 (open until the `asdf_vs_npz` PASS on the `-cd` npz).

**Finding.** Ben asked for a check that we replicate MPoL's workflow. The objective (likelihood, entropy, taper, s/g
mapping, grid, offset sign) is identical; the *data* were not. The DSHARP MS has DATA, CORRECTED_DATA, MODEL_DATA and
WEIGHT_SPECTRUM. MPoL's export (visread via `alma_mpol_ref`'s `run_export.py`; `asdf.export.json` says
`"datacolumn": "corrected_data"`) read CORRECTED_DATA. Our reader (`alma_worker.read_ddid`, default `datacolumn="DATA"`)
read DATA. `asdf_vs_npz` (MPoL's asdf against our npz) showed identical N, keys and weights (rel 6e-8) but
|ΔV|max of 0.03 Jy on ddid 0, 0.4-0.7 Jy on ddids 4 and 17-20, 0.95-1.63 Jy on ddids 22-25. It applied no tolerance,
so nobody noticed. This likely explains MPoL's −ln L being lower than our R fit's by 1.8e6 and the LB obs-6 gain
(ours 1.28, MPoL 0.981). Not yet shown: that needs the new npz and a rerun (the PASS below, then ŝ and the fits).

**Fix.**
1. The column is an explicit, recorded choice. `alma_worker` `read` takes `datacolumn` (case-insensitive; an MS without
   the column is an error, never a silent fall back) and writes it, upper case, to its result JSON. The generic default
   stays `DATA` (the synthetic MS has only DATA); the IM Lup job `alma_imlup_read` (ozstar_scripts#58) defaults to `CORRECTED_DATA` and
   writes `datacolumn.txt` into the npz directory, which every downstream summary prints. The new npz goes to a new tag
   (`<vv commit>-cd`), so the DATA npz stays for comparison.
2. `asdf_vs_npz` is a test: per ddid PASS needs equal N, all keys matched, |ΔV| < 1e-5 Jy, and relative differences of
   per-point relative u, v, weight < 1e-6. `worst.all_pass` is the verdict. Channels are matched by each point's own frequency
   rank, never by channel index or an assumed direction: visread 0.0.5 orders channels by decreasing frequency
   (`process_casa.get_channel_sorted_data` flips an ascending MS), our npz keeps the MS order, which is increasing for some ddids, so the
   index-keyed match broke there (ddids 4, 17, 19, 20, 23, 24). We match rather than re-sort our npz so that existing npz and tags stay
   valid. u, v and weight are compared per point (u, v against max(|u|, 1 λ)). Logic in `external_bridge/npz_compare.py` (numpy only, tested in `tests/test_alma_npz_compare.py`).
3. Weights. `alma_mpol_ref`'s check shows our weights agree with visread's to 6e-8 on every ddid, so for this MS
   WEIGHT_SPECTRUM (which our reader uses when defined) and visread's weights are the same and the choice does not
   matter here (stated from that agreement, not from inspecting the column).
4. virgil's `ComplexVisData.read_ms` already takes `datacolumn="DATA"` (virgil `radio.py`); no virgil change is needed,
   and `alma_imlup_read`'s `read_compare.py` will pass it (ozstar_scripts#58) so the two readers read the same column.

C14 stands, with this addendum: "MPoL's own export" includes its column, and our arrays must be read from the same
column or the comparison is not like for like. All fits, CV and ŝ before this date used DATA and are superseded for
the R-vs-MPoL comparison once rerun on the `-cd` npz.

## Option: `fit --workflow=mpol` (2026-10-22)

Ben: "keep working on the radio analysis, to try and closely replicate MPoL". With the data column fixed (D1) the objective and the
data agree; what still differs is the *route* to the optimum. C9 (an SB stage, then all data, from a 1″ Gaussian of 0.25 Jy, s at ŝ)
is ours; MPoL's `sgd.py` runs two all-data AdamW stages (`Snakefile:26-27`, 10 epochs each) from a flat start. The earlier R fit on the
DATA npz stopped at `max_steps` 20000 without C8 and kept its offsets under 2 mas, where MPoL's are 4–13 mas, so a different basin or
a slow direction is possible. `--workflow=mpol` removes the start as a difference. Default stays `p3`; config R only.

| | `p3` (default, C9) | `mpol` |
|---|---|---|
| stages | SB (obs 0–4), then all | one, all data, to C8 |
| image start | 1″ FWHM Gaussian, 0.25 Jy | uniform `softplus(-3)` = 0.048587 Jy/arcsec² per pixel (`BaseCube`, mpol 0.3.1 `images.py:65-76`: `base_cube = -3`, `Softplus`; `sgd.py:59`), total 1.2837 Jy at 1028² × 5 mas |
| parametrisation | `flux × softmax(log_brightness)` | same: logits 0, `flux = value × dx² × npix²` (so `image_of` returns the uniform value to 1e-12) |
| gains | 1 | 1 (`sgd.py:49`, `log10_amp_factors = zeros`) |
| offsets | 0 | 0 (`sgd.py:56-57`) |
| s per ddid | ŝ | 1 (`sgd.py:53`, `log10_sigma_factors = zeros`; with g = 1, s = g s_MPoL = 1; ŝ is not used, C13 is still checked if `--shat` is given) |
| optimiser | L-BFGS to C8 | L-BFGS to C8 (not AdamW: that stays the "different optimiser" of C1) |

The softmax parametrisation cannot reproduce softplus per pixel *during* the fit (ours keeps the total flux a fitted scalar; MPoL's
pixels are free and non-negative by softplus); only the start is matched. `result.json` and the log carry `workflow` (`p3` or
`mpol`); the log line gives raw χ²/(2N) at s = 1 first. `--workflow=mpol` with `--init` (resume or continue) is the same all-data
stage from the saved values; `--sb-only` is an error.

Code: `va.mpol_start_values`, `va.mpol_base_jy_arcsec2` (`src/virgil_bridge/alma.py`); tests `test_mpol_start_reproduces_*` and
`test_workflow_mpol_runs_exactly_one_stage_*` (`tests/test_alma_maxent_run.py`).

**Offset-scaling diagnostic.** `compare` already ran `va.grad_groups` (vv#165). It now prints physical units (offset per mas, gain per
unit gain, s per ln s, flux per Jy, pixels per unit log-brightness; `grad_table`), and `report --grad` prints the same for one saved
fit without MPoL's export (R only; one value-and-gradient at the fit's eps). Reading it: virgil fits an offset as a logit of
x/bound with bound = 1/(2 q_min) (≈ 10⁴ mas on IM Lup), so dx/dz = bound/2 ≈ 5 × 10³ mas per unit z near 0 and 1 mas ≈ 2 × 10⁻⁴ in z.
In z the offset gradient is therefore bigger than the others' by about dx/dz, and its curvature by about (dx/dz)² ≈ 2.5 × 10⁷
times the curvature per mas², against ~1 for the pixels. L-BFGS has to learn that scale from its history; the first steps and the
line search are set by the large offset gradient and can stall the other groups. On the synthetic test data (bound ≈ 5 × 10³ mas,
dx/dz = 2.6 × 10³): |g_z| max 1.0e4 for offsets vs 1.1e2–3.7e2 for gain, flux and s and 10 for pixels, while per physical unit
the offset gradient is only 3.9 per mas against 3.3e2 per unit gain. The IM Lup numbers need `report --grad` on the A100 fit.

**`--offset-bound-mas X` (fit, cv-task; R and F).** Passes `bounds={"offset": (-X, X)}` to `CalibrationNuisance.priors` (virgil
`radio.py`, `priors(**bounds)`; mas, the same unit and sign as `cal.offset`, symmetric about 0). The default stays virgil's own
±1/(2 q_min), so a run without the option is unchanged. Why: with the default bound ≈ 10⁴ mas, dx/dz ≈ 5 × 10³ mas and the curvature in z
is ≈ 2.5 × 10⁷ times the per-mas² curvature, while MPoL's 4–13 mas offsets are z ≈ 10⁻³ and our earlier R fit's stayed below 2 mas. At
X = 100 mas dx/dz ≈ 50 mas, a factor 100 in gradient and 10⁴ in curvature better, and the offsets still sit well inside the bound.
The prior is flat, so inside the bound the objective (−ln L + λ S_M) is identical at any X: the option changes only the optimiser's
coordinate. It stops being innocent if a fitted offset reaches the bound, so `report` (and `fit`, which reports) records
`offset_bound` in `result.json` (`bound_mas`, `requested_mas` null for the default, `max_abs_over_bound`, `near_bound`, `warning`),
prints the bound and the largest |offset|/bound, and prints `!!! OFFSET BOUND BINDING` if any offset ends within 5% of it; a binding run
is not a fit of the unbounded objective and must be rerun with a larger X. `report` and `report --grad` reuse the saved `--offset-bound-mas`.
`compare` evaluates its gradient table at virgil's default bound. Tests: `test_offset_bound_reaches_the_priors_*`,
`test_report_records_the_bound_and_warns_*` (`tests/test_alma_maxent_run.py`).

## Built: MPoL on the dartboard folds (2026-10-22)

Ben: "start a pr-worker for MPoL cross-validation on our dartboard folds, so we can compare held-out chi^2 on identical folds."
Goal: claim an *improvement* (or parity) of virgil's IM Lup MaxEnt image over MPoL's on the metric neither optimises, comparison metric 1
(held-out dartboard χ², total and per uv annulus, B9), with both codes trained on the same training sets.

- **C14 holds.** MPoL's example runs unmodified; the hold-out is on the input side. `mpol_worker` task `export_fold_asdf` writes, for fold J, the
  asdf with fold J's points **flagged** (loaddata keeps `~flag`): the same points as `cv-task`'s mask `fold != J`, matched by (channel frequency
  rank, time, ant1, ant2) because the asdf's order and ours differ; Hermitian partners share a fold because the dartboard cell is symmetric
  (`virgil.radio.dartboard_folds`). MPoL's `lam_ent` stays 1e-9, i.e. λ_R = 2 n_train λ_M, the same data-to-penalty balance as `cv-task`'s `lam_train`.
- **Scoring.** `score_fold` evaluates the trained checkpoint on the fold's held-out points in `cv-task`'s convention (`s = g s_M`, raw χ² at s = 1, −ln L at the
  fitted s, profiled form; total, per ddid, per annulus). Agreement with `virgil_bridge.alma.score` on a shared synthetic case: 3.5e-5 per ddid, set by MPoL's NuFFT.
- **Aggregate.** `alma_imlup_maxent.py cv-compare` reads MPoL's `fold_J/score.json` and virgil's `cv.json` / `cv_summary.json` and lines them up per fold and per
  annulus at virgil's λ* and at the grid λ nearest λ₀ = C11's equivalent of 2 N λ_M (λ₀ = 3393.69 at N = 42,421,131 and F = F_tot).
- **Not claimed.** A held-out advantage for either code is a statement about uv-plane prediction, not about image-plane structure (B9). The comparison is of
  methods *as run*: MPoL's AdamW at 10 + 10 epochs against our L-BFGS; MPoL's training of each fold is not warm-started from the all-data solution as virgil's `cv-task`
  is (`--init`), which favours virgil if MPoL's stage 0/1 are under-converged on a fold: the report ends with a per-fold table of MPoL's epochs and wall time per stage (the `stage<N>.time.json` beside each `score.json`) and virgil's steps, wall time and stop reason at λ*, and lists a fold without MPoL's times as a problem.
- **Chain of custody.** The checkpoint holds no record of its training data (`sgd.py` is unmodified), so `score_fold` is given the fold's training asdf (it must flag exactly fold J's held-out points per ddid beyond the export) and the fold `train.sbatch` recorded for the checkpoint; `score.json` carries `train_fold`, `train_flagged`, `train_flags_sha256`, `checkpoint_trained_fold`, and `cv-compare` reports a problem when they are missing or for another fold, so an all-data or neighbouring-fold checkpoint cannot be scored as fold J's.
- Code: `src/external_bridge/{cv_folds,cv_compare}.py`, tasks in `mpol_worker.py`, `scripts/alma_imlup_maxent.py cv-compare`, tests `tests/test_alma_mpol_cv.py`,
  formats in `docs/alma_conventions.md` Part 6. The OzSTAR job is `alma_mpol_cv` (ozstar_scripts).


## Built: speed comparison with MPoL (2026-10-23)

Ben: "let's keep working until we are as good as or better than MPoL from a cold start. I would also like to see a speed comparison with MPoL."
`alma_imlup_maxent.py speed` compares virgil's L-BFGS fit with MPoL's two-stage AdamW run on one axis, from small saved JSON only (no data, no GPU):

    python scripts/alma_imlup_maxent.py speed --fit [LABEL=]DIR [--fit ...] --mpol RUN_DIR --out DIR [--log FILE ...] [--target LABEL=R ...]
        [--vg-seconds LO HI] [--mpol-startup-s S]

`--fit DIR` holds a config-R fit's `result.json` (and optionally `anchors.json`, below); `--mpol RUN_DIR` is `alma_mpol_ref`'s `out/<run>/` (`stage<N>.time.json`,
`export/objective.json`, and `export/objective_chk0.json` when it exists); `--log [mpol=|virgil=]FILE` are job logs or digests with the `job=… host=…` and GPU lines (the prefix says whose run; a log without one counts as MPoL's only if its job is in `stage<N>.time.json`, else it is reported as unattributed). It writes `speed.json`,
`speed.md` (raw χ²/N first, the definitions, the tables, the fairness notes) and `speed.png` (objective against wall time, both codes, the targets as horizontal lines, with a
zoom). Code `src/crosscheck/alma_speed.py` (numpy only), tests `tests/test_alma_speed.py` (synthetic trajectories of a known cost, compile time, restarts and constant).

**Definitions.**
- **Objective** is MPoL's `R = −ln L + λ_R S_M` in float64 on all data (C3, C4); MPoL's own value is `objective_full`'s `objective_R`. virgil's fit loss equals R up to a constant, taken from the final
  point (`objective_R − loss_final`; the report evaluates R at eps 1e-12, the loss is at eps 1e-6). The speed report compares that constant between fits and with any anchor (a second evaluation of R
  at another step: `anchors.json`, a list of `{steps, objective, loss}`); a spread above 5 fails the check and the curve is not to be trusted. Within one fit `fit` records no second evaluation of R
  (the report runs at the end only), so the check is across fits. Three existing fits (03531b4/R on DATA, R_mpol and R_mpolwf on CORRECTED_DATA) give the constant −189,545,489.39 to 1e-3.
- **virgil wall time** is the sum of the stages' `wall_s` (the optimiser loop of each stage: it includes the jit compile of each round and the restarts' overhead; the p3 route's SB stage counts in the time
  and not on the curve, its loss being on fewer data). The problem build before the first stage (`result.json` `wall_s` minus the stages) is reported as start-up; reading the npz is not timed. The
  compile of each round is estimated as its first 200-step chunk minus the median chunk.
- **MPoL wall time** is the whole `sgd.py` process per stage (`stage<N>.time.json`: asdf load, CUDA start-up, epochs); it is known only at the checkpoints, so its curve has two points: chk0 and chk1.
- **Time to target**: the wall time of the first record at or below the target (every 200 steps), with the linearly interpolated crossing beside it; for MPoL the end of the first stage whose checkpoint
  is at or below it (an upper bound). Targets: MPoL's final objective (−108,232,184), MPoL's chk0 objective (when evaluated), the best objective known in the inputs, and any `--target`. Warm starts
  (`--init`) give no time to target (they start at or near it) and are drawn dashed.
- **Cost per pass**: MPoL's s/epoch (per stage and its ratio) and s/minibatch (N/40,000 ≈ 1,061); virgil's s/step (median of the 200-step chunks inside rounds, with the 10–90 % spread and the
  first and last quarters, since the line search works harder late). virgil's `info` records **no evaluation count**, so evaluations per step are *estimated* as s/step over the bench's s per
  value+gradient (0.44–0.53 s at 4.24e7 points, f64, eps 1e-6, A100: job 18862592); `--vg-seconds` overrides it. A "pass" is one value+gradient (virgil) or one epoch (MPoL).
- **Fairness notes** (in `speed.md`): hardware from the job logs (host, GPU; "not recorded" when absent), MPoL's 2× stage difference flagged as possible node contention, precision (MPoL float32
  training, virgil float64 at eps 1e-6, every reported objective at 1e-12), start-up and compile separated, data loading in MPoL's wall time and not in virgil's (the bias favours virgil, by minutes),
  MPoL unmodified (C14), AdamW's weight decay.

**MPoL's chk0 objective.** `analyze.sbatch` evaluated only chk1. ozstar_scripts adds `objective_full` on chk0 to `analyze.sbatch`, and `submit.sh --only-chk0 --run=NAME` reruns just that
evaluation for an existing run (ozstar_scripts#63); it writes `out/<run>/export/objective_chk0.json`, which `speed` then picks up as MPoL's second point and a second target.

### Preliminary numbers (OLD env: virgil 0.3.0, jobs 18968244 and the MPoL reference 18913982/18913983; NOT converged)

Both virgil fits stopped at `max_steps` 20,000 without C8 and are from a pre-fix environment, so they say where the comparison stands, not where it ends. Read from the fits' `result.json`
(`speed.md` and `speed.png`, the latter in `docs/imaging/imlup/speed.png`).

| | start | steps / epochs | wall | final R | raw χ²/N | reaches MPoL's final R? |
|---|---|---:|---:|---:|---:|---|
| MPoL (AdamW, float32, 2 × 10 epochs; gina1) | flat | 20 epochs | 21,728 s (6,983 + 14,745; 698 and 1,474 s/epoch) | −108,232,184 | 2.7062 | by definition, at its end |
| virgil `R_mpolwf` (L-BFGS, float64; gina301) | MPoL's flat start | 20,000 steps | 17,078 s (0.80 s/step) | −107,726,506 | 2.7306 | **no**: 505,678 above, not converged |
| virgil `R_mpol` (L-BFGS, float64; gina303) | **warm**, MPoL's solution | 20,000 steps | 15,789 s (0.79 s/step) | −108,304,297 | 2.7052 | not a cold start; ends 72,113 *below* MPoL's |

Per pass virgil is about three orders of magnitude cheaper and needs far more passes: an MPoL epoch (698–1,474 s) is about 880–1,880 virgil steps, or 1,300–3,350 value+gradient
evaluations (at 0.44–0.53 s), while a virgil step is about 1.5–1.8 evaluations; 20,000 steps are about 30,000 evaluations against MPoL's 20 epochs.

**Time to target, cold start: virgil did not reach MPoL's final objective in 17,078 s; MPoL did in 21,728 s.** The ordering does not hinge on MPoL's unexplained 2.1× slowdown of stage 1: at its
stage-0 rate MPoL would take about 13,960 s, and `R_mpolwf` was then still 544,000 above the target (−107,687,746 at 13,552 s). Its last hours fell by about 40,000 per hour against a gap of 505,678, so the
extrapolation is more than 12 h and the rate was still falling; this is a statement about this run (flat start, no preconditioner, old env), not about the best virgil can do. `R_mpolwf` also lost about 1,000 s at
its round-2 restart (its first round stopped at step 60; the record does not say why). The two virgil fits ran on different nodes (gina301, gina303), both A100; MPoL on gina1, also an A100. Not yet known: the time on the
fixed env and with `--offset-bound-mas` or `--precondition diag`, and MPoL's chk0 objective.

## Finding: the cold-start R ob100 fit has dead pixels, and the floor start (2026-10-23)

R from the p3 workflow's 1″ Gaussian start with `--offset-bound-mas 100` and no preconditioner (vv 32a303e had no such flag; job 18968246_0, tag `32a303e-cd-ob100`) ended with a better objective than MPoL's
(−108,309,995.3 against −108,232,183.7; χ²/N 2.7052 against 2.7062; largest |offset| 0.028 of the bound) and an image that does not pass the cold-start gate: an optimiser artefact, not a result.

- **What was measured (local copy).** 64 % of the pixels are below 1e-6 of the peak, with a median of 9e-10 of the peak beyond 2.4″; MPoL's solution and R_mpol have none. The live region ends at 1.78″ and the bright
  arcs (peak 0.11 Jy/arcsec², at 1.4–1.7″) sit on that boundary. No flux lies beyond 1.8″; MPoL has about 10 % of its flux there, which accounts for most of the 0.05 Jy our image is missing.
- **Why.** The image is `flux · softmax(b)`, so ∂L/∂b_i is proportional to I_i. A pixel the Gaussian start leaves at 1e-8 to 1e-15 of the peak cannot grow back, although the entropy term's interior
  optimum puts every pixel above zero (∂S/∂I = ln(I/M) → −∞ as I → 0). The flux that belongs there piles into the outermost live pixels.
- **Fix.** `fit` raises the Gaussian start to MPoL's entropy floor p = 1e-4 Jy/arcsec², as R_clean already does (`--start-floor`, default `va.MPOL["prior"]`; 0 restores the bare Gaussian; R and F; not
  `--workflow mpol`, whose start is uniform, nor `clean`, which has its own floor). That adds about 2.6 mJy (1 % of the flux) at 1028² × 5 mas, and every pixel starts above 1e-6 of the peak (floor / peak ≈ 4.5e-4).
  `result.json` records `start_floor` (also in the first stage) and the arguments.
- **Diagnostic.** `write_report` (so also `report`) logs and records `dead_pixels`: the fractions of pixels below 1e-6 of the peak and below 1e-3 of the floor, the median brightness beyond 2.4″ over the peak, and the
  largest live radius; it prints `!!! DEAD PIXELS … !!!` when more than 1 % are dead, like the offset-bound warning. `report` on the old directory gives the numbers above without refitting.
- **Tests.** `tests/test_alma_start_floor.py` (CPU, seconds): the floored start reads back through `image_of` exactly in R and F, the report flags the bare Gaussian and passes the floored one, and `fit`
  records both. The cold-start gate is rerun with the floor start (arm R, same settings) before it is called passed.

## Decision: the cold-start gate (2026-10-23, approved by Ben)

A cold-start fit (no MPoL information) is called a pass only when the numbers **and the image** agree. `compare` prints and records (`compare.md`, `compare.json` key `gate`) five checks, each PASS or FAIL
(`NOT EVALUATED` when its input is missing, e.g. no `--data`); **the verdict is PASS only if all five pass** (a failed check gives FAIL, a missing one with none failed gives INCOMPLETE):

| | check | source |
|---|---|---|
| a | `objective_R` at or below MPoL's own | R's objective at our solution against MPoL's solution evaluated in virgil, both at eps 1e-12 (`objective.ours_le_mpol`) |
| b | raw χ²/N at s = 1 at or below MPoL's solution evaluated in virgil | the same two evaluations |
| c | C8 converged in every stage | the stages of the fit's `result.json` |
| d | pixels below 1e-6 of the peak under 1 % | `va.dead_pixel_report` of the fit's image (see the finding above) |
| e | total flux within 10 % of the CLEAN flux | `--clean-dir` (an `alma_imlup_clean` output: `clean.json`) when given, else the 0.253 Jy of MPoL's example (`MPOL["f_tot"]`, `sgd.py:298–302`; the first draft said DSHARP's 0.255 Jy, which has no source in the repo); the reference used is recorded |

Flux is the fit's, in MPoL's obs-0 gauge (`gauge_to_obs0`), so a fit with another `--ref-obs` is compared fairly. Code: `virgil_bridge.alma_start.cold_start_gate`, tests `tests/test_alma_start_floor.py`.

## Option: `fit --start gauss-fit`, a data-driven start (2026-10-23)

The p3 workflow starts from a fixed 1″, 0.25 Jy circular Gaussian. `--start gauss-fit` instead fits an **elliptical Gaussian** (flux, centroid, major and minor FWHM, position angle) by weighted least squares
to the visibilities below `--gauss-fit-max-klambda` (default 40 kλ, the short baselines that see the disc's flux, centroid and overall size), from four starting position angles, the lowest χ² kept
(`vs.fit_gaussian`; at most 2 million points, a fixed random subset). The model is that of CASA's `uvmodelfit` (component type G, fitted there by Levenberg–Marquardt); the code is ours, so only the model is CASA's.
Gains and offsets are not fitted (the start has none). The image is the fitted Gaussian on the fit's grid (`vs.gaussian_log_image`), its flux is the fitted flux, and it is then raised to `--start-floor`
as the default start is. Opt-in: the default stays `--start gauss`. `result.json` records the fitted parameters, the points used, χ² per point and the source as `start_gauss_fit` (and in the first stage).

All that is claimed for it: among positive images with a given flux, centroid and second moments the Gaussian has the largest Shannon differential entropy, so it is the maximum-entropy image given those
constraints. That is not our fit's entropy S_M (relative to the flat prior, on a tapered pixel grid), and the fit matches the true moments only as far as the data below the cut see them. Refused with `--workflow mpol`,
`clean` and `--init`, which have their own starts. Tests check that the fit recovers a known ellipse (also with the axes named the other way round) and that the rendered image is the one virgil's forward
model sees (χ² far below the East–West and North–South mirrors).
