# M87* closure-invariant results (Lai et al. 2026): replication plan and criteria

Paper: Lai, Thyagarajan, Wong & Diakogiannis 2026, Nature 659, 317, doi:10.1038/s41586-026-11241-9.
Review of the paper and what virgil takes from it: virgil `design/m87_gendirect_review.local.md` (not public).
Ben (2026-11-13): reproduce the authors' results. Their latent diffusion imager (GenDIReCT) is **not** re-run
or re-implemented: we use its published reconstructions only.

M87* is a worked example here. Nothing in virgil is tuned to it, and a failed criterion is reported, never
loosened (no gate hacking).

**Pre-registration.** The criteria below are fixed by the commit of this file that first contains them. A
criterion changed after a result is a change-log row saying "yes".

## Inputs (stage 0)

All under `$M87_DATA` (default `~/data/m87_eht/`), never in the repo. `scripts/setup_m87.sh data` fetches and
checks them. Licences and acknowledgements: `DATA_ATTRIBUTION.md`.

| Source | What | Size | Check |
| --- | --- | --- | --- |
| Zenodo 10.5281/zenodo.22764433, GenDIReCT v1.1.0-paperM87 (MIT) | code, `results/M87_{2017_101,2018_111,2021_108}_225_sample.npy` (50 reconstructions each, 64×64, 225 μas field), the three fiducial-day UVFITS (`uvfits/`), their `data/ringFitting.py` | 2,119,995 B | md5 764aae1d4ed17b24fdcef7b3c5883037 |
| Zenodo 10.5281/zenodo.21090230, BASIS v0.1.0 (MIT) | xsringauss + bilby/dynesty driver (`examples/runBilby.py`), same three UVFITS, its own `ringFitting.py` | 4,226,811 B | md5 32aab34893130a467b15bedae3973c95 |
| Zenodo 10.5281/zenodo.15524509 (record 15524510), ClosureInvariants v1.0.0 (MIT) | invariant maps | 58,241 B | md5 a2780931ea0e4ad18ad5ac7426bbecfe |
| github.com/eventhorizontelescope/2019-D01-01 (EHT 2017 M87*) | UVFITS, all four 2017 days, both bands | 27 MB checkout | commit 27ae76aacb9067a83388395c3826012b97aac371 |
| github.com/eventhorizontelescope/2024-D01-01 (EHT 2018 M87*) | UVFITS, 21 and 25 Apr 2018 | 47 MB checkout | commit fda41da417419fd9b54505ea3fa70a8cf6fb60ec |
| EHT 2026-D01-01 (2018 + 2021 polarimetric), CyVerse `iplant/home/ckc/EHTC_M87-2018+2021_Jul2026` | 2021 data (13, 18 Apr) | not fetched | CyVerse WebDAV is behind a browser challenge for scripts; Ben downloads by hand if the 13 Apr 2021 cross-validation day is wanted |

Total fetched: about 80 MB, under the 2 GB stop line.

Facts the archives fix, which the paper does not state:

- **Published reconstructions cover only the fiducial days at 225 μas.** Table 1 is the mean over three fields
  of view (128, 225, 450 μas) of the per-field median; the archive has the 225 μas samples only. Stage 1
  therefore compares one field with a three-field mean, and Table 1's ± is the spread between fields.
- **Pixel scale.** 225 μas / 64 = 3.516 μas.
- **BASIS noise.** `estimate_noise_ci` takes σ of each invariant as the standard deviation over 1000 Monte Carlo
  noisy copies of the visibilities (not linear propagation, as Methods says), keeps the diagonal only, and the
  M87 example command in `runBilby.sh` adds `noise_frac=0.05` and `static_noise_floor=1e-4` in quadrature,
  where the paper says "without further systematic error tolerance". Stage 2 runs both.
- **Reference station.** BASIS pins every triangle on the first station in the array table order, one run;
  Methods describes two reference stations combined (that is for GenDIReCT).
- **28 parameters.** xsringauss (14: `I0, Rp, Rn_frac, ecn, f, phi, gax, aq, gq, floor_sigma, floor_brightness,
  bg_gauss_flux, bg_gauss_sigma, gauss_blur_sigma`) + 2 × gauss (7 each: `I0, gauss_sigma, aq, pa, xoff, yoff,
  gauss_blur_sigma`). Invariants carry no absolute flux, so one flux direction is unconstrained by the data
  (only its uniform prior bounds it); each nuisance Gaussian's blur is degenerate with its width.
- **Ring extraction definitions differ.** Lai et al. eqs 6, 9: floor at `2 r_pk`, central disk `r < 0.2 r_pk`.
  EHT Paper IV / virgil `metrics.ring_features`: floor at 50 μas (`2.5 r_pk`), central disk 5 μas (`0.25 r_pk`).
  Diameter and orientation do not depend on these; width and `f_c` do (ruling: `definition`).

## Reference numbers

Table 1 (image domain, GenDIReCT, mean ± SD over three fields):

| Day | 2r_pk (μas) | w_r (μas) | η (°) | A | f_c |
| --- | --- | --- | --- | --- | --- |
| 2017-04-11 | 43.3 ± 1.0 | 16 ± 3 | 167 ± 6 | 0.20 ± 0.01 | 0.037 ± 0.028 |
| 2018-04-21 | 42.7 ± 0.5 | 11 ± 1 | 205 ± 10 | 0.31 ± 0.01 | 0.004 ± 0.004 |
| 2021-04-18 | 39.9 ± 1.8 | 12 ± 1 | 197 ± 4 | 0.26 ± 0.01 | 0.004 ± 0.002 |

ED Table 1 (xsringauss posterior, REX on posterior samples):

| Day | 2r_pk | w_r | η | A | f_c |
| --- | --- | --- | --- | --- | --- |
| 2017-04-05 | 41.5 ± 0.1 | 5.0 ± 0.4 | 165.7 ± 1.3 | 0.262 ± 0.003 | 0.19 ± 0.12 |
| 2017-04-06 | 40.9 ± 0.1 | 5.2 ± 0.2 | 87.2 ± 2.0 | 0.135 ± 0.003 | 0.12 ± 0.08 |
| 2017-04-10 | 40.5 ± 0.2 | 8.5 ± 0.1 | 166.1 ± 1.0 | 0.160 ± 0.002 | 0.13 ± 0.07 |
| 2017-04-11 | 43.0 ± 0.1 | 6.1 ± 0.1 | 167.7 ± 0.7 | 0.284 ± 0.002 | 0.18 ± 0.07 |
| 2018-04-21 | 43.4 ± 0.1 | 5.8 ± 0.1 | 204.3 ± 0.5 | 0.257 ± 0.003 | 0.12 ± 0.07 |
| 2018-04-25 | 42.2 ± 0.1 | 5.5 ± 0.3 | 190.5 ± 0.9 | 0.282 ± 0.003 | 0.19 ± 0.13 |
| 2021-04-13 | 43.9 ± 0.2 | 5.0 ± 0.6 | 184.8 ± 1.9 | 0.183 ± 0.003 | 0.28 ± 0.08 |
| 2021-04-18 | 42.7 ± 0.2 | 8.1 ± 0.2 | 195.6 ± 0.6 | 0.179 ± 0.005 | 0.17 ± 0.01 |

Fig. 3 medians (xsringauss): 2017-04-11 R_out 24.3, R_in 20.4 μas, ε 0.35, fade 0.59, φ 181°;
2018-04-21 R_out 25.5, R_in 19.2 μas, ε 0.49, fade 0.43, φ 206°.

## Stages and criteria

Circular differences for angles. "σ_T" is the paper's quoted ± for that cell.

### Stage 1: their numbers from their products (local, cheap)

Image: the pixelwise median of the 50 published 225 μas samples per fiducial day, as published (no
re-alignment; Methods says the median is of aligned reconstructions, so the archive samples are taken as aligned). Secondary: REX on each of the 50 samples, mean ± SD.

Three extractors on the same image:
(T) their `ringFitting.py` (GenDIReCT archive) in `.venv-basis`, through `external_bridge/m87_worker.py`;
(X) `crosscheck.ring_extraction`, numpy/scipy from Methods eqs 4–9, written without reading theirs or virgil's;
(V) `virgil.metrics.ring_features` through `virgil_bridge/m87.py`.

| Id | Criterion | Kind |
| --- | --- | --- |
| 1a | X vs Table 1: \|Δ 2r_pk\| ≤ max(σ_T, 1.0 μas) and \|Δη\| ≤ max(σ_T, 5°), each fiducial day | reproduction of the paper (one field of three: a fail is a finding about the field dependence or the paper, not a tuning target) |
| 1b | X vs T: \|Δ 2r_pk\| ≤ 0.5 px (1.76 μas), \|Δη\| ≤ 2°, \|Δ w_r\| ≤ 1 px, \|ΔA\| ≤ 0.02 | our implementation of eqs 4–9 against theirs |
| 1c | V vs X: \|Δ 2r_pk\| ≤ 0.5 px, \|Δη\| ≤ 2°, \|ΔA\| ≤ 0.02 | virgil check; width and f_c reported only (definitions differ) |
| 1d | X on a rendered analytic ring (crescent with known r, η, A): \|Δ 2r_pk\| ≤ 0.25 px, \|Δη\| ≤ 1° | reference test of X |

### Stage 2: their geometric fit with their code (OzSTAR)

`runBilby.py` from BASIS v0.1.0 on the three fiducial UVFITS, models `xsringauss,gauss,gauss`, 128 px, 225 μas,
dynesty 250 live points; two seeds; two noise settings: (i) paper (`noise_frac=0`, floor 0), (ii) their example
(`noise_frac=0.05`, floor 1e-4). REX on 200 posterior renders with X.

| Id | Criterion |
| --- | --- |
| 2a | Setting matching the paper (whichever of i, ii does): 2r_pk within max(3σ_T, 0.5 μas) of ED Table 1, η within max(3σ_T, 3°) |
| 2b | Fig. 3 medians: R_out, R_in within 1 μas; ε, fade within 0.1; φ within 5° |
| 2c | Seed-to-seed: if the two seeds' 2r_pk differ by more than 2 σ_T, the quoted errors are too small (finding, expected from the review) |

Reported: χ²_CI/N at the median parameters on quoted σ.

### Stage 3: the same likelihood in virgil (OzSTAR GPU)

UVFITS → MS (casatools `importuvfits`, `.venv-alma`) → `ComplexVisData.read_ms`; per scan, invariants built in
`virgil_bridge` (reference = first station, as BASIS), model = `Crescent` + thick-side `EllipticalGaussian` +
floor `GaussianDisk` in `Blurred`/`Rotated`, + background `GaussianDisk` + 2 `EllipticalGaussian` (no
per-Gaussian blur). Likelihood (a): BASIS's diagonal σ (read from their worker); (b): full `C = J Σ Jᵀ`
(`jax.jacfwd`). Uniform priors on BASIS's bounds. `virgil.smc`.

| Id | Criterion |
| --- | --- |
| 3a | Invariants of a fixed model: virgil bridge vs ClosureInvariants (their worker), max abs difference ≤ 1e-8 (cheap, local once `.venv-basis` exists) |
| 3b | (a) vs stage 2 setting (i): medians of R_out, R_in, ε, fade, φ within 1 posterior SD of stage 2 |
| 3c | (b): posterior SD of 2r_pk and φ reported against (a); prediction recorded now: (b)/(a) ≥ 2 |

### Stage 4: virgil native (OzSTAR GPU)

Complex visibilities, per-scan Laplace-marginalised antenna gains (broad priors), `cal.sigma` fitted; Jeffreys
priors (log-uniform radii, widths, fluxes; uniform angles, ε, fade). Models: m-ring
(`MultipoleDisk(GaussianRing, Harmonics(m ≤ 2))` + background `GaussianDisk`) and the stage 3 composite, each with
0, 1, 2 nuisance `EllipticalGaussian`s; SMC log Z picks.

Reported, no pass/fail (no reference to pass): raw χ²/N on the quoted σ first, then `cal.sigma`; 2r_pk and η per
day from REX (X) on posterior renders and from parameters; evidence table. Prediction recorded now: per-day
2r_pk posterior SD ≥ 3 × ED Table 1's σ.

### Stage 5: nonparametric (OzSTAR)

virgil RML (MEM, TV) and CLEAN with marginalised gains, fields of view 128 / 225 / 450 μas, starts flat /
40 μas Gaussian / 100 μas Gaussian. REX (X) on every image.

| Id | Criterion |
| --- | --- |
| 5a | Spread of 2r_pk over the nine (field, start) runs per day and imager ≤ Table 1 σ_T for that day (the paper's robustness argument for GenDIReCT over RML) |
| 5b | Mean 2r_pk per day within max(2σ_T, 2 μas) of Table 1 |

## Cross-validation days

2017-04-05/06/10, 2018-04-25: stage 2 and 4 only, after the fiducial days. 2021-04-13 needs 2026-D01-01.

## Change log

| Commit | What changed | Changed after seeing a result | Note |
| --- | --- | --- | --- |
| (this commit) | first version | no | criteria written before any stage 1 extraction was run |
