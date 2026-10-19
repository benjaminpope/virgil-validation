# ALMA IM Lup: implementation spec for P1, P2 and P2v

This spec turns [`alma_imlup.md`](alma_imlup.md) (second review, B1–B10)
into work for three Sonnet workers. Where it and the plan disagree, the
plan's decisions hold and this spec is the bug.

| Worker | Repo | Branch | Depends on | Phase |
|---|---|---|---|---|
| W1 | virgil | `radio-complexvis` | — | P2 |
| W2 | virgil-validation | `alma-ingest` | — | P1 |
| W3 | virgil-validation | `alma-radio-validation` | W1 and W2 merged | P2v |

W1 and W2 run at the same time and share nothing but the conventions below.
W2 must not read W1's code (the independence rule); both read this spec.

**`virgil.radio` is instrument-neutral (Ben, 2026-10-19).** ALMA is the
first user and ATCA will follow, so nothing in virgil hard-codes ALMA: no
default dish size, beam factor, reference frequency, band or offset scale.
Anything instrument-specific is an argument, or is derived from the data;
the IM Lup analysis in virgil-validation passes ALMA's values. Names follow
the measurement set, not ALMA: an "observation" (`OBSERVATION_ID`), not an
"execution block". Plan text that says "EB" means an observation here.

## Shared conventions (to be pinned by tests, not assumed)

- **Visibility.** V(u, v) = Σ I(x, y) exp(−2πi(u x + v y)), with x towards
  +RA (East) and y towards +Dec (North), u and v in wavelengths. This is
  virgil's documented convention (`image_visibilities`, `offset_phase`).
  Whether the MS `DATA` column equals V or V* under it, and the sign of u
  relative to `UVW[0]`, is **unknown** until the synthetic-MS test decides
  it. Each reader exposes a `conjugate` choice internally; the test picks
  it; the docs record it.
- **u, v from the MS.** u = UVW[0] · ν_chan / c, v = UVW[1] · ν_chan / c, per
  channel (`SPECTRAL_WINDOW.CHAN_FREQ`), c = 299 792 458 m/s.
- **Which samples.** Cross-correlations only (`ANTENNA1 != ANTENNA2`). One
  sample per (row, channel). Keep it only if `FLAG_ROW` is false, neither
  of the two parallel-hand polarisations is flagged in `FLAG`, and both
  weights are positive.
- **Polarisations.** From `POLARIZATION.CORR_TYPE`: XX and YY (9, 12) or RR
  and LL (5, 8). Average with weights: V = (w₁V₁ + w₂V₂)/(w₁ + w₂),
  w = w₁ + w₂. With equal weights this is Stokes I = (XX + YY)/2.
- **Weights.** `WEIGHT_SPECTRUM` if the column exists and is non-empty,
  otherwise `WEIGHT` broadcast over channels. A weight is 1/σ² for each of
  Re and Im. That is CASA's documented meaning; the noisy synthetic MS and
  the model-free σ test it.
- **Labels.** `ddid` = `DATA_DESC_ID`; `obs` = `OBSERVATION_ID`. Both are
  kept as the MS gives them and relabelled 0..n−1 in sorted order, with the
  originals kept.
- **Fields.** One field only. If the selection holds more than one
  (a mosaic), the reader raises unless a `field` is given. Mosaics are out of
  scope for now.
- **Units.** Frequencies in Hz, times in MJD seconds (`TIME`), offsets in
  milliarcseconds, flux in Jy.

## W1: `virgil.radio` (virgil PR)

New module `src/virgil/radio.py`, exported from `virgil/__init__.py`, with
an optional extra `radio = ["casatools>=6.6"]` in `pyproject.toml`. If
casatools has no wheel for one of virgil's Python versions, add a
`python_version` marker and say so in the docs. Read before writing:
`virgil.orbits.PositionData.term` and `_Term` (the likelihood-term
protocol), `virgil.fitting.fit` (how `likelihoods`, `has_log_norm` and
`log_norm` enter the loss: loss = ½χ² + log_norm + penalty − log prior),
`virgil.models.Image`, `virgil._geometry.image_visibilities`,
`offset_phase` and `set_ft_defaults`.

### `ComplexVisData` (an `equinox.Module`)

Fields: `u`, `v` (float64, wavelengths), `vis` (complex), `weight`,
`freq` (Hz), `ddid`, `obs` (int32, relabelled), static `ddid_ids`,
`obs_ids` (original ids), static `ddid_obs` (the observation of each ddid), and
optional `time`, `ant1`, `ant2`, `chan` (None when unknown).

- `from_arrays(u, v, vis, weight, freq, ddid=0, obs=0, **optional)`:
  validates shapes, positive weights, finite values.
- `read_ms(path, ddids=None, field=None, stride=1, datacolumn="DATA")`: the reader of
  the shared conventions above, through `casatools.table` (lazy import, a
  clear error naming the `radio` extra). It reads one ddid at a time with a
  table query, keeps every `stride`-th row, and never holds more than one
  ddid's columns at once.
- `__len__`, `select(ddids=None, obs=None)`, `subset(mask)`.
- `model(scene, values, nuisance)`: the complex model at every point,
  M_k = F · g_o · (ν_k/ν₀)^α · exp(−2πi(δα_o u_k + δδ_o v_k)) ·
  T(u_k, v_k) · V_scene(u_k, v_k). V_scene is
  `scene.model(u·λ, v·λ, λ)` with λ = c/ν, so that a chromatic scene still
  works. The ramp uses `offset_phase` with the same sign as a component's
  `dra`, `ddec`. T is the optional Gaussian taper (below).
- `term(scene, nuisance, mask=None, chunk_size=10_000_000)`: a likelihood
  term for `fit(likelihoods=[...])`, `scene` a callable of the fitted
  values returning a `SourceModel`, as `PositionData.term` takes `orbit`.
  - `__call__(values)`: whitened residuals [Re, Im] of
    (vis − M)·√w·m / s_d(k), a real vector of length 2M (m the 0/1 mask).
  - `has_log_norm` is true when σ factors are fitted;
    `log_norm(values)` = Σ_k m_k · 2 ln s_d(k).
  - `loglike(values)`, the normalised Gaussian log density, for
    `numpyro_model`.
  - Points are padded with zero weight to a multiple of `chunk_size` and
    evaluated with `jax.lax.map` over chunks. Never `vmap` the transform
    (nufftax batches only one axis).
- `chi2_by_ddid(scene, values, nuisance, mask=None)`: float64 χ² per
  ddid, with N_d, for cross-validation scores.

### `CalibrationNuisance` (static configuration)

`CalibrationNuisance(data, ref_obs=0, gains=True, offsets=True,
sigmas=True, alpha=True, nu0_hz=None, taper_fwhm_mas=None,
prefix="cal")`. `nu0_hz=None` means the weighted mean frequency of `data`. Sites, all keys of `priors` that a function model must
accept:

| Site | Shape | Default prior | When not fitted |
|---|---|---|---|
| `cal.flux` | () | `LogUniform(1e-4, 1e2)` Jy | always fitted |
| `cal.gain` | (n_obs − 1,) | `LogUniform(0.5, 2)` | 1 |
| `cal.offset` | (n_obs − 1, 2), mas | `Uniform(−θ, θ)`, θ = 1/(2 q_min) from the data unless given | 0 |
| `cal.sigma` | (n_ddid,) | `LogUniform(0.1, 10)` | 1 |
| `cal.alpha` | () | `Uniform(0, 5)` | 0 |

The reference observation has g = 1 and zero offset. `priors(**bounds)`
returns this dict, with any bound overridable, `defaults()` the starting values. Check early that `fit` accepts
vector-valued sites; if it does not, use one scalar site per observation and ddid
(`cal.gain_3`, …) and say so in the PR. The taper T = exp(−(π θ q)²/(4
ln 2)), with θ the FWHM in radians and q = √(u² + v²), is a Gaussian
convolution of the image, as MPoL's `GaussConvFourier`.

### Helpers

- `primary_beam(npix, pixel_scale_mas, freq_hz, dish_m, factor)`: the
  log-attenuation −4 ln 2 r²/FWHM² on the pixel grid of `pixel_offsets`,
  FWHM = factor · c/(freq · dish_m). It is added to an `Image`
  log-brightness. `dish_m` and `factor` have no defaults; the docs give
  ALMA (12 m, 1.13) and ATCA (22 m) as examples, with sources.
- `dartboard_folds(data, k=5, n_q=12, n_phi=16, seed=0)`: returns the fold
  (0..k−1) and the q-annulus index of every point. Annuli are log-spaced
  between the smallest and largest q; azimuth φ = arctan2(v, u) mod π, so a
  point and its Hermitian partner share a cell. Cells are shuffled into
  folds with `numpy.random.default_rng(seed)`.

### Tests (`tests/test_radio.py`) and docs

1. `model` against `Image(ft="dft")` in float64 (1e-10 relative) and
   against `ft="nufft"` (relative l2 ≤ 3 eps), for a 64² random image.
2. The offset ramp equals shifting the scene by `dra`, `ddec`.
3. ½χ² + log_norm equals −Σ log N(Re), −Σ log N(Im) up to a constant
   independent of the parameters, checked at two different s.
4. A masked point changes neither χ² nor log_norm; N_d counts unmasked
   points.
5. Gradients with respect to log-brightness and every nuisance site agree
   with central finite differences (float64).
6. A small `fit` (64² image of a Gaussian disk, 2 observations, 3 ddids, 2 000
   points, `data=()`, the term in `likelihoods`, a function model,
   `method="lbfgs"`) recovers g, the offset, s and α within 3σ.
7. `dartboard_folds`: Hermitian partners in the same fold; every fold
   non-empty.
8. `primary_beam`: half power at r = FWHM/2.
9. `read_ms` on a tiny MS from `casatools.simulator` (skipped without
   casatools): counts, u against `UVW` and frequency, and the convention
   test: CASA predicts a point source at +300 mas East, +100 mas North,
   and `model` with `PointSource(dra=300, ddec=100)` must match the read
   data to 1e-6. The opposite conjugation must fail.

Docs: an API page and a short "Radio interferometry" guide page in virgil's
docs (no science targets, no link to design notes), written for any
measurement set, with ALMA and ATCA as examples where an instrument value
is needed. No module, class, argument or default names an instrument.

## W2: independent ingest and checks (virgil-validation PR)

No virgil import anywhere in this work. Read `AGENTS.md`,
`src/external_bridge/_subprocess.py` and `mpol_worker.py` first.

### Environment

`scripts/setup_external.sh alma` installs pinned `casatools`, `casadata`
(or `casaconfig`, whichever the casatools version wants for measures data)
and `visread` into `.venv-alma`. Use the Python version casatools has
wheels for (3.11 worked on 2026-10-17). Measures data go inside the venv
directory, not into `~/.casa` if avoidable.

### `src/external_bridge/alma_worker.py` (runs in `.venv-alma`)

Tasks, JSON in and out through `_subprocess.run`:

- `simulate_ms`: a small MS made from scratch with `casatools.simulator`.
  8 ALMA antennas (positions written in the task), 2 spectral windows (230
  and 245 GHz, 4 channels of 125 MHz), dual linear feeds, field at
  15h56m09.20s −37°56′06.42″, 6 s integrations for 1 h. A component
  list with two points, 10 mJy at (+300, +100) mas and 5 mJy at
  (−200, −400) mas (East, North), predicted by CASA (`sm.predict`).
  Option `noise_jy`: `sm.setnoise(mode="simplenoise")` and `sm.corrupt()`,
  then `WEIGHT` = 1/σ² checked or set explicitly.
- `read`: the reader of the shared conventions, written from CASA's MS
  definition (not from visread). One `.npz` per ddid: `u`, `v` (float64),
  `re`, `im`, `weight` (float32), `freq` (float64), `ant1`, `ant2`,
  `chan` (int16), `time` (float64), `scan` (int32), and scalars `obsid`,
  `ddid`, `chan_width_hz`, `t_int_s`. Options `ddids`, `stride`,
  `conjugate` (default from the convention test).
- `visread_read`: the same output through `visread`, for comparison.
- `info`: per ddid, N_vis, channels, Δν, integration time, frequencies.

### `src/crosscheck/alma.py` (numpy only)

- `dft_dirty_image(u, v, vis, weight, npix, pixel_scale_mas)`: the float64
  natural-weighted dirty image Σ w Re(V e^{+2πi(ux + vy)}) / Σ w, chunked,
  orientation as `crosscheck.sky` (column 0 is East).
- `gridded_dirty_image(...)`: nearest-cell gridding of V and its
  Hermitian partner, then `numpy.fft`, for data too big for the DFT. A
  diagnostic only: it is tapered by sinc(x Δu).
- `successive_difference_sigma(npz)`: for each ddid, pairs (i, i+1) of
  consecutive integrations on the same (ant1, ant2, chan, scan) with
  Δt ≤ 1.5 t_int, non-overlapping. Returns
  ŝ² = mean(|V_i − V_{i+1}|² / (2(1/w_i + 1/w_{i+1}))), its standard
  error, and the number of pairs. ŝ = 1 means the weights are right.

### Docs, tests, data

- `docs/alma_conventions.md`: every convention above, as pinned by the
  tests, with CASA documentation links.
- `tests/test_alma_conventions.py` (tier B, skipped without `.venv-alma`):
  the dirty image of the noiseless synthetic MS peaks within one pixel of
  both sources, the brighter one at (+300, +100) (root `casa`); the
  conjugated mapping as `kind="control"`; on a noisy MS (1 mJy), ŝ is
  1 within 3 standard errors (root `statistics`); `read` agrees with
  `visread_read` to float32 precision (root `golden:visread`).
- `tests/test_alma_crosscheck.py` (tier A): the crosscheck functions on
  numpy-made data with known answers (`kind="reference"`).
- `scripts/alma_imlup_subsample.py`: reads `~/data/alma_imlup/ms/
  IMLup_continuum.ms` with `stride=1000` (≈ 4 × 10⁴ visibilities) into
  `~/data/alma_imlup/subsample_1000/`. Run once, one process, and record
  `info` for all 26 ddids in `docs/alma_conventions.md`. Data never go in
  git.
- `src/evidence/__init__.py`: add the root `casa` (CASA's simulator and
  predict). `trust/graph.yml`: node `crosscheck.alma`.

## W3: validating `virgil.radio` (virgil-validation PR `alma-radio-validation`, built)

Built against virgil 9ee600f (`virgil-astro[radio]`), in its own venv, without
moving `VIRGIL_REF` (CI's pin, 9061a1d, has no `virgil.radio`: every test here
skips there, via `pytest.importorskip("virgil.radio")`, until Ben bumps it).

Files: `src/virgil_bridge/alma.py` (`load_npz`: W2's per-ddid files into
`ComplexVisData.from_arrays`; scene builders `image_scene`, `fixed_image_scene`,
`point_scene`, the numpy `ring_and_blob`; it also points casatools at
`.venv-alma`'s measures data), `tests/test_alma_radio.py`, nodes
`virgil.radio.*` and the planned pipeline `alma-imlup` in `trust/graph.yml`.

| # | Test | Tier | Roots | What |
|---|---|---|---|---|
| 1 | `test_read_ms_agrees_with_our_reader_*` (clean, noisy, IM Lup at stride 1000) | B | casa | `read_ms` (casatools 6.7.6.14) against W2's `read` (6.7.0.31): counts per ddid, u and v to 1e-9, V and weight to float32, frequency, time, antennas, channels, labels |
| 1b | `test_read_ms_returns_the_conjugate_...` and its `control` | B | casa | CASA-predicted point sources; property `conjugation` |
| 2 | `test_model_matches_the_float64_dft_...`, `..._nufft_...` (eps 1e-12, 1e-6), control (offset sign), default nu0 | A | mathematics | F, g, offsets, alpha, taper and a point source on a 64² image, 5000 uv points, 4 ddids, 3 observations |
| 3 | `test_primary_beam_...`, `test_adding_the_beam_...` | A | mathematics | ALMA (12 m, 1.13) and ATCA (22 m); beam times image against the crosscheck DFT |
| 4 | `test_half_chi2_plus_log_norm_...`, `test_masked_points_...` | A | statistics | against `scipy.stats.norm.logpdf` at two parameter sets, with a mask |
| 5 | `test_dartboard_folds_...`, `test_chi2_by_ddid_...` | A | mathematics | Hermitian partners, log-q annuli, occupied folds, one fold per uv cell, held-out chi² |
| 6 | `test_sigma_fit_on_casa_noise_...` | B | statistics, casa | `cal.sigma` on CASA's 1 mJy noise: 1 within 3 SE and equal to the successive-difference estimate |
| 7 | `test_nuisances_are_recovered_...` | B | statistics | injection-recovery on ≤ 10⁴ points of IM Lup's uv, numpy Fisher matrix |

Choices: the documentation was unambiguous everywhere it was tested (image
centre at ((n-1)/2, (n-1)/2); annuli edges `logspace(q_min, q_max, n_q + 1)`;
default nu0 the weighted mean frequency); virgil's behaviour matched each
reading first time. In test 7 the truth flux is 2.5 Jy (ten times IM Lup's),
so that offsets of a few mas are measurable with 10⁴ points of weight ~10³ Jy⁻²,
and the offset prior is narrowed to ±20 mas with `priors(offset=...)`. At
IM Lup's own flux (0.25 Jy, checked once by hand) a cold L-BFGS start from the
defaults and the default offset prior (±1/(2 q_min), about 8000 mas) ended in
a phase-wrapped local mode (gains at their bounds, offsets of hundreds of
mas) with a log-likelihood 1000 below the truth's; at 2.5 Jy the default
prior also recovers (max pull 2.7, recorded as an observation). The free-image fit (MaxEntropy) is reported only.

## W6: the likelihood equals MPoL's (virgil-validation PR `alma-mpol-likelihood`, built)

Ben, 2026-10-19: "We should make sure that the likelihood matches MPoL's."
`tests/test_alma_mpol.py` calls the IM Lup example's own `Net` and
`mpol.losses` (examples commit `5a7c3f0b`, mpol 0.3.1, torch 2.14.1,
torchkbnufft 1.5.2, float64) through `src/external_bridge/mpol_worker.py`,
and virgil.radio (dcf1db8) through `src/virgil_bridge/mpol.py`. The map is in
`docs/alma_conventions.md`, Part 3 (s = g·s_M, δ in arcsec × 1000, taper
FWHM × 1000, `virgil = unpacked[::-1, ::-1]` with dra = ddec = −dx/2).
Tests: (1) loss normalisation to 1e-12 with the control s = s_M; (2) forward
model, MPoL vs virgil vs exact DFT, at 256² and 5 mas; (3) objective and
gradients (torch autograd vs JAX, chain rule through the map); (4, tier B)
the objective on all 26 ddids of the stride-1000 subsample. Result: no
virgil finding; MPoL's default NuFFT is 6e-4 from the exact transform (P12),
and its docstring has a slip (P11, draft upstream text in the PR). The new
tests skip in CI (its pinned virgil has no `virgil.radio`; `VIRGIL_REF` is not bumped).
