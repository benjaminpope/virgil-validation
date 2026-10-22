# Measurement-set conventions (CASA MS; ALMA and DSHARP IM Lup)

How a CASA measurement set (MS) maps to complex visibilities V(u, v), as
pinned by `tests/test_alma_conventions.py` (tier B: needs `.venv-alma`,
`bash scripts/setup_external.sh alma`) and `tests/test_alma_crosscheck.py`
(tier A). The reader is `src/external_bridge/alma_worker.py` (task `read`),
written from CASA's [MS definition](https://casacore.github.io/casacore-notes/229.html)
through `casatools.table`, not from visread. Checks are in
`src/crosscheck/alma.py` (NumPy only). Neither imports virgil.

This page has two parts. **Part 1** is MS-generic: it holds for any CASA
measurement set (ALMA, ATCA, VLA, ...), and the reader does not assume an
instrument. **Part 2** is specific to ALMA and the DSHARP IM Lup data. The
synthetic MS is built with ALMA antennas and frequencies only because it
needs some telescope; none of the conventions depend on that choice.

Versions: casatools 6.7.0.31, casadata 2025.9.22 (measures data, inside
`.venv-alma` through `casasiteconfig.py`; nothing is written to `~/.casa`),
casaconfig 1.5.2, visread 0.0.5, Python 3.11.

# Part 1: MS-generic conventions

## The result: conjugation and the sign of u

CASA's own `simulator` + `predict` put 10 mJy at (+300, +100) mas and 5 mJy
at (−200, −400) mas (East, North) into a synthetic MS (8 antennas, two
spectral windows at 230 and 245 GHz, dual linear feeds, 1 h of 6 s
integrations). With

    V(u, v) = Σ I(x, y) exp(−2πi (u x + v y)),   x towards East, y towards North,

and u = `UVW[0]` ν / c, v = `UVW[1]` ν / c:

- **`DATA` is the complex conjugate of V.** V = conj(`DATA`) matches the
  point-source model to 1 × 10⁻⁷ Jy (float32 rounding) in every sample and
  both windows; `DATA` itself misses by twice the visibility amplitude.
  Equivalently, V(u, v) = `DATA` at (−u, −v): the sign of u is the sign of
  `UVW[0]`, and the conjugation is the whole story (V is Hermitian for a
  real sky, so one flip of u and v equals one conjugation; they are not two
  independent choices).
- **Dirty image.** The natural-weighted dirty image of conj(`DATA`) puts the
  bright source at (+300, +100) mas, within one 25 mas pixel, and the faint
  one at (−200, −400) mas. The control, `DATA` taken as V, puts the
  bright peak at (−300, −100) (the sky mirrored through the phase centre).
- The `read` task takes `conjugate=True` by default for this reason.
  visread (which MPoL uses) also conjugates `DATA`: the two readers give
  identical samples on the synthetic MS (positions to float64 round-off,
  V and weight to float32).

## Other conventions (as the reader applies them)

- **Observations and fields.** Observations are labelled by
  `OBSERVATION_ID` (`obsid` in the npz files), spectral windows by
  `DATA_DESC_ID` (`ddid`). `read` takes an optional `field`; with none, the
  selection must hold a single `FIELD_ID`, and a second one (a mosaic)
  raises an error naming the fields
  (`test_a_selection_with_more_than_one_field_raises_unless_a_field_is_given`).
- **Samples.** Cross-correlations only (`ANTENNA1 != ANTENNA2`), one per
  (row, channel). Dropped: `FLAG_ROW` true; either parallel-hand
  correlation flagged in `FLAG`; either weight not positive
  (`test_samples_are_dropped_for_flags_flag_row_and_zero_weights`).
- **Polarizations.** `POLARIZATION.CORR_TYPE` must contain XX, YY (9, 12) or
  RR, LL (5, 8). V = (w₁V₁ + w₂V₂)/(w₁ + w₂), w = w₁ + w₂.
- **Data column.** `read` takes `datacolumn` (default `DATA`; case-insensitive; a column the MS lacks is an
  error) and records it in its result. The IM Lup MS has `DATA`, `CORRECTED_DATA`, `MODEL_DATA`; MPoL's visread
  export reads `CORRECTED_DATA`, which differs from `DATA` by up to 1.6 Jy, so IM Lup is read with
  `datacolumn=CORRECTED_DATA` (`design/alma_imlup_p3.md`, D1). The conjugation above applies to whichever column is read.
- **Weights.** `WEIGHT_SPECTRUM` if the column exists and its first cell is
  defined with the shape of `DATA`; otherwise `WEIGHT` per row, repeated over
  channels. A weight is 1/σ² of each of Re and Im of a correlation product
  (the casacore MS definition above says `WEIGHT` is the weight per
  polarization; the meaning is established by the noisy synthetic MS, not assumed).
  On the noisy synthetic MS, `setnoise(mode="simplenoise", simplenoise="1mJy")`
  puts σ = 1 mJy on each of Re and Im of each of XX and YY; the combined
  visibility then has σ = 1 mJy/√2, weight w₁ + w₂ = 2 × 10⁶ Jy⁻², the
  scatter about the known model is 0.707 mJy (2%), and the model-free ŝ below
  is 1.000 ± 0.005 in both windows.
- **Frequencies and units.** Hz per channel from `SPECTRAL_WINDOW.CHAN_FREQ`
  (channels are kept in MS order, ascending or not); time in MJD seconds
  (`TIME`); u, v in wavelengths; V in Jy. `ddid` = `DATA_DESC_ID`,
  `obsid` = `OBSERVATION_ID` (the npz files keep the MS's values).
- **Output** (`ddid_NN.npz`): `u`, `v`, `freq` (float64), `re`, `im`,
  `weight` (float32), `ant1`, `ant2` (int32), `chan` (int16), `time`
  (float64), `scan` (int32); scalars `obsid`, `ddid`, `chan_width_hz`,
  `t_int_s` (median `INTERVAL`).
- **Stride.** `stride` keeps every stride-th cross-correlation row of the ddid
  before the flag cuts; all of its channels are kept.

## The model-free weight check

`crosscheck.alma.successive_difference_sigma`: for consecutive integrations
(Δt ≤ 1.5 t_int) of the same scan, baseline and channel, non-overlapping
pairs give

    ŝ² = mean( |V_i − V_{i+1}|² / (2 (1/w_i + 1/w_{i+1})) ),

which is 1 when w = 1/σ² for each of Re and Im. It needs adjacent rows,
so it cannot be run on a strided subsample: `scripts/alma_imlup_subsample.py
--sigma` reads each ddid whole, one at a time, reduces it and discards
it. Two cautions. ŝ is an upper bound on the true weight scale: any change
of the sky or the atmospheric phase over 6 s adds to |ΔV|² (a drift of
0.3 σ between neighbours adds 0.25 to ŝ², tier-A test). And ŝ² has standard
error std(r)/√n over the pairs r; the table quotes ŝ ± SE(ŝ) = SE(ŝ²)/(2ŝ).

# Part 2: ALMA and DSHARP IM Lup

The synthetic MS: 8 ALMA antennas (a fixed layout of about 1.6 km), two
windows at 230 and 245 GHz with 4 channels of 125 MHz, 6 s integrations, dual
linear feeds; IM Lup's phase centre 15h56m09.20s −37°56′06.42″.

## IM Lup (DSHARP continuum) measured on 2026-10-19

`~/data/alma_imlup/ms/IMLup_continuum.ms`, 26 ddids, 7 observations (the
short-baseline EBs 0–4, the long-baseline EBs 5 and 6), read with our
reader. N_vis counts the retained samples at stride 1 and equals the table of
`design/alma_imlup.md` for every ddid (42,421,131 in all). The strided
subsample (stride 1000, 42,424 samples, `~/data/alma_imlup/subsample_1000/`,
never in git) comes from the same reader; the successive-difference
check uses every row.

| ddid | obs | chans | Δν (MHz) | t_int (s) | ν range (GHz) | rows | N_vis (stride 1) | N_vis (stride 1000) | pairs | ŝ |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 1 | 58.594 | 6.048 | 231.392–231.392 | 92,103 | 92,103 | 93 | 45,320 | 1.123 ± 0.003 |
| 1 | 0 | 1 | 58.594 | 6.048 | 231.304–231.304 | 92,103 | 91,476 | 91 | 44,964 | 1.146 ± 0.003 |
| 2 | 0 | 1 | 58.594 | 6.048 | 231.203–231.203 | 92,103 | 92,103 | 93 | 45,320 | 1.126 ± 0.003 |
| 3 | 0 | 1 | 58.594 | 6.048 | 230.520–230.520 | 92,103 | 91,563 | 93 | 45,026 | 1.127 ± 0.003 |
| 4 | 0 | 4 | 117.188 | 6.048 | 234.742–235.093 | 92,103 | 368,412 | 372 | 181,280 | 1.129 ± 0.001 |
| 5 | 1 | 1 | 117.188 | 6.048 | 245.511–245.511 | 94,938 | 66,426 | 59 | 32,700 | 1.140 ± 0.003 |
| 6 | 1 | 1 | 117.188 | 6.048 | 245.542–245.542 | 94,938 | 66,426 | 59 | 32,700 | 1.141 ± 0.003 |
| 7 | 1 | 1 | 58.594 | 6.048 | 241.540–241.540 | 94,938 | 66,426 | 59 | 32,700 | 1.144 ± 0.003 |
| 8 | 1 | 1 | 58.594 | 6.048 | 241.679–241.679 | 94,938 | 66,251 | 59 | 32,623 | 1.138 ± 0.003 |
| 9 | 1 | 1 | 58.594 | 6.048 | 241.746–241.746 | 94,938 | 66,276 | 59 | 32,648 | 1.132 ± 0.003 |
| 10 | 1 | 1 | 58.594 | 6.048 | 241.770–241.770 | 94,938 | 66,126 | 59 | 32,572 | 1.139 ± 0.003 |
| 11 | 2 | 16 | 125.000 | 6.048 | 241.080–242.955 | 97,567 | 1,365,938 | 1,372 | 670,880 | 1.106 ± 0.001 |
| 12 | 2 | 16 | 125.000 | 6.048 | 242.810–244.685 | 97,567 | 1,365,938 | 1,372 | 670,880 | 1.095 ± 0.001 |
| 13 | 3 | 16 | 125.000 | 6.048 | 241.064–242.939 | 129,935 | 1,815,674 | 1,806 | 871,682 | 1.077 ± 0.001 |
| 14 | 3 | 16 | 125.000 | 6.048 | 242.793–244.668 | 129,935 | 1,685,983 | 1,677 | 809,419 | 1.055 ± 0.001 |
| 15 | 4 | 1 | 58.594 | 6.048 | 231.217–231.217 | 221,344 | 219,248 | 221 | 107,344 | 1.128 ± 0.002 |
| 16 | 4 | 1 | 58.594 | 6.048 | 230.534–230.534 | 221,344 | 209,358 | 206 | 102,360 | 1.134 ± 0.002 |
| 17 | 4 | 16 | 125.000 | 6.048 | 231.761–233.636 | 221,276 | 3,088,092 | 3,094 | 1,512,476 | 1.065 ± 0.000 |
| 18 | 5 | 16 | 125.000 | 6.048 | 231.644–233.519 | 282,509 | 3,955,126 | 3,962 | 1,776,096 | 1.665 ± 0.001 |
| 19 | 5 | 16 | 125.000 | 6.048 | 244.032–245.907 | 282,509 | 3,955,126 | 3,962 | 1,776,096 | 1.668 ± 0.001 |
| 20 | 5 | 16 | 125.000 | 6.048 | 245.949–247.824 | 282,509 | 3,955,126 | 3,962 | 1,776,096 | 1.664 ± 0.001 |
| 21 | 5 | 8 | 117.188 | 6.048 | 230.109–230.930 | 282,509 | 2,260,072 | 2,264 | 1,014,912 | 1.747 ± 0.001 |
| 22 | 6 | 16 | 125.000 | 6.048 | 231.651–233.526 | 341,408 | 4,779,712 | 4,788 | 2,147,572 | 1.988 ± 0.001 |
| 23 | 6 | 16 | 125.000 | 6.048 | 244.039–245.914 | 341,408 | 4,779,712 | 4,788 | 2,147,572 | 1.993 ± 0.001 |
| 24 | 6 | 16 | 125.000 | 6.048 | 245.956–247.831 | 356,929 | 4,997,006 | 4,998 | 2,245,194 | 1.991 ± 0.001 |
| 25 | 6 | 8 | 117.188 | 6.048 | 230.117–230.937 | 356,929 | 2,855,432 | 2,856 | 1,282,968 | 2.081 ± 0.001 |

Every ddid of this MS has no `WEIGHT_SPECTRUM` column in use: the reader took
`WEIGHT` for all 26 (the `weight_source` field of `info`), so the table does not
depend on the `WEIGHT_SPECTRUM` branch.

Reading the ŝ column: the short-baseline EBs 0–4 give ŝ = 1.05–1.15; the
long-baseline EB 5 gives 1.66–1.75 and EB 6 gives 1.99–2.08. ŝ overestimates
the weight scale wherever the sky or the atmospheric phase changes within 6 s,
and the long baselines see the phase change fastest, so these are upper
bounds on the true scale, not estimates of it; the bias is bounded only by
the short-baseline EBs, where ŝ² − 1 ≲ 0.3. The per-ddid σ factors of the
multi-EB fit are compared with these ratios, and are not replaced by them.

# Part 3: MPoL and virgil.radio

Pinned by `tests/test_alma_mpol.py` (needs `.venv-mpol` and the examples clone,
`bash scripts/setup_external.sh mpol`; tests skip without them or without
`virgil.radio`). MPoL's side is the IM Lup multi-EB example's own `Net`
(MPoL-dev/examples, commit `5a7c3f0b373f62341ca4baf5869f95e916bbabb3`,
`04-IMLup-multi-EB/mpol_analysis/src/sgd.py`) with `mpol.losses`, imported
unmodified by `src/external_bridge/mpol_worker.py` in float64. virgil's side
is `src/virgil_bridge/mpol.py`.

## The parameter map

| MPoL (`Net`) | virgil.radio | Note |
|---|---|---|
| image `softplus(base_cube)`, Jy/arcsec², packed (`fftshift` layout) | `Image(log_brightness)` with `cal.flux = Σ I · dx²` | `dx` in arcsec; virgil's pixels are a softmax fraction of the flux |
| unpacked cube: column `c` is `l = (c − N/2) dx` East, row `r` is `m = (r − N/2) dx` North | row 0 North, column 0 East, centre ((N−1)/2, (N−1)/2) | `virgil = unpacked[::-1, ::-1]` with `dra = ddec = −dx/2` (half-pixel phase-centre convention) |
| per-ddid `10^log10_sigma` (`s_M`), applied as `w → w / s_M²` | `cal.sigma` | `s = g · s_M`: `s_M` of an observation's ddids is multiplied by that observation's gain |
| per-observation `10^log10_amp` (`g`, observation 0 fixed at 1), model × g and `w → w / g²` | `cal.gain` (observation 0 fixed) | the weight rescaling by g² is why `s` carries `g`: virgil's σ is in the data's units, MPoL's `s_M` in the already-rescaled ones |
| `delta_ra`, `delta_dec` (arcsec), phase `exp(−2πi(δRA·u + δDec·v))` | `cal.offset` = 1000 (δRA, δDec) mas | same sign; checked against the exact DFT |
| `FWHM` (arcsec) Gaussian taper, `GaussConvFourier` | `taper_fwhm_mas` = 1000 × `FWHM` | the same Gaussian visibility multiplier |
| no spectral index | `alpha = 0` | |
| `neg_log_likelihood_avg` = −ln L / (2N) | `½ Σ r² + log_norm + Σ_k (ln 2π − ln w_k)` | −ln L = N ln 2π − Σ ln w + χ²/2 on both sides |

With the map the two likelihoods agree to 1e-15 relative on the same model, the
same weights and the same data (`log_likelihood` and `neg_log_likelihood_avg · 2N`),
and the control `s = s_M` misses by 2e-2. The forward models agree to MPoL's NuFFT
error (`P12`, 6e-4 by default, 1.6e-6 with `sparse_matrices=True`), against an exact
numpy DFT (`crosscheck.alma.image_visibilities`); dropping the half pixel or either
flip misses by 1e-2 to 0.6. The conventions are CASA's, as in Part 1: the data's
`conj(V)` convention is applied by the reader, so here both codes see the same `(u, v, V)`.

MPoL's `log_likelihood` docstring writes Σ w_i where its code correctly uses
Σ ln w_i (`P11`). Its taper buffer is float32 (`P12`), a 3.5e-8 effect.

# Part 4: the MPoL reference run (formats)

The reference run trains the IM Lup example's own `sgd.py` (unmodified, paper flags, only
`--log-interval 200`) on the example's own export of the measurement set, then reads the result back with
three tasks of `src/external_bridge/mpol_worker.py` (`.venv-mpol`, MPoL's code from the pinned clone; no
virgil). `tests/test_alma_mpol_ref.py` pins them on small inputs. Each task is a JSON file run as

    .venv-mpol/bin/python src/external_bridge/mpol_worker.py task.json result.json

`task.json` holds `"task"`, `"examples_dir"` (the clone, `.external/mpol-examples`) and the task's own
fields; `result.json` is the output below plus `mpol_version`, `torch_version`, `torchkbnufft_version`,
`asdf_version`, `visread_version`, `numpy_version`. Defaults are the example's: `npix` 1028, `cell_size` 0.005
arcsec, `fwhm` 0.03 arcsec, `obsid_dict` as in `sgd.py`. Units: u, v in wavelengths (the example's
`loaddata.py` comment says kilolambda; its values are wavelengths); sky offsets in arcsec; images in
Jy/arcsec².

## `export_checkpoint`

Task fields: `checkpoint` (`sgd.py`'s `--save-checkpoint`), `out_npz`. The image is recomputed from the base
cube through the example's forward chain (the checkpoint holds `icube.packed_cube` only after a forward
pass, and in float32); `stored_packed_cube_max_rel_diff` says how far that is from the stored one.

| `out_npz` key | Shape | Meaning |
|---|---|---|
| `image` | (N, N) | the tapered image, Jy/arcsec². **Row 0 is North, column 0 is East** (`imshow` with East left, North up), i.e. `mpol.utils.packed_cube_to_sky_cube(packed)[0][::-1]`; the same layout as `virgil_bridge.mpol.unpack` |
| `base_image` | (N, N) | `softplus(base_cube)`, before the taper, same layout |
| `base_cube` | (N, N) | the raw parameter, MPoL's packed (FFT) layout, as `sgd.py` stores it |
| `l_east_arcsec`, `m_north_arcsec` | (N,) | the pixel centres of the columns and rows of `image` (offset East and North of the grid centre; both axes decrease with index). Column j is at l = (N/2 − 1 − j) dx: MPoL's grid puts a pixel on l = 0, so the half-pixel map (D11) is the comparer's to apply |
| `cell_size_arcsec` | () | dx |
| `log10_amp`, `amp` | (7,) | per observation, **observation 0 (fixed) included as 0 / 1**; `amp` = 10^`log10_amp` (the example's `g`) |
| `log10_sigma`, `sigma` | (26,) | per ddid (`s_M`) |
| `delta_ra_arcsec`, `delta_dec_arcsec` | (7,) | per observation, fixed one (0) included; phase `exp(−2πi(δRA u + δDec v))` |

`result.json`: `npix`, `cell_size_arcsec`, `fwhm_arcsec`, `flux_jy` (Σ image · dx²), `peak_jy_arcsec2`,
`min_jy_arcsec2`; the nuisance vectors as lists (`amp`, `log10_amp`, `sigma`, `log10_sigma`,
`delta_{ra,dec}_arcsec`, `delta_{ra,dec}_mas`); `sigma_eb_mean` (`{"0": ..., "6": ...}`: `analyze_scale_factors.py`'s
`np.average` of the sigma factors over each observation's ddids); `missing_keys`, `unexpected_keys` of the state
dict (`icube.packed_cube` is expected in `unexpected_keys` when the checkpoint has no forward pass).

## `objective_full`

Task fields: `checkpoint`, `asdf` (the example's export), optional `batch` (10⁶), `sparse_matrices` (true),
`lam_ent` (1e-9), `sb_only` (false), `device` (CUDA if present). float64 throughout; the data are
`loaddata.get_ddid_data`'s (float32 weights widened exactly). Per batch the example's `adjust_weights`,
`adjust_amplitudes`, `adjust_positions`, then `mpol.losses.log_likelihood`; the image is computed once.

| `result.json` field | Meaning |
|---|---|
| `n` | complex visibilities |
| `neg_log_likelihood` | −ln L, normalising constants included (N ln 2π − Σ ln w + χ²/2, MPoL's `log_likelihood`, adjusted weights) |
| `neg_log_likelihood_avg` | −ln L / 2N, the example's batch loss without the entropy term |
| `entropy` | S_M = `mpol.losses.entropy` of the tapered image (prior 1e-4, F_tot 0.253), a sum over pixels in Jy/arcsec² |
| `lam_ent`, `loss_mpol` | λ_M and `neg_log_likelihood_avg` + λ_M · S_M |
| `objective_R` | −ln L + 2N λ_M S_M, the quantity decision C4 equates with virgil's R objective |
| `n_ddid`, `chi2_fit`, `chi2_raw`, `*_per_dof`, `*_total` | per ddid (26 entries): the count and MPoL's χ² = Σ w \|model − data\|², over complex points, so ≈ 2 n_ddid for a good fit. `_fit`: weights `w / (s_M g)²` (the fitted s); `_raw`: the quoted weights (s = 1); the model (amplitude-scaled by g, shifted) is the same in both. `_per_dof` is χ² / (2 n_ddid) |
| `sum_ln_weight_fit` | per ddid Σ ln w with the fitted weights |
| `batch`, `sparse_matrices`, `device`, `dtype` | as run |

## `asdf_vs_npz`

Task fields: `asdf`, `npz_dir` (per-ddid `ddid_NN.npz`: `u, v` in wavelengths, `vis` or `re`/`im`, `weight`, and, if
present, `chan, time, ant1, ant2`). The asdf is read by the example's `loaddata.get_ddid_data` (its flag handling,
its conversion to wavelengths, its channel flattening); a second pass over the tree, in the same order, recovers
the channel, antennas and time of each kept point. Points are matched by sorting both sides on (channel rank, time, ant1,
ant2) when both have those keys (`matching: "keys"`; `key_mismatch` counts points whose keys still differ), else
by order (`matching: "order"`). The channel rank is that of the point's own frequency, ascending (the asdf's `frequencies` per channel, our npz's `freq`): visread 0.0.5
orders channels by decreasing frequency, our npz keeps the MS order, and no direction is assumed (a side without `freq` raises).
Per ddid `pass` requires `key_mismatch` 0, `vis_max_abs` < `tol_vis` (default 1e-5 Jy) and `u_max_rel`, `v_max_rel`
(per point, |du| / max(|u|, 1 λ)), `weight_max_rel` < `tol_rel` (default 1e-6); a missing npz or differing N fails. The task **reports**; a conjugation, a unit factor or a different order is a
finding, not corrected.

`result.json`: `ddids` (keyed `"0"`..`"25"`), each `n_asdf`, `n_npz`, `matching`, `key_mismatch`, and, when the
counts agree, `u_max_abs`, `v_max_abs` (wavelengths), `vis_max_abs` (Jy), `weight_max_abs`, `u_max_rel`, `v_max_rel` (per point,
against max(|·|, 1 λ); `u_max_rel` is as before except for the 1 λ floor), `weight_max_rel`, `pass` (bool), and the signatures of a convention difference: `vis_conj_max_abs` (|V_asdf − conj V_npz|:
zero when the two differ by a conjugation), `u_flip_max_abs`, `v_flip_max_abs` (|u_asdf + u_npz|, likewise v: zero when the sign differs; a conjugation pairs with (u, v) → (−u, −v)), `u_ratio_median` (≈ 1; 10³ would be kilolambda). Also `worst` (maxima over ddids, plus `n_mismatch_ddids`: ddids with no npz or differing counts, which the maxima skip, and `key_mismatch_max`) and
`n_asdf_total`, `n_npz_total`; `worst.all_pass` and `tolerances` (`vis_max_abs_jy`, `rel`). An older `result.json` has no `v_max_rel`, `pass` or `tolerances`.

## The environment

`bash scripts/setup_external.sh mpol` builds `.venv-mpol`: Python 3.12, torch (the CUDA wheel index when
`nvidia-smi` exists, else CPU; `MPOL_TORCH_INDEX=cu126` or `cpu` overrides), mpol 0.3.1, asdf, tensorboard,
matplotlib and visread 0.0.5 (installed with `--no-deps`: the example's `loaddata.py` and `sgd.py` use only
`visread.process`, `.scatter` and `.visualization`, none of which needs CASA), then clones the examples at
`5a7c3f0b`. The script prints the resolved versions (laptop, 2026-10-20: torch 2.14.1, torchkbnufft 1.5.2,
mpol 0.3.1, asdf 6.0.0, tensorboard 2.21.0, matplotlib 3.11.2, visread 0.0.5, numpy 2.5.4).
The export of the measurement set (`casa_analysis/src/export_asdf.py`) needs casatools 6.7.0.31 *together
with* visread: that is the **`alma` env** (`.venv-alma`, Python 3.11; on OzSTAR the conda env `alma-casa`),
not `mpol`. casatools wheels pin the Python version and weigh hundreds of MB of CASA that training never
uses, so the two stay apart; the asdf in between is the interface. `alma` also installs asdf for the export.

## Part 5. The MaxEnt fit (P3): MPoL's objective in virgil

`scripts/alma_imlup_maxent.py` fits IM Lup with the objective of the MPoL example
(config R) and with our full model (config F). MPoL's entropy acts on the
**tapered** image in Jy/arcsec², `S_M = (1/F_tot) Σ I ln(I/p)` with `p = 1e-4` and
`F_tot = 0.253` (`mpol.losses.entropy`, `tot_flux=1`, divided by `F_tot`); virgil's
`MaxEntropy` acts on the untapered unit-sum image, so R uses a fit likelihood term
(`virgil_bridge.alma.MPoLEntropy`, empty residuals, `log_norm = λ S_M`) and F keeps
`MaxEntropy` (ledger `D12`). The strengths: `λ_R = 2N λ_M` (the data term is −ln L,
MPoL's loss −ln L / 2N); `λ_F = λ_R · F / (F_tot dx²)` with `dx` in arcsec
(Σ I ln(I/p) = (F/dx²) [Σ b ln b + const] for a unit-sum `b`). R1 (tests/test_alma_maxent.py)
checks the R objective against MPoL's own loss: S_M to 1e-7, the data term to
1.1e-6 (MPoL's accurate NuFFT, `P12`), with controls that must fail.

## Part 6. MPoL on virgil's cross-validation folds

Held-out χ² is the metric neither code optimises (comparison metric 1 of `design/alma_imlup.md`), so MPoL's
example is trained on the training part of each of virgil's dartboard folds and scored on the held-out part. The
example runs unmodified (C14): the hold-out is made on the input side, by a per-fold training asdf. Two more tasks of
`src/external_bridge/mpol_worker.py` (same task/result JSON as Part 4), with the numpy-only helpers in
`external_bridge/cv_folds.py`; `tests/test_alma_mpol_cv.py` pins them.

**The folds** are `alma_imlup_maxent.py folds` (`dartboard_folds(k=5, n_q=12, n_phi=16, seed=0)` on all data): one `fold_NN` and
`annulus_NN` int8 array per ddid in the order of that ddid's points in `ddid_NN.npz`, and a JSON `meta` with the parameters and a
sha256 of the assignment. A cell is `(annulus, φ mod π)`, so a point and its Hermitian partner are always in one fold. The MS has no
explicit partners (each baseline is stored once), so in practice this keeps each point's fold a function of (|q|, φ mod π) alone.

**Matching.** The asdf's kept points (`uu[~flag]`, channel-major) are paired with the npz's by (rank of the channel's frequency, time,
ant1, ant2), never by index (visread numbers channels in decreasing frequency, our npz keeps the MS order; Part 4). The pairing is checked: all
keys equal; u, v (from the asdf's metres and the channel frequency) and the weight within 1e-6 relative of the npz's. A mismatch raises
rather than writing a wrong mask.

### `export_fold_asdf`

Task fields: `asdf`, `npz_dir`, `folds`, `fold` (J), `out_asdf`, optional `tol_rel` (1e-6). Writes `out_asdf`: the same tree with the
**held-out points flagged** (`flag[chan, row] = True` for the points of fold J). `loaddata` keeps `~flag`, so `sgd.py` trains on exactly the
points of the other folds, in the same order, with the same visibilities and weights; nothing else changes. Result: `n_total`, `n_held`,
`n_train` (also per ddid), `folds_sha256`, `folds_params`, `worst_match` (`u_max_rel`, `v_max_rel`, `weight_max_rel`).
MPoL's `lam_ent` stays 1e-9: the data term is a mean over the training points, so λ_R = 2 n_train λ_M, the balance of
`cv-task`'s `lam_train = λ n_train / N`.

### `score_fold`

Task fields: `checkpoint` (the trained model for fold J), `asdf` (the **full** export), `npz_dir`, `folds`, `fold`, optional `batch` (10⁶),
`sparse_matrices` (true), `device`, `tol_rel`. float64; the example's `Net` with MPoL's accurate NuFFT; per batch `adjust_weights`,
`adjust_amplitudes` (V → g V), `adjust_positions`. The convention is that of `virgil_bridge.alma.score` (so of `cv-task`): with `s = g s_M`
per ddid the per-point −ln L is `w|V − M|²/(2s²) + 2 ln s + ln 2π − ln w`; `chi2_raw` sums `w|V − M|²` over the points (≈ 2 per point
for a good fit); the "s = 1" variants put `s = 1` with M unchanged.

| `result.json` field | Meaning |
|---|---|
| `chi2_raw_per_dof` | `chi2_s1 / 2n`: raw χ²/N, reported first |
| `heldout.nll`, `.chi2_s1`, `.chi2_fit` | totals: −ln L at the fitted s, raw χ², χ² at the fitted s |
| `heldout.by_ddid` (26), `.by_annulus` (12) | tallies `n`, `nll`, `chi2_raw`, `chi2_fit` at the fitted s |
| `heldout.by_ddid_s1`, `.by_annulus_s1` | the same with s = 1 (`nll` at s = 1) |
| `nll_mpol` | the sum of `mpol.losses.log_likelihood` on the adjusted weights over the same points (equals `heldout.nll`) |
| `folds_sha256`, `n_test`, `worst_match`, `sparse_matrices`, `device` | provenance |

virgil's `cv.json` carries the same `heldout` layout. The **profiled** score puts one global error scale per set at its maximum-likelihood
value, `s² = χ²_raw/(2n)`: `nll_s1 − χ²_raw/2 + n(1 + ln(χ²_raw/2n))`, computed by `cv_folds.profiled` from the s = 1 tallies of either code, in total and per
annulus (it needs no field virgil does not already write).

**Agreement of the two scorers** (`tests/test_alma_mpol_cv.py`, an IM Lup-like image at 128², random nuisances, 26 ddids, 4 folds): the
held-out counts per ddid and annulus are identical; χ² per ddid agrees to 3.5e-5 relative and per annulus to 2.5e-5, the sum over the folds
to the all-data score to 9.5e-6, set by MPoL's interpolated NuFFT against virgil's at 1e-12. The interpolation error grows with the image's
power at high image frequencies: a white-noise image differs by 1e-3 of the flux (found by the test's first, noise-image checkpoint), so
the test uses a smooth image as R1 does. The held-out numbers of a real fit are therefore good to about the 1e-5 relative level of a smooth solution, far below the differences between methods; on a
solution with fine pixel-scale structure the NuFFT's own error would be the floor.

`alma_imlup_maxent.py cv-compare --mpol DIR --virgil DIR` (`external_bridge/cv_compare.py`) lines the two up: raw χ²/N first, then −ln L at the
fitted s and profiled, in total, per fold and per annulus, for MPoL, virgil at λ* and virgil at the grid λ nearest `λ₀ = λ_F-equivalent(2 N λ_M)` (C11).

