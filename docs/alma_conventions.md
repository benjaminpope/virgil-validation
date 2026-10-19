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

Reading the ŝ column: the short-baseline EBs 0–4 give ŝ = 1.05–1.15; the
long-baseline EB 5 gives 1.66–1.75 and EB 6 gives 1.99–2.08. ŝ overestimates
the weight scale wherever the sky or the atmospheric phase changes within 6 s,
and the long baselines see the phase change fastest, so these are upper
bounds on the true scale, not estimates of it; the bias is bounded only by
the short-baseline EBs, where ŝ² − 1 ≲ 0.3. The per-ddid σ factors of the
multi-EB fit are compared with these ratios, and are not replaced by them.
