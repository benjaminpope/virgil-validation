# Post-AGB rerun: the command line (Worker A, WP-02 to WP-08, reduced scope)

`python scripts/postagb_imaging.py STAGE OUTDIR --target T [options]`. Every option is off by default and its filename tag is absent at its default, so
earlier output names are unchanged. Tags are appended after the existing ones in the order below (`option_suffix`).

## Options

| Flag | Meaning | Filename tag |
| --- | --- | --- |
| `white OUTDIR --target T` (stage) | Fixed pilot (arm-(a) polar GP: σ 1, ℓ_r 2 beams, ℓ_θ 1 rad, m ≤ 3) on the `scale` stage's error scale and geometry. Measures the white scales (channel scatter about a line per (frame, baseline) and (frame, triangle)) and re-measures the night widths about the pilot. Writes `<T>_white.json`: `v2_white`, `t3_white`, `n_groups_*`, `skipped_*`, `night_widths_published`, `night_widths_pilot`, `night_widths_used` (pilot where it differs from the published width by more than 30 %), `pilot_cell`, `pilot_chi2_per_point` (on the arm-(b) noise), counts | file `<T>_white.json` |
| `--white V2,T3` or `--white FILE` | White scales of `--noise vlti|night`: two numbers, or the `white` stage's file (which also sets the night widths used) | `_white<v2>-<t3>` |
| `--inject organic` | Fit simulated data of the ORGANIC model (SPARCO + environment) on the real (u, v), errors and closure triangles | `_inj-organic_nz<white\|night>_s<seed>` |
| `--inject organic+point:E,N,F` | As above plus a grey point source at E, N mas from the primary, flux ratio F to the primary at 1.65 µm | `_inj-orgpt-E_N_F_nz…_s…` |
| `--inject npz:FILE` | The environment replaced by the image of an `image.npz` (`brightness`, `pixel_mas`) | `_inj-npz-<stem>_nz…_s…` |
| `--inject-noise white\|night` | The noise drawn. Must agree with `--noise` (`current`: white at the real target's MacKay factor, then MacKay is found again on the draw; `night`: the measured model, with `--white FILE` for the arm-(b) widths) | in the `_nz` tag |
| `--seed N` | Noise seed (also used by `calibrate`) | `_s<N>` (with `--inject`) |
| `--inject-fstar-sine AMP,PERIOD` | The truth's primary star fraction oscillates, AMP points peak, PERIOD days, one value per night (at the night's mean MJD) | `_fsine<amp>-<period>` |
| `--m-max N` | Azimuthal cap of the polar GP (negative: none). With m ≤ 3 the ℓ_θ grid is 0.25 to 2 rad, above it 0.125 to 2 rad. `--m-max beam` is not built | `_m<N>` or `_mall` |
| `--n-theta N` | Azimuthal nodes of the polar field (default 32). Needs `--m-max` < N / 2 | `_nt<N>` |
| `--polar-rmax support` | The polar grid ends at the support radius (`mask_mas`), not `mask_mas / cos i` | `_rsup` |
| `--report-corner` | After the pick, fits (σ 4, 1 beam), (σ 2, 1 pixel), (σ 4, 1 pixel) cold at the pick's ℓ_θ; writes `corner_dlnz` and `corner_cells` (ln Z minus the pick's, split into fit, prior and Occam parts, and NCC vs the truth under `--inject`). The pick never uses them | `_corner` |
| `--fstar free\|paper\|VALUE` | Primary star fraction at 1.65 µm: free (default), the ORGANIC header's SFLU1, or a number in (0, 1). Environment and background ratios become one `share` parameter; the secondary keeps its own prior (capped at 95 % of what f★ leaves). Writes `fstar_mode`, `fstar_target`, `fstar_fitted`, `fstar_rejected_draws` | `_fs<mode>` |
| `--fstar-night` | AI Sco and IW Car: one f★ offset per night about the common value, Normal(0, sd) truncated (AI Sco 3 points, ±6; IW Car 1.3, ±3). Writes the offsets, their Laplace sds and each night's V² χ² with and without | `_fn` |
| `--binary none` | The scene without a secondary | `_binnone` |
| `--binary free\|motion\|epoch` | As before, but **the secondary is grey in every mode**: index fixed at the paper's SIND2 (0), no prior and no option to free it | `_bin<mode>` |
| `--binary-v-sigma X` | `--binary motion`: velocity prior sd, mas/day (default `BINARY_V_SIGMA`) | `_vs<X>` |
| `--drop-nights a,b` | Drop these nights (0 = first night). A diagnostic: the JSON says `diagnostic: nights dropped` and `nights_dropped` | `_drop<a>-<b>` |
| `image --noise current` | Exits when the `scale` stage's file is missing (the errors would be silently unscaled); `--smoke` excepted | none |

## Outputs added

* Image JSON: `n_data`, `n_independent`, `n_residuals` (and `chi2_per_point` now divides by `n_independent`); `cell_records` (every grid cell: values, ln Z, χ²,
  `chi2_sum`, `log_norm`, `z2`, `logdet`, `n_eff`, steps, stop, history); `pick_flags` (`on_lr_bound`, `on_sigma_bound`, `on_ltheta_edge`); `n_theta`;
  `mp_r_max_used_mas`; with `--inject`: `inject`, `inject_noise`, `seed`, `truth_chi2`, `scale_drawn`, `scale_fitted` (MacKay found on the draw),
  `scale_truth_needs`. `image.npz` also holds `truth_brightness` and `truth_pixel_mas` under `--inject`.
* `compare` JSON: `ncc_beam` (the old `ncc_zero_shift`), `ncc_beam_total` (display blob: environment plus each star as a Gaussian of FWHM λ_min / (2 B_max)
  at its fitted flux, everything at the comparison beam; ORGANIC from its header), `ncc_native`, `flux_in_1mas`, `flux_in_2mas` (ours and reference),
  `inner_peak_ratio` and `inner_peak_ratio_beam` (brightest pixel inside 2.2 mas over the brightest in 2.5 to 3.5 mas), `flux_outside_reference_support`
  (beam-convolved flux where the reference is below 2 % of its peak), `common_field_half_mas` (NCC and fluxes use no area beyond the smaller field),
  `sec_physical`, `sec_implied_companion_mass_msun`, `sec_mass_ceiling_msun`; under `--inject` the reference is the truth image and the injection keys are copied.
* `sec_physical` (D16): false when the fixed source implies a companion above 2.5 M☉ on the target's period: Kepler with the projected separation as the
  semi-major axis, minus a 0.6 M☉ primary. Only AI Sco has a period (977 d, Bailer-Jones distance 2875 pc): the 1.75 mas source gives about 17.8 M☉, so
  `sec_physical` is false; other targets give `null`. Written for `--binary paper` runs only.

## Not built (cut by the scope)

`--m-max beam`, `--line-correction`, `--centre`, `--mean file:`, `--core`, `--sec-index`, the hand-drawn `common` noise of `simulate`, `--inject composite:FILE`.
`postagb_parametric.py` keeps its own χ² per residual row (`chi2_rows`) so the ladder's BIC does not move.
