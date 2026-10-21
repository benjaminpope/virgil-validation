# The 2010 contest: calibration, not optimisation

The 2010 data (`Mystery-Low_HK.oifits`, 20 channels, 1.52-2.42 µm) have V²
errors of exactly 1e-4 on every point, and a chromatic source. Any grey image
therefore has raw χ²/N of 1e3-1e5 in both V² and closure phases. Rescaling
the field and the beam (virgil#361) did not help, and a larger field made the
2010 Low H GP fit worse. The winning entries reached χ²/N ≈ 1 only with
per-block error scales of 60-85. So this is a calibration and systematics
problem, and
[`scripts/contest_images.py`](https://github.com/benjaminpope/virgil-validation/blob/main/scripts/contest_images.py)
treats it as one.

## What it does

All of it is in `scripts/contest_images.py`; it reuses `--systematics block_scales`
(virgil's `error_scale(by_observable=True)`, MacKay's fixed point) and adds three things.

- **One image per channel**, as the winners did: tasks `2010_ch01` ... `2010_ch20`
  (the 20 channels of `Mystery-Low_HK.oifits`; channels 1-10 are H, 11-20 K), next to the
  grey `2010_lowH` and `2010_lowK`. A channel is selected by flagging the others
  (`select_channels`). With a virgil that has `ImagingPipeline` (virgil#457, not in the
  pinned `4b49cce`) the same selection is `ImagingPipeline(data, wavel_range=(lo, hi))`.
- **A calibration report** (`calibration_report`) from the GP fit that estimates the
  scales, saved in the summary and the npz (`calibration`): the scales `s_vis` and `s_phi`,
  and an additive-in-quadrature V² floor *f*, with Σ r²σ²/(σ²+f²) = N (*r* the whitened
  residual). The floor is reported, not applied: the fit uses the scales.
- **Raw and rescaled χ²/N, V² and closure phases separately**, in the same report.

```bash
python scripts/contest_images.py --task 26+k --prior gp --init clean --systematics block_scales
```

The grey default is the band task (`2010_lowH`, `2010_lowK`) run the same way: one scale
per observable across the band; compare it with the per-channel scales.

## What is reported, and what counts

| Quantity | Status |
| --- | --- |
| Raw χ²/N, V² and closure phases separately, on the quoted errors, for the first fit **and** the refit | The measure of the fit. Always reported first. |
| Scales and V² floor | Estimated from this data and this model: **informed by the data**, not blind to it, and only as good as the model (they absorb any structure it cannot fit). |
| Rescaled χ²/N | **Tautological**: the scale is fitted to make it about 1. Printed with that flag, never a success criterion. |
| Grey default | One scale per observable across all channels of a band; the per-channel scales are compared with it. |

!!! warning "Limits of the scale"
    With more image latents than data, MacKay's N − γ is small and `s` is
    inflated (the synthetic test shows it for closure phases: the raw
    χ²/N gives 10, the fixed point 70). Read `s` next to the raw χ²/N, not
    instead of it.

## Blind, informed and truth-model error

- **Blind:** the quoted errors, the channel selection, and the settings of
  `contest_images.py` (nothing from the published images).
- **Informed by the data:** the scales and the floor. They fix the weight
  between V² and closure phases and between channels, using the same data
  that are fitted.
- **Truth-model error:** a model that cannot represent the source (a grey
  image of a chromatic source, a point star where there is a resolved one)
  also shows up as excess χ² and is absorbed by the scale. A scale of 60-85
  does not distinguish that from a mis-stated error bar.

Tests:
[`test_contest_calibration.py`](https://github.com/benjaminpope/virgil-validation/blob/main/tests/test_contest_calibration.py)
(synthetic data of known truth, 16 × 16 pixels).
