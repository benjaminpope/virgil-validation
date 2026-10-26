# IM Lup: the MultipoleField GP image (stages 0-1)

Status: built 2026-10-27 (plan: Ben's IM Lup dots plan, 2026-10-27; gp_plan stages 0-1). Stages 2-4 (held-out χ², NUTS, the comparison notebook) are not built.

## Why

The MaxEnt fits (config R) minimise MPoL's objective, whose taper multiplies only the model, so lowering R pushes the untapered base image onto fewer pixels: chains of dots with dead
pixels between them. A GP image is smooth over its length scales, so it cannot form them. If it fits within Δχ²/N ≈ 1e-4 of R without dots, the data do not need them.

## Model (`virgil_bridge.alma_gp`, `scripts/alma_imlup_gp.py`)

- `Image(MultipoleField(latent, image_shape=(1028, 1028), r_max_mas=2400, ...), pixel_scale_mas=5)` on the gate grid (the half-pixel map of `alma_conventions.md`), flux free.
- Polar grid n_r = 160 (about 15 mas), n_theta = 256: 40 960 whitened latents with a standard-normal prior. That prior is the regulariser: **no taper and no entropy**.
- Geometry fixed at DSHARP's (Huang et al. 2018, ApJ 869, L42): i = 47.5°, PA = 144.5°, centre at the phase centre. Kernel Matérn ν = 3/2 in r and θ.
- σ (field amplitude) fitted, log-uniform on (0.05, 5) from `multipole_field_priors`; `length_r` and `length_theta` are fixed hyperparameters (the array job's grid).
- Nuisances as config R (per-observation gains and offsets, per-ddid error scales, observation 0 the reference), without the taper; χ² is virgil's Toeplitz term, with the offsets' box ±20 mas.
- `--mean-from-clean DIR`: the field's `mean` is the restored CLEAN image (`alma_imlup_clean.py` `image.fits`, Jy/beam → Jy/arcsec², regridded to the grid), floored at 1e-3 of its peak
  (`mean_floor`). The CLEAN beam also gives `image_clean_beam.npy`.

## Fit

Stage `held`: nuisances held at `--nuisances-from values.npz` (the Adam run `7ca7c0b-cd-ob20-tz-adam-lr1e-2/R`), latents, σ and flux fitted with `virgil.fit(method="lbfgs")` (restarted until C8:
loss change < 0.1 over 100 steps). Stage `all`: everything jointly. Per stage `result.json` records steps, wall time, s/step, s per value-and-gradient evaluation and the C8 verdict.

## Correctness (stage 0)

`tests/test_alma_gp.py` runs `Image(MultipoleField)` through `ComplexVisData.toeplitz` on toy data (64², 4 000 visibilities, two observations) and compares loss and every gradient leaf
with the direct χ² (agreement about 1e-13; the test asks 1e-8). Skipped on a virgil without `radio_toeplitz` or `MultipoleField`; the CI pin has both.

## Outputs

`values.npz`, `image.npy`/`image.fits` (Jy/arcsec²), `image_tapered.npy` (30 mas), `image_clean_beam.npy`, `resid_dirty_obs<e>.npy`, `result.json`: raw χ²/N first, then -ln L, flux,
flux beyond 1.8″, nuisances, hyperparameters, timing, `dots` (`dots_report`, when this checkout has it: vv#195).

## Provenance

GP imaging of interferometer data: Junklewitz et al. 2016 (RESOLVE, A&A 586, A76), Arras et al. 2019 (A&A 627, A134), Arras et al. 2022 (Nat. Astron. 6, 259); the kernels and JAX
implementation follow Edenhofer et al. 2024 (NIFTy.re, JOSS 9, 6593). Each checked on ADS on 2026-10-27. `MultipoleField` is provisional in virgil; this check counts towards promoting it.
