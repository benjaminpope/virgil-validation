## Outside the trust graph

- **Bandwidth smearing.** virgil models it (`virgil.models.smeared_model`,
  documented in its `spectro_observables.md`), but nothing here checks that
  model independently: not independently validated. PMOIRED and CANDID model
  smearing too, and fouriever can; none has been compared with virgil's here.
- **harmonix's maps.** Only virgil's wrapper is checked here; the maps are
  harmonix's to validate.
- **AMIGO DISCO mode bases.**
- **Polar-disk GP convenience helpers.** `virgil.fields.polar_interpolate` is validated end to end through `MultipoleField.evaluate` rather than as its own graph node; `virgil.models.RadialGPFunction` and the NumPyro prior-dictionary builders `virgil.models.gp_priors` and `virgil.fields.multipole_field_priors` are not registered separately.
- **Coarse-to-fine uv cut (innovation register B4).** Each level of a multi-resolution radio fit cuts the uv data to its
  grid's Nyquist radius. Experimental in virgil's `design/radio_imaging_innovations.md` (no precedent found, not yet
  built), so there is nothing to validate yet: not yet validated. The CLEAN start and the staging of nuisance and image
  fits (B4's other parts) are published practice and have no independent check here either.
- **Non-scientific, CLI, plotting, and legacy surfaces.** Plotting helpers (`virgil.plotting.*`, matplotlib styling `STYLE`),
  command-line entry points (`virgil.pipeline.cli`), the deprecated `oifits_implaneia` compatibility shim
  (`virgil.legacy.oifits_implaneia.*`), static scalar/tuple constants and schema/tolerance registries (`GAIN_GROUPS`,
  `OFFSET_GROUPS`, `KINDS`, `KKT_TOL`, `BAND_TOL`, `KKT_W_TOL`, `NUISANCE_TOL`, `EPS_PRIOR`, `DELTA`, `FLAG_RANGE`,
  `NIRISS_AMI_HOLES`, `VLTI_UTS`, `SCHEMA`, `CHECKS`), and exception classes perform no independent scientific calculation
  and are intentionally excluded from `trust/graph.yml`.

To ask for any of these, or anything else, [open an
issue](https://github.com/benjaminpope/virgil-validation/issues) naming the
model or function, the independent result it should match, and the precision
you expect.
