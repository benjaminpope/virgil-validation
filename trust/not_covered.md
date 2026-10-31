## Outside the trust graph

- **Bandwidth smearing.** virgil models it (`virgil.models.smeared_model`,
  documented in its `spectro_observables.md`), but nothing here checks that
  model independently: not independently validated. PMOIRED and CANDID model
  smearing too, and fouriever can; none has been compared with virgil's here.
- **harmonix's maps.** Only virgil's wrapper is checked here; the maps are
  harmonix's to validate.
- **AMIGO DISCO mode bases.**
- **Coarse-to-fine uv cut (innovation register B4).** Each level of a multi-resolution radio fit cuts the uv data to its
  grid's Nyquist radius. Experimental in virgil's `design/radio_imaging_innovations.md` (no precedent found, not yet
  built), so there is nothing to validate yet: not yet validated. The CLEAN start and the staging of nuisance and image
  fits (B4's other parts) are published practice and have no independent check here either.

To ask for any of these, or anything else, [open an
issue](https://github.com/benjaminpope/virgil-validation/issues) naming the
model or function, the independent result it should match, and the precision
you expect.
