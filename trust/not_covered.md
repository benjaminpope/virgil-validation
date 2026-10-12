## Outside the trust graph

- **Bandwidth smearing.** virgil does not model it; PMOIRED, CANDID and
  fouriever do.
- **harmonix's maps.** Only virgil's wrapper is checked here; the maps are
  harmonix's to validate.
- **AMIGO DISCO mode bases.**
- **Multimodal posteriors.** The simulation-based calibration of `numpyro_model`
  (SBC v2) uses a 0.5 mas measurement of the companion's position in its low-signal
  regime, which removes the multimodality of the position posterior (without it,
  chains did not mix between the likelihood's lobes). SBC therefore says nothing
  about posteriors with several modes, and its results do not transfer to them.
  Whether the grid search and fit find the right mode is a separate check, the
  cold-start campaign, which runs first.

To ask for any of these, or anything else, [open an
issue](https://github.com/benjaminpope/virgil-validation/issues) naming the
model or function, the independent result it should match, and the precision
you expect.
