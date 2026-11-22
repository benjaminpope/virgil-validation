# EHT M87* conventions (Lai et al. 2026 replication)

How the checks of the M87* closure-invariant results (Lai et al. 2026, Nature 659, 317) read the published products
and which code runs where. The data are the EHT HOPS network-calibrated Stokes I UVFITS of 2017-04-11, 2018-04-21
and 2021-04-18 and the authors' published reconstructions (Zenodo); see `DATA_ATTRIBUTION.md`.

## Images

- The published samples are 64 × 64 pixels over 225 μas (3.52 μas per pixel).
- Row 0 is North and column 0 is East; position angles are measured East of North.
- The image checked is the pixelwise median of the 50 published samples per day, as published (no re-alignment).

## Ring extraction (REX)

Three extractors run on the same image:

- `crosscheck.ring_extraction`: NumPy/SciPy, written from the paper's Methods equations 4–9 (EHT Paper IV REX),
  without reading the authors' code or virgil's. It gives the ring diameter 2r_pk, width w_r (FWHM), orientation
  η, asymmetry A and central fractional brightness f_c. For a ring brightness ∝ 1 + a cos(θ − η), A = a/2.
- `external_bridge.m87_worker`: the authors' own `RingFit` (from their published archive), in its own environment.
- `virgil.metrics.ring_features`, through `virgil_bridge.m87`.

Widths and f_c are reported but not compared between extractors whose definitions differ.

## Closure invariants and the geometric model

- `external_bridge.basis_worker` and `external_bridge.basis_bridge` run the authors' BASIS (xsringauss + two
  Gaussians, closure-invariant likelihood, bilby/dynesty) in its own environment (an Apptainer image on OzSTAR).
  BASIS computes closure invariants with its own code, in single precision.
- `virgil_bridge.m87_fit` builds the same invariants in JAX (N² − 3N + 1 per scan, first station as reference), the
  equivalent virgil model, and the diagonal and full-covariance likelihoods. The full covariance is singular
  (the invariants are redundant and normalised per scan), so residuals are taken in the tangent space of each scan's
  unit invariant vector.
- `external_bridge.m87_casa_worker` converts the UVFITS to a measurement set with casatools `importuvfits`, so
  virgil reads the same visibilities through `ComplexVisData.read_ms`.

None of the files under `src/crosscheck/` or `src/external_bridge/` import virgil.
