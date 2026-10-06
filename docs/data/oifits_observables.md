# Spectra, observables and the OIFITS writer

`tests/test_spectra_more.py` and `tests/test_observables_io.py`.

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `oifits.write_oifits` (V², closure phases, triple amplitudes, `OI_VIS` with `AMPTYP`/`PHITYP`, `OI_FLUX`) | read back with astropy and with our own reader | Standards | exact (float32 `EFF_WAVE`) |
| `VisibilityAmplitude`, `TripleAmplitude` (`extras=`) | the χ² each adds equals Σ((model − data)/σ)² with our own closed-form visibilities | Mathematics + Standards | 1e-6 |
| `observables.continuum_operator`, orders 0 and 1 | the least-squares fit of 1, 1/λ over the continuum channels, by the normal equations | Mathematics | 1e-10 |

`FluxSpectrum` and `DifferentialPhase` are checked below, under
[Grey-scaled spectra and differential phases](../likelihood/spectra_phases.md),
now that virgil#217 states the grey scale's prior rather than taking it
from the data.

