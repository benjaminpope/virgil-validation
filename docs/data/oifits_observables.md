# Spectra, observables and the OIFITS writer

`tests/test_spectra_more.py` and `tests/test_observables_io.py`.

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `spectra.Nodes`, linear | `numpy.interp`; values held, or a fixed number, outside the nodes | Mathematics | 1e-14 |
| `spectra.Nodes`, cubic | SciPy `CubicSpline(bc_type="natural")` | Mathematics | 1e-13 |
| `Nodes.is_physical` with positive nodes and overshoot between them | a dense evaluation of SciPy's spline | Mathematics | same verdict |
| `GaussianLine`, `LorentzianLine` | profiles, half maxima, and the documented integrals by quadrature | Mathematics | 1e-13; integrals 1e-8 |
| `spectra.Sum` | the sum of its parts; reference flux at `wavel0` | Mathematics | 1e-14 |
| `oifits.write_oifits` (V², closure phases, triple amplitudes, `OI_VIS` with `AMPTYP`/`PHITYP`, `OI_FLUX`) | read back with astropy and with our own reader | Standards | exact (float32 `EFF_WAVE`) |
| `VisibilityAmplitude`, `TripleAmplitude` (`extras=`) | the χ² each adds equals Σ((model − data)/σ)² with our own closed-form visibilities | Mathematics + Standards | 1e-6 |
| `observables.continuum_operator`, orders 0 and 1 | the least-squares fit of 1, 1/λ over the continuum channels, by the normal equations | Mathematics | 1e-10 |

`FluxSpectrum` and `DifferentialPhase` are checked below, under
[Grey-scaled spectra and differential phases](../likelihood/spectra_phases.md),
now that virgil#217 states the grey scale's prior rather than taking it
from the data.

