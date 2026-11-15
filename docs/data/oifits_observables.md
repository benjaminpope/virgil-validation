# Spectra, observables and the OIFITS writer

This page checks the OIFITS writer, the extra observable likelihoods and the
continuum operator against read-back files and closed-form sums
(`tests/test_spectra_more.py`, `tests/test_observables_io.py`).

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `oifits.write_oifits` (V², closure phases, triple amplitudes, `OI_VIS` with `AMPTYP`/`PHITYP`, `OI_FLUX`) | read back with astropy and with our own reader | Standards | exact (float32 `EFF_WAVE`) |
| `VisibilityAmplitude`, `TripleAmplitude` (`extras=`) | the χ² each adds equals Σ((model − data)/σ)² with our own closed-form visibilities | Mathematics + Standards | 1e-6 |
| `observables.continuum_operator`, orders 0 and 1 | the least-squares fit of 1, 1/λ over the continuum channels, by the normal equations | Mathematics | 1e-10 |
| `oifits.build_hdulist` (`test_042_data_gains_angles.py`) | OIFITS2 `OI_REVN = 2` extensions, units (`T3PHI` in degrees, `EFF_WAVE`/`EFF_BAND` in metres), custom primary-header injection before writing, and independent read-back via `astropy.io.fits` and `crosscheck.chi2.load` | Standards | `< 1e-7` (float32 `EFF_WAVE`) |
| `oidata.cp_indices`, `oidata.closure_phases`, `observables.in_ranges` (`test_042_data_gains_angles.py`) | oriented triangle-leg indexing (rejecting reversed legs), branch-cut wrapping into `[−π, π)`, and multi-window wavelength selection | Mathematics | `< 1e-14` |
| `OIData` `vis_mode="logamp"` and `vis_mode="v2"` (`0.4.2`, `test_042_data_gains_angles.py`) | drops `V² <= 0` with warning, propagates `σ / (2 V²)`, evaluates `OIData.model` and `whitened_residuals` on both unprojected and `vis_mat`-projected `log\|V\|`, and converts amplitudes to `V²` with exact non-central variance `σ √(4 V² + 2 σ²)` | Mathematics | `< 1e-15` |

`FluxSpectrum` and `DifferentialPhase` are checked under
[Grey-scaled spectra and differential phases](../likelihood/spectra_phases.md),
now that virgil#217 states the grey scale's prior rather than taking it
from the data.

