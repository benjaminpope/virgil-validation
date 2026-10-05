# Spectra

Each spectrum is compared with the formula or interpolation its
documentation states, evaluated with NumPy and SciPy. Tests:
[`test_remaining_models.py`](https://github.com/benjaminpope/virgil-validation/blob/main/tests/test_remaining_models.py) and
[`test_spectra_more.py`](https://github.com/benjaminpope/virgil-validation/blob/main/tests/test_spectra_more.py).

## Continua

| virgil | Reference | Agreement |
| --- | --- | --- |
| `PowerLaw` | (λ/λ₀)^index | 1e-12 |
| `BlackBody` | the Planck ratio with SciPy's CODATA constants, and its Rayleigh–Jeans limit | 9e-9 (virgil carries hc/k to about ten digits) |
| `Tabulated` | linear interpolation, constant beyond the end nodes | 1e-12 |
| `Nodes`, linear | `numpy.interp`, values held or set to a fixed number outside the nodes | 1e-14 |
| `Nodes`, cubic | SciPy's natural cubic spline | 1e-13 |
| `Nodes.is_physical`, positive nodes with overshoot between them | a dense evaluation of SciPy's spline | the same verdict |

`Tabulated`'s reference flux has been the node mean, as documented, since the
fix for [F9](../index.md#F9).

## Lines and sums

| virgil | Reference | Agreement |
| --- | --- | --- |
| `GaussianLine`, `LorentzianLine` | the profiles, their half maxima, and the documented integrals by quadrature | 1e-13; integrals 1e-8 |
| `Sum` | the sum of its parts, and its reference flux at `wavel0` | 1e-14 |

## Chromatic scenes

| virgil | Reference | Agreement |
| --- | --- | --- |
| `System` with a black-body companion and a power-law envelope | the flux-weighted mean of closed-form visibilities, channel by channel | 5e-11 |
