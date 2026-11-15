# AMIGO mixed-DISCO products

This page checks that `virgil.amigo.load_oi_data` reads the AMIGO pipeline's
products for JWST NIRISS AMI correctly. Each filter's record holds DISCO
coefficients with independent errors, plus two operators that map a model's
log-amplitudes and phases onto them. No real product is needed: we build a
small one from the documented record fields, the model vector
`A_logamp @ log|V| + A_phase @ arg V`, and (u, v) stored with the opposite sign.

[`test_amigo_records.py`](https://github.com/benjaminpope/virgil-validation/blob/main/tests/test_amigo_records.py)
uses two filters, 40 uv points and 60 modes each, with random operators.
The coefficients come from a faint binary's closed-form visibility at the
physical (u, v), the negative of the stored values.

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `amigo.load_oi_data`, all filters and one by name | our coefficients and errors | Mathematics | exact |
| `OIData.model` of four binaries on the loaded data | our closed-form DISCO vector | Mathematics | 1e-12 (found 4e-16) |
| control | the stored (u, v) read as physical (a mirrored companion) | Mathematics | differs by > 10σ |
| `coverage.ami_grid_record` (`test_042_data_gains_angles.py`) | half-plane `pitch_m` square lattice where closed-form 7-hole pupil autocorrelation $> 0$, rotated by `rotation_deg` North-through-East; exact `A_logamp @ 1 == 0` flux/piston nullspace and `A_phase @ [u, v] == 0` centroid nullspace (`< 1e-14`), exact nuisance-projected Fisher precision at `keep = 1.0` (`< 3e-15`), linear `disco_sigma` scaling with `sigma`, and `keep` rank truncation (`0.99` vs `0.80`) | Mathematics | `< 3e-15` |
| `amigo.is_mixed_disco_record`, `amigo.validate_mixed_disco_record`, `amigo.mixed_disco_fields` (`test_042_data_gains_angles.py`) | schema/shape/positivity validation and `A_logamp @ log\|V\| + A_phase @ arg V` at negated `(-u, -v)` | Mathematics | `< 1e-14` |

How the DISCO modes are built (the AMIGO pipeline's own work) is outside
the trust graph.
