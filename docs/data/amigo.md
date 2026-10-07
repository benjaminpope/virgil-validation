# AMIGO mixed-DISCO products

`virgil.amigo.load_oi_data` reads the filter-keyed products of the AMIGO
pipeline for JWST NIRISS AMI. Each record holds DISCO coefficients with
independent errors, and two operators that map a model's log-amplitudes and
phases onto them. No real product is needed to check the reader. We build
a small one from what virgil documents: the record fields, the model
vector `A_logamp @ log|V| + A_phase @ arg V`, and (u, v) stored with the
opposite sign.

[`test_amigo_records.py`](https://github.com/benjaminpope/virgil-validation/blob/main/tests/test_amigo_records.py)
uses two filters, 40 uv points and 60 modes each, with random operators.
The coefficients are those of a faint binary's closed-form visibility,
taken at the physical (u, v), the negative of the stored values.

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `amigo.load_oi_data`, all filters and one by name | our coefficients and errors | Mathematics | exact |
| `OIData.model` of four binaries on the loaded data | our closed-form DISCO vector | Mathematics | 1e-12 (found 4e-16) |
| control | the stored (u, v) read as physical (a mirrored companion) | Mathematics | differs by > 10σ |

How the DISCO modes are built (the AMIGO pipeline's own work) is outside
the trust graph.
