# Small helpers

`tests/test_helpers.py`: light checks of helpers that need little trust,
against our own χ².

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `grid_fit.best_grid_point` | NumPy's `nanargmax` | Mathematics | exact |
| `likelihood.build_model`, `likelihood.loglike` | a class and a template build the same scene; loglike differences are −½ our χ² differences | Mathematics | 1e-8 |
| `likelihood.inflated_errors`, relative to the model or the data, quadrature or maximum | the formulae in its docstring | Mathematics | 1e-14 |
| `inference.fisher` | the Hessian of ½ our χ² by central differences | Mathematics | 1e-5 |
| `simulate.simulate` | noiseless: our V² and closure phases; noisy: V² pulls are 𝒩(0, noise_scale²) (sd, KS) | Mathematics, statistics | 1e-12; within 4σ |
| `simulate.bias_test` | each entry is the fit to the simulation drawn with that key | Self-consistency | 1e-10 |

