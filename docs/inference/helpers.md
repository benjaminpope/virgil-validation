# Small helpers

`tests/test_helpers.py`: light checks of helpers that need little trust,
against our own χ². `tests/test_bias_test.py`: `bias_test` against biases we
predict ourselves.

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `grid_fit.best_grid_point` | NumPy's `nanargmax` | Mathematics | exact |
| `likelihood.build_model`, `likelihood.loglike` | a class and a template build the same scene; loglike differences are −½ our χ² differences | Mathematics | 1e-8 |
| `likelihood.inflated_errors`, relative to the model or the data, quadrature or maximum | the formulae in its docstring | Mathematics | 1e-14 |
| `inference.fisher` | the Hessian of ½ our χ² by central differences | Mathematics | 1e-5 |
| `simulate.simulate` | noiseless: our V² and closure phases; noisy: V² pulls are 𝒩(0, noise_scale²) (sd, KS) | Mathematics, statistics | 1e-12; within 4σ |
| `simulate.bias_test` | each entry is the fit to the simulation drawn with that key | Self-consistency | 1e-10 |
| `simulate.bias_test`, uniform disk fitted to itself (40 draws) | mean = truth + Box's (1971) second-order bias, 0.3 % of σ; spread = our Fisher σ | Statistics, mathematics | mean within 4 standard errors (found 2.1); variance χ² test p > 1e-3 (found 0.41) |
| `simulate.bias_test`, Gaussian fitted to a uniform disk (40 draws) | mean = the pseudo-true σ that minimizes our noiseless χ² (White 1982), 0.787 mas | Mathematics | within 4 standard errors (found 1.4) |
| control | no nonlinear bias: σ = D/4, the zero-baseline match | Mathematics | excluded by > 20 standard errors (found 54) |

