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

`tests/test_inference_joint.py`: the inference helpers on problems with
closed forms, and the joint likelihood over several datasets against our
own observables on two simulated three-telescope files.

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `inference.hessian_matrix` | the analytic Hessian of a quartic-and-sine objective | Mathematics | 1e-12 |
| `inference.regularized_inverse` | NumPy's `inv(M + ridge I)`; a `RuntimeWarning` for an indefinite matrix | Mathematics | 1e-12 relative |
| `inference.laplace_covariance` | exactly (Jᵀ C⁻¹ J)⁻¹ for a model linear in its parameters; control: (Jᵀ J)⁻¹ and (Jᵀ C J)⁻¹ do not match | Mathematics | 1e-10 (ridge 0), 1e-6 (default ridge) |
| `inference.gaussian_fisher` | Jᵀ Σ⁻¹ J exactly on a linear model; on the joint prediction, J by central differences of our V² and closure phases; control: unweighted Jᵀ J does not match | Mathematics | 1e-12; 1e-6 |
| `inference.fisher_projection` | P Pᵀ = F⁻¹ and Pᵀ F P = I; flat directions floored at eps × the largest eigenvalue, with a warning | Mathematics | 1e-10 |
| `inference.laplace_parameter_uncertainty` | (½ ∂²χ²/∂θ²)^−½ of our χ² by central differences, the others fixed | Mathematics | 1e-5 relative |
| `likelihood.joint_loglike` | the sum over datasets of SciPy's Gaussian (V²) and von Mises (κ = 1/σ²) log densities, i.e. −½ our χ² plus their normalisations; shared and per-dataset parameters, with and without inflated errors; control: swapped per-dataset parameters do not match | Mathematics | 1e-8 |
| `likelihood.joint_prediction`, `joint_data`, `joint_errors` | our files' values and our observables, dataset by dataset, visibilities then phases; (prediction − data)/errors gives our χ² | Mathematics | 1e-10 |
| `likelihood.noise_sites`, `noise_for` | the documented site names (`noise.<term>`, `noise[i].<term>`); the terms they give per dataset reproduce our likelihood with that dataset's errors scaled | Mathematics | 1e-8 |
| `likelihood.posterior_predictive_summary` | the mean and (population) standard deviation of our V² and closure phases over the samples | Mathematics | 1e-10 |
| `detection.rescale_errors` | s = √(χ²/n) for each block of our χ², after which each block has χ²/n = 1 | Mathematics | 1e-10 |
| `detection.injection_grid` | every separation and flux in the documented order; angles of (dra, ddec) uniform on [0°, 360°) (Kolmogorov–Smirnov; this does not pin the PA sign convention) | Mathematics, statistics | 1e-12; p > 1e-3 |
| `detection.bootstrap_null` | sign flip: each whitened residual about our prediction keeps its magnitude (phases wrapped), also about another scene, with about half flipped; resample: residuals drawn only from their own block | Mathematics, statistics | 1e-8; within 4σ |

