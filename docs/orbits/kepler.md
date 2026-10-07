# Orbits

`tests/test_orbits_more.py`, against our own textbook Kepler code
(`crosscheck.orbits`: Newton's method on Kepler's equation and the
visual-binary projection) and SciPy. orbitize! covers `KeplerOrbit`,
`ThieleInnesOrbit`, `StateVectorOrbit`, `RVData` and the posterior
(`tests/test_orbitize.py`).

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `KeplerOrbit.relative`, three orbits (e = 0 to 0.75, i near 90°, retrograde) | `crosscheck.orbits` | Mathematics | 1e-12 of a |
| `AxialVonMises` | exp(κ cos 2(θ − μ))/(360 I₀(κ)); θ and θ + 180° equal; normalised; KS on samples | Mathematics + Statistics | 1e-12 |
| `orientation_from_varpi`, `orientation_priors` | ϖ = Ω + ω; Ω in [0°, 180°) from 2Ω; the documented keys | Mathematics | 1e-9 |
| `position_angle_log_jacobian` | log\|∂M/∂θ\| by finite differences of our projection; integral over a turn is 2π | Mathematics | 1e-6; 1e-8 |
| `starting_orbits` | each grid point's χ² equals our own weighted least squares in the unit orbit (X, Y); the best orbit is the one the positions came from | Mathematics | 1e-8 |
| `models.Attached` (a companion on an orbit) | the static binary with the companion at our orbit position, at three times | Mathematics | 1e-12 |

## The position angle prior

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `position_angle_prior` with `KeplerOrbit.from_position_angle` (e = 0, 0.3, 0.6) | its `loglike` is log\|∂M/∂θ\| by finite differences of our projection, and `log_norm` its negative | Mathematics | 2e-9 |

