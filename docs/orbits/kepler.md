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
| `orientation_from_varpi`, `orientation_priors` | ϖ = Ω + ω; Ω in [0°, 180°) from 2Ω; the documented keys (`Mathematics`); isotropic `inc` has E[cos² i] = 1/3 and uniform cos i (KS, `Mathematics + Statistics`) | Mathematics + Statistics | 1e-9 |
| `position_angle_log_jacobian` | log\|∂M/∂θ\| by finite differences of our projection; integral over a turn is 2π | Mathematics | 1e-6; 1e-8 |
| `starting_orbits` | each grid point's χ² equals our own weighted least squares in the unit orbit (X, Y); the best orbit is the one the positions came from | Mathematics | 1e-8 |
| `models.Attached` (a companion on an orbit) | the static binary with the companion at our orbit position, at three times | Mathematics | 1e-12 |

## The position angle prior

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `position_angle_prior` with `KeplerOrbit.from_position_angle` (e = 0, 0.3, 0.6) | its `loglike` is log\|∂M/∂θ\| by finite differences of our projection, and `log_norm` its negative | Mathematics | 2e-9 |

## Position data and calibration

`tests/test_orbits_more.py` and `tests/test_position_calibration.py`.

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `PositionData` (`likelihood`) | SciPy multivariate normal log-density (`from_sep_pa` via finite-difference and analytic polar Jacobians, and full `2 × 2` covariances); blind 5-epoch `PMOIRED` astrometry, correlated Schur-complement covariances, and profiled OIFITS `Δχ²` | Mathematics + Statistics + PMOIRED | < 1e-11 (cov), < 1e-6 (FD Jacobian), < 1e-10 (analytic Jacobian), < 5e-3 relative (`PMOIRED` profiled `Δχ²`) |
| `PositionData` (`calibration`: `θ_meas = θ_sky + north_angle`, `ρ_meas = plate_scale · ρ_sky`) | `crosscheck.orbits` positions and BFGS recovery of `(north_angle, plate_scale)` (`−north_angle` and `1/plate_scale` controls fail by > 50 σ; remains `convention` until checked against an external calibration-aware orbit code) | Mathematics | < 1e-10 (mas), < 1e-7 (deg) |

