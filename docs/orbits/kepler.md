# Orbits

`tests/test_orbits_more.py` and `tests/test_standalone_nodes.py`, against our own textbook Kepler code
(`crosscheck.orbits`: Newton's method on Kepler's equation and the
visual-binary projection) and SciPy. orbitize! covers `KeplerOrbit`,
`ThieleInnesOrbit`, `StateVectorOrbit`, `RVData` and the posterior
(`tests/test_orbitize.py`).

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `KeplerOrbit.relative`, three orbits (e = 0 to 0.75, i near 90°, retrograde) | `crosscheck.orbits` | Mathematics | 1e-12 of a |
| `AxialVonMises` | exp(κ cos 2(θ − μ))/(360 I₀(κ)); θ and θ + 180° equal; normalised; KS on samples | Mathematics + Statistics | 1e-12 |
| `orientation_from_varpi`, `orientation_priors` | ϖ = Ω + ω; Ω in [0°, 180°) from 2Ω; the documented keys; isotropic `inc` has E[cos² i] = 1/3 and uniform cos i (KS) | Mathematics + Statistics | 1e-9 |
| `position_angle_log_jacobian` | log\|∂M/∂θ\| by finite differences of our projection; integral over a turn is 2π | Mathematics | 1e-6; 1e-8 |
| `starting_orbits` | each grid point's χ² equals our own weighted least squares in the unit orbit (X, Y); the best orbit is the one the positions came from | Mathematics | 1e-8 |
| `period_grid` | uniform spacing in frequency 1/P from `p_min` to `p_max` with Δ(1/P) ≤ 1/(kT), bounding phase drift over baseline T to ≤ 1/k cycles; control: log-spaced periods violate the short-period bound | Mathematics | 3e-13 |
| `models.Attached` (a companion on an orbit) | the static binary with the companion at our orbit position, at three times | Mathematics | 1e-12 |

## The position angle prior

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `position_angle_prior` with `KeplerOrbit.from_position_angle` (e = 0, 0.3, 0.6) | its `loglike` is log\|∂M/∂θ\| by finite differences of our projection, and `log_norm` its negative | Mathematics | 2e-9 |

## Position data and calibration

`tests/test_orbits_more.py` and `tests/test_position_calibration.py`.

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `PositionData` (`likelihood`) | SciPy multivariate normal density (from `(sep, pa)` via finite-difference Jacobian and from full `2 × 2` covariances); blind 5-epoch astrometry and anisotropic uncertainties from `PMOIRED` (`VmodelOI` grid + `doFit`) and `CANDID` (`fitMap`) | Mathematics + Statistics + PMOIRED + CANDID | < 1e-6 (analytic), < 1e-8 (`loglike` value on external fits) |
| `PositionData` (`calibration`: `θ_meas = θ_sky + north_angle`, `ρ_meas = plate_scale · ρ_sky`) | `crosscheck.orbits` positions and BFGS recovery of `(north_angle, plate_scale)` (`−north_angle` and `1/plate_scale` controls fail by > 50 σ; internal parameter-sign definition per ledger **D25**, enforced via `properties: {calibration: 1}`) | Mathematics | < 1e-10 (mas), < 1e-7 (deg) |

