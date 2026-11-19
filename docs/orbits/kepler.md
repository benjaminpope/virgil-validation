# Orbits

`tests/test_orbits_more.py` and `tests/test_standalone_nodes.py`, against our own textbook Kepler code
(`crosscheck.orbits`: Newton's method on Kepler's equation and the
visual-binary projection) and SciPy. orbitize! covers `KeplerOrbit`,
`ThieleInnesOrbit`, `StateVectorOrbit`, `RVData` and the posterior
(`tests/test_orbitize.py`).

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `KeplerOrbit.relative`, three orbits (e = 0 to 0.75, i near 90°, retrograde) | `crosscheck.orbits` | Mathematics | 1e-12 of a |
| `AxialVonMises` | exp(κ cos 2(θ − μ))/(360 I₀(κ)); θ and θ + 180° equal; normalised; KS on samples | Mathematics + Statistics | 1e-12 relative (density), 1e-12 absolute plus numpy default rtol 1e-7 (θ vs θ + 180°, log density; about 4.4e-6 at κ = 20), 1e-10 absolute (integral − 1); KS p > 1e-3 |
| `orientation_from_varpi`, `orientation_priors` | ϖ = Ω + ω; Ω in [0°, 180°) from 2Ω; the documented keys (`Mathematics`); isotropic `inc` has E[cos² i] = 1/3 and uniform cos i (KS, `Mathematics + Statistics`) | Mathematics + Statistics | 1e-9° (angle wrap, absolute); 1e-6 relative (Ω passthrough); E[cos² i] − 1/3 within 4σ (σ = √(4/45/N), N = 40000, about 6e-3); KS p > 1e-3 |
| `position_angle_log_jacobian` | log\|∂M/∂θ\| by finite differences of our projection; integral over a turn is 2π | Mathematics | 1e-6 absolute (plus numpy's default rtol 1e-7) on log\|∂M/∂θ\|; 1e-8 absolute on (integral − 2π) |
| `starting_orbits` | each grid point's χ² equals our own weighted least squares in the unit orbit (X, Y); the best orbit is the one the positions came from | Mathematics | 1e-8 relative (χ², denominator max(1, χ²)). The best orbit is checked at grid resolution: the grid period exactly, e within 1e-6 relative, t_peri within one phase step (P/36), best χ² < 6 × the number of positions |
| `period_grid` | uniform spacing in frequency 1/P from `p_min` to `p_max` with Δ(1/P) ≤ 1/(kT), bounding phase drift over baseline T to ≤ 1/k cycles | Mathematics | 1e-11 relative (uniform Δ(1/P)); 1e-12 absolute (endpoints; the `pytest.approx` default floor, since `rel=1e-14` is below it for 5 ≤ P ≤ 80); Δ(1/P)·T ≤ 1/k + 1e-14 cycles |
| `models.Attached` (a companion on an orbit) | the static binary with the companion at our orbit position, at three times | Mathematics | 1e-12 |

## The position angle prior

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `position_angle_prior` with `KeplerOrbit.from_position_angle` (e = 0, 0.3, 0.6) | its `loglike` is log\|∂M/∂θ\| by finite differences of our projection, and `log_norm` its negative | Mathematics | 1e-6 absolute (`loglike` vs finite differences); 1e-12 absolute (`log_norm` = −`loglike`) |

## Position data and calibration

`tests/test_orbits_more.py` and `tests/test_position_calibration.py`.

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `PositionData` (`likelihood`) | SciPy multivariate normal log-density (`from_sep_pa` via finite-difference and analytic polar Jacobians, and full `2 × 2` covariances); blind 5-epoch `PMOIRED` astrometry, correlated Schur-complement covariances, and profiled OIFITS `Δχ²` | Mathematics + Statistics + PMOIRED | < 1e-6 absolute (log-likelihood; finite-difference Jacobian and full `2 × 2` covariances, synthetic), < 1e-10 absolute (log-likelihood; `PMOIRED` covariances and analytic Jacobian), < 1e-12 absolute (dRA, dDec in mas), < 5e-3 relative (`PMOIRED` profiled `Δχ²`) |
| `PositionData` (`calibration`: `θ_meas = θ_sky + north_angle`, `ρ_meas = plate_scale · ρ_sky`) | `crosscheck.orbits` positions and BFGS recovery of `(north_angle, plate_scale)` (`−north_angle` and `1/plate_scale` controls fail by > 50 σ; internal parameter-sign definition per ledger **D25**, enforced via `properties: {calibration: 1}`) | Mathematics | < 1e-10 absolute (position, mas); < 1e-9° (forward-model position-angle offset); separation 1e-12 relative (plus numpy's default atol 1e-8 mas); recovery < 1e-7° (`north_angle`) and < 1e-8 (`plate_scale`), absolute; controls > 50 σ (log-likelihood) and > 1.9 · \|`north_angle`\| (position angle) |

