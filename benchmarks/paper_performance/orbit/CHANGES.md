# CHANGES (orbit comparison)

Predeclared settings live in `cases.yml`. Entries here record anything changed after a measured run started, with the
reason; none yet. Two WP-O decisions made before any measured run, for the record:

- 2026-11-18: virgil `matched` uses `dense_mass=True` because Octofitter 9.0.0 samples with AdvancedHMC's dense Euclidean
  metric (`src/sampling.jl`); the plan asked for the same mass-matrix type.
- 2026-11-18: both codes use the IAU nominal GM_sun, au and day for Kepler's third law (virgil's convention); gate 1
  measures the residual difference rather than hiding it.
