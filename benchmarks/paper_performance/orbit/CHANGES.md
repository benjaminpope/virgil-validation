# CHANGES (orbit comparison)

Predeclared settings live in `cases.yml`. Entries here record anything changed after a measured run started, with the
reason; none yet. Two WP-O decisions made before any measured run, for the record:

- 2026-11-18: virgil `matched` uses `dense_mass=True` because Octofitter 9.0.0 samples with AdvancedHMC's dense Euclidean
  metric (`src/sampling.jl`); the plan asked for the same mass-matrix type.
- 2026-11-18: both codes use the IAU nominal GM_sun, au and day for Kepler's third law (virgil's convention); gate 1
  measures the residual difference rather than hiding it.
- 2026-11-18 (WP-R, before the measured run): virgil Level B rows carry the `likelihood_mismatch` flag and `accuracy.likelihood_mismatch_rel` = 2.4e-2 (gate 3, independent closure-phase form; the default likelihood differs by 0.20).
