# CHANGES (orbit comparison)

Predeclared settings live in `cases.yml`. Entries here record anything changed after a measured run started, with the
reason; none yet. Two WP-O decisions made before any measured run, for the record:

- 2026-11-18: virgil `matched` uses `dense_mass=True` because Octofitter 9.0.0 samples with AdvancedHMC's dense Euclidean
  metric (`src/sampling.jl`); the plan asked for the same mass-matrix type.
- 2026-11-23 (PR 275 review, after run r2): O-A-real started every virgil chain on the two endpoints of the period prior (period_grid over the 0.01 d +-5 sigma window returns only its ends), which are infinite in the unconstrained space, so the rows were `error` (NumPyro "Cannot find valid initial parameters"). Starts now come from a +-1 sigma window (`p_start` in `case_info`; unchanged for the simulated cases) and are kept strictly inside the prior. The prior, data and the rest of the setup are unchanged. The two r2 rows are replaced by the rerun stage.
- 2026-11-18: both codes use the IAU nominal GM_sun, au and day for Kepler's third law (virgil's convention); gate 1
  measures the residual difference rather than hiding it.
- 2026-11-18 (WP-R, before the measured run): virgil Level B rows carry the `likelihood_mismatch` flag and `accuracy.likelihood_mismatch_rel` = 2.4e-2 (gate 3, independent closure-phase form; the default likelihood differs by 0.20).
