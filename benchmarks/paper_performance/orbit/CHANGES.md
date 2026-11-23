# CHANGES (orbit comparison)

Predeclared settings live in `cases.yml`. Entries here record anything changed after a measured run started, with the
reason; none yet. Two WP-O decisions made before any measured run, for the record:

- 2026-11-18: virgil `matched` uses `dense_mass=True` because Octofitter 9.0.0 samples with AdvancedHMC's dense Euclidean
  metric (`src/sampling.jl`); the plan asked for the same mass-matrix type.
- 2026-11-23 (PR 275 review, after run r2): O-A-real started every virgil chain on the two endpoints of the period prior (period_grid over the 0.01 d +-5 sigma window returns only its ends), which are infinite in the unconstrained space, so the rows were `error` (NumPyro "Cannot find valid initial parameters"). Starts now come from a +-1 sigma window (`p_start` in `case_info`; unchanged for the simulated cases) and are kept strictly inside the prior. The prior, data and the rest of the setup are unchanged. The two r2 rows are replaced by the rerun stage. Superseded the same day by the next entry.
- 2026-11-23 (after run r2; replaces the +-1 sigma window above before any row was measured with it): one start rule for
  every orbit case, both levels and both configs, derived from the priors instead of per-case numbers
  (`run_virgil.py`, "chain starts"; README "Chain starts").
  - What changed, virgil: (1) the period search covers the central [q, 1-q] quantile interval of the period prior, q = 1e-3
    (virgil's own NUTS-start inset); (2) the trial periods are the frequency midpoints of the `period_grid` cells over that
    interval, never its end points, with at least one cell per chain; (3) every scalar start (P, M_tot, e, inc, flux) is
    clipped into the [q, 1-q] interval of its own prior, replacing the hand-set clips (M_tot x 1.001 / 0.999, e >= 1e-3,
    inc in [1, 179] deg, flux in [0.0101, 0.999], P 1e-6 of the window); (4) `check_starts` refuses a start on or outside a
    support, or with a site missing, and names the site; (5) the row records the rule and the realised trial periods,
    start periods and clips in `settings.init` and `settings.init_detail`. `case_info` no longer carries `p_range` or
    `p_start`. The virgil Level B rows recorded `init: numpyro default (init_to_uniform)`, which was wrong: they always
    used the `epoch_positions -> starting_orbits -> rank_orbits -> chain_starts` route; the string is now right.
  - What changed, Octofitter: `initialize!` gets an explicit generator, `Xoshiro(100 seed)` (it used the unseeded global one, so the
    r2 starting points cannot be reproduced); the starting points are checked finite; `settings.init` and
    `settings.init_detail` are recorded. The route itself (global optimisation + Pathfinder) is unchanged.
  - Why: the r2 O-A-real error came from starts built on the prior bounds, and the first fix moved one parameter of one
    case by a hand-set window. The cases are predeclared, so the start rule should not depend on the case.
  - Effect on the starts (4 chains, no sampling; chi2 of the start orbits on the astrometry): O-A-real 120.8 / 128.7 /
    212.4 / 236.2 at P = mean +0.77, -0.77, +2.32, -2.32 sigma (first fix: 128.1 / 138.3 / 913.4 / 1023.6; r2: both
    prior bounds, not finite). O-A-sim: best trial period 195.69 d instead of 204.55 d, chi2 1517-2092 instead of
    1594-2456. No start of any case needs a clip. Priors, data, sampler settings and likelihoods are unchanged.
  - Rows: every virgil orbit row of r2 was measured with the old starts (simulated cases: the `period_grid` edges), so
    the virgil orbit rows should be rerun together, not only O-A-real; `settings.init_detail` tells the two apart.
- 2026-11-18: both codes use the IAU nominal GM_sun, au and day for Kepler's third law (virgil's convention); gate 1
  measures the residual difference rather than hiding it.
- 2026-11-18 (WP-R, before the measured run): virgil Level B rows carry the `likelihood_mismatch` flag and `accuracy.likelihood_mismatch_rel` = 2.4e-2 (gate 3, independent closure-phase form; the default likelihood differs by 0.20).
