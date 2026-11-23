# Orbit comparison: virgil vs Octofitter.jl 9.0.0 (WP-O)

Plan: `reports/benchmarks-plan.md` section 5. Predeclared settings: `cases.yml`. Changes after a measured run: `CHANGES.md`.

## Files
`make_data.py` data and `manifest.json`; `run_virgil.py` and `run_octofitter.jl` one process = one (case, config) each, printing a
`BENCH_INNER` line; `gates.py` + `gate_octofitter.jl` gates 1-4 (`gates_result.json`); `compare.py` posterior comparison;
`smoke.py` the smoke matrix (rows in `smoke_rows/`); `docs_tutorial.jl` the docs rel-astrom tutorial; `setup_julia.sh` and
`julia/{Project,Manifest}.toml` the pinned Julia project; `test_orbit.py` light tests.

## Julia install (recorded 2026-11-18)
- juliaup 1.24.0 (brew), Julia 1.12.7; Octofitter 9.0.0 (registry); OctofitterInterferometry from the tag v9.0.0 commit
  5649836e82ba2530a6df03d0ee032c05b77f0ceb (not in the registry at 9.0.0). `julia --project=orbit/julia -e 'using Pkg; Pkg.instantiate()'`
  rebuilds from the committed Manifest. `julia/.CondaPkg/` (a Python the dependency tree creates, 100 MB) is git-ignored.
- Sizes: `~/.julia` 2.3 GB (compiled caches 445 MB), juliaup 800 MB. Free disk 13 GB before, 10 GB after.
- Time: resolve + download 4644 s, of which most was the clone of the Octofitter.jl git repository (slow link);
  precompile 550 + 57 + 153 s. Loading the three packages afterwards: 20 s.
- `calib.jl` runs (t_calib 0.010 s). The docs rel-astrom tutorial runs (`docs_tutorial.jl`, prints `TUTORIAL_OK`).

## Gates (`gates_result.json`)
1. Ephemeris vs `crosscheck.orbits.position`, 5 element sets x 20 epochs (i > 90 deg, e = 0.6, 0.8): virgil 2.3e-12 mas,
   Octofitter 7.8e-14 mas. Mapping: Octofitter omega, Omega, M0 are used directly in radians, no sign or offset. Corrections off.
2. Level A log-likelihood difference between two points vs `neutral.astrometry_loglike` (-2101.38381334): relative 2.3e-13
   (virgil), 3.0e-15 (Octofitter).
3. Level B vs the neutral independent-Gaussian V2 + closure-phase chi2 (-109519.9687): Octofitter 8.0e-16; virgil with
   independent closure-phase errors 2.4e-2, virgil's default likelihood 0.20. virgil's closure-phase likelihood is a von Mises
   residual, not a Gaussian; this is `likelihood_mismatch`, as the plan expects. The independent form is obtained with
   `eqx.tree_at(lambda d: d.cp_noise, d, None, is_leaf=lambda x: x is None)`, which is not in virgil's docs.
4. OIFITS reader (Octofitter) vs astropy on the 8 files: V2, errors, closure phases, u, v, triangle baselines agree to
   <1.5e-14; no triangle index 0, none out of range.

## Levels
- Level A matched and native: supported in both codes.
- Level B matched (NUTS) and native (virgil shipped workflow vs `octofit_pigeons`): both codes produce rows; none `unsupported`.
  Octofitter loads the OIFITS files directly (one file per epoch, `use_vis2 = true`), no workaround needed, so the 3 h time-box was not used.
- Octofitter's Gaussian closure-phase likelihood differs from virgil's von Mises one at large residuals (gate 3); Level B
  virgil Level B rows carry `likelihood_mismatch` (set by the matrix driver since db0a0c0; 107 flagged rows in run r2). The 2.4e-2 left with independent errors is attributed to von Mises vs wrapped Gaussian; no sine-chord neutral reference confirms it yet.

## Smoke timings (1 thread, 200 + 200 draws, 4 chains; Level B 30 + 30, 2 chains, native 1 chain; `smoke_rows/`)
| case, config | code | t_init s | t_cold s | peak MB |
|---|---|---|---|---|
| O-A-sim matched | virgil / Octofitter | 5.6 / 37.5 | 12.2 / 5.4 | 518 / 1770 |
| O-A-sim native | virgil / Octofitter | 5.3 / 31.0 | 16.8 / 6.8 | 550 / 1769 |
| O-B-sim matched | virgil / Octofitter | 12.8 / 94.0 | 8.6 / 14.9 | 676 / 1769 |
| O-B-sim native | virgil / Octofitter (pigeons, 4 rounds) | 15.7 / 96.4 | 8.3 / 49.8 | 690 / 1877 |

t_cold includes compilation in each code. Octofitter's t_init is its global optimisation plus Pathfinder (and compilation);
it is a fair part of the cost of "first draws" and is reported separately, as is virgil's start-point search.
Posterior check at 200 + 200 (`smoke_rows/compare_O-A-sim_matched_smoke.json`): median differences below 0.14 sd for all
quantities, sd ratios 0.99-1.10, KS <= 0.062, R-hat <= 1.014, no divergences. Not a headline.

## Notes and caveats
- Unconstrained coordinates differ between the codes (virgil NumPyro transforms vs Octofitter's own); the matched settings
  fix sampler, adaptation, tree depth and metric type, not the parametrisation. Phase is M0 in matched, theta in native.
- Kepler constant: both use the IAU nominal values; gate 1 shows no measurable difference.
- `octofit_pigeons` has no matched-draw equivalent: the native Level B Octofitter row uses `--pigeons-rounds` (default 8, untested at full length).
- Full-matrix estimate (both codes, all configs, 3 repeats): 90-120 min, dominated by Level B and the 20 pull seeds.
- Octofitter draws go to `chain<k>.csv` (32 columns incl. sampler internals); `compare.py` reads them.

## Chain starts (CHANGES.md 2026-11-23)
Both codes start from their own search of the prior volume, with no truth-centred or hand-set values; `t_init` times it.

virgil (`run_virgil.py`, "chain starts"), the same rule for every case, level and config:
1. Period window: the central [q, 1-q] quantile interval of the period prior, q = 1e-3 (`prior_interval`). O-A-real:
   10.655361-10.661541 d (+-3.09 sigma of the +-5 sigma prior); simulated cases: 150.1-249.9 d.
2. Trial periods (`trial_periods`): the frequency midpoints of virgil's `period_grid` cells over the window, so no trial
   is an end point; with fewer cells than chains the window is cut into one cell per chain. O-A-real (window 0.14 of a
   phase-coherence step): 4 trials; O-A-sim and pulls: 6; O-B-sim: 5.
3. Level A: `starting_orbits` over those periods, the `chains` best grid orbits. Level B: `epoch_positions` ->
   `starting_orbits` (200 best) -> `rank_orbits` on the visibilities -> `chain_starts`; flux = median of the per-epoch fits.
4. Every scalar start is clipped into the [q, 1-q] interval of its own prior (`start_values`), so it is strictly inside
   the support and within |logit(q)| = 6.9 of the centre in NUTS's coordinates. The angles start as unit vectors.
   `check_starts` raises, naming the site, if a start is on or outside a support or lacks a site.
5. No random numbers: the starts depend on the data and the priors only (`--seed` does not change them).
6. Recorded per row: `settings.init` (the rule and the realised numbers) and `settings.init_detail` (window, trial
   periods, start periods, start chi2 for Level A, chains clipped per site).

Octofitter (`run_octofitter.jl`): `initialize!(Xoshiro(100 seed), model)`, its documented route: prior draws -> global
optimisation -> Pathfinder, 1000 draws of the Pathfinder approximation kept as starting points in its unconstrained space
(so inside every prior's support); each chain starts at one of them, drawn with the chain's generator. Recorded in
`settings.init` and `settings.init_detail` (number of points, range and median of their periods).
O-A-real matched, seed 1 (local 2 + 2 smoke, identical in two runs): 1000 points, periods 10.656777-10.658860 d
(-1.67 to +0.41 sigma), median 10.658061 d; virgil's four starts span -2.32 to +2.32 sigma.

Differences that remain, by design ("each code's own route", `cases.yml`):
- Octofitter's starts are draws around an optimised mode; virgil's are the best points of a grid, not optimised
  (start chi2 of 120-240 for 24 numbers on O-A-real, 1500-2100 for 20 on O-A-sim). virgil's warm-up therefore begins
  further from the typical set.
- Octofitter sets the initial dense metric from the covariance of its starting points (`src/sampling.jl`); virgil's
  NUTS starts from the identity and adapts during warm-up.
- Octofitter's starts are random given the seed, and reproducible only on the same machine and package versions
  (its docstring); virgil's are deterministic.
- Both are counted in `t_init`, outside `t_cold` and `t_warm`; their effect on warm-up is inside `t_cold` and `t_warm`.

Starts per case (4 chains, both phase parameters; `test_orbit.py` checks every one, all 20 pull seeds included):
| case | start P (d) | prior support of P | -log density at the starts | clips |
|---|---|---|---|---|
| O-A-real | 10.659223, 10.657678, 10.660768, 10.656133 | (10.653451, 10.663451) | -23.9 to 36.6 | none |
| O-A-sim | 195.687 (all four) | (150, 250) | 726 to 1014 | none |
| O-A-sim-pulls, 20 seeds | 195.687 (all) | (150, 250) | 688 to 1158 | none |
| O-B-sim | 208.343 (three), 187.550 | (150, 250) | 2.3e4 to 1.2e5 | none |
