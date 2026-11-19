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
  virgil Level B rows should carry `likelihood_mismatch` (the smoke rows do not yet; to be set by the full-run driver).

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
