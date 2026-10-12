# Mutation testing of the tier A suite

virgil pinned at `0c1be69c61`. 7 of 116 listed mutants run: **4 killed, 0 survived**, 3 timed out, 0 errored. Kill rate (killed / (killed + survived + timed out)) **67%**.

Design `design/mutation.yml` sha256 `2d18963158a8`, mutant list sha256 `ceb194a1a9c7`.

Baseline (no mutation): 4 failed (tier A, finding/upstream deselected; run under load average ~40); 4 tests already failing there, deselected from every mutant run.

## By class of slip

| class | mutants | killed | survived | kill rate |
| --- | ---: | ---: | ---: | ---: |
| bessel_factor | 1 | 1 | 0 | 100% |
| flag_ignored | 1 | 1 | 0 | 100% |
| ft_conj_sign | 1 | 1 | 0 | 100% |
| mjd_time_grouping | 1 | 0 | 0 | 0% |
| pa_origin | 2 | 1 | 0 | 50% |
| whitening_transpose | 1 | 0 | 0 | n/a |

## By trust-graph node

A mutant counts under each node it targets.

| node | mutants | killed | survived | kill rate |
| --- | ---: | ---: | ---: | ---: |
| `virgil.orbits.KeplerOrbit` | 1 | 0 | 0 | 0% |
| `virgil.models.BinaryModelCartesian` | 2 | 1 | 0 | 50% |
| `virgil.oifits.read_oifits` | 2 | 1 | 0 | 50% |
| `virgil.angles.AngleVector` | 1 | 1 | 0 | 100% |
| `virgil.models.PointSource` | 1 | 1 | 0 | 100% |
| `virgil.models.System` | 1 | 1 | 0 | 100% |
| `virgil.models.UniformDisk` | 1 | 1 | 0 | 100% |
| `virgil.models.cvis_binary` | 1 | 1 | 0 | 100% |
| `virgil.models.cvis_uniform_disk` | 1 | 1 | 0 | 100% |
| `virgil.oidata.OIData.cp_noise` | 1 | 0 | 0 | n/a |

## Regression mutants (design amendment 1: every one must be killed)

17 of 17 killed; survivors: none.

| mutant | finding | status |
| --- | --- | --- |
| M89 | F13 | killed |
| M101 | F1 | killed |
| M102 | F2 | killed |
| M103 | F5 | killed |
| M104 | F6 | killed |
| M105 | F7 | killed |
| M106 | F8 | killed |
| M107 | F9 | killed |
| M108 | F10 | killed |
| M109 | F10 | killed |
| M110 | F11 | killed |
| M111 | F12 | killed |
| M112 | F13 | killed |
| M113 | F14 | killed |
| M114 | F15 | killed |
| M115 | F16 | killed |
| M116 | F18 | killed |

## Survivors (0): candidate gaps

Each is a slip no tier A test notices (or an equivalent mutant: check before filing a gap).


## Equivalent mutants (1), in neither count

- **M64** (timeout): Closure-phase whitening solves L^T w = a instead of L w = a. Reason: ClosureNoise.chol is the Cholesky factor of Q R Q^T with Q the left singular vectors of T, which is diagonal (off-diagonal below 3e-16 for 4, 5 and 6 telescopes), so L and L^T are the same matrix and solving with either gives the same whitened vector.

## Timed out or errored

- **M11** timeout: Position angle measured as N of E (arctan2(ddec, dra)) instead of E of N.
- **M64** timeout: Closure-phase whitening solves L^T w = a instead of L w = a.
- **M89** timeout: Exposures grouped on MJD alone (TIME ignored).

## Quick check of the unresolved ones

Only the tests that validate the mutant's nodes and their dependents (`--max-rank 1`). `survived` here means the tests written for these nodes do not notice the slip; the rest of the suite is left unmeasured (timed out).

- **M101** regression: killed in 12s 1 failed, 5 passed, 4 skipped, 778 deselected in 10.35s tests/test_visibilities.py::test_modulated_rim_in_plane_azimuth_and_blur[amps1-pas1-50.0]
- **M102** regression: killed in 11s 1 failed, 1 passed, 4 skipped, 782 deselected in 9.12s tests/test_visibilities.py::test_gaussian_arc_matches_untruncated_arc[5.0-4.0]
- **M103** regression: killed in 20s 1 failed, 4 passed, 7 skipped, 782 deselected, 2 warnings in 18.12s tests/test_vlti.py::test_laplace_cov_with_array_parameters
- **M104** regression: killed in 59s 1 failed, 15 passed, 12 skipped, 758 deselected, 8 warnings in 56.53s tests/test_vlti.py::test_fit_from_the_exact_optimum_converges[lm]
- **M105** regression: killed in 56s 1 failed, 13 passed, 12 skipped, 767 deselected, 8 warnings in 52.91s tests/test_vlti.py::test_fit_uses_lm_for_expanded_uniform_priors[expand]
- **M106** regression: killed in 13s 1 failed, 10 passed, 9 skipped, 572 deselected, 1 warning in 11.19s tests/test_vlti.py::test_all_closure_phases_flagged_leaves_the_visibilities
- **M107** regression: killed in 8s 1 failed, 1 passed, 4 skipped, 797 deselected, 2 warnings in 5.88s tests/test_remaining_models.py::test_tabulated_reference_flux_is_the_documented_mean
- **M108** regression: killed in 54s 1 failed, 54 passed, 29 skipped, 571 deselected, 14 warnings in 50.75s tests/test_epochs.py::test_epoch_positions_grid_and_gaps
- **M109** regression: killed in 55s 1 failed, 57 passed, 29 skipped, 571 deselected, 14 warnings in 52.76s tests/test_epochs.py::test_epoch_positions_grid_and_gaps
- **M11** pa_origin: killed in 13s 1 failed, 11 passed, 10 skipped, 755 deselected in 11.03s tests/test_orbits_more.py::test_kepler_position_angle_from_the_textbook_offsets[elements0]
- **M110** regression: killed in 21s 1 failed, 1 passed, 6 skipped, 614 deselected in 18.96s tests/test_closure_continuity.py::test_four_telescope_chi2_is_continuous
- **M111** regression: killed in 13s 1 failed, 1 passed, 4 skipped, 792 deselected in 10.52s tests/test_clean_finite.py::test_clean_is_finite_with_four_telescopes
- **M112** regression: killed in 11s 1 failed, 1 passed, 9 skipped, 571 deselected in 8.04s tests/test_frames_from_time.py::test_snapshots_in_time_column_are_separate_frames
- **M113** regression: killed in 7s 1 failed, 6 skipped, 804 deselected in 5.19s tests/test_orbits_more.py::test_f14_total_mass_is_keplers_third_law_without_orbitize[elements0]
- **M114** regression: killed in 7s 1 failed, 6 skipped, 803 deselected in 5.39s tests/test_orbits_more.py::test_f15_thiele_innes_node_at_180_is_in_the_documented_range_without_orbitize[0.0]
- **M115** regression: killed in 8s 1 failed, 6 skipped, 804 deselected in 5.81s tests/test_orbits_more.py::test_f16_state_vector_keeps_the_inclination_of_nearly_face_on_orbits_without_orbitize[0.001]
- **M116** regression: killed in 15s 1 failed, 4 passed, 4 skipped, 782 deselected in 11.52s tests/test_epochs.py::test_marginal_loglike_bound_binds[upper]
- **M64** whitening_transpose: survived in 138s 102 passed, 48 skipped, 536 deselected, 45 warnings in 134.07s (0:02:14) 
- **M89** mjd_time_grouping: killed in 7s 1 failed, 9 skipped, 581 deselected in 5.52s tests/test_frames_from_time.py::test_time_column_gives_the_same_frames_as_distinct_mjd

## Kills

| mutant | class | first failing test | s |
| --- | --- | --- | ---: |
| M01 | ft_conj_sign | `tests/test_cone_field.py::test_source_model_invariants[PointSource]` | 9 |
| M35 | bessel_factor | `tests/test_models_spectra_extras.py::test_cvis_uniform_disk_is_the_airy_pattern` | 15 |
| M79 | flag_ignored | `tests/test_vlti.py::test_all_closure_phases_flagged_leaves_the_visibilities` | 166 |
| M96 | pa_origin | `tests/test_priors_angles.py::test_angle_vector_samples` | 91 |
