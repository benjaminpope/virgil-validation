# Mutation testing of the tier A suite

virgil pinned at `0c1be69c61`. 7 of 100 listed mutants run: **4 killed, 0 survived**, 3 timed out, 0 errored. Kill rate (killed / (killed + survived + timed out)) **57%**.

Design `design/mutation.yml` sha256 `b21e1e9d6335`, mutant list sha256 `f80d3513f6c1`.

Baseline (no mutation): 4 failed (tier A, finding/upstream deselected; run under load average ~40); 4 tests already failing there, deselected from every mutant run.

## By class of slip

| class | mutants | killed | survived | kill rate |
| --- | ---: | ---: | ---: | ---: |
| bessel_factor | 1 | 1 | 0 | 100% |
| flag_ignored | 1 | 1 | 0 | 100% |
| ft_conj_sign | 1 | 1 | 0 | 100% |
| mjd_time_grouping | 1 | 0 | 0 | 0% |
| pa_origin | 2 | 1 | 0 | 50% |
| whitening_transpose | 1 | 0 | 0 | 0% |

## By trust-graph node

A mutant counts under each node it targets.

| node | mutants | killed | survived | kill rate |
| --- | ---: | ---: | ---: | ---: |
| `virgil.oidata.OIData.cp_noise` | 1 | 0 | 0 | 0% |
| `virgil.orbits.KeplerOrbit` | 1 | 0 | 0 | 0% |
| `virgil.models.BinaryModelCartesian` | 2 | 1 | 0 | 50% |
| `virgil.oifits.read_oifits` | 2 | 1 | 0 | 50% |
| `virgil.angles.AngleVector` | 1 | 1 | 0 | 100% |
| `virgil.models.PointSource` | 1 | 1 | 0 | 100% |
| `virgil.models.System` | 1 | 1 | 0 | 100% |
| `virgil.models.UniformDisk` | 1 | 1 | 0 | 100% |
| `virgil.models.cvis_binary` | 1 | 1 | 0 | 100% |
| `virgil.models.cvis_uniform_disk` | 1 | 1 | 0 | 100% |

## Survivors (0): candidate gaps

Each is a slip no tier A test notices (or an equivalent mutant: check before filing a gap).


## Timed out or errored

- **M11** timeout: Position angle measured as N of E (arctan2(ddec, dra)) instead of E of N.
- **M64** timeout: Closure-phase whitening solves L^T w = a instead of L w = a.
- **M89** timeout: Exposures grouped on MJD alone (TIME ignored).

## Quick check of the unresolved ones

Only the tests that validate the mutant's nodes and their dependents (`--max-rank 1`). `survived` here means the tests written for these nodes do not notice the slip; the rest of the suite is left unmeasured (timed out).

- **M11** pa_origin: survived in 17s 3 passed, 9 skipped, 674 deselected in 13.40s 
- **M64** whitening_transpose: survived in 138s 102 passed, 48 skipped, 536 deselected, 45 warnings in 134.07s (0:02:14) 
- **M89** mjd_time_grouping: survived in 147s 150 passed, 45 skipped, 491 deselected, 52 warnings in 143.15s (0:02:23) 

## Kills

| mutant | class | first failing test | s |
| --- | --- | --- | ---: |
| M01 | ft_conj_sign | `tests/test_cone_field.py::test_source_model_invariants[PointSource]` | 9 |
| M35 | bessel_factor | `tests/test_models_spectra_extras.py::test_cvis_uniform_disk_is_the_airy_pattern` | 15 |
| M79 | flag_ignored | `tests/test_vlti.py::test_all_closure_phases_flagged_leaves_the_visibilities` | 166 |
| M96 | pa_origin | `tests/test_priors_angles.py::test_angle_vector_samples` | 91 |
