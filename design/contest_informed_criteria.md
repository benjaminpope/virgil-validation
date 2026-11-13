# Criteria for the informed contest fits: checks that say a fit failed

**Status: proposed, awaiting Ben.** Source: `design/contest_informed_plan.md` section 5. The thresholds below are
proposals set from the decision each one protects. They are **not blind**: the second run's outcomes cited in the
"Protects against" column (2004 data1, 2006, 2008 AGN, 2024 Obj1 and Obj2) were known when they were chosen, and the
three smoke runs of the infrastructure PR (50 steps each, no converged fit) were seen before the second and third
commits. No full run with these checks has been seen.

`scripts/contest_informed.py` records this file's last commit (`crosscheck.preregistration.criteria_commit`) in every
report and refuses to run when the file has uncommitted changes. The mandate stage `informed_fit` `requires` the six
checks, so a fit that fails one cannot be recorded as passing. A changed threshold means a rerun; a change made after
a result has been seen is a change-log row that says "yes".

The code is `src/crosscheck/informed_starts.py`. Its constants are the table below; the table is parsed by
`tests/test_contest_informed_starts.py`, which fails if a value here and the value in the code differ.

| Constant | Value | Used by |
|---|---|---|
| `BOUND_EDGE` | 0.01 | `no_bounds` |
| `AGREE_FRACTION` | 0.01 | `nested`, `starts_agree` (with the ln N cap) |
| `MIN_AGREE` | 2 | `starts_agree` |
| `FAMILY_FLUX_LIMIT` | 0.2 | `family_complete` |
| `RING_FRACTION` | 0.2 | `no_collapse` |
| `WIDTH_FLOOR` | 0.05 | the lower bound of a ring's width prior, below `RING_FRACTION` so that a collapsed ring is inside the prior |
| `COMPANION_ELEMENTS` | 2.0 | `no_collapse` |
| `SHORT_V2_MIN` | 0.03 | `extent_known` |
| `EXTENDED_MIN` | 0.05 | `extent_known` |

## Checks

A check is `pass`, `warn` (could not be run: nothing was measured) or `fail`. Mandate accepts `pass` and `warn`; a
`warn` is a missing measurement, never a result, and the report says which.

| Check | Fails when | Threshold (proposed) | Protects against |
|---|---|---|---|
| `no_bounds` | any fitted parameter of the chosen rung lies within `BOUND_EDGE` of the range of a prior bound, in log for a log prior; position angles (periodic) and unbounded priors are exempt | `BOUND_EDGE` = 1% | a fit that is the prior's edge: 2004 data1 (shell flux and modulation), 2008 AGN, 2024 Obj1 and Obj2 in the second run |
| `nested` | a rung has larger -2 ln L than a rung it contains, by more than the tolerance: the smaller of `AGREE_FRACTION` x N and ln N | `AGREE_FRACTION` = 1% (of one chi2 unit per datum), capped at ln N | a richer rung that fits worse than its own sub-model: an optimizer that did not find the optimum (2006 in the second run) |
| `starts_agree` | fewer than `MIN_AGREE` starts finish within the same tolerance of the best -2 ln L | `MIN_AGREE` = 2, tolerance as `nested` | an optimum found once, by luck; a landscape that has not been explored |
| `family_complete` | the image fitted on top of the chosen model carries more than `FAMILY_FLUX_LIMIT` of the model's flux | 0.2 | a family that leaves out much of the object (2004 data1 in the second run: 2.0). also `fail` if the image step ran and errored; `warn` only if it was skipped (`--no-residual-image`) |
| `no_collapse` | an arc shorter than the resolution; a ring narrower than `RING_FRACTION` of the resolution (within 1%); a companion with its own position within `COMPANION_ELEMENTS` resolution elements of the star | 1 element; 0.2 element; 2 elements | a component that has shrunk to something else (the "companion" 0.6 mas from the star in 2024 Obj2) |
| `extent_known` | the extent `E` came from neither the contest's given field nor a constrained visibility-moment fit | a moment fit is unconstrained when V^2 - 3 sigma is below `SHORT_V2_MIN` on the whole shortest quarter of baselines (the noise-aware statistic: pure noise on a source resolved everywhere gives about zero), when the fitted size reached the top of its grid (2 x lambda / B_min), or when its extended light is under `EXTENDED_MIN` of the flux (the size is then not a measurement) | size priors that cut off the object, or a field that was a guess (2006: V^2 below 0.03 everywhere) |

Tolerance of `nested` and `starts_agree`: `-2 ln L` is the log-likelihood with virgil's fitted noise terms
(`vis_error_rel`, `phi_error`), so it carries its normalisation; at fitted noise chi2 is about N, so 1% of N is 1% of
chi2, as the second run's rule ("within 1% of the best chi2") was. But that is an absolute difference in -2 ln L, which
grows with N while the gap that picks a rung (the BIC cost of a parameter) is ln N; so the tolerance is
`min(AGREE_FRACTION x N, ln N)`. For N = 20 000 two optima 200 apart are therefore not "the same" (the tolerance is 9.9).

## Not criteria

The scales, starts and priors of the plan (extent, reach, pixel; moment, stratified, map, rung-below and random
starts; flux priors; noise terms; tempering) are the method, not gates, and appear in each report with their sources.
The rung is chosen by BIC on the fitted-noise likelihood; chi2 on the quoted errors is reported beside it and never
chooses.

## Change log

| commit | what changed | changed after seeing a result: yes/no/unknown | note |
|---|---|---|---|
| (this file's first commit) | file created with the six checks and the thresholds of the plan, section 5 | yes | the thresholds were chosen knowing the second run's outcomes named in the table (2004 data1 flux 2.0, Obj2 companion 0.6 mas, 2006 nested and V^2 below 0.03); no run with these checks; Claude (Sonnet), for Ben, proposed; PR "informed fits: infrastructure" |
| (second commit) | `family_complete` fails when the image step errors (review of the PR); a chromatic rung's `starts_agree` is run on the chromatic stage's own starts; a run with no pinned criteria commit is refused, not warned | yes | the smoke runs of the PR (50 steps, not converged) had been seen; no full run; Claude (Sonnet), PR review |
| (third commit) | `extent_known` uses V^2 - 3 sigma on the short baselines and needs `EXTENDED_MIN` of extended light; the tolerance of `nested` and `starts_agree` is `min(AGREE_FRACTION x N, ln N)`; a ring's width prior floor is `WIDTH_FLOOR` (below `RING_FRACTION`); the constants table is parsed by a test; this change log and the header say the criteria are not blind (PR review) | yes | smoke runs only; the changes make the checks stricter or well-defined, none was tuned to a result; Claude (Sonnet), PR review |
