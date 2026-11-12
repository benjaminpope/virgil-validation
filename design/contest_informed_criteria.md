# Criteria for the informed contest fits: checks that say a fit failed

**Status: proposed, awaiting Ben.** Source: `design/contest_informed_plan.md` section 5. The thresholds below are
proposals set from the decision each one protects, not from a result. No run has been seen when this file was
written.

`scripts/contest_informed.py` records this file's last commit (`crosscheck.preregistration.criteria_commit`) in every
report and refuses to run when the file has uncommitted changes. The mandate stage `informed_fit` `requires` the six
checks, so a fit that fails one cannot be recorded as passing. A changed threshold means a rerun; a change made after
a result has been seen is a change-log row that says "yes".

The code is `src/crosscheck/informed_starts.py` (constants `BOUND_EDGE`, `AGREE_FRACTION`, `MIN_AGREE`,
`FAMILY_FLUX_LIMIT`, `RING_FRACTION`, `COMPANION_ELEMENTS`); the numbers here and there must agree, and
`tests/test_contest_informed_starts.py` pins them.

## Checks

A check is `pass`, `warn` (could not be run: nothing was measured) or `fail`. Mandate accepts `pass` and `warn`; a
`warn` is a missing measurement, never a result, and the report says which.

| Check | Fails when | Threshold (proposed) | Protects against |
|---|---|---|---|
| `no_bounds` | any fitted parameter of the chosen rung lies within `BOUND_EDGE` of the range of a prior bound, in log for a log prior; position angles (periodic) and unbounded priors are exempt | `BOUND_EDGE` = 1% | a fit that is the prior's edge: 2004 data1 (shell flux and modulation), 2008 AGN, 2024 Obj1 and Obj2 in the second run |
| `nested` | a rung has larger -2 ln L than a rung it contains, by more than `AGREE_FRACTION` x N | `AGREE_FRACTION` = 1% (of one chi2 unit per datum) | a richer rung that fits worse than its own sub-model: an optimizer that did not find the optimum (2006 in the second run) |
| `starts_agree` | fewer than `MIN_AGREE` starts finish within the same tolerance of the best -2 ln L | `MIN_AGREE` = 2, tolerance as `nested` | an optimum found once, by luck; a landscape that has not been explored |
| `family_complete` | the image fitted on top of the chosen model carries more than `FAMILY_FLUX_LIMIT` of the model's flux | 0.2 | a family that leaves out much of the object (2004 data1 in the second run: 2.0). also `fail` if the image step ran and errored; `warn` only if it was skipped (`--no-residual-image`) |
| `no_collapse` | an arc shorter than the resolution; a ring narrower than `RING_FRACTION` of the resolution (within 1%); a companion with its own position within `COMPANION_ELEMENTS` resolution elements of the star | 1 element; 0.2 element; 2 elements | a component that has shrunk to something else (the "companion" 0.6 mas from the star in 2024 Obj2) |
| `extent_known` | the extent `E` came from neither the contest's given field nor a constrained visibility-moment fit | a moment fit is unconstrained when V^2 is below 0.03 on the whole shortest quarter of baselines, or when the fitted size reached the top of its grid (2 x lambda / B_min) | size priors that cut off the object, or a field that was a guess (2006: V^2 below 0.03 everywhere) |

Tolerance of `nested` and `starts_agree`: `-2 ln L` is the log-likelihood with virgil's fitted noise terms
(`vis_error_rel`, `phi_error`), so it carries its normalisation; at fitted noise chi2 is about N, so 1% of N is 1% of
chi2, as the second run's rule ("within 1% of the best chi2") was.

## Not criteria

The scales, starts and priors of the plan (extent, reach, pixel; moment, stratified, map, rung-below and random
starts; flux priors; noise terms; tempering) are the method, not gates, and appear in each report with their sources.
The rung is chosen by BIC on the fitted-noise likelihood; chi2 on the quoted errors is reported beside it and never
chooses.

## Change log

| commit | what changed | changed after seeing a result: yes/no/unknown | note |
|---|---|---|---|
| (this file's first commit) | file created with the six checks and the thresholds of the plan, section 5 | no | Claude (Sonnet), for Ben, proposed; PR "informed fits: infrastructure" |
| (second commit) | `family_complete` fails when the image step errors (review of the PR); a chromatic rung's `starts_agree` is run on the chromatic stage's own starts; a run with no pinned criteria commit is refused, not warned | no | no full run had been made; Claude (Sonnet), PR review |
