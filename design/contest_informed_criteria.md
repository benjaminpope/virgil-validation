# Criteria for the informed contest fits: checks that say a fit failed

**Status: proposed by the design session.** Source: `design/contest_informed_plan.md` section 5, as changed by the recipe
move (see the change log). These are **process checks**: each is computed and reported with every fit so that a reader sees
how it was found and how well it fits. Whether a fit is **accepted** is an evaluator's judgement (Ben, 14 Nov: an overall good
fit that visually matches the structure of the truth image), not a numeric threshold. The constants below are the design
session's proposals, set from the decision each check protects. They are **not blind**: the second run's outcomes cited in the
"Protects against" column (2004 data1, 2006, 2008 AGN, 2024 Obj1 and Obj2) were known when they were chosen, and **the
start-agreement, same-optimum and fit-standard constants of this revision (6 of 8, 25, 2, 3 sigma, 2) were set after seeing the
results of the working recipe** that this revision moves into the repository. No full run with these checks has been seen.

`scripts/contest_informed.py` records this file's last commit (`crosscheck.preregistration.criteria_commit`) in every
report and refuses to run when the file has uncommitted changes. The mandate stage `informed_fit` `requires` the gating checks (`converged`, `no_bounds`, `nested`, `starts_agree`,
`family_complete`, `no_collapse`, `extent_known`), so a fit that fails one cannot be recorded as passing; `fit_standard` is reported
and is not a gate. A changed threshold means a rerun; a change made after a
result has been seen is a change-log row that says "yes".

The code is `src/crosscheck/informed_starts.py`. Its constants are the table below; the table is parsed by
`tests/test_contest_informed_starts.py`, which fails if a value here and the value in the code differ.

| Constant | Value | Used by |
|---|---|---|
| `BOUND_EDGE` | 0.01 | `no_bounds` |
| `START_AGREE` | 6 | `start_agreement`, first rung: of `START_OF` broad starts, at least this many end at the best optimum (the same fraction of a larger budget) |
| `START_OF` | 8 | `start_agreement`, first rung: the budget of broad starts; fewer than this cannot pass |
| `START_MARGIN` | 25 | `start_agreement`, higher rungs: the next distinct optimum is worse than the best by at least this in -2 ln L |
| `SAME_DELTA` | 2 | `same_optimum` (-2 ln L within this) and `nested` (a richer rung may not be worse than a rung it contains by more) |
| `SAME_SIGMA` | 3 | `same_optimum`: every parameter within this many combined sigma |
| `SIZE_TOLERANCE` | 0.05 | `same_optimum`: a size (diameter, FWHM, radius, width) within this fraction counts as the same whatever its sigma |
| `ANGLE_TOLERANCE` | 5 | `same_optimum`: an angle within this many degrees, modulo its period, counts as the same |
| `POSITION_ELEMENTS` | 0.25 | `same_optimum`: a position within this fraction of a resolution element counts as the same |
| `FIT_RMS_MAX` | 2 | `fit_standard` |
| `INFORMATIVE_SNR` | 3 | `informative_triangles`: a closure phase is informative when all three V^2 of its triangle exceed this many sigma |
| `INFORMATIVE_SHARE` | 0.05 | the information check: closure phases are flagged as carrying no information below this share of informative triangles |
| `FAMILY_FLUX_LIMIT` | 0.2 | `family_complete` |
| `RING_FRACTION` | 0.2 | `no_collapse` |
| `WIDTH_FLOOR` | 0.05 | the lower bound of a ring's width prior, below `RING_FRACTION` so that a collapsed ring is inside the prior |
| `COMPANION_ELEMENTS` | 2.0 | `no_collapse` |
| `SHORT_V2_MIN` | 0.03 | `extent_known` |
| `EXTENDED_MIN` | 0.05 | `extent_known` |

## Checks

A check is `pass`, `warn` (could not be run: nothing was measured) or `fail`. Mandate accepts `pass` and `warn` for the gating
checks; a `warn` is a missing measurement, never a result, and the report says which. `fit_standard` is reported, never gated,
and does not set the run's status.

| Check | Fails when | Threshold (proposed) | Protects against |
|---|---|---|---|
| `no_bounds` | any fitted parameter of the final fit lies within `BOUND_EDGE` of the range of a prior bound, in log for a log prior; position angles (periodic) and unbounded priors are exempt | `BOUND_EDGE` = 1% | a fit that is the prior's edge: 2004 data1 (shell flux and modulation), 2008 AGN, 2024 Obj1 and Obj2 in the second run |
| `nested` | a rung has larger -2 ln L than a rung it contains, by more than `SAME_DELTA` (-2 ln L on the tempered errors, gains marginalised) | `SAME_DELTA` = 2 | a richer rung that fits worse than its own sub-model: an optimizer that did not find the optimum (2006 in the second run) |
| `starts_agree` | first rung: fewer than `START_OF` broad stratified starts were run, or fewer than `START_AGREE` of `START_OF` (the same fraction of a larger budget) finish at the best optimum. Higher rungs: the best likelihood-map peak and its 3 jittered starts do not all finish at one optimum, or the next distinct optimum is less than `START_MARGIN` worse. The moment start is not counted | 6 of 8; margin 25 in -2 ln L | an optimum found once, by luck; a landscape that has not been explored |
| `fit_standard` (reported, not a gate) | after the last fit, the normalised residual rms at the tempered errors (the fit is judged with no error terms: terms fitted to the residuals would absorb the misfit; the rms at the quoted errors and at the fitted error terms is reported beside it) exceeds `FIT_RMS_MAX` on V^2, or on the informative closure phases when there are any. The verdict is then "incomplete", not "converged" | `FIT_RMS_MAX` = 2 | a fit that converged but leaves structure the data show; a model that is wrong in a way the error terms absorb |
| `family_complete` | the image fitted on top of the chosen model carries more than `FAMILY_FLUX_LIMIT` of the model's flux | 0.2 | a family that leaves out much of the object (2004 data1 in the second run: 2.0). also `fail` if the image step ran and errored; `warn` only if it was skipped (`--no-residual-image`) |
| `no_collapse` | an arc shorter than the resolution; a ring narrower than `RING_FRACTION` of the resolution (within 1%); a companion with its own position within `COMPANION_ELEMENTS` resolution elements of the star | 1 element; 0.2 element; 2 elements | a component that has shrunk to something else (the "companion" 0.6 mas from the star in 2024 Obj2) |
| `extent_known` | the extent `E` came from neither the contest's given field nor a constrained visibility-moment fit | a moment fit is unconstrained when V^2 - 3 sigma is below `SHORT_V2_MIN` on the whole shortest quarter of baselines (the noise-aware statistic: pure noise on a source resolved everywhere gives about zero), when the fitted size reached the top of its grid (2 x lambda / B_min), or when its extended light is under `EXTENDED_MIN` of the flux (the size is then not a measurement) | size priors that cut off the object, or a field that was a guess (2006: V^2 below 0.03 everywhere) |

**Same optimum** (used by `starts_agree`): two fits are at the same optimum when their -2 ln L differ by at most `SAME_DELTA`
and every parameter agrees within `SAME_SIGMA` combined sigma (the Gauss-Newton sigma of each fit), or within the practical
tolerance of its kind (`SIZE_TOLERANCE`, `ANGLE_TOLERANCE` modulo the angle's period, `POSITION_ELEMENTS`) where the sigma is
smaller than that. A parameter the data do not constrain (sigma above half its prior range, or above its value) is listed
in the report as not constrained by the data; it cannot make two fits differ by more than its own sigma.

**Fit standard** (used by `fit_standard`): one number per data kind, the rms of residual / tempered error at the
last fit's values (the same errors every rung was fitted on, not the error terms the last fit chose, which are set by the residuals
themselves), so a fit that would be absorbed into inflated error terms still has to be a fit. Closure phases count only on the
informative triangles (`INFORMATIVE_SNR`); the same rms on all triangles is reported beside it and is not judged.

## Not criteria

The scales, starts and priors of the plan (extent, reach, pixel; moment, stratified, map and rung-below starts; flux
priors; the global visibility gain width of 0.1; tempering; the likelihood-map peaks, which rank starts and never
threshold) are the method, not gates, and appear in each report with their sources. The error terms (`vis_error_rel`,
`phi_error`) are free only in one last fit after the ladder. The rung is chosen by chi2 on the tempered errors
(chi2 on the quoted errors is reported beside it) with ln N per added parameter; the image step runs after the ladder and is not a rung.

## Change log

| commit | what changed | changed after seeing a result: yes/no/unknown | note |
|---|---|---|---|
| (this file's first commit) | file created with the six checks and the thresholds of the plan, section 5 | yes | the thresholds were chosen knowing the second run's outcomes named in the table (2004 data1 flux 2.0, Obj2 companion 0.6 mas, 2006 nested and V^2 below 0.03); no run with these checks; Claude (Sonnet), for Ben, proposed; PR "informed fits: infrastructure" |
| (second commit) | `family_complete` fails when the image step errors (review of the PR); a chromatic rung's `starts_agree` is run on the chromatic stage's own starts; a run with no pinned criteria commit is refused, not warned | yes | the smoke runs of the PR (50 steps, not converged) had been seen; no full run; Claude (Sonnet), PR review |
| (third commit) | `extent_known` uses V^2 - 3 sigma on the short baselines and needs `EXTENDED_MIN` of extended light; the tolerance of `nested` and `starts_agree` is `min(AGREE_FRACTION x N, ln N)`; a ring's width prior floor is `WIDTH_FLOOR` (below `RING_FRACTION`); the constants table is parsed by a test; this change log and the header say the criteria are not blind (PR review) | yes | smoke runs only; the changes make the checks stricter or well-defined, none was tuned to a result; Claude (Sonnet), PR review |
| (recipe move) | `MIN_AGREE`, `AGREE_FRACTION` and the `min(1% of N, ln N)` tolerance are **replaced** by `START_AGREE` and `START_OF` (6 of 8 broad starts on the first rung, the same fraction of a larger budget, fewer than 8 cannot pass), `START_MARGIN` (25, higher rungs: best map peak and its 3 jittered starts at one optimum, the next distinct optimum worse by 25), `SAME_DELTA` (2) and `SAME_SIGMA` (3) with the practical tolerances; `nested` uses `SAME_DELTA`; new process check `fit_standard` (normalised residual rms at most `FIT_RMS_MAX` = 2 on V^2 and on informative closure phases, else "incomplete"), reported and not a gate; new constants `INFORMATIVE_SNR`, `INFORMATIVE_SHARE` for the information check | **yes** | the constants were changed **after seeing results** of the working recipe (README R12 of the contest results); they are proposed by the design session as process checks, not as acceptance thresholds (acceptance is an evaluator's judgement); none has been tested on a full run in this repository; Claude (Sonnet) implemented; PR "informed fits: recipe into the infrastructure" |
