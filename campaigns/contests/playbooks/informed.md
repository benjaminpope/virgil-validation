This stage is informed, not blind. It is told what the contest paper says the object is and nothing about its
values. Never take a diameter, a position, a flux ratio or an angle from a paper or from memory, as a start, a
bound or a constant; and never describe an informed result as a reconstruction or quote it beside blind scores
without the word "informed".

| What you see | What it is | What to do |
|---|---|---|
| `starts_agree` fails: on the first rung fewer than 6 of 8 broad starts end at one optimum (or fewer than 8 were run); on a higher rung the best map peak and its 3 jittered starts do not end at one optimum, or the next distinct optimum is less than 25 worse in -2 ln L | The optimum was found by luck, or is not unique, or the landscape has a rival | A result: say so, with the record's `others`. More starts is a rerun and is counted; ask a person. |
| `fit_standard` is "incomplete": after the last fit the normalised residual rms is over 2 on V^2 or on the informative closure phases | The fit converged but the model leaves structure the data show, or the error terms absorbed it | Not a gate: it is reported with every fit as a process check. Say so in the record, with the rms and which data; whether the fit is accepted is an evaluator's judgement. A new component is a new family, by pull request. |
| `no_bounds` fails: a fitted parameter sits within 1% of the edge of its prior (a size at its smallest, a flux at 1e4) | The data do not constrain it, or the fit is in a local optimum | Report which. Do not narrow the prior toward a published value; the scales (extent, reach) are from the data. |
| `nested` fails: a richer rung fits worse than a rung it contains | The optimizer missed the optimum of the richer rung | A failure of the fit, not a result about the object. Look at the report's starts by origin; more starts is a counted rerun. |
| `family_complete` fails: the image fitted on top of the chosen model carries over a fifth of its flux | The family leaves out much of the object | A result: report it. A new component is a new family, by pull request. |
| `no_collapse` fails: an arc shorter than the resolution, a ring narrower than a fifth of it, or a companion within two resolution elements of the star | A component has shrunk to something else | Report it. Do not widen the prior around a published value. |
| `extent_known` fails: the extent came from neither the contest's field nor a constrained moment fit | The data bound the size only from below | Report it: the size priors may cut off the object. |
| BIC prefers a simpler rung than the paper describes | The data do not support the extra component | A result. Report the chosen rung. Do not force the full model. |
| chi2/N on the quoted errors stays well above 1 on the richest rung | The family does not describe the data, or the quoted errors are wrong | Report it; the rungs are chosen by chi2 on the tempered errors with ln N per added parameter, and the error terms are free only in the one last fit. A new component is a new family, by pull request. |
| The dataset's family is "not built" | No one has written that model yet | Nothing to run. `mandate request` if it is wanted, naming the virgil classes. |

## The recipe, in order

`scripts/contest_informed.py` runs these steps for one dataset and one family. The criteria are `design/contest_informed_criteria.md`;
none of the steps takes a number from a paper.

1. **Pin the criteria.** The file's last commit is recorded; a run with uncommitted criteria is refused before any data are read.
2. **Information check.** Before any fit: the median V^2, the median quoted closure-phase error, and how many triangles have all
   three V^2 above 3 sigma. If under 5 % do, the closure phases carry no information: they are still fitted, the fit is judged on V^2,
   and nothing is oriented by them.
3. **Scales.** Resolution, extent, reach and pixel from the baselines, and the visibility moments (a point, an elliptical Gaussian and
   over-resolved light), each with its source. These start the first rung.
4. **Gain nuisance.** One global visibility gain on log |V| is marginalised, with a prior width of 0.1, at every rung and in the last
   fit. Nothing is divided out of the data; a rung that lacks structure cannot bias the scale.
5. **Tempering.** The first rung's fit gives one factor (at least 1) by which the errors of every ladder fit are multiplied, so the
   ladder compares structure at the errors the data can support. The rung is chosen on chi2 at those tempered errors (chi2 at the quoted errors is
   reported beside it), with ln N per added parameter.
6. **The ladder.** The first rung starts at the moment fit and at broad stratified starts (sizes in log steps, angles on a grid).
   A rung above starts from the rung below with what it adds at negligible flux. A positioned component (a companion, a planet) is
   placed first by a likelihood map over the reach; the map **ranks** its peaks and never thresholds them, the best five are refitted,
   and three more starts are jitters of the best peak (inherited values moved by under 20 % in size and flux, 10 degrees in angle, a
   tenth of a resolution element in position). A rung that places nothing starts from the rung below and jitters of it. A rung whose
   reference component is resolved lets the rung below re-optimise at every map position. Every family ends with a
   `+resolved` rung of over-resolved light.
7. **The last fit.** The chosen rung is fitted once more with the error terms (`vis_error_rel`, `phi_error`) free. Only this fit has
   them free.
8. **Checks.** Start agreement per rung (above), nesting, bounds, collapse, extent, and the **fit standard**: after the last fit the
   normalised residual rms is at most 2 on V^2 and on the informative closure phases, else the fit is "incomplete".
9. **After the ladder, not rungs.** A non-parametric image fitted on top of the chosen model (`family_complete`), a CLEAN probe on top
   of it (where its components cluster says what a next rung would add), and with `--image` an image step that replaces the extended
   light by a Gaussian-process image. None is a rung and none changes the chosen rung.

A run that pauses (`--time-budget`) keeps every fit, grid point and map tile in `--cache` and exits with 75; run it again with the
same cache to continue. `--smoke` checks that the script runs (a few starts of 50 steps, a coarse map); it is never a result.
