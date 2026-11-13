This stage is informed, not blind. It is told what the contest paper says the object is and nothing about its
values. Never take a diameter, a position, a flux ratio or an angle from a paper or from memory, as a start, a
bound or a constant; and never describe an informed result as a reconstruction or quote it beside blind scores
without the word "informed".

| What you see | What it is | What to do |
|---|---|---|
| `starts_agree` fails: fewer than two starts reach the best -2 ln L | The optimum was found by one start, or is not unique | A result: say so. More starts is a rerun and is counted; ask a person. |
| `no_bounds` fails: a fitted parameter sits within 1% of the edge of its prior (a size at its smallest, a flux at 1e4) | The data do not constrain it, or the fit is in a local optimum | Report which. Do not narrow the prior toward a published value; the scales (extent, reach) are from the data. |
| `nested` fails: a richer rung fits worse than a rung it contains | The optimizer missed the optimum of the richer rung | A failure of the fit, not a result about the object. Look at the report's starts by origin; more starts is a counted rerun. |
| `family_complete` fails: the image fitted on top of the chosen model carries over a fifth of its flux | The family leaves out much of the object | A result: report it. A new component is a new family, by pull request. |
| `no_collapse` fails: an arc shorter than the resolution, a ring narrower than a fifth of it, or a companion within two resolution elements of the star | A component has shrunk to something else | Report it. Do not widen the prior around a published value. |
| `extent_known` fails: the extent came from neither the contest's field nor a constrained moment fit | The data bound the size only from below | Report it: the size priors may cut off the object. |
| BIC prefers a simpler rung than the paper describes | The data do not support the extra component | A result. Report the chosen rung. Do not force the full model. |
| chi2/N on the quoted errors stays well above 1 on the richest rung | The family does not describe the data, or the quoted errors are wrong | Report it; the rungs are chosen by BIC at fitted noise, and chi2 on the quoted errors is never what chooses. A new component is a new family, by pull request. |
| The dataset's family is "not built" | No one has written that model yet | Nothing to run. `mandate request` if it is wanted, naming the virgil classes. |
