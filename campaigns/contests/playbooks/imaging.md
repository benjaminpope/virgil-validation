Sources: `design/plan_imaging_contests.md` and `design/contest_convergence_diagnosis.md`. They take methods from
the contest papers and the winning codes, never results. Read them before reporting a problem as new.

Two rules come first. Use only what contestants had before the deadline (`notes/given_<year>.md`); a field,
a morphology or a parameter from a contest paper is a leak, and it has happened here once (the 2004 fields).
And no check measures whether an image looks like the source: chi2/N near 1 is a data-consistent image, not a
correct one. Do not describe an image as right.

| What you see | What it is | What to do |
|---|---|---|
| chi2 flat across the regularisation weights; "line search ran out of float64 precision" | Per-pixel maximum entropy is badly conditioned on high signal-to-noise data | Not this campaign's path: members are Gaussian-process fits from a CLEAN start, which converge on the same data. If a member still stalls, report it. |
| One member far worse than the others | A start that stayed in a poor basin | Nothing. The ensemble's chi2 filter drops it; that is what the ensemble is for. |
| chi2/N far above 1 for every member, error scale well above 1.3 | The quoted errors are wrong (several contest datasets do this on purpose) | The member refits with rescaled errors. The evidence is unreliable there: say so, do not tune. |
| chi2/N in the hundreds or thousands on 2010 or 2024 Obj2 | The source is chromatic; a grey image cannot fit it | Known and open: it needs per-channel images (stage C3, not built). Report; do not rescue. |
| 2004 data2 stays near chi2/N 20 | Known and open; V² dominates the misfit | Report. The field rule already picked a field wide enough from the data alone. |
| The star and no-star members disagree | Whether a central star is present is decided from the data | Nothing: the ensemble compares the best evidence of each. Choosing between them from the data is allowed (Ben, 2026-10-05). |
| Flux reaches the edge of the field | The field may be too small | Data-chosen fields grow by 1.5x, at most three times, while that lowers chi2 by more than 10%. A contest-given field is used as given. Never take a field from a paper. |
| A 2022 GRAVITY header names a real star | Probably unintended by the organisers | Ignored. Treat the data as blind. |
