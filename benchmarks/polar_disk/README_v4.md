# V4 on OzSTAR: the full injection-recovery run

`v4_injection_recovery.py --config full` runs 200 injections for each of NRM and VLTI coverage and
M = 1, 2, 3 (1200 fits), with `n_r = 64`, a large perturbation of the starting point and 8 starts per
injection. Nothing here launches it: Ben submits it. The laptop configuration (`--config laptop`: five
injections, VLTI, M = 2, `n_r = 24`, start at the truth plus a small perturbation) was run for the PR and
gives a pipeline check only (about two minutes; its numbers are in `docs/models/polar_disk.md`).

Before the full run:

* Run it on the same virgil commit that the results record (the script writes it into
  `results/v4_full.json` with the package versions). The pin for this PR is
  `e42973d7478e04d704aaee53e61810e0356c6e49` (`polar-p4`).
* The laptop result flags the gap depth (z-scores of 6.9 and 9.6 in two of five injections, depth near
  its bound of 1). The full run is where that is tested; do not tune the criteria first.
* Cost: the laptop run took about 20 s per fit (compilation included, M = 2, `n_r = 24`); the full run is
  1200 injections x 8 starts at a larger `n_r`, so tens of core-hours on CPU: split it over array tasks (one
  instrument and M per task, a seed offset each). A GPU is not needed.
* Criteria K1 to K4 are in the script's docstring. Report all four, including failures.
