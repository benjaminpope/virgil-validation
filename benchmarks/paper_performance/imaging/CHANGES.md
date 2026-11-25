# Changes to the predeclared imaging cases after the first commit

A later change to `cases.yml` is a new commit with a line here giving the reason; no setting is changed in response to a
timing or to which code is ahead.

| Date | Commit | Change | Reason |
| --- | --- | --- | --- |
| 2026-11-19 | PR 275 | `ladder.time_to_target` replaced by `ladder.time_to_precision` (rungs 10 %, 3 %, 1 %, 0.3 %, 0.1 % of J_ref; headline = tightest rung both codes reach; cold time to precision added) | run 1's single 1e-3 target was unreachable for eht-imaging on I-sim-24/48/96 (its own minimum lies above the J minimum; see cases.yml `why_changed`), so `not_reached` was a property of the target, not a speed. Imaging is compared at equal precision, not per step (Ben, 2026-11-19). No budget, optimiser or data setting changed |
