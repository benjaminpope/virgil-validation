# M87* replication (Lai et al. 2026): results and findings

Results against the criteria in [m87_replication.md](m87_replication.md). Kept in a separate file so the criteria
file, and with it `criteria_commit` (e6e4039), stays unchanged.

## Stage 1 (local; `trust/campaigns/m87_stage1.json`)

REX on the pixelwise median of the 50 published 225 μas samples, X = `crosscheck.ring_extraction`,
V = `virgil.metrics.ring_features`.

| Day | X 2r_pk | X w_r | X η | X A | X f_c | V 2r_pk | V η | V A | Table 1 2r_pk, η, A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2017-04-11 | 44.18 | 14.13 | 166.8 | 0.255 | 0.005 | 44.18 | 166.4 | 0.259 | 43.3 ± 1.0, 167 ± 6, 0.20 |
| 2018-04-21 | 43.40 | 8.55 | 211.7 | 0.320 | 0.002 | 43.40 | 211.7 | 0.322 | 42.7 ± 0.5, 205 ± 10, 0.31 |
| 2021-04-18 | 41.04 | 10.53 | 201.2 | 0.333 | 0.003 | 41.04 | 201.6 | 0.337 | 39.9 ± 1.8, 197 ± 4, 0.26 |

- **1a pass** on all three days (diameter and η within the tolerances).
- **1c pass** on all three days (virgil agrees with the independent extractor).
- **1d pass** (`tests/test_m87_rings.py`).
- **1b not run**: T needs `.venv-basis`, which could not be built here (installing their code locally was refused).
  It runs in `basis.sif` on OzSTAR. Informative only: their own notebook's `RingFit` on the 2021 median gives
  2r_pk 40.66, w_r 7.98, η 203.12, A 0.305. Against X that is ΔA = 0.028 (> 0.02), so 1b would likely fail on A.
  Their notebook value is itself outside Table 1 for η, w_r and A.
- Not in any criterion, reported: A on the median image is above Table 1 on 2017 (0.255 vs 0.20 ± 0.01) and 2021
  (0.333 vs 0.26 ± 0.01); w_r is below Table 1 on all three days. Table 1 is a mean over three fields; only one
  field is published, so these are findings about the field dependence or the table, not tuning targets.

## Facts found while writing stages 2–5

- BASIS computes closure invariants with its own code, not the ClosureInvariants package.
- 3a (≤ 1e-8) is unreachable as pre-registered: BASIS works in complex64 and uses its own arcsec constant
  (206265). The agreement is 7.6e-8 after matching the constant. Recorded as a pre-registered fail with this reason;
  the criterion is not changed.
- BASIS's `sky_map` places the floor Gaussian with the opposite sign of offset to its documented convention.
- The full covariance C = J Σ Jᵀ is singular (invariants are redundant); stage 3b uses a tangent-space projection.
- A nested virgil `System` needs `flux=`.
- `virgil` `read_ms` refuses a Stokes-I-only MS (to be filed as a strict xfail with a coordinator note).
- bilby 2.8.2 does not run with dynesty 3.2.0 (`apply_reflect` removed); dynesty is pinned to 3.1.0.
