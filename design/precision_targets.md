# Precision targets for the calibration campaigns

Registered 2026-10-12 as addenda (design/coldstart_pull_precision.yml,
design/pull_campaign_precision.yml, design/sbc_v2_precision.yml); none of the registered,
hashed criteria files was edited. The calculations are `scripts/precision_power.py`
(`python scripts/precision_power.py power`); the formulas, from textbook definitions, are
`src/crosscheck/precision.py`; `tests/test_precision.py` reproduces every number below.

## The problem

A campaign that fails to reject "calibrated" has not shown calibration. Pull v1 (N = 1250 per
cell) could see a 14% error in the error bars at 90% power; the cold-start campaign as
registered passed a cell with 12 failures in 1000 against a 1% bound (its upper limit is 2.4%);
SBC v2 at 1024 replicates could not see a posterior 10% too narrow. Detection v3 (design/detection_v3.yml)
fixed this for the detection tail by registering an equivalence margin and its power. Here the same.

## Design

For each campaign the smallest miscalibration it must rule out is registered as a margin, and the
claim "calibrated to the margin" is an equivalence test: two one-sided tests (Schuirmann 1987), or
equivalently the 1 - 2 alpha interval lies inside the margin. The claim is a conjunction (every
parameter of every cell, or every regime and quantity), so the tests form an intersection-union test
(Berger 1982): each at the family alpha, no correction, and the claim is made when all reject. The
power of the claim is then the product of the tests' powers; counts are raised until it is at least 0.8
(and each test's at least 0.8). The old Holm tests of miscalibration are unchanged.

## Cold-start campaign (not run)

Rule out a failure rate above `rule_out`; exact one-sided upper limit at alpha 0.01 per cell.
Power is that of showing it when the true rate equals the registered bound.

| cell | bound | rule out | N = 1000 (registered) | N = 1510 (minimum) | N = 1625 (planned) |
| --- | --- | --- | --- | --- | --- |
| binary_bright | 0.01 | 0.025 | 0.866 (k <= 13) | 0.980 (k <= 23) | 0.991 (k <= 26) |
| binary_faint, wide, sparse | 0.05 | 0.075 | 0.790 (k <= 55) | 0.948 (k <= 89) | 0.965 (k <= 97) |
| resolved_companion, envelope | 0.02 | 0.04 | 0.890 (k <= 25) | 0.985 (k <= 42) | 0.991 (k <= 46) |
| all six at once | | | 0.34 | 0.81 | 0.875 |

So the counts rise from 1000 to 1625 per cell (13 tasks of 125), 9750 draws, 78 tasks. Cost scales from
the earlier estimate (6000 draws, 8-17 core-hours) to 14-27 core-hours.
The same-mode pulls get an equivalence flag (|mean| < 0.15, sd in [0.9, 1.1], alpha 0.05 per side;
never pass/fail), with power per test 0.88 (mean) and 0.82 (sd) at 450 same-mode draws.

## Pull campaign (v1 complete; rerun registered)

Margins: sd in [0.9, 1.1], |mean| < 0.1, alpha 0.05 per side, 40 tests (20 parameters). Per-test
power at the true sd 1, mean 0, and all 40 at once:

| N per cell | sd | mean | all 40 |
| --- | --- | --- | --- |
| 1000 | 0.995 | 0.871 | 0.06 |
| 1250 (v1) | 0.999 | 0.941 | 0.29 |
| 1500 | 1.000 | 0.974 | 0.59 |
| 1750 | 1.000 | 0.989 | 0.80 |
| 1875 (rerun plan) | 1.000 | 0.993 | 0.86 |

### Precision v1 achieved (post hoc, N = 1250, 90% intervals)

SD: all 20 inside [0.9, 1.1]; the widest, star_rim_fit rim.flux, is [1.019, 1.090]. Mean: 19 of 20 inside
+-0.1; star_rim_fit rim.inc is [+0.009, +0.104]. So v1 certifies the error bars to 9% and the bias to 0.105 sigma
in every parameter; the claim at exactly 0.1 narrowly fails on one parameter, and at 0.11 holds. v1's own criteria
were met as registered, and this does not change that.

```
binary_fit                  dra            n=1250 mean +0.012 [-0.034,+0.057] in0.1=True  sd 0.979 [0.946,1.011] in=True
binary_fit                  ddec           n=1250 mean -0.043 [-0.089,+0.003] in0.1=True  sd 0.993 [0.961,1.025] in=True
binary_fit                  flux           n=1250 mean -0.038 [-0.085,+0.008] in0.1=True  sd 0.997 [0.965,1.029] in=True
binary_grid_flux            flux           n=1250 mean +0.002 [-0.044,+0.049] in0.1=True  sd 0.999 [0.967,1.031] in=True
resolved_star_companion_fit star.diam      n=1250 mean -0.009 [-0.055,+0.037] in0.1=True  sd 0.990 [0.958,1.023] in=True
resolved_star_companion_fit comp.dra       n=1250 mean -0.036 [-0.082,+0.010] in0.1=True  sd 0.995 [0.964,1.027] in=True
resolved_star_companion_fit comp.ddec      n=1250 mean +0.013 [-0.035,+0.060] in0.1=True  sd 1.021 [0.988,1.054] in=True
resolved_star_companion_fit comp.flux      n=1250 mean +0.034 [-0.013,+0.080] in0.1=True  sd 1.002 [0.968,1.035] in=True
star_envelope_fit           env.fwhm       n=1250 mean -0.033 [-0.079,+0.014] in0.1=True  sd 0.997 [0.965,1.028] in=True
star_envelope_fit           env.ratio      n=1250 mean +0.004 [-0.041,+0.049] in0.1=True  sd 0.969 [0.936,1.002] in=True
star_envelope_fit           env.pa         n=1250 mean -0.008 [-0.054,+0.038] in0.1=True  sd 0.985 [0.951,1.020] in=True
star_envelope_fit           env.flux       n=1250 mean +0.029 [-0.017,+0.075] in0.1=True  sd 0.990 [0.959,1.021] in=True
star_rim_fit                rim.diam       n=1250 mean +0.018 [-0.029,+0.066] in0.1=True  sd 1.015 [0.983,1.048] in=True
star_rim_fit                rim.fwhm       n=1250 mean +0.006 [-0.040,+0.052] in0.1=True  sd 0.994 [0.960,1.027] in=True
star_rim_fit                rim.inc        n=1250 mean +0.056 [+0.009,+0.104] in0.1=False  sd 1.015 [0.983,1.046] in=True
star_rim_fit                rim.pa         n=1250 mean +0.005 [-0.040,+0.049] in0.1=True  sd 0.960 [0.927,0.994] in=True
star_rim_fit                rim.az_amps    n=1250 mean +0.018 [-0.027,+0.063] in0.1=True  sd 0.975 [0.943,1.007] in=True
star_rim_fit                rim.az_pas     n=1250 mean +0.030 [-0.017,+0.078] in0.1=True  sd 1.024 [0.993,1.056] in=True
star_rim_fit                rim.flux       n=1250 mean +0.018 [-0.031,+0.067] in0.1=True  sd 1.054 [1.019,1.090] in=True
uniform_disk_v2_fit         diam           n=1250 mean -0.020 [-0.068,+0.027] in0.1=True  sd 1.021 [0.988,1.055] in=True
{'smallest_mean_margin_certified': 0.10355611302560719, 'smallest_sd_margin_certified': 0.08959491174718437} mean inside: 19/20 sd inside: 20/20
```

## SBC v2 (not run)

36 tests (2 regimes x 6 quantities x {coverage_68, coverage_95, mean_rank}), alpha 0.05 per side.
Margins: coverage_68 +-0.06 (nominal 0.68), coverage_95 +-0.03 (nominal 0.94: 94 of 100 ranks for L = 99),
mean rank +-0.04. They rule out a Gaussian posterior's width being wrong by about 12% (0.88 and 1.13 at 68%;
0.87 and 1.11 at 95%) and a bias of about 0.14 sigma.

| replicates per regime | coverage_68 | coverage_95 | mean rank | all 36 |
| --- | --- | --- | --- | --- |
| 1000 | 0.984 | 0.968 | 0.993 | 0.51 |
| 1024 (registered v2) | 0.986 | 0.971 | 0.994 | 0.56 |
| 1250 | 0.996 | 0.991 | 0.999 | 0.84 |
| 1270 (minimum) | 0.996 | 0.992 | 0.999 | 0.86 |
| 1280 (planned) | 0.997 | 0.993 | 0.999 | 0.87 |

Counts: 160 tasks of 8 per regime. Cost: 2560 replicates, about 7-18 core-hours, against 6-14 for 2048 (256 tasks).
The outcome has three states: pass, fail (a Holm rejection, or a failed sampler-health gate) and inconclusive
(nothing rejected, but a margin is not shown): the last is a signal, rerun on fresh seeds.

## Order and limits

The cold-start campaign runs before SBC v2: SBC v2's chains start from a fit and its low regime carries a
position measurement precisely because finding the mode is the cold-start campaign's job. SBC v2 says nothing
about multimodal posteriors; the limit is on the Trust page (trust/not_covered.md).

## Rerun on fresh seeds

Every criteria file here registers, under `must_hold`, that any campaign signal is rerun on fresh seeds (a
seed base the first run never used: 30261012 cold-start, 30261008 pull, 40261007 SBC v2) before it is ruled on,
and the ruling is the rerun's. The summaries list their `signals` and are `provisional` while one is open. The
rule is also in docs/method/index.md.
