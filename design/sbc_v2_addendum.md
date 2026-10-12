# SBC v2 addendum: what the v1 divergences were, and the low regime (2026-10-12)

Planning note, off the site. Criteria: `design/sbc_v2_addendum.yml` (hashed with
`design/sbc_v2.yml`, which is unchanged). Probe script: `scripts/sbc_geometry_probe.py`; raw
results: `design/probes/*.json`. All numbers below are diagnostics on 6 to 32 replicates, not
a campaign: virgil `0c1be69` (0.4.0), 4 chains, 500 warm-up, 500 samples (1000 for the low regime),
v1's truths and data by seed.

## SBC v1: where the divergences come from

v1 (`trust/campaigns/sbc_numpyro_model.json`, virgil 0.2.0) had divergences in 978 of 1000
replicates (111 with R-hat > 1.01, 97 with bulk ESS < 400), correlating with the companion's
flux (rho 0.38-0.40). Reproduced on today's virgil: all 6 of 6 probe replicates diverge under v1's
sampler, 115 divergent draws in 12 000, with a median of 549 leapfrog steps per draw (the tree
depth limit is 1023), against 6 to 50 for the variants below.

The model's gradient is right: the potential energy's JAX gradient agrees with central finite
differences to 2e-7 (relative, 12 points on two replicates, including small ring radii). It is
the geometry, and it comes from how v1 samples the position angle: an `AngleVector` ring prior
of width 0.25 on a 2-D vector, with numpyro's default diagonal mass matrix. The data fix the angle
to a fraction of a degree, so in the vector's plane the posterior is a thin arc of width
r sigma_theta across and 0.25 along the radius; the arc's width scales with r, so
its curvature varies by (r1/r2)^2 across the radial extent: a funnel, which a diagonal mass matrix
cannot whiten. Divergent draws sit at a smaller ring radius than the rest (0.74 against 1.06
in the first replicate; 0.65 against 0.93 in the last).

Variants on the same 6 bright replicates (all chains at the truth, as v1):

| variant | replicates with divergences | divergent draws | median leapfrog steps | median min bulk ESS | R-hat > 1.01 |
|---|---|---|---|---|---|
| v1 (diag mass, AngleVector 0.25) | 6/6 | 115/12000 | 549 | 898 | 1 |
| v1, target_accept 0.95 | 6/6 | 27/12000 | 745 | 935 | 1 |
| v1, vis_scale fixed | 6/6 | 188/12000 | 517 | 776 | 0 |
| position angle as a plain angle (no ring; diagnostic) | 0/6 | 0 | 52 | 772 | 0 |
| ring 0.1, dense mass on (pa_vec, sep, flux) (v2's sampler) | 0/6 | 0 | 41 | 1000 | 0 |
| Cartesian offsets (dra, ddec) ~ U(-25, 25), diag mass | 0/6 | 0 | 25 | 681 | 0 |
| Cartesian, dense mass | 0/6 | 0 | 24 | 668 | 2 |
| fit, then virgil's `gauss_newton_mass` (polar, fixed errors) | 2/6 | 3 | 6 | 2024 | 0 |
| Cartesian, `gauss_newton_mass` (fixed errors) | 0/6 | 0 | 5 | 2131 | 0 |

So the divergences are our configuration, not a virgil defect: v1 used the default diagonal
mass with the default ring. v2's narrower ring with a dense block already removes them, and
virgil's own `gauss_newton_mass` (documented in `virgil.fitting`, not used by v1 or v2) is
the cheapest cure (5 to 6 steps per draw, ESS 2000+). Raising the target acceptance only
trades divergences for tree depth. No ledger entry (F22 is not needed: there is no virgil mismatch
to pin with a strict-xfail test).

## SBC v2 as registered

Run replicate by replicate with the registered sampler (grid-fit start, jitter, ring 0.1, dense
mass), 4 chains x 1000 draws:

- bright regime: 1 of 32 replicates unhealthy (3%; the gate is 5%): replicate 910005
  (flux 0.020, sep 18.5 mas) had R-hat 25 and ESS 2, chains in different position lobes, no
  divergence. Zero divergences in 32.
- low regime: 5 of 8 unhealthy (63%; gate 5%): R-hat 4 to 27 and bulk ESS 2, with 0 to 4
  divergences. Not geometry: with `jitter` set to 0.1x and to 0 the same replicates
  (10900001, 4, 5) stayed unhealthy, since the grid-fit start itself sits in a lobe that is not
  where the posterior has its mass. Started at the truth, every chain agrees (24/24 in the truth's
  lobe), and Cartesian offsets are healthy (6/6), but the narrow-ring sampler is unhealthy in 2/6
  even from the truth, and `cart_dense` in 1/6.
- started from prior draws, no fit: 0/24 chains reach the truth's lobe in 17 of 18 runs
  (R-hat 300 to 1400, for the ring, Cartesian and Cartesian-dense variants): NUTS does not find the
  mode. Finding it is the grid search's and the fit's job, which the cold-start campaign
  (`design/coldstart_pull.yml`) now tests.

The low regime's posterior in the companion's position is multimodal at fringe spacing, and
no parameterisation mixes between the lobes. The gate would fail it whatever its ranks, for a cause
that is not the model's.

## The change registered (addendum)

`sbc_v2_addendum.yml` replaces `low` by `low_astrometric`: the same flux range (1e-3 to 2e-2), plus one
datum per replicate, a measurement of the companion's offset (truth + N(0, 0.5 mas) per axis), entering
the posterior as a Gaussian likelihood term (`likelihoods=`). The chains start at the measured offset
(the diameter and flux from `likelihood_grid` over (diam, flux) at that offset on a 60-point diameter
axis, then `fit` with the measurement term), jittered by 0.1 mas. The prior is v2's, so the SBC stays
exact (the measurement is data). 24 of 24 probe replicates were healthy (R-hat <= 1.005, ESS >= 1658,
0 divergences). A first version of the start, with v2's seven-point diameter axis, left
replicate 10900006 unhealthy: `fit` from a diameter 0.2 mas off moved the position 3 mas to another lobe;
the finer axis is the fix, and a cold-start lesson in itself.

Everything else in v2 stands, including the 5% unhealthy gate: the bright regime sits at about 3% on 32
replicates, so a pass is not assured, and a failure would be a real mode-finding limit of the v2 start.

## Cost (OzSTAR, CPU, one core and 4 GB per task)

On the loaded laptop (load ~20 on 10 cores): about 20 s per bright replicate and 12 s per
low replicate (compile included), 1024 of each: 5.7 + 3.4 = 9 core-hours; 256 tasks of 8
replicates (2.7 and 1.6 min each, plus start-up). v1 took about 90 s per replicate (25 core-hours).

## Precision, counts and order (addendum 2, 2026-10-12)

design/sbc_v2_precision.yml registers the margins SBC v2 must show (coverage and mean-rank equivalence, power 0.87 at
1280 replicates per regime) and states that the campaign says nothing about multimodal posteriors, which the 0.5 mas
measurement removed from the low regime (see trust/not_covered.md). The cold-start campaign runs before SBC v2.
Rationale and numbers: design/precision_targets.md.
