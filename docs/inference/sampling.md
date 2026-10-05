# Sampling: simulation-based calibration

`scripts/sbc_numpyro.py`, run on OzSTAR (job 18077226, virgil `4afc5b8`):
the method of Talts et al. (2018), with 500 replicates. Each replicate draws a
binary from Jeffreys priors: log-uniform diameter, flux ratio, separation and
V² error scale, and a uniform position angle sampled as an `AngleVector`. Our
own simulator makes three-UT data from it, and NUTS (4 × 1000 draws) samples
`numpyro_model`'s posterior. Each truth is ranked among 99 thinned draws.
`tests/test_sbc_numpyro.py` reads the committed summary
(`trust/campaigns/sbc_numpyro_model.json`), and its evidence is in
`trust/evidence/campaigns.jsonl`, under the commit the campaign ran on.

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `likelihood.numpyro_model`: diameter, flux, separation, PA (`AngleVector`), `noise.vis_scale` | rank uniformity, χ² over 20 bins, Bonferroni family-wise 1 % (each p > 0.002) | Statistics | p = 0.035, 0.50, 0.54, 0.14, 0.35 |
| the same | 68 % and 95 % central-interval coverage, within the 99 % binomial band | Statistics | 64.6–68.0 % and 92.0–94.2 %, all within |

Sampler health is reported, not tested. The median replicate had 28
divergent transitions in 4000 draws. 65 of 500 replicates had a split R̂ above
1.01, and 59 had a bulk ESS below 400, mostly in `vis_scale` and `flux`, the
parameters whose log-uniform priors have hard bounds that the posterior
reaches. A few chains were stuck (worst R̂ 3.2). The ranks are uniform
nonetheless, so `numpyro_model`'s density is calibrated; the divergences
concern NUTS's geometry near those bounds.

