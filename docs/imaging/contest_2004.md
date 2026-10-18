# The 2004 contest score

The 2004 imaging contest (Lawson et al. 2004) is the one year whose truth we
can rebuild from the paper, so it is the one place where an image is scored
against the published table. The scorer is
[`scripts/score_2004.py`](https://github.com/benjaminpope/virgil-validation/blob/main/scripts/score_2004.py)
and its tests are
[`test_score_2004.py`](https://github.com/benjaminpope/virgil-validation/blob/main/tests/test_score_2004.py).
The truth is used for scoring only, never for imaging; the metric is
Lawson et al.'s Eq. 2, which `score_2004.sigma_over_peak` reimplements and a
test checks against `crosscheck.image_metrics.lawson_sigma_over_peak`.

## The rebuilt truth

| Check | Result |
| --- | --- |
| Truth V² against the data, spot included, pixel grid | χ²/N 1.45 |
| Same, analytic visibilities (no pixel grid) | χ²/N 1.32 |
| Truth closure phases | χ²/N 0.61; the inverted truth 947 |
| Same truth, unit-sum normalisation | χ²/N ≈ 50, so the data's V² are normalised without the spot's excess flux (V(0) ≈ 1.07) |
| Spot clipped to the disk instead of added | χ²/N 2.8–3.0 |

Limb darkening is Van Hamme's bolometric u = 0.469 for 7000 K, log g 4; the
data pin u to 0.48 ± 0.02. The score is reported for u = 0.3, 0.469 and 0.7.

The checks against the data need the contest OIFITS (`~/data/imaging_contests`),
which CI does not have, so they are tier C: run them locally with the data
fetched. CI runs the geometry, normalisation, analytic-visibility and metric
checks; the metric is tested against `crosscheck.image_metrics`.

## Scores

σ/peak against the published truth, lower is better. Raw χ²/N is on the
quoted errors, split into V² and closure phase.

| Entry | σ/peak | Raw χ²/N (V², closure) | Status |
| --- | --- | --- | --- |
| BSMEM | 0.116 | | published |
| WISARD | 0.163 | | published |
| Parametric ladder: elliptical limb-darkened star, spot, companion | 0.180 | 0.985 (0.97, 1.01) | informed |
| CLEAN-started GP ensemble (4 of 8 members kept) | 0.253 | 20.7–21.3 (about 31, about 6) | blind |
| MIRA | 0.532 | | published |
| VLBMEM | 0.798 | | published |

The informed ladder was built knowing the object is a spotted star with a
companion; it is not a blind reconstruction and must not be quoted as one. Its
score depends on how the companion is drawn (a point source 0.351, a
free-diameter disk 0.180), and its diameters and limb darkening are 3–6σ from
the rebuilt truth. The blind score moves between 0.249 and 0.254 over u = 0.3
to 0.7, so it is not sensitive to the unpublished limb-darkening table.

Failing fits that remain, in raw χ²/N: 2004 data2 blind images about 20.5, the
2010 data 2500–3800 and 2024 Obj2 GRAVITY about 185.

Metric definitions and the published tables are in
[`design/contest_scoring_sources.md`](https://github.com/benjaminpope/virgil-validation/blob/main/design/contest_scoring_sources.md).
