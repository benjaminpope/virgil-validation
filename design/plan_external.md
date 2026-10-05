# Plan: validating virgil against other packages

## Why

Everything in this repository so far is checked against our own
first-principles code or dLux. Our code shares an author type with virgil
(an AI agent), so a misconception common to both could pass. CANDID
(Gallenne et al. 2015, A&A 579, A68) and PMOIRED (Mérand 2022, SPIE 12183)
are mature, human-written packages in daily use on VLTI data, and
virgil's binary search was modelled on CANDID. Agreement with them, on
files we simulate, is the strongest outside check available. Where the three
codes disagree, our analytic references decide which is right.

## A pool of trusted packages

The external packages are a pool, not a checklist: any link in the
[trust graph](../docs/trust.md) may be validated against whichever of them suits it,
and no stage needs every package. Natural pairings:

| Link in virgil | Packages that suit it |
| --- | --- |
| parametric models, OIFITS reading, fits and their uncertainties | PMOIRED (done: Stages 0–2), CANDID |
| binary detection maps and contrast limits | CANDID, fouriever |
| closure-phase correlations | fouriever (Kammerer et al. 2020) |
| aperture masking | dLux (done), fouriever |
| image reconstruction: regularisers, RML imaging, image Fourier transforms | [eht-imaging](https://github.com/achael/eht-imaging) (Chael et al.), [MPoL](https://github.com/MPoL-dev/MPoL) (Czekala et al.) |
| a whole pipeline | published results reproduced from the authors' data (root: literature) |

The stages below are the plan for each package where it is used; doing them
for every package is not the goal.

## Packages

[fouriever](https://github.com/kammerje/fouriever) (Jens Kammerer;
Kammerer et al. 2019, 2020, 2023) joins as a third external package: CANDID-style
χ² maps and detection limits that can account for correlated observables,
bandwidth smearing, and a focus on JWST NIRISS aperture masking and kernel
phase. It is on PyPI (0.4.3) but states no licence, so, like CANDID, it is
installed and called, never vendored or copied; its `pymultinest`
dependency means it gets its own environment, called through a subprocess.
It ships test data (AB Dor NIRISS AMI, AX Cir PIONIER, β Pic GRAVITY) that
serve the reader checks. Its stages: conventions and binary forward model
alongside CANDID (Stage 7 of the roadmap), closure-phase correlations
against virgil's whitening (7b), and detection limits three ways (8).


| | CANDID | PMOIRED |
| --- | --- | --- |
| Source | <https://github.com/amerand/CANDID> (not on PyPI; the PyPI package `candid` is unrelated) | `pip install pmoired` (26.10.1, Python ≥ 3.8) |
| Licence | none stated: install and call it, never vendor or copy code | BSD-2-Clause |
| Scope | binaries: χ² maps, companion fits, detection limits (Absil and injection methods), bandwidth smearing | general parametric models (disks, Gaussians, rings with azimuthal modulation, binaries), fits, bootstrapping, grid searches, detection limits |
| Data | OIFITS (V², closure phases) | OIFITS (V², closure phases, differential phases, spectra) |

Both go in an optional extra, `pip install -e ".[external]"`, pinned to an
exact version (CANDID to a commit SHA), behind a `pytest.mark.external`
marker so the core suite never depends on them. Updating a pin is a
deliberate PR that reruns everything.

## Stage 0: install, read, and pin conventions (1 day)

**PMOIRED: done**, see [pmoired_conventions.md](../docs/method/pmoired.md). **CANDID: done**, see [candid_notes.md](../docs/method/candid.md). **fouriever: done** for correlated closure phases, see [fouriever_notes.md](../docs/method/fouriever.md).

1. Install both in the validation venv and record the versions in
   `docs/results.md`.
2. Read each package's documentation, not its code, for its parameter
   conventions, as we did for virgil. Write a mapping table in
   `docs/<package>_conventions.md`. Each convention below is
   unverified until a test pins it:
   * position: CANDID `x, y` and PMOIRED `x, y`. Which axis is East, and
     the sign;
   * flux: CANDID `f` (percent of the primary), PMOIRED `f` (absolute,
     per component);
   * size: PMOIRED `ud`, `fwhm`, `diam`/`thick` or `diamin`/`diamout`;
     CANDID `diam*` (the primary's UD diameter);
   * orientation: PMOIRED `incl` and `projang` (which axis it is measured
     to), and `az ampN`/`az projangN` (sky or in-plane azimuth? the same
     ambiguity as virgil's finding 1);
   * closure-phase sign and triangle orientation as each reads T3;
   * bandwidth smearing: how to switch it off (CANDID by spectral
     resolution or config; PMOIRED's smearing setting) so that the
     monochromatic comparison is exact.
3. Pin the conventions with the cheapest test: our OIFITS file of an
   offset point companion, read by each package, whose model V² and closure
   phases at the true parameters must equal ours. Any sign or axis flip
   shows up as O(1) closure-phase errors.

The package code lives in a new `src/external_bridge/`, which may import
CANDID and PMOIRED but not virgil. Comparisons go in tests, as now.

## Stage 1: forward models on identical files (2 days)

**PMOIRED: done** (`tests/test_pmoired_vs_virgil.py`). CANDID: to do.

For every shape the packages share with virgil, evaluate the model
observables of each package on the same OIFITS file (ours) at the same
parameters, and compare V² and closure phases:

| Shape | virgil | PMOIRED | CANDID | Analytic |
| --- | --- | --- | --- | --- |
| point binary | `BinaryModelCartesian` | two components | companion model | yes |
| UD primary + companion | `System(UniformDisk, PointSource)` | `ud` + point | `diam*` + companion | yes |
| Gaussian, elliptical Gaussian | `GaussianDisk`, `EllipticalGaussian` | `fwhm`, `incl`, `projang` | — | yes |
| ring, Gaussian-blurred, inclined | `ModulatedGaussianRim` | `diam`, `thick` (or profile), `incl`, `projang` | — | ring quadrature |
| azimuthally modulated ring | `az_amps`, `az_pas` | `az ampN`, `az projangN` | — | ring quadrature, both readings |

Expected agreement is rounding error for the analytic shapes, once the
conventions are mapped. Where a package's profile differs from virgil's by
definition (e.g. a ring with a uniform rather than Gaussian radial
profile), compare each with our quadrature of its own definition instead.

## Stage 2: fits on identical files (2 days)

**PMOIRED: in progress** (`tests/test_pmoired_fits.py`). Done: binary, disk star + companion and star + envelope (best fits, sigmas, correlations, 200-draw pulls with acceptance bands for both packages). To do: the modulated rim (a difference of definition, see the conventions page) and PMOIRED's bootstrap errors. CANDID, fouriever: to do.

Fit each scene from `virgil_bridge.SCENES`, written by our simulator,
with each package from the same starting point and priors as close as
the packages allow:

* noise-free: all three recover the truth to their optimiser tolerance;
* noisy (the files used for virgil's pulls): best-fit parameters agree to
  well within one σ, and the uncertainties agree (virgil's Laplace
  covariance, PMOIRED's covariance and bootstrap, CANDID's fit errors). For
  50 realisations, compare the pull distributions of all three.

Closure-phase covariance is a known difference: virgil whitens closure
phases as a correlated group, while others may treat triangles as
independent. Run both of our noise models (`phase_noise="baseline"` and
`"triangle"`) and expect their error bars to differ in the predictable
direction.

## Stage 3: binary search and detection limits (3 days)

This is the main comparison with CANDID, the code virgil's binary search
was modelled on.

**CANDID: done** for χ² maps, significance, Absil limits and fits
(`tests/test_candid.py`, [report](../docs/method/index.md#against-candid)). Still to do:
the injection method and our own injection-recovery campaigns.

1. χ² maps: CANDID's fit map vs `virgil.grid_fit` (`likelihood_grid`
   and the optimised grid) on the same file and grid. Best positions agree
   to a small fraction of a beam. Detection significance (nσ) agrees within
   the differences in how they convert χ² to significance; document those.
2. Detection limits: CANDID's Absil and injection methods vs
   `virgil.limits.absil_limits` and `ruffio_upperlimit`, on
   companion-free files at several noise levels. Contrast curves agree to
   ~0.1 mag where the methods are the same; where they differ (Ruffio's
   upper limit has no CANDID equivalent), compare with our own injection
   recovery: inject companions of known flux, record detection fractions,
   and check each code's 99.7 % limit sits where roughly 99.7 % are found.
3. PMOIRED's grid search and detection limits as a third opinion on the
   same files.

## Stage 4: aperture masking files (1 day)

Run Stages 1–3 on the dLux masking files from `tests/test_nrm.py`
(single snapshot, 21 baselines, 35 closure phases). This exercises
closure-phase redundancy, which is heavier for 7 holes than for 4
telescopes, and the wide-separation regime where CANDID's smearing
matters if not switched off.

## Stage 5: reporting and CI (1 day)

* Add an "External packages" section to [`report.md`](../docs/method/index.md) with the
  tag **CANDID** or **PMOIRED** for each check, and the versions used.
* `scripts/report.py --external` writes the comparison tables and a
  three-way contrast-curve figure.
* CI: an `external` job, weekly and on dispatch only (CANDID's maps are
  slow and multiprocess), installing the pinned versions.
* Report disagreements upstream: on virgil as findings here, and to
  CANDID or PMOIRED as Issues on their repositories, but only after our
  analytic references show which code is wrong, and with the user's
  approval before anything is posted.

## Risks

* **Convention mapping is the work.** Most of the effort is Stage 0; a
  wrong mapping looks like a disagreement. Pin each with a single-point or
  single-shape file before comparing anything richer.
* **CANDID's packaging.** It has no PyPI release, no stated licence and
  older dependencies (it uses multiprocessing and interactive plotting).
  Run it in a subprocess with a non-interactive backend and a timeout. If it
  needs an old Python or NumPy, give it its own venv and exchange results
  through JSON files.
* **Different estimators.** Detection limits and significances differ by
  design between packages, so agreement is expected only where the
  methods coincide. Elsewhere our injection-recovery experiment is the
  referee.
* **Cost.** CANDID χ² maps over fine grids take minutes. Keep the test
  grids coarse and the full maps to the weekly job.

## Order of work

Stage 0 → 1 with PMOIRED first: it is pip-installable, licensed, and covers
more shapes. Then CANDID for Stages 0, 1 and 3, then Stage 2 for both,
Stage 4, and Stage 5. About two weeks of agent time, each stage a separate
PR with its own tests.
