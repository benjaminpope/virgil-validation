# virgil-validation

Documentation: <https://benjaminpope.github.io/virgil-validation/>

Independent checks of [virgil](https://github.com/benjaminpope/virgil)
(`pip install virgil-astro`), the interferometry package formerly called
drpangloss.

virgil's own test suite was written mostly by AI agents, and mostly checks
virgil against other parts of virgil. A sign convention, a factor of two or
a misread paper can then be consistent everywhere and still wrong. This
repository simulates data with code that shares nothing with virgil, and only
lets virgil *read, model and fit* the result:

* **Geometric primitives** (`src/crosscheck/sky.py`): every shape is a
  point cloud from a quadrature rule over its brightness (Gauss–Legendre,
  Gauss–Hermite, trapezoid in azimuth), and where one exists a textbook
  closed form evaluated with SciPy (`2 J1(x)/x` for disks, Gaussians,
  shifts). The two routes agree with each other to 1e-15 before either is
  compared with virgil.
* **Long-baseline interferometry** (`array.py`, `oifits_writer.py`,
  `simulate.py`): uv tracks from station coordinates by Earth rotation
  (Thompson, Moran & Swenson eqs. 4.1–4.4), V² and closure phases formed
  from our visibilities, written to OIFITS v2 by our own astropy writer, and
  read back by virgil's reader.
* **Aperture masking** (`nrm.py`): each scene is imaged through a
  NIRISS-like 7-hole mask both by [dLux](https://github.com/LouisDesdoigts/dLux)
  and by the closed-form interferogram (Airy envelope × fringes), point by
  point. Observables are the image's Fourier transform at the baselines
  divided by a point-source calibrator's.
* **Imaging machinery**: virgil's dirty image, beam, Nyquist pixel, field of
  view, beam convolution, rendering and rotated-lattice matrix Fourier
  transform, against analytic answers.

The independent side (`src/crosscheck`) may not import virgil; a test
enforces it. `src/virgil_bridge` is the only code that does.

## What is checked

| Area | Check | Agreement |
| --- | --- | --- |
| Visibilities | `PointSource`, `GaussianDisk`, `EllipticalGaussian`, `UniformDisk` (through four Airy nulls), binaries, `ModulatedGaussianRim` (inclined, m = 1, 2, blurred in the rim plane as since virgil#139), `Image` orientation, `Rotated`, `Resolved`, nested `System`s | 1e-15 (float64) |
| Visibilities | 5 random constellations of 24 points, Gaussians, elliptical Gaussians and disks at random positions and fluxes, plus a nested group | 1e-12 (float64), 3e-5 (float32) |
| Derivatives | the visibilities of every geometric model (including the flared disks; not `HarmonixModel`), `Image` pixels, `whitened_residuals`, `model_loglike` and its Hessian, against finite differences | 1e-5 (check_grads); Hessian < 1e-7 (Richardson) |
| uv geometry | our Earth-rotation tracks vs `virgil.coverage.vlti_oidata` | 1e-9 m |
| OIFITS | our file → `OIData` → `data.model(...)` reproduces what we wrote (V², T3 orientation, signs, units) | 1e-16 |
| Recovery (VLTI) | noise-free fits of a binary, a uniform-disk star with a companion, a star with an elliptical envelope and a star with a modulated rim | 1e-10 to 1e-13 relative (rim: 1e-7, optimiser tolerance) |
| Limb darkening ([#12](https://github.com/benjaminpope/virgil-validation/issues/12)) | `LimbDarkenedDisk` (uniform to order 22), `QuadraticLimbDarkenedDisk`, `SquareRootLimbDarkenedDisk` and `cvis_limb_darkened_disk` with fractional powers, through four nulls, against our quadrature cloud, which agrees with a direct Hankel transform and Hanbury Brown et al.'s (1974) linear-law closed form to 1e-14; Kipping (2013) maps both ways and the unit square ↔ physical laws; `is_physical`; `render`; an oversampled pixel image of an off-centre star with a companion through OIFITS; noise-free recovery of diameter and q₁, q₂; PMOIRED's sampled profiles | 2e-14 (order 22: 7e-12, binomial cancellation); float32 2e-7; pixels 7e-5 in V²; recovery 7e-10 relative; PMOIRED 3e-7 (quadratic) and 2.5e-6 (square root) in V² at `Nr` = 10000, converging as Nr⁻¹·⁵ |
| Uncertainties | pulls (fit − truth)/σ over noisy realisations, with closure phases correlated through shared baselines or independent | 200 draws per case, four scenes including the rim: means within ±0.17, sds 0.87–1.10 ([results](docs/results.md)) |
| Masking (dLux) | calibrated visibilities vs the exact scene | 1e-4 to 8e-4 at a 256-pixel (7.7″) field, falling from 1e-3 at 128 pixels: light lost off the detector. dLux and the closed-form imager agree to 1e-5 |
| Visibility-only data | V²-only files read and fitted (virgil#158): diameters against the truth, PMOIRED and 200-draw pulls | 1e-6 noise-free; 2e-4 σ from PMOIRED; pulls sd 1.01 |
| Grids and limits | `likelihood_grid`, `nsigma`, best flux and its error per position, `absil_limits`, `ruffio_upperlimit` against our own chi-squared, Absil et al. 2011 and Ruffio et al. 2018 (SciPy, mpmath) | 5e-12 to 5e-7; flux pulls sd 1.09 |
| Image reconstruction | `fit` with `TSV` against eht-imaging's imager on the same simulated visibilities (amplitudes + closure phases), objectives matched | minima agree: loss 5e-8, image 1e-7 |
| Image model, closure-phase term | `Image.model` against eht-imaging's transform (opposite Fourier sign, same orientation); closure-phase χ² against `chisq_cphase` | 1e-13; 1e-10 |
| Image regularisers | `TSV`, `TV`, `MaxEntropy` against eht-imaging and MPoL; `Laplacian`, `StarletL1`, `LogSum` against SciPy; values and gradients | 1e-15 |
| Correlated closure phases | `OIData.cp_noise` and the correlated χ² against fouriever (Kammerer et al. 2020's own code) and our own implementation of that model ([notes](docs/fouriever_notes.md)) | correlation exact; χ² 1e-15 against each code's definition |
| Linear marginals | calibration gains (`with_gains`), closure-phase offsets (`with_closure_offsets`) and RV zero points against dense Gaussians; `linear_flux_grid` flux, error and Gaussian-prior evidence against Gauss–Newton and brute-force integration | 1e-11 to 1e-13; 1e-9 (flux map, linear model) |
| Against CANDID | χ², `nsigma`, χ² maps, Absil limits and fits on the same files, CANDID in its own environment ([notes](docs/candid_notes.md)) | 2e-7 on V²-only files; with closure phases, the chord/plain residual difference (≤ 8e-4 on maps); fits 4e-4 σ apart |
| Spectra, flared disks, harmonix wrapper | `PowerLaw`, `BlackBody`, `Tabulated`, chromatic `System`s; `FlaredDiskHG`/`Gaussian`/`PowerLaw` against a direct sum of the documented brightness (Blakely et al. 2024); `HarmonixModel` units and weight | 1e-15 (disks), 1e-12 (spectra; black body 9e-9) |
| Fits against PMOIRED | the same files fitted by both; best fits, uncertainties, 200-draw pulls | best fits < 0.25 σ apart; errors equal with 3 telescopes; PMOIRED's errors ~10 % small with correlated closure phases (it treats them as independent) |
| Masking (dLux) | virgil fits to noise-free dLux observables | bias ≤ 0.05 σ for realistic errors (1° closure phases; largest for the rim) |
| Dirty image | uniform uv disk → Airy beam centred on the source, East left | 3e-3 (sampling of the disk) |
| Beam | filled uv disk and ellipse → FWHM 2.355/(π q), PA perpendicular to the coverage | 1e-4 |
| Grids | `nyquist_pixel_scale`, `field_of_view`, `convolve_beam` (second moments), `render` orientation and sampling, `Image.from_model`, `Image.model_on_grid` on rotated lattices vs direct DFT | 1e-9 to 1e-12 |

The strategy (trust bootstrapped from mathematics, standards and
human-written packages, never from LLM-written code):
[docs/design.md](docs/design.md). Every check, and whether it is against an analytic result, our own
quadrature, first principles or dLux: [docs/report.md](docs/report.md).
Full numbers and figures: [docs/results.md](docs/results.md), regenerated by
`scripts/report.py` (weekly in CI). Which object is checked against which
root of trust, test by test, on exactly which virgil commit:
[docs/evidence.md](docs/evidence.md), from `pytest --evidence`. Next: [validating against CANDID and
PMOIRED](docs/plan_external.md); PMOIRED's conventions are pinned
([Stage 0](docs/pmoired_conventions.md)). Planned: [binaries in the ESO
archive with published orbits](docs/plan_eso_binaries.md), per epoch and
jointly with virgil's Stage 6a.1 orbits; and [the SPIE imaging contests'
blind datasets](docs/plan_imaging_contests.md) for image reconstruction.

## Findings

Disagreements are kept as tests: one pinned to what virgil does, and a
`strict` xfail stating what is wrong, which will start failing (and so be
noticed) once virgil changes.

| # | virgil | Finding | Severity | Status |
| --- | --- | --- | --- | --- |
| 1 | `ModulatedGaussianRim` | The azimuthal modulation of an inclined rim is defined in the **disk plane** (in-plane azimuth, counted from the major axis at `pa`), not in on-sky position angle as the docstring's "polar image coordinates" formula reads. With that reading virgil agrees to 1e-16; with the literal one the visibilities differ by 4e-2. Documented in [virgil#134](https://github.com/benjaminpope/virgil/pull/134). | docs | fixed, virgil#134 |
| 2 | `GaussianArc` | The arc-length weight is cut at ±3.5σ and renormalised, dropping 0.05 % of the flux and giving ~1e-3 visibility errors that no `nodes` setting removes. Once the length FWHM exceeds about half the circumference the weight wraps round the circle more than once; which definition is meant is undocumented. [virgil#134](https://github.com/benjaminpope/virgil/pull/134) spans ±min(6σ, πR) (a Gaussian wrapped once round the circle); the strict xfail is now a passing test. | minor | fixed, virgil#134 |
| 3 | `OIData.uv_grid` | Documented as set "when the samples lie on a regular lattice", but only AMIGO DISCO records get one, so ordinary OIFITS data on a lattice never use `Image`'s fast lattice transform. The transform itself is exact (1e-12) when given the grid. [virgil#134](https://github.com/benjaminpope/virgil/pull/134) corrects the docstring and documents opting in with `find_uv_grid`. | docs / performance | fixed, virgil#134 |
| 4 | `Image.from_model` | The default `floor=1e-6` adds ~1e-5 of spurious flux over a large field (documented; noted here because it limits round-trip tests). | note | documented |
| 5 | `inference.laplace_cov` | Fails (`TypeError: Cannot concatenate arrays with different numbers of dimensions`) when any parameter path is array-valued, e.g. `ModulatedGaussianRim`'s `az_amps`/`az_pas` beside scalar ones, so a rim fit has no Laplace errors. | bug | fixed, virgil#135 |
| 6 | `fitting.fit` | Started exactly at a zero-residual optimum (noise-free data, truth as the start), LM runs to `max_steps` and L-BFGS stops after one step, both reporting non-convergence with a warning; any noise or offset start converges in a few steps. | minor | fixed, virgil#144 |
| 7 | `fitting.fit` | The documented default (LM whenever the objective is least squares) depends on how a prior is written: `Uniform(0, 1).expand([1])` or `.to_event(1)` silently selects L-BFGS, while the equivalent array-shaped `Uniform` gets LM. In rim pull tests one of 120 L-BFGS fits then failed to converge in 20000 steps. | bug | fixed, virgil#142 |
| 8 | `oidata.OIData` | Data with every closure phase flagged (e.g. an OIFITS file whose OI_T3 FLAG is all set) crash inside the closure-phase whitening with `ValueError: zero-size array to reduction operation maximum`, instead of the clear "no phase data" error virgil gives for files without OI_T3. | bug | fixed, virgil#155; visibility-only since virgil#158 |
| 9 | `spectra.reference_flux` | For `Tabulated` it returned every node rather than the documented reference flux (the node mean, which `spectrum(None)` returns). Harmless inside virgil, which used it only to check fluxes are non-negative. | minor | fixed, virgil#163 |
| 10 | `oidata.OIData` | The 2006 imaging-contest files (simulated AMBER, OIFITS v1) keep closure phases under a different `INSNAME` (`AMBER-LR_TR01_OB01`) from their V² (`AMBER-LR_OB01`), with identical wavelength tables, and store some baselines reversed relative to the triangles' legs. The standard allows both; virgil raised `ValueError` ("needs baseline (0, 1), which is not in the visibility table with the same wavelengths", which was untrue). | bug | fixed, [virgil#167](https://github.com/benjaminpope/virgil/pull/167) |
| 11 | `likelihood.whitened_residuals` | For closure phases from four or more telescopes, residuals are wrapped into [-π, π), taken as chords 2 sin(Δ/2) and whitened together. A chord changes sign under Δ → Δ + 2π, so the correlated cross terms make χ² **jump** wherever a residual crosses ±π: in a smooth one-parameter sweep the largest step is 860× the median (48× with three telescopes). Image fits on high-S/N 4T data (2022 GRAVITY, 2010 AMBER contest data) stall on these jumps, where JAX gradients and finite differences disagree by 10²–10⁴. | bug | fixed, [virgil#174](https://github.com/benjaminpope/virgil/pull/174) |
| 12 | `imaging.clean` | Without a base scene, CLEAN seeds the central pixel, whose \|J e_p\| is zero. On even grids that pixel sits half a pixel off the origin, so rounding left \|J e_p\|² ~ 1e-24. Its score won, and its step (−g over a zero curvature) was infinite, giving NaN χ²: every even size from 34 to 68 at 0.4 mas on simulated 4T data. Exposed by virgil#174, which only changed the rounding. | bug | fixed, [virgil#190](https://github.com/benjaminpope/virgil/pull/190) |
| 13 | `oidata.OIData` | Closure-phase frames are grouped by `MJD` alone. OIFITS v1 files that give a night one MJD and tell snapshots apart by `TIME` (the 2004 imaging-contest files, from OYSTER) are merged into one frame, so closure phases from different snapshots are treated as correlated through "shared" baselines: the 2004 data2 file's 130 closure phases whiten to 10 independent combinations instead of 130; on our simulated 4T file, 3 instead of 21. | bug | open |

Differences with other packages (all in the [ledger](docs/trust.md#ledger)):

| # | With | Difference | Ruling |
| --- | --- | --- | --- |
| D2 | PMOIRED | Modulated rims: PMOIRED modulates the profile, virgil blurs a modulated ring | definition |
| D3 | PMOIRED | PMOIRED treats closure phases as independent; virgil whitens them as a correlated group (virgil's pulls calibrated, PMOIRED's errors ~10 % small) | definition |
| D4 | PMOIRED, CANDID | Both scale fit uncertainties by √χ²_r | definition |
| D5 | CANDID, fouriever | Closure-phase residual: plain difference in CANDID and fouriever; in virgil the chord 2 sin(Δ/2) for independent closure phases, and sin Δ plus a periodic penalty (1 − cos Δ)/σ for correlated ones (since virgil#174). All agree to O(Δ³); identical on V²-only data | definition |
| D6 | fouriever | With unequal closure-phase errors within a group, different generalised inverses of the singular covariance (virgil's is the better calibrated) | definition |
| P1–P3 | PMOIRED | NaN models with `auto`, ring sampling not the documented Nr, a ~1e-4 ring precision floor ([notes](docs/pmoired_notes.md)) | to raise |
| P4 | CANDID | Absil limits from `detectionLimit` ~1 % low against its own criterion solved exactly ([notes](docs/candid_notes.md)) | to raise |
| P5 | fouriever | Closure-phase residual not wrapped: data straddling ±180° give a ~2π residual at the true binary ([notes](docs/fouriever_notes.md)) | raised, [fouriever#26](https://github.com/kammerje/fouriever/issues/26) |
| P6 | eht-imaging | Under NumPy 2, closure phases fail when every time has the same baselines (all simulated data); worked around in our worker | to raise |

Nothing else disagreed: every primitive, convention (East, North, position
angle, OIFITS sign, T3 orientation), the OIFITS reader, the fitter, the
Laplace uncertainties, the grid search and the detection limits passed.

## Requesting other validations

If you would like something else validated, please
[open an Issue](https://github.com/benjaminpope/virgil-validation/issues)
detailing the request: the model or function, the analytic or independent
result it should match, and the precision you expect.

Not yet covered: `GravityDarkenedStar` against an independent root,
harmonix's maps, bandwidth smearing, AMIGO DISCO mode bases, regularised
image reconstructions end to end, sampling, and
false-alarm and contrast-limit simulation campaigns.

## Running

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[external]"   # virgil-astro from PyPI, PMOIRED
bash scripts/setup_candid.sh                         # CANDID, in its own .venv-candid
bash scripts/setup_external.sh                       # eht-imaging, MPoL, fouriever, each in its own .venv-<name>
.venv/bin/python -m pytest -m "not slow"             # ~1 min
.venv/bin/python -m pytest                           # + dLux and noisy pulls, ~10 min
.venv/bin/python scripts/report.py                   # docs/results.md and figures
```

To test a local virgil checkout, install it editable instead
(`uv pip install --python .venv/bin/python -e ../virgil -e .`). CI runs the
fast tests on every push and everything weekly against virgil's `main`.

## References

* Pauls, Young, Cotton & Monnier 2005, PASP 117, 1255 (OIFITS)
* Duvert, Young & Hummel 2017, A&A 597, A8 (OIFITS v2)
* Thompson, Moran & Swenson 2017, *Interferometry and Synthesis in Radio
  Astronomy*, 3rd ed., ch. 4
* Absil et al. 2011, A&A 535, A68; Gallenne et al. 2015, A&A 579, A68 (CANDID)
* Chael et al. 2016, ApJ 829, 11; Chael et al. 2018, ApJ 857, 23 (eht-imaging)
* Zawadzki et al. 2023, PASP 135, 064503 (MPoL)
* Kammerer et al. 2020, A&A 644, A110 (fouriever; correlated closure phases)
* Starck, Murtagh & Fadili 2010, *Sparse Image and Signal Processing* (starlets)
* Berger & Segransan 2007, New Astron. Rev. 51, 576
* Hanbury Brown, Davis, Lake & Thompson 1974, MNRAS 167, 475 (linear
  limb darkening)
* Kipping 2013, MNRAS 435, 2152 (q₁, q₂ for two-parameter laws)
* Desdoigts, Pope, Dennis & Tuthill 2023, JATIS 9, 028007 (dLux)
