# PMOIRED

## Conventions

[PMOIRED](https://github.com/amerand/PMOIRED) 26.10.1 (BSD-2-Clause,
`pip install -e ".[external]"`). Conventions were read from its
documentation (the bundled notebook *Model definitions and examples*), then
**pinned** by `tests/test_pmoired_conventions.py`: each test writes a
noise-free OIFITS file from our references (`crosscheck.sky`), has PMOIRED
read it and evaluate its model at the parameters we think correspond, and
requires a match. Analytic shapes must match to rounding error; rings, which
PMOIRED computes from a sampled radial profile, to tolerances set by its
`Nr` (see the table). The ambiguous mappings (position axes and
closure-phase sign, which axis `projang` is, what a component with no size
is, and the modulation angle) also have negative controls: the plausible
wrong mapping must fail. The others (`ud`, `fwhm`, ring forms,
`spatial kernel`) are positive checks only. The
code calling PMOIRED (`src/external_bridge/`) cannot import virgil.

### Pinned conventions

| Quantity | PMOIRED | Documented? | Pinned by | Agreement |
| --- | --- | --- | --- | --- |
| position | `x` East, `y` North, mas | yes | binary with closure phases up to 33°; the mirrored truth fails by 53° | 1e-15 (V²), 6e-14° (CP) |
| closure-phase sign, triangle orientation | T3 = V(ab) V(bc) V*(ac), as OIFITS and ours | via the file's `formula` | same | same |
| flux | `f`: the component's total flux | yes | binary flux ratio 0.2 | same |
| point source | `{'ud': 0}`; a component with **no** size key is fully resolved (V = 0) | yes (resolved); `ud: 0` for a point is our choice | `test_component_without_size_is_resolved` | — |
| uniform disk | `ud`: diameter, mas | yes | offset 2 mas disk | 5e-13 |
| Gaussian | `fwhm`, mas | yes | | 4e-16 |
| inclination, projection angle | image compressed by cos(`incl`) perpendicular to `projang`; `projang` is the major axis, N (0) to E (90) | yes | elliptical Gaussian; the 90°-rotated mapping fails | 4e-16 |
| ring | `diamin`, `diamout`: uniform annulus | yes | difference of two Airy disks | 1e-5 at `Nr` = 100, falling as Nr⁻² to 1e-8 at 3000 (sampled radial profile) |
| limb-darkened disk | ring with `diamin` 0, `diamout` = the limb-darkened diameter, `profile` an expression in `$MU` = √(1 − (2r/`diamout`)²) | yes | quadratic and square-root laws, off-centre, with a companion (`test_limb_darkened_star_and_companion`) | 3e-7 (quadratic) and 2.5e-6 (square root) in V² at `Nr` = 10000, converging as Nr⁻¹·⁵ (sampled radial profile; the square root's slope is singular at the limb) |
| ring, other form | `diam`, `thick`: annulus from `diam` (1 − `thick`) to `diam` | yes | annulus 3–4 mas | 1e-7 at `Nr` = 1000 |
| blur | `spatial kernel`: Gaussian FWHM, mas | yes | annulus × Gaussian transform | 1e-7 at `Nr` = 1000 |
| azimuthal modulation | 1 + Σ `az ampN` cos(N(θ − φN)), θ the **in-plane** azimuth, φN = `projang` + `az projangN` (**relative** to `projang`) | relative: yes; in-plane: **no** | all four readings (in-plane/sky × relative/absolute); only in-plane + relative matches, the others miss by 12–65° in closure phase | 1e-8 (V²), 2e-6° (CP) at `Nr` = 3000; m = 1 and 2 |

### Mapping to virgil (for Stage 1)

| virgil | PMOIRED |
| --- | --- |
| `PointSource(flux, dra, ddec)` | `{'ud': 0, 'f': flux, 'x': dra, 'y': ddec}` |
| `UniformDisk(diam)` | `{'ud': diam}` |
| `GaussianDisk(sigma)` | `{'fwhm': 2.3548 * sigma}` |
| `EllipticalGaussian(fwhm, ratio, pa)` | `{'fwhm': fwhm, 'incl': degrees(arccos(ratio)), 'projang': pa}` |
| `Resolved(flux)` | `{'f': flux}` with no size key |
| `System` weights | per-component `f` (both are total fluxes) |
| `QuadraticLimbDarkenedDisk(diam, q1, q2)` | `{'diamin': 0, 'diamout': diam, 'profile': '1 - u1*(1-$MU) - u2*(1-$MU)**2'}`, with u1 = 2√q1 q2, u2 = √q1 (1 − 2q2) (the model's `u1`, `u2`) |
| `SquareRootLimbDarkenedDisk(diam, q1, q2)` | `{'diamin': 0, 'diamout': diam, 'profile': '1 - c*(1-$MU) - d*(1-np.sqrt($MU))'}`, with c = √q1 (1 − 2q2), d = 2√q1 q2 (the model's `c`, `d`) |
| `LimbDarkenedDisk(diam, u)` | `{'diamin': 0, 'diamout': diam, 'profile': '1 - u[0]*(1-$MU) - u[1]*(1-$MU)**2 - …'}`, one term `u[n-1]*(1-$MU)**n` per coefficient (not tested against PMOIRED; the two above are). For all three, set `Nr` in `setupFit` (10000 for ~1e-6 in V²) |
| `ModulatedGaussianRim(diam, fwhm, inc, pa, az_amps, az_pas)` (blur isotropic in the rim plane since virgil#139) | unmodulated: exactly a ring with the blurred-ring radial profile `exp(-(R² + r0²)/2σ²) I0(R r0/σ²)` over `diamin`/`diamout` = `diam` ∓ 12σ, with `incl`, `projang` (agree to 1e-9). Modulated: PMOIRED's modulation multiplies the radial profile, while virgil blurs the modulated ring (harmonic m gets I_m in place of I0), a **difference of definition** of order (mσ/r0)²; each matches our quadrature of its own definition. Angles map as `az projangN` = `az_pas[N-1]` − `pa` (virgil absolute, PMOIRED relative). `spatial kernel` blurs isotropically on the sky, so it no longer matches. |

Both packages put the modulation in the disk plane. That is virgil's
behaviour (finding 1) and PMOIRED's, and neither documented it before this
check.

### Fits (Stage 2)

| Quantity | PMOIRED | virgil | How they relate |
| --- | --- | --- | --- |
| reported uncertainties | `bestfit["uncer"]`, multiplied by √(reduced χ²) (`"normalized uncertainties": True`) | Laplace covariance, the plain curvature | divide PMOIRED's by √χ²_red (`external_bridge.pmoired_models.fit` does) |
| closure-phase errors | each closure phase independent, with its own σ | the closure phases of a snapshot whitened as a correlated group (baseline-phase noise) | equal with 3 telescopes; with 4, PMOIRED's errors are 4–6 % smaller and, for noise from baseline phases, ~10 % too small in pulls. PMOIRED's `oicorr` module may model the correlations; not yet explored |

### Problems found

Behaviour that looks wrong on PMOIRED's side is listed, with reproducers, in
[pmoired_notes.md](pmoired.md), to be raised upstream in a batch.

### Stage 1

Done: see `tests/test_pmoired_vs_virgil.py` and the Stage 1 section of
[report.md](index.md).

## Problems to raise upstream

A running list of behaviour in [PMOIRED](https://github.com/amerand/PMOIRED)
that our checks suggest is a problem on PMOIRED's side, not ours. Each entry
has a reproducer in `tests/test_pmoired_problems.py`, written as a strict
`xfail` of the behaviour we expect, so the suite fails (and this page must
be updated) if PMOIRED changes. They will be raised as Issues on
PMOIRED once there are enough of them, and only with Ben's approval.

| # | Version | Behaviour | Evidence | Severity |
| --- | --- | --- | --- | --- |
| P1 | 26.10.1 | `OI.setupFit` with the default `auto=True` sets `'wl kernel': 0.0` on our files (a few channels with a nominal 1 nm `EFF_BAND`), and every model observable is then NaN, with no warning. `auto=False` avoids it. | `test_p1_default_setup_gives_finite_models` | medium: silent NaNs |
| P2 | 26.10.1 | Rings with the default radial sampling differ from the analytic annulus by 4e-5 in V², but by 1e-5 with an explicit `'Nr': 100`, the documented default. | `test_p2_default_nr_is_100` | low: docs |
| P3 | 26.10.1 | Ring visibilities are exact (converging as Nr⁻² to 1e-9) when a baseline has at most 30 samples (epochs × channels), but stop at a floor of ~1e-4 in V² (0.1° in closure phase for a modulated ring) once it has more, whatever `Nr` is. It looks like a speed approximation (e.g. interpolation) that switches on silently. Point, disk and Gaussian components are unaffected. | `test_p3_rings_exact_up_to_30_samples` (control) and `test_p3_rings_exact_beyond_30_samples`: 1 epoch × 30 channels: 1.1e-8; 1 × 35: 7.4e-5; 7 × 6: 1.4e-4 (2–4 mas annulus, 4 UTs, `Nr` 1000–8000) | medium: precision |

## PMOIRED conventions (Stage 0)

PMOIRED 26.10.1 evaluates its models on our noise-free OIFITS files
(`tests/test_pmoired_conventions.py`; details in
[pmoired_conventions.md](pmoired.md)). These checks pin
PMOIRED's conventions against our references; virgil enters at Stage 1.

| Check | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `x` East, `y` North; closure-phase sign; flux `f` (binary, CP up to 33°; mirrored truth fails) | sum of shifted points | Analytic + PMOIRED | 1e-15 (V²), 6e-14° |
| `ud`, `fwhm` | `2 J1(x)/x`, Gaussian | Analytic + PMOIRED | 1e-13 |
| `incl`, `projang` (major axis; rotated mapping fails) | elliptical Gaussian | Analytic + PMOIRED | 4e-16 |
| `diamin`/`diamout`, `diam`/`thick` rings | two-Airy annulus | Analytic + PMOIRED | Nr⁻²: 1e-5 at Nr = 100, 1e-8 at 3000 |
| `spatial kernel` | Gaussian blur transform | Analytic + PMOIRED | 1e-7 |
| `az ampN`, `az projangN`: in-plane azimuth, relative to `projang` (3 other readings fail) | annulus quadrature | Quadrature + PMOIRED | 1e-8 (V²), 2e-6° (CP) |


## virgil vs PMOIRED (Stage 1)

PMOIRED evaluates each scene on one of our files and reports the uv
coordinates of every sample and triangle; virgil is evaluated at exactly
those coordinates (`tests/test_pmoired_vs_virgil.py`). The carrier file has
30 samples per baseline, the most for which PMOIRED's rings are exact
(problem P3 in [pmoired_notes.md](pmoired.md)).

| virgil | PMOIRED | Tag | Agreement |
| --- | --- | --- | --- |
| `BinaryModelCartesian`, `BinaryModelAngular` | two `ud: 0` components | PMOIRED | 1e-12 (V²), 1e-9° |
| `System(UniformDisk, PointSource)` | `ud` + point | PMOIRED | same |
| `System(PointSource, EllipticalGaussian)` | `fwhm`, `incl` = arccos(ratio), `projang` | PMOIRED | same |
| `Resolved` in a `System` | component with no size | PMOIRED | 1e-12 |
| binary at masking scales (7 holes, 150 mas) | same | PMOIRED | 1e-12, 1e-9° |
| 3 random constellations of 24 points, Gaussians, elliptical Gaussians and disks | one dictionary of 24 components | PMOIRED | 1e-12, 1e-8° |
| `QuadraticLimbDarkenedDisk`, `SquareRootLimbDarkenedDisk`, off-centre, with a companion | ring from 0 to `diamout` with `profile` in `$MU` | PMOIRED | V² 3e-7 (quadratic) and 2.5e-6 (square root), closure phase 6e-4° and 4.5e-3°, at `Nr` = 10000; the difference falls as Nr⁻¹·⁵ (PMOIRED's sampled profile) |
| `ModulatedGaussianRim`, inclined, unmodulated | ring with the blurred-ring radial profile exp(−(R² + r0²)/2σ²) I0(R r0/σ²), `incl`, `projang` | PMOIRED | 1e-9: confirms virgil#139's in-plane blur |
| `ModulatedGaussianRim`, inclined, m = 1, 2 | same profile with `az ampN`, `az projangN` = `az_pas` − `pa` | PMOIRED + Quadrature | a difference of definition, of order (mσ/r0)² (PMOIRED modulates the profile; virgil blurs the modulated ring); PMOIRED matches our quadrature of its own definition to 1e-8 |


## fits with virgil and PMOIRED (Stage 2)

Both packages fit the same simulated files (`tests/test_pmoired_fits.py`;
binary, disk star + companion and star + elliptical envelope; 30 samples
per baseline). PMOIRED's results are mapped into virgil's parameters
(e.g. axis ratio = cos(incl)), with the covariance carried through the
Jacobian of that mapping.

| Check | Tag | Result |
| --- | --- | --- |
| noise-free fits reach the injected truth | PMOIRED | virgil 1e-6, PMOIRED 1e-4 relative |
| best fits on noisy data, both closure-phase noise models | PMOIRED | agree to < 0.25 σ (typically < 0.1 σ) |
| uncertainties with 3 telescopes (one closure phase per snapshot), after undoing PMOIRED's √(reduced χ²) normalisation | PMOIRED | sigmas agree to < 4 % (0.5–3 % measured); parameter correlations to < 0.05 (≤ 0.011 measured) |
| uncertainties with 4 telescopes | PMOIRED | PMOIRED's are 4–6 % smaller: it counts 4 closure phases per snapshot as independent, where virgil whitens them as 3 correlated ones (a difference of definition) |
| pulls over 200 shared realisations, closure-phase noise from baseline phases (realistic) | PMOIRED + Statistics | virgil sd 1.04 (calibrated, within 1 ± 0.15); PMOIRED sd 1.10 (errors ~10 % small; accepted range 0.85–1.40) |
| the same with independent noise per triangle | PMOIRED + Statistics | PMOIRED sd 0.98 (calibrated, within 1 ± 0.15); virgil sd 0.95 (conservative) |

