# PMOIRED conventions (Stage 0)

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

## Pinned conventions

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

## Mapping to virgil (for Stage 1)

| virgil | PMOIRED |
| --- | --- |
| `PointSource(flux, dra, ddec)` | `{'ud': 0, 'f': flux, 'x': dra, 'y': ddec}` |
| `UniformDisk(diam)` | `{'ud': diam}` |
| `GaussianDisk(sigma)` | `{'fwhm': 2.3548 * sigma}` |
| `EllipticalGaussian(fwhm, ratio, pa)` | `{'fwhm': fwhm, 'incl': degrees(arccos(ratio)), 'projang': pa}` |
| `Resolved(flux)` | `{'f': flux}` with no size key |
| `System` weights | per-component `f` (both are total fluxes) |
| `QuadraticLimbDarkenedDisk(diam, q1, q2)`, `SquareRootLimbDarkenedDisk(diam, q1, q2)`, `LimbDarkenedDisk(diam, u)` | `{'diamin': 0, 'diamout': diam, 'profile': '1 - u1*(1-$MU) - u2*(1-$MU)**2'}` with u1, u2 (or c, d) from Kipping's maps; set `Nr` in `setupFit` (10000 for 1e-6) |
| `ModulatedGaussianRim(diam, fwhm, inc, pa, az_amps, az_pas)` (blur isotropic in the rim plane since virgil#139) | unmodulated: exactly a ring with the blurred-ring radial profile `exp(-(R² + r0²)/2σ²) I0(R r0/σ²)` over `diamin`/`diamout` = `diam` ∓ 12σ, with `incl`, `projang` (agree to 1e-9). Modulated: PMOIRED's modulation multiplies the radial profile, while virgil blurs the modulated ring (harmonic m gets I_m in place of I0), a **difference of definition** of order (mσ/r0)²; each matches our quadrature of its own definition. Angles map as `az projangN` = `az_pas[N-1]` − `pa` (virgil absolute, PMOIRED relative). `spatial kernel` blurs isotropically on the sky, so it no longer matches. |

Both packages put the modulation in the disk plane. That is virgil's
behaviour (finding 1) and PMOIRED's, and neither documented it before this
check.

## Fits (Stage 2)

| Quantity | PMOIRED | virgil | How they relate |
| --- | --- | --- | --- |
| reported uncertainties | `bestfit["uncer"]`, multiplied by √(reduced χ²) (`"normalized uncertainties": True`) | Laplace covariance, the plain curvature | divide PMOIRED's by √χ²_red (`external_bridge.pmoired_models.fit` does) |
| closure-phase errors | each closure phase independent, with its own σ | the closure phases of a snapshot whitened as a correlated group (baseline-phase noise) | equal with 3 telescopes; with 4, PMOIRED's errors are 4–6 % smaller and, for noise from baseline phases, ~10 % too small in pulls. PMOIRED's `oicorr` module may model the correlations; not yet explored |

## Problems found

Behaviour that looks wrong on PMOIRED's side is listed, with reproducers, in
[pmoired_notes.md](pmoired_notes.md), to be raised upstream in a batch.

## Stage 1

Done: see `tests/test_pmoired_vs_virgil.py` and the Stage 1 section of
[report.md](report.md).
