# PMOIRED conventions (Stage 0)

[PMOIRED](https://github.com/amerand/PMOIRED) 26.10.1 (BSD-2-Clause,
`pip install -e ".[external]"`). Conventions were read from its
documentation (the bundled notebook *Model definitions and examples*), then
**pinned** by `tests/test_pmoired_conventions.py`: each test writes a
noise-free OIFITS file from our references (`crosscheck.sky`), has PMOIRED
read it and evaluate its model at the parameters we think correspond, and
requires a match to rounding error. Each test also requires the deliberately
wrong mapping to fail, so a match can't be an accident of symmetry. The
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
| `ModulatedGaussianRim(diam, fwhm, inc, pa, az_amps, az_pas)` | no exact equivalent: PMOIRED's thin-ring limit is `diamin` → `diamout`, and its `spatial kernel` blurs every component. Modulation maps as `az ampN` = `az_amps[N-1]`, `az projangN` = `az_pas[N-1]` − `pa`: virgil's angles are absolute, PMOIRED's relative. |

Both packages put the modulation in the disk plane. That is virgil's
behaviour (finding 1) and PMOIRED's, and neither documented it before this
check.

## Observations to raise (not yet reported upstream)

* `setupFit` with the default `auto=True` sets `'wl kernel': 0.0` for our
  files (four channels with a nominal 1 nm `EFF_BAND`), and every model
  observable is then NaN. `auto=False` avoids it. A real instrument file
  would rarely look like this, but NaN models with no warning are worth
  reporting.
* Rings with the default radial sampling disagree with the analytic annulus
  at 4e-5, but at 1e-5 with an explicit `'Nr': 100`, which the docstring
  gives as the default.

Neither affects the conventions. Both are to be reported to PMOIRED only
with Ben's approval (see the plan's Stage 5).

## Next (Stage 1)

Evaluate virgil and PMOIRED on the same files at mapped parameters, shape
by shape, using the table above; for the rim, compare each package with our
quadrature of its own definition.
