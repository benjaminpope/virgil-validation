# PolarDisk: radial profiles, harmonics and height

`PolarDisk` (virgil P1-P4: radial profiles `GaussianRing`, `RadialPowerLaw`, `ExpTail`, `Gap`,
`RadialSum`, `RadialGP`; azimuthal `Harmonics` in linear and positive forms and `VonMises` arcs;
height `Cone` and `PowerLawHeight`) is checked here in four stages. The code under test is virgil
`e42973d7478e04d704aaee53e61810e0356c6e49` (the tip of `polar-p4`, PR #343, which includes `polar-p3` c82e18c and `main`). **The committed result files were produced at the older pin `791f9fef65eb49ebc387c5c365a13c7af2342341` and are stale: regenerate them at the new pin on OzSTAR** (they still record the older hash, and were not relabelled). Every result file in
[`benchmarks/polar_disk/results`](https://github.com/benjaminpope/virgil-validation/tree/main/benchmarks/polar_disk/results)
records that hash and the versions of the packages it ran with (frank 1.2.3, vis-r at
`f53467077b04878986c967b501d88c54ff3efb86`, astro-frap 1.0.2, PMOIRED 26.10.1, oimodeler 0.8.1, JAX 0.11.2).
The pass criteria are written at the head of each script before it was run. Results are from the working
tree on top of this repository's `main` at the time of the PR: re-run on merge to pin them to a commit.

| Stage | What | Script | Outcome |
| --- | --- | --- | --- |
| V1 | forward transforms against frank, vis-r, FRAP and our own Hankel transform; speed | `v1_forward.py` | smooth profiles pass (1e-9); `ExpTail` does not (F19) |
| V2 | brute-force renders; PMOIRED and oimodeler rings; old against new `TruncatedCone` | `v2_brute_force.py`, `v2_external.py`, `v2_cone_old_new.py` | all pass except oimodeler's `ExpRing` (P9) |
| V3 | public data: AS 209 (frank), HR 4796 (vis-r), HD 169142 (FRAP) | `v3_as209.py`, `v3_hr4796.py`, `v3_frap.py` | pass, except the stated forward-model tolerances in two scripts |
| V4 | injection and recovery on NRM and VLTI coverage | `v4_injection_recovery.py` | laptop configuration run; the full run is for OzSTAR |

## Independence

Reference values come from other codes or from code written in this repository, never from virgil.
`src/crosscheck/radial.py` is a composite Gauss-Legendre Hankel transform (checked against
`scipy.integrate.quad`); `src/crosscheck/polar_images.py` renders a scene finely in the plane of the disk
(radius, azimuth, height) and carries it to the sky, and `crosscheck.sky` sums it point by point;
`src/external_bridge/radial_codes.py` calls frank, vis-r and FRAP as documented.
Tests: [`test_polar_disk_brute_force.py`](https://github.com/benjaminpope/virgil-validation/blob/main/tests/test_polar_disk_brute_force.py),
[`test_polar_disk_vs_codes.py`](https://github.com/benjaminpope/virgil-validation/blob/main/tests/test_polar_disk_vs_codes.py).

## V1 Forward transforms

Criteria A1-A4 (smooth profiles to 1e-6 against truth, frank and FRAP; vis-r to within its own
discretization error; `ExpTail` to its documented 1e-5) and the speed report are in the script. Inclination
50 degrees, position angle 30 degrees, an ALMA-like scale (r0 = 1 arcsec, q to 8e5) where vis-r's defaults make sense,
and an optical/IR scale (r0 = 10 mas, q to 8e7) for virgil, frank and FRAP (the transform depends on q r only).

| Scene | virgil against truth | frank / FRAP against truth | vis-r against truth |
| --- | --- | --- | --- |
| Gaussian ring | 1.3e-9 | 1e-15 | 5.3e-4 |
| error-function edged power law | 6.7e-12 | 3e-16 | 5.2e-3 |
| gap on a plateau | 2.8e-12 | 3e-16 | 3.0e-3 |
| ring with vis-r's scale-height factor | 1.3e-9 | n/a | 5.3e-4 |
| `ExpTail` | **2.0e-4** | **4.9e-3** | n/a |

virgil follows vis-r to within vis-r's own error everywhere (it differs from vis-r by exactly vis-r's
error against truth to 6 digits): vis-r interpolates linearly in q on a grid fixed by the field size.
**`ExpTail` fails A1.** Its docstring promises 1e-5 with about 16 * 2 pi q L Gauss-Laguerre nodes (117 here);
128 nodes give 2e-4, the 150-node cap 4e-5 (ledger **F19**). frank and FRAP are 5e-3 off on the
same profile because a sharp-edged tail is not band-limited by their 300 nodes: a limit of those
codes rather than an error (ledger **D10**); my criterion A2, which presumed a smooth profile, was
wrong for this scene and is reported as such, not rewritten.
`n_r` convergence of the smooth scenes: the error is at its floor by 32 to 64 nodes.

Speed (CPU, Apple M-series laptop, float64, median of 15):

| N_vis | virgil, jitted | virgil, `freeze_geometry` | frank, matrix cached | FRAP, matrix cached | vis-r, cached | frank / FRAP, matrix built per call |
| --- | --- | --- | --- | --- | --- | --- |
| 1e3 | 0.63 ms | 0.14 ms | 0.13 ms | 0.21 ms | 0.13 ms | 60 / 60 ms |
| 1e4 | 7.1 ms | 0.50 ms | 1.9 ms | 1.0 ms | 0.83 ms | 650 / 470 ms |

virgil's per-call cost includes the Bessel matrix; with the geometry frozen it matches the other codes'
cached-matrix cost. GPU and N_vis = 1e5 timings are deferred to OzSTAR.

## V2 Non-axisymmetric and geometric models

Against the brute-force renders, 300 baselines to 130 m at 2.2 um, every case passes (criterion 1e-6; the
imaginary part is above 0.2 everywhere, so each is sensitive to the asymmetry, and the control with the
azimuth or height sense reversed differs by more than 0.2):

| Case | max abs dV |
| --- | --- |
| linear harmonics M = 3 on a thin ring, offset | 8e-16 |
| positive harmonics M = 3, Gaussian ring | 2e-9 |
| positive M = 1 | 9e-16 |
| `VonMises` arc, blurred (degree 29) | 1.2e-7 |
| `Cone` height | 2e-9 |
| `PowerLawHeight` with harmonics | 1.4e-9 |
| chromatic `System` (spectral index) at three wavelengths | 7e-10 |

PMOIRED (rings with `az ampN`/`az projangN`, M = 1 to 3, and a power-law profile; same OIFITS file, virgil
evaluated at PMOIRED's coordinates): max |dV2| 1e-9 to 1.3e-8, |d closure phase| below 1.7e-4 degrees, closure
phases up to 178 degrees. oimodeler 0.8.1 (cycles per radian; `elong` = 1/cos(inc)): `EIRing` and `Ring` to 1e-15,
`ESKRing` to 1e-15 for three skews with the mapping derived in the script (swapping c and s differs by 0.2 to 0.34),
**`ExpRing` fails**: it is 8.3e-3 (dim = 512) and 8.4e-3 (dim = 2048) from the exact transform of its own documented
profile, while virgil's `ExpTail` with the same cut agrees with the exact transform to 1e-14 (ledger **P9**, unexplained;
not raised upstream).

Old against new `TruncatedCone`, with the old form run in a separate environment at virgil `a1dae85584b78d19771e42a19a6146ff74414f2f`
(0.3.0): the old midpoint form converges as 1/n_rings^2 (error ratios 4.0 per doubling; 1.4e-6 at 768 rings against the
n = 8000 value), the new Gauss-Legendre form is at 1.3e-8 from 24 rings (the 1.3e-8 is both forms dropping the tail
beyond five lengths); criteria C1 to C3 pass. Per-call times are in the result file (they are noisy at this size: a few ms).

## V3 Public data

virgil has no loader for complex visibilities and none was added (design/radial_profiles.md, section 13): what follows
compares forward models and profile shapes on data corrected and averaged here, not posteriors from raw data. Downloaded
files are untrusted: each is in its own directory, pickles are read with a whitelisting unpickler
(`src/external_bridge/safe_load.py`), and `.npz` files with `allow_pickle=False`.

**HR 4796** (vis-r, `data/hr4796.selfcal.npy`, 12.9 MB, 321 322 visibilities, binned to 19 493 at 4 arcsec as in `hello_world.py`).
The reference is the authors' own notebook output for their image-based Gaussian ring, not the numbers of the RASTI paper
(not fetched). Eight parameters (offsets, position angle, inclination, flux, r0, dr, dz), MAP and Laplace errors by least squares:

| Criterion | Result |
| --- | --- |
| H1 forward models at the authors' parameters agree to 1e-3 of the flux | **fails**: max dV 5.4e-5 Jy (3.6e-3 of the flux). Against truth vis-r is 5.4e-5 off (its interpolation) and virgil 1.5e-11 |
| H2 virgil MAP within 1 sigma of vis-r's in every parameter, and abs(dchi2) at most 10 | passes: largest difference 0.33 sigma (the flux), dchi2 = 2.5 |
| H3 both within 3 sigma of the notebook values | passes: largest 0.36 (vis-r) and 0.59 (virgil) sigma |

The offsets: vis-r's (dra, ddec) is a phase rotation of the model by exp(+2 pi i (u dra + v ddec)), which in virgil's sky
convention is the offset (-dra, -ddec); the first run passed them unchanged and recovered the same fit with both offsets
of opposite sign. Raw chi2 per data point on the quoted (re-weighted) errors is 0.98 for vis-r and 0.98 for virgil; the weights are
rescaled so that the null model has chi2 equal to the number of points, so this is not an independent goodness of fit.

**AS 209** (frank `docs/tutorials/AS209_continuum.npz`, 37.4 MB, 977 774 visibilities). With frank's tutorial geometry
and settings (fixed 34.97 degrees, 85.76 degrees, offsets 1.9 and 2.5 mas; Rmax = 1.6 arcsec, N = 250, alpha = 1.05,
weights_smooth = 1e-4; 119 s) the visibilities are phase-centred and deprojected with frank's own `apply_correction`,
binned in radius (2 klambda to 4 Mlambda: 2000 bins) and fitted by `PolarDisk(RadialGP(2 mas, 1.6 arcsec, sigma = 2.5, ell = 30 mas), n_r = 200)` (MAP, free flux; 42 s,
reaching the 3000-iteration limit):

| Criterion | Result |
| --- | --- |
| B1 raw chi2 per bin of the MAP at most 1.5 times frank's | passes: 2.3926 against 2.3914 (ratio 1.0005) |
| B2 every frank ring maximum has one within 5 mas | passes: 1.5 and 4.0 mas (frank 0.113 and 0.228 arcsec); the MAP has one more maximum, at 0.16 arcsec, that frank's smoother profile does not |
| B3 unit-flux profiles differ by at most 10 percent of the peak (0.02 to 0.6 arcsec) | passes: 6.0 percent |

The MAP flux, extrapolated by the profile to zero baseline, is 0.262 Jy against frank's 0.319 Jy: the lowest
bin is at 1 klambda and the data do not constrain the missing short baselines.

**HD 169142** (FRAP `example_data/HD169142_band3.pkl`, 6.7 kB, and `band6.pkl`, 23.6 kB; deprojected real visibilities, channel 0 of each band, single-band I(r), no
radiative-transfer layer). Reference: FRAP's Hankel matrix at its Fourier-Bessel nodes (130 to 1.3 arcsec) with a Gaussian-process MAP written
here (the same prior as `RadialGP`: squared-exponential, sigma = 2, ell = 30 mas, whitened latents, exp link), against `PolarDisk(RadialGP(nu = inf), n_r = 130)`:

| Criterion | Band 3 (121 baselines) | Band 6 (232 baselines) |
| --- | --- | --- |
| F1 FRAP's H applied to virgil's profile at its nodes agrees with virgil's V to 2e-3 | **fails**: 7.7e-3 | **fails**: 5.4e-3 |
| F2 raw chi2 per point of the two MAPs within 5 percent | passes: 2.489 and 2.495 (ratio 0.998) | passes: 3.69 and 3.71 (0.996) |
| F3 unit-flux profiles within 10 percent of the peak | passes: 2.5 percent | passes: 0.9 percent |

The F1 difference is the linear interpolation of virgil's profile onto FRAP's differently placed nodes (a property of the check, not of
either transform: V1 compares the transforms of the same function and agrees to 1e-9).

## V4 Injection and recovery

`v4_injection_recovery.py` draws ring + gap + planet scenes with positive harmonics, observes them on synthetic NRM (7 holes,
4.3 um) or VLTI (four UTs, L band) coverage, fits with `virgil.fit` (LM MAP) and takes the Laplace covariance. It checks the
z-scores of the scalar parameters, the central 68 and 95 percent intervals of the image harmonics (Laplace samples) and
the profile band. The criteria K1 to K4 are in the script. **The laptop configuration** (`--config laptop`: VLTI coverage, M = 2,
n_r = 24, five injections starting at the truth plus a small perturbation; about two minutes) was run:

| Quantity | Result |
| --- | --- |
| scalar coverage of the central 68 / 95 percent | 0.67 / 0.93 (55 pairs) |
| image-harmonic coverage 68 / 95 percent | 0.60 / 0.95 |
| z-score rms (K2: within 0.29 of 1) | **1.9, fails**: the gap depth has abs(z) of 6.9 and 9.6 in two injections (the Laplace covariance is too narrow where the depth sits near its bound of 1 and degenerates with the gap width) |
| K3 converged / K4 MAP loglike no worse than the truth's | 5 of 5; yes (worst +6.4) |

Five injections are a pipeline check, not a calibration: the 68 and 95 percent coverages are within their (wide) binomial tolerances,
and the z-score failure is from one parameter. The full configuration (200 injections, NRM and VLTI, M = 1 to 3, multi-start) is
for OzSTAR: [`benchmarks/polar_disk/README_v4.md`](https://github.com/benjaminpope/virgil-validation/blob/main/benchmarks/polar_disk/README_v4.md).
Fitting array-valued harmonic coefficients failed at 791f9fe, fixed in e42973d (ledger **F20**); the script uses per-element paths.

## Findings from this stage

| # | Object | Finding | Ruling |
| --- | --- | --- | --- |
| F19 | virgil `ExpTail` | the documented 1e-5 at 16 * 2 pi q L nodes is 2e-4 (as found at 791f9fe) | virgil, fixed in e42973d |
| F20 | virgil `Harmonics` | array-valued `re`/`im`/`c`/`s` fail (`fit` and `laplace_cov` paths), as found at 791f9fe | virgil, fixed in e42973d |
| D10 | frank, FRAP | cannot represent a discontinuous profile | definition |
| P9 | oimodeler | `ExpRing` 8e-3 from its documented profile, unexplained | to raise |
| P10 | vis-r | `r_prof_erf2_power_ggap` mixes up depth and radius | to raise |
