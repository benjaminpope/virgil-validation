# Aperture masking

NIRISS-like 7-hole mask (0.8 m circular holes on the 1.32 m hexagonal
grid), 4.8 µm, 30 mas pixels; each scene is a point cloud imaged source by
source (`tests/test_nrm.py`).

| Check | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| mask non-redundancy | baseline vectors | Analytic | pass |
| dLux sky orientation (offset x → column, y → row, centre (n − 1)/2) | calibrated phase of an offset point | dLux | pass (a flip would give O(1) errors) |
| point clouds vs exact scene visibilities | closed forms | Quadrature | ≤ 1e-8 |
| dLux image vs closed-form interferogram (Airy envelope × fringes) | each other | dLux + Analytic | calibrated visibilities agree to 1e-5 |
| calibrated visibilities vs exact: binary, disk star + companion, star + envelope, star + rim | closed forms | dLux / Analytic | 1e-4 to 8e-4 at 256 px (7.7″), from 1e-3 at 128 px: light falling off the detector |
| virgil fits to the noise-free dLux observables, written by our OIFITS writer | injected truth | dLux | bias ≤ 0.05 σ for σ(CP) = 1°, σ(V²) = 0.02 (the rim, the most extended scene, is the largest; the test accepts 0.1 σ) |


## Mask geometry

`virgil.coverage`'s synthetic masking data, against our own mask geometry
([`test_limits_coverage_helpers.py`](https://github.com/benjaminpope/virgil-validation/blob/main/tests/test_limits_coverage_helpers.py)).

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `mask_transfer`, NIRISS holes, 0.82 m | the pupil autocorrelation in closed form: Σ over hole pairs of the overlap of two circles, over N times one hole's area; 1 at zero, 1/7 at every splodge centre | Mathematics | 2e-16 |
| `nrm_oidata`, mask turned by 0° and 37° | V² at every hole-pair baseline (either sign) of the mask turned to `rotation_deg` (up axis North through East), each once; 35 closure triangles, each closing (b₁ + b₂ = b₃), every hole triple once; the errors and wavelength as given | Mathematics | 1e-15 m |
