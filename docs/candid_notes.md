# CANDID: conventions and problems

[CANDID](https://github.com/amerand/CANDID) (A. Mérand and A. Gallenne;
Gallenne et al. 2015) is the binary-search code that virgil's grid search
and Absil limits follow. It is one of the external roots of trust for those
steps (see [design](design.md)). It has no PyPI release and states no
licence, so `scripts/setup_candid.sh` installs it at a pinned commit
(`c255e90`, version 1.1.0) into its own environment, and
`src/external_bridge/candid_bridge.py` runs it in a subprocess with JSON in
and out. CANDID reads our OIFITS files with its own reader.

## Conventions

Pinned by `tests/test_candid.py`.

| Quantity | CANDID | virgil |
| --- | --- | --- |
| companion position | `x` East, `y` North, mas | `dra` East, `ddec` North, mas |
| companion flux | `f`, percent of the primary | `flux`, companion/primary ratio |
| primary | uniform disk, `diam*` (mas), fitted before mapping | any model; here `UniformDisk` |
| Fourier sign | exp(−2πi(ux + vy)/λ) | the same (OIFITS) |
| observables | V², closure phase and T3 amplitude by default | V² and closure phase |
| bandwidth smearing | on, from `EFF_BAND` (3-point top hat or Gaussian) | off |
| χ²_r | mean of squared normalised residuals | χ² / number of data points |
| closure-phase residual | plain difference Δ | chord 2 sin(Δ/2) |
| significance | χ² tail in linear space, two-sided Gaussian | the same, in log space |
| degrees of freedom (Absil) | number of data points | the same |
| fit uncertainties | scaled by √χ²_r, χ²_r = χ²/(N − n_fit + 1) | not scaled |

The bridge switches CANDID to V² and closure phases, zero channel width (no
smearing) and one core. With those settings, the only difference of
definition left is the closure-phase residual (ledger D5). On V²-only files
the two codes agree to the float32 precision of the file.

## Differences that are not errors

* **D5, closure-phase residual.** The plain difference and the chord agree
  to O(Δ³). Near a good fit the two χ² values match to ~1e-6. Far from the
  data, where model closure phases are tens of degrees off, they differ by
  up to 1e-3.
* **Significance saturates.** CANDID takes the χ² tail probability in
  linear space, so a strong detection (above ~8σ) comes out as `inf`.
  virgil's `nsigma` works in log space and stays finite to ~37σ in
  float64. Below 8σ the two agree to 2e-13σ.

## Problems to raise (with approval)

| # | CANDID | Problem | Evidence | Severity |
| --- | --- | --- | --- | --- |
| P4 | c255e90 | `detectionLimit` (Absil method) brackets the 3σ flux by factors of 1.4 and interpolates linearly in nσ. Against CANDID's own criterion solved exactly (Brent's method on its χ² and `_nSigmas`), its limits are 1.0 % low on average and 1.3 % at most: slightly optimistic, by ~0.01 mag. | `test_candid_public_limits_are_within_their_interpolation`; `test_p4_candid_public_limits_solve_its_criterion` (strict xfail) | low: precision |
