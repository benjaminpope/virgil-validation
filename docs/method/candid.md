# CANDID: conventions and problems

[CANDID](https://github.com/amerand/CANDID) (A. Mérand and A. Gallenne;
Gallenne et al. 2015) is the binary-search code that virgil's grid search
and Absil limits follow. It is one of the external roots of trust for those
steps (see [design](index.md)). It has no PyPI release and states no
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
| smearing average (†) | `Nsmear` = 3: λ + (−½, 0, +½) Δλ, end points included; V² is the mean of \|V\|², T3 the mean bispectrum over the three | none in virgil; `scripts/oidb_fit.py` `Smeared`: complex V averaged over 7 mid-points of the top hat, then \|·\|² and the closure phase of the mean |
| resolved flux (†) | `fres`, percent of the primary, in the normalisation only: V = (V* + f V_c) / (1 + f + fres) | no virgil model; `scripts/oidb_fit.py` `CandidBinary`: `resolved`, the same, as a fraction |
| χ²_r | mean of squared normalised residuals | χ² / number of data points |
| closure-phase residual | plain difference Δ | chord 2 sin(Δ/2) |
| significance | χ² tail (`chi2.sf`, then `chdtri`), two-sided Gaussian | the same (`gammaincc`, then `ndtri`); saturates instead of `inf` |
| degrees of freedom (Absil) | number of data points | the same |
| fit uncertainties | scaled by √χ²_r, χ²_r = χ²/(N − n_fit + 1) | not scaled |

(†) Read from CANDID's source at `c255e90` (`_VbinSlow` and the Cython
`_V2binFast`/`_T3binFast`), not pinned by a test. The resolved flux is
defined as ours. The smearing is not: averaging V² and the bispectrum is not
the same as averaging V, so the O1 comparison scores De Furio et al.'s
CANDID flux and resolved-flux values under the 2σ parametric rule, not the
0.25σ per-epoch rule.

The bridge switches CANDID to V² and closure phases, zero channel width (no
smearing) and one core. With those settings, the only difference of
definition left is the closure-phase residual (ledger D5). On V²-only files
the two codes agree to the float32 precision of the file.

## Differences that are not errors

* **D5, closure-phase residual.** The plain difference and the chord agree
  to O(Δ³). Near a good fit the two χ² values match to ~1e-6. Far from the
  data, where model closure phases are tens of degrees off, they differ by
  up to 1e-3.
* **Significance at the float64 limit.** Both take the two-sided Gaussian
  equivalent of the χ² tail and agree to 1e-12 relative up to 27σ. Beyond
  ~37σ, where the tail probability underflows, CANDID returns `inf` (as
  `fitMap` reports for a strong detection) and virgil's `nsigma`
  saturates at 37.5σ.

## Problems to raise (with approval)

| # | CANDID | Problem | Evidence | Severity |
| --- | --- | --- | --- | --- |
| P4 | c255e90 | `detectionLimit` (Absil method) brackets the 3σ flux by factors of 1.4 and interpolates linearly in nσ. Against CANDID's own criterion solved exactly (Brent's method on its χ² and `_nSigmas`), its limits are 1.0 % low on average and 1.3 % at most: slightly optimistic, by ~0.01 mag. | `test_candid_public_limits_are_within_their_interpolation`; `test_p4_candid_public_limits_solve_its_criterion` (strict xfail) | low: precision |
