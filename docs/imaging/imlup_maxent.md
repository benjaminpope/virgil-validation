# IM Lup: MaxEnt deconvolution with a pre-registered weight rule

The primary ALMA deconvolution of IM Lup is maximum entropy on √I pixels with one regularizer weight w. A fixed rule picks w from the data alone: no L-curve by eye, no cross-validated optimum, no fallback value. The rule (`lambda_selection` in `design/alma_imlup_gate_criteria.yml`) was written, and revised only on synthetic data, before any real-data run. It is implemented in `virgil_bridge.alma_lambda`; the results block below is filled in by `lambda-report` from the saved run.

For the comparison with MPoL and CLEAN see [IM Lup: virgil against MPoL and CLEAN](imlup.md).

## Method

Each step is a stage of `lambda-rule`; each stage writes one JSON, and a failure ends the run with its reason.

1. **Noise scale (`sigma`).** The weights are rescaled by s², estimated from the scatter of successive differences per baseline, spectral window and scan on the full data, never from an image fit. σ_s/s² must be ≤ 2e-3. Raw χ²/N on the quoted weights is reported first.
2. **Bias check (`d4`).** The same estimator runs on 20 noise realizations of known s² added to the pinned CLEAN model's visibilities at the real sampling; the mean must agree with the injected s² within 2 sd/√20 and 2e-3 s². The sweep refuses to run without a pass.
3. **Pixels and solver.** The image is a `LinearImage` of √I pixels (so I ≥ 0), regularized by [`MaxEntropy`](https://benjaminpope.github.io/virgil/api/imaging/) and fitted by L-BFGS-B ([`fit`](https://benjaminpope.github.io/virgil/api/fitting/)) with the exact Toeplitz χ² ([`radio_toeplitz`](https://benjaminpope.github.io/virgil/api/radio_toeplitz/)).
4. **Sweep (`sweep`).** On the full data, w/s² runs from 1e8 down to 1e2 in steps of ×√10, warm-started from large w. At each w, G = Tr[K(K + wI)⁻¹] is estimated by Hutchinson's method (16 probes, CG tolerance 1e-4), and the discrepancy m(w) = χ²_q/s² − (n_real − G) is formed by [`dof_discrepancy`](https://benjaminpope.github.io/virgil/api/imaging/).
5. **Choice of w.** w is the root of m in ln w, refined by up to 5 secant steps until |m| ≤ 3 se(G). With several crossings the one at the largest w is taken and all are reported; w at s² ± 2σ_s and classic MaxEnt's h(w) are diagnostics only. No root on the grid, or m ≤ 0 at its first point, fails the run: the grid is never extended.
6. **Convergence.** Every fit must reach kkt_w < 1e-4 and f_η/f_floor < 2 ([`kkt_report`](https://benjaminpope.github.io/virgil/api/fitting/)).
7. **Nuisances (`final`).** At that w, per-observation gains and offsets are fitted in stages (image, nuisances, joint) by [`fit_staged_nuisances`](https://benjaminpope.github.io/virgil/api/radio/), with the observation of largest Σw as the reference gain. The run fails if any nuisance ends within 5% of its prior box edge.
8. **Report.** `lambda-report` tabulates every number above against its threshold and draws the figures. It refits nothing.
9. **Known bias.** On synthetic spiral discs this arm under-recovers spiral and ring contrast by about 20% at the chosen w (m = 2 amplitude ratio 0.77 and 0.79). Contrast claims carry that caveat.

## Why this rule

Earlier fits showed chains of dots along the rings. Two causes were found on synthetic discs:

- **Regularization too weak.** A noise scale taken from an unregularized fit puts the discrepancy target at the χ² floor, which selects w → 0, and the dots return. The target is now corrected for the degrees of freedom the fit absorbs (G), and the root at the strong-regularization end is taken.
- **Blurring kernels inside the model.** A smoothing kernel in the forward model lets the image hold structure the data cannot see. The pixels are now unblurred and the regularizer alone sets the resolution.

A first draft that thinned the data to a subsample biased m(w) towards large w by about (k − 1)·G and was withdrawn: the fast loop uses all the data. StarletL1 was dropped (dots, lowest contrast); the Gaussian-field and multipole arms are exploratory and carry no weight.

## Results

<!-- gen:lambda-report -->

**Pending.** No real-data run of the registered rule has been made. When one lands, fill this block with

```bash
python scripts/alma_imlup_maxent.py lambda-report --run ~/data/alma_imlup/rule/<run-id> --clean ~/data/alma_imlup/clean/out/<tag> --page docs/imaging/imlup_maxent.md
```

<!-- /gen:lambda-report -->
