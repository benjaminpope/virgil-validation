# Multi-epoch orbits

`tests/test_epochs.py` checks `virgil.epochs`, the tools that start and
score an orbit fit to interferometric data from several epochs. The data
are three-telescope nights simulated by `crosscheck.simulate`, so every
closure phase is independent, and read with astropy. The reference is our
own χ² (`crosscheck.chi2`), our own Kepler positions (`crosscheck.orbits`)
and SciPy. One night's quoted errors are three times too small in the
orbit data, so that ranking on the quoted errors and ranking with
marginalized scales give different orders.

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `marginal_loglike` | differences between models of ln ∫ L(s) ds/s, one scale per block (V², closure-phase chords), by `scipy.integrate.quad` in ln s; with `dof` = 0.5 the likelihood is tempered, L^dof | Mathematics | 1e-9 (measured 2e-13) |
| `marginal_loglike(s_max=10)` | the same integral bounded to [1/s_max, s_max], with the exact von Mises density exp(κ cos Δ)/(2π I₀(κ)), κ = 1/(sσ)², for weak (40°) closure phases | Mathematics | 1e-6 (measured 2e-7) |
| `marginal_loglike(s_max=1.2)`, bound binding | the mirror image's m difference, with the upper bound (s ≈ 1.1) and the lower bound (errors twice too large, s ≈ 0.5) each within about one posterior width of the peak; the bound moves the answer by 0.3 and 1.8, against the unbounded integral | Mathematics | 1e-6 (measured 2e-14) |
| control: wrong bounds | no upper limit, no lower limit, and [1/s_max², s_max], each on the data that reach it, must miss | Mathematics | misses by 0.1, 2.2 and 0.4 |
| steep edge | an offset model whose likelihood is cut off steeply by `s_max` ([F18](../method/findings.md), fixed in virgil#295) | Mathematics | 1e-6 (measured 5e-13; missed by 0.15-0.73 before virgil#295) |
| control: a uniform prior in s | ∫ L ds gives (ν − 1)/2 in place of ν/2 and must miss | Mathematics | misses by 1.5 |
| `epoch_positions`, `EpochPositions` | our grid search of m = −Σ_b (ν_b/2) ln χ²_b: the same best grid point and flux, `gap_marginal` (on m) and `gap` (on the quoted errors) | Mathematics | 1e-8 (measured 1e-15) |
| `epoch_positions` (refined) | the maximum of our m by SciPy; the covariance as the position block of the inverse curvature of m in position and flux (our Richardson-extrapolated finite differences); the flux-fixed inverse of the position block differs by 4% to 29%, so the flux is shown to be included; `chi2_raw` and `scale` | Mathematics | 1e-6 mas and flux; 1e-4 (measured 2e-6); 1e-6 |
| `epoch_positions` with every error three times too small | positions, covariances and `gap_marginal` unchanged; `gap` and χ²/N grow ninefold; the documented warning | Mathematics | 1e-6 (measured 1e-13) |
| `Epochs.loglike`, `OrbitalBinary` | the normalized Gaussian V² and von Mises closure-phase log likelihood of four epochs, the companion at our orbit position | Mathematics | 1e-10 |
| `rank_orbits(scales="quoted")`, `RankedOrbits` | −Δχ²/2 of eight trial orbits (the truth, its mirror, six perturbed) and the order; `positions` are our Kepler positions | Mathematics | 1e-6 relative (measured 6e-11) |
| `rank_orbits(scales="marginal")` | Δm summed over epochs, and the order | Mathematics | 1e-6 relative (measured 2e-12) |
| `chain_starts` | our greedy choice of distinct orbits over two clusters of trial orbits (one mode when every snapshot position agrees within half our λ/B_max): one start in each mode, never the mirror of the truth, and repeats in turn | Mathematics | exact |
| `start_from_positions`, `OrbitStart` (weekly, about 10 s) | the best fit's positions at each epoch match the true orbit's, and it fits at least as well as the truth by our χ² | Mathematics | 0.1 mas (measured 0.03 mas) |
