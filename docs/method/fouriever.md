# fouriever: conventions and notes

[fouriever](https://github.com/kammerje/fouriever) (J. Kammerer) is the code
of Kammerer et al. (2020, A&A 644, A110), whose model of correlated closure
phases virgil adopts. It is the external root for that part of virgil's
likelihood (`OIData.cp_noise`). It states no licence, so
`scripts/setup_external.sh` installs it at a pinned version (0.4.3) into
its own environment, and `src/external_bridge/fouriever_worker.py` runs it
in a subprocess. It reads our OIFITS files with its own reader. MultiNest
is needed only for its nested sampler, which we don't use.

## Conventions

Pinned by `tests/test_fouriever.py`.

| Quantity | fouriever | virgil |
| --- | --- | --- |
| binary parameters | `[f, ΔRA, ΔDec]`: linear flux ratio, mas, East positive | `flux`, `dra`, `ddec` |
| Fourier sign | exp(−2πi(u ΔRA + v ΔDec)/λ) | the same |
| closure-phase correlation | R = T Tᵀ / 3 per snapshot, identity across channels (`intercorr.add_cpcov`) | the same, per snapshot and channel |
| covariance | C = D^½ R D^½, written to a `CPCOV` extension | the same |
| χ² with correlations | rᵀ C⁺ r (pseudo-inverse) | rᵀ D^−½ R⁺ D^−½ r (whitened, then projected) |
| closure-phase residual | plain difference from the sum of the baseline phases, not wrapped | sin Δ, correlated, plus an uncorrelated periodic penalty (1 − cos Δ)/σ per closure phase (since virgil#174; continuous and 2π periodic) |
| bandwidth smearing | off by default | off |

## Differences that are not errors

* **D5, closure-phase residual.** The same difference as with CANDID,
  except that CANDID wraps the model closure phase into (−π, π] while
  fouriever does not (problem P5).
  fouriever's χ² equals our plain-residual form to 1e-15, and virgil's
  equals its sine-plus-penalty form to 1e-15. Near the true companion the two codes
  agree to 5e-6.
* **D6, generalised inverse.** C is singular: with N telescopes only
  (N−1)(N−2)/2 combinations of a snapshot's closure phases are independent.
  - fouriever uses the pseudo-inverse C⁺. virgil whitens each residual by
    its σ and inverts R on the independent combinations.
  - Both are generalised inverses of C, and they agree whenever all the
    triangles of a group have equal errors.
  - With unequal errors they differ by a few percent.
  - On real closure-phase noise (closures of baseline-phase noise with
    unequal baseline errors), virgil's form gives the mean χ² closer to the
    nominal value. For four telescopes with one baseline three times
    noisier, the means are 2.94 (virgil) and 2.75 (fouriever) against 3,
    and the exact model gives 2.99.

## Problems raised upstream

| # | fouriever | Problem | Evidence | Severity |
| --- | --- | --- | --- | --- |
| P5 | 0.4.3 | The closure-phase residual is `data − model`, with the model the sum of the three baseline phases and nothing wrapped. Take a near-equal binary (flux ratio 0.99, inside its fitting range) whose closure phase is 178°. A measurement 3° away is stored as −179°, so the residual is 357° instead of 3°, and the true binary gets a huge χ². Measurements this close to ±180° are common for near-equal binaries with a few degrees of noise. | `test_p5_residual_wraps_across_the_phase_cut` (strict xfail); `test_plain_residual_across_the_phase_cut` pins the unwrapped definition | medium: wrong fits near ±180°; raised as [fouriever#26](https://github.com/kammerje/fouriever/issues/26) |

## Reading requirements

fouriever needs both `OI_VIS2` and `OI_T3`. Every triangle's baselines must
be VIS2 rows, oriented from `STA_INDEX[0]` to `STA_INDEX[1]`, in
consecutive blocks per snapshot. A primary-header `TELESCOP` other than
the VLTI or JWST raises an error. Our writer meets all of these.

## Not checked yet

A read-only survey of fouriever's source noticed several things that may
be problems. None is confirmed by a test yet, so none is in the ledger:

* `chi2map` looks for `'rho'` where it means `'phi'` in `searchbox`.
* `lincmap` takes `np.max(bmin)`.
* `detlim` returns only a figure, not the limits.
* `util.nsigma` saturates near 8σ.
